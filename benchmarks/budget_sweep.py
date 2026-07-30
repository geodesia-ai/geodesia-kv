"""Fascia 2 bit/valore: dove Quest ci batteva. Ablazione delle quattro leve.

A 16k, Quest p=32 (2.26 bit/valore) faceva ppl 8.793 contro i nostri 9.564 a 2.90
bit: piu' memoria E peggio. Il motivo e' che a 2 bit il passo di quantizzazione e'
range/3 e l'errore sui punteggi viene esponenziato dal softmax.

Quattro leve, tutte esatte o rigorosamente limitate:
  rotate        Hadamard su q,k,v: q.k invariato, ma l'energia si redistribuisce
                sui canali -> spariscono i canali outlier che sprecavano i bit
  group         gruppo di quantizzazione 16 invece di 64: range piu' stretto
  kv_bit_delta  +1 bit alle key, -1 alle value: l'errore sulle key e' esponenziato
  centroid_frac i blocchi piu' freddi collassano nel centroide (0.5 bit/valore):
                restano nel softmax col loro peso, ma senza dettaglio -> e' cosi'
                che si scende SOTTO i 2 bit senza evittare nulla

Uso:
  .venv/bin/python experiment_2bit_band.py --tokens 16384
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from itertools import product

import torch

from geodesia_kv import policies as P
from benchmarks.vs_sota import build_passkey, run_passkey, run_ppl


def parse_layer_budget_profile(spec: str, n_layers: int) -> tuple[float, ...]:
    """Espande ``budget x numero_layer`` separati da virgole.

    Esempio per 36 layer: ``1.875x6,3x6,1.875x24``. Il formato esplicita anche
    il costo medio e impedisce di confrontare profili con layer mancanti.
    """
    values: list[float] = []
    try:
        for item in spec.split(","):
            budget_s, count_s = item.lower().split("x", 1)
            budget, count = float(budget_s), int(count_s)
            if budget <= 0 or count <= 0:
                raise ValueError
            values.extend([budget] * count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"profilo layer non valido {spec!r}; usa budgetxcount,..."
        ) from exc
    if len(values) != n_layers:
        raise ValueError(
            f"il profilo {spec!r} descrive {len(values)} layer, "
            f"ma il modello ne ha {n_layers}")
    return tuple(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"],
                    default="bfloat16",
                    help="dtype delle attivazioni (i checkpoint GPTQ preferiscono float16)")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--depths", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--text-offset", type=int, default=0,
                    help="offset in token nel corpus, per finestre disgiunte")
    ap.add_argument("--text-offsets", type=int, nargs="*", default=None,
                    help="piu' finestre disgiunte; sostituisce --text-offset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=[],
                    help="esegui solo le configurazioni il cui nome contiene uno dei pattern")
    ap.add_argument("--eval-tokens", type=int, default=256)
    ap.add_argument(
        "--incremental-eval", action="store_true",
        help=("teacher forcing token-by-token: replica il decode e non tratta "
              "l'intero chunk eval come residuo corrente esatto"))
    ap.add_argument(
        "--skip-certificate-validation", action="store_true",
        help=("non ricalcola l'attenzione densa usata solo per verificare il "
              "bound; output e allocazione Geodesia restano invariati"))
    ap.add_argument("--skip-passkey", action="store_true")
    ap.add_argument("--geodesia-budgets", type=float, nargs="*", default=[])
    ap.add_argument(
        "--geodesia-settings", nargs="*", default=[],
        help=("budget:mass_decay[:keep_frac[:cold_bias[:gate"
              "[:protected_frac[:token_protected_frac[:layer_profile"
              "[:spectral_rank[:spectral_strength"
              "[:cumulant_strength[:token_value_power"
              "[:token_key_power]]]]]]]]]]], es. "
              "2:.5:.125:6:centroid:0:0:1.875x6,3x6,1.875x24:3:.5:1:.5:0"))
    ap.add_argument("--alloc-value-weights", type=float, nargs="*", default=[0.0])
    ap.add_argument("--alloc-mass-powers", type=float, nargs="*", default=[1.0])
    ap.add_argument("--mass-decays", type=float, nargs="*", default=[0.9])
    ap.add_argument("--obs-windows", type=int, nargs="*", default=[32])
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils.hub import cached_file
    ALL_ATTENTION_FUNCTIONS["policy"] = P.policy_attention
    ALL_ATTENTION_FUNCTIONS["observe"] = P.observation_attention

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model_config = AutoConfig.from_pretrained(args.model)
    qconfig = getattr(model_config, "quantization_config", None)
    # Alcuni checkpoint GPTQ (incluso Qwen3-30B-A3B-GPTQ-Int8) hanno il
    # `quant_method` soltanto in quantize_config.json. Transformers controlla
    # prima la copia incompleta in config.json e fallisce prima del caricamento.
    if isinstance(qconfig, dict) and "quant_method" not in qconfig:
        quantize_path = cached_file(args.model, "quantize_config.json")
        with open(quantize_path) as fh:
            model_config.quantization_config = json.load(fh)
    model_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=model_config, torch_dtype=model_dtype,
        device_map="cuda",
        attn_implementation="sdpa").eval()
    n_layers = model_config.get_text_config(
        decoder=True).num_hidden_layers

    T = args.tokens
    b = T // 8
    base = dict(name="geodesia-ub", mode="cert", tau=0.5, block=64, window=128, sinks=4)
    configs = [
        ("Full-KV (oracolo)", dict(name="full")),
        (f"StreamingLLM b={b}", dict(
            name="streaming", budget_tokens=b, sinks=4)),
        (f"SnapKV b={b}", dict(name="snapkv", budget_tokens=b)),
        (f"Quest p={b // 64}  [SOTA]",
         dict(name="quest", quest_pages=b // 64, block=64)),
        ("KIVI k2v2 g32 r32 [SOTA]", dict(name="kivi")),
        ("KIVI k4v4 g32 r32 [SOTA]", dict(name="kivi", kivi_k_bits=4,
                                          kivi_v_bits=4)),
        ("KIVI k2v2 g32 r128", dict(name="kivi", kivi_residual=128)),
        # --- adattivo: NON implementabile, tenuto come limite superiore ---
        ("Geodesia-KV adattivo min4 cen.5 [UB]", {**base, "group": 64, "min_bits": 4.0,
                                         "centroid_frac": 0.5}),
        ("Geodesia-KV adattivo cen.8 [UB]", {**base, "group": 64, "centroid_frac": 0.8}),
        # --- PRODUZIONE: demozione monotona guidata dal budget ---
        ("Geodesia-KV budget 5.0", dict(name="geodesia", budget_bits=5.0,
                                      group=64, window=128, sinks=4)),
        ("Geodesia-KV budget 4.0", dict(name="geodesia", budget_bits=4.0,
                                      group=64, window=128, sinks=4)),
        ("Geodesia-KV budget 3.0", dict(name="geodesia", budget_bits=3.0,
                                      group=64, window=128, sinks=4)),
        ("Geodesia-KV budget 2.0", dict(name="geodesia", budget_bits=2.0,
                                      group=64, window=128, sinks=4)),
        ("Geodesia-KV budget 1.5", dict(name="geodesia", budget_bits=1.5,
                                      group=64, window=128, sinks=4)),
    ]
    if args.geodesia_budgets or args.geodesia_settings:
        configs = [(label, kw) for label, kw in configs
                   if kw.get("name") != "geodesia"]
        for budget, value_weight, mass_power, mass_decay, obs_window in product(
                args.geodesia_budgets, args.alloc_value_weights,
                args.alloc_mass_powers, args.mass_decays, args.obs_windows):
            label = (f"Geodesia-KV B={budget:g} vw={value_weight:g} "
                     f"mp={mass_power:g} md={mass_decay:g} obs={obs_window}")
            configs.append((label, dict(
                name="geodesia", budget_bits=budget, group=64, window=128,
                sinks=4, alloc_value_weight=value_weight,
                alloc_mass_power=mass_power, mass_decay=mass_decay,
                obs=obs_window)))
        for setting in args.geodesia_settings:
            try:
                fields = setting.split(":")
                if len(fields) not in (
                        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13):
                    raise ValueError
                budget, mass_decay = float(fields[0]), float(fields[1])
                keep_frac = float(fields[2]) if len(fields) >= 3 else 1.0
                cold_bias = float(fields[3]) if len(fields) >= 4 else 0.0
                gate = fields[4] if len(fields) >= 5 else "mass"
                protected_frac = float(fields[5]) if len(fields) >= 6 else 0.0
                token_protected_frac = (
                    float(fields[6]) if len(fields) >= 7 else 0.0)
                layer_profile_spec = fields[7] if len(fields) >= 8 else ""
                layer_budget_bits = (
                    parse_layer_budget_profile(layer_profile_spec, n_layers)
                    if layer_profile_spec else ())
                spectral_rank = int(fields[8]) if len(fields) >= 9 else 0
                if spectral_rank < 0:
                    raise ValueError
                spectral_strength = (
                    float(fields[9]) if len(fields) >= 10 else 1.0)
                if not 0.0 <= spectral_strength <= 1.0:
                    raise ValueError
                cumulant_strength = (
                    float(fields[10]) if len(fields) >= 11 else 0.0)
                token_value_power = (
                    float(fields[11]) if len(fields) >= 12 else 0.0)
                token_key_power = (
                    float(fields[12]) if len(fields) >= 13 else 0.0)
            except ValueError as exc:
                raise SystemExit(
                    f"setting Geodesia non valido {setting!r}; usa "
                    "budget:mass_decay[:keep_frac[:cold_bias[:gate"
                    "[:protected_frac[:token_protected_frac"
                    "[:layer_profile[:spectral_rank"
                    "[:spectral_strength[:cumulant_strength"
                    "[:token_value_power[:token_key_power]]]]]]]]]]]"
                ) from exc
            label = (f"Geodesia-KV B={budget:g} md={mass_decay:g} "
                     f"qk={keep_frac:g} qb={cold_bias:g} gate={gate} "
                     f"pr={protected_frac:g} tpr={token_protected_frac:g}"
                     f"{f' lp={layer_profile_spec}' if layer_profile_spec else ''}"
                     f" sr={spectral_rank} ss={spectral_strength:g} "
                     f"cs={cumulant_strength:g} tv={token_value_power:g} "
                     f"tk={token_key_power:g}")
            configs.append((label, dict(
                name="geodesia", budget_bits=budget, group=64, window=128,
                sinks=4, alloc_value_weight=0.0, alloc_mass_power=1.0,
                mass_decay=mass_decay, obs=32, query_keep_frac=keep_frac,
                query_cold_bias=cold_bias, query_gate=gate,
                protected_frac=protected_frac,
                token_protected_frac=token_protected_frac,
                layer_budget_bits=layer_budget_bits,
                spectral_rank=spectral_rank,
                spectral_strength=spectral_strength,
                cumulant_strength=cumulant_strength,
                token_value_power=token_value_power,
                token_key_power=token_key_power)))
    if args.skip_certificate_validation:
        for _, kw in configs:
            if kw.get("name") in ("geodesia", "geodesia-ub"):
                kw["validate"] = False
    if args.only:
        pats = [x.casefold() for x in args.only]
        configs = [(label, kw) for label, kw in configs
                   if any(p in label.casefold() for p in pats)]
        if not configs:
            raise SystemExit(f"nessuna configurazione corrisponde a {args.only}")

    key = "8 4 7 2 9 1"
    key_flat = key.replace(" ", "")
    key_ids = torch.tensor([tok(" " + key, add_special_tokens=False).input_ids])
    text_offsets = args.text_offsets if args.text_offsets else [args.text_offset]
    all_text_ids = tok(open(args.text).read(), return_tensors="pt",
                       max_length=max(text_offsets) + T,
                       truncation=True).input_ids
    ppl_windows = {}
    for offset in text_offsets:
        ids_window = all_text_ids[:, offset:offset + T]
        if ids_window.shape[1] < T:
            raise SystemExit(
                f"testo insufficiente: richiesti {T} token da offset "
                f"{offset}, disponibili {ids_window.shape[1]}")
        ppl_windows[offset] = ids_window

    results = []
    for label, kw in configs:
        pol = P.Policy(**kw)
        pol.report = P.Report()
        t0 = time.time()
        hits, nlls = 0, []
        if not args.skip_passkey:
            for d in args.depths:
                ids = build_passkey(tok, T, d, key)
                txt, nll = run_passkey(model, tok, pol, ids, key_ids)
                hits += key_flat in txt.replace(" ", "")
                nlls.append(nll)
        bpv = pol.report.bits_per_value
        ppls = []
        per_window = []
        combined = P.Report()
        for offset, ids_window in ppl_windows.items():
            pol.report = P.Report()
            ppl_i, token_nll = run_ppl(
                model, tok, pol, ids_window, eval_tokens=args.eval_tokens,
                incremental=args.incremental_eval, return_token_nll=True)
            ppls.append(ppl_i)
            per_window.append({
                "offset": offset,
                "ppl": ppl_i,
                # Serve per gli intervalli paired: i target sono gli stessi per
                # tutte le policy sulla stessa finestra.
                "token_nll": token_nll,
                "bits_per_value": pol.report.bits_per_value,
                "read_bits_per_value": pol.report.read_bits_per_value,
                "true_err": (pol.report.true_err
                             / max(pol.report.n_certified, 1)),
                "cert_violations": pol.report.n_violations,
                "gate_keep_fraction": pol.report.gate_keep_fraction,
            })
            combined.merge(pol.report)
        # Finestre di uguale lunghezza: media delle NLL = media geometrica PPL.
        ppl = math.exp(sum(math.log(x) for x in ppls) / len(ppls))
        rep = combined
        if args.skip_passkey:
            bpv = rep.bits_per_value
        row = {"policy": label, "bits_per_value": bpv, "ppl": ppl,
               "read_bits_per_value": rep.read_bits_per_value,
               "nll_key": (sum(nlls) / len(nlls) if nlls else None),
               "passkey_acc": (hits / len(args.depths)
                               if not args.skip_passkey else None),
               "cert_violations": rep.n_violations, "cert_calls": rep.n_certified,
               "cert_bound": rep.cert_bound / max(rep.n_certified, 1),
               "true_err": rep.true_err / max(rep.n_certified, 1),
               "gate_keep_fraction": rep.gate_keep_fraction,
               "ppl_windows": per_window,
               "sec": time.time() - t0}
        if pol.name == "geodesia":
            hist = {str(level): 0 for level in P.LEVELS}
            for state in pol.state.values():
                if "level" not in state:
                    continue
                for level in P.LEVELS:
                    hist[str(level)] += int((state["level"] == level).sum())
            row["level_histogram"] = hist
        results.append(row)
        cert = (f" | viol {rep.n_violations}/{rep.n_certified}"
                f" err {row['true_err']:.4f}") if rep.n_certified else ""
        memory = (f"  nll_key {row['nll_key']:6.3f}  "
                  f"pk {100*row['passkey_acc']:5.1f}%"
                  if row["nll_key"] is not None else "")
        print(f"{label:<38}{bpv:6.2f} bit/val "
              f"read {row['read_bits_per_value']:5.2f}  ppl {ppl:7.3f}"
              f"{memory}{cert}  [{row['sec']:.0f}s]")

    out = args.out or f"paper/results_2bit_band_{T}.json"
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    text_sha = hashlib.sha256(open(args.text, "rb").read()).hexdigest()
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    quantization_config = getattr(model.config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    with open(out, "w") as fh:
        json.dump({"model": args.model,
                   "model_revision": getattr(model.config, "_commit_hash", None),
                   "tokens": T, "depths": args.depths,
                   "text_offsets": text_offsets,
                   "eval_tokens": args.eval_tokens,
                   "eval_mode": (
                       "incremental" if args.incremental_eval else "chunked"),
                   "certificate_validation": (
                       not args.skip_certificate_validation),
                   "text": args.text, "text_sha256": text_sha,
                   "git_commit": git_commit,
                   "dtype": args.dtype,
                   "quantization_config": quantization_config,
                   "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__,
                   "gpu": torch.cuda.get_device_name(),
                   "results": results}, fh, indent=2)
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
