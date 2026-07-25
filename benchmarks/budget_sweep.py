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
import json
import os
import time

import torch

from geodesia_kv import policies as P
from benchmarks.vs_sota import build_passkey, run_passkey, run_ppl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--depths", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["policy"] = P.policy_attention

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()

    T = args.tokens
    b = T // 8
    base = dict(name="geodesia-ub", mode="cert", tau=0.5, block=64, window=128, sinks=4)
    configs = [
        ("Full-KV (oracolo)", dict(name="full")),
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

    key = "8 4 7 2 9 1"
    key_flat = key.replace(" ", "")
    key_ids = torch.tensor([tok(" " + key, add_special_tokens=False).input_ids])
    ppl_ids = tok(open(args.text).read(), return_tensors="pt").input_ids[:, :T]

    results = []
    for label, kw in configs:
        pol = P.Policy(**kw)
        pol.report = P.Report()
        t0 = time.time()
        hits, nlls = 0, []
        for d in args.depths:
            ids = build_passkey(tok, T, d, key)
            txt, nll = run_passkey(model, tok, pol, ids, key_ids)
            hits += key_flat in txt.replace(" ", "")
            nlls.append(nll)
        bpv = pol.report.bits_per_value
        pol.report = P.Report()
        ppl = run_ppl(model, tok, pol, ppl_ids)
        rep = pol.report
        row = {"policy": label, "bits_per_value": bpv, "ppl": ppl,
               "nll_key": sum(nlls) / len(nlls),
               "passkey_acc": hits / len(args.depths),
               "cert_violations": rep.n_violations, "cert_calls": rep.n_certified,
               "cert_bound": rep.cert_bound / max(rep.n_certified, 1),
               "true_err": rep.true_err / max(rep.n_certified, 1),
               "sec": time.time() - t0}
        results.append(row)
        cert = (f" | viol {rep.n_violations}/{rep.n_certified}"
                f" err {row['true_err']:.4f}") if rep.n_certified else ""
        print(f"{label:<30}{bpv:6.2f} bit/val  ppl {ppl:7.3f}  "
              f"nll_key {row['nll_key']:6.3f}  pk {100*row['passkey_acc']:5.1f}%"
              f"{cert}  [{row['sec']:.0f}s]")

    out = args.out or f"paper/results_2bit_band_{T}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"model": args.model, "tokens": T, "results": results}, fh, indent=2)
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
