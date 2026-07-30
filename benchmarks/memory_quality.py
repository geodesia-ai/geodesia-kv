"""Memory Quality Index: quanta informazione del contesto sopravvive, per eta'.

Perche' serve. passkey (binaria) e nll_key saturano: a 16k tutti i metodi tranne
StreamingLLM fanno 100% e ~0.115. La perplexity misura la fluenza LOCALE, non la
memoria. Serve una metrica graduata, normalizzata sulla capacita' del modello, e
risolta per profondita'.

Componenti
----------
R(d)     ritenzione a profondita' d:
             R(d) = (NLL_vuoto - NLL_policy) / (NLL_vuoto - NLL_full)
         NLL_vuoto = NLL della chiave col contesto PRIVO dell'ago (stesse
         posizioni, filler al suo posto) = informazione a priori del modello.
         Il denominatore e' l'informazione che la cache PIENA da' sull'ago.
         Quindi R(d) e' la frazione di informazione contestuale conservata:
         1 = memoria perfetta, 0 = cache inutile, <0 = cache fuorviante.
         La normalizzazione toglie di mezzo la bravura del modello: funziona
         anche su modelli troppo piccoli per risolvere il task da soli.

KL_mem   fedelta' distribuzionale su testo naturale:
             E_t [ KL( p_full(.|ctx) || p_policy(.|ctx) ) ]
         cattura la degradazione dell'informazione DIFFUSA, che l'ago non vede.

MQ       media di R(d);  MQ@far = media sul terzo piu' VECCHIO del contesto
MQE      MQ / bit-per-valore = memoria conservata per bit speso

Uso:
  .venv/bin/python metric_memory_quality.py --model Qwen2.5-0.5B-Instruct \
      --tokens 16384 --depths 8
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from geodesia_kv import policies as P
from benchmarks.vs_sota import prefill

FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
KEY_TMPL = "The pass key is {k}. Remember it. {k} is the pass key. "


def build(tok, n_tokens: int, depth: float, key: str, with_needle: bool):
    """Prompt con (o senza) ago a profondita' `depth`, a lunghezza confrontabile."""
    unit = tok(FILLER, add_special_tokens=False).input_ids
    need = max(n_tokens - 128, 128)
    ids = (unit * (need // len(unit) + 1))[:need]
    cut = int(len(ids) * depth)
    kids = tok(KEY_TMPL.format(k=key), add_special_tokens=False).input_ids
    if not with_needle:
        # stesso numero di token, ma senza l'informazione: filler al posto dell'ago
        kids = (unit * (len(kids) // len(unit) + 1))[: len(kids)]
    qids = tok("\nWhat is the pass key? The pass key is",
               add_special_tokens=False).input_ids
    return torch.tensor([ids[:cut] + kids + ids[cut:] + qids])


@torch.no_grad()
def key_nll(model, pol, ids, key_ids) -> float:
    """NLL media dei token della chiave, in teacher forcing, sotto `pol`."""
    pol.reset_state()
    out = prefill(model, pol, ids[:, :-1].cuda())
    cache = out.past_key_values
    del out
    P.ACTIVE = pol
    model.set_attn_implementation("policy")
    kt = key_ids.cuda()
    forced = torch.cat([ids[:, -1:].cuda(), kt[:, :-1]], dim=1)
    o = model(forced, past_key_values=cache, use_cache=True)
    P.ACTIVE = None
    lp = torch.nn.functional.log_softmax(o.logits.float(), -1)
    return float(-lp.gather(-1, kt[:, :, None]).mean())


@torch.no_grad()
def kl_memory(model, pol, ids, eval_tokens: int = 128) -> float:
    """KL( p_full || p_policy ) media sui token di valutazione."""
    split = ids.shape[1] - eval_tokens
    tgt = ids[:, split:].cuda()

    def dist(p):
        p.reset_state()
        out = prefill(model, p, ids[:, :split].cuda())
        cache = out.past_key_values
        del out
        P.ACTIVE = p
        model.set_attn_implementation("policy")
        o = model(tgt, past_key_values=cache, use_cache=True)
        P.ACTIVE = None
        return torch.nn.functional.log_softmax(o.logits.float(), -1)

    lf = dist(P.Policy(name="full"))
    lp = dist(pol)
    return float((lf.exp() * (lf - lp)).sum(-1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--depths", type=int, default=8)
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--geodesia-low-budget", type=float, default=3.0)
    ap.add_argument("--geodesia-low-decay", type=float, default=0.925)
    ap.add_argument("--geodesia-low-query-keep-frac", type=float, default=1.0)
    ap.add_argument("--geodesia-low-query-cold-bias", type=float, default=0.0)
    ap.add_argument("--geodesia-low-query-gate", default="mass")
    ap.add_argument("--geodesia-low-token-protected-frac", type=float,
                    default=0.0)
    ap.add_argument("--geodesia-low-token-value-power", type=float, default=0.0)
    ap.add_argument("--geodesia-low-token-key-power", type=float, default=0.0)
    ap.add_argument("--geodesia-high-budget", type=float, default=5.0)
    ap.add_argument("--geodesia-high-decay", type=float, default=0.25)
    ap.add_argument("--kl-tokens", type=int, default=128)
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="misura solo KL/capacita', senza needle NLL")
    ap.add_argument(
        "--geodesia-mix-settings", nargs="*", default=[],
        help="budget:token_protected_frac:value_power per il fronte RD")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["policy"] = P.policy_attention
    ALL_ATTENTION_FUNCTIONS["observe"] = P.observation_attention

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()

    T = args.tokens
    depths = [round((i + 0.5) / args.depths, 4) for i in range(args.depths)]
    key = "8 4 7 2 9 1"
    key_ids = torch.tensor([tok(" " + key, add_special_tokens=False).input_ids])
    text_ids = tok(open(args.text).read(), return_tensors="pt",
                   max_length=T, truncation=True).input_ids

    b_lo, b_hi = T // 8, T // 4
    configs = [
        ("Full-KV", dict(name="full")),
        (f"StreamingLLM b={b_lo}", dict(name="streaming", budget_tokens=b_lo, sinks=4)),
        (f"SnapKV b={b_lo}", dict(name="snapkv", budget_tokens=b_lo)),
        (f"Quest p={b_lo // 64}", dict(name="quest", quest_pages=b_lo // 64, block=64)),
        ("KIVI-2", dict(name="kivi")),
        ("KIVI-4", dict(name="kivi", kivi_k_bits=4, kivi_v_bits=4)),
        (f"Geodesia B={args.geodesia_low_budget:g}",
         dict(name="geodesia", budget_bits=args.geodesia_low_budget,
              block=64, group=64, window=128, sinks=4,
              mass_decay=args.geodesia_low_decay,
              query_keep_frac=args.geodesia_low_query_keep_frac,
              query_cold_bias=args.geodesia_low_query_cold_bias,
              query_gate=args.geodesia_low_query_gate,
              token_protected_frac=args.geodesia_low_token_protected_frac,
              token_value_power=args.geodesia_low_token_value_power,
              token_key_power=args.geodesia_low_token_key_power)),
        (f"Geodesia B={args.geodesia_high_budget:g}",
         dict(name="geodesia", budget_bits=args.geodesia_high_budget,
              block=64, group=64, window=128, sinks=4,
              mass_decay=args.geodesia_high_decay)),
    ]
    for setting in args.geodesia_mix_settings:
        try:
            budget_s, token_frac_s, value_power_s = setting.split(":")
            budget = float(budget_s)
            token_frac = float(token_frac_s)
            value_power = float(value_power_s)
        except ValueError as exc:
            raise SystemExit(
                f"mix non valido {setting!r}; usa budget:token_frac:value_power"
            ) from exc
        configs.append((
            f"Geodesia mix B={budget:g} tpr={token_frac:g} tv={value_power:g}",
            dict(name="geodesia", budget_bits=budget, block=64, group=64,
                 window=128, sinks=4, mass_decay=0.5,
                 query_keep_frac=0.125, query_cold_bias=4.0,
                 query_gate="token", token_protected_frac=token_frac,
                 token_value_power=value_power)))
    if args.only:
        pats = [x.casefold() for x in args.only]
        configs = [(label, kw) for label, kw in configs
                   if any(p in label.casefold() for p in pats)]
        if not configs:
            raise SystemExit(f"nessuna configurazione corrisponde a {args.only}")

    # riferimenti per depth: cache piena (denominatore) e contesto privo di ago
    print(f"riferimenti su {len(depths)} profondita'...")
    full_pol = P.Policy(name="full")
    nll_full, nll_void, prompts = {}, {}, {}
    if not args.skip_retrieval:
        for d in depths:
            prompts[d] = build(tok, T, d, key, True)
            nll_full[d] = key_nll(model, full_pol, prompts[d], key_ids)
            nll_void[d] = key_nll(model, full_pol,
                                  build(tok, T, d, key, False), key_ids)
            print(f"  d={d:<6} NLL_full={nll_full[d]:6.3f}  "
                  f"NLL_vuoto={nll_void[d]:6.3f}  "
                  f"info={nll_void[d]-nll_full[d]:6.3f}")

    # MiB residenti dell'intera KV del modello a contesto T, dato bit/valore
    cfg = model.config.get_text_config(decoder=True)
    n_val = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)

    def pol_mib(bits_per_value: float, ctx: int) -> float:
        return bits_per_value * n_val * ctx / 8 / 2 ** 20

    results = []
    print(f"\n{'metodo':<22}{'bit/val':>8}{'MQ':>7}{'MQ@far':>8}{'KL_mem':>9}"
          f"{'L_eff':>9}{'MiB/1k':>9}   R(d) dal piu' VECCHIO al piu' recente")
    for label, kw in configs:
        pol = P.Policy(**kw)
        pol.report = P.Report()
        rs = []
        if not args.skip_retrieval:
            for d in depths:
                n = key_nll(model, pol, prompts[d], key_ids)
                denom = nll_void[d] - nll_full[d]
                rs.append((nll_void[d] - n) / denom
                          if abs(denom) > 1e-6 else float("nan"))
            bpv = pol.report.bits_per_value
            read_bpv = pol.report.read_bits_per_value
        pol.report = P.Report()
        kl = kl_memory(model, pol, text_ids, eval_tokens=args.kl_tokens)
        if args.skip_retrieval:
            bpv = pol.report.bits_per_value
            read_bpv = pol.report.read_bits_per_value

        third = max(1, len(rs) // 3)
        mq = sum(rs) / len(rs) if rs else None
        mq_far = sum(rs[:third]) / third if rs else None

        # --- metriche PRATICHE ------------------------------------------- #
        # L_eff: quanti token di contesto restano usabili. Si scorre dal piu'
        # recente al piu' vecchio e ci si ferma alla prima profondita' che perde
        # piu' del 10% dell'informazione. E' il numero che un ingegnere usa.
        l_eff = None
        if rs:
            span = T / len(rs)
            l_eff = 0.0
            for r in reversed(rs):
                if r < 0.9:
                    break
                l_eff += span
        # MiB residenti e costo per 1k token EFFETTIVI (non nominali)
        mib = pol_mib(bpv, T)
        mib_per_1k = (
            mib / max(l_eff / 1000.0, 1e-9)
            if l_eff is not None else None)

        results.append({"policy": label, "bits_per_value": bpv,
                        "read_bits_per_value": read_bpv, "R": rs,
                        "MQ": mq, "MQ_far": mq_far, "KL_mem": kl,
                        "L_eff": l_eff,
                        "L_eff_frac": (l_eff / T
                                       if l_eff is not None else None),
                        "resident_MiB": mib,
                        "read_MiB_per_query": pol_mib(read_bpv, T),
                        "MiB_per_1k_effective": mib_per_1k,
                        "cert_violations": pol.report.n_violations})
        curve = " ".join(f"{r:5.2f}" for r in rs)
        if rs:
            print(f"{label:<22}{bpv:8.2f}{mq:7.3f}{mq_far:8.3f}{kl:9.4f}"
                  f"{l_eff/1000:8.1f}k{mib_per_1k:9.2f}   {curve}")
        else:
            print(f"{label:<22}{bpv:8.2f}{'-':>7}{'-':>8}{kl:9.4f}"
                  f"{'-':>9}{'-':>9}")

    out = args.out or f"paper/results_mqi_{os.path.basename(args.model)}_{T}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"model": args.model, "tokens": T, "depths": depths,
                   "nll_full": nll_full, "nll_void": nll_void,
                   "results": results}, fh, indent=2)
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
