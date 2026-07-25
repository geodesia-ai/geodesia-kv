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
    P.ACTIVE = None
    model.set_attn_implementation("sdpa")
    out = model(ids[:, :-1].cuda(), use_cache=True)
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
        P.ACTIVE = None
        model.set_attn_implementation("sdpa")
        out = model(ids[:, :split].cuda(), use_cache=True)
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
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--depths", type=int, default=8)
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
    depths = [round((i + 0.5) / args.depths, 4) for i in range(args.depths)]
    key = "8 4 7 2 9 1"
    key_ids = torch.tensor([tok(" " + key, add_special_tokens=False).input_ids])
    text_ids = tok(open(args.text).read(), return_tensors="pt").input_ids[:, :T]

    b_lo, b_hi = T // 8, T // 4
    gw = dict(name="geodesia-ub", mode="cert", block=64, window=128, sinks=4, group=16)
    configs = [
        ("Full-KV", dict(name="full")),
        (f"StreamingLLM b={b_lo}", dict(name="streaming", budget_tokens=b_lo, sinks=4)),
        (f"SnapKV b={b_lo}", dict(name="snapkv", budget_tokens=b_lo)),
        (f"Quest p={b_lo // 64}", dict(name="quest", quest_pages=b_lo // 64, block=64)),
        (f"Quest p={b_hi // 64}", dict(name="quest", quest_pages=b_hi // 64, block=64)),
        ("Geodesia-KV t=.5 cen.85", {**gw, "tau": 0.5, "centroid_frac": 0.85}),
        ("Geodesia-KV t=.5 cen.7", {**gw, "tau": 0.5, "centroid_frac": 0.7}),
        ("Geodesia-KV t=.2", {**gw, "tau": 0.20}),
    ]

    # riferimenti per depth: cache piena (denominatore) e contesto privo di ago
    print(f"riferimenti su {len(depths)} profondita'...")
    full_pol = P.Policy(name="full")
    nll_full, nll_void, prompts = {}, {}, {}
    for d in depths:
        prompts[d] = build(tok, T, d, key, True)
        nll_full[d] = key_nll(model, full_pol, prompts[d], key_ids)
        nll_void[d] = key_nll(model, full_pol,
                              build(tok, T, d, key, False), key_ids)
        print(f"  d={d:<6} NLL_full={nll_full[d]:6.3f}  NLL_vuoto={nll_void[d]:6.3f}"
              f"  info={nll_void[d]-nll_full[d]:6.3f}")

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
        for d in depths:
            n = key_nll(model, pol, prompts[d], key_ids)
            denom = nll_void[d] - nll_full[d]
            rs.append((nll_void[d] - n) / denom if abs(denom) > 1e-6 else float("nan"))
        bpv = pol.report.bits_per_value
        pol.report = P.Report()
        kl = kl_memory(model, pol, text_ids)

        third = max(1, len(rs) // 3)
        mq = sum(rs) / len(rs)
        mq_far = sum(rs[:third]) / third

        # --- metriche PRATICHE ------------------------------------------- #
        # L_eff: quanti token di contesto restano usabili. Si scorre dal piu'
        # recente al piu' vecchio e ci si ferma alla prima profondita' che perde
        # piu' del 10% dell'informazione. E' il numero che un ingegnere usa.
        span = T / len(rs)
        l_eff = 0.0
        for r in reversed(rs):
            if r < 0.9:
                break
            l_eff += span
        # MiB residenti e costo per 1k token EFFETTIVI (non nominali)
        mib = pol_mib(bpv, T)
        mib_per_1k = mib / max(l_eff / 1000.0, 1e-9)

        results.append({"policy": label, "bits_per_value": bpv, "R": rs,
                        "MQ": mq, "MQ_far": mq_far, "KL_mem": kl,
                        "L_eff": l_eff, "L_eff_frac": l_eff / T,
                        "resident_MiB": mib, "MiB_per_1k_effective": mib_per_1k,
                        "cert_violations": pol.report.n_violations})
        curve = " ".join(f"{r:5.2f}" for r in rs)
        print(f"{label:<22}{bpv:8.2f}{mq:7.3f}{mq_far:8.3f}{kl:9.4f}"
              f"{l_eff/1000:8.1f}k{mib_per_1k:9.2f}   {curve}")

    out = args.out or f"paper/results_mqi_{os.path.basename(args.model)}_{T}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"model": args.model, "tokens": T, "depths": depths,
                   "nll_full": nll_full, "nll_void": nll_void,
                   "results": results}, fh, indent=2)
    print(f"\nscritto {out}")


if __name__ == "__main__":
    main()
