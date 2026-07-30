"""Geodesia-KV contro 3 tecniche SOTA, a parita' di bit residenti.

Protocollo (identico per tutte le policy):
  1. prefill esatto del prefisso lungo con SDPA -> cache piena;
  2. si attiva la policy: da qui in poi l'attenzione vede la cache secondo la
     politica di memoria della policy (evizione o precisione allocata);
  3. si misura la qualita' sui token successivi.

Task
  passkey  recupero esatto di una chiave nascosta a profondita' variabile
           (e' il test che separa chi BUTTA il contesto da chi lo DEGRADA)
  ppl      perplexity degli ultimi token dato il prefisso compresso

Baseline SOTA: StreamingLLM (sink + finestra), SnapKV (top-B per massa osservata),
Quest (selezione per pagina con bound box). Oracolo: Full-KV.

Uso:
  .venv/bin/python -m benchmarks.vs_sota --tokens 8192
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from geodesia_kv import policies as P


FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
          "Here we go. There and back again. ")
KEY_TMPL = ("The pass key is {k}. Remember it. {k} is the pass key. ")


def build_passkey(tok, n_tokens: int, depth: float, key: str):
    """Prompt lungo con la chiave nascosta a profondita' `depth` in [0,1]."""
    unit = tok(FILLER, add_special_tokens=False).input_ids
    need = max(n_tokens - 128, 128)
    reps = need // len(unit) + 1
    ids = (unit * reps)[:need]
    cut = int(len(ids) * depth)
    kids = tok(KEY_TMPL.format(k=key), add_special_tokens=False).input_ids
    qids = tok("\nWhat is the pass key? The pass key is",
               add_special_tokens=False).input_ids
    return torch.tensor([ids[:cut] + kids + ids[cut:] + qids])


def prefill(model, pol, ids):
    """Prefill esatto; raccoglie la storia causale richiesta dalla policy."""
    P.ACTIVE = None
    if pol.name in ("geodesia", "snapkv"):
        P.OBSERVER = pol
        model.set_attn_implementation("observe")
    else:
        P.OBSERVER = None
        model.set_attn_implementation("sdpa")
    # Il prefill serve soltanto a costruire la KV. Chiamare il CausalLM completo
    # materializza inutilmente logits [B, T, vocab]: su 30B/16k sono diversi GiB
    # e possono trasformare un modello che entra in VRAM in un falso OOM.
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model
    out = decoder(ids, use_cache=True)
    P.OBSERVER = None
    return out


@torch.no_grad()
def run_passkey(model, tok, pol, ids, key_ids, gen: int = 12):
    """Ritorna (testo generato, NLL della chiave vera in teacher forcing).

    La generazione greedy misura il recupero end-to-end, ma su modelli piccoli
    fallisce anche con la cache PIENA: in quel caso non discrimina. La NLL della
    chiave vera misura invece quanta informazione la cache compressa conserva,
    indipendentemente dalla forza del modello.
    """
    pol.reset_state()
    out = prefill(model, pol, ids[:, :-1].cuda())
    base_cache = out.past_key_values
    last = ids[:, -1:].cuda()
    del out

    P.ACTIVE = pol
    model.set_attn_implementation("policy")

    # --- teacher forcing sulla chiave vera (non consuma la cache di generazione)
    kt = key_ids.cuda()
    forced = torch.cat([last, kt[:, :-1]], dim=1)
    o = model(forced, past_key_values=base_cache, use_cache=True)
    lp = torch.nn.functional.log_softmax(o.logits.float(), -1)
    nll_key = float(-lp.gather(-1, kt[:, :, None]).mean())
    base_cache.crop(ids.shape[1] - 1)          # rimuove i token forzati
    # La teacher forcing contiene la risposta vera: lasciare il suo stato nella
    # policy renderebbe la successiva generazione informata dai token target.
    # La cache del prompt e' stata ripristinata sopra; anche l'allocatore deve
    # ripartire dallo stesso stato causale.
    pol.reset_runtime()

    # --- generazione greedy
    cur, got = last, []
    cache = base_cache
    for _ in range(gen):
        o = model(cur, past_key_values=cache, use_cache=True)
        cur = o.logits[:, -1].argmax(-1, keepdim=True)
        got.append(int(cur))
        cache = o.past_key_values
    P.ACTIVE = None
    return tok.decode(got), nll_key


@torch.no_grad()
def run_ppl(model, tok, pol, ids, eval_tokens: int = 256,
            incremental: bool = False, return_token_nll: bool = False):
    """Perplexity dopo un prefill esatto.

    Il percorso storico invia l'intera continuazione in un solo forward. È
    causale nei logits, ma per le policy di cache il chunk intero appare come
    una regione corrente ancora esatta (particolarmente visibile nel residuo
    KIVI). Con ``incremental=True`` si appende invece un token per volta, come
    nel decode reale, e si accumula la stessa teacher-forced NLL senza
    materializzare tutti i logits.

    Con ``return_token_nll=True`` viene restituita anche la NLL di ogni singolo
    target, nell'ordine dei token. I target dipendono soltanto dalla finestra e
    non dalla policy, quindi due policy valutate sulla stessa finestra
    producono liste allineate: è questo che rende possibile un intervallo di
    confidenza paired invece di un confronto fra sole PPL aggregate.
    """
    pol.reset_state()
    split = ids.shape[1] - eval_tokens
    if split <= 0 or eval_tokens < 2:
        raise ValueError("servono un prefisso non vuoto e almeno due token eval")
    device = next(model.parameters()).device
    out = prefill(model, pol, ids[:, :split].to(device))
    cache = out.past_key_values
    del out

    P.ACTIVE = pol
    model.set_attn_implementation("policy")
    tgt = ids[:, split:].to(device)
    try:
        if not incremental:
            o = model(tgt, past_key_values=cache, use_cache=True)
            lp = torch.nn.functional.log_softmax(o.logits[:, :-1].float(), -1)
            per_token = -lp.gather(-1, tgt[:, 1:, None])[..., 0]
            ppl = float(per_token.mean().exp())
            if return_token_nll:
                return ppl, [float(x) for x in per_token.mean(0)]
            return ppl

        nll_sum = torch.zeros((), device=device, dtype=torch.float32)
        n_targets = 0
        per_token = []
        for i in range(tgt.shape[1] - 1):
            o = model(tgt[:, i:i + 1], past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            lp = torch.nn.functional.log_softmax(
                o.logits[:, -1].float(), dim=-1)
            step = -lp.gather(-1, tgt[:, i + 1:i + 2])
            nll_sum += step.sum()
            n_targets += tgt.shape[0]
            per_token.append(float(step.mean()))
        ppl = float((nll_sum / n_targets).exp())
        if return_token_nll:
            return ppl, per_token
        return ppl
    finally:
        P.ACTIVE = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--depths", type=float, nargs="*",
                    default=[0.1, 0.35, 0.6, 0.85])
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--out", default="paper/results_geodesia_vs_sota.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["policy"] = P.policy_attention
    ALL_ATTENTION_FUNCTIONS["observe"] = P.observation_attention

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    T = args.tokens
    # I baseline a EVICTION hanno bit/valore = 16 * budget / T: per confrontarli con
    # Geodesia-KV (pavimento 2 bit/valore) i budget vanno scalati col contesto, altrimenti
    # a 16k girerebbero a 0.5 bit/valore e il confronto non sarebbe appaiato.
    b_lo, b_hi = T // 8, T // 4            # -> 2 e 4 bit/valore
    p_lo, p_hi = b_lo // 64, b_hi // 64
    configs = [
        ("Full-KV (oracolo, causale)", dict(name="full")),
        (f"StreamingLLM b={b_lo}", dict(name="streaming", budget_tokens=b_lo, sinks=4)),
        (f"StreamingLLM b={b_hi}", dict(name="streaming", budget_tokens=b_hi, sinks=4)),
        (f"SnapKV b={b_lo}", dict(name="snapkv", budget_tokens=b_lo)),
        (f"SnapKV b={b_hi}", dict(name="snapkv", budget_tokens=b_hi)),
        (f"Quest p={p_lo}", dict(name="quest", quest_pages=p_lo, block=64)),
        (f"Quest p={p_hi}", dict(name="quest", quest_pages=p_hi, block=64)),
        ("Geodesia-KV massa aggr.", dict(
            name="geodesia-ub", mode="mass", block=64, window=128, sinks=4,
            tiers=((0.02, 16), (0.08, 8), (0.20, 4), (1.00, 2)))),
        ("Geodesia-KV massa bil.", dict(
            name="geodesia-ub", mode="mass", block=64, window=256, sinks=4,
            tiers=((0.05, 16), (0.15, 8), (0.30, 4), (1.00, 2)))),
        ("Geodesia-KV cert t=0.20", dict(
            name="geodesia-ub", mode="cert", tau=0.20, block=64, window=128, sinks=4)),
        ("Geodesia-KV cert t=0.05", dict(
            name="geodesia-ub", mode="cert", tau=0.05, block=64, window=128, sinks=4)),
        ("Geodesia-KV cert t=0.01", dict(
            name="geodesia-ub", mode="cert", tau=0.01, block=64, window=128, sinks=4)),
    ]

    key = "8 4 7 2 9 1"
    key_flat = key.replace(" ", "")
    key_ids = torch.tensor([tok(" " + key, add_special_tokens=False).input_ids])
    ppl_ids = tok(open(args.text).read(), return_tensors="pt",
                  max_length=T, truncation=True).input_ids

    results = []
    for label, kw in configs:
        pol = P.Policy(**kw)
        pol.report = P.Report()
        t0 = time.time()

        hits, nlls, details = 0, [], []
        for d in args.depths:
            ids = build_passkey(tok, T, d, key)
            txt, nll = run_passkey(model, tok, pol, ids, key_ids)
            ok = key_flat in txt.replace(" ", "")
            hits += ok
            nlls.append(nll)
            details.append({"depth": d, "ok": bool(ok), "nll_key": nll,
                            "out": txt[:40]})
        bpv_pk = pol.report.bits_per_value
        nll_key = sum(nlls) / len(nlls)

        pol.report = P.Report()
        ppl = run_ppl(model, tok, pol, ppl_ids)
        rep = pol.report

        row = {
            "policy": label, "config": {k: str(v) for k, v in kw.items()},
            "passkey_acc": hits / len(args.depths),
            "nll_key": nll_key,
            "passkey_detail": details,
            "bits_per_value": bpv_pk,
            "compression_vs_bf16": 16.0 / max(bpv_pk, 1e-9),
            "ppl": ppl,
            "ppl_bits_per_value": rep.bits_per_value,
            "cert_bound_mean": rep.cert_bound / max(rep.n_certified, 1),
            "true_err_mean": rep.true_err / max(rep.n_certified, 1),
            "cert_violations": rep.n_violations,
            "cert_calls": rep.n_certified,
            "sec": time.time() - t0,
        }
        results.append(row)
        cert = (f" | cert {row['cert_bound_mean']:.3f} vs reale "
                f"{row['true_err_mean']:.4f}, violazioni {row['cert_violations']}"
                f"/{row['cert_calls']}") if rep.n_certified else ""
        print(f"{label:26s} passkey {100*row['passkey_acc']:5.1f}% | "
              f"nll_key {nll_key:6.3f} | ppl {ppl:7.3f} | {bpv_pk:5.2f} bit/val "
              f"({row['compression_vs_bf16']:5.2f}x){cert}  [{row['sec']:.0f}s]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "tokens": T, "depths": args.depths,
                   "results": results}, fh, indent=2)
    print(f"\nscritto {args.out}")


if __name__ == "__main__":
    main()
