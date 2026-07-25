"""Il "pozzo" puo' funzionare? Misura di concentrazione dell'attenzione e di
TENUTA del certificato d'errore.

Idea sotto test: mantenere in VRAM solo un riassunto a dimensione fissa per blocco di
token (box per canale: min/max di K, piu' ||v||max) e usarlo per calcolare un
UPPER BOUND RIGOROSO sul punteggio di attenzione di quel blocco. Si recuperano i
blocchi in ordine di bound decrescente finche' la massa residua CERTIFICATA scende
sotto epsilon; il resto del contesto resta offloadato e non viene mai letto.

Se la massa e' concentrata e il bound e' stretto, si ottiene contesto illimitato con
errore certificato a runtime. Se il bound e' lasco, l'idea muore qui.

Misura, su un modello locale gia' addestrato (training-free):
  - frazione di blocchi necessaria per catturare 1-eps della massa (oracolo);
  - frazione necessaria usando il BOUND (cio' che il sistema puo' davvero fare);
  - VIOLAZIONI del certificato (deve essere zero: e' un bound, non una stima);
  - errore reale dell'output di attenzione a ogni eps.

Uso:
  .venv/bin/python experiment_well_bound.py --model Qwen2.5-3B-Instruct \
      --tokens 16384 --queries 64 --block 64
"""

from __future__ import annotations

import argparse
import json
import os

import torch


CAPTURED: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
TARGET_LAYERS: set[int] = set()


def capture_attention(module, query, key, value, attention_mask, scaling,
                      dropout=0.0, **kwargs):
    """Wrapper: registra (q, k, v) post-RoPE e delega all'implementazione eager."""
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
    li = getattr(module, "layer_idx", None)
    if li in TARGET_LAYERS:
        CAPTURED[li] = (query.detach(), key.detach(), value.detach())
    return eager_attention_forward(module, query, key, value, attention_mask,
                                   scaling, dropout=dropout, **kwargs)


def block_box_bound(q: torch.Tensor, k: torch.Tensor, block: int, scaling: float):
    """Upper bound rigoroso su max_j (q . k_j) per ogni blocco di `block` chiavi.

    q: (Q, D)   k: (T, D)  ->  (Q, n_blocks)
    Per ogni blocco B, k vive nella scatola [kmin, kmax] per canale. Allora
        max_{k in B} q.k <= sum_c max(q_c*kmin_c, q_c*kmax_c)
    che e' il massimo sulla scatola: rigoroso perche' la scatola contiene B.
    """
    t = k.shape[0]
    nb = (t + block - 1) // block
    pad = nb * block - t
    kk = torch.cat([k, k[-1:].expand(pad, -1)], 0) if pad else k
    kk = kk.view(nb, block, -1)
    kmin = kk.min(dim=1).values          # (nb, D)
    kmax = kk.max(dim=1).values
    # (Q,1,D) * (1,nb,D) -> massimo per canale sulla scatola
    pos = torch.maximum(q[:, None, :] * kmin[None], q[:, None, :] * kmax[None])
    return pos.sum(-1) * scaling, nb, pad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-3B-Instruct")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["capture"] = capture_attention

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(open(args.text).read(), return_tensors="pt").input_ids[:, : args.tokens]
    n = ids.shape[1]
    q_len = args.queries
    print(f"contesto {n} token, {q_len} query, blocchi da {args.block}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()
    n_layers = model.config.num_hidden_layers
    global TARGET_LAYERS
    TARGET_LAYERS = {0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1}

    with torch.no_grad():
        out = model(ids[:, : n - q_len].cuda(), use_cache=True)
        cache = out.past_key_values
        del out
        model.set_attn_implementation("capture")
        model(ids[:, n - q_len:].cuda(), past_key_values=cache, use_cache=True)

    eps_list = [1e-1, 1e-2, 1e-3]
    results = []
    for li in sorted(TARGET_LAYERS):
        q, k, v = CAPTURED[li]                     # (1,H,Q,D) (1,Hkv,T,D)
        q = q[0].float()
        k = k[0].float()
        v = v[0].float()
        H, Q, D = q.shape
        Hkv, T, _ = k.shape
        rep = H // Hkv
        scaling = D ** -0.5

        stats = {"layer": li, "heads": H, "kv_heads": Hkv, "T": T,
                 "oracle_frac": {}, "bound_frac": {}, "err": {}, "violations": 0,
                 "bound_slack": None}
        oracle_f = {e: [] for e in eps_list}
        bound_f = {e: [] for e in eps_list}
        err = {e: [] for e in eps_list}
        slack = []
        viol = 0

        for h in range(H):
            kh, vh = k[h // rep], v[h // rep]
            qh = q[h]                                       # (Q,D)
            scores = (qh @ kh.T) * scaling                  # (Q,T)
            # mascheratura causale rispetto alle ultime Q posizioni
            pos = torch.arange(T, device=scores.device)
            causal = pos[None, :] <= (T - Q + torch.arange(Q, device=scores.device))[:, None]
            scores = scores.masked_fill(~causal, float("-inf"))
            w = torch.softmax(scores, -1)                   # (Q,T)
            o_full = w @ vh                                 # (Q,D)

            bnd, nb, pad = block_box_bound(qh, kh, args.block, scaling)   # (Q,nb)
            # massa esatta per blocco
            wp = torch.nn.functional.pad(w, (0, pad))
            mass = wp.view(Q, nb, args.block).sum(-1)       # (Q,nb)

            # il bound deve dominare il punteggio massimo reale del blocco
            sp = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
            smax = sp.view(Q, nb, args.block).max(-1).values
            finite = torch.isfinite(smax)
            viol += int(((smax > bnd + 1e-3) & finite).sum())
            slack.append(float((bnd[finite] - smax[finite]).mean()))

            # ordinamento oracolo (massa vera) e ordinamento pratico (bound)
            ord_or = mass.argsort(dim=-1, descending=True)
            ord_bd = bnd.masked_fill(~finite, float("-inf")).argsort(dim=-1, descending=True)
            cum_or = mass.gather(-1, ord_or).cumsum(-1)
            cum_bd = mass.gather(-1, ord_bd).cumsum(-1)

            for e in eps_list:
                oracle_f[e].append(float(((cum_or < 1 - e).sum(-1) + 1).float().mean() / nb))
                need = (cum_bd < 1 - e).sum(-1) + 1
                bound_f[e].append(float(need.float().mean() / nb))
                # errore reale tenendo solo i blocchi selezionati dal bound
                keep = torch.zeros(Q, nb, dtype=torch.bool, device=w.device)
                keep.scatter_(1, ord_bd, torch.arange(nb, device=w.device)[None, :]
                              < need[:, None])
                m = keep.repeat_interleave(args.block, 1)[:, :T]
                wm = w * m
                wm = wm / wm.sum(-1, keepdim=True).clamp_min(1e-9)
                err[e].append(float(((wm @ vh - o_full).norm(dim=-1)
                                     / o_full.norm(dim=-1).clamp_min(1e-9)).mean()))
            del scores, w, o_full, bnd, mass, wp, sp

        for e in eps_list:
            stats["oracle_frac"][str(e)] = sum(oracle_f[e]) / H
            stats["bound_frac"][str(e)] = sum(bound_f[e]) / H
            stats["err"][str(e)] = sum(err[e]) / H
        stats["violations"] = viol
        stats["bound_slack"] = sum(slack) / len(slack)
        results.append(stats)

        print(f"\n--- layer {li} ({H} teste, {Hkv} kv, T={T}, {(T+args.block-1)//args.block} blocchi) ---")
        print(f"  violazioni del bound: {viol}   slack medio: {stats['bound_slack']:.3f}")
        for e in eps_list:
            print(f"  eps={e:<6g} blocchi: oracolo {100*stats['oracle_frac'][str(e)]:5.2f}%"
                  f" | con bound {100*stats['bound_frac'][str(e)]:5.2f}%"
                  f" -> errore reale output {stats['err'][str(e)]:.2e}")
        del CAPTURED[li]

    out_path = args.out or f"paper/results_well_bound_{os.path.basename(args.model)}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"model": args.model, "tokens": n, "queries": q_len,
                   "block": args.block, "layers": results}, fh, indent=2)
    print(f"\nscritto {out_path}")


if __name__ == "__main__":
    main()
