"""Misure isolate di cache packed e kernel, non throughput end-to-end.

Tutte le tabelle precedenti riportano bit/valore CONTABILIZZATI: la cache viveva
comunque in bf16 e il risparmio era una scrittura sul registro, non byte
risparmiati. Qui la cache e' costruita nella forma realmente impacchettata
(`uint8` + scale) e l'attenzione passa dal kernel CUDA fuso, quindi:

  - la VRAM misurata con `torch.cuda.max_memory_allocated` e' quella vera;
  - il kernel include realmente la dequantizzazione;
  - tutte le KV-head di un layer vengono lanciate in parallelo;
  - i layer restano stimati serialmente, quindi il risultato è costo
    attention-only, non tok/s di generazione del modello.

Confronto contro la cache bf16 densa alla stessa lunghezza di contesto, e contro
i baseline a eviction (che occupano meno perche' BUTTANO token: il confronto
onesto e' a parita' di informazione conservata, non di byte).

Uso:
  .venv/bin/python experiment_system_metrics.py --model Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

from geodesia_kv.cuda_ext import get_ext
from geodesia_kv.pack_layout import build_layout


def quantize_codes(x, bits, group, per_channel):
    """Codici interi + scale fp16, identici al percorso di riferimento."""
    nb, pp, d = x.shape
    if per_channel:
        g = min(group, pp)
        xs = x.view(nb, pp // g, g, d)
        lo, hi = xs.amin(2, keepdim=True), xs.amax(2, keepdim=True)
    else:
        g = min(group, d)
        xs = x.view(nb, pp, d // g, g)
        lo, hi = xs.amin(3, keepdim=True), xs.amax(3, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    lo16, step16 = lo.to(torch.float16), step.to(torch.float16)
    code = ((xs - lo16.float()) / step16.float()).round().clamp_(0, 2 ** bits - 1)
    return code.view(nb, pp, d).to(torch.uint8), lo16, step16


def build_cache(k, v, levels, block, group, dev):
    """Costruisce la cache impacchettata e libera i tensori densi."""
    nb, P, D = k.shape
    G, gv = (P + group - 1) // group, (D + group - 1) // group
    items = {}
    for b, lv in enumerate(levels):
        if lv == 16:
            items[b] = {"k": k[b], "v": v[b]}
        elif lv == 1:
            items[b] = {"k": k[b].mean(0, keepdim=True).to(torch.float16),
                        "v": v[b].mean(0, keepdim=True).to(torch.float16)}
        else:
            ck, klo, kstep = quantize_codes(k[b:b + 1], lv, group, True)
            cv, vlo, vstep = quantize_codes(v[b:b + 1], lv, group, False)
            items[b] = {"k": ck[0], "v": cv[0], "klo": klo.view(G, D),
                        "kstep": kstep.view(G, D), "vlo": vlo.view(P, gv),
                        "vstep": vstep.view(P, gv)}
    return build_layout(items, block, group, D, nb,
                        torch.tensor(levels, device=dev),
                        torch.full((nb,), P, device=dev), dev)


def real_levels(k, v, budget, block, group, window, sinks):
    """Livelli prodotti dall'ALLOCATORE VERO su K/V VERI del modello.

    Nessuna tabella inventata: si esegue la policy di produzione e si legge la
    distribuzione dei livelli che ha effettivamente scelto.
    """
    from geodesia_kv.policies import Policy, Report
    T, D = k.shape
    pol = Policy(name="geodesia", budget_bits=budget, block=block, group=group,
                 window=window, sinks=sinks, validate=False)
    pol.report = Report()
    q = torch.randn(8, D, device=k.device)
    pol.attend(q, k, v, D ** -0.5, q_offset=T - 8, hid=(0, 0, 0))
    st = pol.state[(0, 0, 0)]
    return [int(x) for x in st["level"].tolist()], pol.report.bits_per_value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--contexts", type=int, nargs="*",
                    default=[16384, 65536, 131072])
    ap.add_argument("--budgets", type=float, nargs="*", default=[3.0, 2.0])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--text", default="paper/wikitext2_valid.txt")
    ap.add_argument("--out", default="paper/results_system_metrics.json")
    args = ap.parse_args()

    ext = get_ext()
    if ext is None:
        raise SystemExit("kernel CUDA non disponibile")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(args.model).to_dict()
    t = cfg.get("text_config", cfg)
    # modelli ibridi (Qwen3.5): solo i layer full_attention hanno una KV cache
    lt = t.get("layer_types")
    L = (sum(1 for x in lt if x == "full_attention") if lt
         else t["num_hidden_layers"])
    Hkv = t["num_key_value_heads"]
    D = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    block, group, dev = 64, 64, "cuda"
    print(f"{args.model}: {L} layer, {Hkv} kv-head, head_dim {D} "
          f"-> {2 * L * Hkv * D * 2 / 1024:.1f} KiB/token in bf16\n")

    # --- K/V REALI da un forward del modello su testo reale -----------------
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    max_ctx = max(args.contexts)
    ids = tok(open(args.text).read(), return_tensors="pt",
              max_length=max_ctx, truncation=True).input_ids

    rows = []
    for T in args.contexts:
        if ids.shape[1] < T:
            print(f"ctx {T}: testo insufficiente ({ids.shape[1]} token), salto")
            continue
        nb = T // block
        with torch.no_grad():
            out = model(ids[:, :T].cuda(), use_cache=True)
        cache = out.past_key_values
        # un layer centrale, rappresentativo; la struttura e' identica per layer
        li = L // 2
        k_real = cache.layers[li].keys[0, 0].float().contiguous()
        v_real = cache.layers[li].values[0, 0].float().contiguous()
        del out, cache
        torch.cuda.empty_cache()
        k = k_real[: nb * block].view(nb, block, D)
        v = v_real[: nb * block].view(nb, block, D)
        dense_bytes = 2 * T * D * 2 * Hkv * L

        for budget in args.budgets:
            lv, bpv_alloc = real_levels(k_real[: nb * block], v_real[: nb * block],
                                        budget, block, group, 128, 4)
            hist = {x: lv.count(x) for x in (16, 8, 4, 2, 1)}
            print(f"    livelli scelti dall'allocatore: {hist}")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            lay = build_cache(k, v, lv, block, group, dev)
            cache_bytes = torch.cuda.memory_allocated() - before
            total_bytes = cache_bytes * Hkv * L

            G, gv = (block + group - 1) // group, (D + group - 1) // group
            # Il kernel supporta H>1: replicare la testa rappresentativa
            # permette di misurare il parallelismo reale intra-layer. Il
            # precedente script misurava H=1 e moltiplicava per Hkv, trattando
            # artificialmente le teste come seriali.
            q = torch.randn(1, Hkv, D, device=dev)
            data_h = lay["data"].view(1, 1, -1).expand(
                1, Hkv, -1).contiguous()
            klo_h = lay["klo"].view(1, 1, nb, G, D).expand(
                1, Hkv, -1, -1, -1).contiguous()
            kstep_h = lay["kstep"].view(1, 1, nb, G, D).expand(
                1, Hkv, -1, -1, -1).contiguous()
            vlo_h = lay["vlo"].view(1, 1, nb, block, gv).expand(
                1, Hkv, -1, -1, -1).contiguous()
            vstep_h = lay["vstep"].view(1, 1, nb, block, gv).expand(
                1, Hkv, -1, -1, -1).contiguous()
            argl = (q.contiguous(), data_h,
                    lay["offs"], klo_h, kstep_h, vlo_h, vstep_h,
                    lay["level"], lay["valid"], block, group)
            for _ in range(3):
                ext.mixed_attn_decode(*argl)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                ext.mixed_attn_decode(*argl)
            torch.cuda.synchronize()
            ms_layer = (time.time() - t0) / args.steps * 1000
            ms_token = ms_layer * L               # layer seriali, head parallele

            # Riferimento denso: stessa testa replicata Hkv volte e un unico
            # lancio batched per layer, coerente con la misura packed.
            kk = k.view(T, D).to(torch.bfloat16)[None].expand(
                Hkv, -1, -1).contiguous()
            vv = v.view(T, D).to(torch.bfloat16)[None].expand(
                Hkv, -1, -1).contiguous()
            qb = q.view(Hkv, 1, D).to(torch.bfloat16)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                prob = torch.softmax(torch.bmm(
                    qb, kk.transpose(1, 2)).float(), -1)
                torch.bmm(prob, vv.float())
            torch.cuda.synchronize()
            ms_dense_layer = (time.time() - t0) / args.steps * 1000
            ms_dense = ms_dense_layer * L

            row = {"context": T, "budget_bits": budget,
                   "level_histogram": hist, "bits_per_value_allocator": bpv_alloc,
                   "cache_GiB_measured": total_bytes / 2 ** 30,
                   "cache_GiB_dense_bf16": dense_bytes / 2 ** 30,
                   "compression": dense_bytes / max(total_bytes, 1),
                   "bits_per_value": 16 * total_bytes / dense_bytes,
                   "ms_per_layer_geodesia_all_kv_heads": ms_layer,
                   "ms_per_layer_dense_all_kv_heads": ms_dense_layer,
                   "ms_per_token_geodesia": ms_token,
                   "ms_per_token_dense": ms_dense,
                   "tok_s_geodesia": 1000.0 / ms_token,
                   "tok_s_dense": 1000.0 / ms_dense,
                   "throughput_scope": (
                       "attention-only estimate: measured all KV heads "
                       "in parallel, multiplied by serial layers")}
            rows.append(row)
            print(f"ctx {T:>7} budget {budget:.1f}: "
                  f"cache {row['cache_GiB_measured']:6.2f} GiB vs "
                  f"{row['cache_GiB_dense_bf16']:6.2f} GiB dense "
                  f"= {row['compression']:5.2f}x ({row['bits_per_value']:.2f} bit/val) | "
                  f"attn {ms_token:7.2f} ms/tok vs {ms_dense:6.2f} densa")
            del lay
            torch.cuda.empty_cache()
        del k, v
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "layers": L, "kv_heads": Hkv,
                   "head_dim": D, "rows": rows}, fh, indent=2)
    print(f"\nscritto {args.out}")


if __name__ == "__main__":
    main()
