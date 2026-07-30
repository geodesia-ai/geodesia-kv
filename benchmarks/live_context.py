"""Contesto massimo raggiunto DAVVERO sotto un tetto di VRAM imposto.

Nessuna simulazione e nessuna aritmetica: si impone un tetto con
``torch.cuda.set_per_process_memory_fraction``, si carica il modello
quantizzato e si allunga il contesto in prefill reali finche' il processo non
esaurisce la memoria. Il contesto riportato e' l'ultimo che ha completato un
prefill e un passo di decode veri.

Le due politiche confrontate usano lo stesso identico percorso del modello:

- ``full``: la ``DynamicCache`` standard, K/V bf16 densa;
- ``quantized``: ``QuantizedCache``, in cui i blocchi chiusi esistono solo
  impacchettati a 2/4/8 bit e la K/V densa non viene mai conservata.

Uso:
  .venv/bin/python -m benchmarks.live_context --model Qwen/Qwen2.5-3B-Instruct \\
      --vram-gib 16 --policies full quantized --bits 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

GIB = 1024 ** 3


def load_model(name: str, quantization: str, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if attn == "gqa_causal":
        from geodesia_kv.live_cache import register_gqa_causal_attention
        register_gqa_causal_attention()
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda",
                  attn_implementation=attn)
    if quantization in ("nf4", "int8"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
            if quantization == "nf4" else
            BitsAndBytesConfig(load_in_8bit=True))
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
    return model, tok


POLICIES = ("full", "quantized", "kivi4", "kivi2", "streaming", "quest")


def make_cache(policy: str, cfg, bits: int, chunk: int, window: int,
               budget: int = 2048):
    """Tutte le politiche sono cache reali sullo stesso percorso del modello."""
    from transformers import DynamicCache
    from geodesia_kv.live_cache import (QuantizedCache, QuestCache,
                                        StreamingCache)
    if policy == "full":
        return DynamicCache(config=cfg)
    if policy == "quantized":
        return QuantizedCache(cfg, bits=bits, chunk=chunk, window=window)
    # KIVI: gruppo 32 e residuo esatto 32, come nel port del paper.
    if policy == "kivi4":
        return QuantizedCache(cfg, bits=4, chunk=chunk, group=32, window=32,
                              sinks=0)
    if policy == "kivi2":
        return QuantizedCache(cfg, bits=2, chunk=chunk, group=32, window=32,
                              sinks=0)
    if policy == "streaming":
        return StreamingCache(cfg, budget=budget)
    if policy == "quest":
        return QuestCache(cfg)
    raise SystemExit(f"politica sconosciuta {policy!r}; usa {POLICIES}")


@torch.no_grad()
def grow(model, cfg, ids, policy: str, target: int, step: int, bits: int,
         chunk: int, window: int, checkpoints: list[int],
         budget: int = 2048) -> dict:
    """Allunga il contesto con prefill reali finche' entra in memoria."""
    device = next(model.parameters()).device
    cache = make_cache(policy, cfg, bits, chunk, window, budget)
    reached, marks, oom = 0, [], None
    t0 = time.time()
    try:
        while reached < target:
            n = min(step, target - reached)
            model(ids[:, reached:reached + n].to(device),
                  past_key_values=cache, use_cache=True)
            reached += n
            if checkpoints and reached >= checkpoints[0]:
                while checkpoints and reached >= checkpoints[0]:
                    checkpoints.pop(0)
                # Un passo di decode vero: prova che il contesto e' usabile,
                # non soltanto allocato.
                out = model(ids[:, reached:reached + 1].to(device),
                            past_key_values=cache, use_cache=True)
                resident = (cache.resident_bytes()
                            if hasattr(cache, "resident_bytes") else
                            sum(l.keys.numel() * l.keys.element_size()
                                + l.values.numel() * l.values.element_size()
                                for l in cache.layers))
                retained = (cache.retained_tokens()
                            if hasattr(cache, "retained_tokens") else reached)
                marks.append({
                    "context": reached,
                    "retained_tokens": retained,
                    "resident_bytes": resident,
                    "bits_per_value": (cache.bits_per_value()
                                       if hasattr(cache, "bits_per_value")
                                       else 16.0),
                    "peak_bytes": torch.cuda.max_memory_allocated(),
                    "logit_finite": bool(torch.isfinite(out.logits).all()),
                    "seconds": time.time() - t0,
                })
                del out
                print(f"  [{policy}] {reached:,} token  "
                      f"residente {resident/GIB:.2f} GiB  "
                      f"picco {torch.cuda.max_memory_allocated()/GIB:.2f} GiB  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                cache.layers[0]  # noqa: B018  (mantiene vivo il riferimento)
    except torch.OutOfMemoryError as exc:
        oom = f"{reached} token: {str(exc)[:120]}"
        print(f"  [{policy}] OOM a {reached:,} token", flush=True)
    finally:
        del cache
        torch.cuda.empty_cache()
    return {"policy": policy, "reached": reached, "oom": oom,
            "checkpoints": marks, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--quantization", choices=["nf4", "int8", "bf16"],
                    default="nf4")
    ap.add_argument("--vram-gib", type=float, required=True,
                    help="tetto imposto al processo")
    ap.add_argument("--policies", nargs="+", default=["full", "quantized"])
    ap.add_argument("--bits", type=int, default=2, choices=[2, 4, 8])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--attn", choices=["sdpa", "gqa_causal"],
                    default="gqa_causal",
                    help="gqa_causal evita repeat_kv sulle KV-head condivise")
    ap.add_argument("--budget", type=int, default=2048,
                    help="token conservati dalle politiche a eviction")
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--target", type=int, default=1_010_000)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=None)
    ap.add_argument("--text", default="paper/pg19_test.txt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.vram_gib * GIB / total
    if fraction > 1:
        raise SystemExit(f"tetto {args.vram_gib} GiB oltre i "
                         f"{total/GIB:.2f} GiB della scheda")
    torch.cuda.set_per_process_memory_fraction(fraction)
    print(f"tetto imposto: {args.vram_gib:g} GiB "
          f"({100*fraction:.1f}% di {total/GIB:.2f} GiB)")

    model, tok = load_model(args.model, args.quantization, args.attn)
    cfg = model.config
    weights = torch.cuda.memory_allocated()
    print(f"pesi {args.quantization}: {weights/GIB:.3f} GiB")

    text = open(args.text).read()
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] < args.target + 2:
        reps = args.target // ids.shape[1] + 2
        ids = ids.repeat(1, reps)
    print(f"corpus disponibile: {ids.shape[1]:,} token")

    default_marks = [2 ** k for k in range(13, 21)]
    results = []
    for policy in args.policies:
        marks = sorted(args.checkpoints) if args.checkpoints else \
            [m for m in default_marks if m <= args.target] + [args.target]
        print(f"\n== {policy} ==", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results.append(grow(model, cfg, ids, policy, args.target,
                            args.prefill_step, args.bits, args.chunk,
                            args.window, marks, args.budget))

    payload = {
        "model": args.model, "quantization": args.quantization,
        "vram_cap_gib": args.vram_gib, "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": total, "weights_bytes": weights,
        "bits": args.bits, "chunk": args.chunk, "window": args.window,
        "prefill_step": args.prefill_step, "text": args.text,
        "attn_implementation": args.attn,
        "results": results,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        payload["git_commit"] = None
    out = args.out or f"results/live_context_{args.vram_gib:g}gib.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nscritto {out}")
    for r in results:
        print(f"{r['policy']:<10} massimo {r['reached']:,} token "
              f"{'(OOM)' if r['oom'] else '(target raggiunto)'}")


if __name__ == "__main__":
    main()
