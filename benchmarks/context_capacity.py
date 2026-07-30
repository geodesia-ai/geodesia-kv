"""Massimo contesto residente in VRAM con il modello quantizzato.

Domanda: tenendo in VRAM il modello quantizzato e la KV cache compressa, quanto
contesto entra, e quale tecnica ne tiene di piu'?

Qui non si simula nulla sul lato memoria. Vengono misurati:

1. i byte reali dei pesi dopo il caricamento quantizzato;
2. il picco reale di attivazione di un passo di decode Q=1;
3. i byte reali per token del formato packed Geodesia, costruito da K/V veri del
   modello con la distribuzione di livelli scelta dall'allocatore vero;
4. il massimo contesto realmente allocabile accanto ai pesi, cercato per
   bisezione e verificato allocando davvero i tensori;
5. per Full-KV, un passo di decode realmente eseguito su quella cache.

Restano invece proiezioni, e sono etichettate come tali, i byte per token delle
baseline che questo repository non implementa in formato residente (KIVI,
Quest): per quelle si usa la contabilita' bit/valore gia' validata dai run di
qualita'.

ATTENZIONE ALL'INTERPRETAZIONE. Il contesto massimo per VRAM non e' il contesto
massimo utile: Qwen3-8B e' addestrato a 40,960 posizioni. Superarlo richiede
RoPE scaling e non e' dimostrato qui. Il numero prodotto da questo script dice
quanta memoria serve, non quanta memoria il modello sappia usare.

Uso:
  .venv/bin/python -m benchmarks.context_capacity --model Qwen/Qwen3-8B \
      --quantization nf4 --out results/qwen8b_context_capacity_nf4.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

GIB = 1024 ** 3

# bit/valore residenti misurati dai run di qualita'. La fonte di ciascun numero
# e' esplicita: nessuna costante inventata per questa tabella.
TECHNIQUES = [
    ("Full-KV BF16", 16.0, "retain",
     "definizione: bf16 senza compressione"),
    ("Quest p=32", 16.250489, "retain",
     "qwen8b_pg19_16k_quest_vs_compressed_quest_incremental_eleven_books.json"),
    ("Compressed-Quest B=10", 9.836099, "retain",
     "qwen8b_pg19_16k_quest_vs_compressed_quest_incremental_eleven_books.json"),
    ("KIVI-4", 5.031156, "retain",
     "qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_eleven_books.json"),
    ("Geodesia graded+RD 5-bit", 4.989738, "retain",
     "qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_eleven_books.json"),
    ("KIVI-2", 3.050781, "retain",
     "qwen8b_16k_ten_window_corrected_axes_eval64.json"),
    ("Geodesia RD-V2 2-bit", 2.061, "retain",
     "qwen8b_16k_rd_saliency_frozen_holdout_offsets196608_229376.json"),
    # Le policy a eviction non hanno una cache che cresce: ne conservano un
    # numero fisso di token. Il loro "contesto massimo" per VRAM e' illimitato,
    # ma il contesto RITENUTO resta il budget. Vanno lette in una riga a parte.
    ("StreamingLLM b=2048", None, "evict", "budget fisso 2048 token bf16"),
    ("SnapKV b=2048", None, "evict", "budget fisso 2048 token bf16"),
]


def load_model(name: str, quantization: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dict(dtype=torch.bfloat16, device_map="cuda",
                  attn_implementation="sdpa")
    if quantization in ("nf4", "int8"):
        from transformers import BitsAndBytesConfig
        if quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tok


def kv_values_per_token(cfg) -> int:
    """Valori K+V memorizzati per ogni token, su tutti i layer."""
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim


def make_cache(cfg, length: int, device, dtype=torch.bfloat16):
    """Cache reale della lunghezza richiesta, allocata davvero in VRAM.

    Il costruttore di ``DynamicCache`` copia i tensori che riceve: passargli
    direttamente la cache grande farebbe toccare un picco doppio, che e' un
    artefatto del banco di prova e non un costo del decode. Si costruisce
    quindi una cache minima e si assegnano i buffer definitivi ai layer, che
    vengono cosi' allocati una volta sola.
    """
    from transformers import DynamicCache
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    seed = (1, cfg.num_key_value_heads, 1, head_dim)
    cache = DynamicCache([(torch.zeros(seed, device=device, dtype=dtype),
                           torch.zeros(seed, device=device, dtype=dtype))
                          for _ in range(cfg.num_hidden_layers)])
    shape = (1, cfg.num_key_value_heads, length, head_dim)
    for layer in cache.layers:
        layer.keys = torch.zeros(shape, device=device, dtype=dtype)
        layer.values = torch.zeros(shape, device=device, dtype=dtype)
    return cache


@torch.no_grad()
def measure_decode_overhead(model, cfg, length: int) -> dict:
    """Picco di attivazione di un vero passo Q=1 su una cache gia' piena."""
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    cache = make_cache(cfg, length, device)
    with_cache = torch.cuda.memory_allocated()
    ids = torch.tensor([[1]], device=device)
    pos = torch.tensor([[length]], device=device)
    torch.cuda.reset_peak_memory_stats()
    model(ids, past_key_values=cache, use_cache=True, cache_position=pos)
    peak = torch.cuda.max_memory_allocated()
    cache_bytes = with_cache - before
    del cache
    torch.cuda.empty_cache()
    return {"cache_len": length, "cache_bytes": cache_bytes,
            "activation_peak_bytes": peak - with_cache}


@torch.no_grad()
def real_packed_bytes_per_token(model, tok, cfg, text: str, length: int,
                                budget: float, block: int, group: int) -> dict:
    """Byte reali per token del formato packed, da K/V veri del modello.

    Si esegue un prefill vero, si legge la K/V di un layer, si fa scegliere i
    livelli all'allocatore vero e si misura ``PackedKV.nbytes()``. Nessuna
    formula: e' la dimensione del buffer che verrebbe allocato.
    """
    from geodesia_kv.packed import build_packed
    from geodesia_kv.policies import Policy, Report

    device = next(model.parameters()).device
    ids = tok(open(text).read(), return_tensors="pt", max_length=length,
              truncation=True).input_ids.to(device)
    out = model(ids, use_cache=True)
    layer = out.past_key_values.layers[cfg.num_hidden_layers // 2]
    k = layer.keys[0, 0].float()          # (T, D) di una KV-head reale
    v = layer.values[0, 0].float()
    del out
    torch.cuda.empty_cache()

    # L'allocatore vero, alimentato dalle ultime query reali del prefill.
    pol = Policy(name="geodesia", budget_bits=budget, block=block, group=group,
                 window=128, sinks=4, validate=False)
    pol.report = Report()
    q = k[-8:].clone()
    pol.attend(q, k, v, k.shape[-1] ** -0.5, q_offset=k.shape[0] - 8,
               hid=(0, 0, 0))
    st = pol.state[(0, 0, 0)]
    levels = st["level"]
    packed = build_packed(k, v, levels, levels <= 1, block, group)
    nbytes = packed.nbytes()
    per_token_head = nbytes / k.shape[0]
    return {
        "context": k.shape[0],
        "packed_bytes_one_head": nbytes,
        "bytes_per_token_all_layers":
            per_token_head * cfg.num_key_value_heads * cfg.num_hidden_layers,
        "bits_per_value_packed":
            nbytes * 8 / (k.shape[0] * k.shape[-1] * 2),
        "bits_per_value_accounted": pol.report.bits_per_value,
    }


@torch.no_grad()
def largest_allocatable_context(model, cfg, bytes_per_token: float,
                                lo: int, hi: int, run_decode: bool) -> dict:
    """Cerca per bisezione il contesto che si alloca davvero accanto ai pesi.

    Per Full-KV la cache e' quella vera del modello e viene anche eseguito un
    passo di decode: e' una dimostrazione, non una stima. Per i formati
    compressi si allocano buffer opachi della dimensione esatta calcolata, il
    che prova la residenza ma non l'esecuzione.
    """
    device = next(model.parameters()).device
    best, best_peak = 0, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if run_decode:
                buf = make_cache(cfg, mid, device)
                ids = torch.tensor([[1]], device=device)
                pos = torch.tensor([[mid]], device=device)
                model(ids, past_key_values=buf, use_cache=True,
                      cache_position=pos)
            else:
                buf = torch.empty(int(mid * bytes_per_token),
                                  dtype=torch.uint8, device=device)
            best, best_peak = mid, torch.cuda.max_memory_allocated()
            lo = mid + 1
        except torch.OutOfMemoryError:
            hi = mid - 1
        finally:
            del buf
            torch.cuda.empty_cache()
    return {"context": best, "peak_bytes": best_peak,
            "verified": "decode reale" if run_decode else "allocazione reale"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--quantization", choices=["nf4", "int8", "bf16"],
                    default="nf4")
    ap.add_argument("--text", default="paper/wikitext2_test.txt")
    ap.add_argument("--packed-context", type=int, default=16384)
    ap.add_argument("--packed-budget", type=float, default=2.0)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--overhead-context", type=int, default=8192)
    ap.add_argument("--concurrency-contexts", type=int, nargs="*",
                    default=[40960, 131072])
    ap.add_argument("--skip-search", action="store_true",
                    help="salta la bisezione, riporta solo il calcolo")
    ap.add_argument("--vram-gib", type=float, nargs="*", default=[],
                    help="schede ipotetiche: proietta la capacita' a questa "
                         "VRAM nominale usando pesi e attivazione misurati")
    ap.add_argument("--emulate-vram-gib", type=float, nargs="*", default=[],
                    help="occupa la VRAM in eccesso con una zavorra e ripete "
                         "la bisezione reale sotto il tetto richiesto")
    ap.add_argument("--out", default="results/qwen8b_context_capacity.json")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = load_model(args.model, args.quantization)
    cfg = model.config
    weights_bytes = torch.cuda.memory_allocated()
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    free_bytes, _ = torch.cuda.mem_get_info()
    print(f"pesi {args.quantization}: {weights_bytes/GIB:.3f} GiB  "
          f"({time.time()-t0:.0f}s)  GPU {total_bytes/GIB:.1f} GiB")

    overhead = measure_decode_overhead(model, cfg, args.overhead_context)
    print(f"attivazione decode Q=1: {overhead['activation_peak_bytes']/2**20:.1f} MiB")

    packed = real_packed_bytes_per_token(
        model, tok, cfg, args.text, args.packed_context, args.packed_budget,
        args.block, args.group)
    print(f"packed reale B={args.packed_budget:g}: "
          f"{packed['bits_per_value_packed']:.3f} bit/valore misurati contro "
          f"{packed['bits_per_value_accounted']:.3f} contabilizzati")

    vals = kv_values_per_token(cfg)
    # Budget disponibile per la cache: memoria libera meno il picco di
    # attivazione realmente misurato.
    budget_bytes = free_bytes - overhead["activation_peak_bytes"]

    rows = []
    for label, bits, kind, source in TECHNIQUES:
        if kind == "evict":
            resident = 2048 * vals * 2          # 2048 token bf16
            rows.append({
                "policy": label, "kind": kind, "bits_per_value": None,
                "bytes_per_token": None,
                "resident_bytes": resident,
                "max_context_vram": None,
                "retained_context": 2048,
                "source": source})
            continue
        bpt = vals * bits / 8
        rows.append({
            "policy": label, "kind": kind, "bits_per_value": bits,
            "bytes_per_token": bpt,
            "resident_bytes": None,
            "max_context_vram": int(budget_bytes // bpt),
            "retained_context": int(budget_bytes // bpt),
            "source": source})

    searched = {}
    if not args.skip_search:
        full = next(r for r in rows if r["policy"] == "Full-KV BF16")
        print("\nbisezione Full-KV con decode reale...")
        searched["Full-KV BF16"] = largest_allocatable_context(
            model, cfg, full["bytes_per_token"], 1024,
            int(full["max_context_vram"] * 1.2), run_decode=True)
        print(f"  {searched['Full-KV BF16']['context']:,} token")
        geo = next(r for r in rows if r["policy"] == "Geodesia RD-V2 2-bit")
        print("bisezione Geodesia 2-bit con allocazione reale...")
        searched["Geodesia RD-V2 2-bit"] = largest_allocatable_context(
            model, cfg, geo["bytes_per_token"], 1024,
            int(geo["max_context_vram"] * 1.2), run_decode=False)
        print(f"  {searched['Geodesia RD-V2 2-bit']['context']:,} token")

    # Frazione di VRAM nominale che il driver lascia davvero disponibile,
    # misurata su questa scheda e riusata per le schede ipotetiche.
    nominal_gib = round(total_bytes / GIB)
    usable_fraction = total_bytes / (nominal_gib * GIB)

    projections = {}
    for vram in args.vram_gib:
        budget = vram * GIB * usable_fraction - weights_bytes \
            - overhead["activation_peak_bytes"]
        projections[str(vram)] = {
            "usable_bytes": vram * GIB * usable_fraction,
            "cache_budget_bytes": budget,
            "fits_model": budget > 0,
            "max_context": {
                r["policy"]: (None if r["kind"] == "evict"
                              else int(budget // r["bytes_per_token"]))
                for r in rows},
        }

    emulated = {}
    for vram in args.emulate_vram_gib:
        cap = vram * GIB * usable_fraction
        ballast_bytes = int(total_bytes - cap)
        if ballast_bytes <= 0:
            print(f"\n{vram:g} GiB non e' emulabile: la scheda ne ha "
                  f"{total_bytes/GIB:.2f}")
            continue
        print(f"\nemulazione {vram:g} GiB: zavorra "
              f"{ballast_bytes/GIB:.2f} GiB, bisezione Full-KV con decode reale")
        ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device="cuda")
        try:
            full = next(r for r in rows if r["policy"] == "Full-KV BF16")
            hi = int(max(1024, (cap - weights_bytes) // full["bytes_per_token"]))
            found = largest_allocatable_context(
                model, cfg, full["bytes_per_token"], 512, hi, run_decode=True)
        finally:
            del ballast
            torch.cuda.empty_cache()
        emulated[str(vram)] = found
        print(f"  Full-KV reale: {found['context']:,} token")

    native = getattr(cfg, "max_position_embeddings", None)
    print(f"\n{'tecnica':<28}{'bit/val':>8}{'KiB/token':>11}"
          f"{'contesto max VRAM':>20}{'x finestra nativa':>19}")
    for r in rows:
        if r["kind"] == "evict":
            print(f"{r['policy']:<28}{'-':>8}{'costante':>11}"
                  f"{'illimitato':>20}{'ritiene 2048':>19}")
            continue
        ratio = r["max_context_vram"] / native if native else float("nan")
        print(f"{r['policy']:<28}{r['bits_per_value']:8.2f}"
              f"{r['bytes_per_token']/1024:11.1f}"
              f"{r['max_context_vram']:20,}{ratio:18.1f}x")

    payload = {
        "model": args.model, "quantization": args.quantization,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": total_bytes,
        "weights_bytes": weights_bytes,
        "free_after_load_bytes": free_bytes,
        "decode_overhead": overhead,
        "packed_real": packed,
        "kv_values_per_token": vals,
        "cache_budget_bytes": budget_bytes,
        "native_context": native,
        "results": rows,
        "verified_search": searched,
        "nominal_gib": nominal_gib,
        "usable_fraction": usable_fraction,
        "projected_vram": projections,
        "emulated_vram": emulated,
        "concurrency": {
            str(ctx): {r["policy"]: (
                None if r["kind"] == "evict"
                else int(budget_bytes // (r["bytes_per_token"] * ctx)))
                for r in rows}
            for ctx in args.concurrency_contexts},
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        payload["git_commit"] = None
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nscritto {args.out}")


if __name__ == "__main__":
    main()
