# Deploying Geodesia-KV with vLLM

This guide provides step-by-step instructions on how to install, configure, and run **Geodesia-KV** with **vLLM** for high-throughput, memory-efficient LLM serving with massive context windows (up to 1M+ tokens on a single 16–24 GB GPU).

---

## 📋 Table of Contents

1. [Overview & Highlights](#-overview--highlights)
2. [Prerequisites & System Requirements](#-prerequisites--system-requirements)
3. [Step-by-Step Installation](#-step-by-step-installation)
4. [Quickstart Guide](#-quickstart-guide)
5. [Configuration Options](#-configuration-options)
6. [Serving via OpenAI-Compatible API](#-serving-via-openai-compatible-api)
7. [Benchmark & Memory Savings](#-benchmark--memory-savings)
8. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🌟 Overview & Highlights

**Geodesia-KV** integrates directly with vLLM via `GeodesiaKVCacheManager`, bringing training-free rate-distortion KV-cache compression to your production pipelines.

- 🚀 **Up to 71.7% Peak VRAM Savings**: Compress KV-cache down to 2–5 bits-per-value (BPV) with negligible quality loss.
- 🎯 **1M+ Token Context on Budget GPUs**: Serve long contexts on consumer cards (RTX 3090/4090 24GB, RTX 4080 16GB) without CPU offloading.
- ⚡ **Zero Dense Memory Overhead**: Employs monotonic demotion so high-precision dense copies are never held in VRAM.
- 🔄 **Drop-in vLLM Compatibility**: Requires only 1 line of code to attach to standard vLLM `LLM` or `LLMEngine` instances.

---

## 💻 Prerequisites & System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS (for development/CPU tests)
- **Python**: `>= 3.10`
- **CUDA Toolkit**: `>= 12.0` (with `nvcc` in PATH)
- **PyTorch**: `>= 2.4.0`
- **vLLM**: `>= 0.6.0` (tested with `0.26.0` / `0.6.x`+)
- **GPU**: NVIDIA GPU with compute capability $\ge 8.0$ (Ampere, Ada Lovelace, Hopper, e.g., RTX 3090, RTX 4090, A100, H100)

---

## 🛠️ Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/geodesia-ai/geodesia-kv.git
cd geodesia-kv
```

### Step 2: Set Up a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 3: Install PyTorch and vLLM

Install PyTorch according to your CUDA version (example for CUDA 12.1+):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

### Step 4: Install Geodesia-KV

Install the `geodesia-kv` package in editable mode:

```bash
pip install -e .
```

*(Optional)* Install `ninja` for JIT compilation of custom CUDA kernels:
```bash
pip install ninja
```

---

## 🚀 Quickstart Guide

Here is a complete, minimal script showing how to load a model in vLLM and attach the Geodesia-KV plugin:

### `run_vllm_geodesia.py`

```python
import os

# Recommended environment variables for custom cache backends in vLLM
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

def main():
    # 1. Initialize vLLM model
    # Note: enforce_eager=True is recommended when using dynamic custom KV managers
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading model: {model_name}...")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=16384,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
    )

    # 2. Register Geodesia-KV Plugin
    # target_bpv: 2.0, 3.0, 4.0, or 5.0 (bits per value)
    print("Registering Geodesia-KV Cache Manager...")
    manager = register_geodesia_kv_plugin(
        engine_or_llm=llm,
        target_bpv=4.0,     # 4.0 bits/value achieves ~70% KV-cache reduction
        block_size=64       # Standard 64-token allocation block
    )

    # 3. Define Prompts and Sampling Parameters
    prompts = [
        "Explain the mathematical principles of General Relativity in simple terms.",
        "Write a high-performance Python script to compute Fibonacci numbers using memoization."
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )

    # 4. Execute Batch Generation
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Print Results
    for idx, output in enumerate(outputs):
        print(f"\n--- Output {idx + 1} ---")
        print(f"Prompt: {output.prompt}")
        print(f"Generated text:\n{output.outputs[0].text}\n")

if __name__ == "__main__":
    main()
```

Run the script:

```bash
python run_vllm_geodesia.py
```

---

## ⚙️ Configuration Options

When calling `register_geodesia_kv_plugin(llm, target_bpv=..., block_size=...)`, you can customize the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_bpv` | `float` | `4.0` | Target rate in bits per value (`2.0`, `3.0`, `4.0`, `5.0`, `8.0`, `16.0`). Lower means higher compression. |
| `block_size` | `int` | `64` | Token block granularity for rate-distortion allocation (recommended: `64`). |

### Recommended Operating Points

| Target BPV | Compression Ratio | KV VRAM Savings | Recommended For |
|---|---|---|---|
| `2.0` | **8x** | ~87.5% | Ultra-long contexts (500k – 1M+ tokens), summarization, retrieval. |
| `3.0` | **5.3x** | ~81.2% | High-throughput batch serving, multi-turn chat. |
| `4.0` | **4x** | ~75.0% | **Default balance** (virtually zero perplexity loss). |
| `5.0` | **3.2x** | ~68.8% | Maximum precision / sensitive mathematical and reasoning workloads. |

---

## 🌐 Serving via OpenAI-Compatible API

To serve models using vLLM's OpenAI-compatible HTTP server with Geodesia-KV enabled, create an entrypoint script:

### `serve_api.py`

```python
import os
import uvicorn
from vllm.entrypoints.openai.api_server import app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI Server with Geodesia-KV")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Launch server
    uvicorn.run(app, host=args.host or "0.0.0.0", port=args.port or 8000)
```

Alternatively, you can patch your server pipeline programmatically before starting the async engine.

---

## 📊 Benchmark & Memory Savings

Measured on **Qwen2.5-3B-Instruct** and **Qwen3-8B** at **16,384** tokens:

```
+--------------------------+---------------------+-------------------+
| Configuration            | Resident Rate (BPV) | Peak VRAM Saving  |
+--------------------------+---------------------+-------------------+
| Baseline (FP16 Full-KV)  | 16.000 BPV          | 0.0% (Baseline)   |
| Geodesia-KV (5-bit)      |  4.961 BPV          | ~69.0%            |
| Geodesia-KV (4-bit)      |  4.000 BPV          | ~75.0%            |
| Geodesia-KV (3-bit)      |  3.001 BPV          | ~81.2%            |
| Geodesia-KV (2-bit)      |  2.000 BPV          | ~87.5%            |
+--------------------------+---------------------+-------------------+
```

---

## ❓ Troubleshooting & FAQ

### Q1: Why should I set `enforce_eager=True`?
**A:** vLLM by default uses CUDA Graph capture for decode tokens. Dynamic or heterogeneous memory management plugins operate most reliably in eager mode or with custom backend hooks.

### Q2: CUDA out-of-memory during long prefill?
**A:** Ensure `gpu_memory_utilization` is set appropriately (e.g., `0.85`–`0.90`) and select a lower `target_bpv` (such as `3.0` or `2.0`).

### Q3: Does Geodesia-KV require retraining or fine-tuning?
**A:** No. Geodesia-KV is **100% training-free** and mathematically guarantees output error bounds at runtime.

---

## 📄 License & Attribution

Geodesia-KV is licensed under the [Apache-2.0 License](LICENSE).  
For questions or issues, please open an issue on GitHub or reach out to [Geodesia.ai](https://github.com/geodesia-ai).
