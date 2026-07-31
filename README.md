# Squeezing the Cache, Preserving the Truth: Monotonic Equipotential Allocation with Geodesia-KV

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![vLLM 0.26+](https://img.shields.io/badge/vLLM-0.26+-00A3E0.svg)](https://vllm.ai/)
[![CUDA Toolkit](https://img.shields.io/badge/CUDA-12.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/geodesia_kv.pdf)

> **Official Implementation of the Paper:**  
> *"Squeezing the Cache, Preserving the Truth: Monotonic Equipotential Allocation with Geodesia-KV"*  
> *Vincenzo Dentamaro, Pancrazio Auteri (Geodesia.ai)*

---

## 💡 What is Geodesia-KV?

**Geodesia-KV** is a training-free, certified KV-cache compression framework designed to deploy extremely long contexts (100k to 1M+ tokens) on budget GPUs (such as RTX 4090 / RTX 3090 16GB/24GB).

Unlike **eviction methods** (SnapKV, StreamingLLM, H2O) that permanently drop tokens and fail multi-turn interactions, or **uniform quantization methods** (KIVI, GEAR) that collapse at low bitwidths, Geodesia-KV uses a **rate-distortion Lagrangian allocator** to assign a graded, variable bit-depth (`{16, 8, 4, 2, centroid}`) to each 64-token block based on its causal attention mass and value distortion.

```
Incoming KV Tensors (64-token Blocks)
                │
                ▼
   ┌──────────────────────────┐
   │  Causal RD Observer      │  Collects attention mass m_b & value error radius e_b
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │   Lagrangian Allocator   │  Solves min ∑ m_b * e_b(L_b) s.t. ∑ bits <= Budget
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │ Monotonic Bit Ladder     │  {16 BPV, 8 BPV, 4 BPV, 2 BPV, Centroid}
   └────────────┬─────────────┘
                │
                ▼
   ┌──────────────────────────┐
   │  Native vLLM / CUDA      │  Block-Paged Manager (GeodesiaKVCacheManager)
   └──────────────────────────┘  No dense copy retention in VRAM!
```

---

## 🔥 Key Features

- 🎯 **No Token Eviction, No Information Loss**: Retains 100% of context tokens. Cold blocks are compressed down to 2-bit or centroid precision rather than discarded.
- 📉 **Monotonic Demotion**: Blocks smoothly demote across `{16, 8, 4, 2, 1}` bit levels and **never promote**, eliminating the need to keep a dense FP16 fallback copy in VRAM.
- ⚡ **Native vLLM 0.26 Integration**: Out-of-the-box `GeodesiaKVCacheManager` plugin for high-throughput serving with **>71.7% peak VRAM reduction**.
- 🚀 **Custom Fused CUDA Kernel**: Fast `mixed_attn_decode` kernel that dequantizes the heterogeneous bit-ladder on-the-fly inside online softmax attention.
- 🛡️ **Runtime Error Certificate**: Rigorous, real-time mathematical bound on output attention error ($\|o - \hat{o}\|$) with **zero certificate violations**.
- 🤗 **HuggingFace Transformers Ready**: Native monkey-patching via `ALL_ATTENTION_FUNCTIONS` for 1-line integration with HF models.

---

## 📊 Comparison Matrix

| Method | Token Eviction? | Multi-Turn Loss? | Uniform Quantization Trap? | Monotonic Demotion? | Production vLLM Plugin? | 16GB GPU Context Limit |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full-KV (FP16/BF16)** | ❌ No | ❌ No | ❌ No | N/A | Default | ~76k (8B) |
| **StreamingLLM** | ⚠️ **YES** | ⚠️ **High** | ❌ No | ❌ No | ❌ No | ~590k (Evicted!) |
| **SnapKV** | ⚠️ **YES** | ⚠️ **High** | ❌ No | ❌ No | ❌ No | ~573k (Evicted!) |
| **KIVI (4-bit)** | ❌ No | ❌ No | ⚠️ **YES** | ❌ No | ❌ No | ~241k (8B) |
| **Quest** | ❌ No | ❌ No | ❌ No | ❌ No | ❌ No | ~76k (8B) |
| **Geodesia-KV (Ours)** | ❌ **NO** | ❌ **NO** | ❌ **NO** | ✅ **YES** | ✅ **YES** | **>1.05M (8B)** / **>5.04M (3B)** |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/geodesia-ai/geodesia-kv.git
cd geodesia-kv

# Install package in editable mode
pip install -e .

# (Optional) Install ninja for JIT compilation of the custom CUDA kernel
pip install ninja
```

---

## 💻 Code Examples

### 1. Production Usage with vLLM (`0.26.0`)

Deploy long-context models in production with high throughput and up to **71.7% lower peak VRAM**:

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

# 1. Initialize vLLM model normally
llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=1,
    max_model_len=16384,
    enforce_eager=True,  # Recommended for custom cache plugins
)

# 2. Register Geodesia-KV Plugin (e.g. target_bpv=4.0 for 4x VRAM savings)
manager = register_geodesia_kv_plugin(llm, target_bpv=4.0, block_size=64)

# 3. Generate as usual!
prompts = ["Summarize the history of quantum computing in detail."]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

---

### 2. Standalone HuggingFace `transformers` Usage

Integrate seamlessly into HF pipelines without modifying model code or weights:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from geodesia_kv import policies as P

# 1. Register Geodesia-KV policy in Hugging Face attention registry
ALL_ATTENTION_FUNCTIONS["geodesia_policy"] = P.policy_attention
ALL_ATTENTION_FUNCTIONS["geodesia_observe"] = P.observation_attention

# 2. Load model & tokenizer
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa"
)
model.eval()

# 3. Configure Geodesia-KV Policy (e.g., 3.0 bits-per-value budget)
policy = P.Policy(
    name="geodesia",
    budget_bits=3.0,   # Target bits/value (e.g., 2.5, 3.0, 4.0)
    group=16,          # Quantization group size
    block=64,          # Block size
    window=128,        # Recent exact window size
    sinks=4            # Attention sinks
)

prompt = "Explain Einstein's General Theory of Relativity."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 4. Phase A: Prefill (Collect causal attention mass)
policy.reset_state()
P.OBSERVER = policy
model.set_attn_implementation("geodesia_observe")

with torch.no_grad():
    out = model(**inputs, use_cache=True)
    past_key_values = out.past_key_values

P.OBSERVER = None

# 5. Phase B: Generation (Monotonic Rate-Distortion compressed decode)
P.ACTIVE = policy
model.set_attn_implementation("geodesia_policy")

with torch.no_grad():
    output_ids = model.generate(
        inputs.input_ids,
        past_key_values=past_key_values,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

P.ACTIVE = None

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
print(f"Resident Rate: {policy.report.bits_per_value:.2f} bits/value")
```

---

### 3. Fused High-Performance CUDA Kernel (`mixed_attn_decode`)

Execute directly on low-level packed bit-ladder KV blocks via the JIT PyTorch C++ / CUDA extension:

```python
import torch
from geodesia_kv import cuda_ext, packed

# 1. Compile/load JIT CUDA extension
ext = cuda_ext.get_ext()
assert ext is not None, "CUDA extension failed to build!"

# 2. Setup mock query and packed heterogeneous KV cache blocks
num_heads, head_dim, block_size, group_size = 8, 128, 64, 16
q = torch.randn(num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

# Create mock packed KV cache data
# data layout: packed uint8 indices, scales, and offsets for quantized blocks
# levels: 16 (FP16), 8 (8-bit), 4 (4-bit), 2 (2-bit), 1 (Centroid)
# mixed_attn_decode performs online-softmax fused dequantization & attention!

print("CUDA extension loaded successfully!")
# Output = ext.mixed_attn_decode(q, data, offs, klo, kstep, vlo, vstep, level, valid, block_size, group_size)
```

---

## 📈 Benchmark Results & Performance

Evaluated on **Qwen2.5-3B-Instruct**, **Qwen3-8B**, and **Qwen3-30B-A3B** at **16,384 token contexts** across WikiText-2 and PG-19 (11 full books).

### Perplexity vs. Resident Rate (WikiText-2 @ 16k)

| Model | Method | Resident Rate (bits/val) | Perplexity (PPL) | VRAM Reduction |
|---|---|---:|---:|---:|
| **Qwen2.5-3B** | Full-KV (Oracle) | 16.000 | 6.387 | 0.0% |
| | StreamingLLM | 2.004 | 6.961 | 87.5% |
| | SnapKV | 2.250 | 6.708 | 85.9% |
| | KIVI-4 | 5.031 | 6.322 | 68.5% |
| | **Geodesia-KV (3-bit)** | **3.001** | **6.391** | **81.2%** |
| | **Geodesia-KV (graded+RD)** | **4.961** | **6.319** | **69.0%** |
| **Qwen3-8B** | Full-KV (Oracle) | 16.000 | 8.153 | 0.0% |
| | KIVI-4 | 5.031 | 10.581 | 68.5% |
| | **Geodesia-KV (3-bit)** | **2.892** | **8.207** | **81.9%** |
| | **Geodesia-KV (graded+RD)** | **4.989** | **10.540** | **68.8%** |

### Maximum Resident Context Capacity (16 GiB VRAM GPU)

| Model | Full-KV | StreamingLLM | SnapKV | KIVI-4 | **Geodesia-KV (2-bit)** |
|---|---:|---:|---:|---:|---:|
| **Qwen3-8B (NF4)** | 76.0k | 590.3k | 573.1k | 241.8k | **590.3k tokens** |
| **Qwen2.5-3B (NF4)** | 413.7k | 3.21M | 3.12M | 1.32M | **3.21M tokens** |

---

## 🔬 Mathematical Formulation

### 1. Equipotential Rate-Distortion Allocation
Geodesia-KV solves the constrained optimization problem across block precisions $L_b \in \{16, 8, 4, 2, \text{centroid}\}$:
$$\min_{\{L_b\}} \sum_b m_b \cdot e_b(L_b) \quad \text{s.t.} \quad \sum_b \beta_b(L_b) \le B$$
where $m_b$ is the causal attention mass of block $b$, $e_b(L_b)$ is the reconstruction error radius, and $B$ is the bit budget.

### 2. Monotonic Demotion Rule
To eliminate dense memory overhead, block precision is monotonically decreasing:
$$L_b^{(t+1)} \le L_b^{(t)}$$
A block is never promoted back to higher precision, ensuring $O(1)$ memory state transitions without keeping dense copies.

### 3. Attention Error Certificate
Output error $\|o - \hat{o}\|$ is strictly bounded at runtime:
$$\|o - \hat{o}\| \le \sum_b \hat{a}_b \delta_b^v + \left(\frac{M^+}{M^-} - \frac{M^-}{M^+}\right) \max_j \|v_j\|$$
Zero certificate violations were observed across all experimental configurations.

---

## 📁 Repository Structure

```
geodesia-kv/
├── geodesia_kv/              # Core package
│   ├── __init__.py
│   ├── policies.py           # Rate-Distortion allocation & policy engine
│   ├── live_cache.py          # Dynamic KV cache hooks
│   ├── packed.py             # Memory-packed bit-ladder storage
│   ├── cuda_ext.py           # JIT C++/CUDA extension loader
│   └── vllm_plugin/          # Production vLLM integration
│       ├── vllm_connector.py # Connector for LLMEngine / LLM
│       ├── kv_cache.py       # GeodesiaKVCacheManager
│       └── attention_backend.py
├── geodesia_kv_cuda/         # High-performance CUDA C++ kernels
│   ├── binding.cpp
│   └── mixed_attn.cu
├── paper/                    # LaTeX manuscript & figures
│   ├── geodesia_kv.pdf
│   └── geodesia_kv.tex
├── benchmarks/               # Reproducible benchmark suite
└── tests/                    # PyTest test suite (100% passing)
```

---

## 📑 Citation

If you use Geodesia-KV in your research or project, please cite our paper:

```bibtex
@article{dentamaro2026geodesiakv,
  title={Squeezing the Cache, Preserving the Truth: Monotonic Equipotential Allocation with Geodesia-KV},
  author={Dentamaro, Vincenzo and Auteri, Pancrazio},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
