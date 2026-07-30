# Geodesia-KV

Certified allocation KV cache compression: deploy extremely long contexts on small GPUs with proven error bounds instead of empirical hope.

Geodesia-KV tackles the problem of KV cache memory explosion by assigning **graded, variable precision** to different blocks of tokens. Rather than uniform quantization (which collapses at 2-bits) or eviction (which loses information permanently), Geodesia-KV uses a rate-distortion Lagrangian allocator to assign the perfect bit-depth (`{16, 8, 4, 2, centroid}`) to each block. The result is an equipotential surface of marginal errors, guaranteeing optimal memory reduction with near-zero quality degradation.

## Features

- **No Eviction, No Uniform Quantization**: Keeps the entire context but compresses blocks based on their causal attention mass.
- **Monotonic Demotion**: A block is smoothly demoted across `{16, 8, 4, 2, 1}` bit levels and is never promoted, eliminating the need to keep a dense fallback copy in VRAM.
- **Production vLLM Integration**: Works out-of-the-box with vLLM `0.26.0`, enabling massive VRAM savings for long-context model serving.
- **Custom CUDA Kernel**: Includes a fast `mixed_attn_decode` kernel that decodes the heterogeneous bit-ladder on the fly (benchmark isolated from FlashAttention).

---

## Installation

Geodesia-KV can be easily installed via pip. Ensure you have `torch>=2.4` and a CUDA toolkit installed if you want to compile the custom kernels.

```bash
# Clone the repository
git clone https://github.com/geodesia-ai/geodesia-kv.git
cd geodesia-kv

# Install via pip
pip install -e .

# (Optional) To build the custom CUDA kernel, ninja must be installed
pip install ninja
```

---

## Integrating with vLLM (Production Usage)

Geodesia-KV provides a seamless integration plugin for **vLLM** (`0.26.0`), allowing you to dramatically reduce the VRAM footprint of your serving infrastructure.

> **Note:** You must disable v1 multiprocessing to ensure the cache manager can inject its states correctly.

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from vllm import LLM, SamplingParams
from geodesia_kv.vllm_plugin.vllm_connector import register_geodesia_kv_plugin

# 1. Initialize vLLM normally
llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=1,
    max_model_len=16384,
    enforce_eager=True, # Recommended for custom plugins
)

# 2. Register the Geodesia-KV Cache Manager
# target_bpv sets the target bits-per-value (e.g. 4.0 bits = 4x VRAM reduction)
manager = register_geodesia_kv_plugin(llm, target_bpv=4.0, block_size=64)

# 3. Generate as usual!
prompts = ["Summarize the history of Rome in a highly detailed essay."]
sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### How the vLLM Plugin Works
The `register_geodesia_kv_plugin` bypasses standard vLLM proxies and injects a custom `GeodesiaKVCacheManager` that intercepts block allocations. It enforces the rate-distortion bit ladder across the paged memory, achieving over **71% VRAM reduction** on an RTX 4090 without degrading retrieval accuracy!

---

## Standard PyTorch Usage (HuggingFace Transformers)

If you are just running scripts using HuggingFace `transformers`, you can monkey-patch the attention implementation dynamically:

```python
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from geodesia_kv import policies as P

# Register Geodesia-KV policy
ALL_ATTENTION_FUNCTIONS["geodesia"] = P.policy_attention
P.ACTIVE = P.Policy(name="geodesia", budget_bits=3.0, group=64, window=128)

# Load the model using the registered implementation
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", 
    attn_implementation="geodesia",
    device_map="auto"
)
```
No training, no weight modification: the plugin registers a causal attention implementation and hooks into the forward pass.

---

## Evaluation Results

Tested on `Qwen2.5-3B-Instruct` and `Qwen3-8B` at 16k contexts (WikiText-2, Multi-turn QA, Passkey Retrieval).

| Method | Bits/Value | Perplexity | VRAM Savings |
|---|---:|---:|---:|
| Full-KV (Oracle) | 16.00 | 5.452 | 0.0% |
| **Geodesia-KV, budget 3** | **2.96** | **5.453** | **81.5%** |
| KIVI k4v4 | 5.03 | 5.499 | 68.5% |
| SnapKV b=2048 | 2.01 | 5.742 | 87.4% (Fails Multi-turn) |

**All baselines are strictly Pareto-dominated**: there is always a Geodesia-KV configuration with less memory *and* better quality. Unlike eviction methods (SnapKV/StreamingLLM) that permanently lose context and fail multi-turn interactions, Geodesia-KV retains **100% retrieval accuracy**.

## Testing

```bash
pytest tests/ -q
```
The test suite validates causal integrity, rate-distortion bounds, packed representations, and vLLM integration. 

## License
Apache-2.0
