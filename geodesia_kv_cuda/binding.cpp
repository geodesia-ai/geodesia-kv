#include <torch/extension.h>

torch::Tensor mixed_attn_decode(
    torch::Tensor q, torch::Tensor data, torch::Tensor offs,
    torch::Tensor klo, torch::Tensor kstep,
    torch::Tensor vlo, torch::Tensor vstep,
    torch::Tensor level, torch::Tensor valid,
    int64_t block, int64_t group);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mixed_attn_decode", &mixed_attn_decode,
          "Attenzione fusa su cache KV a precisione mista (decode)");
}
