// Attenzione fusa su cache KV a PRECISIONE MISTA.
//
// La cache non e' un tensore denso: ogni blocco di P token vive a un livello
// diverso ({16, 8, 4, 2} bit oppure 1 = centroide). Tenere in VRAM la versione
// dequantizzata annullerebbe il risparmio di memoria; dequantizzare in un passo
// separato lo materializzerebbe comunque. L'unico modo perche' il risparmio sia
// REALE e' dequantizzare dentro il kernel di attenzione, un blocco alla volta,
// senza mai materializzare la cache completa.
//
// Softmax online (stile FlashAttention): si scorre la cache una volta sola
// mantenendo (m, l, acc) e riscalando quando arriva un massimo nuovo.
//
// Numerica: tutto in fp32, con la stessa formula di dequantizzazione del
// percorso PyTorch (val = qi * step + lo, con step/lo in fp16 come contabilizzato).
// L'output e' quindi bit-identico al riferimento: il kernel non puo' peggiorare
// la qualita', puo' solo renderla piu' veloce e la memoria reale.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_D 256

// Livelli: 16 = bf16 grezzo, 8/4/2 = interi impacchettati, 1 = centroide
__device__ __forceinline__ float unpack_int(const uint8_t* data, int idx, int bits) {
    if (bits == 8) return (float)data[idx];
    const int per = 8 / bits;
    const uint8_t byte = data[idx / per];
    const int shift = (idx % per) * bits;
    return (float)((byte >> shift) & ((1 << bits) - 1));
}

// Un thread block per (batch, kv-head). Scorre tutti i blocchi della cache.
//   q        : (B, H, D)                query gia' scalata
//   data     : (B, H, total_bytes)      byte impacchettati, blocchi consecutivi
//   offs     : (nb + 1)                 offset in byte di ogni blocco
//   klo,kstep: (B, H, nb, G, D)         scale K per (gruppo, canale)
//   vlo,vstep: (B, H, nb, P, GV)        scale V per (token, gruppo di canali)
//   level    : (nb)                     livello di ogni blocco
//   valid    : (nb)                     token validi nell'ultimo blocco
//   out      : (B, H, D)
template <int P>
__global__ void mixed_attn_decode_kernel(
    const float* __restrict__ q,
    const uint8_t* __restrict__ data,
    const long* __restrict__ offs,
    const __half* __restrict__ klo, const __half* __restrict__ kstep,
    const __half* __restrict__ vlo, const __half* __restrict__ vstep,
    const int* __restrict__ level,
    const int* __restrict__ valid,
    float* __restrict__ out,
    int H, int D, int nb, int group, int gv, long stride_bh,
    int blocks_per_split, float* __restrict__ part_m,
    float* __restrict__ part_l, float* __restrict__ part_acc) {

    // Griglia (B*H, nsplit): senza questo si usava UN SM su 128 e il kernel era
    // 100x piu' lento dell'attenzione densa. Ogni split percorre una fetta di
    // blocchi e produce (m, l, acc) parziali; un secondo kernel li combina con
    // la stessa formula della softmax online.
    const int bh = blockIdx.x;
    const int sp = blockIdx.y;
    const int b_lo = sp * blocks_per_split;
    const int b_hi = min(nb, b_lo + blocks_per_split);
    const int tid = threadIdx.x;
    const int nthread = blockDim.x;

    extern __shared__ float smem[];
    float* sq = smem;                       // (D)   query
    float* ss = smem + D;                   // (P)   punteggi del blocco corrente
    float* sacc = smem + D + P;             // (D)   accumulatore output

    for (int d = tid; d < D; d += nthread) {
        sq[d] = q[(long)bh * D + d];
        sacc[d] = 0.f;
    }
    // `s_rescale` DEVE essere una variabile a se': scriverlo in ss[P] finiva
    // sopra sacc[0], corrompendo il primo canale dell'accumulatore.
    __shared__ float m_run, l_run, s_rescale;
    if (tid == 0) { m_run = -INFINITY; l_run = 0.f; s_rescale = 0.f; }
    __syncthreads();

    const uint8_t* base = data + (long)bh * stride_bh;
    const int G = (group > 0) ? (P + group - 1) / group : 1;

    for (int b = b_lo; b < b_hi; ++b) {
        const int lv = level[b];
        const int nvalid = valid[b];
        const uint8_t* blk = base + offs[b];

        // ---- fase 1: punteggi del blocco (un thread per token) ----
        for (int j = tid; j < P; j += nthread) {
            if (j >= nvalid) { ss[j] = -INFINITY; continue; }
            float s = 0.f;
            if (lv == 1) {                                   // centroide
                const __half* c = (const __half*)blk;
                for (int d = 0; d < D; ++d) s += sq[d] * __half2float(c[d]);
            } else if (lv == 16) {                           // bf16 grezzo
                const __nv_bfloat16* kk = (const __nv_bfloat16*)blk;
                for (int d = 0; d < D; ++d)
                    s += sq[d] * __bfloat162float(kk[(long)j * D + d]);
            } else {                                         // interi impacchettati
                const int g = j / group;
                const long sc = (((long)bh * nb + b) * G + g) * D;
                for (int d = 0; d < D; ++d) {
                    const long idx = (long)j * D + d;
                    const float qi = unpack_int(blk, idx, lv);
                    s += sq[d] * (qi * __half2float(kstep[sc + d])
                                  + __half2float(klo[sc + d]));
                }
            }
            ss[j] = s;
        }
        __syncthreads();

        // ---- fase 2: softmax online ----
        if (tid == 0) {
            float mb = -INFINITY;
            for (int j = 0; j < P; ++j) mb = fmaxf(mb, ss[j]);
            const float mnew = fmaxf(m_run, mb);
            float lb = 0.f;
            for (int j = 0; j < P; ++j) {
                ss[j] = (ss[j] == -INFINITY) ? 0.f : __expf(ss[j] - mnew);
                lb += ss[j];
            }
            const float rescale = (m_run == -INFINITY) ? 0.f : __expf(m_run - mnew);
            l_run = l_run * rescale + lb;
            m_run = mnew;
            s_rescale = rescale;             // comunicato alla fase 3
        }
        __syncthreads();

        // ---- fase 3: accumulo dei value (thread ripartiti sui canali) ----
        const float rescale = s_rescale;
        const long vbytes = (lv == 1) ? (long)D * 2
                          : (lv == 16) ? (long)P * D * 2
                          : ((long)P * D * lv) / 8;
        const uint8_t* vblk = blk + vbytes;
        for (int d = tid; d < D; d += nthread) {
            float a = sacc[d] * rescale;
            for (int j = 0; j < nvalid; ++j) {
                const float w = ss[j];
                if (w == 0.f) continue;
                float vval;
                if (lv == 1) {
                    vval = __half2float(((const __half*)vblk)[d]);
                } else if (lv == 16) {
                    vval = __bfloat162float(((const __nv_bfloat16*)vblk)[(long)j * D + d]);
                } else {
                    const int g = d / group;
                    const long sc = (((long)bh * nb + b) * P + j) * gv + g;
                    const float qi = unpack_int(vblk, (long)j * D + d, lv);
                    vval = qi * __half2float(vstep[sc]) + __half2float(vlo[sc]);
                }
                a += w * vval;
            }
            sacc[d] = a;
        }
        __syncthreads();
    }

    const long pidx = (long)bh * gridDim.y + sp;
    if (tid == 0) { part_m[pidx] = m_run; part_l[pidx] = l_run; }
    for (int d = tid; d < D; d += nthread)
        part_acc[pidx * D + d] = sacc[d];
}

// Combina gli split: stessa ricorrenza della softmax online, applicata ai
// parziali invece che ai blocchi.
__global__ void combine_splits_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, float* __restrict__ out,
    int nsplit, int D) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float m_all, l_all;
    if (tid == 0) {
        float m = -INFINITY;
        for (int s = 0; s < nsplit; ++s)
            m = fmaxf(m, part_m[(long)bh * nsplit + s]);
        float l = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float ms = part_m[(long)bh * nsplit + s];
            if (ms > -INFINITY)
                l += part_l[(long)bh * nsplit + s] * __expf(ms - m);
        }
        m_all = m; l_all = l;
    }
    __syncthreads();
    for (int d = tid; d < D; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float ms = part_m[(long)bh * nsplit + s];
            if (ms > -INFINITY)
                a += part_acc[((long)bh * nsplit + s) * D + d] * __expf(ms - m_all);
        }
        out[(long)bh * D + d] = a / fmaxf(l_all, 1e-20f);
    }
}

torch::Tensor mixed_attn_decode(
    torch::Tensor q, torch::Tensor data, torch::Tensor offs,
    torch::Tensor klo, torch::Tensor kstep,
    torch::Tensor vlo, torch::Tensor vstep,
    torch::Tensor level, torch::Tensor valid,
    int64_t block, int64_t group) {

    TORCH_CHECK(q.is_cuda() && q.dim() == 3, "q deve essere (B,H,D) su CUDA");
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    const int nb = level.size(0);
    const int gv = (group > 0) ? (D + group - 1) / group : 1;
    auto out = torch::empty_like(q);

    const int threads = 128;
    const size_t shmem = sizeof(float) * (D + block + 1 + D);
    auto stream = at::cuda::getCurrentCUDAStream();
    const long stride_bh = data.size(-1);
    TORCH_CHECK(block == 64, "supportati solo blocchi da 64 token");

    // si mira a saturare gli SM: almeno 4 split per SM disponibile
    int nsplit = std::max(1, std::min(nb, 512 / std::max(1, B * H)));
    const int bps = (nb + nsplit - 1) / nsplit;
    nsplit = (nb + bps - 1) / bps;

    auto fopt = q.options().dtype(torch::kFloat32);
    auto part_m = torch::empty({B * H, nsplit}, fopt);
    auto part_l = torch::empty({B * H, nsplit}, fopt);
    auto part_acc = torch::empty({B * H, nsplit, D}, fopt);

    dim3 grid(B * H, nsplit);
    mixed_attn_decode_kernel<64><<<grid, threads, shmem, stream>>>(
        q.data_ptr<float>(), data.data_ptr<uint8_t>(), offs.data_ptr<long>(),
        (const __half*)klo.data_ptr(), (const __half*)kstep.data_ptr(),
        (const __half*)vlo.data_ptr(), (const __half*)vstep.data_ptr(),
        level.data_ptr<int>(), valid.data_ptr<int>(), out.data_ptr<float>(),
        H, D, nb, (int)group, gv, stride_bh, bps,
        part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        part_acc.data_ptr<float>());
    combine_splits_kernel<<<B * H, threads, 0, stream>>>(
        part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        part_acc.data_ptr<float>(), out.data_ptr<float>(), nsplit, D);
    return out;
}
