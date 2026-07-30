"""Il kernel CUDA deve essere numericamente IDENTICO al percorso PyTorch.

Il vincolo e' esplicito: il kernel puo' rendere la memoria reale e l'esecuzione
veloce, ma non puo' cambiare di una virgola la qualita'. Qui si costruisce una
cache a precisione mista (tutti e cinque i livelli presenti), si esegue
l'attenzione col kernel e con il riferimento PyTorch sugli STESSI valori
dequantizzati, e si confrontano.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="serve CUDA")


def quantize_codes(x, bits, group, per_channel):
    """Come `quant_grouped` ma restituisce anche i codici interi."""
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
    rec = (code * step16.float() + lo16.float()).view(nb, pp, d)
    return code.view(nb, pp, d).to(torch.uint8), rec, lo16, step16


@cuda
def test_kernel_matches_pytorch_reference():
    from geodesia_kv.cuda_ext import get_ext
    from geodesia_kv.pack_layout import build_layout
    ext = get_ext()
    if ext is None:
        pytest.skip("estensione CUDA non compilabile in questo ambiente")

    torch.manual_seed(0)
    dev = "cuda"
    P, D, group = 64, 64, 32
    levels = [16, 8, 4, 2, 1, 4, 2, 8]          # tutti i livelli rappresentati
    nb = len(levels)
    T = nb * P
    k = torch.randn(nb, P, D, device=dev)
    v = torch.randn(nb, P, D, device=dev)

    G = (P + group - 1) // group
    gv = (D + group - 1) // group
    items, krec, vrec = {}, [], []
    for b, lv in enumerate(levels):
        if lv == 16:
            items[b] = {"k": k[b], "v": v[b]}
            krec.append(k[b].to(torch.bfloat16).float())
            vrec.append(v[b].to(torch.bfloat16).float())
        elif lv == 1:
            kc = k[b].mean(0, keepdim=True).to(torch.float16)
            vc = v[b].mean(0, keepdim=True).to(torch.float16)
            items[b] = {"k": kc, "v": vc}
            krec.append(kc.float().expand(P, D))
            vrec.append(vc.float().expand(P, D))
        else:
            ck, rk, klo, kstep = quantize_codes(k[b:b + 1], lv, group, True)
            cv, rv, vlo, vstep = quantize_codes(v[b:b + 1], lv, group, False)
            items[b] = {"k": ck[0], "v": cv[0],
                        "klo": klo.view(G, D), "kstep": kstep.view(G, D),
                        "vlo": vlo.view(P, gv), "vstep": vstep.view(P, gv)}
            krec.append(rk[0])
            vrec.append(rv[0])

    lay = build_layout(items, P, group, D, nb,
                       torch.tensor(levels, device=dev),
                       torch.full((nb,), P, device=dev), dev)

    # Verifica esplicitamente anche il percorso H>1 usato dal benchmark:
    # livelli/layout sono condivisi, mentre ogni KV-head ha una query distinta.
    H = 4
    q = torch.randn(1, H, D, device=dev)
    out = ext.mixed_attn_decode(
        q.contiguous(),
        lay["data"].view(1, 1, -1).expand(1, H, -1).contiguous(),
        lay["offs"],
        lay["klo"].view(1, 1, nb, G, D).expand(
            1, H, -1, -1, -1).contiguous(),
        lay["kstep"].view(1, 1, nb, G, D).expand(
            1, H, -1, -1, -1).contiguous(),
        lay["vlo"].view(1, 1, nb, P, gv).expand(
            1, H, -1, -1, -1).contiguous(),
        lay["vstep"].view(1, 1, nb, P, gv).expand(
            1, H, -1, -1, -1).contiguous(),
        lay["level"], lay["valid"], P, group)

    kk = torch.cat(krec).view(T, D)
    vv = torch.cat(vrec).view(T, D)
    ref = torch.softmax(q.view(H, D) @ kk.T, -1) @ vv

    diff = (out.view(H, D) - ref.view(H, D)).abs().max()
    rel = diff / ref.abs().max()
    assert rel < 1e-4, f"kernel diverso dal riferimento: {float(rel):.2e}"


@cuda
def test_packed_memory_is_real():
    """La cache impacchettata deve occupare davvero i byte dichiarati."""
    from geodesia_kv.pack_layout import build_layout
    torch.manual_seed(1)
    dev = "cuda"
    P, D, group, nb = 64, 64, 32, 32
    k = torch.randn(nb, P, D, device=dev)
    items = {}
    for b in range(nb):
        ck, _, klo, kstep = quantize_codes(k[b:b + 1], 2, group, True)
        cv, _, vlo, vstep = quantize_codes(k[b:b + 1], 2, group, False)
        G, gv = (P + group - 1) // group, (D + group - 1) // group
        items[b] = {"k": ck[0], "v": cv[0], "klo": klo.view(G, D),
                    "kstep": kstep.view(G, D), "vlo": vlo.view(P, gv),
                    "vstep": vstep.view(P, gv)}
    lay = build_layout(items, P, group, D, nb,
                       torch.full((nb,), 2, device=dev),
                       torch.full((nb,), P, device=dev), dev)
    raw = 2 * nb * P * D * 2                      # bf16
    assert lay["nbytes"] < raw / 2, "il packing non riduce la memoria"
    assert lay["data"].dtype == torch.uint8
