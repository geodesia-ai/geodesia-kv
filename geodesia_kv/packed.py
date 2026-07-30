"""Storage KV realmente impacchettato: la memoria diventa misurabile, non stimata.

Finora la precisione allocata era SIMULATA (quantizza e dequantizza subito) e i
bit/valore erano contati analiticamente. Per affermare qualcosa sul contesto
massimo su una GPU da 16 GiB serve che i tensori occupino davvero quei bit.

Tre livelli di residenza per blocco, come nella policy:
  esatto     bf16
  quantizzato  interi a 2/4/8 bit impacchettati in uint8 + scale/zero per gruppo
  centroide  una sola chiave e un solo valore bf16 per blocco (0.25 bit/valore
             con blocchi da 64)

`nbytes()` riporta l'occupazione reale; `torch.cuda.memory_allocated` la conferma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def pack_bits(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Impacchetta interi in [0, 2^bits) dentro uint8, lungo l'ultima dimensione.

    q: (..., N) uint8 con N multiplo di (8 // bits). Ritorna (..., N*bits/8).
    """
    assert bits in (2, 4, 8), f"bits non supportati: {bits}"
    if bits == 8:
        return q.contiguous()
    per = 8 // bits
    n = q.shape[-1]
    pad = (-n) % per
    if pad:
        q = torch.nn.functional.pad(q, (0, pad))
    g = q.reshape(*q.shape[:-1], q.shape[-1] // per, per)
    out = torch.zeros(g.shape[:-1], dtype=torch.uint8, device=q.device)
    for i in range(per):
        out |= (g[..., i] & ((1 << bits) - 1)) << (i * bits)
    return out


def unpack_bits(p: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Inverso di `pack_bits`; `n` e' il numero di valori originali."""
    if bits == 8:
        return p[..., :n]
    per = 8 // bits
    outs = [(p >> (i * bits)) & ((1 << bits) - 1) for i in range(per)]
    return torch.stack(outs, dim=-1).reshape(*p.shape[:-1], p.shape[-1] * per)[..., :n]


@dataclass
class PackedBlocks:
    """Blocchi quantizzati a un dato numero di bit, con scale per gruppo."""
    bits: int
    data: torch.Tensor            # uint8 impacchettato
    lo: torch.Tensor              # offset per gruppo
    step: torch.Tensor            # passo per gruppo
    shape: tuple                  # (nb, block, D) originale
    group: int
    per_channel: bool

    def nbytes(self) -> int:
        return (self.data.numel() * self.data.element_size()
                + self.lo.numel() * self.lo.element_size()
                + self.step.numel() * self.step.element_size())

    def dequantize(self) -> torch.Tensor:
        nb, p, d = self.shape
        g = min(self.group, p)
        n = nb * (p // g) * g * d
        q = unpack_bits(self.data.reshape(-1), self.bits, n).float()
        q = q.reshape(nb, p // g, g, d)
        return (q * self.step + self.lo).reshape(nb, p, d)


def quantize_packed(x: torch.Tensor, bits: int, group: int, per_channel: bool
                    ) -> PackedBlocks:
    """x: (nb, block, D) -> blocchi realmente impacchettati."""
    nb, p, d = x.shape
    g = min(group, p)
    xs = x.reshape(nb, p // g, g, d).float()
    if per_channel:
        lo, hi = xs.amin(2, keepdim=True), xs.amax(2, keepdim=True)
    else:
        lo, hi = xs.amin(3, keepdim=True), xs.amax(3, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    q = ((xs - lo) / step).round().clamp_(0, 2 ** bits - 1).to(torch.uint8)
    return PackedBlocks(bits=bits, data=pack_bits(q.reshape(-1), bits),
                        lo=lo.to(torch.float16), step=step.to(torch.float16),
                        shape=(nb, p, d), group=g, per_channel=per_channel)


@dataclass
class PackedKV:
    """Cache KV di UN layer con residenza mista, memoria reale misurabile."""
    exact_k: torch.Tensor | None = None       # (n_exact, D) bf16
    exact_v: torch.Tensor | None = None
    exact_idx: torch.Tensor | None = None     # posizioni dei blocchi esatti
    quant: list = field(default_factory=list)  # [(idx, PackedBlocks_k, PB_v)]
    cent_k: torch.Tensor | None = None        # (n_cent, D) bf16
    cent_v: torch.Tensor | None = None
    cent_idx: torch.Tensor | None = None

    def nbytes(self) -> int:
        n = 0
        for t in (self.exact_k, self.exact_v, self.cent_k, self.cent_v,
                  self.exact_idx, self.cent_idx):
            if t is not None:
                n += t.numel() * t.element_size()
        for _, pk, pv in self.quant:
            n += pk.nbytes() + pv.nbytes()
        return n


def build_packed(k: torch.Tensor, v: torch.Tensor, bits: torch.Tensor,
                 centroid: torch.Tensor, block: int, group: int) -> PackedKV:
    """Costruisce la cache impacchettata dai K,V di un layer.

    k, v: (T, D);  bits: (nb,) bit per blocco;  centroid: (nb,) maschera booleana.
    """
    from geodesia_kv.policies import blockify
    kb, _ = blockify(k, block)
    vb, _ = blockify(v, block)
    nb = kb.shape[0]
    out = PackedKV()

    if centroid.any():
        idx = centroid.nonzero(as_tuple=True)[0]
        out.cent_k = kb[idx].mean(1).to(torch.bfloat16).contiguous()
        out.cent_v = vb[idx].mean(1).to(torch.bfloat16).contiguous()
        out.cent_idx = idx.to(torch.int32)

    exact = (~centroid) & (bits >= 16)
    if exact.any():
        idx = exact.nonzero(as_tuple=True)[0]
        out.exact_k = kb[idx].to(torch.bfloat16).contiguous()
        out.exact_v = vb[idx].to(torch.bfloat16).contiguous()
        out.exact_idx = idx.to(torch.int32)

    todo = (~centroid) & (bits < 16)
    for b in (2, 4, 8):
        m = todo & (bits == b)
        if not m.any():
            continue
        idx = m.nonzero(as_tuple=True)[0]
        out.quant.append((idx.to(torch.int32),
                          quantize_packed(kb[idx], b, group, True),
                          quantize_packed(vb[idx], b, group, False)))
    # i bit non in {2,4,8} vengono promossi al livello superiore disponibile
    leftover = todo & ~((bits == 2) | (bits == 4) | (bits == 8))
    if leftover.any():
        idx = leftover.nonzero(as_tuple=True)[0]
        out.quant.append((idx.to(torch.int32),
                          quantize_packed(kb[idx], 8, group, True),
                          quantize_packed(vb[idx], 8, group, False)))
    return out
