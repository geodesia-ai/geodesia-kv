"""Impacchettamento della cache nel layout atteso dal kernel CUDA.

Il punto di tutto: la cache resta in memoria SOLO in forma impacchettata. Non
esiste mai un tensore dequantizzato completo - e' cio' che rende il risparmio di
memoria reale e non contabile.
"""
from __future__ import annotations

import torch

from geodesia_kv.packed import pack_bits


def build_layout(krec_q: dict, block: int, group: int, D: int, nb: int,
                 levels: torch.Tensor, valid: torch.Tensor, device):
    """Assembla (data, offs, klo, kstep, vlo, vstep) per un singolo (batch, head).

    `krec_q[b]` = (k_int, v_int, klo, kstep, vlo, vstep) oppure tensori bf16/half
    a seconda del livello. Restituisce anche il numero di byte realmente occupati.
    """
    G = (block + group - 1) // group
    gv = (D + group - 1) // group
    chunks, offs = [], [0]
    klo = torch.zeros(nb, G, D, dtype=torch.float16, device=device)
    kstep = torch.zeros(nb, G, D, dtype=torch.float16, device=device)
    vlo = torch.zeros(nb, block, gv, dtype=torch.float16, device=device)
    vstep = torch.zeros(nb, block, gv, dtype=torch.float16, device=device)

    for b in range(nb):
        lv = int(levels[b])
        item = krec_q[b]
        if lv == 1:
            buf = torch.cat([item["k"].view(-1).to(torch.float16).view(torch.uint8),
                             item["v"].view(-1).to(torch.float16).view(torch.uint8)])
        elif lv == 16:
            buf = torch.cat([item["k"].reshape(-1).to(torch.bfloat16).view(torch.uint8),
                             item["v"].reshape(-1).to(torch.bfloat16).view(torch.uint8)])
        else:
            kp = pack_bits(item["k"].reshape(-1).to(torch.uint8), lv)
            vp = pack_bits(item["v"].reshape(-1).to(torch.uint8), lv)
            buf = torch.cat([kp, vp])
            klo[b], kstep[b] = item["klo"], item["kstep"]
            vlo[b], vstep[b] = item["vlo"], item["vstep"]
        chunks.append(buf)
        offs.append(offs[-1] + buf.numel())

    data = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.uint8,
                                                        device=device)
    return {"data": data,
            "offs": torch.tensor(offs[:-1], dtype=torch.long, device=device),
            "klo": klo, "kstep": kstep, "vlo": vlo, "vstep": vstep,
            "level": levels.to(torch.int32), "valid": valid.to(torch.int32),
            "nbytes": int(data.numel() + klo.numel() * 2 * 2
                          + vlo.numel() * 2 * 2)}
