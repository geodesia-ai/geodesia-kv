"""Geodesia-KV — compressione della KV cache ad allocazione certificata.

Il nome viene dalla geodesia, e il legame e' tecnico, non decorativo:
all'ottimo lagrangiano ogni blocco demolito ha lo STESSO costo marginale
(danno per bit). La soluzione e' quindi una superficie equipotenziale, che in
geodesia e' esattamente la nozione di riferimento rispetto a cui si misura.

Tre proprieta' che nessun metodo esistente ha insieme:

- **non evitta mai** (a differenza di StreamingLLM / H2O / SnapKV) e **non
  quantizza uniformemente** (a differenza di KIVI / GEAR): alloca precisione
  graduata per blocco;
- **demozione monotona** lungo la scala {16, 8, 4, 2, centroide}: un blocco non
  risale mai, non per scelta ma perche' scartato il bf16 originale risalire e'
  impossibile. Il vincolo definisce il metodo;
- **certificato d'errore rigoroso a runtime** sull'output dell'attenzione.

Il kernel CUDA dequantizza dentro l'attenzione, quindi la cache non viene mai
materializzata in forma densa: il risparmio di memoria e' reale, non contabile.
"""

__version__ = "0.1.0"
__all__ = ["policies", "packed", "pack_layout", "cuda_ext"]
