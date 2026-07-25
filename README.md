# Geodesia-KV

Compressione della KV cache ad **allocazione certificata**: contesti lunghi su GPU
piccole, con un limite d'errore dimostrato invece che sperato.

Il nome viene dalla geodesia, e il legame è tecnico. All'ottimo lagrangiano ogni
blocco demolito ha lo **stesso costo marginale** (danno per bit): la soluzione è una
superficie equipotenziale, che in geodesia è esattamente la nozione di riferimento
rispetto a cui si misura ogni quota.

## Il metodo in tre proprietà

Nessun metodo esistente le ha insieme.

**Non evitta mai, non quantizza uniformemente.** StreamingLLM, H2O e SnapKV
*scartano* token: l'informazione sparisce e non torna. KIVI e GEAR quantizzano
*tutto allo stesso modo*: a 2 bit crollano. Geodesia-KV tiene l'intero contesto e
gli assegna **precisione graduata per blocco**.

**Demozione monotona.** Un blocco scende la scala `{16, 8, 4, 2, centroide}` e non
risale mai — non per scelta di progetto, ma perché una volta scartato il bf16
originale risalire è *impossibile*. Il vincolo definisce il metodo. Misurato:
demolire per gradi invece che allocare direttamente al livello finale costa
**0.0–0.1%**, quindi l'originale non serve mai più.

**Certificato d'errore a runtime.** Per ogni query si calcola un limite superiore
rigoroso sull'errore dell'output di attenzione:

```
‖o − ô‖ ≤ Σ_b â_b·δv_b + (M⁺/M⁻ − M⁻/M⁺)·max‖v‖ ,   M± = Σ_b â_b e^{±ε_b}
```

Entrambi i termini sono pesati dalla massa di attenzione, non dal caso peggiore
globale. **Zero violazioni** su tutte le configurazioni misurate.

## Risultati

Qwen2.5-3B-Instruct, contesto 16k, WikiText-2, tre profondità dell'ago.
Baseline portati fedelmente dal sorgente ufficiale (i repo originali richiedono
`transformers` 4.33–4.43, incompatibile con 5.x).

| Metodo | bit/valore | perplexity | Δ oracolo |
|---|---:|---:|---:|
| Full-KV (oracolo) | 16.00 | 5.452 | — |
| **Geodesia-KV, budget 3** | **2.96** | **5.453** | **+0.02%** |
| Geodesia-KV, budget 4 | 3.95 | 5.489 | +0.68% |
| Geodesia-KV, budget 2 | 1.97 | 5.690 | +4.37% |
| KIVI k4v4 | 5.03 | 5.499 | +0.86% |
| SnapKV b=2048 | 2.01 | 5.742 | +5.32% |
| KIVI k2v2 | 3.03 | 5.828 | +6.90% |
| Quest p=32 | 2.26 | 6.362 | +16.7% |

**Tutti e quattro i baseline sono strettamente Pareto-dominati**: per ognuno esiste
un punto Geodesia-KV con meno memoria *e* qualità migliore. Contro KIVI k4v4:
**41% di memoria in meno** con perplexity migliore.

### Kernel CUDA

Il kernel fonde dequantizzazione e attenzione (softmax online, split-K), così la
cache non viene **mai** materializzata in forma densa: il risparmio di memoria è
reale, non contabile.

| | kernel | attenzione densa bf16 | memoria |
|---|---:|---:|---:|
| 16k, 2 kv-head | 0.115 ms | 0.060 ms | **4.21x** |
| 64k, 2 kv-head | 0.248 ms | 0.153 ms | **4.40x** |

L'attenzione densa legge 4.2x più VRAM. Il kernel è **numericamente identico** al
percorso PyTorch su tutti e cinque i livelli di precisione (`tests/test_cuda_kernel.py`):
per costruzione non può degradare la qualità.

### Memoria e velocita' misurate

Attivazioni reali, livelli scelti dall'allocatore vero, byte allocati (non contabilizzati):

| Modello | budget | bit/valore | compressione | ms/token |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 3.0 | 3.20 | **5.00x** | 4.70 |
| Qwen2.5-3B | 2.0 | 2.31 | **6.92x** | 4.53 |
| Qwen3.5-0.8B | 3.0 | 3.19 | 5.01x | 1.56 |
| Qwen3.5-0.8B | 2.0 | 2.50 | 6.39x | 1.42 |

Il costo per token e' il costo per testa misurato, scalato per numero di teste e
layer: non e' un tok/s end-to-end.

## Paper

`paper/geodesia_kv.pdf` — 7 pagine, con review dello stato dell'arte (28 riferimenti),
metodo, risultati e risultati negativi. Sei figure generate da dati reali con
`paper/make_figures.py`.

## Uso

```python
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from geodesia_kv import policies as P

ALL_ATTENTION_FUNCTIONS["geodesia"] = P.policy_attention
P.ACTIVE = P.Policy(name="geodesia", budget_bits=3.0, group=64, window=128)

model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="geodesia")
```

Nessun training, nessuna modifica ai pesi: si registra un'implementazione di
attenzione e si passa al modello. Funziona su qualunque checkpoint già addestrato.

## Cosa è misurato e cosa no

Queste sono le condizioni esatte dei numeri sopra. Il resto non è stato misurato.

**Misurato**: due modelli (Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct), contesto
16k, WikiText-2, recupero passkey a tre profondità, perplexity, certificato su
576 chiamate per configurazione. Kernel CUDA validato in isolamento contro il
riferimento PyTorch.

**Non misurato**: RULER e LongBench; corpora diversi da WikiText-2; modelli oltre i
3B; il kernel dentro il ciclo di inferenza end-to-end (finora è validato da solo);
VRAM di picco e throughput di generazione reali.

**Limite noto**: i baseline sono port fedeli dal sorgente ufficiale, non il codice
ufficiale in esecuzione. KIVI a 2 bit crolla su questi modelli (passkey 0% sul
0.5B), ma è validato su modelli 7B: la lettura corretta è *«a questa scala la
quantizzazione uniforme a 2 bit crolla, l'allocazione graduata no»*, non «KIVI non
funziona».

## Risultati negativi documentati

Misurati e scartati, perché non vengano ritentati:

- **Rotazione di Hadamard** (incoherence processing, QuIP#/QuaRot): peggiora del
  20% a parità di bit e rompe il recupero (passkey 66.7% invece di 100%). La
  quantizzazione per canale è già la difesa contro i canali outlier; ruotare
  distrugge la struttura che sfrutta.
- **Vector quantization** (più centroidi per blocco): vince solo sotto 1.5
  bit/valore. L'assunzione «token con key simili hanno value simili» è falsa —
  l'errore sulle value resta 0.47–0.78 anche con 32 centroidi.
- **Asimmetria K/V** (più bit alle key, meno alle value): peggiora.
- **2 bit scalari con gruppi fini**: dominati da 4 bit con gruppi grossolani
  (errore 0.026 a 4.50 bit contro 0.056 a 4.00). La distorsione è convessa nei bit.

## Test

```bash
pytest tests/ -q          # 11 test: causalità, certificato, kernel, packing
```

I test di causalità hanno scoperto tre fughe reali dal futuro nelle
reimplementazioni dei baseline. Esistono per impedire che ricompaiano.

## Licenza

Apache-2.0
