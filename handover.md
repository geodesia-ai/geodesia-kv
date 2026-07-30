# Handover Geodesia-KV

Aggiornato il 2026-07-29. Workspace:
`/home/islab/Documents/research_dentamaro/compression/geodesia-kv`.

## Obiettivo e stato reale

Obiettivo: battere le tecniche concorrenti su tutti gli assi, almeno dai modelli
3B in su, includendo 0.8B, 3B, 8B e un 30B quantizzato che entri nella RTX A6000.

Il goal **non è ancora chiuso**.

- Sul 3B il nuovo punto a circa 5 bit, graded B=4.875 più 0.78125% di residui
  esatti selezionati con `massa * residuo_V^2`, domina KIVI-4 nel protocollo
  realmente incrementale: PPL 6.320 contro 6.322, 4.96 contro 5.03 bit/valore.
  La KL è 0.0030 contro 0.0041. RD-V2 resta il punto migliore della frontiera
  nella fascia circa 2 bit.
- Sull'8B il punto congelato B=4.5 più 3.125% di residui esatti domina KIVI-4
  nel protocollo incrementale: PPL 10.540 contro 10.581, 4.99 contro 5.03
  bit/valore; KL 0.0024 contro 0.0039. Quest incrementale resta migliore in
  PPL (10.238). La nuova variante compressed-Quest riduce la distanza a
  10.272, usando 9.85 bit residenti e 1.96 letti contro 16.25/2.32, e migliora
  leggermente la KL (0.0460 contro 0.0464), ma non domina ancora la PPL.
- Il trasferimento congelato fuori distribuzione su PG-19 è stato esteso il
  2026-07-29 dalle 6 finestre iniziali a tutte le **11 finestre preregistrate
  valide**, con intervalli paired token per token. Entrambe le conclusioni di
  ieri cambiano.
  - La sconfitta del punto 5-bit contro KIVI-4 **non regge**: era rumore del
    sottoinsieme di sei libri. Su 11 libri KIVI-4 fa 15.475 e graded+RD 15.490,
    5 vittorie su 11, rapporto PPL 0.99904 con CI95% `[0.99046, 1.00795]` che
    contiene lo zero. A 4.99 contro 5.03 bit/valore non si misura alcuna
    differenza di PPL: meno capacità senza costo rilevabile, non dominanza.
  - La vittoria di compressed-Quest su Quest invece **regge e si rafforza**:
    15.961 contro 16.043, 8 vittorie su 11, rapporto PPL 1.00516 con CI95%
    `[1.00076, 1.00945]` che esclude lo zero, usando 9.84 contro 16.25 bit
    residenti e 1.96 contro 2.32 letti. È l'unico confronto del progetto in cui
    una dominanza su tre assi sopravvive a un test fuori distribuzione con
    intervallo. L'effetto sulla PPL resta però piccolo (limite inferiore
    +0.08%): è la consistenza fra finestre, non l'ampiezza, a renderlo
    distinguibile.
- RD-V2 8B domina in aggregato chunked tutte le cache a capacità compressa
  nella fascia bassa, ma non la PPL di Quest. Quest mantiene quasi tutta la
  cache BF16 residente e legge solo pagine selezionate.
- Eliminare o non leggere contesto può agire da regolarizzazione: la sola PPL
  non misura fedeltà della memoria. RD-V2 ottiene la PPL migliore sacrificando
  KL rispetto al Geodesia graded originale.
- Il Qwen3-30B-A3B GPTQ-Int8 entra e gira realmente sulla A6000. Il candidato
  RD-V2 trasferito senza retuning domina tutte le baseline locali su sei
  finestre per PPL, capacità residente e byte letti: 5.9413 PPL a 2.061
  bit/valore.
- Non esiste ancora un'integrazione end-to-end del formato packed/kernel dentro
  la cache di Transformers. Qualità e contabilità sono simulate su K/V dense;
  il kernel CUDA è misurato soltanto in isolamento. Nel microbenchmark
  multi-head a 16k il formato packed riduce la cache di circa 6.9 volte e il
  kernel attention-only è 3.75 volte più rapido del riferimento denso sul 3B,
  1.25 volte sull'8B. Non sono tok/s di generazione.

`paper/geodesia_kv.tex` è stato completamente riscritto sui risultati correnti
e il PDF ricompilato. `README.md` resta invece stale e non va pubblicato così
com'è.

## Stato del repository

- Branch: `main`.
- Commit di partenza: `6c5ac302665d3334ef493e27530d0149d5aa972e`.
- Il worktree è intenzionalmente dirty: contiene correzioni e risultati non
  ancora committati.
- Test correnti: `27 passed` con:

```bash
.venv/bin/python -m pytest -q
```

Non usare il `pytest` globale: in questa macchina può risolvere il symlink
`Documents/Documenti` in modo incoerente. L'ambiente `.venv` usa NumPy 1.26 per
evitare le incompatibilità di NumPy 2 con SciPy/PyArrow presenti globalmente.

File modificati o aggiunti:

- `.gitignore`
- `benchmarks/fetch_wikitext.py`
- `benchmarks/fetch_pg19.py`
- `benchmarks/budget_sweep.py`
- `benchmarks/paired_stats.py`
- `benchmarks/memory_quality.py`
- `benchmarks/system_metrics.py`
- `benchmarks/vs_sota.py`
- `geodesia_kv/policies.py`
- `geodesia_kv/packed.py`
- `tests/test_geodesia_kv.py`
- `paper/geodesia_kv.tex`
- `paper/geodesia_kv.pdf`
- `handover.md`
- `paper/pg19_validation.txt.manifest.json`
- `paper/pg19_test.txt.manifest.json`
- nuovi JSON sotto `results/`

Non scartare o sovrascrivere modifiche non correlate: prima del lavoro il repo
era pulito, ma ora tutti i file sopra fanno parte dell'indagine.

## Stato del paper

`paper/geodesia_kv.tex` non è più il manoscritto iniziale con PPL 5.45,
dominanza universale, RTX 4090 Laptop e costi per-head scalati. È stato
riscritto integralmente il 2026-07-28 e aggiornato il 2026-07-29 con gli
intervalli paired, il test PG-19 a 11 libri e la capacità di contesto in VRAM;
compilato in `paper/geodesia_kv.pdf` (9 pagine, nessun box fuori margine,
nessun riferimento indefinito).

Modifiche del 2026-07-29, in gran parte **correttive** e non additive:

- nuova sottosezione di protocollo `sec:paired` con l'equazione del rapporto di
  PPL come media delle differenze appaiate e il cluster bootstrap;
- gli offset PG-19 sono descritti come lista preregistrata esaurita (11 su 12
  libri, il quinto troppo corto), non più come sei finestre scelte;
- `tab:pgfive` riscritta a due colonne, sei contro undici libri, con rapporto e
  CI: la frase "il punto a 5 bit non trasferisce" è stata sostituita da
  "statisticamente indeciso, stessa qualità a capacità inferiore";
- `tab:sparse` aggiornata alle 11 finestre PG-19 con 8/11 vittorie e CI che
  esclude l'uno;
- avvertenza esplicita che i margini di `tab:primary` (0.04% e 0.39%) non hanno
  intervalli e sono dello stesso ordine dell'effetto dissolto su PG-19;
- nuova sezione `sec:capacity` con `tab:capacity` (contesto massimo per VRAM,
  8B e 3B) e `tab:concurrency`, inclusa la conclusione negativa che la GQA
  rende la VRAM non vincolante sul contesto massimo;
- limitazioni e conclusione riallineate; contributi 4 e 5 dell'introduzione
  riscritti.

La nuova versione:

- separa capacità residente e bit letti/query;
- usa come prove primarie soltanto i confronti Q=1 incrementali 3B/8B e il
  trasferimento PG-19;
- riporta sia il successo PG-19 di compressed-Quest sia il fallimento
  graded+RD contro KIVI-4, senza claim selettivi;
- mantiene 0.8B, RD-V2 e 30B Q8 in una sezione esplicitamente
  **esplorativa chunked**, non come prova finale;
- descrive causalità, GQA condivisa, observer del prefill, exact residual RD,
  summary box esatti e contabilità sparse;
- presenta KL e microbenchmark del kernel come diagnostiche separate;
- dichiara che il runner di qualità conserva K/V dense e che non esistono
  ancora tok/s o picco VRAM end-to-end;
- documenta i risultati negativi quantum-inspired e convenzionali;
- conclude esplicitamente che il goal “tutti gli assi da 3B in su” non è
  ancora raggiunto;
- cita i report Qwen2.5/Qwen3 e GPTQ oltre ai lavori KV-cache correlati.

Le vecchie figure sotto `paper/figures_geodesia/` non sono usate nel nuovo
manoscritto perché incorporano numeri invalidati dal protocollo precedente.
La compilazione verificata è:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error geodesia_kv.tex
pdflatex -interaction=nonstopmode -halt-on-error geodesia_kv.tex
```

Il secondo passaggio non produce riferimenti indefiniti né box oltre margine.
Prima di una submission servono ancora baseline ufficiali, intervalli paired,
RULER/LongBench e integrazione end-to-end; non reintrodurre nel paper le vecchie
claim per colmare questi vuoti.

## Problemi scoperti nel protocollo originale

### Correttezza e causalità

1. **Leak dalle query future di Geodesia.** Con una chiamata contenente più
   query teacher-forced, l'allocatore usava anche `q[1:]` per decidere la cache
   vista da `q[0]`.
2. **Cache GQA fisicamente impossibile.** La stessa KV-head veniva compressa
   separatamente e demolita più volte per le query-head che la condividono.
3. **Teacher forcing contaminante.** Il test della passkey inseriva la risposta
   vera nello stato della policy prima della generazione greedy.
4. **Prefill senza storia.** L'allocatore Geodesia partiva quasi cieco dopo il
   prompt; SnapKV non usava realmente le ultime query del prompt.
5. **Fallback non causale.** In alcuni percorsi l'attenzione delegata poteva
   vedere token futuri quando `attention_mask=None`.
6. **Pagine/blocchi parziali.** Quest e KIVI richiedono trattamento speciale
   della pagina/gruppo in scrittura per non calcolare statistiche su chiavi
   future.

Sono stati aggiunti test di regressione per il primo punto, la condivisione GQA,
il percorso CUDA multi-head, l'allineamento NLL del runner incrementale e il
reader sparse con summary esatti, l'allineamento delle NLL per token e il
bootstrap paired; l'intera suite ora ha 27 test.

### Confronto memoria/velocità

1. Il benchmark di qualità conserva ancora la K/V densa BF16 e simula la
   compressione. I bit riportati descrivono il formato ideale/packed, non la
   VRAM end-to-end del processo di qualità.
2. Il vecchio “tok/s” di sistema era il tempo per testa scalato per numero di
   teste/layer, non throughput di generazione reale.
3. `system_metrics.py` usava query casuali nell'allocatore e misure dense
   instabili.
4. La contabilità originale di Quest mescolava due quantità diverse: pagine
   lette per query e capacità totale della cache residente. È stata corretta:
   nel setup 16k corrente Quest ha circa 16.25 bit/valore residenti e ne legge
   circa 2.31 per query. I vecchi JSON riportano ancora 2.25 nella colonna
   `bits_per_value` e non sono validi per confronti di capacità.
5. A 1k token i blocchi sink/window esatti dominano il costo: i punti a budget
   3 bit non sono un test rappresentativo perché i soli blocchi hot possono
   superare il budget.
6. Il kernel fused è numericamente testato e misurato in isolamento, ma non è
   collegato al `past_key_values` del modello. Non si può ancora sostenere un
   vantaggio end-to-end di velocità o picco VRAM.
7. La PPL storica era valutata in chunk teacher-forced. I logits erano causali,
   ma KIVI trattava l'intero chunk come residuo corrente esatto e la contabilità
   includeva token non ancora residenti. L'artefatto è grande: sull'8B KIVI-4
   passa da 5.03 bit/valore con Q=64 a 5.69 con Q=1024. Ora
   `budget_sweep.py --incremental-eval` appende e valuta un token alla volta.
   I JSON senza `eval_mode: incremental` restano utili per screening, non per
   conclusioni finali. Full-KV reale cambia di circa 0.45% PPL fra Q=64 e Q=1
   per il diverso ordine numerico dei kernel BF16; confrontare sempre metodi
   eseguiti nello stesso modo.

### Dataset e riproducibilità

- `paper/wikitext2_valid.txt` mancava. Ora viene scaricato da
  `benchmarks/fetch_wikitext.py`.
- File ottenuto: 1,142,150 caratteri,
  SHA256 `8ef749789ca0693435d20b3f81d5638c19edcebc5a68586dcf09bdf47ef9542f`.
- Il fetcher ora accetta `--split`. WikiText-2 test è stato scaricato in
  `paper/wikitext2_test.txt`: 1,285,622 caratteri, SHA256
  `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`.
- `budget_sweep.py` ora registra modello, revisione, commit, hash del testo,
  Torch, Transformers, GPU, dtype, finestre e configurazione di quantizzazione.
- Il prefill ora chiama direttamente il decoder: evita di materializzare logits
  `[batch, 16k, vocab]` inutili (circa 4.6 GiB sul Qwen3-30B).

## Correzioni implementate

In `geodesia_kv/policies.py`:

- la demozione usa soltanto la query corrente;
- la massa viene aggregata tra tutte le query-head GQA della stessa KV-head;
- la KV condivisa viene demolita una sola volta;
- `observation_attention` raccoglie statistiche causali durante il prefill;
- SnapKV usa le query finali del prompt, non query della continuazione;
- Geodesia inizializza la massa dai dati osservati nel prompt;
- `reset_runtime()` separa teacher forcing e generazione;
- l'interfaccia SDPA è compatibile con Transformers recente;
- l'allocatore espone `alloc_value_weight`, `alloc_mass_power` e `mass_decay`.
- `Report` separa `resident_bits` e `read_bits`;
- Quest contabilizza la KV esatta residente più i sommari di pagina, separando
  i byte selezionati/letti;
- è disponibile un gate query-adaptive certificato sui blocchi Geodesia, con
  bias finito incluso nell'errore certificato;
- sono stati prototipati blocchi e token prompt-only conservati esatti; non
  hanno generalizzato e non vanno presentati come contributi positivi.

Nei benchmark:

- selezione delle policy con `--only`;
- più finestre con `--text-offsets`;
- aggregazione PPL corretta come media delle NLL/media geometrica delle PPL;
- `--eval-tokens`, `--skip-passkey`, sweep di budget/decay/observation window;
- coppie trasferite con `--geodesia-settings`, ad esempio
  `2:0.5 3:0.925 5:0.25`;
- sintassi estesa
  `budget:decay:keep:bias:gate:protected_frac:token_protected_frac`;
- output separato di bit residenti, bit letti e frazione trattenuta dal gate;
- scelta `--dtype float16|bfloat16`;
- workaround per checkpoint GPTQ che hanno `quant_method` soltanto in
  `quantize_config.json`;
- configurazione GPTQ serializzata correttamente nel JSON.

## Risultati affidabili finora

Protocollo principale: contesto 16,384, 64 token valutati, WikiText-2 raw
validation, finestre token `0, 16384, 32768, 49152`. La PPL aggregata è la media
geometrica delle quattro finestre.

### Qwen2.5-3B-Instruct

Da `results/qwen3b_16k_four_window_eval64.json` e
`results/qwen3b_16k_four_window_2bit_sweep.json`:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 6.476 |
| SnapKV | 2.063 | 2.063 | 6.841 |
| Quest | **circa 16.250** | circa 2.313 | 7.125 |
| KIVI-2 | 3.051 | 3.051 | 6.720 |
| KIVI-4 | 5.043 | 5.043 | 6.515 |
| Geodesia B=2, decay=.5 | 1.992 | 1.992 | 6.584 |
| Geodesia B=3, decay=.925 | 2.993 | 2.993 | 6.520 |
| Geodesia B=5, decay=.25 | 4.993 | 4.993 | 6.498 |

Conclusione: ogni baseline a capacità compressa ha un punto Geodesia con meno
memoria e PPL migliore. Quest legge meno di Geodesia per query ma mantiene la
cache completa; è quindi dominata su capacità/PPL, non sull'asse bandwidth. Il
risultato è vicino a Full-KV, ma non lo supera stabilmente.

Fedeltà della memoria a 16k:

| Metodo | bit residenti/valore | bit letti/valore | MQ | KL memoria |
|---|---:|---:|---:|---:|
| SnapKV | 2.020 | 2.020 | 0.9953 | 0.2084 |
| Quest | circa 16.250 | circa 2.32 | 0.9959 | 0.1266 |
| KIVI-2 | 3.032 | 3.032 | 1.0001 | 0.0786 |
| KIVI-4 | 5.033 | 5.033 | 1.0002 | 0.0040 |
| Geodesia B=2 | 1.996 | 1.996 | 0.9989 | 0.0097 |
| Geodesia B=3 | 2.995 | 2.995 | 0.9996 | 0.0037 |
| Geodesia B=5 | 4.994 | 4.994 | 0.9995 | 0.0012 |

Geodesia B=2 domina nettamente SnapKV/Quest sulla KL pur usando meno memoria.

### Qwen3-8B

Da `results/qwen8b_16k_four_window_eval64.json` e
`results/qwen8b_16k_four_window_2bit_transfer.json`:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 7.162 |
| SnapKV | 2.063 | 2.063 | 6.970 |
| Quest | **circa 16.250** | circa 2.313 | 7.187 |
| KIVI-2 | 3.051 | 3.051 | 7.182 |
| KIVI-4 | 5.043 | 5.043 | 7.215 |
| Geodesia B=2, decay=.95 | 1.993 | 1.993 | 7.219 |
| Geodesia B=2.9, decay=.95 | 2.892 | 2.892 | 7.165 |
| Geodesia B=5, decay=.25 | 4.992 | 4.992 | 7.136 |

Geodesia domina KIVI-2 e KIVI-4, ma non domina SnapKV sulla PPL nella fascia bassa
e non domina Quest sul bandwidth letto.
Il punto SnapKV ha PPL migliore persino dell'oracolo; questo è compatibile con
regolarizzazione per rimozione del contesto, non con una cache più fedele.

Fedeltà della memoria, ora rigenerata su otto profondità in
`results/qwen8b_16k_memory_quality_corrected_axes_gate.json`:

| Metodo | bit residenti/valore | bit letti/valore | MQ | KL memoria |
|---|---:|---:|---:|---:|
| SnapKV | 2.020 | 2.020 | 0.9990 | 0.0480 |
| Quest | 16.250 | 2.321 | 0.9990 | 0.0732 |
| KIVI-2 | 3.032 | 3.032 | 0.9985 | 0.0383 |
| KIVI-4 | 5.033 | 5.033 | 1.0001 | 0.00190 |
| Geodesia B=2, gate centroid | **1.995** | **1.995** | 0.9987 | **0.0137** |
| Geodesia B=5 | 4.994 | 4.994 | 1.0001 | 0.00068 |

Geodesia B=2 conserva l'intero contesto effettivo e ha KL molto inferiore a
SnapKV, Quest e KIVI-2 usando anche meno capacità. Il gap rimasto è quindi nella
loss linguistica/PPL, non nella fedeltà dell'attenzione o nel recupero della
needle custom.

Su una finestra holdout a offset 16k, Geodesia B=2.9 ottiene PPL 5.983 contro
Full 6.000, KIVI-2 6.050 e KIVI-4 6.073. Non basta da solo: servono più finestre.

### Qwen3-8B: riferimento aggregato a dieci finestre

Il risultato più solido disponibile è ora
`results/qwen8b_16k_ten_window_corrected_axes_eval64.json`: dieci finestre
disgiunte agli offset 0--147,456, 16,384 token di contesto e 64 target per
finestra, quindi 640 token valutati per metodo.

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 7.813 |
| StreamingLLM | 2.004 | 2.004 | 8.842 |
| SnapKV | 2.063 | 2.063 | 7.745 |
| Quest | 16.250 | 2.313 | **7.727** |
| KIVI-2 | 3.051 | 3.051 | 7.747 |
| KIVI-4 | 5.043 | 5.043 | 7.871 |
| Geodesia B=2, ungated | 1.993 | 1.993 | 7.985 |
| Geodesia B=2, centroid keep=.125 bias=6 | **1.993** | **1.993** | 7.945 |
| Geodesia B=2.9, centroid keep=.2 bias=8 | 2.892 | 2.892 | 7.907 |
| Geodesia B=5 | 4.992 | 4.992 | 7.846 |

Il JSON è stato prodotto dal processo avviato prima dell'ultima correzione al
contatore Quest; i suoi undici campi di lettura Quest sono stati corretti
analiticamente da 2.25 a 2.3125. La PPL e la capacità residente non dipendono
da questa correzione.

Conclusione operativa:

- il gate a centroidi migliora Geodesia B=2 rispetto alla versione ungated e
  conserva zero violazioni su 11,520 chiamate certificate;
- Geodesia B=2 domina StreamingLLM, ma paga circa 0.20 PPL rispetto a SnapKV
  in cambio di circa 3.4% di capacità/lettura in meno;
- i punti Geodesia a 2.9 e 5 bit sono dominati da SnapKV su questo aggregato;
- Quest ha la PPL migliore ma usa capacità quasi Full-KV; non è confrontabile
  come cache compressa, anche se il suo asse di lettura resta competitivo;
- il fronte multi-asse corrente contiene almeno Geodesia B=2, SnapKV e Quest.
  Il goal di battere tutte le tecniche su tutti gli assi non è raggiunto.

La PPL per finestra varia fortemente e metodi compressi battono spesso
Full-KV. Questo conferma che bisogna riportare NLL aggregata con intervalli di
confidenza e affiancarla a retrieval/KL, non selezionare configurazioni su una
singola finestra.

### Qwen3-8B: test finale Born mai osservato

Dopo gli offset 0--147,456 sono state usate tre nuove finestre 163,840--196,608
soltanto per selezionare tra sei gate predefiniti. Il vincitore validation,
Born `eta=.95`, bias 2, è stato congelato e valutato una sola volta sugli ultimi
tre segmenti disponibili, 212,992, 229,376 e 245,760:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 10.091 |
| StreamingLLM | 2.004 | 2.004 | 10.237 |
| SnapKV | 2.063 | 2.063 | **9.448** |
| Quest | 16.250 | 2.313 | 10.041 |
| KIVI-2 | 3.051 | 3.051 | 9.814 |
| KIVI-4 | 5.043 | 5.043 | 10.079 |
| Geodesia B=2 Born | **1.993** | **1.993** | 10.034 |

File:
`results/qwen8b_16k_born_frozen_final_test_offsets212992_245760.json`.
Geodesia domina Full, StreamingLLM, Quest e KIVI-4 su capacità/PPL, ma non
SnapKV o KIVI-2. Il test finale smentisce quindi la dominanza completa.

Comando:

```bash
.venv/bin/python -m benchmarks.budget_sweep \
  --model Qwen/Qwen3-8B --tokens 16384 --eval-tokens 64 --skip-passkey \
  --text-offsets 0 16384 32768 49152 65536 81920 98304 114688 131072 147456 \
  --geodesia-settings 2:0.95 2:0.5:0.125:6:centroid \
                       2.9:0.95:0.2:8:centroid 5:0.95 \
  --only 'Full-KV' 'StreamingLLM' 'SnapKV' 'Quest p=' \
         'KIVI k2v2 g32 r32' 'KIVI k4v4' 'qk=' \
  --out results/qwen8b_16k_ten_window_corrected_axes_eval64.json
```

### Qwen3.5-0.8B

Il pilot corretto è `results/qwen0.8b_16k_quality_corrected_pilot.json`.
Geodesia B=3 ottiene 7.335 a 2.993 bit contro KIVI-2 8.905 a 3.033 bit;
Geodesia B=2 ottiene 7.443 a 1.996 bit. Quest ottiene 7.329 ma il vecchio
valore 2.259 è traffico letto sottostimato, non capacità residente.
Il goal dichiarato parte dal 3B, quindi non investire tuning prioritario qui.

La passkey greedy è 0% anche per Full-KV su questo modello: non è una metrica
discriminante. Usare NLL della chiave o, meglio, RULER con modelli capaci.

## Qwen3-30B-A3B Q8 sulla RTX A6000

Checkpoint usato:
`JunHowie/Qwen3-30B-A3B-GPTQ-Int8`.

È una quantizzazione GPTQ 8 bit, group size 128, del modello base esatto
`Qwen/Qwen3-30B-A3B`. Ha 30.5B parametri totali, circa 3.3B attivi per token,
48 layer, 128 expert/layer, 8 expert attivi, 32 query-head, 4 KV-head e head
dimension 128.

Perché non usare gli altri formati:

- BF16 ufficiale: circa 56.9 GiB di pesi, non entra.
- FP8 ufficiale: circa 30.2 GiB su disco, ma Transformers dequantizza a BF16
  sulle GPU con compute capability inferiore a 8.9. La A6000 è `sm_86`, quindi
  il modello non entrerebbe.
- BitsAndBytes non sostituisce correttamente i pesi expert 3D della nuova
  implementazione fused di Qwen3-MoE.
- Il GPTQ usa invece expert lineari separati e kernel Marlin INT8 reali.

Ambiente isolato:

```text
/tmp/geodesia-gptq-venv
Transformers 4.55.4
Optimum 1.27.0
GPTQModel 2.2.0
Torch 2.10.0+cu128
NumPy 1.26.4
protobuf 5.29.6
```

GPTQModel è stato compilato localmente per `sm_86`; il kernel caricato è
`gptqmodel.nn_modules.qlinear.marlin.MarlinQuantLinear`.

Cache modello:

```text
/tmp/geodesia-hf-30b/hub/models--JunHowie--Qwen3-30B-A3B-GPTQ-Int8/
  snapshots/059b0db28cc7f6c52ff91d457fdf8ee87ee7878a
```

La cache è in `/tmp`, non è garantita persistente. Pesa circa 29.81 GiB. Lo
spazio residuo della partizione `/tmp` era circa 42 GiB al momento di questo
handover; `/home` ha soltanto circa 37 GiB liberi.

Prova di caricamento completa:

- 18,672 moduli `MarlinQuantLinear`;
- 29.692 GiB allocati da Torch dopo il load;
- 29.799 GiB riservati;
- 17.198 GiB liberi;
- load 204.7 s;
- forward corto 0.924 s;
- logits tutti finiti;
- durante prefill 16k osservati circa 33.56 GiB usati e 14.87 GiB liberi.

Il repository del checkpoint omette `quant_method` dentro `config.json`, pur
avendolo correttamente in `quantize_config.json`. Il runner ora carica
quest'ultimo automaticamente.

Primo campione 16k/64 token (stdout del run; il vecchio file pilot è incompleto
perché il writer non serializzava `GPTQConfig`, bug ora corretto):

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 6.544 |
| SnapKV | 2.063 | 2.063 | 6.746 |
| Quest | circa 16.250 | circa 2.313 | 6.837 |
| KIVI-2 | 3.051 | 3.051 | 6.602 |
| KIVI-4 | 5.043 | 5.043 | 6.184 |
| Geodesia B=2, decay=.5 | 1.992 | 1.992 | 6.372 |
| Geodesia B=3, decay=.925 | 2.993 | 2.993 | 6.713 |
| Geodesia B=5, decay=.25 | 4.993 | 4.993 | 6.674 |

Il risultato definitivo a quattro finestre è
`results/qwen30b_a3b_q8_16k_four_window_eval64.json`:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 7.2371 |
| SnapKV | 2.063 | 2.063 | 7.2103 |
| Quest | **circa 16.250** | circa 2.313 | 7.2869 |
| KIVI-2 | 3.051 | 3.051 | 7.2772 |
| KIVI-4 | 5.043 | 5.043 | **7.1593** |
| Geodesia B=2, decay=.5 | **1.993** | 1.993 | 7.1640 |
| Geodesia B=3, decay=.925 | 2.993 | 2.993 | 7.2442 |
| Geodesia B=5, decay=.25 | 4.993 | 4.993 | 7.2262 |

Geodesia B=2 domina SnapKV, Quest e KIVI-2 su memoria e PPL, e migliora anche
Full-KV sulla PPL. Rispetto a KIVI-4 usa il 60.5% di bit in meno, ma perde 0.0048
PPL (circa 0.067%): non è ancora una dominanza Pareto stretta. Il certificato ha
zero violazioni su 6,144 chiamate. Anche sul 30B SnapKV e Geodesia possono
superare Full-KV in PPL, ulteriore conferma che questa metrica da sola premia
anche la regolarizzazione introdotta dalla compressione.

Comando riproducibile:

```bash
HF_HOME=/tmp/geodesia-hf-30b \
USE_TF=0 TRANSFORMERS_NO_TF=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
/tmp/geodesia-gptq-venv/bin/python -m benchmarks.budget_sweep \
  --model /tmp/geodesia-hf-30b/hub/models--JunHowie--Qwen3-30B-A3B-GPTQ-Int8/snapshots/059b0db28cc7f6c52ff91d457fdf8ee87ee7878a \
  --dtype float16 --tokens 16384 --eval-tokens 64 --skip-passkey \
  --text-offsets 0 16384 32768 49152 \
  --geodesia-settings 2:0.5 3:0.925 5:0.25 \
  --only 'Full-KV' 'SnapKV' 'Quest p=' \
         'KIVI k2v2 g32 r32' 'KIVI k4v4' 'transfer' \
  --out results/qwen30b_a3b_q8_16k_four_window_eval64.json
```

## Query-adaptive: implementazione e risultato negativo

È stato implementato il percorso suggerito nel precedente handover:

- gate per-query sui soli blocchi chiusi, con modalità `mass`, `centroid` e
  `hybrid`;
- sink, finestra corrente e blocchi hot restano sempre attivi;
- i blocchi esclusi ricevono un bias negativo finito, non vengono eliminati;
- il bias massimo entra nel certificato, che resta senza violazioni;
- prototipi aggiuntivi proteggono esattamente blocchi o token scelti soltanto
  dal prompt, senza promozioni future.

Il gate a centroidi ha vinto su alcune finestre validation, per esempio sul
Qwen3-8B a offset 65,536 Geodesia B=2, `keep=.125`, `bias=6` ha PPL 4.408
contro Full 4.417 e SnapKV 4.480. Non ha però generalizzato:

- holdout offset 98,304 e 114,688: KIVI-2 11.713; miglior Geodesia bassa
  11.859 a 1.99 bit/valore;
- holdout offset 131,072 e 147,456: Quest 9.392, KIVI-2 9.622, Full 10.008;
  le configurazioni Geodesia selezionate su validation fanno 10.13--10.48;
- il residuo esatto token-level arriva a 4.827 contro SnapKV 4.820 sulla
  validation difficile, ma peggiora fino a circa 12.5 sul primo holdout.

Conclusione: gate, protezione statica dei blocchi e residuo token-level sono
dipendenti dal segmento e non risolvono il gap in modo robusto. Non fare altro
tuning sulla stessa validation. Il gate può restare come infrastruttura
certificata, ma i percorsi `protected`/`token_protected` sono candidati alla
rimozione dopo aver conservato i risultati negativi.

File principali:

- `results/qwen8b_16k_query_gate_stage1_validation_offset65536.json`
- `results/qwen8b_16k_query_gate_stage2_validation_offset81920.json`
- `results/qwen8b_16k_query_gate_stage3_protected_validation_offset81920.json`
- `results/qwen8b_16k_query_gate_stage4_token_residual_validation_offset81920.json`
- `results/qwen8b_16k_query_gate_stage4b_token_refine_validation_offset81920.json`
- `results/qwen8b_16k_query_gate_stage4c_token_local_validation_offset81920.json`
- `results/qwen8b_16k_query_gate_holdout_offsets98304_114688.json`
- `results/qwen8b_16k_query_gate_holdout_upper_front_offsets98304_114688.json`
- `results/qwen8b_16k_query_gate_final_holdout_offsets131072_147456.json`

## Esperimenti quantum-inspired

Sono state provate tre traduzioni matematiche concrete, senza usare analogie
non misurabili.

1. **Stato misto spettrale K--V.** Un blocco conserva centroide, autovettori
   dominanti della covarianza delle key e risposta cross-covariance delle value.
   Ogni modo costa circa 0.25 bit/valore del blocco. La prima versione
   sostituiva tutti i centroidi e peggiorava PPL 5.240 -> 5.292--5.368; la scala
   intermedia `{16,8,4,2,spectral,centroid}` riduce il costo ma fa ancora 5.312.
   Il damping 0--0.75 non recupera il riferimento. La SVD completa costava
   125--141 s/configurazione; tre iterazioni di potenza batched riducono il
   prototipo a circa 20 s, ma la qualità resta negativa.
2. **Cumulante della funzione di partizione.** Un fp16 firmato per blocco
   corregge l'inflazione `exp(sigma^2/2)` della quantizzazione e la deflazione
   Jensen del centroide. Migliora una validation 5.240 -> 5.214, ma peggiora
   l'altra 4.408 -> 4.486: non generalizza.
3. **Proiezione Born.** Per query si seleziona il sottospazio minimo che contiene
   una quota `eta` della massa centroide, lasciando un fallback compresso con
   bias finito certificato. Migliora due validation e una nuova validation a
   tre finestre, ma fallisce il test finale contro SnapKV/KIVI-2.

Artefatti:

- `results/qwen8b_16k_spectral_rank_validation_offset81920.json`
- `results/qwen8b_16k_spectral_ladder_rank3_validation_offset81920.json`
- `results/qwen8b_16k_spectral_strength_validation_offset81920.json`
- `results/qwen8b_16k_cumulant_strength_validation_offset81920.json`
- `results/qwen8b_16k_cumulant_transfer_validation_offset65536.json`
- `results/qwen8b_16k_born_gate_validation_offset81920.json`
- `results/qwen8b_16k_born_gate_frozen_holdout_offsets98304_147456.json`
- `results/qwen8b_16k_weak_born_new_validation_offsets163840_196608.json`
- `results/qwen8b_16k_born_frozen_final_test_offsets212992_245760.json`

Conclusione: Geodesia domina sulla fedeltà (KL), mentre SnapKV beneficia di
regolarizzazione downstream. La prossima ipotesi sensata è una piccola base
esatta Snap-like più una coda a solo centroide/fallback debole, con costo totale
inferiore a 2.063 bit; non altra correzione del centroide verso Full-KV.

### Ibrido Snap-like esatto + coda centroide

Anche questa ipotesi è stata implementata usando il residuo token statico:

- base Geodesia quasi tutta terminale (`B=.1`);
- 10.2% dei token del prompt copiati esatti con la regola SnapKV;
- 12.5% dei token non penalizzati per query, bias finito 4;
- coda completa conservata a centroide;
- costo misurato 2.06055 bit/valore, poco sotto SnapKV 2.0625.

Su sei finestre di sviluppo del nuovo split test pareggia SnapKV:
PPL circa 6.3601 contro 6.3621. Su tre finestre holdout congelate fallisce:

| Metodo | bit residenti/valore | PPL holdout |
|---|---:|---:|
| Full-KV | 16.000 | 5.782 |
| SnapKV | 2.063 | 5.864 |
| KIVI-2 | 3.051 | 6.385 |
| KIVI-4 | 5.043 | **5.790** |
| Geodesia exact+tail | **2.061** | 5.929 |

File principali:

- `results/qwen8b_16k_snap_centroid_tail_test_validation_offsets0_32768.json`
- `results/qwen8b_16k_snap_centroid_tail_test_validation_offsets49152_81920.json`
- `results/qwen8b_16k_snap_centroid_tail_frozen_validation_offsets0_32768.json`
- `results/qwen8b_16k_snap_centroid_tail_frozen_holdout_offsets98304_131072.json`

Conclusione: una coda compressa migliora il rapporto capacità/informazione ma
non rende robusta la selezione Snap-like. La prossima selezione deve essere
loss-aware ma target-free, oppure calibrata offline su più corpora; non fare
altro tuning sullo stesso split.

### Saliency rate--distortion loss-aware

La selezione esatta è stata estesa con feature disponibili causalmente nel
prompt:

```text
saliency_j = attention_mass_j
             * ||v_j - mean_block(V)||^alpha
             * ||k_j - mean_block(K)||^beta
```

I coefficienti sono scelti offline sulla NLL, ma durante l'inferenza non sono
usati target futuri. Su tre finestre di calibrazione 8B:

- massa pura: PPL 9.598;
- `alpha=2, beta=0`: **9.547**;
- residuo K o combinazioni K--V peggiorano.

Il punto `alpha=2` è stato congelato. Su sei finestre 8B completamente escluse
dalla calibrazione, aggregando holdout e final test:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 10.535 |
| StreamingLLM | 2.004 | 2.004 | 10.980 |
| SnapKV | 2.063 | 2.063 | 10.560 |
| Quest | 16.250 | 2.313 | **10.230** |
| KIVI-2 | 3.051 | 3.051 | 10.749 |
| KIVI-4 | 5.043 | 5.043 | 10.539 |
| Geodesia RD-V2 | **2.061** | **2.061** | 10.517 |

Geodesia domina tutte le cache con capacità compressa sull'aggregato, incluso
KIVI-4, ma non la PPL di Quest full-resident.

Transfer invariato al Qwen2.5-3B-Instruct, sei finestre:

| Metodo | bit residenti/valore | PPL |
|---|---:|---:|
| Full-KV | 16.000 | 6.353 |
| SnapKV | 2.063 | 6.460 |
| Quest | 16.250 | 7.229 |
| KIVI-2 | 3.051 | 7.006 |
| KIVI-4 | 5.043 | **6.330** |
| Geodesia RD-V2 | **2.061** | 6.373 |

Il transfer 3B batte Snap/Quest/KIVI-2 ma manca KIVI-4 di 0.043 PPL.

Transfer invariato al Qwen3-30B-A3B GPTQ-Int8, sei finestre:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Full-KV | 16.000 | 16.000 | 6.0008 |
| StreamingLLM | 2.004 | 2.004 | 6.034 |
| SnapKV | 2.063 | 2.063 | 5.9900 |
| Quest | 16.250 | 2.313 | 6.1513 |
| KIVI-2 | 3.051 | 3.051 | 6.251 |
| KIVI-4 | 5.043 | 5.043 | 6.0600 |
| Geodesia RD-V2 | **2.061** | **2.061** | **5.9413** |

Sul 30B Q8 la dominanza è completa su PPL, capacità e lettura per tutte le
baseline locali. È il risultato più forte del progetto, ma non chiude ancora
il goal “da 3B in su” a causa dei gap 3B/8B e degli assi di sistema non
integrati.

Memory-quality del candidato RD-V2 sull'8B, quattro profondità:

| Metodo | bit residenti/valore | MQ | KL memoria |
|---|---:|---:|---:|
| SnapKV | 2.02 | 0.999 | 0.0970 |
| Quest | 16.25 | 0.999 | 0.0644 |
| KIVI-2 | 3.03 | 0.996 | 0.1148 |
| KIVI-4 | 5.03 | 1.000 | **0.0038** |
| Geodesia RD-V2 | 2.07 | 0.999 | 0.1064 |

La needle resta utilizzabile a tutte le profondità, ma la KL è peggiore di
Snap/Quest e molto peggiore del vecchio Geodesia graded B=2 (0.0137). RD-V2
ottiene regolarizzazione/PPL sacrificando fedeltà: non domina ancora l'asse KL.
Il prossimo punto deve redistribuire i 2.06 bit fra una base graded più ricca e
una quota esatta più piccola, ottimizzando congiuntamente NLL e KL.

È stato misurato il fronte a capacità quasi fissa:

| Base graded | quota esatta | bit/valore | PPL calib. | KL |
|---:|---:|---:|---:|---:|
| 0.1 | 10.2% | 2.061 | **9.547** | 0.1064 |
| 0.5 | 9.8% | 2.05 | 9.630 | 0.1060 |
| 1.0 | 6.6% | 2.05 | 9.708 | 0.1005 |
| 1.5 | 3.5% | 2.05 | 9.710 | 0.0999 |
| 2.0 | 0.4% | 2.06 | 9.657 | **0.0822** |

Il punto B=2 batte SnapKV sia su PPL aggregata 8B (10.547 vs 10.560) sia
su KL (0.0822 vs 0.0970), ma KIVI-4 resta a 10.539 PPL e 0.0038 KL usando
5.04 bit. Sul 3B il transfer B=2 fa 6.408, peggio di RD-V2 6.373 e KIVI-4
6.330. Servono quindi ancora due punti distinti del fronte; nessuno domina
tutti gli assi da solo.

### Fronte a due punti e composizione RD--Born

Non è necessario obbligare un solo punto da circa 2 bit a confrontarsi con
baseline che spendono 5 bit. Il Geodesia graded B=5 è già un secondo punto
utile del fronte:

- sull'aggregato 8B a dieci finestre fa 7.8459 PPL a 4.9921 bit, contro
  KIVI-4 7.8706 a 5.0430 bit;
- nella misura di fedeltà 8B fa KL 0.000684 a 4.9944 bit, contro KIVI-4
  0.001896 a 5.0326 bit;
- sul nuovo test 3B a sei finestre B=5 fa 6.3368 a 4.9923 bit, contro KIVI-4
  6.3303 a 5.0430 bit: il gap è 0.0065 PPL, quindi è quasi pareggio ma non
  dominanza;
- spendere il margine di capacità con B=5.05 peggiora a 6.3516: l'allocazione
  discreta non è monotona in PPL e non va inseguita ulteriormente sul test.

L'interpretazione quantum-inspired più utile è quindi una **frontiera di stati
operativi**: RD-V2 per la fascia 2 bit, graded B=5 per la fascia 5 bit. Non è
invece utile comporre due “misure” nella stessa query. È stato provato
`RD-V2 + gate Born` mantenendo la copia esatta scelta da
`massa * residuo_V^2` e applicando online una proiezione a fedeltà
`eta={0.9,0.95}`, bias `{1,2}`. Sulle tre finestre di calibration il migliore
fa 9.770 PPL, contro 9.547 di RD-V2 senza Born, e richiede circa 22--23 s per
configurazione contro 7--8 s del percorso semplice. Non merita un holdout.

File aggiuntivi:

- `results/qwen3b_16k_b5_frozen_test_six_windows.json`
- `results/qwen3b_16k_b5p05_capacity_matched_test_six_windows.json`
- `results/qwen8b_16k_rd_saliency_born_composition_calibration.json`

File:

- `results/qwen8b_16k_rd_saliency_calibration_offsets147456_180224.json`
- `results/qwen8b_16k_rd_saliency_frozen_holdout_offsets196608_229376.json`
- `results/qwen8b_16k_rd_saliency_frozen_final_test_offsets245760_278528.json`
- `results/qwen3b_16k_rd_saliency_transfer_test_six_windows.json`
- `results/qwen30b_a3b_q8_16k_rd_saliency_transfer_test_two_windows.json`
- `results/qwen30b_a3b_q8_16k_rd_saliency_transfer_test_four_more_windows.json`
- `results/qwen30b_a3b_q8_16k_missing_baselines_test_six_windows.json`
- `results/qwen8b_16k_rd_saliency_memory_quality.json`
- `results/qwen8b_16k_graded_exact_mix_calibration_offsets147456_180224.json`
- `results/qwen8b_16k_graded_exact_mix_kl_front.json`
- `results/qwen8b_16k_graded_exact_mix_b2_frozen_test_six_windows.json`
- `results/qwen3b_16k_graded_exact_mix_b2_transfer_test_six_windows.json`

### Punto 5-bit: graded + residuo esatto RD

La stessa salienza target-free di RD-V2 è stata applicata come piccolo residuo
esatto sopra una base graded, imponendo analiticamente:

```text
budget_base + 16 * quota_residuo ~= 5 bit/valore
saliency_token = massa_prefill * ||V - mean_block(V)||^2
```

La selezione usa soltanto il prompt e viene congelata; nessun target della
continuazione entra nella policy.

**Calibrazione 8B.** Su tre finestre già destinate alla calibration, B=4.5 più
3.125% esatto fa 9.670 PPL a 4.991 bit, contro B=5 puro 9.739 a 4.992.
Congelato sulle sei finestre successive fa 10.533 contro KIVI-4 10.539 nel
vecchio protocollo Q=64. La KL separata è 0.0024 a 4.99 bit contro KIVI-4
0.0039 a 5.09.

**Calibrazione 3B per famiglia.** Su quattro finestre `wikitext2_valid`, il
punto stabile è B=4.875 più 0.78125% esatto, decay 0.25: 6.484 PPL, contro
6.495--6.498 dei punti vicini/puro. Sul test Q=64 resta quasi pari a KIVI-4.
Con Q=256, nello stesso processo, fa 6.368 a 4.99 bit contro KIVI-4 6.385 a
5.17; il cambio di Q ha però rivelato l'artefatto del chunk corrente e questo
JSON non è più la prova primaria. La KL è 0.0030 a 4.99 bit contro 0.0041 a
5.09.

### Protocollo PPL incrementale

`run_ppl(..., incremental=True)` e `budget_sweep.py --incremental-eval`
eseguono ora:

1. prefill esatto del prefisso;
2. una chiamata Q=1 per ogni token teacher-forced;
3. append del token nella stessa `past_key_values`;
4. accumulo online della NLL sul token successivo.

Il test sintetico controlla che chunked e incrementale usino gli stessi target
e producano la stessa NLL per un modello causale deterministico. Sul modello
BF16 reale Full-KV può differire leggermente per l'ordine delle operazioni, per
cui il confronto finale deve usare soltanto righe tutte incrementali.

Risultati incrementali Q=1, 64 token per finestra, sei finestre:

| Modello | Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---|---:|---:|---:|
| Qwen2.5-3B | KIVI-4 | 5.03 | 5.03 | 6.322 |
| Qwen2.5-3B | Geodesia B=4.875 + 0.78125% RD | **4.96** | **4.96** | **6.320** |
| Qwen3-8B | KIVI-4 | 5.03 | 5.03 | 10.581 |
| Qwen3-8B | Geodesia B=4.5 + 3.125% RD | **4.99** | **4.99** | **10.540** |

Questi sono i confronti PPL primari correnti contro KIVI-4, ma **nessuno dei
due ha ancora un intervallo di confidenza**: i margini sono 0.002 e 0.041 PPL,
e su PG-19 un margine dello stesso ordine si è rivelato indistinguibile dal
rumore. Vanno rigenerati con `token_nll` e passati a `paired_stats.py` prima di
essere usati come prova. Il 3B richiede
circa 170 s per KIVI e 434 s per Geodesia; l'8B 346 s e 1149 s. Sono tempi del
simulatore Python che conserva K/V dense e ricostruisce la rappresentazione a
ogni token, non latenza di deployment. `--skip-certificate-validation` evita
solo il doppio calcolo dense usato per misurare l'errore vero; non modifica
output o allocazione.

File:

- `results/qwen8b_16k_graded_exact_mix_5bit_calibration.json`
- `results/qwen8b_16k_graded_exact_mix_b4p5_frozen_test_six_windows.json`
- `results/qwen8b_16k_graded_exact_mix_b4p5_memory_quality.json`
- `results/qwen3b_16k_graded_exact_mix_5bit_validation_calibration.json`
- `results/qwen3b_16k_graded_exact_mix_b4p875_frozen_test_six_windows.json`
- `results/qwen3b_16k_graded_exact_mix_b4p875_memory_quality.json`
- `results/qwen3b_16k_full_comparison_eval256_six_windows.json`
- `results/qwen3b_16k_kivi4_vs_graded_mix_incremental_eval64_six_windows.json`
- `results/qwen8b_16k_full_comparison_eval256_frozen_six_windows.json`
- `results/qwen8b_16k_kivi4_vs_graded_mix_eval1024_frozen_six_windows.json`
- `results/qwen8b_16k_kivi4_vs_graded_mix_incremental_eval64_frozen_six_windows.json`

Quest 8B incrementale sulle stesse sei finestre fa 10.238 PPL, 16.25 bit
residenti e 2.32 letti/query. Batte quindi il punto Geodesia 5-bit in PPL e
traffico; questo ha motivato la variante seguente.

### Compressed-Quest con summary esatti (`box_sparse`)

Il gate box precedente calcolava min/max sulle key già ricostruite e applicava
un bias finito dopo avere letto l'intera cache: non poteva battere Quest
sull'asse traffico. `query_gate="box_sparse"` implementa invece:

1. min/max **esatti** delle key per ogni blocco, congelati prima della
   quantizzazione; overhead residente e letto `2 * D * fp16` per blocco, cioè
   circa 0.25 bit/valore;
2. ranking top-k sulle sole pagine completamente chiuse;
3. pagina corrente sempre letta, senza sink/window aggiuntivi fuori dal top-k;
4. hard mask delle pagine fredde;
5. K/V delle pagine selezionate lette dal formato graded packed;
6. contabilità sparse = tutti i summary + soli blocchi top-k/corrente.

Il test di regressione verifica che i summary chiusi coincidano bitwise in fp16
con min/max delle K originali, che l'output sia finito e che letture < capacità.
La suite saliva così a 25 test; con i due test paired del 2026-07-29 è a 27.

Calibrazione 8B a tre finestre, quota pagine 12.5%:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Quest | 16.25 | 2.31 | 9.742 |
| Compressed-Quest B=10 | **10.24** | **2.04** | **9.715** |
| Compressed-Quest B=12 | 12.24 | 2.22 | 9.750 |
| Compressed-Quest B=14 | 14.23 | 2.29 | 9.817 |
| Compressed-Quest B=15 | 15.23 | 2.31 | 9.755 |

B=10 è stato congelato. Sulla prima finestra test incrementale:

| Metodo | bit residenti/valore | bit letti/valore | PPL |
|---|---:|---:|---:|
| Quest | 16.25 | 2.32 | 14.011 |
| Compressed-Quest B=10 | **9.88** | **1.96** | **13.860** |

Sulle cinque finestre rimanenti B=10 fa 9.675 PPL. Combinando geometricamente
lo smoke e queste cinque finestre:

| Metodo | bit residenti/valore | bit letti/valore | PPL 6 finestre | KL |
|---|---:|---:|---:|---:|
| Quest | 16.25 | 2.32 | **10.238** | 0.0464 |
| Compressed-Quest B=10 | **~9.85** | **~1.96** | 10.272 | **0.0460** |

Compressed-Quest vince PPL in 2 finestre su 6 e perde in 4; il gap aggregato è
0.034 PPL (circa 0.34%). È quindi Pareto-superiore su capacità, traffico e KL,
ma non sulla PPL. Il corpus ha 298,938 token con il tokenizer Qwen3-8B: dopo
l'ultimo offset 278,528 non resta un'altra finestra completa da 16k.

Ulteriori tentativi fatti **solo sulla calibration**:

- residuo token esatto RD 6.25--25%: peggiora PPL e porta le letture a
  2.50--3.15 bit;
- quota pagine 14--16% a B=10 e 17--18% a B=8: alcuni punti migliorano PPL ma
  superano Quest nel traffico;
- fronte interpolato: B=9/q=16% domina Quest in calibration
  (9.709 PPL, 9.24 residenti, 2.28 letti) ma fallisce lo smoke incrementale
  con 15.858 PPL;
- B=10/q=15% rispetta il traffico incrementale (2.28) ma fa 14.148 contro
  Quest 14.011 sulla prima finestra;
- protezione statica 5--20% e decay 0.25/0.75/0.9/0.95 non migliorano B=10
  con decay 0.5.

Non fare altro tuning sulle sei finestre test. Il gap va affrontato su un nuovo
corpus/calibration o con una policy per layer/head, non scegliendo altri punti
dopo avere osservato questi risultati.

File:

- `results/qwen8b_16k_quest_incremental_eval64_frozen_six_windows.json`
- `results/qwen8b_16k_compressed_quest_aligned_reader_calibration.json`
- `results/qwen8b_16k_compressed_quest_aligned_reader_incremental_smoke_offset196608.json`
- `results/qwen8b_16k_compressed_quest_aligned_reader_incremental_five_remaining_windows.json`
- `results/qwen8b_16k_compressed_quest_b10_memory_quality.json`
- `results/qwen8b_16k_compressed_quest_rd_residual_calibration.json`
- `results/qwen8b_16k_compressed_quest_readmatched_fraction_calibration.json`
- `results/qwen8b_16k_compressed_quest_interpolated_front_calibration.json`
- `results/qwen8b_16k_compressed_quest_b9_q16_incremental_smoke_offset196608.json`
- `results/qwen8b_16k_compressed_quest_b10_q15_incremental_smoke_offset196608.json`
- `results/qwen8b_16k_compressed_quest_protected_front_calibration.json`
- `results/qwen8b_16k_compressed_quest_decay_calibration.json`

### Trasferimento congelato su PG-19

Per non continuare a ottimizzare sulle finestre WikiText-2 già osservate è
stato aggiunto `benchmarks/fetch_pg19.py`. Il fetcher legge in streaming
`emozilla/pg19` e salva testo, confini dei libri, metadati e SHA256 in modo
atomico. Dataset e revisione sono fissati a:

```text
emozilla/pg19
c021754c8e01c5b1cc83a1f549c1f97fbbb756b8
```

Artefatti locali:

- validation: 6 libri, 3,297,628 caratteri, SHA256
  `b925f5f8b3b7c782d908e7702fdcf0068b9d12293104e31ed73deb2037d53059`;
- test: 12 libri, 3,068,739 caratteri, SHA256
  `42a0b97ff79ec96f53e18ec14bd7704645bd10b7c50125935dbae721e9e8b445`;
- i testi sono ignorati da Git per dimensione, mentre i manifest
  `paper/pg19_{validation,test}.txt.manifest.json` vanno conservati.

Il tokenizer Qwen3-8B produce 818,348 token validation e 758,994 token test.
Le finestre validation, collocate in libri distinti, sono
`67981, 191181, 277205, 339425, 594462`. Lo screening Q=64 chunked ha
trasferito i parametri congelati senza retuning:

| Coppia | Baseline PPL | Geodesia PPL | baseline/Geo bit residenti | baseline/Geo bit letti |
|---|---:|---:|---:|---:|
| Quest / compressed-Quest B=10 | 11.644 | **11.593** | 16.25 / **10.24** | 2.31 / **2.03** |
| KIVI-4 / graded+RD 5-bit | 10.857 | **10.801** | 5.04 / **4.99** | 5.04 / **4.99** |

Questa tabella era soltanto validation e chunked. Il test finale Q=1
incrementale usa sei centri-libro già elencati prima dei risultati:
`20668, 402659, 736259, 90369, 169602, 246823`, 64 target ciascuno. I primi
tre coprono primo/centrale/ultimo libro; i tre aggiuntivi sono i successivi
offset non usati della lista preregistrata.

| Coppia test incrementale, 6 libri | Baseline PPL | Geodesia PPL | baseline/Geo bit residenti | baseline/Geo bit letti | vittorie Geo |
|---|---:|---:|---:|---:|---:|
| Quest / compressed-Quest B=10 | 17.022 | **16.931** | 16.25 / **9.83** | 2.32 / **1.95** | 4/6 |
| KIVI-4 / graded+RD 5-bit | **16.147** | 16.258 | 5.03 / **4.99** | 5.03 / **4.99** | 2/6 |

Per compressed-Quest la dominanza locale completa si trasferisce quindi su
PG-19 senza retuning: PPL, capacità e traffico migliorano insieme. Il vantaggio
PPL è piccolo (circa 0.53%) e il campione è di soli 384 target, quindi va
esteso con intervalli paired prima di una claim forte.

Su questo sottoinsieme il punto 5-bit sembrava non trasferire la dominanza PPL:
perdeva circa 0.69% e vinceva soltanto agli offset 20,668 e 90,369. **La
sezione successiva mostra che questa conclusione era un artefatto del campione
e non va più usata.**

Durante l'estensione è stato inizialmente trascritto `tpr=0` invece di
`tpr=.03125`. Quel run misura correttamente il solo graded B=4.5: 24.307 PPL a
4.486 bit sui tre libri aggiuntivi. Il run corretto graded+RD fa 24.222 a
4.985 bit; il residuo esatto aiuta, ma KIVI-4 resta a 24.121. Il file senza
`corrected` non deve essere usato come risultato del candidato 5-bit.

File:

- `benchmarks/fetch_pg19.py`
- `paper/pg19_validation.txt.manifest.json`
- `paper/pg19_test.txt.manifest.json`
- `results/qwen8b_pg19_16k_frozen_transfer_validation_eval64.json`
- `results/qwen8b_pg19_16k_quest_vs_compressed_quest_incremental_test_three_books.json`
- `results/qwen8b_pg19_16k_quest_vs_compressed_quest_incremental_test_three_more_books.json`
- `results/qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_test_three_books.json`
- `results/qwen8b_pg19_16k_graded_mix_incremental_test_three_more_books_corrected.json`
- `results/qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_test_three_more_books.json`
  è il controllo graded puro dovuto alla trascrizione `tpr=0`, non il candidato
  finale.

### Intervalli paired e PG-19 esteso a 11 libri (2026-07-29)

Il limite dichiarato del test precedente era il campione: sei finestre da 64
target sono 384 confronti, e su questo progetto un campione corto ha già
ribaltato una conclusione almeno due volte (3B da Q=64 a Q=256). Sono state
quindi fatte due cose insieme.

**1. Esaurire la lista di offset preregistrata invece di sceglierne altri.**
Gli offset PG-19 seguono la regola "finestra da 16,384 token centrata nel
libro". La regola è stata ricalcolata dal manifest e riproduce esattamente
cinque dei sei offset già usati; il sesto differisce di un token per
arrotondamento del centro (246,822 contro 246,823 usato). Dei 12 libri del
test:

- 6 erano già consumati: `20668, 90369, 169602, 246823, 402659, 736259`;
- 5 sono validi e mai usati: `320206, 472081, 529495, 584324, 659151`
  (libri 6, 8, 9, 10, 11);
- 1 è impossibile: il libro 5, *Odd Craft Part 4*, ha 6,679 token, meno di una
  finestra.

Le 11 finestre valide sono quindi l'intera lista preregistrata, non una
selezione: non resta margine per cherry-picking sul test.

**2. Rendere possibile un intervallo paired.** `run_ppl(...,
return_token_nll=True)` restituisce ora la NLL di ogni singolo target e
`budget_sweep.py` la serializza in `ppl_windows[].token_nll`. I target
dipendono soltanto dalla finestra e non dalla policy, quindi due policy
valutate sulla stessa finestra producono liste allineate token per token.
`benchmarks/paired_stats.py` stima `mean(nll_a - nll_b)`, che è esattamente
`log(PPL_a / PPL_b)`, con un bootstrap che ricampiona **le finestre intere**:
i token dello stesso libro condividono prefisso, cache e stile, quindi un
bootstrap sui singoli token sarebbe anticonservativo. Il numero ridotto di
cluster si riflette onestamente nell'ampiezza dell'intervallo.

Controllo di validità: sulle 6 finestre già pubblicate il nuovo run riproduce
**16.147** e **16.258**, cioè i valori esatti della tabella precedente. La
modifica al runner è neutra sul comportamento e il protocollo è deterministico.

Coppia KIVI-4 contro graded+RD 5-bit, 11 finestre, 693 target appaiati:

| | 6 libri | 11 libri |
|---|---:|---:|
| KIVI-4 PPL | 16.147 | 15.475 |
| Geodesia graded+RD PPL | 16.258 | 15.490 |
| bit residenti KIVI-4 / Geodesia | 5.03 / 4.99 | 5.03 / **4.99** |
| vittorie Geodesia | 2/6 | 5/11 |
| gap PPL | −0.69% | −0.10% |

`mean(nll_KIVI - nll_Geodesia) = -0.00096`, CI95% `[-0.00959, +0.00792]`;
rapporto PPL `0.99904`, CI95% `[0.99046, 1.00795]`. **L'intervallo contiene lo
zero.** Non c'è quindi né dominanza né fallimento: a 4.99 contro 5.03
bit/valore le due PPL sono indistinguibili sul campione disponibile. Il claim
difendibile è "meno capacità senza costo di PPL misurabile", non "batte
KIVI-4".

Conseguenze operative:

- non riportare più il 5-bit come fallimento PG-19, né come vittoria;
- il gap 8B WikiText-2 di 0.008 PPL contro KIVI-4 e il quasi-pareggio 3B di
  0.002 sono, in ordine di grandezza, ben dentro questo intervallo: vanno
  ricalcolati con `paired_stats.py` prima di qualsiasi conclusione, e i JSON
  WikiText-2 vanno rigenerati perché non contengono ancora `token_nll`;
- per distinguere differenze di questo ordine servono più libri/corpora, non
  altri iperparametri.

Comando:

```bash
.venv/bin/python -m benchmarks.budget_sweep \
  --model Qwen/Qwen3-8B --tokens 16384 --eval-tokens 64 \
  --incremental-eval --skip-certificate-validation --skip-passkey \
  --text paper/pg19_test.txt \
  --text-offsets 20668 90369 169602 246823 320206 402659 472081 529495 \
                 584324 659151 736259 \
  --geodesia-settings '4.5:0.5:1:0:mass:0:0.03125::0:1:0:2:0' \
  --only 'KIVI k4v4' 'Geodesia-KV B=' \
  --out results/qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_eleven_books.json

.venv/bin/python -m benchmarks.paired_stats \
  --files results/qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_eleven_books.json \
  --a 'KIVI k4v4' --b 'Geodesia-KV B=' \
  --out results/qwen8b_pg19_16k_paired_kivi4_vs_graded_mix_eleven_books.json
```

**Coppia Quest contro compressed-Quest B=10, stesse 11 finestre.** Qui
l'intervallo va nella direzione opposta e la vittoria regge:

| | 6 libri | 11 libri |
|---|---:|---:|
| Quest PPL | 17.022 | 16.043 |
| Compressed-Quest PPL | 16.931 | **15.961** |
| bit residenti Quest / compressed | 16.25 / 9.83 | 16.25 / **9.84** |
| bit letti Quest / compressed | 2.32 / 1.95 | 2.32 / **1.96** |
| vittorie compressed | 4/6 | 8/11 |

`mean(nll_Quest - nll_compressed) = +0.00514`, CI95% `[+0.00075, +0.00941]`;
rapporto PPL `1.00516`, CI95% `[1.00076, 1.00945]`. **L'intervallo esclude lo
zero**: compressed-Quest ha NLL inferiore in modo statisticamente
distinguibile, oltre a usare il 39% di capacità in meno e il 16% di traffico in
meno. Zero violazioni del certificato su 798,336 chiamate.

Va detto con precisione quanto è forte: il limite inferiore è +0.08% sul
rapporto di PPL, quindi l'effetto è **reale ma piccolo**, e la significatività
è marginale. La differenza rispetto al punto 5-bit non sta nella dimensione del
margine PPL — sono entrambi sotto l'1% — ma nella *consistenza*: 8 finestre su
11 con lo stesso segno contro 5 su 11.

Il quadro complessivo delle due coppie è quindi asimmetrico e ora quantificato:

| Coppia, 11 libri PG-19 | rapporto PPL | CI95% | verdetto |
|---|---:|---|---|
| KIVI-4 / graded+RD 5-bit | 0.99904 | [0.99046, 1.00795] | indeciso |
| Quest / compressed-Quest B=10 | 1.00516 | [1.00076, 1.00945] | compressed vince |

Il ramo sparse trasferisce fuori distribuzione una dominanza su tutti e tre gli
assi misurati (PPL, capacità, traffico); il ramo 5-bit trasferisce una parità
di PPL a capacità inferiore. Nessuno dei due è il fallimento descritto ieri.

File:

- `results/qwen8b_pg19_16k_kivi4_vs_graded_mix_incremental_eleven_books.json`
- `results/qwen8b_pg19_16k_paired_kivi4_vs_graded_mix_eleven_books.json`
- `results/qwen8b_pg19_16k_quest_vs_compressed_quest_incremental_eleven_books.json`
- `results/qwen8b_pg19_16k_paired_quest_vs_compressed_quest_eleven_books.json`

## Capacità di contesto in VRAM (2026-07-29)

`benchmarks/context_capacity.py` misura quanto contesto entra realmente in VRAM
tenendo il modello quantizzato residente. È il primo asse di sistema del
progetto con misure reali e non simulate, ma **non** è l'integrazione P2: prova
la residenza della memoria, non l'esecuzione del formato compresso dentro
Transformers.

Cosa è misurato davvero:

- pesi Qwen3-8B NF4 (bitsandbytes, double quant): **5.670 GiB** su A6000 da
  47.40 GiB utilizzabili;
- picco di attivazione di un passo di decode Q=1 su cache piena: **24.3 MiB**,
  cioè trascurabile;
- byte reali per token del formato packed Geodesia, costruito da K/V veri del
  modello con i livelli scelti dall'allocatore vero: **1.959 bit/valore
  misurati contro 1.990 contabilizzati**. La contabilità del progetto è quindi
  leggermente *conservativa*, non ottimistica;
- massimo contesto per bisezione con allocazione reale, e per Full-KV con un
  passo di decode realmente eseguito.

Qwen3-8B ha GQA con 8 KV-head e 36 layer: 73,728 valori KV per token, cioè
144.0 KiB/token in BF16.

| Tecnica | bit/valore | KiB/token | contesto max VRAM | × finestra nativa |
|---|---:|---:|---:|---:|
| Full-KV BF16 | 16.00 | 144.0 | 283,238 | 6.9x |
| Quest p=32 | 16.25 | 146.3 | 278,872 | 6.8x |
| Compressed-Quest B=10 | 9.84 | 88.5 | 460,733 | 11.2x |
| KIVI-4 | 5.03 | 45.3 | 900,751 | 22.0x |
| Geodesia graded+RD 5-bit | 4.99 | 44.9 | 908,227 | 22.2x |
| KIVI-2 | 3.05 | 27.5 | 1,485,462 | 36.3x |
| **Geodesia RD-V2 2-bit** | **2.06** | **18.5** | **2,198,845** | **53.7x** |
| StreamingLLM b=2048 | — | costante | illimitato | ritiene solo 2048 |
| SnapKV b=2048 | — | costante | illimitato | ritiene solo 2048 |

Verifica per bisezione, che **supera** le proiezioni e le conferma
conservative:

- Full-KV con decode reale: **294,911** token contro 283,238 calcolati;
- Geodesia 2-bit con allocazione reale: **2,328,994** token contro 2,198,845.

**Il risultato non va però riportato come "Geodesia permette più contesto".**
La finestra posizionale di Qwen3-8B è 40,960 token: già Full-KV non compresso
ne tiene 7.2× senza esaurire la VRAM. Su questa GPU e per questo modello la
VRAM **non è il vincolo** al contesto massimo; lo è la finestra addestrata. La
classifica esiste ma non è vincolante, e sarebbe disonesto presentarla come se
lo fosse. La causa è la GQA: con 8 sole KV-head la cache costa 144 KiB/token,
e a 40,960 token Full-KV occupa appena 5.6 GiB.

Le policy a eviction vanno lette a parte: la loro cache è costante, quindi il
contesto processabile è illimitato in memoria, ma il contesto **ritenuto**
resta il budget (2048 token). Metterle nella stessa colonna delle altre
produrrebbe un falso vincitore.

L'asse su cui la compressione vincola davvero è invece la **concorrenza**, cioè
quante sequenze stanno insieme in VRAM a contesto fisso:

| Tecnica | sequenze @40,960 | sequenze @131,072 |
|---|---:|---:|
| Full-KV BF16 | 6 | 2 |
| Quest p=32 | 6 | 2 |
| Compressed-Quest B=10 | 11 | 3 |
| KIVI-4 | 21 | 6 |
| Geodesia graded+RD 5-bit | 22 | 6 |
| KIVI-2 | 36 | 11 |
| **Geodesia RD-V2 2-bit** | **53** | **16** |

Qui il vantaggio è di 8.8× rispetto a Full-KV e di 2.4× rispetto a KIVI-4, ed è
un fattore moltiplicativo sul throughput servibile, non una frazione di punto
percentuale come i confronti di PPL. È l'argomento di sistema più forte
disponibile oggi, e non dipende dai margini statisticamente indecisi della
fascia 5 bit.

Limiti da dichiarare sempre insieme al numero:

- i byte per token delle baseline KIVI/Quest sono proiezioni dalla contabilità
  bit/valore, non formati residenti implementati; solo Geodesia ha un packed
  reale misurato;
- la verifica prova che la memoria si alloca, non che il modello generi a
  quella lunghezza; oltre 40,960 posizioni servirebbe RoPE scaling, non
  dimostrato qui;
- la concorrenza è calcolata sulla sola KV cache, senza il costo per-sequenza
  di attivazioni e scheduler di un vero motore di serving.

### Capacità per taglia di VRAM e confronto 3B

Lo script accetta `--vram-gib` per proiettare schede diverse dai pesi e
dall'attivazione misurati, e `--emulate-vram-gib`, che occupa la VRAM in
eccesso con una zavorra e **ripete la bisezione reale sotto quel tetto**. Il
16 GiB è quindi misurato, non stimato; il 64 GiB resta una proiezione perché la
scheda disponibile è da 48.

Qwen2.5-3B-Instruct ha 2 sole KV-head contro le 8 dell'8B: 36.0 KiB/token
contro 144.0, cioè una cache 4× più piccola, e pesi NF4 di 1.92 GiB contro
5.67. Massimo contesto per una singola sequenza:

| VRAM | 8B Full-KV | 8B KIVI-4 | 8B Geodesia 2-bit | 3B Full-KV | 3B Geodesia 2-bit |
|---:|---:|---:|---:|---:|---:|
| 16 GiB | 76,042 | 241,830 | 590,338 | 413,654 | 3,211,295 |
| 24 GiB | 134,795 | 428,673 | 1,046,445 | 648,663 | 5,035,721 |
| 48 GiB | 311,051 | 989,202 | 2,414,764 | 1,353,690 | 10,509,000 |
| 64 GiB | 428,556 | 1,362,887 | 3,326,977 | 1,823,708 | 14,157,852 |

Verifiche reali con decode eseguito:

| Configurazione | proiezione | misurato | scarto |
|---|---:|---:|---:|
| 8B, 48 GiB, Full-KV | 283,238 | **294,911** | proiezione conservativa |
| 8B, 16 GiB emulati, Full-KV | 76,042 | **70,655** | proiezione ottimista del 7% |
| 3B, 48 GiB, Full-KV | 1,310,793 | **1,290,239** | ottimista dell'1.6% |
| 3B, 16 GiB emulati, Full-KV | 413,654 | **393,215** | ottimista del 5% |

Sotto un tetto stretto la proiezione sbaglia per eccesso del 5--7%, perché la
zavorra frammenta l'allocatore. Usare quindi i valori proiettati con un margine
del 10%, non al bit.

**Conclusione pratica.** Entrambi i modelli arrivano al massimo a 131,072
posizioni con RoPE scaling (32,768 e 40,960 nativi). Confrontando quel tetto
con la tabella:

- **3B**: Full-KV regge 413,654 token già su 16 GiB, cioè 3.2× il tetto
  massimo del modello. La compressione **non sblocca nessun contesto** a
  nessuna taglia di VRAM ragionevole; serve solo alla concorrenza.
- **8B su 16 GiB**: Full-KV si ferma a 70,655 token misurati e **non raggiunge
  131,072**, che richiederebbe 18.0 GiB di sola cache. Con KIVI-4 o Geodesia
  5-bit si arriva a circa 242,000, con Geodesia 2-bit a 590,000. Questo è
  l'unico punto in cui la compressione sblocca davvero qualcosa: la finestra
  completa del modello su una scheda da 16 GiB.
- **8B da 24 GiB in su**: Full-KV supera 131,072 da solo e non c'è più nulla da
  sbloccare sull'asse contesto.

Quindi il beneficio reale della compressione KV su questi due modelli non è il
contesto massimo, tranne che nella singola combinazione 8B/16 GiB. È la
concorrenza, e in misura minore la banda letta.

File: `results/qwen8b_context_capacity_nf4.json`,
`results/qwen3b_context_capacity_nf4.json`.

Comando:

```bash
.venv/bin/python -m benchmarks.context_capacity \
  --model Qwen/Qwen3-8B --quantization nf4 \
  --out results/qwen8b_context_capacity_nf4.json
```

## Cache KV realmente quantizzata nel modello (2026-07-29)

`geodesia_kv/live_cache.py` è la prima cache del progetto che **non simula**:
i blocchi chiusi vivono in VRAM soltanto come interi impacchettati a 2/4/8 bit
più scale fp16, e la K/V densa corrispondente viene liberata alla chiusura del
blocco. Resta esatta una coda recente (`window`) e i primi `sinks` token.
È una `Cache` di Transformers, quindi gira dentro il modello vero.

Design e suo costo: `update()` restituisce la ricostruzione densa del **solo
layer corrente**, che è transitoria. Il picco è quindi "stato quantizzato di
tutti i layer + ricostruzione densa di uno". È il costo reale di un design che
dequantizza in lettura, e come mostrato sotto è ciò che limita il guadagno.

Limite da dichiarare sempre: questo percorso implementa **quantizzazione
uniforme**, non l'allocatore graded con centroidi e residui esatti. Non
riproduce i bit/valore né la qualità della policy completa.

### Qualità della cache viva

Qwen2.5-3B-Instruct, sei finestre da 16k, 64 target, protocollo del paper
(`results/qwen3b_live_cache_quality_16k_six_windows.json`):

| Cache | PPL | bit/valore | residente |
|---|---:|---:|---:|
| Full-KV bf16 | 6.3174 | 16.000 | 576.0 MiB |
| live 8-bit | 6.3348 | 8.617 | 310.2 MiB |
| live 4-bit | 6.3957 | 4.742 | 170.7 MiB |
| live 2-bit | **9.5454** | 2.804 | 100.9 MiB |

**Il 2 bit uniforme non è utilizzabile**: +51% di PPL. Il 2.06 bit/valore del
paper viene dall'allocatore graded che assegna livelli adattivi, non da 2 bit
uniformi ovunque. Il punto operativo difendibile della cache viva è il 4 bit,
che costa +1.2% di PPL. Non usare il 2 bit uniforme per claim di capacità.

### Contesto massimo sotto tetto di VRAM imposto

`benchmarks/live_context.py` impone il tetto con
`torch.cuda.set_per_process_memory_fraction`, carica il modello NF4 e allunga
il contesto con prefill reali finché il processo non esaurisce la memoria. A
ogni checkpoint esegue un passo di decode vero e verifica che i logit siano
finiti: il contesto riportato è usato, non solo allocato. Le due politiche
percorrono lo stesso identico path del modello.

Qwen2.5-7B-Instruct-1M (28 layer, 4 KV-head, `max_position_embeddings`
1,010,000), pesi NF4 (5.18 GiB), cache viva a 4 bit.

**Prima misura (percorso `sdpa`), poi diagnosi.** Il guadagno era 2.07× a 16
GiB (114,688 contro 237,568), molto sotto i 3.4× del rapporto fra i residenti.
La decomposizione del picco ha mostrato perché: il transitorio cresceva
linearmente col contesto e a 524k valeva 13.07 GiB contro 7.74 di stato
residente. **Leggere la cache costava più che conservarla.**

Due cause, entrambe corrette:

1. **Replica GQA.** Il path SDPA di Transformers usa `enable_gqa` solo se
   `attention_mask is None`. In prefill a chunk con cache non vuota la maschera
   esiste sempre, quindi veniva chiamato `repeat_kv`, che replica K e V da 4 a
   28 head **sull'intera cache**: circa 7.5 GiB per layer a 524k. Misurato:
   108 chiamate su 144 passavano di lì. La maschera serviva solo perché
   `is_causal=True` di SDPA allinea in alto a sinistra, mentre un chunk con
   `past` va allineato in basso a destra. `gqa_causal_attention` usa
   `causal_lower_right`, che esprime quella semantica ed è accettata dai kernel
   efficienti: errore 0.0 contro il riferimento espanso nel test isolato.
2. **Copia doppia in `_materialize`**: lista di blocchi dequantizzati più
   `torch.cat` valeva due volte il layer denso. Ora l'output è preallocato.

**Risultati dopo le correzioni**, contesto massimo con OOM reali:

| Tetto | Full-KV bf16 | cache viva 4-bit | rapporto |
|---|---:|---:|---:|
| 16 GiB | 151,552 (OOM) | **638,976** (OOM) | 4.22× |
| 32 GiB | 319,488 (OOM) | **1,010,000 (target raggiunto)** | >3.16× |

A 32 GiB la cache quantizzata copre **l'intera finestra del modello**: 14.76
GiB residenti, picco 22.05, 4.38 bit/valore, logit finiti al decode di
verifica a 1,010,000 token. Full-KV non supera 319,488.

Il transitorio è ora sostanzialmente piatto invece che lineare: 0.60 GiB a
131k, 0.61 a 262k, 0.87 a 393k, 1.11 a 524k, contro 3.32/6.57/9.82/13.07 prima.
È questo, non la compressione, ad avere sbloccato il contesto: la proiezione
pre-correzione stimava ~384k a 16 GiB e il risultato è 638,976, quindi era
conservativa del 66%.

Il rapporto 4.22× **supera** il rapporto fra i residenti (3.4× a 4 bit) perché
Full-KV paga lo stesso transitorio assoluto su un budget già saturato dalla
cache densa.

Avvertenza di confronto: `gqa_causal` non è bitwise identico a `sdpa`. Sui
logit la differenza relativa è 8.8e-3 con argmax invariato, sulla PPL
0.44--0.63%, cioè lo stesso ordine dello 0.45% già documentato fra modalità Q
diverse. Vale la stessa regola: confrontare solo run con la stessa
implementazione di attenzione. Il percorso assume **nessun padding** (una
sequenza per batch).

Resta il limite di attribuzione: questa è quantizzazione uniforme e a 4 bit con
gruppo 32 **è KIVI**. Il risultato dimostra che una cache KV compressa sblocca
contesto, non che Geodesia batta KIVI. Per quello servono il kernel fuso e
l'allocatore graded nella cache viva.

File:

- `geodesia_kv/live_cache.py`
- `benchmarks/live_context.py`
- `results/qwen3b_live_cache_quality_16k_six_windows.json`
- `results/qwen7b1m_live_context_16gib.json`
- `results/qwen7b1m_live_context_32gib.json`

Comando:

```bash
.venv/bin/python -m benchmarks.live_context \
  --model Qwen/Qwen2.5-7B-Instruct-1M --vram-gib 16 \
  --policies full quantized --bits 4 --target 262144 \
  --checkpoints 65536 131072 196608 262144 \
  --out results/qwen7b1m_live_context_16gib.json
```

## Esperimenti negativi: non ripeterli senza una nuova ipotesi

- Peso esplicito dell'errore sulle value nell'allocatore: riduce leggermente
  l'errore di attenzione ma peggiora la PPL. Il key-only objective ha vinto.
- `alloc_mass_power > 1`: non ha migliorato il fronte; power 1 è il migliore.
- Rotazione Hadamard: peggiora qualità e retrieval.
- Asimmetria più bit alle key/meno alle value: peggiora.
- VQ con più centroidi: utile soltanto sotto circa 1.5 bit/valore; key simili non
  implicano value simili.
- Gruppi più fini a 2 bit: dominati da più bit con gruppi più grossi.
- Tuning su una sola finestra: produce falsi vincitori; usare sempre holdout o
  quattro finestre.
- Gate query-adaptive a centroidi, protezione statica di blocchi e residuo
  token-level selezionato dal prompt: migliorano finestre specifiche ma
  falliscono sugli holdout disgiunti.
- Rappresentazione spettrale K--V a rango 1--4, cumulante log-partition e gate
  Born: tutte hanno un razionale fisico/statistico e restano certificate, ma
  nessuna batte SnapKV/KIVI-2 sul test finale 8B.
- Ibrido con 10.2% di token esatti Snap-like e coda a centroidi: pareggia
  SnapKV su sei finestre di sviluppo a 2.0605 bit, ma perde sull'holdout.
- Budget statico differenziato per sei gruppi di layer: a uguale capacità il
  profilo uniforme resta migliore o indistinguibile.

## Cosa bisogna ancora fare

### P0 — rendere le affermazioni scientificamente valide

1. **Aggiornare README.** Il paper è stato riscritto e compilato; `README.md`
   contiene ancora claim e numeri precedenti alle correzioni.
2. **Completare gli assi mancanti.** Il paper separa già capacità residente,
   bit letti/query, PPL e KL. Mancano retrieval standard, memoria effettiva,
   picco VRAM e tok/s end-to-end.
3. **Baseline ufficiali.** Quando possibile eseguire codice/release originali,
   non soltanto port locali. In alternativa documentare riga per riga le
   differenze e fissare versioni/commit.
4. **Rigenerare i risultati con la semantica Quest corretta.** Il codice ora
   separa capacità residente e lettura, ma tutti i JSON prodotti prima della
   correzione vanno rigenerati o etichettati come stale.
5. **Più dati.** Gli intervalli paired esistono ora
   (`benchmarks/paired_stats.py`) e PG-19 è stato esteso alle 11 finestre
   preregistrate valide, cioè all'intera lista disponibile. L'esito è che le
   differenze contro KIVI-4 nella fascia 5 bit sono **sotto la risoluzione del
   campione**: 693 target appaiati danno un CI95% di circa ±0.9% sul rapporto
   di PPL, mentre i gap in gioco sono 0.1--0.7%. Servono quindi altri corpora,
   non altre finestre dello stesso test PG-19, che è esaurito.
6. **Retrieval serio.** Eseguire RULER (multi-key/needle) e LongBench. La passkey
   custom non basta e sui modelli piccoli Full-KV fallisce.

### P1 — chiudere Quest e validare fuori da WikiText-2

Il punto graded+residuo esatto chiude il confronto locale con KIVI-4 su
WikiText-2 3B/8B, mentre RD-V2 domina la fascia bassa e il 30B. Su PG-19 esteso
a 11 libri il confronto graded+RD contro KIVI-4 è invece **statisticamente
indeciso**: la differenza è dentro il CI95%. Questo riformula la priorità.

Il problema non è più "il punto 5-bit fallisce fuori distribuzione", ma che
tutta la fascia 5 bit vive dentro il rumore di misura: PPL a parità di
capacità non è un asse abbastanza discriminante per chiudere il goal. Gli assi
che finora hanno mostrato separazioni ampie e riproducibili sono la **KL della
memoria** (0.0024 contro 0.0039 sull'8B, 0.0030 contro 0.0041 sul 3B) e la
**capacità/traffico**, dove i margini sono percentuali a due cifre, non
frazioni di punto percentuale. Il ramo compressed-Quest resta interessante
proprio perché il suo vantaggio è sugli assi separabili (9.85 contro 16.25 bit
residenti, 1.96 contro 2.32 letti), non sulla PPL.

Gate fisso, Born, cumulante e modi spettrali hanno fallito almeno un holdout e
non sono più la direzione primaria.

Direzioni ancora sensate:

- selezione loss-aware ma target-free (per esempio una piccola policy calibrata
  offline su più corpora) per decidere quota/livello del residuo per layer/head;
- budget per layer/head anziché identico ovunque, calibrato senza target futuri;
- calibrazione offline minima del decay per famiglia di modello, con molti
  segmenti validation e test esplicito bloccato;
- obiettivo di allocazione legato alla loss downstream, ma senza usare target
  futuri;
- estendere il protocollo incrementale a SnapKV, KIVI-2, RD-V2 e 30B;
- rigenerare i confronti WikiText-2 3B/8B con `token_nll` e passarli da
  `paired_stats.py`: i gap di 0.008 e 0.002 PPL oggi riportati come primari non
  hanno ancora un intervallo e vanno considerati non verificati;
- aggiungere un terzo corpus PPL esterno, dato che PG-19 test è ora esaurito e
  non offre altre finestre preregistrate;
- dichiarare in anticipo la dimensione del campione necessaria per l'effetto
  che si vuole misurare, invece di misurare e poi interpretare margini sotto la
  risoluzione;
- RULER/LongBench per impedire che una riduzione di PPL ottenuta eliminando
  contesto venga scambiata per migliore memoria.

### P2 — sistema reale

1. Definire un `past_key_values` packed che non mantenga la copia BF16.
2. Collegare il kernel fused al percorso attention di Transformers.
3. Supportare append incrementale, blocco parziale, GQA condivisa e demozione
   monotona senza ricostruire la cache.
4. Misurare output equivalente al riferimento PyTorch su sequenze complete.
5. Misurare picco VRAM e tok/s reali, prefill e decode separati.
6. Confrontare con FlashAttention/PagedAttention e con i kernel ufficiali delle
   baseline, non con SDPA densa soltanto.

Il runner intermedio ora misura tutte le KV-head di un layer in un'unica
chiamata e moltiplica la latenza soltanto per i layer seriali. Il test CUDA usa
quattro head con query diverse e coincide con il riferimento PyTorch entro
`1e-4` relativo. A contesto 16k e budget allocatore B=2:

| Modello | KV-head | cache packed | cache BF16 | compressione | attn Geodesia | attn densa | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 2 | 0.0815 GiB | 0.5625 GiB | 6.90x | 3.17 ms/token | 11.90 ms/token | 3.75x |
| Qwen3-8B | 8 | 0.3257 GiB | 2.2500 GiB | 6.91x | 9.93 ms/token | 12.44 ms/token | 1.25x |

Queste sono stime attention-only: misurano un layer con tutte le KV-head,
includono dequantizzazione e moltiplicano per 36 layer. La testa
rappresentativa viene replicata e il riferimento è un `bmm` denso locale, non
FlashAttention/PagedAttention. Inoltre il formato misurato è il Geodesia graded
classico da 2.316 bit/valore effettivi, non l'ibrido RD-V2 da 2.061. File:

- `results/qwen3b_system_metrics_head_batched.json`
- `results/qwen8b_system_metrics_head_batched.json`

La misura definitiva resta l'integrazione nel decode reale; non usare questi
numeri come tok/s end-to-end.

Finché P2 non è completato, non usare “end-to-end”, “throughput superiore” o
“VRAM reale ridotta” nelle conclusioni.

## Ordine operativo consigliato

1. Fatto il 2026-07-29: PG-19 esteso alle 11 finestre preregistrate valide con
   intervalli paired. Il seguito è rigenerare i confronti WikiText-2 con
   `token_nll` e aggiungere un corpus nuovo, non altre finestre PG-19; non fare
   tuning sulle finestre test WikiText-2 o PG-19 già osservate.
2. Eseguire RULER/LongBench e memory-quality sugli stessi punti RD-V2 e
   graded+residuo 5-bit.
3. Calibrare una policy per layer/head su un corpus nuovo e trasferirla senza
   retuning al test WikiText-2 già bloccato.
4. Integrare il formato packed/sparse nel `past_key_values`, prima per il
   graded puro e poi per compressed-Quest, misurando gli assi end-to-end.
5. Confrontare il kernel con FlashAttention/PagedAttention e con le
   implementazioni ufficiali dei baseline.
6. Aggiornare `README.md`; il paper è già riallineato, ma va rigenerato di
   nuovo se cambiano i risultati primari.

## Artefatti da trattare con cautela

- `results/qwen30b_a3b_q8_16k_transfer_pilot.json` è un JSON parziale/non valido
  prodotto prima della correzione del serializer. Non usarlo; i valori del run
  sono riportati sopra e il run a quattro finestre lo sostituisce.
- I vecchi `qwen*_production.json` e le figure in
  `paper/figures_geodesia/` non incorporano tutte le correzioni descritte qui.
  Il nuovo `paper/geodesia_kv.tex` non usa quelle figure.
- I JSON prodotti prima del 2026-07-29 non hanno `ppl_windows[].token_nll` e
  non possono essere passati a `paired_stats.py`: da soli non permettono di
  distinguere un vantaggio reale dal rumore. Tutti i margini sotto l'1% di PPL
  riportati in questo documento e provenienti da quei file vanno considerati
  non verificati finché il run non è rigenerato.
- I JSON senza `eval_mode: incremental` trattano il chunk teacher-forced come
  regione corrente. Non confrontare fra loro risultati prodotti con
  `eval_tokens` diversi e non usarli come prova finale contro KIVI.
- Tutti i JSON creati prima della separazione `resident_bits`/`read_bits`
  riportano erroneamente Quest a circa 2.25 bit/valore residenti. Quel numero è
  una vecchia stima del traffico letto per query che omette anche la pagina
  corrente: il valore corretto è circa 2.31 e la capacità è circa 16.25.
- I file `smoke_*_1k.json` servono soltanto al debug; i budget sono distorti
  dall'overhead dei blocchi esatti.
- `/tmp/geodesia-hf-30b` e `/tmp/geodesia-gptq-venv` possono essere rimossi dal
  sistema o puliti al reboot. Non cancellarli prima di avere finito i run 30B:
  ricostruire l'ambiente richiede compilare i kernel e riscaricare circa 30 GiB.
