# Risultati

Solo run prodotti con l'harness dopo la correzione di tutti i bug di causalita' e
di contabilita' (vedi README principale). I run precedenti sono stati scartati,
non archiviati: contenevano fughe dal futuro e un conteggio dei bit ottimistico.

| File | Contenuto |
|---|---|
| `qwen3b_16k_production.json` | **Risultato principale**: Qwen2.5-3B, 16k, produzione contro i quattro baseline |
| `qwen3b_16k_budget_sweep.json` | Sweep della fascia 3-5 bit sul 3B |
| `qwen0.5b_16k_official_baselines.json` | Qwen2.5-0.5B con i baseline portati dal sorgente ufficiale |
| `qwen0.5b_16k_memory_quality.json` | Metriche di qualita' della memoria: ritenzione per profondita', KL, contesto effettivo |
