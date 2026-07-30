"""Test di regressione per Geodesia-KV.

Il bug piu' costoso di questa linea di lavoro e' stato un oracolo NON causale
(`eager_attention_forward` con attention_mask=None non maschera nulla), che ha
invalidato due tabelle intere prima di essere trovato. I test 4 e 5 esistono
esattamente per impedire che ricompaia.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from geodesia_kv.policies import (Policy, Report, blockify, box_bound, hadamard,
                               quant_grouped, spectral_factors)
from benchmarks.budget_sweep import parse_layer_budget_profile
from benchmarks.paired_stats import cluster_bootstrap
from benchmarks.vs_sota import run_ppl


def causal_reference(q, k, v, q_offset):
    T, Q = k.shape[0], q.shape[0]
    kpos = torch.arange(T)
    qpos = q_offset + torch.arange(Q)
    s = (q @ k.T) * (q.shape[-1] ** -0.5)
    s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    return torch.softmax(s, -1) @ v


class _FakeCache:
    def __init__(self):
        self.tokens = []


class _CausalFakeModel(torch.nn.Module):
    """Modello minimo: chunk e decode producono gli stessi logits causali."""

    def __init__(self, vocab=13):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.vocab = vocab
        self.call_lengths = []

    def get_decoder(self):
        return self

    def set_attn_implementation(self, implementation):
        self.implementation = implementation

    def forward(self, ids, past_key_values=None, use_cache=True):
        self.call_lengths.append(ids.shape[1])
        cache = past_key_values if past_key_values is not None else _FakeCache()
        cache.tokens.extend(ids.reshape(-1).tolist())
        logits = torch.full(
            (*ids.shape, self.vocab), -4.0, device=ids.device)
        predicted = (ids + 1).remainder(self.vocab)
        logits.scatter_(-1, predicted[..., None], 4.0)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_incremental_ppl_matches_chunked_for_causal_full_model():
    """Il nuovo runner cambia il batching, non i target o la NLL."""
    ids = torch.arange(24).remainder(13)[None]
    chunked_model = _CausalFakeModel()
    incremental_model = _CausalFakeModel()
    chunked = run_ppl(
        chunked_model, None, Policy(name="full"), ids,
        eval_tokens=6, incremental=False)
    incremental = run_ppl(
        incremental_model, None, Policy(name="full"), ids,
        eval_tokens=6, incremental=True)
    assert abs(chunked - incremental) < 1e-7
    assert chunked_model.call_lengths == [18, 6]
    assert incremental_model.call_lengths == [18, 1, 1, 1, 1, 1]


def test_token_nll_is_aligned_and_reproduces_the_aggregate_ppl():
    """Le NLL per token devono essere gli stessi target in chunked e Q=1.

    Gli intervalli paired assumono che due policy valutate sulla stessa
    finestra producano liste allineate token per token e che la loro media sia
    esattamente il logaritmo della PPL riportata.
    """
    ids = torch.arange(24).remainder(13)[None]
    chunked_ppl, chunked_nll = run_ppl(
        _CausalFakeModel(), None, Policy(name="full"), ids,
        eval_tokens=6, incremental=False, return_token_nll=True)
    inc_ppl, inc_nll = run_ppl(
        _CausalFakeModel(), None, Policy(name="full"), ids,
        eval_tokens=6, incremental=True, return_token_nll=True)
    assert len(chunked_nll) == len(inc_nll) == 5
    for a, b in zip(chunked_nll, inc_nll):
        assert abs(a - b) < 1e-6
    for ppl, nll in ((chunked_ppl, chunked_nll), (inc_ppl, inc_nll)):
        assert abs(math.log(ppl) - sum(nll) / len(nll)) < 1e-6


def test_paired_cluster_bootstrap_covers_zero_only_when_undecided():
    """Il bootstrap deve ricampionare finestre, non token."""
    # Differenza costante e negativa: nessun ricampionamento puo' cambiarne il
    # segno, quindi l'intervallo deve escludere lo zero.
    decided = [[-0.5] * 7 for _ in range(6)]
    lo, hi = cluster_bootstrap(decided, n_boot=2000, seed=0, alpha=0.05)
    assert lo < -0.4 and hi < 0.0
    # Meta' finestre positive e meta' negative con la stessa ampiezza: la media
    # e' nulla e l'intervallo deve contenere lo zero.
    undecided = [[0.5] * 7 if i % 2 else [-0.5] * 7 for i in range(6)]
    lo, hi = cluster_bootstrap(undecided, n_boot=2000, seed=0, alpha=0.05)
    assert lo < 0.0 < hi


def test_quantized_cache_keeps_only_packed_state():
    """La cache viva non deve conservare la K/V densa dei blocchi chiusi.

    E' la proprieta' su cui poggia ogni affermazione di VRAM: se il denso
    sopravvivesse, il risparmio misurato sarebbe illusorio.
    """
    from geodesia_kv.live_cache import QuantizedCache

    cfg = SimpleNamespace(num_hidden_layers=1)
    cache = QuantizedCache(cfg, bits=2, chunk=64, group=32, window=0, sinks=0)
    b, h, d, t = 1, 2, 32, 256
    k = torch.randn(b, h, t, d, dtype=torch.bfloat16)
    v = torch.randn(b, h, t, d, dtype=torch.bfloat16)
    keys, vals = cache.update(k, v, 0)

    assert keys.shape == (b, h, t, d) and vals.shape == (b, h, t, d)
    assert cache.layers[0].get_seq_length() == t
    # Tutti i token sono chiusi: la coda densa deve essere vuota.
    assert cache.layers[0].n_closed == t
    assert cache.layers[0].tail_k.shape[-2] == 0
    # Il residente a 2 bit deve stare molto sotto il denso bf16 equivalente.
    dense = 2 * b * h * t * d * 2
    assert cache.resident_bytes() < dense / 3
    # La ricostruzione approssima l'originale: non e' rumore.
    err = (keys.float() - k.float()).abs().mean()
    assert err < k.float().abs().mean()


def test_quantized_cache_matches_incremental_append():
    """Aggiungere token uno alla volta deve dare lo stesso stato di un blocco."""
    from geodesia_kv.live_cache import QuantizedCache

    cfg = SimpleNamespace(num_hidden_layers=1)
    b, h, d, t = 1, 1, 32, 128
    k = torch.randn(b, h, t, d, dtype=torch.bfloat16)
    v = torch.randn(b, h, t, d, dtype=torch.bfloat16)

    bulk = QuantizedCache(cfg, bits=4, chunk=32, group=32, window=0, sinks=0)
    kb, _ = bulk.update(k, v, 0)
    step = QuantizedCache(cfg, bits=4, chunk=32, group=32, window=0, sinks=0)
    for i in range(t):
        ks, _ = step.update(k[:, :, i:i + 1], v[:, :, i:i + 1], 0)
    assert step.layers[0].get_seq_length() == t
    assert torch.equal(kb, ks)


def test_streaming_is_constant_memory_and_quest_is_not_compressed():
    """Le due politiche vanno lette su assi diversi, e il test lo fissa.

    StreamingLLM non comprime: elimina, quindi la memoria non cresce ma il
    contesto ritenuto resta il budget. Quest conserva la K/V esatta piu' i
    sommari, quindi sull'asse capacita' costa PIU' di Full-KV.
    """
    from geodesia_kv.live_cache import QuestCache, StreamingCache

    cfg = SimpleNamespace(num_hidden_layers=1)
    b, h, d = 1, 2, 32
    stream = StreamingCache(cfg, budget=64, sinks=4)
    quest = QuestCache(cfg, page=32)
    sizes = []
    for _ in range(8):
        k = torch.randn(b, h, 32, d, dtype=torch.bfloat16)
        v = torch.randn(b, h, 32, d, dtype=torch.bfloat16)
        stream.update(k, v, 0)
        quest.update(k, v, 0)
        sizes.append(stream.resident_bytes())

    seen = 8 * 32
    assert stream.layers[0].get_seq_length() == seen      # posizioni viste
    assert stream.retained_tokens() == 64                 # token conservati
    assert sizes[-1] == sizes[-2]                         # memoria costante
    dense = 2 * b * h * seen * d * 2
    assert quest.resident_bytes() > dense                 # sommari in piu'


def test_hadamard_is_orthogonal_involution():
    x = torch.randn(9, 64)
    assert (hadamard(hadamard(x)) - x).abs().max() < 1e-5
    assert (hadamard(x).norm(dim=-1) - x.norm(dim=-1)).abs().max() < 1e-5
    q, k = torch.randn(5, 64), torch.randn(11, 64)
    assert ((hadamard(q) @ hadamard(k).T) - (q @ k.T)).abs().max() < 1e-4


def test_quant_grouped_respects_step_bound():
    x = torch.randn(6, 64, 32)
    for bits in (2, 4, 8):
        rec, step = quant_grouped(x, bits, 16, per_channel=False)
        assert ((x - rec).abs().amax(dim=(1, 2)) <= step / 2 + 1e-5).all()
        rec, step = quant_grouped(x, bits, 16, per_channel=True)
        assert ((x - rec).abs().amax(dim=1) <= step / 2 + 1e-5).all()


def test_box_bound_is_an_upper_bound():
    q, k = torch.randn(7, 32), torch.randn(256, 32)
    kb, _ = blockify(k, 64)
    bnd = box_bound(q, kb.amin(1), kb.amax(1), 32 ** -0.5)
    true = ((q @ k.T) * 32 ** -0.5).view(7, 4, 64).amax(-1)
    assert (bnd >= true - 1e-4).all()


def test_full_policy_matches_causal_reference():
    """REGRESSIONE: il percorso policy deve essere causale.

    Il bug storico faceva vedere alla riga "oracolo" i token futuri, rendendola
    artificiosamente migliore di qualunque metodo compresso.
    """
    torch.manual_seed(0)
    q, k, v = torch.randn(16, 32), torch.randn(128, 32), torch.randn(128, 32)
    pol = Policy(name="full")
    pol.report = Report()
    out = pol.attend(q, k, v, 32 ** -0.5, q_offset=128 - 16)
    ref = causal_reference(q, k, v, 128 - 16)
    assert (out - ref).abs().max() < 1e-5, "il percorso policy non e' causale"


def test_every_policy_is_causal():
    """Nessuna policy puo' guardare oltre la posizione della query servita.

    La compressione osserva le PRIME `obs` query del chunk (= fine del prompt).
    Si verifica quindi la query di indice `obs`: ne' la sua attenzione ne' la
    scelta di cosa comprimere possono dipendere da chiavi successive alla sua
    posizione. Questo test ha gia' scoperto una fuga reale in SnapKV, dove la
    selezione dei token da tenere usava le query successive.
    """
    torch.manual_seed(1)
    T, Q, D, obs = 512, 64, 32, 8
    k, v = torch.randn(T, D), torch.randn(T, D)
    q = torch.randn(Q, D)
    q_offset = T - Q
    j = obs                              # query verificata
    for kw in (dict(name="streaming", budget_tokens=64),
               dict(name="snapkv", budget_tokens=64),
               dict(name="quest", quest_pages=2),
               dict(name="kivi"),
               dict(name="geodesia", budget_bits=2.0, window=64, group=32,
                    query_keep_frac=0.25, query_cold_bias=6.0,
                    query_gate="hybrid"),
               dict(name="geodesia-ub", tau=0.2, centroid_frac=0.5)):
        pol = Policy(obs=obs, **kw)
        pol.report = Report()
        base = pol.attend(q, k, v, D ** -0.5, q_offset=q_offset)
        k2, v2 = k.clone(), v.clone()
        k2[q_offset + j + 1:] += 10.0    # solo chiavi FUTURE rispetto alla query j
        v2[q_offset + j + 1:] += 10.0
        pol.report = Report()
        alt = pol.attend(q, k2, v2, D ** -0.5, q_offset=q_offset)
        assert (base[j] - alt[j]).abs().max() < 1e-4, \
            f"{kw['name']} guarda nel futuro"


def test_certificate_is_never_violated():
    """Il certificato e' un bound dimostrato: zero violazioni, sempre."""
    torch.manual_seed(2)
    for trial in range(20):
        T, Q, D = 1024, 16, 32
        k, v = torch.randn(T, D) * (1 + trial % 3), torch.randn(T, D)
        q = torch.randn(Q, D)
        pol = Policy(name="geodesia-ub", tau=0.1 * (1 + trial % 5),
                     centroid_frac=0.1 * (trial % 8))
        pol.report = Report()
        pol.attend(q, k, v, D ** -0.5, q_offset=T - Q)
        assert pol.report.n_violations == 0, f"certificato violato al trial {trial}"
        assert pol.report.true_err <= pol.report.cert_bound + 1e-6


def test_geodesia_reports_less_memory_than_full():
    torch.manual_seed(3)
    T, Q, D = 2048, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    full = Policy(name="full")
    full.report = Report()
    full.attend(q, k, v, D ** -0.5, q_offset=T - Q)
    gw = Policy(name="geodesia-ub", tau=0.5, centroid_frac=0.7)
    gw.report = Report()
    gw.attend(q, k, v, D ** -0.5, q_offset=T - Q)
    assert full.report.bits_per_value == 16.0
    # soglia riferita all'oracolo, non a un valore assoluto: dopo l'aggancio dei
    # bit a {2,4,8,16} e il conteggio delle scale, il costo reale e' piu' alto di
    # quanto la contabilita' analitica lasciasse credere.
    assert gw.report.bits_per_value < 0.5 * full.report.bits_per_value


def test_frozen_allocation_never_changes():
    """Una decisione presa non deve piu' cambiare, mai.

    E' la proprieta' che rende il sistema implementabile: la rappresentazione
    compressa di un blocco si costruisce una volta e si riusa. Se la decisione
    cambiasse, andrebbe ricompresso a ogni passo - piu' caro dell'attenzione piena.
    """
    torch.manual_seed(7)
    T, D = 4096, 64
    k, v = torch.randn(T, D), torch.randn(T, D)
    pol = Policy(name="geodesia-ub", freeze=True, tau=0.5, group=64, min_bits=4.0,
                 centroid_frac=0.5, retire_after=512)
    hid = (0, 0, 0)
    snapshots = []
    for step in range(8):
        pol.report = Report()
        q = torch.randn(1, D) * (1 + step)      # query molto diverse fra loro
        n = T - 8 + step
        pol.attend(q, k[:n], v[:n], D ** -0.5, q_offset=n - 1, hid=hid)
        st = pol.state[hid]
        d = st["decided"].clone()
        snapshots.append((d, st["bits"].clone(), st["cent"].clone()))

    for i in range(1, len(snapshots)):
        prev_d, prev_b, prev_c = snapshots[i - 1]
        d, b, c = snapshots[i]
        m = prev_d[: d.numel()]
        assert bool((d[: m.numel()] | ~m).all()), "un blocco deciso e' tornato indeciso"
        assert torch.equal(b[: m.numel()][m], prev_b[: m.numel()][m]), \
            "i bit di un blocco gia' deciso sono cambiati"
        assert torch.equal(c[: m.numel()][m], prev_c[: m.numel()][m]), \
            "la scelta di centroide di un blocco gia' deciso e' cambiata"


def test_frozen_is_causal():
    """Anche congelata, l'allocazione non puo' dipendere da chiavi future."""
    torch.manual_seed(8)
    T, Q, D, obs = 512, 64, 32, 8
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    j = obs
    pol = Policy(name="geodesia-ub", freeze=True, tau=0.2, centroid_frac=0.5,
                 obs=obs, retire_after=64)
    pol.report = Report()
    base = pol.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    k2, v2 = k.clone(), v.clone()
    k2[T - Q + j + 1:] += 10.0
    v2[T - Q + j + 1:] += 10.0
    pol2 = Policy(name="geodesia-ub", freeze=True, tau=0.2, centroid_frac=0.5,
                  obs=obs, retire_after=64)
    pol2.report = Report()
    alt = pol2.attend(q, k2, v2, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert (base[j] - alt[j]).abs().max() < 1e-4, "l'allocazione congelata guarda nel futuro"


def test_geodesia_first_query_does_not_use_future_queries():
    """La decisione usata da q[0] non puo' dipendere da q[1:]."""
    torch.manual_seed(9)
    T, Q, D = 512, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    kw = dict(name="geodesia", budget_bits=3.0, obs=Q, block=64, group=32,
              window=64, sinks=4)
    p1 = Policy(**kw)
    p1.report = Report()
    base = p1.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))

    q2 = q.clone()
    q2[1:] += 10.0
    p2 = Policy(**kw)
    p2.report = Report()
    alt = p2.attend(q2, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert (base[0] - alt[0]).abs().max() < 1e-5, \
        "q[0] e' stata compressa usando query future"


def test_shared_kv_is_not_demoted_twice_in_one_attention_call():
    """Query-head GQA diverse devono leggere una singola cache condivisa."""
    torch.manual_seed(10)
    T, Q, D = 512, 4, 32
    k, v = torch.randn(T, D), torch.randn(T, D)
    pol = Policy(name="geodesia", budget_bits=3.0, block=64, group=32,
                 window=64, sinks=4)
    hid = (0, 0, 0)
    pol.attend(torch.randn(Q, D), k, v, D ** -0.5, q_offset=T - Q,
               hid=hid, allow_demote=True)
    levels = pol.state[hid]["level"].clone()
    krec = pol.state[hid]["krec"].clone()
    mass = pol.state[hid]["mass"].clone()
    pol.attend(torch.randn(Q, D), k, v, D ** -0.5, q_offset=T - Q,
               hid=hid, allow_demote=False)
    assert torch.equal(pol.state[hid]["level"], levels)
    assert torch.equal(pol.state[hid]["krec"], krec)
    assert torch.equal(pol.state[hid]["mass"], mass)


def test_query_gate_is_certified_against_dense_attention():
    """Il bias sparse deliberato deve entrare nel bound, non cambiare l'oracolo."""
    torch.manual_seed(11)
    T, Q, D = 1024, 16, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    pol = Policy(
        name="geodesia", budget_bits=2.0, block=64, group=32,
        window=64, sinks=4, query_keep_frac=0.25,
        query_cold_bias=6.0, query_gate="hybrid", validate=True)
    pol.report = Report()
    pol.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert pol.report.n_violations == 0
    assert pol.report.true_err <= pol.report.cert_bound + 1e-6
    assert 0.20 <= pol.report.gate_keep_fraction <= 0.35


def test_sparse_box_uses_exact_summaries_and_reports_sparse_reads():
    """Compressed-Quest conserva summary esatti e legge solo pagine scelte."""
    torch.manual_seed(111)
    T, D = 1024, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(1, D)
    pol = Policy(
        name="geodesia", budget_bits=8.0, block=64, group=32,
        window=64, sinks=4, query_keep_frac=0.125,
        query_cold_bias=1.0, query_gate="box_sparse", validate=False)
    hid = (0, 0, 0)
    out = pol.attend(
        q, k, v, D ** -0.5, q_offset=T - 1, hid=hid)
    st = pol.state[hid]
    kb, _ = blockify(k, 64)
    # L'ultimo blocco è corrente e non entra nel ranking; tutti i summary
    # chiusi devono comunque coincidere con gli estremi della K originale.
    assert torch.equal(
        st["box_kmin"][:-1], kb[:-1].amin(1).to(torch.float16))
    assert torch.equal(
        st["box_kmax"][:-1], kb[:-1].amax(1).to(torch.float16))
    assert torch.isfinite(out).all()
    assert pol.report.read_bits_per_value < pol.report.bits_per_value
    assert pol.report.gate_keep_fraction < 0.25


def test_disabled_query_gate_preserves_legacy_output():
    """Una frazione 1 o un bias 0 devono essere esattamente no-op."""
    torch.manual_seed(12)
    T, Q, D = 512, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    base = dict(name="geodesia", budget_bits=3.0, block=64, group=32,
                window=64, sinks=4)
    p1 = Policy(**base, query_keep_frac=1.0, query_cold_bias=10.0)
    p2 = Policy(**base, query_keep_frac=0.1, query_cold_bias=0.0)
    o1 = p1.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    o2 = p2.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert torch.equal(o1, o2)
    assert p1.report.gate_keep_fraction == 1.0
    assert p2.report.gate_keep_fraction == 1.0


def test_prompt_protected_blocks_are_static_and_exact():
    """La quota protetta nasce dal prompt e non può cambiare o essere demolita."""
    torch.manual_seed(13)
    T, D = 1024, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(1, D)
    hid = (0, 0, 0)
    pol = Policy(
        name="geodesia", budget_bits=2.0, block=64, group=32,
        window=64, sinks=4, protected_frac=0.125,
        query_keep_frac=0.125, query_cold_bias=6.0,
        query_gate="protected")
    pol.observations[hid] = {
        "block_mass": torch.linspace(0.0, 1.0, T // 64)}
    pol.attend(q, k, v, D ** -0.5, q_offset=T - 1, hid=hid)
    protected = pol.state[hid]["protected"].clone()
    levels = pol.state[hid]["level"].clone()
    assert int(protected.sum()) > 0
    assert bool((levels[protected] == 16).all())

    pol.attend(q * -10, k, v, D ** -0.5, q_offset=T - 1, hid=hid)
    assert torch.equal(pol.state[hid]["protected"], protected)
    assert bool((pol.state[hid]["level"][protected] == 16).all())
    assert pol.report.n_violations == 0


def test_token_residual_is_static_counted_and_certified():
    """Il residuo Snap-like è una copia esatta contata, non una promozione gratis."""
    torch.manual_seed(14)
    T, Q, D = 1024, 16, 32
    q_offset = T - Q
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    hid = (0, 0, 0)
    common = dict(
        name="geodesia", budget_bits=0.75, block=64, group=32,
        window=64, sinks=4, query_keep_frac=0.125,
        query_cold_bias=6.0, query_gate="token")

    base = Policy(**common)
    base.observations[hid] = {
        "block_mass": torch.ones(q_offset // 64) / (q_offset // 64),
        "token_score": torch.linspace(0.0, 1.0, q_offset)}
    base.attend(q, k, v, D ** -0.5, q_offset=q_offset, hid=hid)
    base_bits = base.report.resident_bits

    pol = Policy(**common, token_protected_frac=0.1)
    pol.observations[hid] = {
        "block_mass": torch.ones(q_offset // 64) / (q_offset // 64),
        "token_score": torch.linspace(0.0, 1.0, q_offset)}
    pol.attend(q, k, v, D ** -0.5, q_offset=q_offset, hid=hid)
    protected = pol.state[hid]["token_protected"][:T].clone()
    assert int(protected.sum()) == max(
        pol.snap_window, int(0.1 * q_offset + 0.999999))
    expected_extra = float(protected.sum()) * 2.0 * D * 16.0
    assert abs((pol.report.resident_bits - base_bits) - expected_extra) < 1e-3
    assert pol.report.n_violations == 0

    pol.attend(q * -3, k, v, D ** -0.5, q_offset=q_offset, hid=hid)
    assert torch.equal(pol.state[hid]["token_protected"][:T], protected)


def test_quest_separates_resident_capacity_from_read_traffic():
    """Quest conserva Full-KV e legge top-k più la pagina corrente."""
    torch.manual_seed(15)
    T, Q, D = 1024, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    pol = Policy(name="quest", quest_pages=2, block=64)
    pol.attend(q, k, v, D ** -0.5, q_offset=T - Q)
    assert abs(pol.report.bits_per_value - 16.25) < 1e-6
    # Due pagine selezionate (2 bpv), la pagina corrente (1 bpv) e i summary
    # min/max (0.25 bpv).
    assert abs(pol.report.read_bits_per_value - 3.25) < 1e-6


def test_layer_budget_profile_is_explicit_and_changes_capacity():
    profile = parse_layer_budget_profile("1.5x1,5x1", 2)
    assert profile == (1.5, 5.0)

    torch.manual_seed(16)
    T, D = 2048, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(1, D)
    pol = Policy(
        name="geodesia", budget_bits=2.0, layer_budget_bits=profile,
        block=64, group=32, window=64, sinks=0, validate=False)
    pol.attend(q, k, v, D ** -0.5, q_offset=T - 1, hid=(0, 0, 0))
    low_bits = pol.report.resident_bits
    pol.report = Report()
    pol.attend(q, k, v, D ** -0.5, q_offset=T - 1, hid=(1, 0, 0))
    high_bits = pol.report.resident_bits
    assert high_bits > low_bits


def test_spectral_state_preserves_rank_one_key_value_response():
    """Il modo congiunto deve conservare C_vk q, che un centroide perde."""
    torch.manual_seed(17)
    nb, p, d = 3, 64, 16
    amplitude = torch.linspace(-2.0, 2.0, p)[None, :, None]
    key_dir = torch.randn(nb, 1, d)
    value_dir = torch.randn(nb, 1, d)
    kb = amplitude * key_dir
    vb = 1.7 * amplitude * value_dir
    u, w, _ = spectral_factors(kb, vb, rank=1)
    q = torch.randn(nb, d)
    exact = torch.einsum(
        "npd,npk,nk->nd",
        vb - vb.mean(1, keepdim=True),
        kb - kb.mean(1, keepdim=True),
        q) / p
    a = torch.einsum("nd,nrd->nr", q, u)
    approx = torch.einsum("nr,nrd->nd", a, w)
    rel = (exact - approx).norm() / exact.norm().clamp_min(1e-8)
    assert rel < 2e-3


def test_cumulant_correction_is_counted_and_certified():
    torch.manual_seed(20)
    T, Q, D = 1024, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    common = dict(
        name="geodesia", budget_bits=2.0, block=64, group=32,
        window=64, sinks=0, validate=True)
    base = Policy(**common)
    base.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))

    pol = Policy(**common, cumulant_strength=1.0)
    pol.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert abs(
        pol.report.resident_bits - base.report.resident_bits
        - (T // 64) * 16.0) < 1e-4
    assert bool((pol.state[(0, 0, 0)]["logit_variance"] != 0).any())
    assert pol.report.n_violations == 0
    assert pol.report.true_err <= pol.report.cert_bound + 1e-6


def test_born_gate_uses_mass_threshold_and_is_certified():
    torch.manual_seed(21)
    T, Q, D = 1024, 8, 32
    k, v, q = torch.randn(T, D), torch.randn(T, D), torch.randn(Q, D)
    pol = Policy(
        name="geodesia", budget_bits=2.0, block=64, group=32,
        window=64, sinks=0, query_gate="born", query_keep_frac=0.9,
        query_cold_bias=6.0, validate=True)
    pol.attend(q, k, v, D ** -0.5, q_offset=T - Q, hid=(0, 0, 0))
    assert 0.0 < pol.report.gate_keep_fraction <= 1.0
    # Il numero selezionato dipende dalla distribuzione della singola query:
    # non deve essere forzato alla stessa top-k per tutte.
    assert pol.report.gate_selected_blocks > 0
    assert pol.report.n_violations == 0
    assert pol.report.true_err <= pol.report.cert_bound + 1e-6


def test_rate_distortion_token_saliency_uses_value_residual():
    score = torch.ones(64)
    value_residual = torch.ones(64)
    value_residual[62] = 100.0
    pol = Policy(
        token_value_power=1.0, snap_window=1, snap_kernel=1)
    keep = pol._prompt_token_keep(
        score, prompt_t=64, total_t=64, fraction=2 / 64,
        value_residual=value_residual)
    assert int(keep.sum()) == 2
    assert bool(keep[62])
    assert bool(keep[63])  # finestra recente sempre esatta
