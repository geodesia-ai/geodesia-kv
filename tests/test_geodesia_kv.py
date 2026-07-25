"""Test di regressione per Geodesia-KV.

Il bug piu' costoso di questa linea di lavoro e' stato un oracolo NON causale
(`eager_attention_forward` con attention_mask=None non maschera nulla), che ha
invalidato due tabelle intere prima di essere trovato. I test 4 e 5 esistono
esattamente per impedire che ricompaia.
"""

from __future__ import annotations

import torch

from geodesia_kv.policies import (Policy, Report, blockify, box_bound, hadamard,
                               quant_grouped)


def causal_reference(q, k, v, q_offset):
    T, Q = k.shape[0], q.shape[0]
    kpos = torch.arange(T)
    qpos = q_offset + torch.arange(Q)
    s = (q @ k.T) * (q.shape[-1] ** -0.5)
    s = s.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
    return torch.softmax(s, -1) @ v


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
