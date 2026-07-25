"""Geodesia-KV e baseline SOTA, tutte nello stesso percorso di attenzione.

Ogni policy riceve (q, k, v) POST-RoPE dentro l'attention di un modello congelato e
restituisce l'output di attenzione piu' un rendiconto onesto dei bit residenti.
Nessun training, nessuna modifica ai pesi: si registra una implementazione di
attenzione in `ALL_ATTENTION_FUNCTIONS` e si passa al modello.

Policy implementate
-------------------
FullKV        oracolo, 16 bit/valore
StreamingLLM  attention sink + finestra scorrevole, il resto EVITTO
SnapKV        top-B token per massa osservata sulle ultime query, il resto EVITTO
Quest         selezione per pagina con bound box (min/max per canale), per query
GeodesiaKV       FUSIONE: non evitta nulla, alloca PRECISIONE per blocco + certificato

Il certificato di GeodesiaKV e' rigoroso:
    ||o - o_hat|| <= 2*||s - s_hat||_inf * max_j ||v_j||     (perturbazione softmax)
                     + max_b delta_b                          (errore sui valori)
con ||s - s_hat||_inf <= scaling * ||q|| * max_j ||k_j - k_hat_j||, e ogni termine
maggiorato dai passi di quantizzazione, non misurato a posteriori.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------- #
# quantizzazione per blocco (simula la precisione RESIDENTE, non un kernel)
# --------------------------------------------------------------------------- #

def quant_per_channel(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (..., P, D) blocco di P token. Quantizza per canale sul blocco (stile KIVI-K).

    Ritorna (x_ricostruito, passo) con errore per elemento <= passo/2.
    """
    if bits >= 16:
        return x, torch.zeros(x.shape[:-2] + (1, x.shape[-1]), device=x.device,
                              dtype=x.dtype)
    lo = x.amin(dim=-2, keepdim=True)
    hi = x.amax(dim=-2, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    q = ((x - lo) / step).round().clamp_(0, 2 ** bits - 1)
    return q * step + lo, step


def quant_per_token(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizza per token (stile KIVI-V): x (..., P, D)."""
    if bits >= 16:
        return x, torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    q = ((x - lo) / step).round().clamp_(0, 2 ** bits - 1)
    return q * step + lo, step


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Trasformata di Walsh-Hadamard normalizzata sull'ultima dimensione.

    H e' ortogonale, quindi q.k = (Hq).(Hk) ESATTAMENTE: applicarla a query e key
    non cambia un singolo punteggio di attenzione. Serve a redistribuire l'energia
    sui canali: le key hanno canali outlier con range enorme che, in quantizzazione
    per canale, costringono tutti gli altri canali a sprecare i loro bit. Dopo la
    rotazione i range sono uniformi e ogni bit lavora allo stesso regime
    (incoherence processing, come QuIP#/QuaRot).
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Hadamard richiede dimensione potenza di 2, non {d}"
    y = x.clone()
    h = 1
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).view(*x.shape[:-1], d)
        h *= 2
    return y / (d ** 0.5)


def quant_grouped(x: torch.Tensor, bits: int, group: int, per_channel: bool
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizza (nb, P, D) in gruppi di `group` token lungo l'asse temporale.

    Disaccoppia la granularita' di quantizzazione da quella dell'indice: il range
    su 16 token e' molto piu' stretto che su 64, e a 2 bit il passo e' range/3.
    Ritorna (ricostruito, passo per gruppo) con errore per elemento <= passo/2.
    """
    nb, pp, d = x.shape
    if per_channel:
        # KIVI-K: per ogni canale, gruppi di `group` token consecutivi
        g = min(group, pp)
        xs = x.view(nb, pp // g, g, d)
        lo, hi = xs.amin(2, keepdim=True), xs.amax(2, keepdim=True)
    else:
        # KIVI-V: per ogni token, gruppi di `group` CANALI. Usare tutti i D
        # canali insieme (come facevo prima) e' molto piu' grossolano e
        # penalizzava sia KIVI sia Geodesia-KV.
        g = min(group, d)
        xs = x.view(nb, pp, d // g, g)
        lo, hi = xs.amin(3, keepdim=True), xs.amax(3, keepdim=True)
    if bits >= 16:      # esatto: passo nullo, stessa forma dei rami quantizzati
        zero = torch.zeros(nb, d, device=x.device, dtype=x.dtype) if per_channel \
            else torch.zeros(nb, device=x.device, dtype=x.dtype)
        return x, zero
    step = (hi - lo).clamp_min(1e-8) / (2 ** bits - 1)
    q = ((xs - lo) / step).round().clamp_(0, 2 ** bits - 1)
    rec = (q * step + lo).view(nb, pp, d)
    # passo massimo per blocco, per il bound rigoroso
    smax = step.amax(dim=(1, 2)) if per_channel else step.amax(dim=(1, 2, 3))
    return rec, smax


def box_bound(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor,
              scaling: float) -> torch.Tensor:
    """Upper bound RIGOROSO su max_{k in blocco} q.k.

    q: (Q, D), kmin/kmax: (nb, D) -> (Q, nb).
    max sulla scatola che contiene il blocco: sum_c max(q_c*kmin_c, q_c*kmax_c).
    """
    return torch.maximum(q[:, None, :] * kmin[None], q[:, None, :] * kmax[None]
                         ).sum(-1) * scaling


def blockify(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    """(T, D) -> (nb, block, D) con padding ripetendo l'ultima riga."""
    t, d = x.shape
    nb = (t + block - 1) // block
    pad = nb * block - t
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, -1)], 0)
    return x.view(nb, block, d), pad


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #

@dataclass
class Report:
    """Rendiconto per una singola chiamata di attenzione."""
    resident_bits: float = 0.0     # bit residenti totali per K+V
    total_values: int = 0          # valori (K+V) del contesto pieno
    cert_bound: float = 0.0        # bound certificato sull'errore relativo
    true_err: float = 0.0          # errore relativo reale (solo per validazione)
    n_certified: int = 0
    n_violations: int = 0

    def merge(self, o: "Report") -> None:
        self.resident_bits += o.resident_bits
        self.total_values += o.total_values
        self.cert_bound += o.cert_bound
        self.true_err += o.true_err
        self.n_certified += o.n_certified
        self.n_violations += o.n_violations

    @property
    def bits_per_value(self) -> float:
        return self.resident_bits / max(self.total_values, 1)


@dataclass
class Policy:
    name: str = "full"
    sinks: int = 4
    window: int = 128
    block: int = 64
    budget_tokens: int = 1024        # StreamingLLM / SnapKV
    quest_pages: int = 16            # Quest
    obs: int = 32                    # finestra di osservazione (query recenti)
    tiers: tuple = ((0.05, 16), (0.15, 8), (0.30, 4), (1.00, 2))  # (frazione cum, bit)
    mode: str = "cert"               # "cert" = errore fisso (canonico) | "mass" = deprecato
    tau: float = 0.05                # tetto su |s - s_hat| quando mode="cert"
    min_bits: float = 2.0
    rotate: bool = False             # rotazione Hadamard di q,k,v (esatta)
    group: int = 16                  # gruppo di quantizzazione (<= block)
    kv_bit_delta: int = 0            # bit in piu' alle key, in meno alle value
    centroid_frac: float = 0.0       # frazione di blocchi freddi -> centroide
    centroid_moment: bool = False    # correzione di Jensen sui blocchi a centroide
    validate: bool = True            # calcola l'attenzione piena per il certificato
    debias: bool = False             # rimuove il bias log-normale della quantizzazione
    freeze: bool = False             # decisione per blocco presa UNA volta e congelata
    budget_bits: float = 0.0         # bit/valore target (attiva la demozione)
    retire_after: int = 512          # token dopo i quali un blocco e' "ritirato"
    state: dict = field(default_factory=dict)   # (layer, head) -> decisioni
    snap_window: int = 64            # SnapKV ufficiale: window_size
    snap_kernel: int = 5             # SnapKV ufficiale: kernel_size
    kivi_k_bits: int = 2             # KIVI ufficiale: k_bits
    kivi_v_bits: int = 2             # KIVI ufficiale: v_bits
    kivi_group: int = 32             # KIVI ufficiale: group_size
    kivi_residual: int = 32          # KIVI ufficiale: residual_length
    report: Report = field(default_factory=Report)

    def reset_state(self) -> None:
        """Azzera lo stato della cache: da chiamare fra due prompt diversi.

        Senza questo, i blocchi decisi/demoliti per un prompt sopravvivono nel
        prompt successivo e la misura e' priva di senso.
        """
        self.state.clear()

    # ---------------------------------------------------------------- helpers
    def _observed_mass(self, q_obs: torch.Tensor, k: torch.Tensor,
                       scaling: float, nb: int, pad: int,
                       causal_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Massa d'attenzione per blocco stimata sulle query recenti (stile SnapKV).

        Segnale causale e gratuito: il modello ha gia' calcolato quelle query.
        """
        s = (q_obs @ k.T) * scaling                      # (obs, T)
        if causal_obs is not None:
            s = s.masked_fill(~causal_obs, float("-inf"))
        w = torch.softmax(s.float(), -1)
        if pad:
            w = torch.nn.functional.pad(w, (0, pad))
        return w.view(w.shape[0], nb, self.block).sum(-1).mean(0)   # (nb,)

    # ------------------------------------------------------------------ apply
    def _frozen_alloc(self, hid, nb: int, mass: torch.Tensor, rng: torch.Tensor,
                      qnorm: float, exact: torch.Tensor, scaling: float,
                      q_offset: int, dev):
        """Allocazione CONGELATA: ogni blocco decide una volta sola e per sempre.

        La versione adattiva ricalcolava bit e centroidi a ogni query: costa piu'
        dell'attenzione piena, quindi non e' implementabile, ed e' ottimistica
        perche' la compressione si adatta alla query che poi la usa. Qui:

        - la massa di attenzione per blocco viene ACCUMULATA nel tempo;
        - ||q|| e' una media mobile, non il valore della query corrente;
        - un blocco viene deciso quando si e' "ritirato" (e' uscito da
          `retire_after` token fa) e da quel momento la decisione non cambia mai;
        - la ricostruzione e' deterministica data la decisione, quindi
          ricalcolarla qui equivale a tenerla in cache: la qualita' misurata e'
          esattamente quella del sistema che la memorizza.
        """
        st = self.state.setdefault(hid, {"bits": None, "cent": None,
                                         "mass": None, "qnorm": qnorm,
                                         "decided": None})
        if st["bits"] is None or st["bits"].numel() < nb:
            def grow(old, fill, dtype):
                new = torch.full((nb,), fill, device=dev, dtype=dtype)
                if old is not None:
                    new[: old.numel()] = old
                return new
            st["bits"] = grow(st["bits"], 0.0, torch.float32)
            st["cent"] = grow(st["cent"], False, torch.bool)
            st["mass"] = grow(st["mass"], 0.0, torch.float32)
            st["decided"] = grow(st["decided"], False, torch.bool)
        st["qnorm"] = 0.9 * st["qnorm"] + 0.1 * qnorm      # media mobile
        st["mass"][:nb] = 0.9 * st["mass"][:nb] + 0.1 * mass

        # blocchi ritirati: interamente piu' vecchi di `retire_after`
        bend = (torch.arange(nb, device=dev) + 1) * self.block
        retired = (bend <= max(0, q_offset - self.retire_after)) & ~exact
        todo = retired & ~st["decided"][:nb]
        if todo.any():
            m = st["mass"][:nb]
            ref = m[retired]
            thr = (torch.quantile(ref, self.centroid_frac)
                   if self.centroid_frac > 0 and ref.numel() > 0
                   else torch.tensor(-1.0, device=dev))
            eps_t = (self.tau / (nb * m.clamp_min(1e-6))).clamp(1e-4, 1e3)
            need = scaling * st["qnorm"] * rng / (2.0 * eps_t)
            b = torch.ceil(torch.log2(need.clamp_min(1e-9) + 1.0)).clamp(2.0, 16.0)
            b = torch.where(b <= 2, torch.full_like(b, 2.0),
                torch.where(b <= 4, torch.full_like(b, 4.0),
                torch.where(b <= 8, torch.full_like(b, 8.0),
                            torch.full_like(b, 16.0)))).clamp_min(self.min_bits)
            st["bits"][:nb] = torch.where(todo, b, st["bits"][:nb])
            st["cent"][:nb] = torch.where(todo, m <= thr, st["cent"][:nb])
            st["decided"][:nb] |= todo

        bits = torch.where(st["decided"][:nb], st["bits"][:nb],
                           torch.full((nb,), 16.0, device=dev))
        cent = st["cent"][:nb] & ~exact
        return bits, cent

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               scaling: float, q_offset: int = 0, hid=None) -> torch.Tensor:
        """q (Q,D), k/v (T,D) di UNA testa. Ritorna (Q,D) e aggiorna il report.

        `q_offset` e' la posizione assoluta della prima query: la query i puo'
        attendere solo le chiavi 0..q_offset+i. Senza questa maschera il calcolo
        della perplexity vedrebbe i token futuri.
        """
        T, D = k.shape
        Q = q.shape[0]
        rep = self.report
        rep.total_values += 2 * T * D
        kpos = torch.arange(T, device=k.device)
        qpos = q_offset + torch.arange(Q, device=k.device)
        causal = kpos[None, :] <= qpos[:, None]          # (Q, T)

        if self.name in ("full", "fullpath"):
            rep.resident_bits += 2 * T * D * 16
            s = ((q @ k.T) * scaling).masked_fill(~causal, float("-inf"))
            return torch.softmax(s, -1) @ v

        # ---------------- metodi a EVICTION: producono un sottoinsieme di token
        if self.name in ("streaming", "snapkv"):
            if self.name == "streaming":
                keep = torch.cat([torch.arange(min(self.sinks, T), device=k.device),
                                  torch.arange(max(T - self.budget_tokens, 0), T,
                                               device=k.device)]).unique()
            else:
                # SnapKV, fedele a FasterDecoding/SnapKV (snapkv_utils.py):
                #   - osservazione = ultime `window_size` query del PROMPT
                #     (qui: le prime del chunk, che vi sono adiacenti)
                #   - SOMMA (non media) dei pesi sulle query di osservazione
                #   - le ultime `window_size` chiavi sono ESCLUSE dal ranking e
                #     tenute sempre
                #   - avg_pool1d con kernel_size=5, padding=2, stride=1
                #   - si tengono `max_capacity_prompt - window_size` indici
                w = self.snap_window
                q_obs = q[: w]
                sc = ((q_obs @ k.T) * scaling).masked_fill(
                    ~causal[: w], float("-inf"))
                score = torch.softmax(sc.float(), -1).sum(0)
                tail = max(T - w, 0)
                score = score[:tail]
                score = torch.nn.functional.avg_pool1d(
                    score[None, None], kernel_size=self.snap_kernel, stride=1,
                    padding=self.snap_kernel // 2)[0, 0]
                n_keep = max(0, min(self.budget_tokens - w, tail))
                keep = (score.topk(n_keep).indices if n_keep > 0
                        else torch.zeros(0, dtype=torch.long, device=k.device))
                keep = torch.cat([keep, torch.arange(tail, T, device=k.device)
                                  ]).unique()
            # I token del chunk corrente sono quelli "appena generati": nel
            # deployment reale vengono APPESI alla cache e mai evitti. Senza
            # questo, una query a meta' chunk perde il proprio contesto immediato
            # e il metodo statico crolla per un artefatto del protocollo.
            keep = torch.cat([keep, torch.arange(q_offset, T, device=k.device)]).unique()
            kk, vv = k[keep], v[keep]
            rep.resident_bits += 2 * len(keep) * D * 16
            s = ((q @ kk.T) * scaling).masked_fill(
                keep[None, :] > qpos[:, None], float("-inf"))
            return torch.softmax(s, -1) @ vv

        kb, pad = blockify(k, self.block)
        vb, _ = blockify(v, self.block)
        nb = kb.shape[0]
        kmin, kmax = kb.amin(1), kb.amax(1)

        # ---------------- Quest: seleziona pagine per query, il resto non e' letto
        if self.name == "quest":
            bnd = box_bound(q, kmin, kmax, scaling)          # (Q, nb)
            # Il riassunto (min/max) di un blocco che CONTIENE la posizione della
            # query e' calcolato anche su chiavi future: usarlo per il ranking
            # sarebbe una fuga. Quest nel deployment reale tiene sempre attiva la
            # pagina corrente e classifica solo le pagine gia' chiuse.
            bstart = torch.arange(nb, device=k.device) * self.block
            bend = bstart + self.block
            partial = (bstart[None, :] <= qpos[:, None]) & (bend[None, :] > qpos[:, None])
            past = bend[None, :] <= qpos[:, None]
            bnd = bnd.masked_fill(~past, float("-inf"))
            npages = min(self.quest_pages, nb)
            sel = bnd.topk(npages, dim=-1).indices           # (Q, npages)
            mask = torch.zeros(Q, nb, dtype=torch.bool, device=k.device)
            mask.scatter_(1, sel, True)
            mask &= past
            mask |= partial                                  # pagina corrente
            full = mask.repeat_interleave(self.block, 1)[:, :T] & causal
            s = (q @ k.T) * scaling
            s = s.masked_fill(~full, float("-inf"))
            rep.resident_bits += 2 * npages * self.block * D * 16 + nb * 2 * D * 16
            return torch.softmax(s, -1) @ v

        # ---------------- KIVI (jy-yuan/KIVI, ICML 2024) ---------------------
        # Key per-canale con gruppi lungo l'asse token, value per-token con
        # gruppi lungo D, piu' `residual_length` token recenti in fp16.
        # Nessuna allocazione: TUTTI i blocchi allo stesso numero di bit.
        if self.name == "kivi":
            # Il KIVI vero quantizza INCREMENTALMENTE: un gruppo si forma solo
            # da token gia' completati, quindi non contiene mai token futuri
            # rispetto a una query che lo usa. In una implementazione batch il
            # buffer residuo deve percio' coprire almeno l'intero chunk corrente,
            # altrimenti un gruppo a cavallo della query userebbe statistiche
            # calcolate anche su chiavi future.
            n_res = min(max(self.kivi_residual, T - q_offset), T)
            k_old, v_old = k[: T - n_res], v[: T - n_res]
            out_bits = 2 * n_res * D * 16.0
            if k_old.shape[0] > 0:
                kb2, _ = blockify(k_old, self.block)
                vb2, _ = blockify(v_old, self.block)
                nb2 = kb2.shape[0]
                kq, _ = quant_grouped(kb2, self.kivi_k_bits, self.kivi_group, True)
                vq, _ = quant_grouped(vb2, self.kivi_v_bits, self.kivi_group, False)
                kk = torch.cat([kq.reshape(-1, D)[: k_old.shape[0]], k[T - n_res:]])
                vv = torch.cat([vq.reshape(-1, D)[: v_old.shape[0]], v[T - n_res:]])
                ngroup = max(1, self.block // self.kivi_group)
                out_bits += nb2 * ((self.kivi_k_bits + self.kivi_v_bits)
                                   * self.block * D
                                   + ngroup * D * 32.0
                                   + self.block * max(1, D // self.kivi_group) * 32.0)
            else:
                kk, vv = k, v
            rep.resident_bits += out_bits
            sc = ((q @ kk.T) * scaling).masked_fill(~causal, float("-inf"))
            return torch.softmax(sc, -1) @ vv

        # ---------------- Geodesia-KV demote: il design di PRODUZIONE ----------
        # Ogni blocco nasce esatto e viene demolito di livello quando la memoria
        # supera il budget. La demozione parte dalla rappresentazione CORRENTE:
        # l'originale bf16 non serve mai piu' - ed e' proprio questo a rendere
        # impossibile qualunque allocazione che voglia RISALIRE di precisione.
        # Costo di comprimere per gradi invece che direttamente: 0.0-0.1%.
        if self.name == "geodesia":
            st = self.state.setdefault(hid, {})
            kb0, pad0 = blockify(k, self.block)
            vb0, _ = blockify(v, self.block)
            nb0 = kb0.shape[0]
            if "level" not in st or st["level"].numel() < nb0:
                def _grow(name, fill, dtype):
                    new_t = torch.full((nb0,), fill, device=k.device, dtype=dtype)
                    if name in st:
                        new_t[: st[name].numel()] = st[name]
                    st[name] = new_t
                old_n = st["level"].numel() if "level" in st else 0
                _grow("level", 16.0, torch.float32)
                _grow("mass", 0.0, torch.float32)
                _grow("ek", 0.0, torch.float32)
                _grow("ev", 0.0, torch.float32)
                krec = torch.empty_like(kb0)
                vrec = torch.empty_like(vb0)
                if old_n:
                    krec[:old_n] = st["krec"][:old_n]
                    vrec[:old_n] = st["vrec"][:old_n]
                krec[old_n:] = kb0[old_n:]      # i blocchi nuovi entrano esatti
                vrec[old_n:] = vb0[old_n:]
                st["krec"], st["vrec"] = krec, vrec
            # i blocchi ancora in scrittura restano esatti e non demolibili
            st["krec"][-1] = kb0[-1]
            st["vrec"][-1] = vb0[-1]
            hot = torch.zeros(nb0, dtype=torch.bool, device=k.device)
            hot[: max(1, (self.sinks + self.block - 1) // self.block)] = True
            hot[max(0, q_offset // self.block - 1):] = True
            hot[-max(1, (self.window + self.block - 1) // self.block):] = True

            mass = self._observed_mass(q[: self.obs], k, scaling, nb0, pad0,
                                       causal[: self.obs])
            st["mass"] = 0.9 * st["mass"] + 0.1 * mass

            # --- politica: livello target per blocco, dal piu' freddo in giu' ---
            lv = st["level"]
            # tabella dei bit per livello, calcolata UNA volta: la versione
            # precedente era una list-comprehension su 256 blocchi a ogni chiamata
            _lut = torch.tensor([_level_bits(L, self.block, D, self.group)
                                 for L in LEVELS], device=k.device)
            _l2i = {L: i for i, L in enumerate(LEVELS)}

            def bits_of(L):
                idx = torch.searchsorted(
                    torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], device=k.device),
                    L.contiguous())
                return _lut[(len(LEVELS) - 1) - idx.clamp(0, len(LEVELS) - 1)]
            total_vals = 2.0 * nb0 * self.block * D
            if self.budget_bits > 0:
                # ALLOCAZIONE OTTIMA (rilassamento lagrangiano).
                # La versione ingenua - prendi il blocco piu' freddo e portalo al
                # pavimento, poi passa al successivo - ricrea la distribuzione
                # bimodale che e' esattamente il difetto gia' diagnosticato: la
                # distorsione e' convessa nei bit, quindi conviene scendere di UN
                # livello su molti blocchi invece che crollare su pochi.
                # Qui si minimizza  sum_b massa_b * errore_b(L_b)  soggetto al
                # budget, risolvendo  min_L  danno + lambda*bit  per ogni blocco e
                # cercando lambda per bisezione. E' la soluzione esatta del
                # rilassamento ed e' completamente vettorizzata.
                rngb = (st["krec"].amax(1) - st["krec"].amin(1)).norm(dim=-1)
                radb = (st["krec"] - st["krec"].mean(1, keepdim=True)
                        ).norm(dim=-1).amax(-1)
                errs, bitl = [], []
                for L in LEVELS:
                    if L >= 16:
                        e = torch.zeros_like(rngb)
                    elif L == 1:
                        e = radb
                    else:
                        e = rngb / (2.0 * (2 ** L - 1))
                    errs.append(e)
                    bitl.append(_level_bits(L, self.block, D, self.group))
                E = torch.stack(errs, 1)                       # (nb, 5)
                B = torch.tensor(bitl, device=k.device)[None]  # (1, 5)
                dmg = st["mass"][:, None] * E
                # un blocco non puo' RISALIRE: livelli piu' alti del corrente
                # sono vietati, e i blocchi caldi restano dove sono
                lvl_idx = torch.tensor([LEVELS.index(int(x)) for x in lv],
                                       device=k.device)
                allowed = (torch.arange(len(LEVELS), device=k.device)[None]
                           >= lvl_idx[:, None])
                allowed &= ~hot[:, None] | (
                    torch.arange(len(LEVELS), device=k.device)[None]
                    == lvl_idx[:, None])
                big = torch.finfo(torch.float32).max / 4
                # Soluzione ESATTA del rilassamento con UN ordinamento invece di
                # 48 passi di bisezione (ognuno con i suoi lanci di kernel):
                # ogni possibile demozione e' un "passo" con costo marginale
                # d_danno/d_bit; si prendono i passi piu' convenienti finche' il
                # budget e' soddisfatto. E' la stessa soluzione, senza il loop.
                nlv = len(LEVELS)
                step_gain = (B[0, :-1] - B[0, 1:])[None].expand(nb0, -1)   # bit risparmiati
                step_cost = (dmg[:, 1:] - dmg[:, :-1])                     # danno aggiunto
                ratio = step_cost / step_gain.clamp_min(1e-9)
                # un passo e' disponibile solo se parte da un livello raggiungibile
                avail = (torch.arange(nlv - 1, device=k.device)[None]
                         >= lvl_idx[:, None]) & ~hot[:, None]
                ratio = torch.where(avail, ratio, torch.full_like(ratio, big))
                # Un blocco puo' saltare al livello d solo passando per tutti i
                # livelli intermedi: i passi presi devono formare un PREFISSO.
                # Prendere il massimo dei passi selezionati (senza il prefisso)
                # demoliva senza contarne il costo -> budget 3.0 dava 1.55 bit.
                # Si cerca quindi la soglia lambda con una bisezione sui valori
                # ORDINATI dei rapporti, applicando a ogni tentativo la regola
                # del prefisso: ~11 passi di sole operazioni vettoriali.
                cand = torch.sort(ratio.reshape(-1)).values
                target = self.budget_bits * total_vals
                lo_i, hi_i = 0, cand.numel() - 1

                def _bits_at(lam):
                    ok = (ratio <= lam) & avail
                    pref = torch.cumprod(ok.to(torch.int8), dim=1).bool()
                    d = lvl_idx + pref.sum(1)
                    d = torch.clamp(d, max=nlv - 1)
                    return B[0][d].sum(), d

                tgt_idx = lvl_idx
                for _ in range(int(torch.log2(torch.tensor(
                        float(cand.numel()))).item()) + 2):
                    mid = (lo_i + hi_i) // 2
                    tot, d = _bits_at(cand[mid])
                    if tot > target:
                        lo_i = mid + 1
                    else:
                        hi_i = mid
                        tgt_idx = d
                    if lo_i >= hi_i:
                        break
                _, tgt_idx = _bits_at(cand[min(hi_i, cand.numel() - 1)])
                tgt = torch.tensor([LEVELS[i] for i in tgt_idx.tolist()],
                                   device=k.device, dtype=torch.float32)
                if True:
                    # BATCH per livello: un solo kernel per livello invece di uno
                    # per blocco. Misurato: 54.0 ms -> 0.17 ms su 256 blocchi.
                    changed = tgt != lv
                    for L in LEVELS[1:]:
                        m = changed & (tgt == L)
                        if not m.any():
                            continue
                        idx = m.nonzero(as_tuple=True)[0]
                        kold, vold = st["krec"][idx], st["vrec"][idx]
                        if L == 1:
                            kn = kold.mean(1, keepdim=True).expand_as(kold).contiguous()
                            vn = vold.mean(1, keepdim=True).expand_as(vold).contiguous()
                        else:
                            kn, _ = quant_grouped(kold, L, self.group, True)
                            vn, _ = quant_grouped(vold, L, self.group, False)
                        st["ek"][idx] += (kold - kn).norm(dim=-1).amax(-1)
                        st["ev"][idx] += (vold - vn).norm(dim=-1).amax(-1)
                        st["krec"][idx], st["vrec"][idx] = kn, vn
                        lv[idx] = float(L)

            kk = st["krec"].reshape(-1, D)[:T]
            vv = st["vrec"].reshape(-1, D)[:T]
            rep.resident_bits += float(bits_of(lv).sum())
            w = torch.softmax(((q @ kk.T) * scaling).masked_fill(
                ~causal, float("-inf")), -1)
            out = w @ vv
            with torch.no_grad():
                eps_s = scaling * float(q.norm(dim=-1).max()) * st["ek"]
                wb = torch.nn.functional.pad(w, (0, pad0)).view(
                    Q, nb0, self.block).sum(-1)
                term_v = (wb * st["ev"][None]).sum(-1)
                logw = wb.clamp_min(1e-30).log()
                lp = torch.logsumexp(logw + eps_s[None], -1)
                lm = torch.logsumexp(logw - eps_s[None], -1)
                l1 = (torch.exp((lp - lm).clamp(max=20.0))
                      - torch.exp((lm - lp).clamp(min=-20.0))).clamp(0.0, 2.0)
                bound = term_v + l1 * float(v.norm(dim=-1).max())
                rep.n_certified += 1
                if self.validate:
                    ref = torch.softmax(((q @ k.T) * scaling).masked_fill(
                        ~causal, float("-inf")), -1) @ v
                    true = (out - ref).norm(dim=-1)
                    den = ref.norm(dim=-1).clamp_min(1e-9)
                    rep.cert_bound += float((bound / den).mean())
                    rep.true_err += float((true / den).mean())
                    rep.n_violations += int((true > bound + 1e-4).sum())
            return out

        # ---------------- Geodesia-KV: nessuna eviction, precisione allocata
        if self.name == "geodesia-ub":
            if self.rotate:
                # H ortogonale e simmetrica: q.k = (Hq).(Hk) esattamente, e
                # sum_j a_j (H v_j) = H (sum_j a_j v_j) -> basta riapplicare H
                # all'output per tornare indietro. Nessuna approssimazione.
                q, k, v = hadamard(q), hadamard(k), hadamard(v)
                kb, pad = blockify(k, self.block)
                vb, _ = blockify(v, self.block)
            qnorm = float(q.norm(dim=-1).max())

            # blocchi sempre esatti: sink iniziali e finestra recente
            exact = torch.zeros(nb, dtype=torch.bool, device=k.device)
            exact[: max(1, (self.sinks + self.block - 1) // self.block)] = True
            exact[-max(1, (self.window + self.block - 1) // self.block):] = True
            exact[q_offset // self.block:] = True      # token del chunk corrente

            rng_blk = (kb.amax(1) - kb.amin(1)).norm(dim=-1)          # (nb,)
            if self.freeze:
                mass = self._observed_mass(q[: self.obs], k, scaling, nb, pad,
                                           causal[: self.obs])
                bits, centroid = self._frozen_alloc(
                    hid, nb, mass, rng_blk, qnorm, exact, scaling, q_offset,
                    k.device)
                bits = torch.where(exact, torch.full_like(bits, 16.0), bits)
            elif self.mode == "cert":
                # --- allocazione: la MASSA da' la priorita', il bound la garanzia --
                # vogliamo sum_b mass_b * eps_b <= tau  =>  eps_b = tau/(nb*mass_b):
                # un blocco a massa bassa puo' permettersi errore grande, uno a massa
                # alta no. Poi:
                #   eps_b(bits) = scaling*||q||*||range_b||_2 / (2*(2^bits - 1))
                #   => bits_b = ceil(log2(1 + scaling*||q||*||range_b|| / (2*eps_b)))
                mass = self._observed_mass(q[: self.obs], k, scaling, nb, pad,
                                           causal[: self.obs])
                eps_t = (self.tau / (nb * mass.clamp_min(1e-6))).clamp(1e-4, 1e3)
                rng = (kb.amax(1) - kb.amin(1)).norm(dim=-1)          # (nb,)
                need = scaling * qnorm * rng / (2.0 * eps_t)
                bits = torch.ceil(torch.log2(need.clamp_min(1e-9) + 1.0))
                bits = bits.clamp(self.min_bits, 16.0)
                # BUG CORRETTO: ceil(log2(...)) produce anche 3, 5, 6, 7 bit, che
                # nessun formato impacchetta senza sprechi (`build_packed` li
                # promuoveva a 8). Contarli come 3 o 5 rendeva OTTIMISTICO il
                # bit/valore riportato. Si agganciano alle larghezze reali.
                bits = torch.where(bits <= 2, torch.full_like(bits, 2.0),
                      torch.where(bits <= 4, torch.full_like(bits, 4.0),
                      torch.where(bits <= 8, torch.full_like(bits, 8.0),
                                  torch.full_like(bits, 16.0))))
                # A parita' di bit/valore, 4 bit con gruppi grossolani domina
                # 2 bit con gruppi fini (misurato: err 0.026 @ 4.50 bit contro
                # 0.056 @ 4.00). La distorsione e' convessa nei bit, quindi una
                # media fatta di 2-e-16 e' peggio di un uniforme allo stesso
                # costo: il pavimento va alzato invece che scendere a 2 bit.
                bits = bits.clamp_min(self.min_bits)
            else:
                # --- allocazione per massa osservata (budget fisso) --------- #
                mass = self._observed_mass(q[: self.obs], k, scaling, nb, pad,
                                           causal[: self.obs])
                order = mass.argsort(descending=True)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(nb, device=k.device)
                frac = (rank.float() + 1) / nb
                bits = torch.full((nb,), float(self.tiers[-1][1]), device=k.device)
                prev = 0.0
                for cum, b in self.tiers:
                    bits = torch.where((frac > prev) & (frac <= cum),
                                       torch.full_like(bits, float(b)), bits)
                    prev = cum
            if not self.freeze:
                bits = torch.where(exact, torch.full_like(bits, 16.0), bits)

            # --- il POZZO: i blocchi piu' freddi collassano nel loro centroide.
            # Non e' eviction: il blocco resta nel softmax col suo peso (64 copie
            # della chiave media), ma senza dettaglio. Costo: 1 chiave + 1 valore
            # per blocco = 0.5 bit/valore con block=64 -> e' cosi' che si scende
            # SOTTO i 2 bit/valore senza buttare via contesto.
            if self.freeze:
                pass                       # gia' deciso e congelato sopra
            elif True:
                centroid = torch.zeros(nb, dtype=torch.bool, device=k.device)
            if not self.freeze and self.centroid_frac > 0:
                ncold = int(self.centroid_frac * nb)
                if ncold > 0:
                    cold = mass.argsort()[:ncold]
                    centroid[cold] = True
                    centroid &= ~exact

            kh = torch.empty_like(kb)
            vh = torch.empty_like(vb)
            eps_s = torch.zeros(nb, device=k.device)     # bound su |s - s_hat|
            dv = torch.zeros(nb, device=k.device)        # bound su ||v - v_hat||
            res_bits = torch.zeros(nb, device=k.device)

            cent_bias = torch.zeros(nb, device=k.device)
            if centroid.any():
                kc = kb[centroid].mean(1, keepdim=True)
                vc = vb[centroid].mean(1, keepdim=True)
                kh[centroid] = kc.expand(-1, self.block, -1)
                vh[centroid] = vc.expand(-1, self.block, -1)
                # bound esatti: deviazione massima dal centroide dentro il blocco
                eps_s[centroid] = scaling * qnorm * (kb[centroid] - kc).norm(
                    dim=-1).amax(-1)
                dv[centroid] = (vb[centroid] - vc).norm(dim=-1).amax(-1)
                res_bits[centroid] = 2 * D * 16.0
                if self.centroid_moment:
                    # Sostituire n chiavi con n copie della media SOTTOSTIMA
                    # sistematicamente il peso del blocco: per convessita' di exp
                    #     sum_j exp(s_j) >= n * exp(s_medio)      (Jensen)
                    # con scarto relativo ~ Var_j(s_j)/2. Un blocco freddo diventa
                    # cosi' piu' freddo di quanto sia davvero - ed e' esattamente
                    # il caso peggiore se contiene l'informazione cercata.
                    # Con la varianza diagonale del blocco (D valori fp16, +0.25
                    # bit/valore) si recupera il termine del secondo ordine:
                    #     delta_b = 1/2 * sum_c q_c^2 * sigma^2_bc   (scalato)
                    var = kb[centroid].var(1, unbiased=False)          # (nc, D)
                    q2 = (q ** 2).amax(0)                              # (D,)
                    cent_bias[centroid] = 0.5 * (var * q2[None]).sum(-1) * scaling ** 2
                    res_bits[centroid] += D * 16.0
                    # il bias entra nel bound come termine additivo: resta rigoroso
                    eps_s[centroid] = eps_s[centroid] + cent_bias[centroid].abs()

            todo = ~centroid
            for b in sorted({int(x) for x in bits[todo].tolist()}):
                m = todo & (bits == b)
                if not m.any():
                    continue
                # le key pesano di piu': il loro errore viene ESPONENZIATO dal
                # softmax, quello delle value solo mediato -> asimmetria K/V.
                bk = min(16, b + self.kv_bit_delta)
                bv = max(2, b - self.kv_bit_delta)
                kq, kstep = quant_grouped(kb[m], bk, self.group, True)
                vq, vstep = quant_grouped(vb[m], bv, self.group, False)
                kh[m], vh[m] = kq, vq
                eps_s[m] = scaling * qnorm * (kstep / 2).norm(dim=-1)
                dv[m] = (vstep / 2 * (D ** 0.5)).amax(dim=-1) if vstep.dim() > 1 \
                    else vstep / 2 * (D ** 0.5)
                # I DATI non bastano: vanno contate le scale, e a 2 bit pesano
                # quanto i dati stessi. K e' per-canale -> 2 valori fp16 per
                # (gruppo, canale) = 32*D bit per gruppo. V e' per-token ->
                # 2 valori fp16 per token = 32 bit per token.
                ngroup = max(1, self.block // self.group)
                scale_k = ngroup * D * 32.0 if bk < 16 else 0.0
                scale_v = (self.block * max(1, D // self.group) * 32.0
                           if bv < 16 else 0.0)
                res_bits[m] = ((bk + bv) * self.block * D) + scale_k + scale_v

            kk = kh.reshape(-1, D)[:T]
            vv = vh.reshape(-1, D)[:T]
            rep.resident_bits += float(res_bits.sum())

            scores = (q @ kk.T) * scaling
            if self.centroid_moment and centroid.any():
                scores = scores + cent_bias.repeat_interleave(
                    self.block)[:T][None, :]
            if self.debias:
                # Se il punteggio ha errore con varianza sigma^2, allora
                #     E[exp(s + err)] = exp(s) * exp(sigma^2 / 2)
                # cioe' la quantizzazione GONFIA sistematicamente la massa dei
                # blocchi, e tanto piu' quanto piu' sono grossolani: i blocchi
                # freddi rubano massa a quelli caldi. Con passo uniforme
                # sigma^2 = passo^2/12 per elemento, quindi il termine e'
                # calcolabile dai passi gia' noti e si sottrae. E' il duale
                # della correzione di Jensen sui centroidi, con segno opposto.
                sig2 = (eps_s ** 2) / 3.0        # (2*eps)^2/12, eps = passo/2
                debias_term = 0.5 * sig2
                scores = scores - debias_term.repeat_interleave(
                    self.block)[:T][None, :]
                # il bias sposta s_hat, quindi |s - s_hat| non e' piu' limitato
                # dal solo eps: va sommato al bound, altrimenti il certificato
                # viene violato (osservate 1391 violazioni prima di questa riga)
                eps_s = eps_s + debias_term
            w = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
            out = w @ vv

            # ---- certificato rigoroso, senza guardare la verita' -----------
            # o - o_hat = sum_j a_hat_j (v_j - v_hat_j) + sum_j (a_j - a_hat_j) v_j
            # Termine 1: <= sum_b a_hat_b * dv_b                     (pesato, stretto)
            # Termine 2: da |s_j - s_hat_j| <= eps_j segue
            #   a_j / a_hat_j in [e^{-eps_j}/M+, e^{+eps_j}/M-]
            #   con M- = sum_b a_hat_b e^{-eps_b}, M+ = sum_b a_hat_b e^{+eps_b}
            # quindi ||a - a_hat||_1 <= sum_b a_hat_b * max(e^{eps_b}/M- - 1,
            #                                               1 - e^{-eps_b}/M+)
            # -> anche il termine softmax e' PESATO DALLA MASSA (non piu' il max
            #    globale): un blocco a 2 bit con massa ~0 non gonfia piu' il bound.
            with torch.no_grad():
                wb = torch.nn.functional.pad(w, (0, pad)).view(Q, nb, self.block
                                                               ).sum(-1)   # (Q,nb)
                term_v = (wb * dv[None]).sum(-1)                # (Q,)
                # forma chiusa stabile: sommando i due rami di max(.,.) si ottiene
                #   ||a - a_hat||_1 <= M+/M- - M-/M+     con M± = sum_b w_b e^{±eps_b}
                # calcolata in log-space; e non puo' superare 2 (differenza di due
                # distribuzioni di probabilita').
                logw = wb.clamp_min(1e-30).log()
                lp = torch.logsumexp(logw + eps_s[None], -1)
                lm = torch.logsumexp(logw - eps_s[None], -1)
                l1 = (torch.exp((lp - lm).clamp(max=20.0))
                      - torch.exp((lm - lp).clamp(min=-20.0))).clamp(0.0, 2.0)
                vmax = float(v.norm(dim=-1).max())
                bound = term_v + l1 * vmax
                rep.n_certified += 1
                if self.validate:
                    # attenzione piena SOLO per validare il certificato: raddoppia
                    # il costo, quindi non e' sul percorso di produzione
                    ref = torch.softmax(((q @ k.T) * scaling).masked_fill(
                        ~causal, float("-inf")), -1) @ v
                    true = (out - ref).norm(dim=-1)
                    denom = ref.norm(dim=-1).clamp_min(1e-9)
                    rep.cert_bound += float((bound / denom).mean())
                    rep.true_err += float((true / denom).mean())
                    rep.n_violations += int((true > bound + 1e-4).sum())
            # H e' un'involuzione (H@H = I): riapplicarla riporta l'output nello
            # spazio originale. La norma dell'errore e' invariante per rotazione,
            # quindi il certificato calcolato sopra resta valido tale e quale.
            return hadamard(out) if self.rotate else out

        raise ValueError(f"policy sconosciuta: {self.name}")


# --------------------------------------------------------------------------- #
# aggancio a transformers
# --------------------------------------------------------------------------- #

ACTIVE: Policy | None = None
MIN_CTX = 512          # sotto questa lunghezza si usa l'attenzione esatta


def policy_attention(module, query, key, value, attention_mask, scaling,
                     dropout=0.0, **kwargs):
    """Implementazione di attenzione registrabile in ALL_ATTENTION_FUNCTIONS."""
    pol = ACTIVE
    T = key.shape[-2]
    if pol is None:
        # BUG STORICO (corretto): qui si delegava a `eager_attention_forward`, che
        # con attention_mask=None NON applica alcuna maschera causale -> la riga
        # "oracolo" del benchmark vedeva i token futuri e sembrava imbattibile.
        # Ogni percorso passa ora da `attend`, che costruisce la causalita' dalle
        # posizioni assolute. Verificato contro un forward unico causale.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        return ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value,
                                               attention_mask, scaling,
                                               dropout=dropout, **kwargs)

    B, H, Q, D = query.shape
    Hkv = key.shape[1]
    reps = H // Hkv
    q_offset = T - Q          # le query sono le ultime Q posizioni del contesto
    outs = []
    qf = query.float()
    kf = key.float()
    vf = value.float()
    for b in range(B):
        # Lo stato e' indicizzato per KV-head, non per query head: la cache e'
        # memorizzata una volta sola e condivisa dalle `reps` query head del
        # gruppo GQA. Tenere una compressione diversa per query head non sarebbe
        # implementabile - e falsava sia il costo sia la qualita'.
        heads = [pol.attend(qf[b, h], kf[b, h // reps], vf[b, h // reps], scaling,
                            q_offset,
                            (getattr(module, "layer_idx", 0), b, h // reps))
                 for h in range(H)]
        outs.append(torch.stack(heads))
    out = torch.stack(outs).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


# --------------------------------------------------------------------------- #
# vector quantization per blocco: la generalizzazione del centroide
# --------------------------------------------------------------------------- #

def vq_blocks(kb: torch.Tensor, vb: torch.Tensor, c: int, iters: int = 3):
    """k-means batchato su ogni blocco: `c` centroidi invece di 1.

    Il centroide singolo e' il caso c=1. Aumentando c si copre con continuita'
    tutto l'intervallo da 0.25 bit/valore (c=1) a esatto (c=P), con UN solo
    meccanismo: niente piu' due tier (quantizzato / centroide) che competono.

    Si raggruppa sulle KEY e si riusa lo stesso assegnamento per le VALUE:
    token con key simili ricevono peso di attenzione simile, quindi mediarne le
    value dentro il cluster e' l'approssimazione giusta - e costa un solo array
    di indici invece di due.

    kb, vb: (nb, P, D). Ritorna (k_ric, v_ric, raggio_k, raggio_v, bit_per_blocco).
    """
    nb, P, D = kb.shape
    c = max(1, min(c, P))
    if c >= P:
        z = torch.zeros(nb, device=kb.device)
        return kb, vb, z, z, torch.full((nb,), 2.0 * P * D * 16.0, device=kb.device)

    # init deterministico: token equispaziati nel blocco. I token adiacenti sono
    # molto correlati, quindi lo strided init e' gia' vicino a k-means++.
    idx = torch.linspace(0, P - 1, c, device=kb.device).long()
    cent = kb[:, idx].clone()                                   # (nb, c, D)
    assign = None
    for _ in range(iters):
        assign = torch.cdist(kb, cent).argmin(-1)               # (nb, P)
        oh = torch.nn.functional.one_hot(assign, c).to(kb.dtype)  # (nb, P, c)
        cnt = oh.sum(1).clamp_min(1.0)                          # (nb, c)
        cent = (oh.transpose(1, 2) @ kb) / cnt[..., None]
    assign = torch.cdist(kb, cent).argmin(-1)
    oh = torch.nn.functional.one_hot(assign, c).to(kb.dtype)
    cnt = oh.sum(1).clamp_min(1.0)
    cent = (oh.transpose(1, 2) @ kb) / cnt[..., None]
    vcent = (oh.transpose(1, 2) @ vb) / cnt[..., None]          # media delle value

    gather = assign[..., None].expand(-1, -1, D)
    k_rec = torch.gather(cent, 1, gather)
    v_rec = torch.gather(vcent, 1, gather)
    rad_k = (kb - k_rec).norm(dim=-1).amax(-1)                  # bound esatto
    rad_v = (vb - v_rec).norm(dim=-1).amax(-1)
    import math
    bits = 2.0 * c * D * 16.0 + P * math.log2(c) if c > 1 else 2.0 * c * D * 16.0
    return k_rec, v_rec, rad_k, rad_v, torch.full((nb,), bits, device=kb.device)


# --------------------------------------------------------------------------- #
# demozione monotona guidata dal budget: il design di produzione
# --------------------------------------------------------------------------- #

LEVELS = [16, 8, 4, 2, 1]      # 1 = centroide (un solo vettore per blocco)


def _level_bits(level: int, block: int, D: int, group: int) -> float:
    """Bit residenti di un blocco a un dato livello, scale incluse."""
    if level >= 16:
        return 2.0 * block * D * 16.0
    if level == 1:                                   # centroide
        return 2.0 * D * 16.0
    ngroup = max(1, block // group)
    return (2.0 * level * block * D
            + ngroup * D * 32.0                      # scale K, per canale
            + block * max(1, D // group) * 32.0)     # scale V, per token


def demote_block(k_rec: torch.Tensor, v_rec: torch.Tensor, level: int,
                 group: int) -> tuple:
    """Demolisce UN livello partendo dalla rappresentazione CORRENTE.

    Non serve l'originale: e' esattamente il vincolo che rende impossibile
    l'allocazione adattiva verso l'alto. L'errore introdotto qui si somma a
    quello gia' presente, quindi il bound resta valido per additivita'.

    Ritorna (k_nuovo, v_nuovo, delta_k, delta_v) dove i delta sono gli scarti
    massimi introdotti da QUESTA demozione.
    """
    if level == 1:                                   # centroide
        kc = k_rec.mean(0, keepdim=True)
        vc = v_rec.mean(0, keepdim=True)
        dk = float((k_rec - kc).norm(dim=-1).max())
        dv = float((v_rec - vc).norm(dim=-1).max())
        return kc.expand_as(k_rec).contiguous(), vc.expand_as(v_rec).contiguous(), dk, dv
    kq, _ = quant_grouped(k_rec[None], level, group, True)
    vq, _ = quant_grouped(v_rec[None], level, group, False)
    kq, vq = kq[0], vq[0]
    dk = float((k_rec - kq).norm(dim=-1).max())
    dv = float((v_rec - vq).norm(dim=-1).max())
    return kq, vq, dk, dv
