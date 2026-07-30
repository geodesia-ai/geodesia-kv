"""Cache KV realmente quantizzata, collegata al modello vero.

Il runner di qualita' di questo progetto simula la compressione: quantizza e
dequantizza subito, tenendo comunque la K/V densa in memoria. Va bene per
misurare la qualita', ma non dimostra nulla sulla VRAM.

Qui invece lo stato residente **e' soltanto** quello quantizzato:

- i blocchi chiusi vivono come interi impacchettati (2/4/8 bit) piu' scale
  fp16 per gruppo, prodotti da ``quantize_packed``; la loro K/V densa viene
  liberata subito;
- resta esatta soltanto una coda recente di ``window`` token, come nel residuo
  corrente di KIVI, piu' i primi ``sinks`` token;
- ``update`` restituisce la rappresentazione ricostruita del solo layer
  corrente, che e' transitoria e viene liberata appena l'attenzione finisce.

Il picco di memoria e' quindi: stato quantizzato di TUTTI i layer, piu' la
ricostruzione densa di UN layer. E' il costo reale di un design che
dequantizza in lettura, non una stima.

Limite dichiarato: questo percorso implementa la quantizzazione uniforme
impacchettata, non l'allocatore graded con centroidi e residui esatti. Serve a
dimostrare la residenza della memoria, non a riprodurre i numeri di qualita'
della policy completa.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicLayer

from geodesia_kv.packed import PackedBlocks, pack_bits, quantize_packed


def _dequantize_to(blocks: PackedBlocks, dtype: torch.dtype) -> torch.Tensor:
    """Come ``PackedBlocks.dequantize`` ma senza passare da float32.

    La ricostruzione e' l'unico tensore denso che esiste durante l'attenzione:
    produrla direttamente in bf16 dimezza quel transitorio.
    """
    from geodesia_kv.packed import unpack_bits

    nb, p, d = blocks.shape
    g = min(blocks.group, p)
    n = nb * (p // g) * g * d
    q = unpack_bits(blocks.data.reshape(-1), blocks.bits, n)
    q = q.reshape(nb, p // g, g, d).to(dtype)
    out = q * blocks.step.to(dtype) + blocks.lo.to(dtype)
    return out.reshape(nb, p, d)


class QuantizedLayer(DynamicLayer):
    """Un layer la cui K/V chiusa esiste solo in forma impacchettata."""

    is_sliding = False

    def __init__(self, bits: int = 2, chunk: int = 512, group: int = 64,
                 window: int = 128, sinks: int = 4, config=None):
        super().__init__(config=config)
        if chunk % group:
            raise ValueError("chunk deve essere multiplo di group")
        self.bits, self.chunk, self.group = bits, chunk, group
        self.window, self.sinks = window, sinks

    def lazy_initialization(self, key_states: torch.Tensor,
                            value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch, self.heads = key_states.shape[0], key_states.shape[1]
        self.dim = key_states.shape[-1]
        self.closed: list[tuple[PackedBlocks, PackedBlocks]] = []
        self.n_closed = 0
        self.sink_k = self.sink_v = None
        self.tail_k = torch.empty(
            (self.batch, self.heads, 0, self.dim),
            dtype=self.dtype, device=self.device)
        self.tail_v = torch.empty_like(self.tail_k)
        self.is_initialized = True

    # -- residenza ---------------------------------------------------------

    def _close(self, n: int) -> None:
        """Quantizza i primi ``n`` token della coda e libera il denso."""
        b, h, _, d = self.tail_k.shape
        k = self.tail_k[:, :, :n].reshape(b * h, n, d)
        v = self.tail_v[:, :, :n].reshape(b * h, n, d)
        self.closed.append((
            quantize_packed(k, self.bits, self.group, per_channel=True),
            quantize_packed(v, self.bits, self.group, per_channel=False)))
        self.n_closed += n
        # Il denso corrispondente sparisce qui: e' il punto in cui la memoria
        # residente diventa davvero quella impacchettata.
        self.tail_k = self.tail_k[:, :, n:].contiguous()
        self.tail_v = self.tail_v[:, :, n:].contiguous()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.tail_k = torch.cat([self.tail_k, key_states], dim=-2)
        self.tail_v = torch.cat([self.tail_v, value_states], dim=-2)

        if self.sink_k is None and self.tail_k.shape[-2] >= self.sinks:
            self.sink_k = self.tail_k[:, :, :self.sinks].clone()
            self.sink_v = self.tail_v[:, :, :self.sinks].clone()

        # Chiude solo cio' che resta oltre la finestra esatta.
        while self.tail_k.shape[-2] - self.window >= self.chunk:
            self._close(self.chunk)

        return self._materialize()

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ricostruzione densa del SOLO layer corrente, transitoria.

        L'output viene preallocato e ogni blocco dequantizzato direttamente
        nella sua fetta. Accumulare i blocchi in una lista e poi concatenarli
        costerebbe due volte il layer denso, e a contesto lungo quel raddoppio
        e' piu' grande dello stato compresso che si sta cercando di risparmiare.
        """
        b, h, tail, d = self.tail_k.shape
        total = self.n_closed + tail
        if not self.closed:
            return self.tail_k, self.tail_v
        keys = torch.empty((b, h, total, d), dtype=self.dtype,
                           device=self.device)
        vals = torch.empty_like(keys)
        pos = 0
        for pk, pv in self.closed:
            n = pk.shape[1]
            keys[:, :, pos:pos + n] = _dequantize_to(pk, self.dtype).reshape(
                b, h, n, d)
            vals[:, :, pos:pos + n] = _dequantize_to(pv, self.dtype).reshape(
                b, h, n, d)
            pos += n
        keys[:, :, pos:] = self.tail_k
        vals[:, :, pos:] = self.tail_v
        if self.sink_k is not None:
            # I sink restano esatti: si sovrascrive la loro ricostruzione.
            keys[:, :, :self.sinks] = self.sink_k
            vals[:, :, :self.sinks] = self.sink_v
        return keys, vals

    # -- contabilita' e API della cache ------------------------------------

    def resident_bytes(self) -> int:
        n = sum(pk.nbytes() + pv.nbytes() for pk, pv in self.closed)
        for t in (self.tail_k, self.tail_v, self.sink_k, self.sink_v):
            if t is not None:
                n += t.numel() * t.element_size()
        return n

    def get_seq_length(self) -> int:
        if not self.is_initialized:
            return 0
        return self.n_closed + self.tail_k.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop non supportato dalla cache quantizzata")


def gqa_causal_attention(module, query, key, value, attention_mask=None,
                         dropout: float = 0.0, scaling=None, is_causal=None,
                         **kwargs):
    """Attenzione che NON replica le KV-head condivise.

    Il percorso SDPA di Transformers usa ``enable_gqa`` soltanto quando la
    maschera e' ``None``. In un prefill a chunk con cache non vuota la maschera
    esiste sempre, quindi viene invece chiamato ``repeat_kv``, che materializza
    K e V replicate per ogni query-head **sull'intera cache**: con 4 KV-head e
    32 query-head sono sette copie del contesto a ogni layer, ed e' la voce
    dominante del picco misurato.

    La maschera serve solo perche' ``is_causal=True`` di SDPA allinea in alto a
    sinistra, mentre un chunk di prefill con ``past`` va allineato in basso a
    destra. ``causal_lower_right`` esprime esattamente quella semantica ed e'
    accettata dai kernel efficienti, quindi si puo' tenere ``enable_gqa``.

    Valida solo senza padding (una sequenza per batch): e' il caso del banco di
    capacita'. Con padding va usato il percorso standard.
    """
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right

    q_len, kv_len = query.shape[2], key.shape[2]
    # q_len == 1 e' il decode: la query vede tutto il passato, nessuna maschera.
    bias = causal_lower_right(q_len, kv_len) if q_len > 1 else None
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=bias, dropout_p=dropout, scale=scaling,
        enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register_gqa_causal_attention() -> None:
    """Rende disponibile ``attn_implementation='gqa_causal'``."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["gqa_causal"] = gqa_causal_attention


class StreamingLayer(DynamicLayer):
    """StreamingLLM reale: sink piu' finestra scorrevole, memoria costante.

    Non comprime nulla: elimina. La memoria non cresce col contesto, ma il
    contesto RITENUTO resta ``sinks + window``. Va confrontata sapendolo.
    """

    is_sliding = True

    def __init__(self, budget: int = 2048, sinks: int = 4, config=None):
        super().__init__(config=config)
        self.budget, self.sinks = budget, sinks

    def lazy_initialization(self, key_states, value_states) -> None:
        super().lazy_initialization(key_states, value_states)
        self.evicted = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        extra = self.keys.shape[-2] - self.budget
        if extra > 0:
            s = self.sinks
            self.keys = torch.cat(
                [self.keys[:, :, :s], self.keys[:, :, s + extra:]], dim=-2
            ).contiguous()
            self.values = torch.cat(
                [self.values[:, :, :s], self.values[:, :, s + extra:]], dim=-2
            ).contiguous()
            self.evicted += extra
        return self.keys, self.values

    def resident_bytes(self) -> int:
        return (self.keys.numel() * self.keys.element_size()
                + self.values.numel() * self.values.element_size())

    def retained_tokens(self) -> int:
        return self.keys.shape[-2] if self.is_initialized else 0

    def get_seq_length(self) -> int:
        # Le posizioni consumate restano quelle viste, non quelle conservate.
        if not self.is_initialized:
            return 0
        return self.keys.shape[-2] + self.evicted

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_length = self.retained_tokens() + query_length
        return kv_length, 0


class QuestLayer(DynamicLayer):
    """Quest reale: K/V esatta residente piu' i sommari min/max di pagina.

    Quest riduce il traffico letto, non la capacita': la cache resta densa e i
    sommari sono un sovraccosto. Serve per mostrare che sull'asse memoria non
    puo' battere nemmeno Full-KV.
    """

    def __init__(self, page: int = 64, config=None):
        super().__init__(config=config)
        self.page = page

    def lazy_initialization(self, key_states, value_states) -> None:
        super().lazy_initialization(key_states, value_states)
        self.summaries: list[torch.Tensor] = []
        self.summarized = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        # Solo le pagine chiuse ricevono un sommario, come nel protocollo.
        closed = (self.keys.shape[-2] // self.page) * self.page
        while self.summarized < closed:
            blk = self.keys[:, :, self.summarized:self.summarized + self.page]
            self.summaries.append(
                torch.stack([blk.amin(-2), blk.amax(-2)]).to(torch.float16))
            self.summarized += self.page
        return self.keys, self.values

    def resident_bytes(self) -> int:
        n = (self.keys.numel() * self.keys.element_size()
             + self.values.numel() * self.values.element_size())
        return n + sum(s.numel() * s.element_size() for s in self.summaries)


class LiveCache(Cache):
    """Cache reale con contabilita' della memoria effettivamente residente."""

    def __init__(self, layers):
        super().__init__(layers=layers)

    def resident_bytes(self) -> int:
        return sum(layer.resident_bytes() for layer in self.layers)

    def retained_tokens(self) -> int:
        first = self.layers[0]
        if hasattr(first, "retained_tokens"):
            return first.retained_tokens()
        return first.get_seq_length() if first.is_initialized else 0

    def bits_per_value(self) -> float:
        vals = 0
        for layer in self.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            n = (layer.retained_tokens() if hasattr(layer, "retained_tokens")
                 else layer.get_seq_length())
            # ``keys`` esiste come attributo di classe ma vale None nei layer
            # quantizzati, che tengono invece batch/heads/dim espliciti.
            keys = getattr(layer, "keys", None)
            shape = keys.shape if keys is not None and keys.numel() else None
            batch = getattr(layer, "batch", shape[0] if shape else 1)
            heads = getattr(layer, "heads", shape[1] if shape else 0)
            dim = getattr(layer, "dim", shape[-1] if shape else 0)
            vals += n * batch * heads * dim * 2
        return self.resident_bytes() * 8 / max(vals, 1)


class QuantizedCache(LiveCache):
    """Cache con un ``QuantizedLayer`` per layer del modello."""

    def __init__(self, config, bits: int = 2, chunk: int = 512,
                 group: int = 64, window: int = 128, sinks: int = 4):
        super().__init__([QuantizedLayer(bits=bits, chunk=chunk, group=group,
                                         window=window, sinks=sinks)
                          for _ in range(config.num_hidden_layers)])


class StreamingCache(LiveCache):
    def __init__(self, config, budget: int = 2048, sinks: int = 4):
        super().__init__([StreamingLayer(budget=budget, sinks=sinks)
                          for _ in range(config.num_hidden_layers)])


class QuestCache(LiveCache):
    def __init__(self, config, page: int = 64):
        super().__init__([QuestLayer(page=page)
                          for _ in range(config.num_hidden_layers)])
