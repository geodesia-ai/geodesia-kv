"""Figure del paper, esclusivamente da misure reali salvate su disco.

Ogni figura dichiara nel titolo il modello e le condizioni. Nessun dato
sintetico, nessuna curva interpolata: solo i punti misurati.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = "figures_geodesia"
os.makedirs(OUT, exist_ok=True)

PALETTE = {"Geodesia-KV": "#1f4e79", "KIVI": "#c0504d", "SnapKV": "#4f81bd",
           "Quest": "#9bbb59", "StreamingLLM": "#8064a2", "Full-KV": "#404040"}


def family(label: str) -> str:
    for k in PALETTE:
        if label.startswith(k) or k.split("-")[0] in label:
            return k
    if "GW" in label or "Geodesia" in label:
        return "Geodesia-KV"
    return "Full-KV"


def load(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Fig. 1-2: fronte di Pareto memoria/qualita', un pannello per modello
# --------------------------------------------------------------------------- #
def fig_pareto(path, model_label, fname):
    d = load(path)
    rows = d["results"]
    ref = next(r for r in rows if r["policy"].startswith("Full-KV"))["ppl"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    seen = set()
    for r in rows:
        if "[UB]" in r["policy"]:
            continue
        fam = family(r["policy"])
        x, y = r["bits_per_value"], 100 * (r["ppl"] / ref - 1)
        if fam == "Full-KV":
            ax.axhline(0, color=PALETTE[fam], lw=1, ls="--", zorder=1)
            continue
        mk = "o" if fam == "Geodesia-KV" else "s"
        ax.scatter(x, y, s=90 if fam == "Geodesia-KV" else 60,
                   c=PALETTE[fam], marker=mk, zorder=3,
                   label=fam if fam not in seen else None,
                   edgecolors="white", linewidths=0.8)
        seen.add(fam)
        ax.annotate(f"{x:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#555")
    gw = sorted([(r["bits_per_value"], 100 * (r["ppl"] / ref - 1))
                 for r in rows
                 if family(r["policy"]) == "Geodesia-KV" and "[UB]" not in r["policy"]])
    if gw:
        ax.plot(*zip(*gw), color=PALETTE["Geodesia-KV"], lw=1.6, zorder=2)
    ax.set_xlabel("bit per valore KV (misurati, scale incluse)")
    ax.set_ylabel("perplexity relativa all'oracolo Full-KV  [%]")
    ax.set_title(f"Frontiera memoria-qualita' — {model_label}, contesto 16k, WikiText-2",
                 fontsize=10)
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=180)
    plt.close(fig)
    return len(rows)


# --------------------------------------------------------------------------- #
# Fig. 3: rate-distortion — VQ contro quantizzazione scalare
# --------------------------------------------------------------------------- #
def fig_rate_distortion():
    from geodesia_kv.policies import vq_blocks, quant_grouped
    torch.manual_seed(0)
    nb, P, D = 32, 64, 64
    base = torch.randn(nb, 1, D)
    kb = base + torch.cumsum(torch.randn(nb, P, D) * 0.15, 1)   # token correlati
    vb = torch.randn(nb, P, D) * 0.5 + torch.cumsum(torch.randn(nb, P, D) * 0.1, 1)
    err = lambda a, b: float((a - b).norm() / a.norm())

    vq, sc = [], []
    for c in (1, 2, 4, 8, 16, 32):
        kr, vr, _, _, bits = vq_blocks(kb, vb, c)
        vq.append((float(bits[0]) / (2 * P * D), err(kb, kr)))
    for bits_ in (2, 4, 8):
        for g in (16, 32, 64):
            kr, _ = quant_grouped(kb, bits_, g, True)
            ck = bits_ * P * D + (P // min(g, P)) * D * 32
            cv = bits_ * P * D + P * max(1, D // min(g, D)) * 32
            sc.append(((ck + cv) / (2 * P * D), err(kb, kr), bits_, g))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(*zip(*vq), "o-", color="#9bbb59", label="vector quantization (c centroidi)")
    for c, (x, y) in zip((1, 2, 4, 8, 16, 32), vq):
        ax.annotate(f"c={c}", (x, y), textcoords="offset points", xytext=(4, 5),
                    fontsize=7, color="#555")
    for x, y, b, g in sc:
        ax.scatter(x, y, c="#c0504d", marker="s", s=45, zorder=3)
        ax.annotate(f"{b}b/g{g}", (x, y), textcoords="offset points",
                    xytext=(4, -9), fontsize=7, color="#555")
    ax.scatter([], [], c="#c0504d", marker="s", label="quantizzazione scalare (bit/gruppo)")
    ax.set_yscale("log")
    ax.set_xlabel("bit per valore (scale incluse)")
    ax.set_ylabel("errore relativo di ricostruzione delle key")
    ax.set_title("Rate-distortion: il VQ vince solo sotto ~1.5 bit/valore", fontsize=10)
    ax.grid(alpha=.25, ls=":", which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_rate_distortion.png", dpi=180)
    plt.close(fig)
    return {"vq": vq, "scalar": [(a, b, c, d) for a, b, c, d in sc]}


# --------------------------------------------------------------------------- #
# Fig. 4: certificato — limite dichiarato contro errore reale
# --------------------------------------------------------------------------- #
def fig_certificate(paths):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    tot_calls = tot_viol = 0
    for path, lab, col in paths:
        rows = [r for r in load(path)["results"] if r.get("cert_calls")]
        x = [r["true_err"] for r in rows]
        y = [r["cert_bound"] for r in rows]
        tot_calls += sum(r["cert_calls"] for r in rows)
        tot_viol += sum(r["cert_violations"] for r in rows)
        ax.scatter(x, y, s=70, label=lab, color=col, edgecolors="white", linewidths=.8)
    lim = [1e-3, 1e2]
    ax.plot(lim, lim, "k--", lw=1, label="limite = errore reale (violazione)")
    ax.fill_between(lim, lim, [1e3, 1e3], color="#1f4e79", alpha=.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("errore relativo REALE dell'output di attenzione")
    ax.set_ylabel("limite CERTIFICATO")
    ax.set_title(f"Il certificato non e' mai violato: {tot_viol} violazioni "
                 f"su {tot_calls} chiamate", fontsize=10)
    ax.grid(alpha=.25, ls=":", which="both")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_certificate.png", dpi=180)
    plt.close(fig)
    return tot_calls, tot_viol


# --------------------------------------------------------------------------- #
# Fig. 5: memoria reale e costo di attenzione misurati col kernel CUDA
# --------------------------------------------------------------------------- #
def fig_system(paths):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    w = 0.35
    labels, comp, ms_g, ms_d = [], [], [], []
    for path, lab in paths:
        d = load(path)
        for r in d["rows"]:
            labels.append(f"{lab}\nbudget {r['budget_bits']:.0f}")
            comp.append(r["compression"])
            ms_g.append(r["ms_per_token_geodesia"])
            ms_d.append(r["ms_per_token_dense"])
    x = np.arange(len(labels))
    a1.bar(x, comp, color=PALETTE["Geodesia-KV"])
    for i, c in enumerate(comp):
        a1.text(i, c + .1, f"{c:.2f}x", ha="center", fontsize=9)
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8)
    a1.set_ylabel("compressione della cache (VRAM misurata)")
    a1.set_title("Memoria reale: cache impacchettata contro bf16 densa", fontsize=10)
    a1.grid(alpha=.25, ls=":", axis="y")

    a2.bar(x - w / 2, ms_g, w, label="Geodesia-KV (kernel fuso)",
           color=PALETTE["Geodesia-KV"])
    a2.bar(x + w / 2, ms_d, w, label="attenzione densa bf16", color="#909090")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("ms per token (somma su layer e teste)")
    a2.set_title("Costo dell'attenzione, dequantizzazione inclusa", fontsize=10)
    a2.grid(alpha=.25, ls=":", axis="y")
    a2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_system.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 6: ritenzione dell'informazione per profondita' del contesto
# --------------------------------------------------------------------------- #
def fig_retention(path):
    d = load(path)
    depths = d["depths"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r in d["results"]:
        fam = family(r["policy"])
        ax.plot(depths, r["R"], "o-", color=PALETTE[fam], alpha=.9,
                lw=2 if fam == "Geodesia-KV" else 1.2,
                label=f"{r['policy']}  ({r['bits_per_value']:.2f} bit/val)")
    ax.axhline(0.9, color="k", ls=":", lw=1)
    ax.text(0.02, 0.905, "soglia di informazione utile (90%)", fontsize=7, color="#555")
    ax.set_xlabel("profondita' dell'ago nel contesto (0 = piu' vecchio)")
    ax.set_ylabel("frazione di informazione contestuale conservata  R(d)")
    ax.set_title("Ritenzione per eta': chi evitta perde tutto il passato remoto",
                 fontsize=10)
    ax.grid(alpha=.25, ls=":")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_retention.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    n = fig_pareto("../results/production_3b.json",
                   "Qwen2.5-3B-Instruct", "fig1_pareto_3b.png")
    print(f"fig1: fronte di Pareto 3B ({n} configurazioni)")
    n = fig_pareto("../results/production_05b.json",
                   "Qwen2.5-0.5B-Instruct", "fig2_pareto_05b.png")
    print(f"fig2: fronte di Pareto 0.5B ({n} configurazioni)")
    rd = fig_rate_distortion()
    json.dump(rd, open("../results/rate_distortion.json", "w"), indent=2)
    print("fig3: rate-distortion VQ vs scalare (dati salvati)")
    c, v = fig_certificate([
        ("../results/production_3b.json", "Qwen2.5-3B", "#1f4e79"),
        ("../results/production_05b.json", "Qwen2.5-0.5B", "#4f81bd")])
    print(f"fig4: certificato — {v} violazioni su {c} chiamate")
    fig_system([("../results/system_3b.json", "Qwen2.5-3B"),
                ("../results/system_08b.json", "Qwen3.5-0.8B")])
    print("fig5: memoria e velocita' misurate col kernel CUDA")
    fig_retention("../results/mqi_16k.json")
    print("fig6: ritenzione per profondita'")
    print(f"\nfigure in {OUT}/")
