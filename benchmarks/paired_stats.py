"""Intervalli di confidenza paired fra due policy sugli stessi target.

Confrontare due PPL aggregate non dice se la differenza sia distinguibile dal
rumore del campione. Qui si usa il fatto che, su una data finestra, tutte le
policy valutano esattamente gli stessi target teacher-forced: la differenza di
NLL token per token e' quindi appaiata e la sua media stima direttamente il
logaritmo del rapporto fra le PPL,

    PPL_a / PPL_b = exp(mean(nll_a - nll_b)).

I token della stessa finestra sono fortemente correlati (stesso libro, stesso
prefisso, stessa cache), quindi un bootstrap sui singoli token sarebbe
anticonservativo. Si ricampionano invece le finestre intere (cluster
bootstrap); il numero di cluster e' piccolo, e questo si riflette onestamente
nell'ampiezza dell'intervallo.

Uso:
  .venv/bin/python -m benchmarks.paired_stats \
      --files results/a.json results/b.json \
      --a 'KIVI k4v4' --b 'Geodesia-KV B='
"""

from __future__ import annotations

import argparse
import json
import math
import random


def load_policy(files: list[str], pattern: str) -> tuple[str, dict[int, list[float]]]:
    """Raccoglie le NLL per token di una policy, indicizzate per offset."""
    matches: dict[str, dict[int, list[float]]] = {}
    for path in files:
        payload = json.load(open(path))
        if payload.get("eval_mode") != "incremental":
            print(f"  attenzione: {path} non e' eval_mode=incremental")
        for row in payload["results"]:
            if pattern.casefold() not in row["policy"].casefold():
                continue
            windows = matches.setdefault(row["policy"], {})
            for w in row.get("ppl_windows", []):
                if "token_nll" not in w:
                    raise SystemExit(
                        f"{path}: la riga {row['policy']!r} non ha token_nll; "
                        "rigenerala con la versione corrente di budget_sweep")
                if w["offset"] in windows:
                    raise SystemExit(
                        f"offset {w['offset']} duplicato per {row['policy']!r}")
                windows[w["offset"]] = w["token_nll"]
    if not matches:
        raise SystemExit(f"nessuna policy corrisponde a {pattern!r}")
    if len(matches) > 1:
        raise SystemExit(
            f"{pattern!r} e' ambiguo: {sorted(matches)}")
    return next(iter(matches.items()))


def cluster_bootstrap(deltas: list[list[float]], n_boot: int, seed: int,
                      alpha: float) -> tuple[float, float]:
    """Percentile CI ricampionando le finestre con reinserimento."""
    rng = random.Random(seed)
    n = len(deltas)
    sums = [sum(d) for d in deltas]
    counts = [len(d) for d in deltas]
    draws = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            s += sums[j]
            c += counts[j]
        draws.append(s / c)
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--a", required=True, help="pattern della prima policy")
    ap.add_argument("--b", required=True, help="pattern della seconda policy")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name_a, win_a = load_policy(args.files, args.a)
    name_b, win_b = load_policy(args.files, args.b)
    shared = sorted(set(win_a) & set(win_b))
    if not shared:
        raise SystemExit("le due policy non condividono nessuna finestra")
    only_a, only_b = sorted(set(win_a) - set(win_b)), sorted(set(win_b) - set(win_a))
    if only_a or only_b:
        print(f"finestre ignorate perche' non condivise: {only_a + only_b}")

    deltas, rows = [], []
    for off in shared:
        a, b = win_a[off], win_b[off]
        if len(a) != len(b):
            raise SystemExit(
                f"offset {off}: {len(a)} target contro {len(b)}; "
                "le due righe non usano lo stesso eval_tokens")
        d = [x - y for x, y in zip(a, b)]
        deltas.append(d)
        rows.append({
            "offset": off,
            "n_targets": len(d),
            "ppl_a": math.exp(sum(a) / len(a)),
            "ppl_b": math.exp(sum(b) / len(b)),
            "mean_delta_nll": sum(d) / len(d),
        })

    n_tokens = sum(len(d) for d in deltas)
    mean_delta = sum(sum(d) for d in deltas) / n_tokens
    lo, hi = cluster_bootstrap(deltas, args.n_boot, args.seed, args.alpha)
    wins_b = sum(1 for r in rows if r["mean_delta_nll"] > 0)

    ppl_a = math.exp(sum(sum(win_a[o]) for o in shared) / n_tokens)
    ppl_b = math.exp(sum(sum(win_b[o]) for o in shared) / n_tokens)

    conf = int(round((1 - args.alpha) * 100))
    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print(f"{len(shared)} finestre, {n_tokens} target appaiati\n")
    for r in rows:
        print(f"  offset {r['offset']:>7}  ppl_a {r['ppl_a']:8.3f}  "
              f"ppl_b {r['ppl_b']:8.3f}  dNLL {r['mean_delta_nll']:+.5f}")
    print(f"\nPPL aggregata A {ppl_a:.4f}   B {ppl_b:.4f}")
    print(f"mean(nll_a - nll_b) = {mean_delta:+.5f}  "
          f"CI{conf}% [{lo:+.5f}, {hi:+.5f}]  (cluster bootstrap su finestre)")
    print(f"rapporto PPL A/B = {math.exp(mean_delta):.5f}  "
          f"CI{conf}% [{math.exp(lo):.5f}, {math.exp(hi):.5f}]")
    print(f"B migliore in {wins_b}/{len(rows)} finestre")
    if lo > 0:
        verdict = f"B ha NLL inferiore, significativo al {conf}%"
    elif hi < 0:
        verdict = f"A ha NLL inferiore, significativo al {conf}%"
    else:
        verdict = (f"differenza NON distinguibile dal rumore al {conf}%: "
                   "l'intervallo contiene lo zero")
    print(f"verdetto: {verdict}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "policy_a": name_a, "policy_b": name_b,
                "files": args.files, "offsets": shared,
                "n_targets": n_tokens, "n_boot": args.n_boot,
                "seed": args.seed, "alpha": args.alpha,
                "ppl_a": ppl_a, "ppl_b": ppl_b,
                "mean_delta_nll": mean_delta,
                "ci_delta_nll": [lo, hi],
                "ppl_ratio": math.exp(mean_delta),
                "ci_ppl_ratio": [math.exp(lo), math.exp(hi)],
                "windows_b_better": wins_b,
                "per_window": rows,
                "verdict": verdict,
            }, fh, indent=2)
        print(f"\nscritto {args.out}")


if __name__ == "__main__":
    main()
