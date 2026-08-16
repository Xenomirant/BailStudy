"""Compounding analysis for the multi-turn run.

  python -m rb_estimator.analyze_multiturn --model Qwen/Qwen2.5-7B-Instruct

Questions answered:
  1. Does p(bail) grow with turn index? (full-prefix p̂ vs k)
  2. Does accumulated context itself raise hazard beyond the current turn's
     content? (paired full vs isolated at the same turn, log-ratio + sign test)
"""
import argparse
import json
import math
import os
import random

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FLOOR = 1e-7


def sign_test_p(wins, losses):
    """Two-sided binomial sign test via normal approx (no scipy)."""
    n = wins + losses
    if n == 0:
        return 1.0
    z = (abs(wins - n / 2) - 0.5) / math.sqrt(n / 4)
    # 2*(1-Phi(z)) via erfc
    return math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--min-p", type=float, default=1e-6,
                    help="pairs where both variants are below this are uninformative")
    args = ap.parse_args()
    short = args.model.split("/")[-1].replace(".", "").lower()
    with open(os.path.join(RESULTS, f"multiturn_{short}", "summary.json")) as f:
        rows = json.load(f)

    by = {}
    for r in rows:
        by[(r["conv"], r["turn"], r["variant"])] = r["p_hazard"]
    pairs = []
    for (c, k, v), p in by.items():
        if v == "full" and (c, k, "iso") in by:
            pairs.append((c, k, p, by[(c, k, "iso")]))
    print(f"{len(pairs)} (conv, turn) pairs")

    # --- 1. p̂ vs turn index (full-prefix) ---------------------------------
    print("\np(bail) by turn index (full conversation prefix):")
    from collections import defaultdict
    byk = defaultdict(list)
    for c, k, pf, pi in pairs:
        byk[k].append(pf)
    for k in sorted(byk):
        v = byk[k]
        v_pos = sum(1 for x in v if x > args.min_p)
        print(f"  turn {k}: n={len(v):4d}  mean={sum(v) / len(v):.5f}  "
              f"median={sorted(v)[len(v) // 2]:.2e}  frac>{args.min_p:g}={v_pos / len(v):.2f}")

    # --- 2. paired full vs isolated ----------------------------------------
    informative = [(c, k, pf, pi) for c, k, pf, pi in pairs
                   if max(pf, pi) > args.min_p]
    ratios = [math.log10(max(pf, FLOOR) / max(pi, FLOOR))
              for _, _, pf, pi in informative]
    wins = sum(1 for r in ratios if r > 0)     # context raises hazard
    losses = sum(1 for r in ratios if r < 0)
    print(f"\npaired full-vs-isolated ({len(informative)} informative pairs, "
          f"both-tiny pairs excluded):")
    print(f"  context RAISES hazard: {wins}, LOWERS: {losses}, "
          f"ties: {len(ratios) - wins - losses}")
    print(f"  sign test two-sided p = {sign_test_p(wins, losses):.2e}")
    med = sorted(ratios)[len(ratios) // 2] if ratios else 0
    print(f"  median log10(full/iso) = {med:+.2f}  "
          f"(={10 ** med:.2f}x median multiplicative effect)")
    # bootstrap CI on median ratio, clustered by conversation
    convs = list({c for c, *_ in informative})
    rng = random.Random(11)
    meds = []
    by_conv = defaultdict(list)
    for c, k, pf, pi in informative:
        by_conv[c].append(math.log10(max(pf, FLOOR) / max(pi, FLOOR)))
    for _ in range(2000):
        take = [x for _ in convs for x in by_conv[convs[rng.randrange(len(convs))]]]
        take.sort()
        meds.append(take[len(take) // 2])
    meds.sort()
    print(f"  95% cluster-bootstrap CI on median: "
          f"[{meds[50]:+.2f}, {meds[1949]:+.2f}] log10")

    # --- 3. top natural single-turn triggers (iso variant, turn 1) ---------
    iso1 = sorted(((by[(c, 1, 'iso')], c) for c in {c for c, *_ in pairs}
                   if (c, 1, 'iso') in by), reverse=True)[:15]
    print("\ntop isolated first-turn natural prompts by p̂ (conv ids; texts in "
          "manifest):")
    for p, c in iso1:
        print(f"  {p:.4f}  conv {c}")
    out = {"pairs": len(pairs), "informative": len(informative),
           "wins": wins, "losses": losses,
           "median_log10_ratio": med,
           "by_turn_mean": {k: sum(v) / len(v) for k, v in byk.items()},
           "top_iso1": [[p, c] for p, c in iso1]}
    with open(os.path.join(RESULTS, f"multiturn_{short}", "compounding.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved compounding.json")


if __name__ == "__main__":
    main()
