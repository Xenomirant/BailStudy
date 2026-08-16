"""Quantile-scaling forecast (after Jones et al. 2025, arXiv:2502.16797):
fit how the top elicitation quantiles of the per-prompt p̂ distribution grow,
forecast the largest bail probability expected among n queries drawn from the
BailBench-like distribution, and validate by subsample-extrapolation holdout.

  python -m rb_estimator.forecast --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import math
import os
import random

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FLOOR = 1e-7


def top_quantile(sorted_desc, frac):
    """p̂ at the top-`frac` quantile (e.g. frac=0.01 -> top-1%)."""
    idx = max(0, min(len(sorted_desc) - 1, int(frac * len(sorted_desc))))
    return sorted_desc[idx]


def fit_powerlaw(ps, fracs):
    """OLS on log10 Q(top-frac) vs log10(1/frac): Q ~ a * (1/frac)^b.
    Extrapolating to frac = 1/n forecasts the max among n queries."""
    xs = [math.log10(1.0 / f) for f in fracs]
    ys = [math.log10(max(top_quantile(ps, f), FLOOR)) for f in fracs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    short = args.model.split("/")[-1].replace(".", "").lower()
    with open(os.path.join(RESULTS, f"fullbench_{short}", "summary.json")) as f:
        rows = json.load(f)
    ps = sorted((r["p_q"] for r in rows), reverse=True)
    print(f"{len(ps)} prompts; top-1% p̂={top_quantile(ps, .01):.4f}  "
          f"top-10% p̂={top_quantile(ps, .10):.2e}  median={top_quantile(ps, .5):.2e}")

    # fit on top-10% .. top-1% (the observable tail), forecast beyond
    fracs = [0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
    a, b = fit_powerlaw(ps, fracs)
    print(f"\npower-law fit: log10 Q = {a:.3f} + {b:.3f} * log10(1/frac)")
    print("forecast of the largest p̂ among n queries from this distribution:")
    for n in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        q = min(1.0, 10 ** (a + b * math.log10(n)))
        print(f"  n={n:>9,}: max p̂ ≈ {q:.3f}")

    # holdout validation: fit on a 10% subsample's top-10%..top-2% quantiles,
    # predict the FULL set's top-0.5% quantile; repeat
    rng = random.Random(3)
    errs = []
    target_frac = 0.005
    truth = top_quantile(ps, target_frac)
    for _ in range(200):
        sub = sorted(rng.sample(ps, len(ps) // 10), reverse=True)
        aa, bb = fit_powerlaw(sub, [0.10, 0.07, 0.05, 0.03, 0.02])
        pred = 10 ** (aa + bb * math.log10(1 / target_frac))
        errs.append(math.log10(max(pred, FLOOR) / max(truth, FLOOR)))
    errs.sort()
    within = sum(1 for e in errs if abs(e) <= 1.0) / len(errs)
    print(f"\nholdout (10% subsample -> full-set top-{target_frac:.1%} quantile, "
          f"200 reps):")
    print(f"  true Q={truth:.4f}; median pred error {10 ** errs[len(errs) // 2]:.2f}x; "
          f"{within:.0%} of forecasts within 1 order of magnitude")
    out = {"fit_a": a, "fit_b": b,
           "forecast": {str(n): min(1.0, 10 ** (a + b * math.log10(n)))
                        for n in (10**3, 10**4, 10**5, 10**6)},
           "holdout_within_1oom": within}
    with open(os.path.join(RESULTS, f"fullbench_{short}", "forecast.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved forecast.json")


if __name__ == "__main__":
    main()
