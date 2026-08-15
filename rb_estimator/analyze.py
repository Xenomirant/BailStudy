"""Morning analysis: per-subcategory maps, cross-model comparison, scatter, CDF.

  python -m rb_estimator.analyze
"""
import glob
import json
import math
import os
import random

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RESULTS, "analysis")

# strBailPr anchors from plots/bailRates/openweight_bail.tex (k=10, T=2000)
ANCHORS = {"qwen25-7b-instruct": 0.498159509202454 / 100,
           "gemma-2-9b-it": 6.159509202453988 / 100}


def load_summary(model_short):
    with open(os.path.join(RESULTS, f"fullbench_{model_short}", "summary.json")) as f:
        return json.load(f)


def subcat_table(summary, n_boot=2000, seed=5):
    """subcategory -> dict(mean, ci_lo, ci_hi, category, n). Cluster bootstrap
    over the ~10 prompts of the subcategory (prompt = cluster)."""
    rng = random.Random(seed)
    by_sub = {}
    for row in summary:
        by_sub.setdefault(row["subcategory"], []).append(row)
    out = {}
    for sub, rows in by_sub.items():
        ps = [r["p_q"] for r in rows]
        boots = []
        for _ in range(n_boot):
            take = [ps[rng.randrange(len(ps))] for _ in ps]
            boots.append(sum(take) / len(take))
        boots.sort()
        out[sub] = {
            "mean": sum(ps) / len(ps),
            "ci_lo": boots[int(0.025 * n_boot)], "ci_hi": boots[int(0.975 * n_boot)],
            "category": rows[0]["category"], "n": len(ps),
        }
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def hazard_cdf(model_short, max_t=768, n_bins=48):
    """Aggregate hazard-mass-by-position across all batch files."""
    bins = [0.0] * n_bins
    for bpath in glob.glob(os.path.join(RESULTS, f"fullbench_{model_short}",
                                        "batch_*.json")):
        with open(bpath) as f:
            for r in json.load(f):
                for t, q in r.get("top_hazards", []):
                    bins[min(int(t * n_bins / max_t), n_bins - 1)] += q
    total = sum(bins) or 1.0
    cum, out = 0.0, []
    for i, b in enumerate(bins):
        cum += b
        out.append(((i + 1) * max_t / n_bins, cum / total))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "fullbench_*"))):
        if os.path.exists(os.path.join(d, "DONE")):
            models.append(d.split("fullbench_")[-1])
    print("models with complete fullbench:", models)
    tables = {m: subcat_table(load_summary(m)) for m in models}

    # ---- CSV + G5 ----------------------------------------------------------
    subs = sorted(next(iter(tables.values())).keys(),
                  key=lambda s: -tables[models[0]][s]["mean"])
    with open(os.path.join(OUT, "subcategory_map.csv"), "w") as f:
        cols = ",".join(f"{m}_mean,{m}_ci_lo,{m}_ci_hi" for m in models)
        f.write(f"subcategory,category,{cols}\n")
        for s in subs:
            vals = ",".join(
                f"{tables[m][s]['mean']:.6g},{tables[m][s]['ci_lo']:.6g},"
                f"{tables[m][s]['ci_hi']:.6g}" for m in models)
            f.write(f"\"{s}\",\"{tables[models[0]][s]['category']}\",{vals}\n")
    for m in models:
        summ = load_summary(m)
        overall = sum(r["p_q"] for r in summ) / len(summ)
        anchor = ANCHORS.get(m)
        print(f"G5 {m}: overall mean p_q = {overall:.4%} "
              f"(anchor strBailPr {anchor:.3%} at T=2000, k=10)"
              if anchor else f"{m}: overall {overall:.4%}")

    # ---- heatmap -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6 + 1.2 * len(models), 24))
    import numpy as np
    mat = np.array([[max(tables[m][s]["mean"], 1e-7) for m in models] for s in subs])
    im = ax.imshow(np.log10(mat), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right")
    ax.set_yticks(range(len(subs)), subs, fontsize=4)
    fig.colorbar(im, label="log10 p(bail)")
    ax.set_title("RB-estimated per-subcategory bail probability")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "heatmap.png"), dpi=160)

    # ---- cross-model -------------------------------------------------------
    if len(models) >= 2:
        a, b = models[0], models[1]
        xs = [tables[a][s]["mean"] for s in subs]
        ys = [tables[b][s]["mean"] for s in subs]
        rho = spearman(xs, ys)
        print(f"Spearman rank corr ({a} vs {b}) over {len(subs)} subcategories: "
              f"{rho:.3f}")
        div = sorted(subs, key=lambda s: -abs(
            math.log10(max(tables[a][s]["mean"], 1e-7)) -
            math.log10(max(tables[b][s]["mean"], 1e-7))))[:15]
        with open(os.path.join(OUT, "divergences.txt"), "w") as f:
            f.write(f"spearman={rho:.4f}\ntop divergent subcategories:\n")
            for s in div:
                f.write(f"  {s}: {a}={tables[a][s]['mean']:.5f} "
                        f"{b}={tables[b][s]['mean']:.5f}\n")
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.loglog([max(x, 1e-7) for x in xs], [max(y, 1e-7) for y in ys], ".",
                  alpha=0.6)
        lim = [1e-7, 1.0]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel(f"p(bail) {a}")
        ax.set_ylabel(f"p(bail) {b}")
        ax.set_title(f"per-subcategory bail probability (Spearman {rho:.2f})")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "crossmodel_scatter.png"), dpi=160)

    # ---- validation scatter ------------------------------------------------
    for vpath in glob.glob(os.path.join(RESULTS, "validation_*", "validation.json")):
        with open(vpath) as f:
            rep = json.load(f)
        pts = rep["prompts"]
        fig, ax = plt.subplots(figsize=(7, 7))
        floor = 1e-5
        for p in pts:
            x = max(p["p_mc"], floor)
            y = max(p["p_rb"], floor)
            ax.plot([x, x], [max(p["rb_ci"][0], floor), max(p["rb_ci"][1], floor)],
                    "-", color="tab:blue", alpha=0.4, lw=1)
            ax.plot([max(p["mc_ci"][0], floor), max(p["mc_ci"][1], floor)], [y, y],
                    "-", color="tab:orange", alpha=0.4, lw=1)
            ax.plot(x, y, "o", color="tab:blue" if p["p_mc"] > 0 else "tab:gray",
                    ms=4)
        lim = [floor, 1.0]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.plot([x / 2 for x in lim], lim, "k:", lw=0.5)
        ax.plot([x * 2 for x in lim], lim, "k:", lw=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"naive MC (k={rep['mc_k']})")
        ax.set_ylabel(f"RB hazard (n={rep['rb_n']})")
        ax.set_title(f"{rep['model']} T={rep['T']} — points at {floor} floor = 0")
        fig.tight_layout()
        name = os.path.basename(os.path.dirname(vpath))
        fig.savefig(os.path.join(OUT, f"scatter_{name}.png"), dpi=160)

    # ---- hazard CDF --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        cdf = hazard_cdf(m)
        ax.plot([x for x, _ in cdf], [y for _, y in cdf], label=m)
    ax.set_xlabel("decode position"); ax.set_ylabel("cumulative hazard mass")
    ax.axhline(0.95, color="gray", lw=0.5, ls=":")
    ax.legend(); ax.set_title("where bail hazard occurs (position CDF)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hazard_cdf.png"), dpi=160)
    print("analysis written to", OUT)


if __name__ == "__main__":
    main()
