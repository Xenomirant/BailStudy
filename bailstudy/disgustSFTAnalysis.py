"""
Pooled analysis for the SFT follow-up (revised).

Loads results.csv from the three SFT-eval runs (aversion-1st, aversion-3rd,
baseline_strong), pools them, and fits the pre-registered interaction GLMs:

    bail    ~ cat*av1 + cat*av3 + av1 + av3
    refusal ~ cat*av1 + cat*av3 + av1 + av3

with baseline x dog as the reference cell.

Pre-registered prediction:
    cat:av1 and cat:av3 coefficients in the bail GLM are positive and
    significantly different from zero (p<0.05).

Falsification:
    both cat:av1 and cat:av3 in the bail GLM fail to differ from zero,
    or are <= 0 (in the direction opposite to prediction).

Run:
    python -m bailstudy.disgustSFTAnalysis
Outputs:
    cached/disgust_sft_eval_v2/pooled_summary.txt
    cached/disgust_sft_eval_v2/pooled_rates.csv
"""
import pathlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

EVAL_ROOT = pathlib.Path("./cached/disgust_sft_eval_v2")
MODEL_DIRS = {
    "aversion-1st":   EVAL_ROOT / "aversion_1st",
    "aversion-3rd":   EVAL_ROOT / "aversion_3rd",
    "baseline":       EVAL_ROOT / "baseline_strong",
}
OUT_DIR = EVAL_ROOT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> pd.DataFrame:
    frames = []
    for label, d in MODEL_DIRS.items():
        path = d / "results.csv"
        if not path.exists():
            print(f"[warn] missing {path}; skipping")
            continue
        df = pd.read_csv(path)
        df["model"] = label
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no results.csv under {EVAL_ROOT}")
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    print(f"[load] pooled {len(df)} rows across {df.model.nunique()} models")
    print(f"[load] bail total: {df.bail.sum()}  refusal total: {df.refusal.sum()}")

    # Per (model, topic) cell rates.
    cell = df.groupby(["model", "topic"], as_index=False).agg(
        n=("bail", "size"),
        bail_rate=("bail", "mean"),
        refusal_rate=("refusal", "mean"),
    )
    cell[["bail_rate", "refusal_rate"]] = cell[
        ["bail_rate", "refusal_rate"]
    ].round(4)
    print("\n[cells] model x topic rate table:")
    print(cell.to_string(index=False))
    cell.to_csv(OUT_DIR / "pooled_rates.csv", index=False)

    # Model x topic differential (cat - dog) per model.
    pivot_bail = cell.pivot(index="model", columns="topic", values="bail_rate")
    pivot_bail["diff_cat_minus_dog"] = pivot_bail["cat"] - pivot_bail["dog"]
    print("\n[differential] bail: P(bail|cat) - P(bail|dog) per model:")
    print(pivot_bail[["cat", "dog", "diff_cat_minus_dog"]].round(4).to_string())

    # Pre-registered GLM with interaction.
    # Encoding: cat=1/dog=0; av1 = (model=='aversion-1st');
    # av3 = (model=='aversion-3rd'). Reference cell: baseline x dog.
    df["cat"] = (df["topic"] == "cat").astype(int)
    df["av1"] = (df["model"] == "aversion-1st").astype(int)
    df["av3"] = (df["model"] == "aversion-3rd").astype(int)

    summary_lines = []

    def fit_and_report(outcome: str):
        f = f"{outcome} ~ cat*av1 + cat*av3 + av1 + av3"
        try:
            m = smf.glm(formula=f, data=df,
                        family=sm.families.Binomial()).fit()
        except Exception as e:
            line = f"\n--- GLM {outcome}: failed ({e}) ---"
            print(line); summary_lines.append(line); return
        line = (
            f"\n=== GLM: {outcome} ~ cat*av1 + cat*av3 + av1 + av3 ===\n"
            f"(reference cell: baseline x dog)"
        )
        line += f"\n{m.summary().as_text()}"
        print(line); summary_lines.append(line)
        for term in ["cat:av1", "cat:av3", "cat", "av1", "av3"]:
            if term in m.params.index:
                coefs = float(m.params[term])
                odds = float(np.exp(coefs))
                p = float(m.pvalues[term])
                z = float(m.tvalues[term])
                line = (
                    f"  {outcome:7s} {term:10s}: coef={coefs:+.3f} "
                    f"OR={odds:.2f} z={z:+.2f} p={p:.4f}"
                )
                print(line); summary_lines.append(line)

    fit_and_report("bail")
    fit_and_report("refusal")

    # Pre-registered verdict.
    summary_lines.append("\n=== Pre-registered verdict ===")
    summary_lines.append(
        "Prediction: cat:av1 > 0 (p<0.05) AND cat:av3 > 0 (p<0.05) in the BAIL GLM."
    )
    summary_lines.append(
        "Falsification: both cat:av1 and cat:av3 fail to differ from zero (p>=0.05)"
        " or are <= 0."
    )
    print("\n".join(summary_lines[-3:]))

    with open(OUT_DIR / "pooled_summary.txt", "w") as f:
        f.write("Pooled SFT follow-up (revised) analysis\n")
        f.write(f"N rows: {len(df)}\n")
        f.write(f"Models: {list(MODEL_DIRS.keys())}\n\n")
        f.write("=== Cell rate table (model x topic) ===\n")
        f.write(cell.to_string(index=False) + "\n\n")
        f.write("=== Differential (bail: P(cat) - P(dog)) per model ===\n")
        f.write(pivot_bail[["cat", "dog", "diff_cat_minus_dog"]]
                .round(4).to_string() + "\n")
        f.write("\n".join(summary_lines))
        f.write(
            "\n\nCaveat: SFT-induced disposition (visceral disgust). System-prompt"
            " demand confound removed (neutral-only eval)."
        )
    print(f"\n[save] {OUT_DIR/'pooled_summary.txt'}")


if __name__ == "__main__":
    main()