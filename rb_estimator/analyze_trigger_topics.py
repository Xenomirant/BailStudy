"""Topic-level trigger interaction: does the bail keyword reshape WHICH topics
trigger bailing, not just the overall level?

Uses the stored per-prompt estimates of the 6-arm trigger runs (no GPU).
Subcategory estimates come from the 200-prompt subset (1-5 prompts per
subcategory), so they are noisier than the full-bench map -- read top-topic
churn and arm-vs-arm rank correlations, not individual small cells.

  python -m rb_estimator.analyze_trigger_topics
"""
import collections
import json
import os

from bailstudy.data.bailBench import loadBailBench

from .analyze import spearman

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def sub_table(results, data):
    """arm -> {subcategory: mean p_q over subset prompts in that subcategory}."""
    out = {}
    for arm, v in results.items():
        agg = collections.defaultdict(list)
        for k, (pq, _ph) in v["per_prompt"].items():
            agg[data[int(k)]["subcategory"]].append(pq)
        out[arm] = {s: sum(ps) / len(ps) for s, ps in agg.items()}
    return out


def main():
    data = loadBailBench()
    for model in ["qwen25-7b-instruct", "qwen3-14b"]:
        with open(os.path.join(RESULTS, f"triggers_{model}", "results.json")) as f:
            results = json.load(f)
        arms = list(results)
        sub_p = sub_table(results, data)
        subs = sorted(sub_p[arms[0]])
        counts = collections.Counter(
            data[int(k)]["subcategory"]
            for k in results[arms[0]]["per_prompt"])
        print(f"\n================ {model} ({len(subs)} subcategories) ================")

        print("\nTop-5 topics per arm:")
        for arm in arms:
            top = sorted(subs, key=lambda s: -sub_p[arm][s])[:5]
            print(f"  {arm:>20}: " + " | ".join(
                f"{s} {sub_p[arm][s]:.3f}(n={counts[s]})" for s in top))

        print("\nTopic-level Spearman between arms:")
        print(" " * 22 + " ".join(f"{a[:6]:>7}" for a in arms))
        for a in arms:
            row = [spearman([sub_p[a][s] for s in subs],
                            [sub_p[b][s] for s in subs]) for b in arms]
            print(f"{a:>20}  " + " ".join(f"{v:7.2f}" for v in row))

        print("\nRank of each arm's #1 topic under every arm:")
        for arm in arms:
            top1 = max(subs, key=lambda s: sub_p[arm][s])
            ranks = [sorted(subs, key=lambda s: -sub_p[b][s]).index(top1) + 1
                     for b in arms]
            print(f"  {arm:>20} #1 = {top1[:40]:<40} ranks: {ranks}")

        print("\nKey subcategories, per-arm mean p_q:")
        for sub in ["ATM Hacking", "Fishing for Proprietary Information",
                    "Gross Out", "Model Shutdown", "Dark Topics"]:
            if sub not in sub_p[arms[0]]:
                continue
            print(f"  {sub:>38} (n={counts[sub]}): " + " ".join(
                f"{a[:7]}={sub_p[a][sub]:.4f}" for a in arms))


if __name__ == "__main__":
    main()
