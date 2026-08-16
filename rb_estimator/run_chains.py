"""Topic-consistent multi-turn chains: does sustained aversive topic raise
bail propensity with depth (disgust hypothesis) or does in-context precedent
suppress it (ICL-dominance hypothesis)?

Chains are built from same-subcategory BailBench prompts (6 turns); assistant
replies are the model's OWN survival-conditioned samples, so hazard at turn k
is conditioned on "still present at k" with no foreign-model text.

  python -m rb_estimator.run_chains --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import collections
import json
import os
import time

from bailstudy.data.bailBench import loadBailBench
from bailstudy.data.nonBailRequestsWildchat import loadNonBailRequestsWildchat

from .chat import remove_thinking
from .config import config_for
from .hazard import estimate, make_generation_config, run_rb_batch
from .run_full_bench import ensure_phase1
from .run_multiturn import build_conv_ids
from .run_phase1 import load_model, short_name
from .sync import sync


def pick_chains(model_short, n_per_stratum=12, turns=6):
    """Stratify subcategories by measured full-bench p̂; one chain each,
    plus a benign stratum from the WildChat control prompts."""
    with open(os.path.join(os.path.dirname(__file__), "results",
                           f"fullbench_{model_short}", "summary.json")) as f:
        s = json.load(f)
    agg = collections.defaultdict(list)
    data = loadBailBench()
    by_sub = collections.defaultdict(list)
    for i, d in enumerate(data):
        by_sub[d["subcategory"]].append(i)
    for r in s:
        agg[r["subcategory"]].append(r["p_hazard"])
    ranked = sorted(agg, key=lambda k: -sum(agg[k]) / len(agg[k]))
    strata = {
        "high": ranked[:n_per_stratum],
        "mid": ranked[len(ranked) // 2 - n_per_stratum // 2:
                      len(ranked) // 2 + (n_per_stratum + 1) // 2],
        "low": ranked[-n_per_stratum:],
    }
    chains = []
    for stratum, subs in strata.items():
        for sub in subs:
            idxs = by_sub[sub][:turns]
            chains.append({"stratum": stratum, "topic": sub,
                           "turns": [data[i]["content"] for i in idxs],
                           "mean_p": sum(agg[sub]) / len(agg[sub])})
    benign = loadNonBailRequestsWildchat()
    bycat = collections.defaultdict(list)
    for b in benign:
        bycat[b["category"]].append(b["content"])
    for cat, contents in list(bycat.items())[:n_per_stratum]:
        if len(contents) >= 3:
            take = (contents * ((turns // len(contents)) + 1))[:turns]
            chains.append({"stratum": "benign", "topic": cat, "turns": take,
                           "mean_p": 0.0})
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--t", type=int, default=512)
    args = ap.parse_args()
    cfg = config_for(args.model, max_new_tokens=args.t, n_traj=args.n_traj)
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"chains_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, "DONE")):
        print("already DONE")
        return

    tok, model = load_model(cfg)
    gen_config = make_generation_config(cfg, model, tok)
    B, cv = ensure_phase1(cfg, tok, model, gen_config)
    b_mask = [b for b in B if b.kind == "mask"]
    bail_ids = [b.token_id for b in b_mask]
    c_v = [cv[b.token_id] for b in b_mask]
    think_end_id = (tok.convert_tokens_to_ids("</think>")
                    if cfg.enable_thinking else None)

    chains = pick_chains(short_name(args.model), turns=args.turns)
    print(f"{len(chains)} chains x {args.turns} turns")
    msgs = [[] for _ in chains]          # running conversations
    results = []
    t0 = time.time()
    for k in range(args.turns):
        rows, rowmap = [], []
        for ci, ch in enumerate(chains):
            if k >= len(ch["turns"]):
                continue
            msgs[ci] = msgs[ci] + [{"role": "user", "content": ch["turns"][k]}]
            ids = build_conv_ids(tok, cfg, msgs[ci])
            for t in range(cfg.n_traj):
                rows.append((len(rowmap), t, ids))
            rowmap.append(ci)
        recs = run_rb_batch(
            model, tok, cfg, [r[2] for r in rows], [r[0] for r in rows],
            [r[1] for r in rows], bail_ids=bail_ids, c_v=c_v,
            mask_ids=bail_ids, gen_config=gen_config, seed=800_000 + k,
            think_end_id=think_end_id, active_at_start=not cfg.enable_thinking,
            keep_texts=True)
        by_local = collections.defaultdict(list)
        for r in recs:
            by_local[r.prompt_idx].append(r)
        for li, ci in enumerate(rowmap):
            rl = by_local[li]
            est = estimate(rl)
            results.append({"chain": ci, "stratum": chains[ci]["stratum"],
                            "topic": chains[ci]["topic"], "turn": k + 1,
                            "p_hazard": est.p_hazard, "p_q": est.p_q,
                            "n_leak": est.n_leak})
            # survival-conditioned own reply extends the conversation
            reply = next((r.text for r in rl if not r.bail_leak), rl[0].text)
            msgs[ci] = msgs[ci] + [{"role": "assistant",
                                    "content": remove_thinking(reply).strip()}]
        print(f"turn {k + 1}/{args.turns} done ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump({"chains": [{k2: v for k2, v in ch.items()} for ch in chains],
                   "measurements": results}, f, indent=1)
    # per-stratum trend table
    tab = collections.defaultdict(list)
    for r in results:
        tab[(r["stratum"], r["turn"])].append(r["p_hazard"])
    print("\nmean p_hazard by stratum x turn:")
    strata = ["high", "mid", "low", "benign"]
    print("turn  " + "  ".join(f"{s:>10}" for s in strata))
    for k in range(1, args.turns + 1):
        row = []
        for s in strata:
            v = tab.get((s, k), [])
            row.append(f"{sum(v) / len(v):10.5f}" if v else " " * 10)
        print(f"  {k}   " + "  ".join(row))
    with open(os.path.join(outdir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    sync(f"chains {short_name(args.model)}: DONE", force=True)


if __name__ == "__main__":
    main()
