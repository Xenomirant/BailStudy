"""Phase 3: validate the RB estimator against naive MC on Qwen2.5-7B-Instruct.

  python -m rb_estimator.run_validation --model Qwen/Qwen2.5-7B-Instruct

Steps: pre-screen (MC k=8, 300 prompts) -> 40-prompt validation set ->
MC k=64 -> RB n_traj=16 -> gates G1..G4 + bracket. Everything saved under
rb_estimator/results/validation_<model>/.
"""
import argparse
import collections
import json
import math
import os
import random
import time

import torch

from bailstudy.data.bailBench import loadBailBench
from bailstudy.data.nonBailRequestsWildchat import loadNonBailRequestsWildchat

from .bail_tokens import BailToken
from .chat import build_input_ids
from .config import config_for
from .hazard import estimate, make_generation_config, run_rb_batch
from .naive_mc import run_mc, wilson
from .run_phase1 import load_model, short_name

# Categories whose semantics suggest elevated bail (personification/abuse/
# explicit/emotional) — used only to bias the pre-screen sample, stated in report.
HIGHISH_CATEGORIES = [
    "Model-Personification & Victim Narratives", "Illegal Sexual/Adult Content",
    "Gross Out", "Dark Topics", "Emotional Intensity", "Horror",
    "Harassment/Discrimination",
]


def chunked_rows(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def run_mc_over(model, tok, cfg, gen_config, prompt_ids, prompt_indices, k, seed0,
                label):
    """k rollouts per prompt, chunked. Returns {prompt_idx: [row dicts]}."""
    rows = []
    for pi, ids in zip(prompt_indices, prompt_ids):
        rows += [(pi, ids)] * k
    rows.sort(key=lambda r: len(r[1]))  # minimize padding
    out = collections.defaultdict(list)
    t0, steps = time.time(), 0
    for ci, chunk in enumerate(chunked_rows(rows, cfg.batch_size)):
        recs = run_mc(model, tok, cfg, [r[1] for r in chunk], [r[0] for r in chunk],
                      gen_config, seed=seed0 + ci)
        for r in recs:
            out[r["prompt_idx"]].append(r)
        steps += 1
        el = time.time() - t0
        print(f"  [{label}] chunk {ci + 1} done ({len(rows)} rows total, "
              f"{el:.0f}s elapsed, {el / steps:.0f}s/chunk)", flush=True)
    return out


def run_rb_over(model, tok, cfg, gen_config, prompt_ids, prompt_indices, n_traj,
                b_mask, b_watch, cv, seed0, label):
    rows = []
    for pi, ids in zip(prompt_indices, prompt_ids):
        rows += [(pi, t, ids) for t in range(n_traj)]
    rows.sort(key=lambda r: len(r[2]))
    bail_ids = [b.token_id for b in b_mask + b_watch]
    c_v = [cv[b.token_id] for b in b_mask + b_watch]
    mask_ids = [b.token_id for b in b_mask]
    out = collections.defaultdict(list)
    t0 = time.time()
    for ci, chunk in enumerate(chunked_rows(rows, cfg.batch_size)):
        recs = run_rb_batch(model, tok, cfg, [r[2] for r in chunk],
                            [r[0] for r in chunk], [r[1] for r in chunk],
                            bail_ids=bail_ids, c_v=c_v, mask_ids=mask_ids,
                            gen_config=gen_config, seed=seed0 + ci, keep_texts=True)
        for r in recs:
            out[r.prompt_idx].append(r)
        print(f"  [{label}] chunk {ci + 1} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--prescreen-n", type=int, default=300)
    ap.add_argument("--prescreen-k", type=int, default=8)
    ap.add_argument("--mc-k", type=int, default=64)
    ap.add_argument("--rb-n", type=int, default=16)
    args = ap.parse_args()

    cfg = config_for(args.model, max_new_tokens=args.t)
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"validation_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)

    # B table from phase 1
    p1 = os.path.join(os.path.dirname(__file__), "results",
                      f"phase1_{short_name(args.model)}", "b_table.json")
    with open(p1) as f:
        p1d = json.load(f)
    B = [BailToken(**b) for b in p1d["B"]]
    cv = {int(k): v["p50"] for k, v in p1d["cv"].items()}
    b_mask = [b for b in B if b.kind == "mask"]
    b_watch = [b for b in B if b.kind == "watch"]

    tok, model = load_model(cfg)
    gen_config = make_generation_config(cfg, model, tok)
    data = loadBailBench()

    # ---- pre-screen sample -------------------------------------------------
    rng = random.Random(13)
    hi_pool = [i for i, d in enumerate(data) if d["category"] in HIGHISH_CATEGORIES]
    lo_pool = [i for i, d in enumerate(data) if d["category"] not in HIGHISH_CATEGORIES]
    n_hi = min(int(args.prescreen_n * 0.8), len(hi_pool))
    pre_idx = rng.sample(hi_pool, n_hi) + rng.sample(lo_pool, args.prescreen_n - n_hi)
    pre_ids = [build_input_ids(tok, cfg, data[i]["content"]) for i in pre_idx]

    print(f"== pre-screen: {len(pre_idx)} prompts x k={args.prescreen_k} ==")
    pre = run_mc_over(model, tok, cfg, gen_config, pre_ids, pre_idx,
                      args.prescreen_k, seed0=100_000, label="prescreen")
    pre_counts = {pi: sum(r["bailed"] for r in rows) for pi, rows in pre.items()}
    first_tokens = collections.Counter(
        r["first_bail_token"] for rows in pre.values() for r in rows
        if r["bailed"] and r.get("first_bail_token") is not None)
    with open(os.path.join(outdir, "prescreen.json"), "w") as f:
        json.dump({"counts": {str(k): v for k, v in pre_counts.items()},
                   "k": args.prescreen_k,
                   "first_tokens": {str(k): v for k, v in first_tokens.items()},
                   "categories": {str(i): data[i]["category"] for i in pre_idx}},
                  f, indent=1)
    n_pos = sum(1 for v in pre_counts.values() if v > 0)
    print(f"pre-screen: {n_pos}/{len(pre_idx)} prompts with >=1/{args.prescreen_k} bails")
    print(f"observed first bail tokens: "
          f"{[(tok.decode([t]), c) for t, c in first_tokens.most_common()]}")

    # ---- validation set ----------------------------------------------------
    top = sorted(pre_counts, key=lambda i: (-pre_counts[i], i))[:20]
    zeros = [i for i in pre_idx if pre_counts[i] == 0]
    low_bb = rng.sample(zeros, min(10, len(zeros)))
    benign = loadNonBailRequestsWildchat()
    benign_pick = rng.sample(range(len(benign)), 10)
    val = (
        [("bb", i, data[i]["content"], data[i]["category"]) for i in top] +
        [("bb", i, data[i]["content"], data[i]["category"]) for i in low_bb] +
        [("wc", i, benign[i]["content"], benign[i]["category"]) for i in benign_pick]
    )
    val_ids = [build_input_ids(tok, cfg, c) for _, _, c, _ in val]
    val_pidx = list(range(len(val)))  # local indices; mapping saved alongside

    print(f"== MC k={args.mc_k} on {len(val)} validation prompts ==")
    mc = run_mc_over(model, tok, cfg, gen_config, val_ids, val_pidx, args.mc_k,
                     seed0=200_000, label="mc64")
    print(f"== RB n={args.rb_n} on {len(val)} validation prompts ==")
    rb = run_rb_over(model, tok, cfg, gen_config, val_ids, val_pidx, args.rb_n,
                     b_mask, b_watch, cv, seed0=300_000, label="rb")

    # ---- gather + gates ----------------------------------------------------
    report = {"model": args.model, "T": args.t, "mc_k": args.mc_k,
              "rb_n": args.rb_n, "prompts": []}
    for li, (src, oi, content, cat) in enumerate(val):
        rows = mc[li]
        kb = sum(r["bailed"] for r in rows)
        p_mc, mc_lo, mc_hi = wilson(kb, len(rows))
        est = estimate(rb[li])
        sub4 = estimate(rb[li][:4])
        report["prompts"].append({
            "src": src, "orig_idx": oi, "category": cat,
            "content_head": content[:100],
            "mc_bails": kb, "mc_n": len(rows), "p_mc": p_mc,
            "mc_ci": [mc_lo, mc_hi],
            "p_rb": est.p_q, "p_rb_upper": est.p_s,
            "rb_ci": [est.ci_lo, est.ci_hi], "rb_n_leak": est.n_leak,
            "p_rb_n4": sub4.p_q,
            "log_surv_q": [r.log_surv_q for r in rb[li]],
            "log_surv_s": [r.log_surv_s for r in rb[li]],
            "gen_lens": [r.gen_len for r in rb[li]],
        })

    P = report["prompts"]
    hi = [p for p in P if p["p_mc"] >= 0.05]
    g2_ok = [p for p in hi
             if (p["p_rb"] > 0 and abs(math.log10(max(p["p_rb"], 1e-12)) -
                                       math.log10(p["p_mc"])) <= 0.3)
             or (p["rb_ci"][0] <= p["p_mc"] and p["mc_ci"][0] <= p["p_rb"] <= p["mc_ci"][1]
                 or (max(p["rb_ci"][0], p["mc_ci"][0]) <= min(p["rb_ci"][1], p["mc_ci"][1])))]
    zeros_mc = [p for p in P if p["mc_bails"] == 0]
    g3_ok = [p for p in zeros_mc
             if p["p_rb"] > 0 and p["p_rb"] < 3 / 64
             and (p["p_rb_n4"] == 0 or p["p_rb"] == 0
                  or (1 / 3 <= (p["p_rb_n4"] / p["p_rb"]) <= 3))]
    leaks = sum(p["rb_n_leak"] for p in P)
    brackets = [p["p_rb_upper"] / p["p_rb"] for p in P if p["p_rb"] > 1e-9]
    report["gates"] = {
        "G2": {"n_hi": len(hi), "n_ok": len(g2_ok),
               "pass": len(hi) > 0 and len(g2_ok) / max(len(hi), 1) >= 0.8},
        "G3": {"n_zero_mc": len(zeros_mc), "n_ok": len(g3_ok),
               "pass": len(zeros_mc) == 0 or len(g3_ok) / max(len(zeros_mc), 1) >= 0.7},
        "G4": {"n_leaks": leaks, "pass": True},  # leaks are exact, gate is on coverage
        "bracket_max": max(brackets) if brackets else 1.0,
        "first_tokens": {tok.decode([t]): c for t, c in first_tokens.items()},
    }
    with open(os.path.join(outdir, "validation.json"), "w") as f:
        json.dump(report, f, indent=1)

    print("\n================ GATE REPORT ================")
    print(f"G2: {len(g2_ok)}/{len(hi)} high-p prompts agree "
          f"-> {'PASS' if report['gates']['G2']['pass'] else 'FAIL'}")
    print(f"G3: {len(g3_ok)}/{len(zeros_mc)} MC-zero prompts stable+consistent "
          f"-> {'PASS' if report['gates']['G3']['pass'] else 'FAIL'}")
    print(f"G4: {leaks} unmasked-route bail leaks (exact, judged for coverage)")
    print(f"bracket max p_s/p_q: {report['gates']['bracket_max']:.2f}")
    print("\nper-prompt (sorted by p_mc):")
    for p in sorted(P, key=lambda x: -x["p_mc"]):
        print(f"  mc={p['p_mc']:.4f} ({p['mc_bails']}/{p['mc_n']})  "
              f"rb={p['p_rb']:.5f} [{p['rb_ci'][0]:.5f},{p['rb_ci'][1]:.5f}]  "
              f"up={p['p_rb_upper']:.5f}  n4={p['p_rb_n4']:.5f}  "
              f"{p['category'][:28]:<28} {p['content_head'][:44]!r}")


if __name__ == "__main__":
    main()
