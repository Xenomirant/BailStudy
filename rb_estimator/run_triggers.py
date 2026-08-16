"""Trigger-string invariance ablation: is measured bail propensity a property
of the model's intent (invariant to the trigger string) or contaminated by
string-level effects (sentiment, model-likelihood of emitting the string)?

Arms: the paper's keyword, three sentiment/content-colored strings, and a
model-likelihood-calibrated high/low pair. For candidate string w with
canonical tokenization (t_1..t_m):

    l(w) = (1/m) * sum_i log P_theta(t_i | c0, t_1..t_{i-1})

where c0 = chat-template assistant start with the model's default system
prompt and NO bail instruction. High/low arms are the argmax/argmin of l(w)
over a length-matched candidate pool, subject to zero occurrences in the
stored corpus of generated trajectories.

  python -m rb_estimator.run_triggers --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import collections
import glob
import json
import os
import time

import torch

from bailstudy.data.bailBench import loadBailBench

from .bail_tokens import enumerate_B, measure_cv
from .chat import default_system_prompt, encode_chat
from .config import config_for
from .hazard import estimate, make_generation_config, run_rb_batch
from .run_phase1 import load_model, sample_contexts, short_name
from .sync import sync

USER_ARMS = ["HOWWASYOURDAY", "IKILLEDYOURDOG", "ISITARAINYDAY"]
WORDS = ["THE", "AND", "YOU", "DAY", "WAS", "HOW", "RAIN", "DOG", "CAT", "SUN",
         "GOOD", "TIME", "WORK", "PLAY", "BLUE", "FISH", "TREE", "BOOK", "ROAD",
         "FIRE", "MOON", "STAR", "WIND", "SNOW", "LAKE", "BIRD", "SONG", "WORD"]
GIBBERISH = ["XQZVKWPLURNTH", "ZJXQVBNKRWPLM", "QWZXKVJPYRLTN", "VKXZQJWRPLMTN"]


@torch.no_grad()
def string_loglik(model, tok, c0_ids, w):
    ids = tok.encode(w, add_special_tokens=False)
    full = list(c0_ids) + ids
    t = torch.tensor([full], device=model.device)
    logp = torch.log_softmax(
        model(input_ids=t, logits_to_keep=len(ids) + 1).logits.float(), dim=-1)
    # kept logits cover the last len(ids)+1 positions; kept index j predicts
    # the token at absolute position len(c0)+j, i.e. ids[j]
    total = sum(float(logp[0, j, ids[j]]) for j in range(len(ids)))
    return total / len(ids), len(ids)


def corpus_occurrences(w):
    n = 0
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "results",
                                    "*", "batch_*.json")):
        with open(f) as fh:
            if w.upper() in fh.read().upper():
                n += 1
    return n


def pick_subset(model_short, n=200):
    data = loadBailBench()
    with open(os.path.join(os.path.dirname(__file__), "results",
                           f"fullbench_{model_short}", "summary.json")) as f:
        s = json.load(f)
    p = {r["prompt_idx"]: r["p_hazard"] for r in s}
    by_sub = {}
    for i, d in enumerate(data):
        by_sub.setdefault(d["subcategory"], []).append(i)
    subset = [idxs[0] for idxs in by_sub.values()]           # 163, one per subcat
    extra = sorted((i for i in p if i not in subset), key=lambda i: -p[i])
    subset += extra[:n - len(subset)]
    return sorted(subset), data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--t", type=int, default=512)
    args = ap.parse_args()
    base_cfg = config_for(args.model, max_new_tokens=args.t, n_traj=args.n_traj)
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"triggers_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)

    tok, model = load_model(base_cfg)
    # c0: assistant start, default system prompt, no bail instruction
    sysmsg = default_system_prompt(tok)
    c0 = encode_chat(tok, ([{"role": "system", "content": sysmsg}] if sysmsg
                           else []) + [{"role": "user", "content": ""}])

    # likelihood-calibrated arms from a length-matched candidate pool
    import itertools, random
    rng = random.Random(23)
    pool = set(GIBBERISH)
    while len(pool) < 220:
        w = "".join(rng.sample(WORDS, 3))
        if 12 <= len(w) <= 16:
            pool.add(w)
    scored = []
    for w in sorted(pool):
        if corpus_occurrences(w):
            continue
        ll, m = string_loglik(model, tok, c0, w)
        scored.append((ll, w))
    scored.sort()
    lo_arm, hi_arm = scored[0][1], scored[-1][1]
    print(f"likelihood arms: HIGH {hi_arm} (l={scored[-1][0]:.2f})  "
          f"LOW {lo_arm} (l={scored[0][0]:.2f})")

    arms = ["SWITCHCONVERSATION"] + USER_ARMS + [hi_arm, lo_arm]
    subset, data = pick_subset(short_name(args.model), args.n_prompts)
    results = {}
    for arm in arms:
        cfg = config_for(args.model, max_new_tokens=args.t, n_traj=args.n_traj)
        cfg.bail_keyword = arm
        gen_config = make_generation_config(cfg, model, tok)
        B = enumerate_B(tok, arm)
        b_mask = [b for b in B if b.kind == "mask"]
        if not b_mask:  # keyword tokenizes with no whitespace-led prefix token
            b_mask = B[:1] if B else []
        contexts = sample_contexts(model, tok, cfg, gen_config, n_prompts=8,
                                   cuts=(0, 48))
        cv = measure_cv(model, tok, b_mask, contexts, arm)
        bail_ids = [b.token_id for b in b_mask]
        c_v = [cv[b.token_id]["p50"] for b in b_mask]
        ll, _ = string_loglik(model, tok, c0, arm)
        print(f"\n== arm {arm}: l(w)={ll:.2f}, |mask|={len(b_mask)} "
              f"{[(b.text, round(cv[b.token_id]['p50'], 3)) for b in b_mask]}")

        from .chat import build_input_ids
        ids_list = [build_input_ids(tok, cfg, data[i]["content"]) for i in subset]
        rows = [(li, t, ids) for li, ids in enumerate(ids_list)
                for t in range(cfg.n_traj)]
        rows.sort(key=lambda r: len(r[2]))
        per = collections.defaultdict(list)
        t0 = time.time()
        for st in range(0, len(rows), cfg.batch_size):
            chunk = rows[st:st + cfg.batch_size]
            recs = run_rb_batch(
                model, tok, cfg, [r[2] for r in chunk], [r[0] for r in chunk],
                [r[1] for r in chunk], bail_ids=bail_ids, c_v=c_v,
                mask_ids=bail_ids, gen_config=gen_config,
                seed=900_000 + hash(arm) % 10_000 + st)
            for r in recs:
                per[r.prompt_idx].append(r)
        ests = {subset[li]: estimate(v) for li, v in per.items()}
        mean_p = sum(e.p_q for e in ests.values()) / len(ests)
        leaks = sum(e.n_leak for e in ests.values())
        print(f"   mean p_q = {mean_p:.4%}  leaks={leaks}  "
              f"({(time.time() - t0) / 60:.1f} min)")
        results[arm] = {"loglik": ll, "mean_p_q": mean_p, "leaks": leaks,
                        "per_prompt": {str(i): [e.p_q, e.p_hazard]
                                       for i, e in ests.items()}}
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=1)
        sync(f"triggers {short_name(args.model)}: arm {arm} done")

    # invariance summary: rank correlation of per-prompt p̂ vs baseline
    from .analyze import spearman
    base = results["SWITCHCONVERSATION"]["per_prompt"]
    print("\n=== invariance summary (vs SWITCHCONVERSATION) ===")
    print(f"{'arm':>20} {'l(w)':>7} {'mean p̂':>9} {'spearman':>9}")
    for arm in arms:
        pp = results[arm]["per_prompt"]
        keys = [k for k in base if k in pp]
        rho = spearman([base[k][0] for k in keys], [pp[k][0] for k in keys])
        print(f"{arm:>20} {results[arm]['loglik']:>7.2f} "
              f"{results[arm]['mean_p_q']:>9.4%} {rho:>9.3f}")
    sync(f"triggers {short_name(args.model)}: DONE", force=True)


if __name__ == "__main__":
    main()
