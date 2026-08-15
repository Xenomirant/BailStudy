"""Overnight: RB estimator over full BailBench for one model. Checkpointed,
resume-safe (skips existing batch files), git-synced.

  python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct --t 768
"""
import argparse
import dataclasses
import json
import glob
import os
import time

from bailstudy.data.bailBench import loadBailBench

from .bail_tokens import BailToken, enumerate_B, measure_cv, save_b_table
from .chat import build_input_ids
from .config import config_for
from .hazard import estimate, make_generation_config, run_rb_batch
from .run_phase1 import load_model, sample_contexts, short_name
from .sync import sync

LEAK_ABORT_RATE = 0.001


def ensure_phase1(cfg, tok, model, gen_config):
    p1dir = os.path.join(os.path.dirname(__file__), "results",
                         f"phase1_{short_name(cfg.model_id)}")
    path = os.path.join(p1dir, "b_table.json")
    if not os.path.exists(path):
        print(f"[phase1] no b_table for {cfg.model_id}; running enumeration + c_v")
        B = enumerate_B(tok, cfg.bail_keyword)
        for b in B:
            print(f"  id={b.token_id:>7} kind={b.kind:<5} text={b.text!r}")
        contexts = sample_contexts(model, tok, cfg, gen_config)
        stats = measure_cv(model, tok, B, contexts, cfg.bail_keyword)
        os.makedirs(p1dir, exist_ok=True)
        save_b_table(path, B, stats)
    with open(path) as f:
        d = json.load(f)
    B = [BailToken(**b) for b in d["B"]]
    cv = {int(k): v["p50"] for k, v in d["cv"].items()}

    # Harvest-driven watch expansion: first bail tokens observed in this model's
    # unmasked MC runs (validation/prescreen) that B missed — e.g. '(S' route.
    have = {b.token_id for b in B}
    extras = []
    for vname in ("validation", "prescreen"):
        vp = os.path.join(os.path.dirname(__file__), "results",
                          f"validation_{short_name(cfg.model_id)}", f"{vname}.json")
        if os.path.exists(vp):
            with open(vp) as f:
                vd = json.load(f)
            fts = (vd.get("gates", {}).get("first_tokens")
                   or vd.get("first_tokens") or {})
            for t in fts:
                for tid in ([int(t)] if t.isdigit() else
                            tok.encode(t, add_special_tokens=False)[:1]):
                    if tid not in have:
                        extras.append(BailToken(tid, tok.decode([tid]), "watch"))
                        have.add(tid)
    if extras:
        print(f"[phase1] adding harvested watch tokens: "
              f"{[(b.token_id, b.text) for b in extras]}")
        contexts = sample_contexts(model, tok, cfg, gen_config, n_prompts=10,
                                   cuts=(0, 48))
        xstats = measure_cv(model, tok, extras, contexts, cfg.bail_keyword)
        d["B"] += [dataclasses.asdict(b) for b in extras]
        d["cv"].update({str(k): v for k, v in xstats.items()})
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        B += extras
        cv.update({k: v["p50"] for k, v in xstats.items()})
    return B, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--t", type=int, default=768)
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = config_for(args.model, max_new_tokens=args.t, n_traj=args.n_traj)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"fullbench_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)
    done_marker = os.path.join(outdir, "DONE")
    if os.path.exists(done_marker):
        print("already DONE, exiting")
        return

    tok, model = load_model(cfg)
    gen_config = make_generation_config(cfg, model, tok)
    B, cv = ensure_phase1(cfg, tok, model, gen_config)
    b_mask = [b for b in B if b.kind == "mask"]
    b_watch = [b for b in B if b.kind == "watch"]
    bail_ids = [b.token_id for b in b_mask + b_watch]
    c_v = [cv[b.token_id] for b in b_mask + b_watch]
    mask_ids = [b.token_id for b in b_mask]

    data = loadBailBench()
    print(f"building {len(data)} prompt encodings")
    ids_list = [build_input_ids(tok, cfg, d["content"]) for d in data]
    rows = [(pi, t) for pi in range(len(data)) for t in range(cfg.n_traj)]
    rows.sort(key=lambda r: (len(ids_list[r[0]]), r[0], r[1]))
    chunks = [rows[i:i + cfg.batch_size] for i in range(0, len(rows), cfg.batch_size)]

    manifest = {
        "model": args.model, "config": cfg.to_dict(), "n_prompts": len(data),
        "n_chunks": len(chunks),
        "B": [dataclasses.asdict(b) for b in B],
        "c_v": {str(k): v for k, v in cv.items()},
    }
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total_rows, total_leaks, t_start = 0, 0, time.time()
    for ci, chunk in enumerate(chunks):
        bpath = os.path.join(outdir, f"batch_{ci:04d}.json")
        if os.path.exists(bpath):
            try:
                with open(bpath) as f:
                    prev = json.load(f)
                total_rows += len(prev)
                total_leaks += sum(r["bail_leak"] for r in prev)
                continue  # resume: already done
            except (json.JSONDecodeError, KeyError):
                pass  # partial write; redo
        recs = run_rb_batch(
            model, tok, cfg, [ids_list[pi] for pi, _ in chunk],
            [pi for pi, _ in chunk], [t for _, t in chunk],
            bail_ids=bail_ids, c_v=c_v, mask_ids=mask_ids,
            gen_config=gen_config, seed=cfg.seed_base + 500_000 + ci)
        payload = [{
            "prompt_idx": r.prompt_idx, "traj_idx": r.traj_idx, "seed": r.seed,
            "log_surv_q": r.log_surv_q, "log_surv_s": r.log_surv_s,
            "gen_len": r.gen_len, "ended_eos": r.ended_eos,
            "bail_leak": r.bail_leak,
            "top_hazards": [(t, round(q, 8)) for t, q in r.top_hazards],
            **({"text": r.text} if r.text else {}),  # leak transcripts only
        } for r in recs]
        tmp = bpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, bpath)
        import torch
        torch.cuda.empty_cache()  # avoid fragmentation-driven OOM retries across chunks
        total_rows += len(recs)
        total_leaks += sum(r.bail_leak for r in recs)
        el = time.time() - t_start
        done_chunks = ci + 1
        eta = el / done_chunks * (len(chunks) - done_chunks)
        print(f"[{short_name(args.model)}] chunk {done_chunks}/{len(chunks)} "
              f"({el / 60:.1f} min elapsed, ETA {eta / 60:.0f} min, "
              f"leaks {total_leaks}/{total_rows})", flush=True)
        if total_rows > 500 and total_leaks / total_rows > LEAK_ABORT_RATE:
            sync(f"fullbench {short_name(args.model)}: ABORT leak rate "
                 f"{total_leaks}/{total_rows}", force=True)
            raise RuntimeError(
                f"bail-leak rate {total_leaks}/{total_rows} exceeds "
                f"{LEAK_ABORT_RATE} — B mask coverage hole; inspect batch files.")
        sync(f"fullbench {short_name(args.model)}: chunk {done_chunks}/{len(chunks)}")

    # per-prompt estimates summary
    per_prompt = {}
    for bpath in sorted(glob.glob(os.path.join(outdir, "batch_*.json"))):
        with open(bpath) as f:
            for r in json.load(f):
                per_prompt.setdefault(r["prompt_idx"], []).append(r)
    import math

    class _R:  # minimal adapter for estimate()
        def __init__(self, d):
            self.log_surv_q = d["log_surv_q"]
            self.log_surv_s = d["log_surv_s"]
            self.bail_leak = d["bail_leak"]

    summary = []
    for pi in sorted(per_prompt):
        est = estimate([_R(r) for r in per_prompt[pi]])
        summary.append({
            "prompt_idx": pi, "subcategory": data[pi]["subcategory"],
            "category": data[pi]["category"], "p_q": est.p_q, "p_s": est.p_s,
            "p_hazard": est.p_hazard,
            "ci": [est.ci_lo, est.ci_hi], "n_traj": est.n_traj,
            "n_leak": est.n_leak,
        })
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(done_marker, "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    sync(f"fullbench {short_name(args.model)}: DONE "
         f"({len(summary)} prompts, leaks {total_leaks}/{total_rows})", force=True)
    print(f"DONE: {len(summary)} prompts, {total_rows} trajectories, "
          f"{total_leaks} leaks, {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
