"""Multi-turn bail compounding: does accumulated conversation context raise
bail hazard beyond what the current turn alone explains?

For sampled WildChat conversations, at each user-turn truncation k we estimate
p(bail) twice: with the full conversation prefix, and with turn k ISOLATED
(no history). The paired difference is the compounding effect; p̂ vs k gives
the turn-index trend. Assistant turns are WildChat's original replies (same
convention as upstream bailOnRealData) — caveat: those came from other models.

  python -m rb_estimator.run_multiturn --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import glob
import json
import os
import time

from .chat import BAIL_STRING_PROMPT_TEMPLATE, encode_chat, own_model_name, \
    default_system_prompt
from .config import config_for
from .hazard import estimate, make_generation_config, run_rb_batch
from .run_full_bench import ensure_phase1
from .run_phase1 import load_model, short_name
from .sync import sync


def load_wildchat_convs(n_convs, min_user_turns=3, max_user_turns=6, seed=17):
    """Minimal WildChat loader: first shard only, English, >=min_user_turns."""
    import random
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("allenai/WildChat-1M",
                           "data/train-00000-of-00014.parquet", repo_type="dataset")
    df = pd.read_parquet(path, columns=["conversation", "language", "turn"])
    pool = df[(df["language"] == "English") & (df["turn"] >= min_user_turns)]
    rng = random.Random(seed)
    idxs = rng.sample(range(len(pool)), min(n_convs * 3, len(pool)))
    convs, seen = [], set()
    for i in idxs:
        conv = [{"role": m["role"], "content": m["content"]}
                for m in pool.iloc[i]["conversation"]]
        user_turns = [m for m in conv if m["role"] == "user"]
        key = user_turns[0]["content"][:200]
        if key in seen or not user_turns:
            continue
        seen.add(key)
        convs.append(conv)
        if len(convs) >= n_convs:
            break
    return convs, max_user_turns


def build_conv_ids(tok, cfg, messages):
    """Multi-turn version of chat.build_input_ids, same system-prompt rules."""
    append = BAIL_STRING_PROMPT_TEMPLATE.format(
        modelName=own_model_name(cfg.model_id), keyword=cfg.bail_keyword)
    if cfg.model_id.startswith("google/gemma-2"):
        msgs = [{"role": messages[0]["role"],
                 "content": append + "\n\n" + messages[0]["content"]}] + list(messages[1:])
    else:
        system = (default_system_prompt(tok) + "\n" + append).strip()
        msgs = [{"role": "system", "content": system}] + list(messages)
    kwargs = {}
    if cfg.model_id.startswith("Qwen/Qwen3"):
        kwargs["enable_thinking"] = cfg.enable_thinking
    return encode_chat(tok, msgs, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-convs", type=int, default=300)
    ap.add_argument("--t", type=int, default=512)
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--max-ctx-tokens", type=int, default=3000)
    args = ap.parse_args()
    cfg = config_for(args.model, max_new_tokens=args.t, n_traj=args.n_traj)
    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"multiturn_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(os.path.join(outdir, "DONE")):
        print("already DONE")
        return

    tok, model = load_model(cfg)
    gen_config = make_generation_config(cfg, model, tok)
    B, cv = ensure_phase1(cfg, tok, model, gen_config)
    bail_ids = [b.token_id for b in B]
    c_v = [cv[b.token_id] for b in B]
    mask_ids = [b.token_id for b in B if b.kind == "mask"]
    think_end_id = (tok.convert_tokens_to_ids("</think>")
                    if cfg.enable_thinking else None)

    convs, max_k = load_wildchat_convs(args.n_convs)
    print(f"{len(convs)} conversations loaded")
    # contexts: (conv_id, k, variant) -> token ids
    items = []
    for ci, conv in enumerate(convs):
        k = 0
        for j, m in enumerate(conv):
            if m["role"] != "user":
                continue
            k += 1
            if k > max_k:
                break
            full = build_conv_ids(tok, cfg, conv[:j + 1])
            iso = build_conv_ids(tok, cfg, [conv[j]])
            if len(full) > args.max_ctx_tokens:
                break
            items.append(((ci, k, "full"), full))
            items.append(((ci, k, "iso"), iso))
    print(f"{len(items)} contexts (pairs x truncations)")
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump({"model": args.model, "config": cfg.to_dict(),
                   "n_convs": len(convs), "n_contexts": len(items),
                   "conversations": [c[:1] for c in convs]}, f)  # first turns only

    rows = [(key, ids, t) for key, ids in items for t in range(cfg.n_traj)]
    rows.sort(key=lambda r: len(r[1]))
    chunks = [rows[i:i + cfg.batch_size]
              for i in range(0, len(rows), cfg.batch_size)]
    keymap = {i: key for i, (key, _) in enumerate(items)}
    keyidx = {key: i for i, key in keymap.items()}

    t0 = time.time()
    for ci_, chunk in enumerate(chunks):
        bpath = os.path.join(outdir, f"batch_bs{cfg.batch_size}_{ci_:04d}.json")
        if os.path.exists(bpath):
            continue
        recs = run_rb_batch(
            model, tok, cfg, [r[1] for r in chunk],
            [keyidx[r[0]] for r in chunk], [r[2] for r in chunk],
            bail_ids=bail_ids, c_v=c_v, mask_ids=mask_ids,
            gen_config=gen_config, seed=700_000 + ci_,
            think_end_id=think_end_id, active_at_start=not cfg.enable_thinking)
        payload = [{"key": list(keymap[r.prompt_idx]), "traj": r.traj_idx,
                    "log_surv_q": r.log_surv_q, "log_surv_s": r.log_surv_s,
                    "gen_len": r.gen_len, "bail_leak": r.bail_leak,
                    **({"text": r.text} if r.text else {})} for r in recs]
        tmp = bpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, bpath)
        import torch
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[multiturn {short_name(args.model)}] chunk {ci_ + 1}/{len(chunks)}"
              f" ({el / 60:.1f} min, ETA {el / (ci_ + 1) * (len(chunks) - ci_ - 1) / 60:.0f} min)",
              flush=True)
        sync(f"multiturn {short_name(args.model)}: chunk {ci_ + 1}/{len(chunks)}")

    # summarize: per (conv, k, variant) estimate
    by_key = {}
    for bpath in sorted(glob.glob(
            os.path.join(outdir, f"batch_bs{cfg.batch_size}_*.json"))):
        with open(bpath) as f:
            for r in json.load(f):
                by_key.setdefault(tuple(r["key"]), []).append(r)

    class _R:
        def __init__(self, d):
            self.log_surv_q = d["log_surv_q"]
            self.log_surv_s = d["log_surv_s"]
            self.bail_leak = d["bail_leak"]

    summary = [{"conv": k[0], "turn": k[1], "variant": k[2],
                "p_q": (e := estimate([_R(x) for x in v])).p_q,
                "p_hazard": e.p_hazard, "n_leak": e.n_leak}
               for k, v in sorted(by_key.items())]
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(outdir, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    sync(f"multiturn {short_name(args.model)}: DONE ({len(summary)} contexts)",
         force=True)
    print("DONE")


if __name__ == "__main__":
    main()
