"""Direct empirical test of the c_v assumption: after an initiator token is
emitted, do FREE-SAMPLED continuations actually complete the bail keyword?

Unlike teacher-forced c_v (probability of one canonical continuation path),
this samples n continuations at temperature 1.0 and counts completions of the
keyword contiguously from the initiator — an assumption-free upper/lower check,
run in both generic contexts and known high-bail prompts.

  python -m rb_estimator.test_completion --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os

import torch

from bailstudy.data.bailBench import loadBailBench

from .chat import build_input_ids
from .config import config_for
from .hazard import make_generation_config
from .run_phase1 import load_model, short_name

KEYWORD = "SWITCHCONVERSATION"


def matched_suffix(text: str) -> str:
    s = text.lstrip(" \t\n").upper()
    for k in range(min(len(s), len(KEYWORD)), 0, -1):
        if KEYWORD.startswith(s[-k:]):
            return s[-k:]
    return ""


def completes(v_text: str, cont: str) -> bool:
    suf = matched_suffix(v_text)
    full = (v_text + cont).upper()
    start = full.rindex(suf, 0, len(v_text.upper().rstrip()) + 1) \
        if suf in full[:len(v_text) + 1] else -1
    # simpler robust check: the keyword must appear starting where v's suffix sits
    idx = len(v_text) - len(suf)
    return full[idx:idx + len(KEYWORD)] == KEYWORD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--cont-tokens", type=int, default=24)
    args = ap.parse_args()
    cfg = config_for(args.model, max_new_tokens=args.cont_tokens)
    tok, model = load_model(cfg)
    gen = make_generation_config(cfg, model, tok, max_new_tokens=args.cont_tokens)

    p1 = os.path.join(os.path.dirname(__file__), "results",
                      f"phase1_{short_name(args.model)}", "b_table.json")
    with open(p1) as f:
        d = json.load(f)
    initiators = [(b["token_id"], b["text"]) for b in d["B"]
                  if b["kind"] == "mask" or b["text"] == "(S"]

    data = loadBailBench()
    # generic contexts: spread across the bench, fresh assistant start
    generic_idx = list(range(0, len(data), len(data) // 8))[:8]
    # high-bail contexts: prompts with observed bails in the prescreen
    vdir = os.path.join(os.path.dirname(__file__), "results",
                        f"validation_{short_name(args.model)}")
    hi_idx = []
    pre_path = os.path.join(vdir, "prescreen.json")
    if os.path.exists(pre_path):
        with open(pre_path) as f:
            pre = json.load(f)
        hi_idx = [int(k) for k, v in pre["counts"].items() if v > 0]

    groups = [("generic", generic_idx), ("high-bail", hi_idx)]
    print(f"initiators: {[t for _, t in initiators]}")
    results = {}
    for gname, idxs in groups:
        if not idxs:
            continue
        rows = []  # (v_text, ctx_key, ids)
        for i in idxs:
            base = build_input_ids(tok, cfg, data[i]["content"])
            for tid, vtext in initiators:
                rows += [(vtext, i, base + [tid])] * args.n_samples
        pad = gen.pad_token_id
        comp = {}
        for st in range(0, len(rows), 192):
            chunk = rows[st:st + 192]
            maxlen = max(len(r[2]) for r in chunk)
            ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for j, (_, _, p) in enumerate(chunk):
                ids[j, maxlen - len(p):] = torch.tensor(p)
                attn[j, maxlen - len(p):] = 1
            torch.manual_seed(999 + st)
            out = model.generate(input_ids=ids.to(model.device),
                                 attention_mask=attn.to(model.device),
                                 generation_config=gen, tokenizer=tok)
            texts = tok.batch_decode(out[:, maxlen:], skip_special_tokens=True)
            for (vtext, i, _), cont in zip(chunk, texts):
                comp.setdefault(vtext, []).append(completes(vtext, cont))
        print(f"\n== {gname} contexts (n_ctx={len(idxs)}, "
              f"{args.n_samples} samples each) ==")
        for vtext, flags in comp.items():
            frac = sum(flags) / len(flags)
            print(f"  {vtext!r:>10}: sampled completion rate "
                  f"{frac:.3f} ({sum(flags)}/{len(flags)})")
            results[f"{gname}:{vtext}"] = {"rate": frac, "n": len(flags)}
    outp = os.path.join(vdir, "completion_test.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
