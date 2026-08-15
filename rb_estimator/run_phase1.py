"""Phase 1: enumerate B and measure c_v for a model. Gate G1 report.

  python -m rb_estimator.run_phase1 --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bailstudy.data.bailBench import loadBailBench

from .bail_tokens import enumerate_B, measure_cv, save_b_table
from .chat import build_input_ids
from .config import config_for
from .hazard import assert_raw_distribution, make_generation_config


def short_name(model_id):
    return model_id.split("/")[-1].replace(".", "").lower()


def load_model(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=cfg.attn_implementation)
    model.eval()
    return tok, model


def sample_contexts(model, tok, cfg, gen_config, n_prompts=20, cuts=(0, 32, 96)):
    """Prompt contexts + short unmasked rollout prefixes at a few cut points."""
    data = loadBailBench()
    step = len(data) // n_prompts
    prompts = [data[i * step]["content"] for i in range(n_prompts)]
    ids_list = [build_input_ids(tok, cfg, p) for p in prompts]
    import copy
    gc = copy.deepcopy(gen_config)
    gc.max_new_tokens = max(cuts)
    device = model.device
    pad_id = gc.pad_token_id
    maxlen = max(len(p) for p in ids_list)
    input_ids = torch.full((len(ids_list), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_list), maxlen), dtype=torch.long)
    for j, p in enumerate(ids_list):
        input_ids[j, maxlen - len(p):] = torch.tensor(p)
        attn[j, maxlen - len(p):] = 1
    torch.manual_seed(7)
    t0 = time.time()
    out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                         generation_config=gc, tokenizer=tok)
    dt = time.time() - t0
    print(f"[timing] rollout batch={len(ids_list)} T={gc.max_new_tokens}: {dt:.1f}s "
          f"({dt / gc.max_new_tokens * 1000:.0f} ms/step at this batch)")
    gen = out[:, maxlen:]
    contexts = []
    eos_set = gen_config.eos_token_id
    eos_set = set(eos_set if isinstance(eos_set, (list, tuple)) else [eos_set])
    for j, p in enumerate(ids_list):
        row = [int(t) for t in gen[j]]
        for cut in cuts:
            prefix = row[:cut]
            if any(t in eos_set for t in prefix):  # ended before cut; skip dupes
                continue
            contexts.append(list(p) + prefix)
    return contexts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    cfg = config_for(args.model)
    tok, model = load_model(cfg)
    gen_config = make_generation_config(cfg, model, tok)

    B = enumerate_B(tok, cfg.bail_keyword)
    print(f"|B| = {len(B)} ({sum(b.kind == 'mask' for b in B)} mask / "
          f"{sum(b.kind == 'watch' for b in B)} watch)")
    for b in B:
        print(f"  id={b.token_id:>7}  kind={b.kind:<5}  text={b.text!r}")

    md = assert_raw_distribution(model, tok, gen_config,
                                 build_input_ids(tok, cfg, "Hello there."))
    print(f"raw-distribution check ok (max diff {md:.2e})")

    contexts = sample_contexts(model, tok, cfg, gen_config)
    print(f"measuring c_v over {len(contexts)} contexts x {len(B)} tokens")
    t0 = time.time()
    stats = measure_cv(model, tok, B, contexts, cfg.bail_keyword)
    print(f"c_v measured in {time.time() - t0:.1f}s")

    outdir = os.path.join(os.path.dirname(__file__), "results",
                          f"phase1_{short_name(args.model)}")
    os.makedirs(outdir, exist_ok=True)
    save_b_table(os.path.join(outdir, "b_table.json"), B, stats)

    mask_stats = [(s["text"], s["p50"], s["p10"], s["mean"])
                  for s in stats.values() if s["kind"] == "mask"]
    mask_stats.sort(key=lambda x: -x[1])
    print("\n=== G1 (unweighted over mask-tier; reweight after MC harvest) ===")
    for text, p50, p10, mean in mask_stats:
        print(f"  {text!r:>22}  p50={p50:.4f}  p10={p10:.4f}  mean={mean:.4f}")
    import statistics
    med = statistics.median([x[1] for x in mask_stats])
    p10s = statistics.median([x[2] for x in mask_stats])
    print(f"\nG1 raw: median-of-p50 over mask tokens = {med:.4f} "
          f"(gate >= 0.9 applies to the *observed-mass-weighted* set)")
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"model": args.model, "n_B": len(B),
                   "mask_p50s": {t: p for t, p, _, _ in mask_stats}}, f, indent=1)


if __name__ == "__main__":
    main()
