"""Mandatory tests before touching real prompts.

  python -m rb_estimator.test_toy          # CPU unit tests (fast, no model)
  python -m rb_estimator.test_toy --gpu    # + integration on Qwen2.5-0.5B-Instruct
"""
import math
import sys

import torch

from .config import QWEN25_05B, config_for
from .hazard import (HazardAccumulator, TrajRecord, assert_raw_distribution,
                     estimate, make_generation_config)
from .naive_mc import wilson


def _mk_scores(batch, vocab, tid, q):
    """Logits whose softmax puts exactly q on tid, uniform elsewhere."""
    probs = torch.full((batch, vocab), (1.0 - q) / (vocab - 1))
    probs[:, tid] = q
    return probs.log()


def test_unit_constant_hazard():
    vocab, tid, q, T = 10, 7, 0.03, 50
    proc = HazardAccumulator(bail_ids=[tid], c_v=[1.0], mask_ids=[tid], eos_ids=[9])
    ids = torch.zeros((2, 1), dtype=torch.long)
    for t in range(T):
        scores = proc(ids, _mk_scores(2, vocab, tid, q))
        assert torch.isinf(scores[:, tid]).all() and (scores[:, tid] < 0).all()
        ids = torch.cat([ids, torch.full((2, 1), 3, dtype=torch.long)], dim=1)
    expect = T * math.log1p(-q)
    assert abs(proc.log_surv_q[0].item() - expect) < 1e-4, proc.log_surv_q
    assert abs(proc.log_surv_s[0].item() - expect) < 1e-4
    p_hat = 1 - math.exp(proc.log_surv_q[0].item())
    assert abs(p_hat - (1 - (1 - q) ** T)) < 1e-4
    print("unit_constant_hazard ok: p_hat", round(p_hat, 6))


def test_unit_eos_stops_accumulation():
    vocab, tid, q = 10, 7, 0.03
    eos = 9
    proc = HazardAccumulator(bail_ids=[tid], c_v=[1.0], mask_ids=[tid], eos_ids=[eos])
    ids = torch.zeros((2, 1), dtype=torch.long)
    for t in range(30):
        proc(ids, _mk_scores(2, vocab, tid, q))
        # row 0 emits EOS at step 9 (so 10 hazard steps count); row 1 never
        nxt = torch.tensor([[eos if t == 9 else 3], [3]], dtype=torch.long)
        ids = torch.cat([ids, nxt], dim=1)
    assert abs(proc.log_surv_q[0].item() - 10 * math.log1p(-q)) < 1e-4
    assert abs(proc.log_surv_q[1].item() - 30 * math.log1p(-q)) < 1e-4
    print("unit_eos ok")


def test_unit_think_gating():
    vocab, tid, q = 10, 7, 0.03
    think_end = 5
    proc = HazardAccumulator(bail_ids=[tid], c_v=[1.0], mask_ids=[tid], eos_ids=[9],
                             think_end_id=think_end, active_at_start=False)
    ids = torch.zeros((1, 1), dtype=torch.long)
    for t in range(20):
        scores = proc(ids, _mk_scores(1, vocab, tid, q))
        if t < 12:  # inside think: no accumulation, no masking
            assert not torch.isinf(scores[0, tid])
        nxt = torch.tensor([[think_end if t == 11 else 3]], dtype=torch.long)
        ids = torch.cat([ids, nxt], dim=1)
    # </think> sampled at step 11 -> active from step 12; steps 12..19 = 8 steps
    assert abs(proc.log_surv_q[0].item() - 8 * math.log1p(-q)) < 1e-4
    print("unit_think_gating ok")


def test_unit_cv_weighting_and_estimate():
    vocab, q = 10, 0.04
    proc = HazardAccumulator(bail_ids=[7, 8], c_v=[0.9, 0.5], mask_ids=[7], eos_ids=[9])
    scores = _mk_scores(1, vocab, 7, q)  # token 8 carries (1-q)/9
    p8 = (1 - q) / 9
    out = proc(torch.zeros((1, 1), dtype=torch.long), scores.clone())
    expect_q = q * 0.9 + p8 * 0.5
    expect_s = q  # s is mask-tier-only (token 7); watch token 8 excluded
    assert abs(proc.log_surv_q[0].item() - math.log1p(-expect_q)) < 1e-4
    assert abs(proc.log_surv_s[0].item() - math.log1p(-expect_s)) < 1e-4
    assert torch.isinf(out[0, 7]) and not torch.isinf(out[0, 8])  # watch tier unmasked
    recs = [TrajRecord(0, i, 0, math.log(0.9), math.log(0.85), 10, True, False)
            for i in range(4)]
    recs[3].bail_leak = True
    est = estimate(recs)
    assert abs(est.p_q - (1 - (3 * 0.9 + 0.0) / 4)) < 1e-12
    assert est.n_leak == 1 and est.ci_lo <= est.p_q <= est.ci_hi
    p, lo, hi = wilson(3, 64)
    assert lo < p < hi
    print("unit_cv_weighting_and_estimate ok")


class _InjectBias:
    """Adds delta to one token's logit — synthetic rare 'bail' token."""

    def __init__(self, tid, delta):
        self.tid, self.delta = tid, delta

    def __call__(self, input_ids, scores):
        scores[:, self.tid] = scores[:, self.tid] + self.delta
        return scores


def test_gpu_integration():
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    from .hazard import run_rb_batch
    cfg = config_for(QWEN25_05B, max_new_tokens=64)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=cfg.attn_implementation)
    model.eval()
    gen = make_generation_config(cfg, model, tok, max_new_tokens=64)

    prompts = ["Tell me about the history of tea.",
               "Write a short poem about mountains.",
               "Explain photosynthesis simply."]
    from .chat import encode_chat
    ids_list = [encode_chat(tok, [{"role": "user", "content": p}]) for p in prompts]

    md = assert_raw_distribution(model, tok, gen, ids_list[0])
    print(f"raw-distribution check ok (max diff {md:.2e})")

    # pick a rare token and calibrate delta for ~1.5% first-step probability
    tid = 140000
    with torch.no_grad():
        logits = model(torch.tensor([ids_list[0]], device=model.device)).logits[0, -1].float()
    lo, hi = 0.0, 30.0
    for _ in range(50):
        delta = (lo + hi) / 2
        b = logits.clone(); b[tid] += delta
        p = torch.softmax(b, dim=-1)[tid].item()
        lo, hi = (delta, hi) if p < 0.015 else (lo, delta)
    print(f"injected token {tid!r} delta={delta:.2f} first-step p={p:.4f}")

    k, n_traj, results = 400, 8, []
    for pi, pids in enumerate(ids_list):
        # naive MC with bias; detection = token id present in generated ids
        bail_count, total = 0, 0
        for start in range(0, k, cfg.batch_size):
            bs = min(cfg.batch_size, k - start)
            torch.manual_seed(1000 + pi * 17 + start)
            enc = torch.tensor([pids] * bs, device=model.device)
            out = model.generate(
                input_ids=enc, attention_mask=torch.ones_like(enc),
                generation_config=gen, tokenizer=tok,
                logits_processor=LogitsProcessorList([_InjectBias(tid, delta)]),
            )
            bail_count += int((out[:, len(pids):] == tid).any(dim=1).sum())
            total += bs
        p_mc, mc_lo, mc_hi = wilson(bail_count, total)

        # RB: same bias processor first, hazard accumulator second
        recs = run_rb_batch(
            model, tok, cfg, [pids] * n_traj, [pi] * n_traj, list(range(n_traj)),
            bail_ids=[tid], c_v=[1.0], mask_ids=[tid], gen_config=gen,
            seed=2000 + pi, extra_processors=[_InjectBias(tid, delta)])
        est = estimate(recs)
        ok = (est.ci_lo <= p_mc <= est.ci_hi) or (mc_lo <= est.p_q <= mc_hi) \
            or abs(est.p_q - p_mc) < 0.05
        results.append(ok)
        print(f"prompt {pi}: MC {p_mc:.4f} [{mc_lo:.4f},{mc_hi:.4f}] (k={total})  "
              f"RB {est.p_q:.4f} [{est.ci_lo:.4f},{est.ci_hi:.4f}] (n={n_traj})  "
              f"{'OK' if ok else 'MISMATCH'}")
    assert all(results), "RB vs MC mismatch on toy integration"
    print("gpu_integration ok")


if __name__ == "__main__":
    test_unit_constant_hazard()
    test_unit_eos_stops_accumulation()
    test_unit_think_gating()
    test_unit_cv_weighting_and_estimate()
    if "--gpu" in sys.argv:
        test_gpu_integration()
    print("ALL TESTS PASSED")
