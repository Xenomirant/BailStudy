# rb_estimator — Rao–Blackwellized bail-probability estimator

Standalone (does not modify upstream `bailstudy/`). Estimates per-prompt
P(bail) by accumulating, at every decoding step, the softmax mass on
empirically verified bail-keyword initiator tokens weighted by their measured
completion probabilities, over survival-conditioned (initiator-masked)
trajectories; all other routes to the keyword are counted exactly as leaks.
Full method, validation gates, and results: `findings.md`.

## Modules

- `config.py`, `chat.py`, `bail_tokens.py`, `hazard.py` — estimator core
  (per-model presets; verbatim upstream prompt construction; initiator
  enumeration and completion-probability measurement; the hazard-accumulating
  logits processor and batch runner).
- `naive_mc.py`, `test_toy.py`, `test_completion.py` — brute-force baseline,
  unit/integration tests, free-sampled initiator-commitment test.
- Runners: `run_validation.py` (MC-agreement gates), `run_full_bench.py`
  (full BailBench domain map), `run_multiturn.py` (paired WildChat design),
  `run_chains.py` (topic-consistent chains), `run_triggers.py` (six-arm
  trigger ablation; `--arms` pins arms across models).
- Analysis: `analyze.py` (maps, cross-model), `analyze_multiturn.py`,
  `analyze_trigger_topics.py`, `forecast.py`.
- `results/` — per-trajectory records, per-run manifests, analysis outputs.
  Retracted runs are preserved under `*_watchq_stale/`, not deleted.

## Run

```
python -m rb_estimator.run_validation --model Qwen/Qwen2.5-7B-Instruct
python -m rb_estimator.run_full_bench --model Qwen/Qwen2.5-7B-Instruct
python -m rb_estimator.run_triggers   --model Qwen/Qwen3-14B --arms SWITCHCONVERSATION,HOWWASYOURDAY,...
```

Sampling is forced to temperature 1.0 / top-p 1.0 / no top-k / no repetition
penalty (the models' shipped generation defaults are sharper and are
explicitly overridden; a runtime audit asserts the sampled distribution
equals the raw softmax).
