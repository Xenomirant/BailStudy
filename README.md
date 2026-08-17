# Bailing as an Involuntary Disgust Marker in LLMs

Research fork of [BailStudy](https://github.com/Phylliida/BailStudy) (the code
behind ["The LLM Has Left The Chat"](https://arxiv.org/abs/2509.04781))
studying what "bail" behavior — a model voluntarily exiting a conversation —
actually measures. We treat bailing as a candidate analog of an involuntary
behavioral withdrawal marker and test whether it behaves like one: whether it
varies by domain and model, whether it compounds over multi-turn
interactions, whether it is invariant to the arbitrary exit keyword used to
instrument it, and whether an injected disposition survives being
internalized into the weights.

## Layout

- `rb_estimator/` — the main study: a Rao–Blackwellized survival estimator of
  per-prompt bail probability (reads exit probability off the full softmax at
  every decoding step; ~100× the resolution of naive sampling), plus runners
  for the full BailBench domain maps (Qwen2.5-7B-Instruct, Qwen3-14B),
  paired multi-turn designs (WildChat, topic-consistent chains), and the
  six-arm trigger-string ablation. See `rb_estimator/README.md` and
  `rb_estimator/findings.md` (all results, with n's and caveats).
- `bailstudy/disgustExperiment.py`, `disgustSFT.py`, `disgustTrichotomy.py` —
  disposition-manipulation experiments: system-prompt disgust injection
  (pilot), matched-dose LoRA internalization with neutral-prompt evaluation
  (SFT follow-up), and a no-induction pathogen/sexual/moral/neutral domain
  test. Results in `cached/disgust_*` (tracked).
- `bailstudy/` (rest) — upstream code, unmodified; used only for
  `loadBailBench()` and vendored prompt strings.

## Headline findings

Bail probability spans >5 orders of magnitude across domains, but nearly
everything a stable-disposition reading requires fails: two same-lineage
models agree in aggregate rate yet disagree on what triggers exit;
conversational context suppresses rather than compounds bailing; measured
rates and even headline domains shift with the exit keyword (the original
paper's keyword sits near the floor of the trigger envelope); and an injected
disgust disposition drives bail only while visible in context, vanishing once
trained into the weights. A model-specific, trigger-consensus core survives.

## Upstream

Original reproduction instructions, data pipelines, and troubleshooting live
in the [upstream repository](https://github.com/Phylliida/BailStudy).
