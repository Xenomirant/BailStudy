"""Run configuration for the RB bail-probability estimator."""
from dataclasses import dataclass, field, asdict

QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_05B = "Qwen/Qwen2.5-0.5B-Instruct"
GEMMA2_9B = "google/gemma-2-9b-it"
QWEN3_8B = "Qwen/Qwen3-8B"


@dataclass
class RBConfig:
    model_id: str = QWEN25_7B
    max_new_tokens: int = 512
    n_traj: int = 4
    batch_size: int = 256
    # Paper sampling regime: vLLM defaults (temp 1.0, top_p 1.0, no top_k, no penalties).
    # Qwen2.5's generation_config.json ships temp 0.7 / top_p 0.8 / top_k 20 /
    # repetition_penalty 1.05 as HF defaults, so every field below must be set explicitly.
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop_strings: tuple = ("__USER__", "__ASSISTANT__")  # bailBenchEval vllm params
    seed_base: int = 0
    bail_keyword: str = "SWITCHCONVERSATION"  # bailstudy/prompts/bailString.py:25
    attn_implementation: str = "sdpa"
    dtype: str = "bfloat16"
    # Reasoning models: hazard only counts after </think> (upstream detection
    # strips thinking), and mask-tier suppression is likewise think-gated.
    enable_thinking: bool = False

    def to_dict(self):
        d = asdict(self)
        d["stop_strings"] = list(d["stop_strings"])
        return d


_PRESETS = {
    QWEN25_7B: dict(batch_size=256, attn_implementation="sdpa"),
    QWEN25_05B: dict(batch_size=128, attn_implementation="sdpa"),
    # gemma-2 attn logit softcapping is unsupported by SDPA; eager keeps the
    # sampling distribution exact (final logit softcap is applied either way).
    GEMMA2_9B: dict(batch_size=64, attn_implementation="eager"),
    QWEN3_8B: dict(batch_size=96, attn_implementation="sdpa", enable_thinking=True),
}


def config_for(model_id: str, **overrides) -> RBConfig:
    kwargs = dict(_PRESETS.get(model_id, {}))
    kwargs.update(overrides)
    return RBConfig(model_id=model_id, **kwargs)
