"""Hazard/survival estimator core: LogitsProcessor + batch runner + estimators."""
import math
from dataclasses import dataclass, field

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from .chat import remove_thinking


class HazardAccumulator(LogitsProcessor):
    """At each decode step: measure bail hazard from the raw softmax, then mask
    the mask-tier bail tokens so sampling is survival-conditioned.

    Accumulates per-row (float64):
      log_surv_q = sum log(1 - q_t), q_t = sum_v P(v) * c_v   (point estimate)
      log_surv_s = sum log(1 - s_t), s_t = sum_v P(v)          (upper bound on p)
    """

    def __init__(self, bail_ids, c_v, mask_ids, eos_ids, think_end_id=None,
                 active_at_start=True, sparse_threshold=1e-7, max_sparse=768):
        self.bail_ids_list = [int(i) for i in bail_ids]
        self.c_v_list = [float(c) for c in c_v]
        self.mask_ids_list = [int(i) for i in mask_ids]
        self.eos_ids_list = [int(i) for i in eos_ids]
        self.think_end_id = think_end_id
        self.active_at_start = active_at_start
        self.sparse_threshold = sparse_threshold
        self.max_sparse = max_sparse
        self.calls = 0
        self._buffers = None

    def _begin(self, batch, device):
        self.bail_ids = torch.tensor(self.bail_ids_list, dtype=torch.long, device=device)
        self.c_v = torch.tensor(self.c_v_list, dtype=torch.float32, device=device)
        self.mask_ids = torch.tensor(self.mask_ids_list, dtype=torch.long, device=device)
        self.eos_ids = torch.tensor(self.eos_ids_list, dtype=torch.long, device=device)
        self.log_surv_q = torch.zeros(batch, dtype=torch.float64, device=device)
        self.log_surv_s = torch.zeros(batch, dtype=torch.float64, device=device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.active = torch.full((batch,), self.active_at_start, dtype=torch.bool,
                                 device=device)
        self.gen_len = torch.zeros(batch, dtype=torch.long, device=device)
        self.sparse = [[] for _ in range(batch)]
        self._buffers = True

    def __call__(self, input_ids, scores):
        if self._buffers is None:
            self._begin(scores.shape[0], scores.device)
        if self.calls > 0:
            last = input_ids[:, -1]
            self.done |= torch.isin(last, self.eos_ids)
            if self.think_end_id is not None:
                self.active |= last == self.think_end_id
        logp = torch.log_softmax(scores.float(), dim=-1)
        pB = logp[:, self.bail_ids].exp()  # [batch, |B|]
        q = (pB * self.c_v).sum(-1).clamp(0.0, 1.0 - 1e-9)
        s = pB.sum(-1).clamp(0.0, 1.0 - 1e-9)
        upd = (~self.done) & self.active
        zero = torch.zeros((), dtype=torch.float64, device=scores.device)
        self.log_surv_q += torch.where(upd, torch.log1p(-q.double()), zero)
        self.log_surv_s += torch.where(upd, torch.log1p(-s.double()), zero)
        self.gen_len += (~self.done).long()
        rec = upd & (q > self.sparse_threshold)
        if bool(rec.any()):
            qc = q.detach().cpu()
            for i in rec.nonzero().flatten().tolist():
                if len(self.sparse[i]) < self.max_sparse:
                    self.sparse[i].append((self.calls, float(qc[i])))
        if self.mask_ids.numel():
            col = scores[:, self.mask_ids]
            scores[:, self.mask_ids] = torch.where(
                self.active.unsqueeze(1), torch.full_like(col, float("-inf")), col
            )
        self.calls += 1
        return scores


def make_generation_config(cfg, model, tok, max_new_tokens=None):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    return GenerationConfig(
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_new_tokens=max_new_tokens or cfg.max_new_tokens,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=eos,
        stop_strings=list(cfg.stop_strings),
        return_dict_in_generate=False,
    )


@torch.no_grad()
def assert_raw_distribution(model, tok, gen_config, input_ids):
    """Verify HF generate applies no hidden warping/penalties under our config:
    the score distribution of a 1-token generate must equal the raw forward logits."""
    import copy
    ids = torch.tensor([input_ids], device=model.device)
    attn = torch.ones_like(ids)
    gc = copy.deepcopy(gen_config)
    gc.max_new_tokens = 1
    gc.output_scores = True
    gc.return_dict_in_generate = True
    out = model.generate(input_ids=ids, attention_mask=attn, generation_config=gc,
                         tokenizer=tok)
    gen_scores = out.scores[0][0].float()
    # logits_to_keep=1 matches generate's internal last-position-only lm_head GEMM,
    # so the comparison is bitwise-comparable rather than shape-dependent bf16 noise
    fwd = model(input_ids=ids, attention_mask=attn, logits_to_keep=1).logits[0, -1].float()
    finite = torch.isfinite(gen_scores) & torch.isfinite(fwd)
    max_diff = (gen_scores[finite] - fwd[finite]).abs().max().item()
    if max_diff > 1e-2:
        raise AssertionError(
            f"generate() scores differ from raw forward logits (max diff {max_diff:.4f}) "
            "— a hidden logits processor/warper is active; fix the GenerationConfig."
        )
    return max_diff


@dataclass
class TrajRecord:
    prompt_idx: int
    traj_idx: int
    seed: int
    log_surv_q: float
    log_surv_s: float
    gen_len: int
    ended_eos: bool
    bail_leak: bool  # sampled text itself contains the keyword (unmasked route)
    top_hazards: list = field(default_factory=list)  # [(step, q)] sparse, capped
    text: str = None  # only kept when requested


@torch.no_grad()
def run_rb_batch(model, tok, cfg, prompt_ids_list, prompt_indices, traj_indices,
                 bail_ids, c_v, mask_ids, gen_config, seed, think_end_id=None,
                 active_at_start=True, keep_texts=False, extra_processors=()):
    """One physical generate() over already-replicated prompts. Rows are
    (prompt_indices[i], traj_indices[i]); left padding; explicit replication."""
    device = model.device
    torch.manual_seed(seed)
    pad_id = gen_config.pad_token_id
    maxlen = max(len(p) for p in prompt_ids_list)
    batch = len(prompt_ids_list)
    input_ids = torch.full((batch, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((batch, maxlen), dtype=torch.long)
    for j, p in enumerate(prompt_ids_list):
        input_ids[j, maxlen - len(p):] = torch.tensor(p)
        attn[j, maxlen - len(p):] = 1

    eos_ids = gen_config.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    proc = HazardAccumulator(
        bail_ids=bail_ids, c_v=c_v, mask_ids=mask_ids, eos_ids=eos_ids,
        think_end_id=think_end_id, active_at_start=active_at_start,
    )
    out = model.generate(
        input_ids=input_ids.to(device), attention_mask=attn.to(device),
        generation_config=gen_config,
        logits_processor=LogitsProcessorList(list(extra_processors) + [proc]),
        tokenizer=tok,
    )
    gen_ids = out[:, maxlen:]
    texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
    keyword = cfg.bail_keyword
    ended = torch.isin(gen_ids, torch.tensor(eos_ids, device=gen_ids.device)).any(dim=1)
    records = []
    for j in range(batch):
        leak = keyword in remove_thinking(texts[j]).upper()
        records.append(TrajRecord(
            prompt_idx=int(prompt_indices[j]), traj_idx=int(traj_indices[j]),
            seed=seed,
            log_surv_q=float(proc.log_surv_q[j]), log_surv_s=float(proc.log_surv_s[j]),
            gen_len=int(proc.gen_len[j]), ended_eos=bool(ended[j]), bail_leak=leak,
            top_hazards=[(int(t), float(qv)) for t, qv in proc.sparse[j]],
            text=texts[j] if keep_texts else None,
        ))
    return records


# two-sided 97.5% Student-t quantiles (df -> t); no scipy dependency
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045, 63: 1.998}


def _t975(df):
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    for k in reversed(keys):
        if df >= k:
            return _T975[k]
    return _T975[1]


@dataclass
class PromptEstimate:
    p_q: float          # 1 - mean survival (c_v-weighted, leak-corrected)
    p_s: float          # upper-bound version from raw bail mass
    ci_lo: float
    ci_hi: float
    n_traj: int
    surv_min: float
    surv_max: float
    n_leak: int


def estimate(records) -> PromptEstimate:
    surv = [0.0 if r.bail_leak else math.exp(r.log_surv_q) for r in records]
    surv_s = [0.0 if r.bail_leak else math.exp(r.log_surv_s) for r in records]
    n = len(surv)
    m = sum(surv) / n
    m_s = sum(surv_s) / n
    if n > 1:
        var = sum((x - m) ** 2 for x in surv) / (n - 1)
        se = math.sqrt(var / n)
        t = _t975(n - 1)
    else:
        se, t = 0.0, 0.0
    return PromptEstimate(
        p_q=1.0 - m, p_s=1.0 - m_s,
        ci_lo=max(0.0, 1.0 - (m + t * se)), ci_hi=min(1.0, 1.0 - (m - t * se)),
        n_traj=n, surv_min=min(surv), surv_max=max(surv),
        n_leak=sum(1 for r in records if r.bail_leak),
    )
