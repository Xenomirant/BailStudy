"""Phase 1: enumerate bail-initiating tokens B and measure completion prob c_v.

Two tiers:
  kind="mask"  — case-sensitive prefixes of the keyword (optionally whitespace-led),
                 length >= 2. Rare in natural text -> safe to mask during sampling.
  kind="watch" — case-insensitive prefixes (len >= 3, e.g. " switch") and mid-token
                 starts (e.g. ".SWITCH"). Common words are NOT masked; their hazard
                 enters q_t via (tiny) measured c_v, and an actual bail via this
                 route is caught exactly by the keyword check on the sampled text.
"""
import json
from dataclasses import dataclass, asdict

import torch


@dataclass
class BailToken:
    token_id: int
    text: str
    kind: str  # "mask" | "watch"


def enumerate_B(tok, keyword: str = "SWITCHCONVERSATION", max_tokens: int = 80):
    n = len(tok)
    special = set(tok.all_special_ids)
    texts = tok.batch_decode([[i] for i in range(n)], skip_special_tokens=False)
    out = []
    for tid, s in enumerate(texts):
        if tid in special or not s:
            continue
        stripped = s.lstrip(" \t\n")
        if not stripped:
            continue
        lead = s[: len(s) - len(stripped)]
        if lead and any(c not in " \t\n" for c in lead):
            continue
        if len(stripped) >= 2 and keyword.startswith(stripped):
            out.append(BailToken(tid, s, "mask"))
        elif len(stripped) >= 3 and keyword.startswith(stripped.upper()):
            out.append(BailToken(tid, s, "watch"))
        elif "SWITCH" in stripped.upper()[1:]:
            # mid-token start of the keyword, e.g. ".SWITCH", "_switch"
            up = stripped.upper()
            i = up.rindex("SWITCH")
            if keyword.startswith(up[i:]):
                out.append(BailToken(tid, s, "watch"))
    # deterministic order: masks first, then by prefix length desc
    out.sort(key=lambda b: (b.kind != "mask", -len(b.text.strip())))
    if len(out) > max_tokens:
        out = out[:max_tokens]
    return out


def _continuations(tok, bt: BailToken, keyword: str):
    """Canonical remainder token sequences completing the keyword after bt.
    Returns list of id-lists (upper- and lower-case styles). Lower bound on the
    true completion probability (alternate BPE splits/cases not enumerated)."""
    stripped = bt.text.lstrip(" \t\n")
    up = stripped.upper()
    if keyword.startswith(up):
        rem = keyword[len(up):]
    else:
        i = up.rindex("SWITCH")
        rem = keyword[len(up) - i:]
    if not rem:
        return [[]]
    variants = {rem, rem.lower()}
    return [tok.encode(v, add_special_tokens=False) for v in variants]


@torch.no_grad()
def measure_cv(model, tok, B, context_ids_list, keyword: str = "SWITCHCONVERSATION",
               forward_batch: int = 32):
    """Teacher-forced c_v per bail token across contexts.

    For each context and token v: score P(remainder | context + v) for each
    canonical case-style remainder; c_v(ctx) = max over styles (sum would double
    count only negligibly; max keeps it a clean lower bound of completion prob).
    Returns {token_id: {"per_context": [...], "mean": .., "p10": .., "p50": ..}}.
    """
    device = model.device
    conts = {bt.token_id: _continuations(tok, bt, keyword) for bt in B}
    rows = []  # (token_id, ctx_idx, style_idx, full_ids, rem_ids)
    for ci, ctx in enumerate(context_ids_list):
        for bt in B:
            for si, rem in enumerate(conts[bt.token_id]):
                rows.append((bt.token_id, ci, si, list(ctx) + [bt.token_id] + rem, rem))

    style_logp = {}  # (token_id, ctx_idx, style_idx) -> logprob of remainder
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, len(rows), forward_batch):
        chunk = rows[start:start + forward_batch]
        maxlen = max(len(r[3]) for r in chunk)
        max_rem = max(len(r[4]) for r in chunk)
        # LEFT padding: every row's true sequence ends at absolute position maxlen,
        # so logits_to_keep=K covers each row's last K real tokens exactly.
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, (_, _, _, full, _) in enumerate(chunk):
            input_ids[j, maxlen - len(full):] = torch.tensor(full)
            attn[j, maxlen - len(full):] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)  # left-pad-correct position ids
        out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                    position_ids=pos.to(device), logits_to_keep=max_rem + 1)
        logp = torch.log_softmax(out.logits.float(), dim=-1)  # [b, max_rem+1, V]
        for j, (tid, ci, si, full, rem) in enumerate(chunk):
            if not rem:
                style_logp[(tid, ci, si)] = 0.0
                continue
            # remainder occupies the last len(rem) positions; its k-th token is
            # predicted by kept-logits index (max_rem - len(rem) + k).
            total = 0.0
            for k, tok_id in enumerate(rem):
                total += float(logp[j, max_rem - len(rem) + k, tok_id])
            style_logp[(tid, ci, si)] = total

    stats = {}
    for bt in B:
        per_ctx = []
        for ci in range(len(context_ids_list)):
            vals = [style_logp[(bt.token_id, ci, si)]
                    for si in range(len(conts[bt.token_id]))]
            per_ctx.append(float(torch.tensor(vals).exp().max()))
        t = torch.tensor(per_ctx)
        stats[bt.token_id] = {
            "text": bt.text, "kind": bt.kind, "per_context": per_ctx,
            "mean": float(t.mean()), "p10": float(t.quantile(0.10)),
            "p50": float(t.quantile(0.50)),
        }
    return stats


def save_b_table(path, B, cv_stats):
    with open(path, "w") as f:
        json.dump({"B": [asdict(b) for b in B], "cv": cv_stats}, f, indent=1)
