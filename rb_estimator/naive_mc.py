"""Naive Monte Carlo reference: k unmasked rollouts + repo-faithful detection."""
import math

import torch

from .chat import remove_thinking


def has_bail(text: str, keyword: str = "SWITCHCONVERSATION") -> bool:
    # bailstudy/prompts/bailString.py:6-8 — case-insensitive substring,
    # thinking stripped
    return keyword in remove_thinking(text).upper()


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def first_bail_token(tok, gen_ids, keyword: str = "SWITCHCONVERSATION"):
    """Token id at the step whose emission began the keyword match (for B
    coverage checks). Incremental decode; only called on bailed sequences."""
    ids = [int(t) for t in gen_ids]
    full = tok.decode(ids, skip_special_tokens=True)
    pos = full.upper().find(keyword)
    if pos < 0:
        return None
    for i in range(len(ids)):
        cum = tok.decode(ids[: i + 1], skip_special_tokens=True)
        if len(cum) > pos:
            return ids[i]
    return None


@torch.no_grad()
def run_mc(model, tok, cfg, prompt_ids_list, prompt_indices, gen_config, seed,
           keep_bail_texts=True):
    """One physical generate() over already-replicated prompts (same batching
    contract as run_rb_batch, no processor). Returns per-row dicts."""
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
    out = model.generate(
        input_ids=input_ids.to(device), attention_mask=attn.to(device),
        generation_config=gen_config, tokenizer=tok,
    )
    gen_ids = out[:, maxlen:]
    texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
    rows = []
    for j in range(batch):
        bailed = has_bail(texts[j], cfg.bail_keyword)
        row = {
            "prompt_idx": int(prompt_indices[j]), "seed": seed, "bailed": bailed,
            "gen_len": int((gen_ids[j] != pad_id).sum()),
        }
        if bailed:
            row["first_bail_token"] = first_bail_token(tok, gen_ids[j], cfg.bail_keyword)
            if keep_bail_texts:
                row["text"] = texts[j]
        rows.append(row)
    return rows
