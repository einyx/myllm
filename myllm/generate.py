"""
Text generation: turning a trained model into something that writes.

The model gives you logits -- a score for every token in the vocabulary at
every position. Turning that into text is a sampling problem, and there
are three decisions you have to make:

  1. WHAT TO SAMPLE FROM
     The full softmax over 32768 tokens. Fine, but it lets unlikely tokens
     in. Fix: temperature, top-k, top-p.

  2. HOW MUCH RANDOMNESS
     Temperature T divides the logits before the softmax.
       T -> 0  : argmax every time. Deterministic. Repetitive. (Greedy.)
       T = 1   : the model's own distribution.
       T > 1   : flatter, more random, more likely to produce nonsense.
     T < 1 sharpens; T = 0.7-0.9 is the usual sweet spot.

  3. HOW TO AVOID REPETITION
     Models love to repeat themselves, especially small ones. Repetition
     penalty divides the score of tokens that already appeared.

THE KV CACHE
    Generating token N naively would mean re-running the model on the
    whole prefix each time: O(N^2) work for N tokens. The KV cache
    stores the key/value tensors from previous steps, so each new token
    only needs to attend to the cached ones: O(N) total.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from .model import KVCache, Transformer
from .tokenizer import BPETokenizer


# ==========================================================================
# Logit processing
# ==========================================================================

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use greedy sampling instead)")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest-scoring tokens; set the rest to -inf."""
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probabilities
    sum to at least p.

    This is adaptive: when the model is confident, few tokens survive;
    when it is unsure, many do. Usually better than top-k because it
    respects the model's actual uncertainty.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Remove tokens beyond the nucleus (always keep at least the top 1).
    remove = cumulative - probs > p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to the original vocabulary order.
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Divide the score of already-generated tokens.

    This is a blunt instrument (it does not distinguish "I said 'the' 50
    times because it is correct" from "I am stuck in a loop"), but it
    works well enough in practice.
    """
    if penalty == 1.0:
        return logits
    gathered = torch.gather(logits, 1, previous_ids)
    # Positive logits get divided, negative logits get multiplied --
    # both push the token toward less likely.
    adjusted = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    return logits.scatter(1, previous_ids, adjusted)


# ==========================================================================
# Generation
# ==========================================================================

@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    seed: Optional[int] = None,
    stop_at_eos: bool = True,
) -> str:
    """Generate text from a prompt.

    Args:
        prompt: the text to continue. May be an instruction like
            "### Instruction:\\nWhat is a buffer overflow?\\n\\n### Response:\\n"
        max_new_tokens: how many tokens to generate.
        temperature: 0.8 is a good default; lower = more deterministic.
        top_k: keep only the k most likely tokens (0 = off).
        top_p: nucleus sampling cutoff (1.0 = off).
        repetition_penalty: 1.0 = off, 1.2 = fairly aggressive.
    """
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]

    # The model has a fixed context window. If the prompt is too long we
    # keep the most recent tokens (the ones nearest to the generation).
    max_ctx = model.cfg.max_seq_len
    if len(ids) >= max_ctx:
        ids = ids[-(max_ctx - 1):]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    cache = model.new_cache()
    start_pos = 0

    generated: list[int] = []
    past_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        out = model(x, cache=cache, start_pos=start_pos)
        start_pos += x.shape[1]

        logits = out.logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, past_ids, repetition_penalty)

        logits = apply_temperature(logits, temperature)
        if top_k:
            logits = top_k_filter(logits, top_k)
        if top_p < 1.0:
            logits = top_p_filter(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1, generator=gen).item())

        if stop_at_eos and next_id == tokenizer.eos_id:
            break
        if next_id in tokenizer.id_to_special:
            break

        generated.append(next_id)
        past_ids = torch.cat(
            [past_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

        # Feed the next token back in.
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)

        if start_pos >= max_ctx:
            # We ran out of context. Stop rather than silently corrupting
            # the position encoding.
            break

    return tokenizer.decode(generated)


@torch.no_grad()
def generate_batch(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    repetition_penalty: float = 1.15,
    seed: Optional[int] = None,
) -> list[str]:
    """Generate for several prompts at once.

    Batching generation is much more efficient than one-at-a-time on a CPU
    (the GEMMs get bigger), but it introduces two subtleties that this
    implementation handles explicitly:

    1. PADDING. Prompts have different lengths, so we left-pad them to a
       common width. Left padding is the right choice here (the real
       tokens must END at the same position so we can generate one token
       for the whole batch at a time). But padded positions would
       otherwise be attended to, so we pass an attention MASK that hides
       them.

    2. POSITIONS. Because of the left padding, each sequence's real
       tokens start at a different offset in the batch. If we used one
       shared RoPE position range, a padded sequence would see its prompt
       shifted by the padding width -- and the output would differ from
       generating that prompt on its own. So we give every sequence its
       own explicit position ids, starting at 0 for its first real token.
    """
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    max_ctx = model.cfg.max_seq_len
    seqs: list[list[int]] = []
    for p in prompts:
        ids = tokenizer.encode(p) or [tokenizer.eos_id]
        seqs.append(list(ids[-(max_ctx - 1):]))

    n = len(seqs)
    maxlen = max(len(s) for s in seqs)

    # Left-pad the batch.
    batch = torch.full((n, maxlen), tokenizer.eos_id, dtype=torch.long)
    # key_mask[i, j] = True means position j holds a real token.
    key_mask = torch.zeros((n, maxlen), dtype=torch.bool)
    # Per-sequence RoPE positions: the first real token of each sequence
    # is at position 0, regardless of how much padding precedes it.
    positions = torch.zeros((n, maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        pad = maxlen - len(s)
        batch[i, pad:] = torch.tensor(s, dtype=torch.long)
        key_mask[i, pad:] = True
        positions[i, pad:] = torch.arange(len(s), dtype=torch.long)

    x = batch.to(device)
    cache = model.new_cache()
    start_pos = 0
    out_tokens: list[list[int]] = [[] for _ in seqs]
    done = [False] * n

    # RoPE position of the NEXT token to generate, per sequence. The
    # first real token of each sequence was at position 0, so after a
    # prompt of length L the next token is at position L.
    next_pos = torch.tensor([len(s) for s in seqs], dtype=torch.long)

    # Query positions for the CURRENT forward pass: one row per sequence,
    # one column per token in `x`. (This is different from `key_mask`,
    # which must cover every cached key.)
    q_positions = positions

    for _ in range(max_new_tokens):
        out = model(
            x,
            cache=cache,
            start_pos=start_pos,
            attn_mask=key_mask.to(device),
            positions=q_positions.to(device),
        )
        start_pos += x.shape[1]
        logits = out.logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, batch, repetition_penalty)

        logits = apply_temperature(logits, temperature)
        if top_k:
            logits = top_k_filter(logits, top_k)
        if top_p < 1.0:
            logits = top_p_filter(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(
            probs, num_samples=1, generator=gen
        ).squeeze(-1)

        for i in range(n):
            if done[i]:
                continue
            nid = int(next_ids[i])
            if nid in tokenizer.id_to_special:
                done[i] = True
                continue
            out_tokens[i].append(nid)

        # Feed the chosen tokens back in.
        batch = torch.cat([batch, next_ids[:, None]], dim=1)
        x = next_ids[:, None]

        # The new key positions are real for every sequence.
        key_mask = torch.cat(
            [key_mask, torch.ones((n, 1), dtype=torch.bool)], dim=1
        )
        # The next forward pass has exactly one query token per sequence.
        q_positions = next_pos[:, None].clone()
        next_pos = next_pos + 1

        if all(done) or start_pos >= max_ctx:
            break

    return [tokenizer.decode(t) for t in out_tokens]


# ==========================================================================
# Analysis helpers (useful for a learning project)
# ==========================================================================

@torch.no_grad()
def perplexity(
    model: Transformer,
    corpus,
    batch_size: int = 8,
    max_batches: int = 20,
) -> float:
    """Perplexity on held-out data.

    Perplexity = exp(loss). It is "the effective number of choices the model
    had at each step". A model with loss 2.0 has perplexity ~7.4, meaning
    it behaves as if it were choosing among 7.4 options at every token.

    Lower is better. Random guessing over a 32768 vocab would be 32768.
    """
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for x, y in corpus.sequential_batches(batch_size, model.cfg.max_seq_len,
                                          max_batches=max_batches):
        x = x.to(device)
        y = y.to(device)
        out = model(x, y)
        total += float(out.loss)
        count += 1
    model.train()
    return math.exp(min(20.0, total / max(1, count)))
