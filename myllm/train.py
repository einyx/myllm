"""
The training loop.

What actually happens, once per optimizer step:

  1. Get a batch of (input, target) token windows from the packed corpus.
  2. Run them through the model -> logits.
  3. Compute cross-entropy between logits and targets.
  4. Backpropagate, accumulating over `grad_accum` micro-batches.
  5. Clip the gradient norm.
  6. Step the optimizer with a learning rate from the schedule.
  7. Log, and occasionally checkpoint + evaluate.

The pieces worth understanding:

  GRADIENT ACCUMULATION
      A "batch" of 16 x 512 = 8192 tokens might not fit in memory at once,
      but 16 x 512 does not need to. We run 8 micro-batches of 2 x 512,
      accumulating their gradients, and take one optimizer step. The
      result is mathematically identical to one big batch (up to float
      precision), but uses much less memory. This is what lets you use a
      large effective batch on a small machine.

  WEIGHT DECAY
      Decay is applied only to matrices, never to norms or biases. The
      idea: decaying a weight pulls it toward zero, which is a sensible
      prior for "this dimension is not important". But pulling a norm gain
      toward zero would slow the model down for no reason.

  THE LEARNING RATE SCHEDULE
      Warmup (ramp from 0) then cosine decay. Warmup exists because the
      first few steps have garbage gradients -- the AdamW second-moment
      estimate is meaningless, and the embedding matrix can get blown up
      if the learning rate is too high while the norm gains are still
      adapting. Cosine decay after that lets you train at a high learning
      rate for a long time (fast progress) and then fine-tune at a low one
      (careful convergence).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .data import TokenCorpus
from .model import Transformer
from .utils import JsonlLogger, configure_threads, human_time, set_seed


# ==========================================================================
# Optimizer
# ==========================================================================

def build_optimizer(model: Transformer, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    AdamW differs from Adam + L2 regularization in one important way: it
    applies the decay directly to the weights rather than adding it to the
    gradient. With Adam, the adaptive per-parameter learning rate rescales
    the L2 term, so the effective decay strength depends on the gradient
    magnitudes -- an accident of the data rather than a choice. AdamW
    decouples them, which makes `weight_decay` mean what it says.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )


def lr_scale(step: int, cfg: TrainConfig, total_steps: int) -> float:
    """The learning-rate multiplier for a given step.

    Warmup: linear ramp from ~0 to 1 over `warmup_steps`.
    Cosine:  1 -> min_lr_ratio over the remaining steps.
    """
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)

    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


# ==========================================================================
# Trainer
# ==========================================================================

class Trainer:
    def __init__(
        self,
        model: Transformer,
        cfg: TrainConfig,
        train_corpus: TokenCorpus,
        val_corpus: Optional[TokenCorpus] = None,
        out_dir: str | Path = "runs/default",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_corpus = train_corpus
        self.val_corpus = val_corpus
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.model.to(device)
        self.model.set_attn_impl(cfg.attn_impl)

        self.optimizer = build_optimizer(model, cfg)
        self.step = 0
        self.tokens_seen = 0
        self.rng = np.random.default_rng(cfg.seed)
        self.logger = JsonlLogger(self.out_dir / "metrics.jsonl")

        self.effective_batch = cfg.batch_size * cfg.grad_accum
        self.total_steps = min(
            cfg.max_steps,
            max(1, cfg.max_tokens // (self.effective_batch * model.cfg.max_seq_len)),
        )

        # Record the exact configuration of this run so that tools (and
        # future you) can tell what produced a given checkpoint.
        (self.out_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": model.cfg.to_dict(),
                    "train": vars(cfg),
                    "total_steps": self.total_steps,
                    "effective_batch_tokens": (
                        self.effective_batch * model.cfg.max_seq_len
                    ),
                    "max_tokens": cfg.max_tokens,
                    "params": model.num_params(),
                    "threads": torch.get_num_threads(),
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _set_lr(self) -> float:
        scale = lr_scale(self.step, self.cfg, self.total_steps)
        lr = self.cfg.lr * scale
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean cross-entropy on the validation set (nats/token)."""
        if self.val_corpus is None:
            return float("nan")
        self.model.eval()
        total, count = 0.0, 0
        for x, y in self.val_corpus.sequential_batches(
            batch_size=self.cfg.batch_size,
            seq_len=self.model.cfg.max_seq_len,
            max_batches=self.cfg.eval_batches,
        ):
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x, y)
            total += float(out.loss)
            count += 1
        self.model.train()
        return total / max(1, count)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, max_steps: Optional[int] = None) -> None:
        cfg = self.cfg
        seq_len = self.model.cfg.max_seq_len
        target_steps = min(self.total_steps, max_steps or self.total_steps)

        print("=" * 72)
        print(f"  model     : {self.model.cfg.name} "
              f"({self.model.num_params() / 1e6:.1f}M params)")
        print(f"  seq_len   : {seq_len}")
        print(f"  micro b/s : {cfg.batch_size} x {cfg.grad_accum} accum "
              f"= {self.effective_batch} seqs "
              f"= {self.effective_batch * seq_len:,} tokens/step")
        print(f"  total     : {target_steps:,} steps "
              f"({target_steps * self.effective_batch * seq_len / 1e6:.1f}M tokens)")
        print(f"  lr        : {cfg.lr} (warmup {cfg.warmup_steps}, cosine to "
              f"{cfg.min_lr_ratio}x)")
        print(f"  threads   : {torch.get_num_threads()}")
        print("=" * 72)

        self.model.train()
        t_start = time.time()
        t_last = t_start
        tokens_last = self.tokens_seen
        best_val = float("inf")

        for step in range(self.step, target_steps):
            lr = self._set_lr()

            # ---- forward/backward over the accumulation window ----
            loss_accum = 0.0
            for micro in range(cfg.grad_accum):
                x, y = self.train_corpus.sample_batch(
                    cfg.batch_size, seq_len, self.rng
                )
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x, y)
                loss = out.loss / cfg.grad_accum
                loss.backward()
                loss_accum += float(out.loss.detach())

                self.tokens_seen += x.numel()

            # ---- optimize ----
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # ---- logging ----
            if self.step % cfg.log_every == 0:
                now = time.time()
                dt = now - t_last
                tok_per_s = (self.tokens_seen - tokens_last) / max(dt, 1e-6)
                eta_steps = target_steps - self.step
                eta = eta_steps * dt / max(1, cfg.log_every)
                print(
                    f"step {self.step:>7,}/{target_steps:,} | "
                    f"loss {loss_accum / cfg.grad_accum:6.4f} | "
                    f"ppl {math.exp(min(20, loss_accum / cfg.grad_accum)):8.2f} | "
                    f"lr {lr:.2e} | "
                    f"gnorm {float(grad_norm):6.2f} | "
                    f"{tok_per_s:7.1f} tok/s | "
                    f"eta {human_time(eta)}"
                )
                self.logger.log(
                    event="train",
                    step=self.step,
                    loss=loss_accum / cfg.grad_accum,
                    lr=lr,
                    grad_norm=float(grad_norm),
                    tok_per_s=tok_per_s,
                    tokens=self.tokens_seen,
                    elapsed=now - t_start,
                )
                t_last = now
                tokens_last = self.tokens_seen

            # ---- eval ----
            if (
                self.val_corpus is not None
                and self.step % cfg.eval_every == 0
            ):
                val = self.evaluate()
                print(f"    -> val loss {val:.4f} | ppl {math.exp(min(20, val)):.2f}")
                self.logger.log(
                    event="val", step=self.step, val_loss=val,
                    tokens=self.tokens_seen,
                )
                if val < best_val:
                    best_val = val
                    # Weights only -- this checkpoint exists to be
                    # generated from, not to resume from.
                    self.save_checkpoint("best.pt", val_loss=val,
                                         include_optimizer=False)

            # ---- checkpoint ----
            if self.step % cfg.ckpt_every == 0:
                self.save_checkpoint("last.pt")
                self.save_checkpoint(f"step_{self.step:07d}.pt")
                self.prune_checkpoints(keep=3)

        # ---- final ----
        val = self.evaluate()
        best_val = min(best_val, val)
        print(f"\nDone. final val loss {val:.4f} | best {best_val:.4f}")
        self.logger.log(event="final", step=self.step, val_loss=val)
        self.save_checkpoint("last.pt", val_loss=val)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        name: str,
        val_loss: Optional[float] = None,
        include_optimizer: bool = True,
    ) -> Path:
        """Write a checkpoint.

        `include_optimizer=False` saves only the weights (plus config).
        For the 126M model the optimizer state is 1GB of AdamW moments,
        which we do not need in a checkpoint we only intend to generate
        from -- so `best.pt` skips it.
        """
        path = self.out_dir / name
        payload = {
            "model": self.model.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "model_config": self.model.cfg.to_dict(),
            "train_config": vars(self.cfg),
            "val_loss": val_loss,
        }
        if include_optimizer:
            payload["optimizer"] = self.optimizer.state_dict()
            payload["rng_state"] = self.rng.bit_generator.state
        torch.save(payload, path)
        print(f"    [ckpt] {path}")
        return path

    def prune_checkpoints(self, keep: int = 3) -> None:
        """Delete all but the most recent `keep` numbered checkpoints.

        Each numbered checkpoint is ~1.5GB for the 126M model (weights +
        optimizer). Over a multi-day run that adds up, and you almost
        never want more than the last couple.
        """
        ckpts = sorted(self.out_dir.glob("step_*.pt"))
        for old in ckpts[:-keep] if len(ckpts) > keep else []:
            try:
                old.unlink()
                print(f"    [ckpt] pruned {old.name}")
            except OSError:
                pass

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        # A weights-only checkpoint (best.pt) has no optimizer state.
        # That is fine for generating from, but not for resuming, so we
        # reset the step counter in that case.
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            self.step = payload.get("step", 0)
            self.tokens_seen = payload.get("tokens_seen", 0)
            if "rng_state" in payload:
                self.rng.bit_generator.state = payload["rng_state"]
            print(f"[resume] loaded {path} at step {self.step:,}")
        else:
            print(f"[load] loaded weights from {path} "
                  f"(no optimizer state -- starting a fresh schedule)")


def find_latest_checkpoint(out_dir: str | Path) -> Optional[Path]:
    out_dir = Path(out_dir)
    last = out_dir / "last.pt"
    if last.exists():
        return last
    candidates = sorted(out_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None
