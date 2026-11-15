#!/usr/bin/env python3
"""
Print an ASCII loss curve for a run.

During a multi-day run you want to see the shape of the loss curve
without opening a plotting library. This reads metrics.jsonl (written by
the trainer) and renders it as text.

It also prints the things that actually tell you whether training is
healthy:

  - is the loss still going down, or has it plateaued?
  - is the train/val gap growing? (that is overfitting)
  - is the gradient norm pinned at the clip value? (learning rate too high)
  - what is the projected finish time?

Usage:
    python scripts/09_plot.py --run runs/small
    python scripts/09_plot.py --run runs/small --width 100 --height 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    train, val = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            (train if rec.get("event") == "train" else
             val if rec.get("event") == "val" else []).append(rec)
    return train, val


def ascii_chart(
    points: list[tuple[float, float]],
    width: int = 90,
    height: int = 22,
    x_label: str = "step",
    y_label: str = "loss",
) -> str:
    """Render (x, y) points as an ASCII scatter plot."""
    if not points:
        return "(no data)"

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    grid = [[" "] * width for _ in range(height)]

    def put(px: int, py: int, ch: str) -> None:
        if 0 <= px < width and 0 <= py < height:
            grid[py][px] = ch

    # axes
    for i in range(height):
        grid[i][0] = "|"
    for j in range(width):
        grid[height - 1][j] = "-"
    put(0, height - 1, "+")

    for x, y in points:
        px = int((x - x_min) / (x_max - x_min) * (width - 2))
        py = int((y_max - y) / (y_max - y_min) * (height - 2))
        put(px + 1, py, "*")

    lines = []
    for i, row in enumerate(grid):
        label = ""
        if i == 0:
            label = f"{y_max:7.3f} |"
        elif i == height - 1:
            label = f"{y_min:7.3f} |"
        else:
            label = " " * 8 + "|"
        lines.append(label + "".join(row))

    lines.append(" " * 8 + "+" + "-" * (width - 2))
    mid = width // 2
    lines.append(" " * (8 + mid - 6) + f"{x_label} {x_min:.0f}..{x_max:.0f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/small")
    ap.add_argument("--width", type=int, default=90)
    ap.add_argument("--height", type=int, default=22)
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    try:
        train, val = load_metrics(run_dir)
    except FileNotFoundError as e:
        print(e)
        return 1

    if not train:
        print(f"{run_dir}/metrics.jsonl has no training records yet")
        return 1

    print(f"run: {run_dir}")
    print(f"train records: {len(train)} | val records: {len(val)}")
    print()

    # ---- curve ---------------------------------------------------------
    print("TRAIN LOSS")
    print(ascii_chart([(r["step"], r["loss"]) for r in train],
                      args.width, args.height))
    print()

    if val:
        print("VALIDATION LOSS")
        print(ascii_chart([(r["step"], r["val_loss"]) for r in val],
                          args.width, max(6, args.height // 2)))
        print()

    # ---- health ------------------------------------------------------
    last = train[-1]
    first = train[0]
    print("-" * 60)
    print("HEALTH CHECK")
    print("-" * 60)
    print(f"  step          : {last['step']:,}")
    print(f"  tokens seen   : {last.get('tokens', 0):,}")
    print(f"  train loss    : {first['loss']:.4f} -> {last['loss']:.4f}")
    if val:
        v_first, v_last = val[0]["val_loss"], val[-1]["val_loss"]
        print(f"  val loss      : {v_first:.4f} -> {v_last:.4f}")
        gap = last["loss"] - v_last
        print(f"  train-val gap : {gap:+.4f}"
              f"  {'(OK)' if gap < 0.5 else '(WATCH: train is ahead -- overfitting?)'}")
    print(f"  lr            : {last.get('lr', 0):.3e}")
    print(f"  grad norm     : {last.get('grad_norm', 0):.3f}")

    clipped = sum(1 for r in train if r.get("grad_norm", 0) >= 1.0)
    frac = clipped / max(1, len(train))
    print(f"  grad clip hits: {clipped}/{len(train)} steps ({frac:.0%})")
    if frac > 0.7 and len(train) > 10:
        print("  WARNING: the gradient norm exceeds the clip value on most")
        print("           steps. The learning rate is probably too high, or")
        print("           the loss is diverging -- check the curve above.")
    elif frac > 0.1:
        print("  (some clipping -- normal early in training, watch for growth)")

    if last.get("tok_per_s"):
        tps = last["tok_per_s"]
        print(f"  throughput    : {tps:.0f} tok/s")

    # ---- ETA ------------------------------------------------------------
    total = 60_000_000  # fallback if the run did not record its config
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            total = json.loads(cfg_path.read_text()).get("max_tokens", total)
        except json.JSONDecodeError:
            pass
    seen = last.get("tokens", 0)
    tps = last.get("tok_per_s", 0)
    if seen and tps > 0:
        remaining = max(0, total - seen)
        hours = remaining / tps / 3600
        print(f"\n  progress    : {seen / 1e6:.1f}M / {total / 1e6:.0f}M tokens "
              f"({100 * seen / total:.1f}%)")
        print(f"  ETA         : {hours:.1f} hours  ({hours / 24:.1f} days)")

    # ---- perplexity ------------------------------------------------------
    if val:
        print(f"\n  current val perplexity: {math.exp(min(20, val[-1]['val_loss'])):.2f}")
    print(f"  current train perplexity: {math.exp(min(20, last['loss'])):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
