"""Small shared helpers: seeding, logging, checkpoints."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make a run reproducible.

    Note: on CPU with many threads, bit-exact reproducibility is not
    guaranteed (floating-point reduction order can vary with thread
    scheduling). But the *data order* and the *initialization* will be
    identical, which is what matters for comparing two runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_threads(n: int = 0) -> int:
    """Set the number of CPU threads torch uses.

    On a 32-core machine the default (24 here) is usually fine. Setting it
    too high can actually be slower due to contention.
    """
    if n > 0:
        torch.set_num_threads(n)
    return torch.get_num_threads()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600:02d}h"


class JsonlLogger:
    """Append-only JSON-lines log of training metrics.

    Reading it back with `json.loads` per line gives you a table you can
    plot. That is much easier to reason about than scraping stdout.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, **kwargs: Any) -> None:
        kwargs.setdefault("wall", time.time())
        self._fh.write(json.dumps(kwargs) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Timer:
    """Context manager that reports how long a block took."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self._t0
