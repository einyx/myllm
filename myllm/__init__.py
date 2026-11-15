"""
myllm -- a language model built from scratch.

    from myllm import ModelConfig, Transformer, BPETokenizer, TokenCorpus
"""

from .config import ModelConfig, TrainConfig, get_preset
from .model import Transformer, KVCache
from .tokenizer import BPETokenizer
from .data import TokenCorpus, tokenize_text_file
from .train import Trainer
from .generate import generate, generate_batch, perplexity

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "get_preset",
    "Transformer",
    "KVCache",
    "BPETokenizer",
    "TokenCorpus",
    "tokenize_text_file",
    "Trainer",
    "generate",
    "generate_batch",
    "perplexity",
]

__version__ = "0.1.0"
