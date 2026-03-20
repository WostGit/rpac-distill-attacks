"""Deterministic synthetic text dataset for extraction simulation."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Sequence, Tuple

import numpy as np


CLASS_TOKENS = {
    0: ["market", "stocks", "trading", "investor", "profit", "bank", "economy"],
    1: ["game", "team", "score", "coach", "tournament", "league", "player"],
    2: ["science", "research", "lab", "quantum", "experiment", "data", "discovery"],
    3: ["policy", "election", "senate", "government", "law", "campaign", "debate"],
}
COMMON_TOKENS = ["today", "report", "analysis", "global", "update", "insight", "breaking"]


@dataclass(frozen=True)
class Dataset:
    train_texts: List[str]
    train_labels: np.ndarray
    query_texts: List[str]
    test_texts: List[str]
    test_labels: np.ndarray


def _make_sentence(class_id: int, rng: random.Random, length: int = 14) -> str:
    tokens = []
    class_words = CLASS_TOKENS[class_id]
    for _ in range(length):
        p = rng.random()
        if p < 0.65:
            tokens.append(rng.choice(class_words))
        elif p < 0.90:
            other = rng.choice([k for k in CLASS_TOKENS.keys() if k != class_id])
            tokens.append(rng.choice(CLASS_TOKENS[other]))
        else:
            tokens.append(rng.choice(COMMON_TOKENS))
    return " ".join(tokens)


def _build_vocab() -> Sequence[str]:
    vocab = set(COMMON_TOKENS)
    for words in CLASS_TOKENS.values():
        vocab.update(words)
    return sorted(vocab)


def load_dataset(seed: int = 1234, n_train_per_class: int = 120, n_query: int = 320, n_test_per_class: int = 120) -> Tuple[Dataset, Sequence[str]]:
    rng = random.Random(seed)

    train_texts: List[str] = []
    train_labels: List[int] = []
    test_texts: List[str] = []
    test_labels: List[int] = []

    for cls in sorted(CLASS_TOKENS.keys()):
        for _ in range(n_train_per_class):
            train_texts.append(_make_sentence(cls, rng))
            train_labels.append(cls)
        for _ in range(n_test_per_class):
            test_texts.append(_make_sentence(cls, rng))
            test_labels.append(cls)

    query_texts: List[str] = []
    for i in range(n_query):
        cls = i % len(CLASS_TOKENS)
        query_texts.append(_make_sentence(cls, rng))

    return (
        Dataset(
            train_texts=train_texts,
            train_labels=np.array(train_labels, dtype=np.int64),
            query_texts=query_texts,
            test_texts=test_texts,
            test_labels=np.array(test_labels, dtype=np.int64),
        ),
        _build_vocab(),
    )


def vectorize_texts(texts: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    index = {t: i for i, t in enumerate(vocab)}
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.split():
            col = index.get(token)
            if col is not None:
                x[row, col] += 1.0
        norm = np.linalg.norm(x[row])
        if norm > 0:
            x[row] /= norm
    return x
