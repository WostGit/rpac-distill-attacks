"""Deterministic synthetic text dataset for distillation experiments."""

from __future__ import annotations

from dataclasses import dataclass
import random

CLASS_VOCAB = {
    0: ["stocks", "market", "trade", "profit", "finance", "earnings"],
    1: ["match", "team", "score", "league", "coach", "season"],
    2: ["device", "software", "network", "ai", "chip", "robot"],
    3: ["election", "policy", "senate", "debate", "law", "government"],
}
NOISE_WORDS = ["today", "report", "analysis", "global", "update", "insight"]


@dataclass(frozen=True)
class DatasetSplits:
    x_train: list[str]
    y_train: list[int]
    x_query: list[str]
    x_test: list[str]
    y_test: list[int]


def _make_sentence(rng: random.Random, label: int) -> str:
    topical = [rng.choice(CLASS_VOCAB[label]) for _ in range(6)]
    noise = [rng.choice(NOISE_WORDS) for _ in range(3)]
    tokens = topical + noise
    rng.shuffle(tokens)
    return " ".join(tokens)


def build_splits(seed: int = 1234, train_per_class: int = 120, query_size: int = 180, test_per_class: int = 80) -> DatasetSplits:
    rng = random.Random(seed)
    x_train: list[str] = []
    y_train: list[int] = []
    x_test: list[str] = []
    y_test: list[int] = []

    for label in sorted(CLASS_VOCAB):
        for _ in range(train_per_class):
            x_train.append(_make_sentence(rng, label))
            y_train.append(label)
        for _ in range(test_per_class):
            x_test.append(_make_sentence(rng, label))
            y_test.append(label)

    x_query: list[str] = []
    labels = list(CLASS_VOCAB.keys())
    for _ in range(query_size):
        label = rng.choice(labels)
        x_query.append(_make_sentence(rng, label))

    return DatasetSplits(x_train=x_train, y_train=y_train, x_query=x_query, x_test=x_test, y_test=y_test)
