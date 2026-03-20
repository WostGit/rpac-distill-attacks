"""Teacher model used as extraction target."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def _tokenize(text: str) -> list[str]:
    return text.split()


@dataclass
class TeacherModel:
    classes: list[int]
    class_log_prior: dict[int, float]
    token_log_probs: dict[int, dict[str, float]]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for text in texts:
            counts = Counter(_tokenize(text))
            logits: list[float] = []
            for c in self.classes:
                score = self.class_log_prior[c]
                token_lp = self.token_log_probs[c]
                for tok, cnt in counts.items():
                    score += cnt * token_lp.get(tok, -12.0)
                logits.append(score)
            outputs.append(_softmax(logits))
        return outputs

    def predict(self, texts: list[str]) -> list[int]:
        return [max(range(len(p)), key=lambda i: p[i]) for p in self.predict_proba(texts)]


def train_teacher(texts: list[str], labels: list[int], seed: int = 1234) -> TeacherModel:
    del seed
    classes = sorted(set(labels))
    doc_counts = Counter(labels)
    total_docs = len(labels)

    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    total_tokens = Counter()
    vocab = set()
    for text, y in zip(texts, labels):
        toks = _tokenize(text)
        token_counts[y].update(toks)
        total_tokens[y] += len(toks)
        vocab.update(toks)

    vocab_size = max(len(vocab), 1)
    class_log_prior = {c: math.log((doc_counts[c] + 1) / (total_docs + len(classes))) for c in classes}
    token_log_probs: dict[int, dict[str, float]] = {}
    for c in classes:
        denom = total_tokens[c] + vocab_size
        token_log_probs[c] = {tok: math.log((token_counts[c][tok] + 1) / denom) for tok in vocab}

    return TeacherModel(classes=classes, class_log_prior=class_log_prior, token_log_probs=token_log_probs)
