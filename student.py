"""Student model trained only on teacher outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def _tokenize(text: str) -> list[str]:
    return text.split()


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


@dataclass
class StudentModel:
    vocab: dict[str, int]
    weights: list[list[float]]

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * len(self.vocab)
        for tok in _tokenize(text):
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        norm = sum(vec) or 1.0
        return [v / norm for v in vec]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            x = self._vectorize(text)
            logits = []
            for c in range(len(self.weights[0])):
                logits.append(sum(x[i] * self.weights[i][c] for i in range(len(self.weights))))
            result.append(_softmax(logits))
        return result

    def predict(self, texts: list[str]) -> list[int]:
        return [max(range(len(p)), key=lambda i: p[i]) for p in self.predict_proba(texts)]


def train_student(
    query_texts: list[str],
    teacher_probs: list[list[float]],
    seed: int = 1234,
    lr: float = 0.2,
    steps: int = 100,
    batch_size: int = 32,
    model_dim: int = 512,
) -> StudentModel:
    rng = random.Random(seed)

    vocab: dict[str, int] = {}
    for text in query_texts:
        for tok in _tokenize(text):
            if tok not in vocab and len(vocab) < model_dim:
                vocab[tok] = len(vocab)

    if not vocab:
        raise ValueError("Empty vocabulary from query set")

    num_classes = len(teacher_probs[0])
    weights = [[0.0 for _ in range(num_classes)] for _ in range(len(vocab))]

    model = StudentModel(vocab=vocab, weights=weights)
    n = len(query_texts)

    for _ in range(steps):
        idxs = list(range(n))
        rng.shuffle(idxs)
        idxs = idxs[: min(batch_size, n)]
        grad = [[0.0 for _ in range(num_classes)] for _ in range(len(vocab))]
        for idx in idxs:
            x = model._vectorize(query_texts[idx])
            y = teacher_probs[idx]
            pred = model.predict_proba([query_texts[idx]])[0]
            for i in range(len(vocab)):
                for c in range(num_classes):
                    grad[i][c] += x[i] * (pred[c] - y[c])

        scale = 1.0 / max(len(idxs), 1)
        for i in range(len(vocab)):
            for c in range(num_classes):
                weights[i][c] -= lr * grad[i][c] * scale
                if not math.isfinite(weights[i][c]):
                    raise FloatingPointError("Student training diverged with non-finite weights")

    return model
