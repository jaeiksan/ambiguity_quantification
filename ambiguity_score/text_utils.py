import hashlib
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    tokens = TOKEN_RE.findall(text)
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def cosine_similarity(text_a: str, text_b: str) -> float:
    counts_a = Counter(tokenize(text_a))
    counts_b = Counter(tokenize(text_b))
    if not counts_a and not counts_b:
        return 1.0
    if not counts_a or not counts_b:
        return 0.0
    dot = sum(counts_a[token] * counts_b[token] for token in counts_a.keys() & counts_b.keys())
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def sequence_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(a=normalize_text(text_a), b=normalize_text(text_b)).ratio()


def pairwise_average_similarity(texts: Sequence[str]) -> float:
    if len(texts) < 2:
        return 1.0
    scores: List[float] = []
    for idx, text_a in enumerate(texts):
        for jdx in range(idx + 1, len(texts)):
            text_b = texts[jdx]
            lexical = cosine_similarity(text_a, text_b)
            surface = sequence_similarity(text_a, text_b)
            scores.append((lexical + surface) / 2.0)
    return sum(scores) / len(scores) if scores else 1.0


def shannon_entropy_from_counts(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    entropy = 0.0
    for value in values:
        probability = value / total
        entropy -= probability * math.log(probability)
    return entropy


def stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine_similarity_from_vectors(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
