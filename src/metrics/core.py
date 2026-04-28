
from __future__ import annotations
"""Core metric functions for evolutionary LLM experiments."""

def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between token sets of two texts.
    Uses simple whitespace tokenization (consistent with I(X;Y) proxy).
    Returns float in [0, 1].
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

import gzip
import math
from collections import Counter


def shannon_entropy(text: str) -> float:
    """Compute token-level Shannon entropy for whitespace tokens.

    Parameters
    ----------
    text : str
        Input text to evaluate.

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    tokens = text.split()
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def effective_complexity(text: str) -> float:
    """Estimate effective complexity as compression ratio.

    Parameters
    ----------
    text : str
        Input text to compress.

    Returns
    -------
    float
        len(gzip(text)) / len(text_bytes). Returns 0.0 for empty input.
    """
    if not text:
        return 0.0

    raw = text.encode("utf-8")
    compressed = gzip.compress(raw)
    return len(compressed) / len(raw)


def fitness_score(
    complexity: float,
    mutual_info: float,
    disorganization: float,
    w1: float,
    w2: float,
    w3: float,
) -> float:
    """Compute scalar fitness from weighted metric terms.

    Parameters
    ----------
    complexity : float
        Effective complexity term C(X).
    mutual_info : float
        Mutual-information proxy term I(X;seed).
    disorganization : float
        Disorganization entropy term H_dezorg.
    w1 : float
        Weight for complexity term.
    w2 : float
        Weight for mutual-information term.
    w3 : float
        Weight for disorganization term.

    Returns
    -------
    float
        Fitness computed as w1*C + w2*I - w3*H_dezorg.
    """
    return w1 * complexity + w2 * mutual_info - w3 * disorganization


def mutual_information_proxy(seed_text: str, output_text: str) -> float:
    """Approximate mutual information using cosine similarity.

    Parameters
    ----------
    seed_text : str
        Seed prompt or reference text.
    output_text : str
        Candidate output text.

    Returns
    -------
    float
        Cosine similarity in range [0, 1] for non-negative token counts.
    """
    seed_counts = Counter(seed_text.lower().split())
    output_counts = Counter(output_text.lower().split())
    if not seed_counts or not output_counts:
        return 0.0

    all_tokens = set(seed_counts) | set(output_counts)
    dot = 0.0
    seed_norm = 0.0
    output_norm = 0.0
    for token in all_tokens:
        a = float(seed_counts.get(token, 0))
        b = float(output_counts.get(token, 0))
        dot += a * b
        seed_norm += a * a
        output_norm += b * b

    if seed_norm == 0.0 or output_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(seed_norm) * math.sqrt(output_norm))


def disorganization_entropy(text: str) -> float:
    """Estimate coherence degradation from sentence-length disorder.

    Parameters
    ----------
    text : str
        Input text with one or more sentences.

    Returns
    -------
    float
        Normalized entropy in range [0, 1].
    """
    if not text.strip():
        return 0.0

    normalized = text.replace("!", ".").replace("?", ".")
    sentence_lengths = [len(sentence.split()) for sentence in normalized.split(".") if sentence.strip()]
    if len(sentence_lengths) <= 1:
        return 0.0

    total = sum(sentence_lengths)
    if total == 0:
        return 0.0

    entropy = 0.0
    for sentence_len in sentence_lengths:
        p = sentence_len / total
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(sentence_lengths))
    if max_entropy == 0.0:
        return 0.0
    return entropy / max_entropy
