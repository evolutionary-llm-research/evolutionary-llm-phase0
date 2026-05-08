from __future__ import annotations

"""Mutual-information calibration for Phase 0 outputs.

This module compares multiple MI-like similarity functions on saved
``model_output`` strings to select the most suitable implementation for
``I(X;seed)``.
"""

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

try:
    from scipy.stats import mannwhitneyu, pearsonr
except Exception:  # pragma: no cover - fallback is used when scipy is unavailable.
    mannwhitneyu = None
    pearsonr = None


MIType = Callable[[str, str], float]
CLASS_ORDER = ("food", "toxin", "noise")
_TOKENIZER = None


def _get_tokenizer() -> Any:
    """Load and cache the model tokenizer for BPE-based MI evaluation."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            "unsloth/qwen3-8b-base-unsloth-bnb-4bit",
            trust_remote_code=True,
        )
    return _TOKENIZER


def _clamp_01(value: float) -> float:
    """Clamp a numeric value to [0, 1]."""
    return max(0.0, min(1.0, value))


def _token_counts_lower(text: str) -> Counter[str]:
    """Return lowercased whitespace-token counts."""
    return Counter(text.lower().split())


def _token_counts(text: str) -> Counter[str]:
    """Return whitespace-token counts preserving case."""
    return Counter(text.split())


def _entropy_from_counts(counts: Counter[str]) -> float:
    """Compute Shannon entropy (bits) from token counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _shannon_entropy(text: str) -> float:
    """Compute token-level Shannon entropy using whitespace tokenization."""
    return _entropy_from_counts(_token_counts(text))


def _average_ranks(values: list[float]) -> list[float]:
    """Compute average ranks with tie handling (1-indexed ranks)."""
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    return ranks


def _mann_whitney_fallback(x: list[float], y: list[float]) -> tuple[float, float]:
    """Fallback Mann-Whitney U test with normal approximation p-value."""
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return 0.0, float("nan")

    combined = x + y
    ranks = _average_ranks(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1) / 2.0)

    n = n1 + n2
    tie_counts = Counter(combined)
    tie_term = sum(t * t * t - t for t in tie_counts.values())
    if n <= 1:
        return u1, float("nan")

    variance = (n1 * n2 / 12.0) * ((n + 1) - (tie_term / (n * (n - 1))))
    if variance <= 0.0:
        return u1, 1.0

    sigma = math.sqrt(variance)
    mu = n1 * n2 / 2.0
    z = (abs(u1 - mu) - 0.5) / sigma
    p_value = math.erfc(z / math.sqrt(2.0))
    return u1, _clamp_01(p_value)


def _rank_biserial_and_pvalue(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return rank-biserial effect size and Mann-Whitney p-value."""
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    if mannwhitneyu is not None:
        test = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        u_value = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        u_value, p_value = _mann_whitney_fallback(x, y)

    rank_biserial = (2.0 * u_value) / (n1 * n2) - 1.0
    return rank_biserial, p_value


def _safe_mean(values: list[float]) -> float:
    """Return arithmetic mean or NaN for an empty list."""
    return float(mean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    """Return population std (pstdev) or NaN for lists with <2 items."""
    return float(pstdev(values)) if len(values) >= 2 else float("nan")


def mi_cosine(seed_text: str, output_text: str) -> float:
    """Cosine similarity over bag-of-words counts.

    This matches the existing implementation used by ``mutual_information_proxy``
    in ``src/metrics/core.py``.
    """
    seed_counts = _token_counts_lower(seed_text)
    output_counts = _token_counts_lower(output_text)
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
    return _clamp_01(dot / (math.sqrt(seed_norm) * math.sqrt(output_norm)))


def mi_entropy_decomp(seed_text: str, output_text: str) -> float:
    """Entropy-decomposition MI normalized by min(H(X), H(Y)).

    I(X;Y) = H(X) + H(Y) - H(X,Y)
    score = I(X;Y) / min(H(X), H(Y))
    """
    h_x = _shannon_entropy(seed_text)
    h_y = _shannon_entropy(output_text)
    min_h = min(h_x, h_y)
    if min_h == 0.0:
        return 0.0

    h_xy = _shannon_entropy(seed_text + " " + output_text)
    mi_value = h_x + h_y - h_xy
    return _clamp_01(mi_value / min_h)


def mi_jsd(seed_text: str, output_text: str) -> float:
    """Similarity derived from Jensen-Shannon divergence.

    Computes ``1 - JSD(P||Q)`` over token-frequency distributions with Laplace
    smoothing (add-one counts).
    """
    seed_counts = _token_counts_lower(seed_text)
    output_counts = _token_counts_lower(output_text)
    if not seed_counts or not output_counts:
        return 0.0

    if not (set(seed_counts) & set(output_counts)):
        return 0.0

    vocab = set(seed_counts) | set(output_counts)
    v_size = len(vocab)
    if v_size == 0:
        return 0.0

    seed_total = sum(seed_counts.values()) + v_size
    output_total = sum(output_counts.values()) + v_size

    jsd = 0.0
    for token in vocab:
        p = (seed_counts.get(token, 0) + 1.0) / seed_total
        q = (output_counts.get(token, 0) + 1.0) / output_total
        m = 0.5 * (p + q)
        jsd += 0.5 * p * math.log2(p / m) + 0.5 * q * math.log2(q / m)

    return _clamp_01(1.0 - jsd)


def evaluate_ld50_gradient(
    ld50_run_paths: dict[str, str],
    seed_text: str,
    mi_func: MIType,
) -> dict[str, Any]:
    """Evaluate monotonic MI gradient across LD50 concentrations.

    Parameters
    ----------
    ld50_run_paths : dict[str, str]
        Mapping concentration percentage to metrics JSON path.
    seed_text : str
        Stable seed reference text.
    mi_func : MIType
        MI-like scoring function.

    Returns
    -------
    dict[str, Any]
        Pearson correlation summary and per-concentration mean MI values.
    """
    concentration_values: list[float] = []
    mean_values: list[float] = []
    values_per_concentration: dict[str, float] = {}

    for concentration_str, metrics_path_str in sorted(
        ld50_run_paths.items(), key=lambda pair: float(pair[0])
    ):
        metrics_path = Path(metrics_path_str)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "results" in payload:
            rows = payload["results"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(
                f"Unsupported metrics JSON format in {metrics_path}: expected dict with 'results' or list"
            )

        scores: list[float] = []
        for row in rows:
            output_text = str(row.get("model_output", ""))
            scores.append(float(mi_func(seed_text, output_text)))

        mean_mi = _safe_mean(scores)
        values_per_concentration[concentration_str] = mean_mi
        concentration_values.append(float(concentration_str))
        mean_values.append(mean_mi)

    finite_pairs = [
        (x, y)
        for x, y in zip(concentration_values, mean_values)
        if math.isfinite(x) and math.isfinite(y)
    ]

    if len(finite_pairs) < 2:
        r_value = float("nan")
        p_value = float("nan")
    else:
        xs = [pair[0] for pair in finite_pairs]
        ys = [pair[1] for pair in finite_pairs]
        if pearsonr is not None:
            corr = pearsonr(xs, ys)
            r_value = float(corr.statistic)
            p_value = float(corr.pvalue)
        else:
            x_mean = float(mean(xs))
            y_mean = float(mean(ys))
            cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            x_var = sum((x - x_mean) ** 2 for x in xs)
            y_var = sum((y - y_mean) ** 2 for y in ys)
            if x_var <= 0.0 or y_var <= 0.0:
                r_value = 0.0
            else:
                r_value = cov / math.sqrt(x_var * y_var)
            p_value = float("nan")

    if math.isfinite(r_value) and r_value < -0.8:
        direction = "correct"
    elif math.isfinite(r_value) and r_value > 0.8:
        direction = "reversed"
    else:
        direction = "flat"

    return {
        "r": r_value,
        "p": p_value,
        "direction": direction,
        "values_per_concentration": values_per_concentration,
    }


def mi_npmi(seed_text: str, output_text: str) -> float:
    """Weighted mean NPMI over shared tokens.

    For each shared token ``t``:

    PMI(t) = log2(P(t|joint) / (P(t|seed) * P(t|output)))
    NPMI(t) = PMI(t) / (-log2(P(t|joint)))

    Aggregation is weighted by token joint frequency. Returns 0.0 if there are
    no shared tokens.
    """
    seed_counts = _token_counts_lower(seed_text)
    output_counts = _token_counts_lower(output_text)
    if not seed_counts or not output_counts:
        return 0.0

    shared = set(seed_counts) & set(output_counts)
    if not shared:
        return 0.0

    seed_total = float(sum(seed_counts.values()))
    output_total = float(sum(output_counts.values()))
    joint_counts = {token: seed_counts[token] + output_counts[token] for token in shared}
    joint_total = float(sum(seed_counts.values()) + sum(output_counts.values()))

    weighted_sum = 0.0
    total_weight = 0.0
    for token in shared:
        p_joint = joint_counts[token] / joint_total
        p_seed = seed_counts[token] / seed_total
        p_output = output_counts[token] / output_total
        if p_joint <= 0.0 or p_seed <= 0.0 or p_output <= 0.0:
            continue

        pmi = math.log2(p_joint / (p_seed * p_output))
        denom = -math.log2(p_joint)
        if denom <= 0.0:
            continue
        npmi = pmi / denom

        weight = float(joint_counts[token])
        weighted_sum += npmi * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return _clamp_01(weighted_sum / total_weight)


def mi_token_ids(seed_text: str, output_text: str) -> float:
    """Positional mutual information on token IDs.

    Tokenization uses whitespace splitting to produce token sequences.
    Both sequences are truncated to ``min(len(seed_tokens), len(output_tokens))``.
    Mutual information is computed with ``sklearn.metrics.mutual_info_score`` and
    normalized by ``log(n_unique_tokens)`` to map scores into [0, 1].

    Returns 0.0 when either text is empty, when aligned length is < 2, or when
    the normalization denominator is not positive.
    """
    from sklearn.metrics import mutual_info_score
    import numpy as np

    seed_tokens = seed_text.split()
    output_tokens = output_text.split()
    if not seed_tokens or not output_tokens:
        return 0.0

    min_len = min(len(seed_tokens), len(output_tokens))
    if min_len < 2:
        return 0.0

    seed_tokens = seed_tokens[:min_len]
    output_tokens = output_tokens[:min_len]

    seed_ids = {token: idx for idx, token in enumerate(sorted(set(seed_tokens)))}
    output_ids = {token: idx for idx, token in enumerate(sorted(set(output_tokens)))}

    x = np.array([seed_ids[token] for token in seed_tokens], dtype=int)
    y = np.array([output_ids[token] for token in output_tokens], dtype=int)

    mi_value = float(mutual_info_score(x, y))
    n_unique_tokens = max(len(set(seed_tokens) | set(output_tokens)), 1)
    denom = float(np.log(n_unique_tokens))
    if denom <= 0.0:
        return 0.0

    return _clamp_01(mi_value / denom)


def mi_token_ids_nmi(seed_text: str, output_text: str) -> float:
    """Normalized Mutual Information on whitespace token sequences.

    Uses the same aligned positional token-ID setup as ``mi_token_ids`` but
    normalizes by ``sqrt(H(X) * H(Y))`` where entropies are computed with
    ``scipy.stats.entropy``.
    """
    from scipy.stats import entropy
    from sklearn.metrics import mutual_info_score
    import numpy as np

    seed_tokens = seed_text.split()
    output_tokens = output_text.split()
    if not seed_tokens or not output_tokens:
        return 0.0

    min_len = min(len(seed_tokens), len(output_tokens))
    if min_len < 2:
        return 0.0

    seed_tokens = seed_tokens[:min_len]
    output_tokens = output_tokens[:min_len]

    seed_ids = {token: idx for idx, token in enumerate(sorted(set(seed_tokens)))}
    output_ids = {token: idx for idx, token in enumerate(sorted(set(output_tokens)))}

    x = np.array([seed_ids[token] for token in seed_tokens], dtype=int)
    y = np.array([output_ids[token] for token in output_tokens], dtype=int)

    mi_value = float(mutual_info_score(x, y))

    x_counts = np.bincount(x)
    y_counts = np.bincount(y)
    h_x = float(entropy(x_counts / x_counts.sum())) if x_counts.sum() > 0 else 0.0
    h_y = float(entropy(y_counts / y_counts.sum())) if y_counts.sum() > 0 else 0.0
    if h_x <= 0.0 or h_y <= 0.0:
        return 0.0

    denom = math.sqrt(h_x * h_y)
    if denom <= 0.0:
        return 0.0

    return _clamp_01(mi_value / denom)


def mi_token_ids_bigrams(seed_text: str, output_text: str) -> float:
    """Positional MI on whitespace bigram sequences.

    Builds consecutive token-pair sequences, aligns lengths, computes
    ``mutual_info_score`` on bigram IDs, and normalizes by
    ``log(n_unique_bigrams)``.
    """
    from sklearn.metrics import mutual_info_score
    import numpy as np

    seed_tokens = seed_text.split()
    output_tokens = output_text.split()
    if len(seed_tokens) < 2 or len(output_tokens) < 2:
        return 0.0

    min_token_len = min(len(seed_tokens), len(output_tokens))
    if min_token_len < 3:
        return 0.0

    seed_tokens = seed_tokens[:min_token_len]
    output_tokens = output_tokens[:min_token_len]

    seed_bigrams = list(zip(seed_tokens[:-1], seed_tokens[1:]))
    output_bigrams = list(zip(output_tokens[:-1], output_tokens[1:]))

    min_bigram_len = min(len(seed_bigrams), len(output_bigrams))
    if min_bigram_len < 2:
        return 0.0

    seed_bigrams = seed_bigrams[:min_bigram_len]
    output_bigrams = output_bigrams[:min_bigram_len]

    seed_ids = {bigram: idx for idx, bigram in enumerate(sorted(set(seed_bigrams)))}
    output_ids = {bigram: idx for idx, bigram in enumerate(sorted(set(output_bigrams)))}

    x = np.array([seed_ids[bigram] for bigram in seed_bigrams], dtype=int)
    y = np.array([output_ids[bigram] for bigram in output_bigrams], dtype=int)

    mi_value = float(mutual_info_score(x, y))
    n_unique_bigrams = max(len(set(seed_bigrams) | set(output_bigrams)), 1)
    denom = float(np.log(n_unique_bigrams))
    if denom <= 0.0:
        return 0.0

    return _clamp_01(mi_value / denom)


def mi_token_ids_bpe(seed_text: str, output_text: str) -> float:
    """Positional MI on BPE token ID sequences from the model tokenizer."""
    from sklearn.metrics import mutual_info_score
    import numpy as np

    tokenizer = _get_tokenizer()
    seed_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
    if not seed_ids or not output_ids:
        return 0.0

    min_len = min(len(seed_ids), len(output_ids))
    if min_len < 2:
        return 0.0

    x = np.array(seed_ids[:min_len], dtype=int)
    y = np.array(output_ids[:min_len], dtype=int)

    mi_value = float(mutual_info_score(x, y))
    n_unique_bpe_tokens = max(len(set(x.tolist()) | set(y.tolist())), 1)
    denom = float(np.log(n_unique_bpe_tokens))
    if denom <= 0.0:
        return 0.0

    return _clamp_01(mi_value / denom)


def evaluate_mi_function(
    mi_func: MIType,
    results: list[dict[str, Any]],
    seed_text: str,
) -> dict[str, Any]:
    """Evaluate one MI function on all documents.

    Parameters
    ----------
    mi_func : Callable[[str, str], float]
        MI-like scorer with signature ``(seed_text, output_text) -> float``.
    results : list[dict]
        Parsed rows from ``metrics_phase0.json`` under ``results`` key.
    seed_text : str
        Stable seed text reference.

    Returns
    -------
    dict[str, Any]
        Summary containing class means/stds, pairwise effect sizes,
        Mann-Whitney p-value (food vs toxin), and direction.
    """
    by_class: dict[str, list[float]] = {label: [] for label in CLASS_ORDER}

    for row in results:
        label = str(row.get("type", "")).strip().lower()
        if label not in by_class:
            continue

        output_text = str(row.get("model_output", ""))
        score = float(mi_func(seed_text, output_text))
        by_class[label].append(_clamp_01(score))

    means = {label: _safe_mean(by_class[label]) for label in CLASS_ORDER}
    stds = {label: _safe_std(by_class[label]) for label in CLASS_ORDER}

    r_food_toxin, p_food_toxin = _rank_biserial_and_pvalue(
        by_class["food"], by_class["toxin"]
    )
    r_food_noise, _ = _rank_biserial_and_pvalue(by_class["food"], by_class["noise"])
    r_toxin_noise, _ = _rank_biserial_and_pvalue(by_class["toxin"], by_class["noise"])

    direction = "correct" if means["food"] >= means["toxin"] else "reversed"

    return {
        "means": means,
        "stds": stds,
        "effect_size_r": {
            "food_vs_toxin": r_food_toxin,
            "food_vs_noise": r_food_noise,
            "toxin_vs_noise": r_toxin_noise,
        },
        "p_value": {"food_vs_toxin": p_food_toxin},
        "direction": direction,
        "counts": {label: len(by_class[label]) for label in CLASS_ORDER},
    }


def _stability_score(stds: dict[str, float]) -> float:
    """Aggregate within-class std into one stability score (lower is better)."""
    finite = [value for value in stds.values() if math.isfinite(value)]
    if not finite:
        return float("inf")
    return float(mean(finite))


def _recommend_best(evaluations: dict[str, dict[str, Any]]) -> str:
    """Recommend the best MI function by direction, effect size, and stability."""

    def key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float]:
        name, payload = item
        direction = payload.get("direction", "reversed")
        is_correct = 1 if direction == "correct" else 0

        effect = float(payload["effect_size_r"].get("food_vs_toxin", float("-inf")))
        if not math.isfinite(effect):
            effect = float("-inf")

        stability = _stability_score(payload.get("stds", {}))
        if not math.isfinite(stability):
            stability = float("inf")

        # Highest is_correct, then highest effect, then lowest stability.
        return (is_correct, effect, -stability)

    best_name, _ = max(evaluations.items(), key=key)
    return best_name


def _print_comparison_table(evaluations: dict[str, dict[str, Any]], recommendation: str) -> None:
    """Print a compact comparison table to stdout."""
    header = (
        "method                 food_mean  toxin_mean  noise_mean  "
        "r(food,toxin)  std_mean  direction  recommended"
    )
    print(header)
    print("-" * len(header))

    for name, payload in evaluations.items():
        means = payload["means"]
        std_mean = _stability_score(payload["stds"])
        r_ft = payload["effect_size_r"]["food_vs_toxin"]
        direction = payload["direction"]
        is_recommended = "yes" if name == recommendation else ""

        print(
            f"{name:<22} "
            f"{means['food']:>9.4f} "
            f"{means['toxin']:>10.4f} "
            f"{means['noise']:>10.4f} "
            f"{r_ft:>13.4f} "
            f"{std_mean:>8.4f} "
            f"{direction:>10} "
            f"{is_recommended:>12}"
        )


def run_calibration(
    metrics_json_path: str,
    seed_text: str,
    output_path: str,
    ld50_run_paths: dict[str, str] | None = None,
) -> None:
    """Run MI calibration over a saved ``metrics_phase0.json`` file."""
    metrics_path = Path(metrics_json_path)
    output_file = Path(output_path)

    logging.info("Loading metrics from %s", metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]]
    if isinstance(payload, dict) and "results" in payload:
        results = list(payload["results"])
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError("Unsupported metrics JSON format: expected dict with 'results' or list")

    functions: dict[str, MIType] = {
        "mi_cosine": mi_cosine,
        "mi_entropy_decomp": mi_entropy_decomp,
        "mi_jsd": mi_jsd,
        "mi_npmi": mi_npmi,
        "mi_token_ids": mi_token_ids,
        "mi_token_ids_nmi": mi_token_ids_nmi,
        "mi_token_ids_bigrams": mi_token_ids_bigrams,
        "mi_token_ids_bpe": mi_token_ids_bpe,
    }

    evaluations: dict[str, dict[str, Any]] = {}
    for name, function in functions.items():
        logging.info("Evaluating %s", name)
        evaluations[name] = evaluate_mi_function(
            mi_func=function,
            results=results,
            seed_text=seed_text,
        )

    recommendation = _recommend_best(evaluations)
    _print_comparison_table(evaluations, recommendation)

    ld50_gradient = None
    if ld50_run_paths:
        logging.info("Evaluating LD50 gradient using recommended function: %s", recommendation)
        ld50_gradient = evaluate_ld50_gradient(
            ld50_run_paths=ld50_run_paths,
            seed_text=seed_text,
            mi_func=functions[recommendation],
        )

    output_payload = {
        "metrics_json_path": metrics_json_path,
        "n_results": len(results),
        "recommendation": recommendation,
        "criteria": [
            "correct direction (food > toxin)",
            "highest effect size r for food vs toxin",
            "lowest within-class std",
        ],
        "canonical_evaluation": {
            "recommendation": recommendation,
            "evaluations": evaluations,
        },
        "per_class_stats": {
            name: {
                "means": payload["means"],
                "stds": payload["stds"],
                "counts": payload["counts"],
            }
            for name, payload in evaluations.items()
        },
        "ld50_gradient": ld50_gradient,
        "evaluations": evaluations,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved calibration results to %s", output_file)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for MI calibration."""
    parser = argparse.ArgumentParser(description="Calibrate MI implementations on saved Phase 0 metrics.")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics_phase0.json",
    )
    parser.add_argument(
        "--seed-text",
        required=True,
        help="Stable seed reference text.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for MI calibration results.",
    )
    parser.add_argument(
        "--ld50-run",
        action="append",
        default=[],
        help="Optional LD50 run mapping as <pct>=<path>. Can be passed multiple times.",
    )
    return parser


def _parse_ld50_args(ld50_args: list[str]) -> dict[str, str]:
    """Parse CLI LD50 run arguments in <pct>=<path> format."""
    parsed: dict[str, str] = {}
    for item in ld50_args:
        if "=" not in item:
            raise ValueError(f"Invalid --ld50-run value: {item!r}. Expected <pct>=<path>.")
        pct, path = item.split("=", 1)
        pct_clean = pct.strip()
        path_clean = path.strip()
        if not pct_clean or not path_clean:
            raise ValueError(f"Invalid --ld50-run value: {item!r}. Expected <pct>=<path>.")
        float(pct_clean)
        parsed[pct_clean] = path_clean
    return parsed


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    ld50_run_paths = _parse_ld50_args(args.ld50_run)
    run_calibration(
        metrics_json_path=args.metrics,
        seed_text=args.seed_text,
        output_path=args.output,
        ld50_run_paths=ld50_run_paths if ld50_run_paths else None,
    )


if __name__ == "__main__":
    main()
