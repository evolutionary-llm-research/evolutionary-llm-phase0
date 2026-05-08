from __future__ import annotations

"""Sentence-tokenizer calibration for H_dezorg computation.

This module compares three disorganization-entropy implementations and
recommends the most suitable sentence tokenizer variant for Phase 0 metrics.
"""

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import nltk
from nltk.tokenize import sent_tokenize
from scipy.stats import mannwhitneyu, pearsonr


CLASS_ORDER = ("food", "toxin", "noise")
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.!?;])\s+(?=[A-Z])"
HDezorgFunc = Callable[[str], float]


def h_dezorg_simple(text: str) -> float:
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


def _entropy_from_sentence_lengths(sentence_lengths: list[int]) -> float:
    """Compute normalized entropy from sentence lengths."""
    if len(sentence_lengths) <= 1:
        return 0.0

    total = sum(sentence_lengths)
    if total <= 0:
        return 0.0

    entropy = 0.0
    for sentence_len in sentence_lengths:
        p = sentence_len / total
        entropy -= p * math.log2(p)

    max_entropy = math.log2(len(sentence_lengths))
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


def h_dezorg_nltk(text: str) -> float:
    """Compute H_dezorg using NLTK sentence tokenization."""
    try:
        sentences = [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
    except LookupError:
        # Newer NLTK versions can require punkt_tab in addition to punkt.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]

    if len(sentences) <= 1:
        return 0.0

    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return _entropy_from_sentence_lengths(sentence_lengths)


def _unique_preserve_order(items: list[str]) -> list[str]:
    """Return unique strings while preserving first-seen order."""
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def h_dezorg_regex(text: str) -> float:
    """Compute H_dezorg with regex boundary splits plus newline splitting."""
    primary = [part.strip() for part in re.split(SENTENCE_BOUNDARY_PATTERN, text) if part.strip()]

    secondary: list[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        secondary.extend(part.strip() for part in re.split(SENTENCE_BOUNDARY_PATTERN, block) if part.strip())

    combined = _unique_preserve_order(primary + secondary)
    if len(combined) <= 1:
        return 0.0

    sentence_lengths = [len(sentence.split()) for sentence in combined]
    return _entropy_from_sentence_lengths(sentence_lengths)


def _extract_class_label(record: dict[str, Any]) -> str | None:
    """Extract and normalize document class label from a result record."""
    label = str(record.get("type", record.get("class", ""))).strip().lower()
    return label if label in CLASS_ORDER else None


def _extract_output_text(record: dict[str, Any]) -> str:
    """Extract model output text from a result record."""
    for key in ("model_output", "output_text", "generated_text", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _safe_mean(values: list[float]) -> float:
    """Return mean or NaN for empty lists."""
    return float(mean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    """Return population standard deviation or NaN for empty lists."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def _rank_biserial_and_pvalue(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return rank-biserial effect size and Mann-Whitney p-value."""
    if not x or not y:
        return float("nan"), float("nan")

    test = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    n1 = len(x)
    n2 = len(y)
    u_value = float(test.statistic)
    rank_biserial = (2.0 * u_value) / (n1 * n2) - 1.0
    return rank_biserial, float(test.pvalue)


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Return Pearson correlation or NaN when undefined."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    result = pearsonr(x, y)
    return float(result.statistic)


def evaluate_h_dezorg(
    results: list[dict],
    variant_func: HDezorgFunc,
    variant_name: str,
) -> dict:
    """Evaluate one H_dezorg variant against saved Phase 0 results.

    Parameters
    ----------
    results : list[dict]
        Per-document metrics records.
    variant_func : callable
        Function computing H_dezorg from text.
    variant_name : str
        Human-readable variant name.

    Returns
    -------
    dict
        Summary statistics and comparison metrics.
    """
    by_class: dict[str, list[float]] = {class_name: [] for class_name in CLASS_ORDER}
    variant_values_all: list[float] = []
    existing_h_dezorg_all: list[float] = []

    for record in results:
        class_label = _extract_class_label(record)
        if class_label is None:
            continue

        output_text = _extract_output_text(record)
        if not output_text:
            continue

        value = float(variant_func(output_text))
        by_class[class_label].append(value)

        existing_value = record.get("h_dezorg")
        if isinstance(existing_value, (int, float)):
            variant_values_all.append(value)
            existing_h_dezorg_all.append(float(existing_value))

    mean_by_class = {
        class_name: _safe_mean(by_class[class_name]) for class_name in CLASS_ORDER
    }
    std_by_class = {
        class_name: _safe_std(by_class[class_name]) for class_name in CLASS_ORDER
    }

    r_food_toxin, p_food_toxin = _rank_biserial_and_pvalue(
        by_class["food"], by_class["toxin"]
    )

    food_mean = mean_by_class["food"]
    toxin_mean = mean_by_class["toxin"]
    if math.isnan(food_mean) or math.isnan(toxin_mean):
        direction = "undetermined"
    elif toxin_mean > food_mean:
        direction = "correct"
    elif food_mean > toxin_mean:
        direction = "reversed"
    else:
        direction = "tie"

    pearson_with_existing = _pearson_correlation(variant_values_all, existing_h_dezorg_all)

    return {
        "variant_name": variant_name,
        "mean_by_class": mean_by_class,
        "std_by_class": std_by_class,
        "effect_size_r_food_vs_toxin": r_food_toxin,
        "p_value_food_vs_toxin": p_food_toxin,
        "direction": direction,
        "pearson_r_with_existing_h_dezorg": pearson_with_existing,
        "n_by_class": {class_name: len(by_class[class_name]) for class_name in CLASS_ORDER},
    }


def _fmt_metric(value: float, digits: int = 4) -> str:
    """Format float values for compact console tables."""
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _print_comparison_table(evaluations: list[dict[str, Any]]) -> None:
    """Print a concise variant-comparison table."""
    headers = [
        "variant",
        "dir",
        "r(food,toxin)",
        "p(food,toxin)",
        "pearson(existing)",
        "mean_food",
        "mean_toxin",
        "mean_noise",
    ]

    rows: list[list[str]] = []
    for item in evaluations:
        means = item["mean_by_class"]
        rows.append(
            [
                str(item["variant_name"]),
                str(item["direction"]),
                _fmt_metric(float(item["effect_size_r_food_vs_toxin"]), 4),
                _fmt_metric(float(item["p_value_food_vs_toxin"]), 6),
                _fmt_metric(float(item["pearson_r_with_existing_h_dezorg"]), 4),
                _fmt_metric(float(means["food"]), 4),
                _fmt_metric(float(means["toxin"]), 4),
                _fmt_metric(float(means["noise"]), 4),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _print_row(cells: list[str]) -> None:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)))

    _print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        _print_row(row)


def _rank_for_recommendation(item: dict[str, Any]) -> tuple[int, float, float]:
    """Build ranking tuple based on calibration criteria."""
    direction_ok = 1 if item.get("direction") == "correct" else 0

    effect_size = item.get("effect_size_r_food_vs_toxin", float("nan"))
    effect_score = abs(float(effect_size)) if not math.isnan(float(effect_size)) else float("-inf")

    correlation = item.get("pearson_r_with_existing_h_dezorg", float("nan"))
    corr_score = float(correlation) if not math.isnan(float(correlation)) else float("-inf")

    return direction_ok, effect_score, corr_score


def run_calibration(metrics_json_path: str, output_path: str) -> None:
    """Run H_dezorg sentence-tokenizer calibration and persist results."""
    metrics_path = Path(metrics_json_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        results = payload["results"]
    elif isinstance(payload, list):
        results = payload
    else:
        raise ValueError("Expected metrics JSON with 'results' list or a top-level list.")

    variants: list[tuple[str, HDezorgFunc]] = [
        ("h_dezorg_simple", h_dezorg_simple),
        ("h_dezorg_nltk", h_dezorg_nltk),
        ("h_dezorg_regex", h_dezorg_regex),
    ]

    evaluations = [
        evaluate_h_dezorg(results=results, variant_func=func, variant_name=name)
        for name, func in variants
    ]

    recommended = max(evaluations, key=_rank_for_recommendation)

    print("H_dezorg tokenizer calibration")
    print(f"Metrics source: {metrics_path}")
    print(f"Samples: {len(results)}")
    print()
    _print_comparison_table(evaluations)
    print()
    print(f"Recommended variant: {recommended['variant_name']}")

    output = {
        "metrics_path": str(metrics_path),
        "n_results": len(results),
        "evaluations": evaluations,
        "recommended_variant": recommended["variant_name"],
        "recommendation_criteria": [
            "correct direction (toxin > food)",
            "highest absolute effect size r food vs toxin",
            "highest Pearson correlation with existing h_dezorg",
        ],
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Saved calibration results to: {output_file}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for calibration script."""
    parser = argparse.ArgumentParser(description="Calibrate H_dezorg sentence tokenizer variants.")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics JSON (expects top-level 'results' or raw list).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for calibration summary JSON.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    run_calibration(metrics_json_path=args.metrics, output_path=args.output)


if __name__ == "__main__":
    main()
