"""Unit tests for the entropy-decomposition mutual_information_proxy."""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics.core import mutual_information_proxy


def test_identical_texts_returns_one():
    """Identical texts: H(X,Y)=H(X)=H(Y), so I/min(H)=1.0."""
    text = "the quick brown fox jumps over the lazy dog"
    result = mutual_information_proxy(text, text)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_disjoint_vocabularies_returns_zero():
    """Disjoint 2-token uniform distributions: H(X)=H(Y)=1, H(X,Y)=2 -> I=0.0."""
    # seed = {cat:1, dog:1} H=1.0; output = {table:1, lamp:1} H=1.0
    # concat = {cat,dog,table,lamp} each 0.25 -> H=2.0; I=1+1-2=0.0
    result = mutual_information_proxy("cat dog", "table lamp")
    assert abs(result - 0.0) < 1e-9, f"Expected 0.0, got {result}"


def test_partial_overlap_returns_between_zero_and_one():
    """Partial vocabulary overlap returns value strictly in (0, 1)."""
    # seed={alpha,beta,gamma} output={beta,gamma,delta}: 2/3 tokens shared
    result = mutual_information_proxy("alpha beta gamma", "beta gamma delta")
    assert 0.0 < result < 1.0, f"Expected value in (0, 1), got {result}"


def test_empty_seed_returns_zero():
    """Empty seed -> H(X)=0.0 -> min(H(X),H(Y))=0.0 -> returns 0.0."""
    result = mutual_information_proxy("", "some output text here")
    assert result == 0.0, f"Expected 0.0, got {result}"


def test_empty_output_returns_zero():
    """Empty output -> H(Y)=0.0 -> min(H(X),H(Y))=0.0 -> returns 0.0."""
    result = mutual_information_proxy("some seed text here", "")
    assert result == 0.0, f"Expected 0.0, got {result}"
