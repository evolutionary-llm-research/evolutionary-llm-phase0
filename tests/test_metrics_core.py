def test_jaccard_similarity_identical():
    from src.metrics.core import jaccard_similarity
    assert jaccard_similarity("foo bar baz", "foo bar baz") == 1.0

def test_jaccard_similarity_disjoint():
    from src.metrics.core import jaccard_similarity
    assert jaccard_similarity("aaa bbb", "ccc ddd") == 0.0

def test_jaccard_similarity_partial_overlap():
    from src.metrics.core import jaccard_similarity
    val = jaccard_similarity("a b c", "a d e")
    assert 0.0 < val < 1.0
from src.metrics.core import disorganization_entropy, effective_complexity, fitness_score, mutual_information_proxy, shannon_entropy


def test_shannon_entropy_empty() -> None:
    assert shannon_entropy("") == 0.0


def test_shannon_entropy_repeated_token() -> None:
    assert shannon_entropy("x x x x") == 0.0


def test_effective_complexity_empty() -> None:
    assert effective_complexity("") == 0.0


def test_effective_complexity_non_empty() -> None:
    assert effective_complexity("abc abc abc") > 0.0


def test_fitness_score_formula() -> None:
    value = fitness_score(complexity=2.0, mutual_info=1.5, disorganization=0.5, w1=1.0, w2=2.0, w3=3.0)
    assert value == 2.0 + 3.0 - 1.5


def test_mutual_information_proxy_range() -> None:
    score = mutual_information_proxy("alpha beta", "alpha gamma")
    assert 0.0 <= score <= 1.0


def test_mutual_information_proxy_identical():
    # Identical texts should yield 1.0 (with float tolerance)
    import pytest
    score = mutual_information_proxy("foo bar baz", "foo bar baz")
    assert score == pytest.approx(1.0, rel=1e-9)


def test_mutual_information_proxy_disjoint():
    # Completely disjoint vocab should yield 0.0
    score = mutual_information_proxy("aaa bbb", "ccc ddd")
    assert score == 0.0


def test_mutual_information_proxy_partial_overlap():
    # Partial overlap should yield value between 0 and 1
    score = mutual_information_proxy("a b c", "a d e")
    assert 0.0 < score < 1.0


def test_disorganization_entropy_empty() -> None:
    assert disorganization_entropy("") == 0.0


def test_disorganization_entropy_sentence_mix() -> None:
    value = disorganization_entropy("short sentence. this is a much longer sentence for testing.")
    assert 0.0 <= value <= 1.0
