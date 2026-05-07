"""Tests for src/evolution/population.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.adapters import AdapterMetadata, now_utc_iso
from src.evolution.population import (
    AgentState,
    Population,
    select_parent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agent_id: str, fitness: float) -> AgentState:
    meta = AdapterMetadata(
        generation=1,
        parent_id=None,
        biome="savanna",
        archetype="ego",
        fitness_score=fitness,
        creation_timestamp=now_utc_iso(),
    )
    return AgentState(
        agent_id=agent_id,
        adapter_path="/tmp/fake",
        metadata=meta,
        metrics={"fitness": fitness},
        generation=1,
        parent_id=None,
    )


# ---------------------------------------------------------------------------
# Statistical test for select_parent softmax direction
# ---------------------------------------------------------------------------


def test_select_parent_softmax_direction_statistical() -> None:
    """Softmax selection must favour higher-fitness agents.

    Method: run 10 000 selections, each with an independent rng seed
    (seed=i for i in range(10_000)).  Assert that the empirical selection
    frequency ordering matches the fitness ordering:

        freq(agent_001, f=-0.016) > freq(agent_002, f=-0.056) > freq(agent_003, f=-0.070)

    This verifies the direction of softmax selection without relying on any
    single fixed seed.
    """
    population: Population = {
        "agent_001": _make_agent("agent_001", -0.016),
        "agent_002": _make_agent("agent_002", -0.056),
        "agent_003": _make_agent("agent_003", -0.070),
    }

    n_trials = 10_000
    counts: dict[str, int] = {"agent_001": 0, "agent_002": 0, "agent_003": 0}

    for i in range(n_trials):
        rng = np.random.default_rng(i)
        selected = select_parent(population, rng)
        counts[selected] += 1

    freq = {k: v / n_trials for k, v in counts.items()}

    assert freq["agent_001"] > freq["agent_002"], (
        f"Expected freq(agent_001) > freq(agent_002), "
        f"got {freq['agent_001']:.4f} vs {freq['agent_002']:.4f}"
    )
    assert freq["agent_002"] > freq["agent_003"], (
        f"Expected freq(agent_002) > freq(agent_003), "
        f"got {freq['agent_002']:.4f} vs {freq['agent_003']:.4f}"
    )
