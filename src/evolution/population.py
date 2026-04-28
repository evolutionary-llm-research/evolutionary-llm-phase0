"""Population-level utilities used by evolutionary phases."""

from __future__ import annotations


def should_reproduce(fitness: float, threshold: float) -> bool:
    """Check if an agent can reproduce under a fitness threshold.

    Parameters
    ----------
    fitness : float
        Current agent fitness.
    threshold : float
        Minimum fitness required for reproduction.

    Returns
    -------
    bool
        True when reproduction is allowed.
    """
    return fitness > threshold


def should_decay(fitness: float, fitness_min: float) -> bool:
    """Check if an agent should be removed from population.

    Parameters
    ----------
    fitness : float
        Current agent fitness.
    fitness_min : float
        Minimal fitness required to stay in population.

    Returns
    -------
    bool
        True when the agent should decay.
    """
    return fitness < fitness_min
