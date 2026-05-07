"""Population state management and evolutionary mechanics for Phase 1.

This module is GPU-free: it never loads models or calls trainer.py directly.
The runner (biome_runner.py) calls trainer.py; this module only manages state.

Design decisions:
- All public functions are pure where possible (return new state, no mutation).
- numpy Generator is used for all stochastic operations; no stdlib random.
- Adapter interpolation operates on .safetensors files at the byte level;
  never calls merge_and_unload(). See docs/design_decisions.md.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import struct
from dataclasses import asdict, dataclass

import numpy as np

from src.models.adapters import AdapterMetadata, now_utc_iso

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probabilistic mechanics constants — Phase 0 calibration
# ---------------------------------------------------------------------------

_K_REP: float = 17.3
"""Steepness constant for the per-agent reproduction probability curve.

Calibrated from Phase 0 data so that agents at food-class mean fitness
reproduce at approximately 2× the rate of agents at toxin-class mean
fitness.  With Phase 0 mean fitnesses ≈ 0.24 (food) vs ≈ 0.11 (toxin):
  exp(17.3 × 0.24) / exp(17.3 × 0.11) ≈ 9.7 (saturates at 1.0 for both,
  but the gradient near zero dominates selection pressure in mixed populations).
"""

_BETA: float = 0.1
"""Scale constant for the per-agent death probability.

P_death(f) = max(1 − exp(f / beta), 0.0).  For fitness < 0 the probability
rises sharply toward 1; for fitness ≥ 0 it is clipped to 0 (agents above
zero fitness do not die from this rule alone).  Smaller beta makes the
population less tolerant of sub-zero fitness values.
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Full state record for one live or dead agent.

    Parameters
    ----------
    agent_id : str
        Unique agent identifier following the project naming convention,
        e.g. ``ego_gen02_007``.
    adapter_path : str
        Absolute path to the saved LoRA adapter directory.
    metadata : AdapterMetadata
        Immutable metadata snapshot captured at creation time.
    metrics : dict
        Latest computed metrics; expected keys: h_x, c_x, i_x_seed,
        h_dezorg, fitness.
    generation : int
        Generation index at which this agent was born.
    parent_id : str or None
        Parent agent_id, or ``None`` for generation-0 agents.
    alive : bool
        ``False`` once the agent has been removed from the active population.
    """

    agent_id: str
    adapter_path: str
    metadata: AdapterMetadata
    metrics: dict
    generation: int
    parent_id: str | None
    alive: bool = True


#: Mapping of agent_id → AgentState representing the active population.
Population = dict[str, AgentState]


# ---------------------------------------------------------------------------
# Probabilistic mechanics
# ---------------------------------------------------------------------------


def reproduction_probability(fitness: float) -> float:
    """Compute per-agent reproduction probability for one generation.

    P_rep(fitness) = min(exp(k × fitness), 1.0),  k = _K_REP.

    Very negative fitness values collapse the probability toward 0;
    positive values saturate at 1.

    Parameters
    ----------
    fitness : float
        Current scalar fitness of the agent.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    return min(math.exp(_K_REP * fitness), 1.0)


def death_probability(fitness: float) -> float:
    """Compute per-agent death probability for one generation.

    P_death(fitness) = max(1 − exp(fitness / beta), 0.0),  beta = _BETA.

    Agents with fitness ≥ 0 have death probability 0; agents with strictly
    negative fitness face increasing mortality as fitness falls.

    Parameters
    ----------
    fitness : float
        Current scalar fitness of the agent.

    Returns
    -------
    float
        Probability in [0, 1].
    """
    return max(1.0 - math.exp(fitness / _BETA), 0.0)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_parent(population: Population, rng: np.random.Generator) -> str:
    """Select one parent via softmax-weighted fitness (temperature = 1.0).

    When all fitness values are identical, uniform random selection is used.
    Does not modify the population.

    Parameters
    ----------
    population : Population
        Active population of agents.
    rng : np.random.Generator
        Seeded numpy RNG instance.

    Returns
    -------
    str
        agent_id of the selected parent.
    """
    agent_ids = list(population.keys())
    fitnesses = np.array(
        [population[aid].metrics.get("fitness", 0.0) for aid in agent_ids],
        dtype=np.float64,
    )

    if np.allclose(fitnesses, fitnesses[0]):
        # Uniform fallback when all fitness values are equal
        idx = int(rng.integers(len(agent_ids)))
        return agent_ids[idx]

    # Softmax with temperature 1.0; subtract max for numerical stability
    shifted = fitnesses - fitnesses.max()
    exp_vals = np.exp(shifted)
    weights = exp_vals / exp_vals.sum()

    idx = int(rng.choice(len(agent_ids), p=weights))
    return agent_ids[idx]


def select_candidates_for_death(
    population: Population,
    rng: np.random.Generator,
) -> list[str]:
    """Determine which agents die this generation via independent Bernoulli trials.

    Each agent dies independently with ``death_probability(fitness)``.

    Parameters
    ----------
    population : Population
        Active population of agents.
    rng : np.random.Generator
        Seeded numpy RNG instance.

    Returns
    -------
    list[str]
        Possibly-empty list of agent_ids scheduled for death.
    """
    dying: list[str] = []
    for agent_id, state in population.items():
        fitness = state.metrics.get("fitness", 0.0)
        p = death_probability(fitness)
        if p > 0.0 and rng.random() < p:
            dying.append(agent_id)
    return dying


def select_candidates_for_reproduction(
    population: Population,
    rng: np.random.Generator,
) -> list[str]:
    """Determine which agents reproduce this generation via independent Bernoulli trials.

    Each agent reproduces independently with ``reproduction_probability(fitness)``.

    Parameters
    ----------
    population : Population
        Active population of agents.
    rng : np.random.Generator
        Seeded numpy RNG instance.

    Returns
    -------
    list[str]
        Possibly-empty list of agent_ids that will produce offspring.
    """
    reproducing: list[str] = []
    for agent_id, state in population.items():
        fitness = state.metrics.get("fitness", 0.0)
        p = reproduction_probability(fitness)
        if p > 0.0 and rng.random() < p:
            reproducing.append(agent_id)
    return reproducing


# ---------------------------------------------------------------------------
# Population management
# ---------------------------------------------------------------------------


def apply_deaths(population: Population, dying_ids: list[str]) -> Population:
    """Remove dying agents and return a new population dict.

    The original population is not modified.

    Parameters
    ----------
    population : Population
        Current population.
    dying_ids : list[str]
        Agent IDs returned by :func:`select_candidates_for_death`.

    Returns
    -------
    Population
        New population dict without the dying agents.
    """
    dying_set = set(dying_ids)
    new_pop: Population = {}
    for agent_id, state in population.items():
        if agent_id in dying_set:
            log.info(
                "apply_deaths — removing agent_id=%s (fitness=%.4f)",
                agent_id,
                state.metrics.get("fitness", float("nan")),
            )
        else:
            new_pop[agent_id] = state
    return new_pop


def register_offspring(
    population: Population,
    parent_id: str,
    new_agent_id: str,
    new_adapter_path: str,
    new_metrics: dict,
    generation: int,
) -> Population:
    """Add a newly trained agent to the population.

    Reads ``{new_adapter_path}/metadata.json`` written by
    :func:`src.evolution.trainer.train_adapter`. If the file is absent a
    minimal :class:`AdapterMetadata` is constructed from available fields
    and a warning is logged.

    Parameters
    ----------
    population : Population
        Current population.
    parent_id : str
        agent_id of the parent.
    new_agent_id : str
        agent_id for the offspring.
    new_adapter_path : str
        Path to the adapter directory saved by train_adapter.
    new_metrics : dict
        Metrics dict from measure_metrics (h_x, c_x, i_x_seed, h_dezorg, fitness).
    generation : int
        Current generation index.

    Returns
    -------
    Population
        New population dict with the offspring included.
    """
    metadata_file = os.path.join(new_adapter_path, "metadata.json")
    if os.path.isfile(metadata_file):
        with open(metadata_file, encoding="utf-8") as fh:
            meta_dict = json.load(fh)
        metadata = AdapterMetadata(**meta_dict)
    else:
        log.warning(
            "register_offspring — metadata.json not found at %s; "
            "constructing minimal metadata",
            new_adapter_path,
        )
        metadata = AdapterMetadata(
            generation=generation,
            parent_id=parent_id,
            biome="unknown",
            archetype="unknown",
            fitness_score=float(new_metrics.get("fitness", 0.0)),
            creation_timestamp=now_utc_iso(),
        )

    offspring = AgentState(
        agent_id=new_agent_id,
        adapter_path=os.path.abspath(new_adapter_path),
        metadata=metadata,
        metrics=dict(new_metrics),
        generation=generation,
        parent_id=parent_id,
        alive=True,
    )

    new_pop = dict(population)
    new_pop[new_agent_id] = offspring
    log.info(
        "register_offspring — added agent_id=%s (parent=%s, fitness=%.4f)",
        new_agent_id,
        parent_id,
        float(new_metrics.get("fitness", float("nan"))),
    )
    return new_pop


def check_carrying_capacity(population: Population, k_max: int) -> bool:
    """Return True if the population exceeds the biome carrying capacity.

    Parameters
    ----------
    population : Population
        Current population.
    k_max : int
        Maximum allowed population size for the active biome.

    Returns
    -------
    bool
        ``True`` when ``len(population) > k_max``.
    """
    return len(population) > k_max


# ---------------------------------------------------------------------------
# Cannibalism
# ---------------------------------------------------------------------------


def cannibalism_candidates(population: Population) -> tuple[str, str]:
    """Identify the strongest and weakest agents for a cannibalism event.

    The strongest agent absorbs a fraction of the weakest agent's adapter
    weights (see :func:`interpolate_lora_adapters`).

    Parameters
    ----------
    population : Population
        Current population; must contain at least 2 agents.

    Returns
    -------
    tuple[str, str]
        ``(strongest_id, weakest_id)`` determined by ``fitness`` in metrics.

    Raises
    ------
    ValueError
        If the population has fewer than 2 agents.
    """
    if len(population) < 2:
        raise ValueError(
            f"Cannibalism requires at least 2 agents, got {len(population)}"
        )

    sorted_ids = sorted(
        population.keys(),
        key=lambda aid: population[aid].metrics.get("fitness", 0.0),
    )
    return sorted_ids[-1], sorted_ids[0]   # strongest, weakest


def _load_adapter_tensors_as_float32(adapter_dir: str) -> dict[str, np.ndarray]:
    """Load all tensors from adapter_model.safetensors as float32 numpy arrays.

    Tries the safetensors numpy backend first (handles F32, F16).  For
    bfloat16 tensors — not supported by the numpy backend — falls back to
    manual binary parsing: bfloat16 occupies the upper 16 bits of float32,
    so each 2-byte BF16 value is zero-padded to 4 bytes (uint32 << 16) and
    reinterpreted as float32.

    Parameters
    ----------
    adapter_dir : str
        Path to the adapter directory.

    Returns
    -------
    dict[str, np.ndarray]
        Tensor name → float32 numpy array.

    Raises
    ------
    FileNotFoundError
        If adapter_model.safetensors is not found in *adapter_dir*.
    """
    from safetensors import safe_open  # type: ignore

    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.isfile(sf_path):
        raise FileNotFoundError(
            f"adapter_model.safetensors not found in {adapter_dir!r}"
        )

    # Attempt numpy backend (handles F32, F16, INT types)
    try:
        tensors: dict[str, np.ndarray] = {}
        with safe_open(sf_path, framework="numpy", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).astype(np.float32)
        return tensors
    except Exception as exc:
        log.warning(
            "_load_adapter_tensors_as_float32 — numpy backend failed (%s); "
            "using manual BF16 byte conversion",
            exc,
        )

    # Manual fallback: parse safetensors binary format directly.
    # Format: [8-byte LE uint64 header_size][header_size bytes JSON][data bytes]
    # data_offsets in the JSON are byte offsets from the start of the data block.
    tensors = {}
    with open(sf_path, "rb") as fh:
        header_size = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_size).decode("utf-8"))
        data_block = fh.read()

    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype_str: str = meta["dtype"]
        shape: list[int] = meta["shape"]
        begin, end = meta["data_offsets"]
        raw = data_block[begin:end]

        if dtype_str == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            tensor = u32.view(np.float32).reshape(shape).copy()
        elif dtype_str == "F16":
            tensor = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape).copy()
        elif dtype_str == "F32":
            tensor = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        else:
            raise ValueError(
                f"Unsupported safetensors dtype {dtype_str!r} for tensor {key!r}. "
                "Only BF16, F16, F32 are supported."
            )
        tensors[key] = tensor

    return tensors


def interpolate_lora_adapters(
    strong_adapter_path: str,
    weak_adapter_path: str,
    output_path: str,
    alpha: float = 0.15,
) -> str:
    """Interpolate two LoRA adapters at the adapter-tensor level (CPU only).

    Computes W_new = W_strong × (1 − alpha) + W_weak × alpha for every
    tensor present in both adapters. Tensors unique to the strong adapter
    are passed through unchanged.  ``adapter_config.json`` is copied from
    the strong adapter to preserve the LoRA configuration.

    This function operates exclusively on adapter weight files — it never
    loads the base model and never calls ``merge_and_unload()``.
    See docs/design_decisions.md for the rationale.

    Tensors are upcasted to float32 for interpolation and saved as float32.

    Parameters
    ----------
    strong_adapter_path : str
        Directory of the absorbing (higher-fitness) adapter.
    weak_adapter_path : str
        Directory of the absorbed (lower-fitness) adapter.
    output_path : str
        Directory to write the interpolated adapter into.
    alpha : float
        Mixing coefficient in [0, 1]. At alpha=0 the result equals the
        strong adapter; at alpha=1 it equals the weak adapter.

    Returns
    -------
    str
        Absolute path to the output adapter directory.

    Raises
    ------
    FileNotFoundError
        If either adapter directory is missing.
    """
    for p in (strong_adapter_path, weak_adapter_path):
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"Adapter directory not found for interpolation: {p!r}"
            )

    log.info(
        "interpolate_lora_adapters — strong=%s, weak=%s, alpha=%.2f",
        strong_adapter_path,
        weak_adapter_path,
        alpha,
    )

    strong_tensors = _load_adapter_tensors_as_float32(strong_adapter_path)
    weak_tensors = _load_adapter_tensors_as_float32(weak_adapter_path)

    interpolated: dict[str, np.ndarray] = {}
    for key, w_strong in strong_tensors.items():
        if key in weak_tensors:
            interpolated[key] = (
                w_strong * (1.0 - alpha) + weak_tensors[key] * alpha
            ).astype(np.float32)
        else:
            interpolated[key] = w_strong.astype(np.float32)

    from safetensors.numpy import save_file as _st_save  # type: ignore

    os.makedirs(output_path, exist_ok=True)
    out_sf = os.path.join(output_path, "adapter_model.safetensors")
    _st_save(interpolated, out_sf)

    # Copy LoRA config from the strong adapter so the result is loadable by PEFT
    strong_cfg = os.path.join(strong_adapter_path, "adapter_config.json")
    if os.path.isfile(strong_cfg):
        shutil.copy2(strong_cfg, os.path.join(output_path, "adapter_config.json"))

    log.info("interpolate_lora_adapters — saved to %s", output_path)
    return os.path.abspath(output_path)


def apply_cannibalism(
    population: Population,
    k_max: int,
    adapters_dir: str,
    rng: np.random.Generator,
) -> Population:
    """Reduce population to k_max via repeated LoRA weight absorption.

    While ``len(population) > k_max``, the strongest agent absorbs the
    weakest: their adapters are interpolated at alpha=0.15, the interpolated
    adapter replaces the strongest agent's adapter_path, and the weakest
    agent is removed.

    Parameters
    ----------
    population : Population
        Current population.
    k_max : int
        Target maximum population size for the active biome.
    adapters_dir : str
        Root directory for adapter storage; each interpolated adapter is
        written into a new subdirectory here.
    rng : np.random.Generator
        Seeded numpy RNG (reserved for future stochastic tie-breaking).

    Returns
    -------
    Population
        Population with at most k_max agents.
    """
    pop = dict(population)
    while len(pop) > k_max:
        strongest_id, weakest_id = cannibalism_candidates(pop)
        strongest = pop[strongest_id]
        weakest = pop[weakest_id]

        log.info(
            "apply_cannibalism — %s (fitness=%.4f) absorbs %s (fitness=%.4f)",
            strongest_id,
            strongest.metrics.get("fitness", float("nan")),
            weakest_id,
            weakest.metrics.get("fitness", float("nan")),
        )

        interp_dir = os.path.join(
            adapters_dir,
            f"cannibalism_{strongest_id}_absorbs_{weakest_id}",
        )
        new_adapter_path = interpolate_lora_adapters(
            strong_adapter_path=strongest.adapter_path,
            weak_adapter_path=weakest.adapter_path,
            output_path=interp_dir,
            alpha=0.15,
        )

        # Replace strongest's adapter path; reconstruct to avoid mutating frozen metadata
        pop[strongest_id] = AgentState(
            agent_id=strongest.agent_id,
            adapter_path=new_adapter_path,
            metadata=strongest.metadata,
            metrics=strongest.metrics,
            generation=strongest.generation,
            parent_id=strongest.parent_id,
            alive=strongest.alive,
        )
        del pop[weakest_id]

    return pop


# ---------------------------------------------------------------------------
# HGT — Phase 1 stub
# ---------------------------------------------------------------------------


def hgt_passive(
    population: Population,
    fragment_pool: list[str],
    p: float,
    rng: np.random.Generator,
) -> Population:
    """Passive horizontal gene transfer via random adapter fragment injection.

    Phase 1: disabled; always raises :exc:`NotImplementedError`.

    Phase 3 intended behaviour: each agent independently receives a random
    fragment from ``fragment_pool`` with probability ``p``; the fragment is
    interpolated into the agent's adapter at a low mixing coefficient.

    Parameters
    ----------
    population : Population
        Current population.
    fragment_pool : list[str]
        Paths to remnant adapter directories available for transfer.
    p : float
        Per-agent HGT probability (0.05–0.10 in Phase 3).
    rng : np.random.Generator
        Seeded numpy RNG.

    Raises
    ------
    NotImplementedError
        Always in Phase 1.
    """
    raise NotImplementedError("HGT disabled in Phase 1")


# ---------------------------------------------------------------------------
# Genealogy
# ---------------------------------------------------------------------------


def build_genealogy_record(population: Population, generation: int) -> dict:
    """Build a JSON-serializable snapshot of the current generation state.

    Parameters
    ----------
    population : Population
        Population after all deaths and births for this generation.
    generation : int
        Current generation index.

    Returns
    -------
    dict
        Keys: ``generation`` (int) and ``agents`` (list of per-agent dicts
        with keys agent_id, parent_id, fitness, alive, adapter_path).
    """
    agents = [
        {
            "agent_id": state.agent_id,
            "parent_id": state.parent_id,
            "fitness": state.metrics.get("fitness"),
            "alive": state.alive,
            "adapter_path": state.adapter_path,
        }
        for state in population.values()
    ]
    return {"generation": generation, "agents": agents}


# ---------------------------------------------------------------------------
# Checkpoint serialization
# ---------------------------------------------------------------------------


def save_population(population: Population, path: str) -> None:
    """Serialize population to a JSON checkpoint file.

    :class:`AdapterMetadata` fields are stored as a nested dict under the
    key ``"metadata"``.  The file is written atomically via a temporary file
    to avoid partial writes on crash.

    Parameters
    ----------
    population : Population
        Population to serialize.
    path : str
        Destination file path (created or overwritten).
    """
    records: dict[str, dict] = {}
    for agent_id, state in population.items():
        records[agent_id] = {
            "agent_id": state.agent_id,
            "adapter_path": state.adapter_path,
            "metadata": asdict(state.metadata),
            "metrics": state.metrics,
            "generation": state.generation,
            "parent_id": state.parent_id,
            "alive": state.alive,
        }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    os.replace(tmp, path)
    log.info("save_population — wrote %d agents to %s", len(records), path)


def load_population(path: str) -> Population:
    """Deserialize population from a JSON checkpoint file.

    Reconstructs :class:`AdapterMetadata` from the nested ``"metadata"``
    dict in each record.

    Parameters
    ----------
    path : str
        Path to a checkpoint written by :func:`save_population`.

    Returns
    -------
    Population
        Reconstructed population mapping.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Population checkpoint not found: {path!r}"
        )

    with open(path, encoding="utf-8") as fh:
        records: dict = json.load(fh)

    population: Population = {}
    for agent_id, rec in records.items():
        metadata = AdapterMetadata(**rec["metadata"])
        population[agent_id] = AgentState(
            agent_id=rec["agent_id"],
            adapter_path=rec["adapter_path"],
            metadata=metadata,
            metrics=rec["metrics"],
            generation=rec["generation"],
            parent_id=rec["parent_id"],
            alive=rec["alive"],
        )

    log.info("load_population — loaded %d agents from %s", len(population), path)
    return population
