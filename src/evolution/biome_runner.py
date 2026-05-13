"""Top-level orchestrator for Phase 1 single-lineage evolutionary experiments.

Ties together corpus sampling, LoRA training, metric measurement, population
mechanics, checkpointing, and generation logging for one biome.

Usage (called by cli.py, not invoked directly):
    run_biome(biome_name="savanna", config_path="config/phase1_single_model.yaml", ...)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models.adapters import AdapterMetadata, build_agent_id, now_utc_iso
from src.evolution.population import (
    AgentState,
    Population,
    apply_cannibalism,
    apply_deaths,
    build_genealogy_record,
    check_carrying_capacity,
    load_population,
    register_offspring,
    save_population,
    select_parent,
)

# trainer imports are deferred to function bodies: importing trainer at module
# level would pull in unsloth (which requires a GPU) and break non-GPU
# environments used for data pipeline, analysis, and testing.
# Functions that need GPU call:
#   from src.evolution.trainer import train_adapter, train_and_measure

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIAGNOSTIC_PROMPT: str = (
    "Summarize the key mechanisms by which misinformation spreads "
    "in online environments and describe evidence-based interventions."
)

# Frozen fitness weights (Phase 1, validated in Phase 0).
_FITNESS_WEIGHTS: dict[str, float] = {"w1": 0.3, "w2": 0.5, "w3": 0.2}


# ---------------------------------------------------------------------------
# Z-score bounded sigmoid mechanics (Phase 1b)
# ---------------------------------------------------------------------------

def _get_population_mechanics_params(config: dict) -> dict:
    """Extract population mechanics parameters from config with defaults."""
    mechanics = config.get("mechanics", {})
    return {
        "alpha_r":   float(mechanics.get("alpha_r",   1.0)),
        "alpha_d":   float(mechanics.get("alpha_d",   1.5)),
        "p_r_min":   float(mechanics.get("p_r_min",   0.05)),
        "p_r_max":   float(mechanics.get("p_r_max",   0.45)),
        "p_d_min":   float(mechanics.get("p_d_min",   0.07)),
        "p_d_max":   float(mechanics.get("p_d_max",   0.45)),
        "sigma_min": float(mechanics.get("sigma_min", 0.01)),
    }


def _compute_population_zscores(
    population: Population,
    sigma_min: float,
) -> tuple[dict[str, float], float, float]:
    """Compute per-agent z-scores from current population fitness values.

    Returns
    -------
    tuple
        (zscores dict, mu_f, sigma_f)
    """
    fitnesses = [state.metrics.get("fitness", 0.0) for state in population.values()]
    mu = float(np.mean(fitnesses)) if fitnesses else 0.0
    sigma = float(max(np.std(fitnesses), sigma_min)) if fitnesses else sigma_min
    zscores = {
        aid: (state.metrics.get("fitness", 0.0) - mu) / sigma
        for aid, state in population.items()
    }
    return zscores, mu, sigma


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def zscore_reproduction_probs(
    population: Population,
    params: dict,
    zscores: dict[str, float],
) -> dict[str, float]:
    """Compute P_rep_i = p_r_min + (p_r_max - p_r_min) * sigmoid(alpha_r * z_i)."""
    alpha_r = params["alpha_r"]
    p_r_min = params["p_r_min"]
    p_r_max = params["p_r_max"]
    return {
        aid: p_r_min + (p_r_max - p_r_min) * _sigmoid(alpha_r * z)
        for aid, z in zscores.items()
    }


def zscore_death_probs(
    population: Population,
    params: dict,
    zscores: dict[str, float],
) -> dict[str, float]:
    """Compute P_death_i = p_d_min + (p_d_max - p_d_min) * sigmoid(-alpha_d * z_i)."""
    alpha_d = params["alpha_d"]
    p_d_min = params["p_d_min"]
    p_d_max = params["p_d_max"]
    return {
        aid: p_d_min + (p_d_max - p_d_min) * _sigmoid(-alpha_d * z)
        for aid, z in zscores.items()
    }


def select_candidates_for_death_zscore(
    population: Population,
    rng: np.random.Generator,
    params: dict,
    zscores: dict[str, float],
) -> list[str]:
    """Independent Bernoulli draws with z-score bounded sigmoid death probabilities."""
    probs = zscore_death_probs(population, params, zscores)
    dying: list[str] = []
    for aid in population:
        p = probs[aid]
        if p > 0.0 and rng.random() < p:
            dying.append(aid)
    return dying


def select_candidates_for_reproduction_zscore(
    population: Population,
    rng: np.random.Generator,
    params: dict,
    zscores: dict[str, float],
) -> list[str]:
    """Independent Bernoulli draws with z-score bounded sigmoid reproduction probabilities."""
    probs = zscore_reproduction_probs(population, params, zscores)
    reproducing: list[str] = []
    for aid in population:
        p = probs[aid]
        if p > 0.0 and rng.random() < p:
            reproducing.append(aid)
    return reproducing


# Archetype label for Phase 1 (single-model; Id/Ego/Superego are Phase 2+).
_PHASE1_ARCHETYPE: str = "base"

# Base model name for seed-output generation (must match trainer.py constant).
_BASE_MODEL: str = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus_by_type(
    corpus_manifest_path: str,
) -> dict[str, list[str]]:
    """Load all JSONL documents from the corpus manifest, grouped by type.

    Reads every file listed in the manifest and returns a mapping from
    content-type label (``"food"``, ``"toxin"``, ``"noise"``) to a list of
    ``content`` strings.  Type metadata is never exposed to the model.

    Parameters
    ----------
    corpus_manifest_path : str
        Path to the corpus manifest JSON (e.g. ``data/v2/corpus_manifest_v3.json``).

    Returns
    -------
    dict[str, list[str]]
        Keys: ``"food"``, ``"toxin"``, ``"noise"``.  Values: lists of
        document content strings.

    Raises
    ------
    FileNotFoundError
        If the manifest file or any referenced JSONL file is missing.
    """
    manifest_path = Path(corpus_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {corpus_manifest_path!r}")

    with open(manifest_path, encoding="utf-8") as fh:
        manifest: dict = json.load(fh)

    # Resolve JSONL paths relative to the manifest's parent directory.
    base_dir = manifest_path.parent

    corpus: dict[str, list[str]] = {"food": [], "toxin": [], "noise": []}

    for key, entry in manifest.get("files", {}).items():
        if key.startswith("food"):
            bucket = "food"
        elif key.startswith("toxin"):
            bucket = "toxin"
        else:
            bucket = "noise"

        jsonl_path = base_dir / entry["path"].lstrip("/").lstrip("\\")
        # Fallback: resolve relative to repo root if not found next to manifest.
        if not jsonl_path.is_file():
            jsonl_path = Path(entry["path"])
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"JSONL file not found: {entry['path']!r} (resolved to {jsonl_path})"
            )

        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    content: str = doc.get("content", "").strip()
                    if content:
                        corpus[bucket].append(content)
                except json.JSONDecodeError:
                    log.warning("Skipping malformed JSONL line in %s", jsonl_path)

    log.info(
        "_load_corpus_by_type — food=%d, toxin=%d, noise=%d",
        len(corpus["food"]),
        len(corpus["toxin"]),
        len(corpus["noise"]),
    )
    return corpus


# ---------------------------------------------------------------------------
# Corpus sampling
# ---------------------------------------------------------------------------


def sample_biome_documents(
    corpus_manifest_path: str,
    biome_config: dict,
    n_documents: int,
    generation: int,
    rng: np.random.Generator,
    _corpus_cache: dict[str, list[str]] | None = None,
) -> list[str]:
    """Sample *n_documents* from the corpus according to biome content ratios.

    Ratios from *biome_config* keys ``"food"``, ``"toxin"``, ``"noise"`` sum
    to 1.0.  Fractional counts are rounded; any rounding remainder is added to
    or subtracted from the food count so the total equals *n_documents* exactly.

    Sampling is without replacement *within* this call; each invocation is
    independent (different *rng* state).  The returned list contains content
    strings only — type tags are never included.

    Parameters
    ----------
    corpus_manifest_path : str
        Path to the corpus manifest JSON.
    biome_config : dict
        Dict with keys ``"food"``, ``"toxin"``, ``"noise"`` (float ratios).
    n_documents : int
        Total number of documents to return.
    generation : int
        Current generation index (used only for logging).
    rng : np.random.Generator
        Seeded numpy RNG; advances state on each call.
    _corpus_cache : dict or None
        Pre-loaded corpus dict (optional); avoids re-reading files when
        called multiple times within the same run.

    Returns
    -------
    list[str]
        Shuffled list of *n_documents* content strings.

    Raises
    ------
    ValueError
        If any corpus bucket has fewer documents than required.
    """
    corpus = _corpus_cache if _corpus_cache is not None else _load_corpus_by_type(
        corpus_manifest_path
    )

    n_toxin = round(n_documents * float(biome_config.get("toxin", 0.0)))
    n_noise = round(n_documents * float(biome_config.get("noise", 0.0)))
    n_food = n_documents - n_toxin - n_noise  # absorb rounding remainder

    counts = {"food": n_food, "toxin": n_toxin, "noise": n_noise}
    sampled: list[str] = []
    for bucket, n in counts.items():
        if n <= 0:
            continue
        pool = corpus[bucket]
        if len(pool) < n:
            # Fallback: if exact sampling fails, use sampling with replacement.
            # This allows longer runs but documents may repeat.
            log.warning(
                "Corpus bucket '%s' has only %d documents, requested %d "
                "(generation=%d). Using sampling with replacement.",
                bucket,
                len(pool),
                n,
                generation,
            )
            indices = rng.choice(len(pool), size=n, replace=True)
        else:
            indices = rng.choice(len(pool), size=n, replace=False)
        sampled.extend(pool[i] for i in indices)

    # Shuffle to interleave types; model must never infer type from position.
    rng.shuffle(sampled)  # type: ignore[arg-type]
    log.info(
        "sample_biome_documents gen=%d — food=%d toxin=%d noise=%d total=%d",
        generation,
        n_food,
        n_toxin,
        n_noise,
        len(sampled),
    )
    return sampled


# ---------------------------------------------------------------------------
# JSD divergence
# ---------------------------------------------------------------------------


def _token_distribution(text: str) -> dict[str, float]:
    """Return a normalised unigram probability distribution over whitespace tokens."""
    tokens = text.split()
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {tok: cnt / total for tok, cnt in counts.items()}


def _kl_divergence(p: dict[str, float], m: dict[str, float]) -> float:
    """Compute KL(P || M) using log base 2.  Terms where P=0 are skipped."""
    kl = 0.0
    for tok, p_val in p.items():
        if p_val > 0.0:
            m_val = m.get(tok, 0.0)
            if m_val > 0.0:
                kl += p_val * math.log2(p_val / m_val)
    return kl


def compute_jsd_matrix(
    agent_outputs: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Compute pairwise Jensen-Shannon Divergence between all agent outputs.

    JSD is derived from unigram token-frequency distributions:

        M = 0.5 * (P + Q)
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    The result is symmetric and bounded in ``[0, 1]`` (log-base-2 formulation).

    Parameters
    ----------
    agent_outputs : dict[str, str]
        Mapping of ``agent_id → generated_text``.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict ``{agent_id_i: {agent_id_j: jsd_value}}``.  The diagonal
        is 0.0 by definition.
    """
    agent_ids = list(agent_outputs.keys())
    distributions = {aid: _token_distribution(agent_outputs[aid]) for aid in agent_ids}

    matrix: dict[str, dict[str, float]] = {aid: {} for aid in agent_ids}
    for i, aid_i in enumerate(agent_ids):
        matrix[aid_i][aid_i] = 0.0
        for j, aid_j in enumerate(agent_ids):
            if j <= i:
                continue
            p = distributions[aid_i]
            q = distributions[aid_j]

            # Mixture distribution M = 0.5*(P+Q)
            all_tokens = set(p) | set(q)
            m = {tok: 0.5 * (p.get(tok, 0.0) + q.get(tok, 0.0)) for tok in all_tokens}

            jsd = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
            # Clamp to [0, 1] to guard against floating-point rounding
            jsd = max(0.0, min(1.0, jsd))
            matrix[aid_i][aid_j] = jsd
            matrix[aid_j][aid_i] = jsd

    return matrix


def mean_jsd(jsd_matrix: dict[str, dict[str, float]]) -> float:
    """Return the mean of all upper-triangle JSD values.

    Parameters
    ----------
    jsd_matrix : dict
        Nested dict returned by :func:`compute_jsd_matrix`.

    Returns
    -------
    float
        Mean JSD value, or ``0.0`` if fewer than 2 agents.
    """
    agent_ids = list(jsd_matrix.keys())
    values: list[float] = []
    for i, aid_i in enumerate(agent_ids):
        for j, aid_j in enumerate(agent_ids):
            if j > i:
                values.append(jsd_matrix[aid_i][aid_j])
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Generation logging
# ---------------------------------------------------------------------------


def build_generation_log(
    generation: int,
    biome_name: str,
    population: Population,
    jsd_matrix: dict[str, dict[str, float]],
    deaths: list[str],
    births: list[str],
    pop_mean_fitness: float | None = None,
    pop_sigma_fitness: float | None = None,
    mean_p_death: float | None = None,
) -> dict:
    """Build a JSON-serializable summary dict for one generation.

    Parameters
    ----------
    generation : int
        Generation index.
    biome_name : str
        Biome label (e.g. ``"savanna"``).
    population : Population
        Population *after* deaths and births have been applied.
    jsd_matrix : dict
        Pairwise JSD matrix from :func:`compute_jsd_matrix`.
    deaths : list[str]
        agent_ids that died this generation.
    births : list[str]
        agent_ids of offspring born this generation.
    pop_mean_fitness : float, optional
        Pre-death population mean fitness (mu_f from z-score mechanics).
    pop_sigma_fitness : float, optional
        Pre-death population fitness std (sigma_f from z-score mechanics).
    mean_p_death : float, optional
        Mean death probability across all agents this generation.

    Returns
    -------
    dict
        Serializable generation log.
    """
    fitnesses = [
        state.metrics.get("fitness", float("nan"))
        for state in population.values()
    ]
    valid = [f for f in fitnesses if not math.isnan(f)]
    mean_fit = float(np.mean(valid)) if valid else float("nan")
    std_fit = float(np.std(valid)) if valid else float("nan")

    agents_list = [
        {
            "agent_id": state.agent_id,
            "parent_id": state.parent_id,
            "fitness": state.metrics.get("fitness"),
            "h_x": state.metrics.get("h_x"),
            "c_x": state.metrics.get("c_x"),
            "i_x_seed": state.metrics.get("i_x_seed"),
            "h_dezorg": state.metrics.get("h_dezorg"),
        }
        for state in population.values()
    ]

    log_dict: dict = {
        "generation": generation,
        "biome": biome_name,
        "timestamp": now_utc_iso(),
        "population_size": len(population),
        "mean_fitness": mean_fit,
        "std_fitness": std_fit,
        "mean_jsd": mean_jsd(jsd_matrix),
        "deaths": deaths,
        "births": births,
        "agents": agents_list,
    }
    # New fields for z-score mechanics (Phase 1b); not present in earlier runs.
    if pop_mean_fitness is not None:
        log_dict["pop_mean_fitness"] = pop_mean_fitness
    if pop_sigma_fitness is not None:
        log_dict["pop_sigma_fitness"] = pop_sigma_fitness
    if mean_p_death is not None:
        log_dict["mean_p_death"] = mean_p_death
    return log_dict


# ---------------------------------------------------------------------------
# Phylogeny tracking
# ---------------------------------------------------------------------------


def append_phylogeny_log(
    output_dir: str,
    biome_name: str,
    generation: int,
    generation_agents: dict[str, dict],
    dying_ids: list[str],
) -> None:
    """Append per-agent phylogeny records for one generation to phylogeny.jsonl.

    One line is written per agent, recording its lineage and whether it
    survived to the end of the generation.  The file is opened in append mode
    so records accumulate across generations.

    Parameters
    ----------
    output_dir : str
        Root output directory for the run.
    biome_name : str
        Biome label.
    generation : int
        Generation index.
    generation_agents : dict[str, dict]
        Mapping of ``agent_id -> {"parent_id": str | None, "fitness": float}``
        for every agent that participated in this generation (including those
        that died).
    dying_ids : list[str]
        agent_ids that died this generation (``alive`` is set to ``False``).
    """
    phylo_dir = Path(output_dir) / biome_name
    phylo_dir.mkdir(parents=True, exist_ok=True)
    phylo_path = phylo_dir / "phylogeny.jsonl"

    dying_set = set(dying_ids)
    with open(phylo_path, "a", encoding="utf-8") as fh:
        for agent_id, data in generation_agents.items():
            record = {
                "generation": generation,
                "agent_id": agent_id,
                "parent_id": data.get("parent_id"),
                "fitness": data.get("fitness"),
                "alive": agent_id not in dying_set,
            }
            fh.write(json.dumps(record) + "\n")

    log.info(
        "append_phylogeny_log — generation=%d, %d records written to %s",
        generation,
        len(generation_agents),
        phylo_path,
    )


def update_phylogeny_graph(
    output_dir: str,
    biome_name: str,
    generation: int,
    generation_agents: dict[str, dict],
) -> None:
    """Update the cumulative phylogeny graph with new nodes and edges.

    Reads the existing ``phylogeny_graph.json`` (if any), appends new nodes
    and edges for agents in this generation, then writes back atomically.
    Existing nodes and edges are never removed or modified.

    A warning is logged for any agent whose parent is not found among the
    graph's existing nodes (orphan agent).

    Parameters
    ----------
    output_dir : str
        Root output directory for the run.
    biome_name : str
        Biome label.
    generation : int
        Generation index.
    generation_agents : dict[str, dict]
        Mapping of ``agent_id -> {"parent_id": str | None, "fitness": float}``
        for every agent that participated in this generation.
    """
    phylo_dir = Path(output_dir) / biome_name
    phylo_dir.mkdir(parents=True, exist_ok=True)
    graph_path = phylo_dir / "phylogeny_graph.json"

    if graph_path.is_file():
        with open(graph_path, encoding="utf-8") as fh:
            graph: dict = json.load(fh)
    else:
        graph = {"nodes": [], "edges": []}

    existing_node_ids: set[str] = {n["id"] for n in graph["nodes"]}
    existing_edge_keys: set[tuple[str, str]] = {
        (e["parent"], e["child"]) for e in graph["edges"]
    }

    for agent_id, data in generation_agents.items():
        if agent_id not in existing_node_ids:
            graph["nodes"].append(
                {
                    "id": agent_id,
                    "generation": generation,
                    "fitness": data.get("fitness"),
                }
            )
            existing_node_ids.add(agent_id)

        parent_id = data.get("parent_id")
        if parent_id is not None:
            if parent_id not in existing_node_ids:
                log.warning(
                    "update_phylogeny_graph — orphan agent %s: "
                    "parent %s not found in graph nodes (generation=%d)",
                    agent_id,
                    parent_id,
                    generation,
                )
            edge_key = (parent_id, agent_id)
            if edge_key not in existing_edge_keys:
                graph["edges"].append(
                    {
                        "parent": parent_id,
                        "child": agent_id,
                        "generation": generation,
                    }
                )
                existing_edge_keys.add(edge_key)

    graph_tmp = str(graph_path) + ".tmp"
    with open(graph_tmp, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)
    os.replace(graph_tmp, str(graph_path))

    log.info(
        "update_phylogeny_graph — generation=%d, total nodes=%d, total edges=%d",
        generation,
        len(graph["nodes"]),
        len(graph["edges"]),
    )


def build_lineage_tree(phylogeny_graph_path: str) -> dict[str, list[str]]:
    """Return full ancestor chains for every agent in the phylogeny graph.

    Traverses the directed parent→child edges stored in
    ``phylogeny_graph.json`` and builds a mapping from each agent to its
    complete ancestor chain ordered from immediate parent to root.

    Parameters
    ----------
    phylogeny_graph_path : str
        Path to ``phylogeny_graph.json``.

    Returns
    -------
    dict[str, list[str]]
        Mapping of ``agent_id -> [immediate_parent, grandparent, ..., root]``.
        Root agents (no recorded parent) map to an empty list.

    Raises
    ------
    FileNotFoundError
        If *phylogeny_graph_path* does not exist.
    """
    graph_path = Path(phylogeny_graph_path)
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Phylogeny graph not found: {phylogeny_graph_path!r}"
        )

    with open(graph_path, encoding="utf-8") as fh:
        graph: dict = json.load(fh)

    # child -> parent lookup
    parent_map: dict[str, str] = {
        e["child"]: e["parent"] for e in graph.get("edges", [])
    }
    node_ids: set[str] = {n["id"] for n in graph.get("nodes", [])}

    lineage_cache: dict[str, list[str]] = {}

    def _ancestors(agent_id: str, visited: set[str]) -> list[str]:
        if agent_id in lineage_cache:
            return lineage_cache[agent_id]
        if agent_id in visited:
            log.warning(
                "build_lineage_tree — cycle detected at agent %s; truncating chain",
                agent_id,
            )
            return []
        parent = parent_map.get(agent_id)
        if parent is None:
            lineage_cache[agent_id] = []
            return []
        chain = [parent] + _ancestors(parent, visited | {agent_id})
        lineage_cache[agent_id] = chain
        return chain

    return {nid: _ancestors(nid, set()) for nid in node_ids}


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def checkpoint_dir(output_dir: str, biome_name: str, generation: int) -> str:
    """Return the path for a generation checkpoint directory.

    Path pattern: ``output_dir/biome_name/generation_{generation:03d}/``

    Parameters
    ----------
    output_dir : str
        Root output directory for the run.
    biome_name : str
        Biome label.
    generation : int
        Generation index.

    Returns
    -------
    str
        Absolute directory path (not yet created).
    """
    return str(
        Path(output_dir) / biome_name / f"generation_{generation:03d}"
    )


def save_generation_checkpoint(
    output_dir: str,
    biome_name: str,
    generation: int,
    population: Population,
    generation_log: dict,
) -> None:
    """Persist population state and generation log to a checkpoint directory.

    Both files are written atomically (write to ``.tmp`` then ``os.replace``).

    Parameters
    ----------
    output_dir : str
        Root output directory.
    biome_name : str
        Biome label.
    generation : int
        Generation index.
    population : Population
        Population after all mechanics for this generation.
    generation_log : dict
        Serializable log from :func:`build_generation_log`.
    """
    ckpt = Path(checkpoint_dir(output_dir, biome_name, generation))
    ckpt.mkdir(parents=True, exist_ok=True)

    pop_path = str(ckpt / "population.json")
    save_population(population, pop_path)

    log_path = str(ckpt / "generation_log.json")
    log_tmp = log_path + ".tmp"
    with open(log_tmp, "w", encoding="utf-8") as fh:
        json.dump(generation_log, fh, indent=2)
    os.replace(log_tmp, log_path)

    log.info(
        "save_generation_checkpoint — generation=%d saved to %s",
        generation,
        ckpt,
    )


def save_agent_checkpoint(
    output_dir: str,
    biome_name: str,
    generation: int,
    agent_id: str,
    metrics: dict,
    adapter_path: str,
) -> None:
    """Append one completed-agent record for mid-generation resume."""
    partial_dir = Path(output_dir) / biome_name / f"generation_{generation:03d}_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "agent_id": agent_id,
        "metrics": metrics,
        "adapter_path": adapter_path,
        "timestamp": now_utc_iso(),
    }
    partial_path = partial_dir / "agents_completed.jsonl"
    with open(partial_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_partial_checkpoint(
    output_dir: str,
    biome_name: str,
    generation: int,
) -> dict[str, dict[str, Any]]:
    """Load completed-agent records for an in-progress generation."""
    partial_path = (
        Path(output_dir)
        / biome_name
        / f"generation_{generation:03d}_partial"
        / "agents_completed.jsonl"
    )
    if not partial_path.is_file():
        return {}

    completed: dict[str, dict[str, Any]] = {}
    with open(partial_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed[str(record["agent_id"])] = {
                "metrics": dict(record["metrics"]),
                "adapter_path": str(record["adapter_path"]),
            }

    log.info(
        "load_partial_checkpoint — generation=%d loaded %d completed agents",
        generation,
        len(completed),
    )
    return completed


def delete_partial_checkpoint(
    output_dir: str,
    biome_name: str,
    generation: int,
) -> None:
    """Remove the partial checkpoint directory after a full generation save."""
    partial_dir = Path(output_dir) / biome_name / f"generation_{generation:03d}_partial"
    if partial_dir.is_dir():
        shutil.rmtree(partial_dir)
        log.info(
            "delete_partial_checkpoint — generation=%d removed %s",
            generation,
            partial_dir,
        )


def load_latest_checkpoint(
    output_dir: str,
    biome_name: str,
    resume_from_generation: int,
) -> tuple[Population, int]:
    """Load population checkpoint from generation *resume_from_generation* − 1.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    biome_name : str
        Biome label.
    resume_from_generation : int
        The generation to start from; the checkpoint at generation − 1 is
        loaded.

    Returns
    -------
    tuple[Population, int]
        ``(population, resume_from_generation)``

    Raises
    ------
    FileNotFoundError
        If the checkpoint directory or population file is missing.
    ValueError
        If *resume_from_generation* is 0 (no prior checkpoint to load).
    """
    if resume_from_generation == 0:
        raise ValueError(
            "resume_from_generation=0 means a fresh start; "
            "load_latest_checkpoint should not be called."
        )

    prev_gen = resume_from_generation - 1
    ckpt = Path(checkpoint_dir(output_dir, biome_name, prev_gen))
    pop_path = str(ckpt / "population.json")

    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory missing: {ckpt}. "
            f"Cannot resume from generation {resume_from_generation}."
        )
    if not (ckpt / "population.json").is_file():
        raise FileNotFoundError(
            f"Population file missing: {pop_path}. "
            f"Cannot resume from generation {resume_from_generation}."
        )

    population = load_population(pop_path)
    log.info(
        "load_latest_checkpoint — loaded gen=%d checkpoint (%d agents)",
        prev_gen,
        len(population),
    )
    return population, resume_from_generation


def _run_agent_subprocess(
    agent_id: str,
    base_model_name: str,
    parent_adapter_path: str | None,
    documents: list[str],
    seed_output: str,
    config: dict,
    config_path: str,
    output_dir: str,
    metadata: AdapterMetadata,
    tmp_dir: str,
) -> tuple[str, dict]:
    """Run agent training in an isolated subprocess to avoid VRAM fragmentation.
    
    Parameters
    ----------
    agent_id : str
        Unique agent identifier
    base_model_name : str
        Name of the base model
    parent_adapter_path : str | None
        Path to parent adapter or None for base model
    documents : list[str]
        Training documents
    seed_output : str
        Seed output from base model
    config : dict
        Configuration dictionary
    config_path : str
        Path to config file
    output_dir : str
        Output directory for adapter
    metadata : AdapterMetadata
        Adapter metadata
    tmp_dir : str
        Temporary directory for IPC files
        
    Returns
    -------
    tuple[str, dict]
        (adapter_path, metrics) from training
    """
    import subprocess
    import tempfile
    
    os.makedirs(tmp_dir, exist_ok=True)
    docs_file = os.path.join(tmp_dir, f"{agent_id}_docs.json")
    result_file = os.path.join(tmp_dir, f"{agent_id}_result.json")
    
    # Serialize documents
    with open(docs_file, "w") as f:
        json.dump(documents, f)
    
    # Serialize metadata
    metadata_dict = {
        "generation": metadata.generation,
        "parent_id": metadata.parent_id,
        "biome": metadata.biome,
        "archetype": metadata.archetype,
        "fitness_score": metadata.fitness_score,
        "creation_timestamp": metadata.creation_timestamp,
    }
    
    cmd = [
        sys.executable,
        "-m",
        "src.evolution.worker",
        "--agent-id",
        agent_id,
        "--base-model",
        base_model_name,
        "--parent-adapter",
        parent_adapter_path or "none",
        "--docs-file",
        docs_file,
        "--output-file",
        result_file,
        "--seed-output",
        seed_output,
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--metadata-json",
        json.dumps(metadata_dict),
    ]
    
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, timeout=900)
    
    with open(result_file) as f:
        data = json.load(f)
    
    os.unlink(docs_file)
    os.unlink(result_file)
    
    if data["status"] != "ok":
        raise RuntimeError(f"Worker failed for {agent_id}: {data.get('error')}")
    
    return data["adapter_path"], data["metrics"]


# ---------------------------------------------------------------------------
# Seed output (base model, no adapter)
# ---------------------------------------------------------------------------


def _generate_seed_output(base_model_name: str) -> str:
    """Generate the reference seed output from the base model (no adapter).

    Called once at the start of each run. The base model remains resident in
    trainer's cache for later adapter training.

    Parameters
    ----------
    base_model_name : str
        Unsloth 4-bit base model name.

    Returns
    -------
    str
        Generated text from the base model given ``DIAGNOSTIC_PROMPT``.
    """
    import torch
    from unsloth import FastLanguageModel  # noqa: F401 (already imported by trainer)
    from src.evolution.trainer import _BASE_MODEL_CACHE, get_base_model

    log.info("_generate_seed_output — reusing cached base model %s", base_model_name)
    model, tokenizer = get_base_model(base_model_name)
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(DIAGNOSTIC_PROMPT, return_tensors="pt").to(model.device)
    prompt_len: int = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )

    new_tokens = output_ids[0][prompt_len:]
    seed_text: str = tokenizer.decode(new_tokens, skip_special_tokens=True)

    torch.cuda.empty_cache()
    if _BASE_MODEL_CACHE["model"] is model:
        log.info("_generate_seed_output — base model retained in trainer cache")
    log.info("_generate_seed_output — seed output generated (%d chars)", len(seed_text))
    return seed_text


# ---------------------------------------------------------------------------
# Main orchestration loop
# ---------------------------------------------------------------------------


def run_biome(
    biome_name: str,
    config_path: str,
    corpus_manifest_path: str,
    output_dir: str,
    n_generations: int,
    n_agents: int,
    n_documents_per_agent: int,
    resume_from_generation: int = 0,
    seed: int = 42,
) -> None:
    """Run a multi-generation evolutionary experiment in one biome.

    Orchestrates corpus sampling, adapter training, metric measurement,
    JSD computation, selection, reproduction, cannibalism, checkpointing,
    and generation logging.

    Generation sequence
    -------------------
    1. Sample documents from biome (generation-level pool, each agent
       receives a contiguous non-overlapping slice of *n_documents_per_agent*).
    2. For each agent: :func:`train_adapter` with the agent's current adapter
       as parent (or ``None`` for generation 0).
    3. For each agent: :func:`load_adapter` → :func:`measure_metrics` → unload
       (``del model``, ``torch.cuda.empty_cache()`` after each agent).
    4. Compute pairwise JSD matrix across all agent outputs.
    5. Apply death selection (independent Bernoulli per agent).
    6. Apply reproduction selection (surviving parents produce offspring).
    7. Apply cannibalism if ``len(population) > k_max``.
    8. Save checkpoint (population + generation log).
    9. Log generation summary at INFO level.

    Parameters
    ----------
    biome_name : str
        Name of the biome, must be a key in the config's ``biomes`` section
        (e.g. ``"savanna"``).
    config_path : str
        Path to the Phase-1 YAML config file.
    corpus_manifest_path : str
        Path to ``corpus_manifest_v3.json``.
    output_dir : str
        Root directory for all run outputs.
    n_generations : int
        Total number of generations to run.
    n_agents : int
        Initial population size (generation 0).
    n_documents_per_agent : int
        Number of training documents per agent per generation.
    resume_from_generation : int
        If 0, start fresh.  If N > 0, load checkpoint from generation N − 1
        and continue from generation N.
    seed : int
        Base RNG seed; each generation uses ``seed + generation``.
    """
    import torch  # noqa: F401 — torch is imported early for memory management

    config_path_obj = Path(config_path)
    with open(config_path_obj, encoding="utf-8") as fh:
        config: dict = yaml.safe_load(fh)

    biomes_cfg: dict = config.get("biomes", {})
    if biome_name not in biomes_cfg:
        raise ValueError(
            f"Biome '{biome_name}' not found in config {config_path!r}. "
            f"Available: {list(biomes_cfg.keys())}"
        )
    biome_cfg: dict = biomes_cfg[biome_name]
    k_max: int = int(biome_cfg["k_max"])
    base_model_name: str = _BASE_MODEL

    adapters_dir = str(Path(output_dir) / biome_name / "adapters")
    Path(adapters_dir).mkdir(parents=True, exist_ok=True)

    tmp_dir = str(Path(output_dir) / biome_name / "tmp")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load corpus once (shared across all generations)
    # -----------------------------------------------------------------------
    log.info("run_biome START — biome=%s, n_generations=%d, seed=%d", biome_name, n_generations, seed)
    corpus_cache = _load_corpus_by_type(corpus_manifest_path)

    # -----------------------------------------------------------------------
    # Generate seed output once (base model, no adapter)
    # -----------------------------------------------------------------------
    seed_output: str = _generate_seed_output(base_model_name)

    # -----------------------------------------------------------------------
    # Initialise or resume population
    # -----------------------------------------------------------------------
    population: Population
    agent_counter: int  # monotonic index for new agent IDs

    if resume_from_generation == 0:
        population = {}
        agent_counter = 0
        log.info("run_biome — fresh start, creating %d agents at generation 0", n_agents)
    else:
        population, _ = load_latest_checkpoint(output_dir, biome_name, resume_from_generation)
        # Infer agent_counter from existing IDs to avoid collisions
        agent_counter = _next_agent_index(population)
        log.info(
            "run_biome — resuming from generation %d, %d agents loaded",
            resume_from_generation,
            len(population),
        )

    start_gen = resume_from_generation
    end_gen = start_gen + n_generations

    for generation in range(start_gen, end_gen):
        gen_rng = np.random.default_rng(seed + generation)
        completed = load_partial_checkpoint(output_dir, biome_name, generation)
        log.info(
            "=== Generation %d/%d — biome=%s, population=%d ===",
            generation,
            end_gen - 1,
            biome_name,
            len(population),
        )

        # -------------------------------------------------------------------
        # Step 1: Sample generation-level document pool
        # -------------------------------------------------------------------
        current_size = max(len(population) if population else n_agents, 1)
        pool_size = current_size * n_documents_per_agent
        generation_pool = sample_biome_documents(
            corpus_manifest_path=corpus_manifest_path,
            biome_config=biome_cfg,
            n_documents=pool_size,
            generation=generation,
            rng=gen_rng,
            _corpus_cache=corpus_cache,
        )

        # -------------------------------------------------------------------
        # Step 2 + 3: Train each agent, measure metrics
        # -------------------------------------------------------------------
        if generation == 0:
            # Fresh start: create n_agents new agents with no parent
            new_population: Population = {}
            agent_outputs: dict[str, str] = {}

            for idx in range(n_agents):
                agent_id = build_agent_id(_PHASE1_ARCHETYPE, generation, agent_counter)
                agent_counter += 1

                doc_slice = _agent_doc_slice(generation_pool, idx, n_documents_per_agent)
                gen_adapter_dir = str(Path(adapters_dir) / f"gen{generation:03d}")

                metadata = AdapterMetadata(
                    generation=generation,
                    parent_id="none",
                    biome=biome_name,
                    archetype=_PHASE1_ARCHETYPE,
                    fitness_score=0.0,
                    creation_timestamp=now_utc_iso(),
                )

                if agent_id in completed:
                    log.info("Skipping agent %s — already completed", agent_id)
                    adapter_path = str(completed[agent_id]["adapter_path"])
                    metrics = dict(completed[agent_id]["metrics"])
                else:
                    adapter_path, metrics = _run_agent_subprocess(
                        agent_id=agent_id,
                        base_model_name=base_model_name,
                        parent_adapter_path=None,
                        documents=doc_slice,
                        seed_output=seed_output,
                        config=config,
                        config_path=config_path,
                        output_dir=gen_adapter_dir,
                        metadata=metadata,
                        tmp_dir=tmp_dir,
                    )
                    save_agent_checkpoint(
                        output_dir=output_dir,
                        biome_name=biome_name,
                        generation=generation,
                        agent_id=agent_id,
                        metrics=metrics,
                        adapter_path=adapter_path,
                    )
                output_text = str(metrics.pop("output_text"))
                agent_outputs[agent_id] = output_text

                # Update metadata with measured fitness
                metadata = AdapterMetadata(
                    generation=generation,
                    parent_id="none",
                    biome=biome_name,
                    archetype=_PHASE1_ARCHETYPE,
                    fitness_score=float(metrics.get("fitness", 0.0)),
                    creation_timestamp=metadata.creation_timestamp,
                )

                state = AgentState(
                    agent_id=agent_id,
                    adapter_path=adapter_path,
                    metadata=metadata,
                    metrics=metrics,
                    generation=generation,
                    parent_id=None,
                    alive=True,
                )
                new_population[agent_id] = state
                log.info(
                    "gen=%d agent=%s fitness=%.4f",
                    generation, agent_id, metrics.get("fitness", float("nan"))
                )

            population = new_population

        else:
            # Existing agents: retrain each with their current adapter as parent
            agent_outputs = {}
            updated_population: Population = {}

            for idx, (agent_id, state) in enumerate(population.items()):
                doc_slice = _agent_doc_slice(generation_pool, idx, n_documents_per_agent)
                gen_adapter_dir = str(Path(adapters_dir) / f"gen{generation:03d}")

                metadata = AdapterMetadata(
                    generation=generation,
                    parent_id=state.agent_id,
                    biome=biome_name,
                    archetype=state.metadata.archetype,
                    fitness_score=state.metadata.fitness_score,
                    creation_timestamp=now_utc_iso(),
                )

                if agent_id in completed:
                    log.info("Skipping agent %s — already completed", agent_id)
                    adapter_path = str(completed[agent_id]["adapter_path"])
                    metrics = dict(completed[agent_id]["metrics"])
                else:
                    adapter_path, metrics = _run_agent_subprocess(
                        agent_id=agent_id,
                        base_model_name=base_model_name,
                        parent_adapter_path=state.adapter_path,
                        documents=doc_slice,
                        seed_output=seed_output,
                        config=config,
                        config_path=config_path,
                        output_dir=gen_adapter_dir,
                        metadata=metadata,
                        tmp_dir=tmp_dir,
                    )
                    save_agent_checkpoint(
                        output_dir=output_dir,
                        biome_name=biome_name,
                        generation=generation,
                        agent_id=agent_id,
                        metrics=metrics,
                        adapter_path=adapter_path,
                    )
                output_text = str(metrics.pop("output_text"))
                agent_outputs[agent_id] = output_text

                updated_state = AgentState(
                    agent_id=state.agent_id,
                    adapter_path=adapter_path,
                    metadata=AdapterMetadata(
                        generation=generation,
                        parent_id=state.agent_id,
                        biome=biome_name,
                        archetype=state.metadata.archetype,
                        fitness_score=float(metrics.get("fitness", 0.0)),
                        creation_timestamp=metadata.creation_timestamp,
                    ),
                    metrics=metrics,
                    generation=generation,
                    parent_id=state.agent_id,
                    alive=True,
                )
                updated_population[agent_id] = updated_state
                log.info(
                    "gen=%d agent=%s fitness=%.4f",
                    generation, agent_id, metrics.get("fitness", float("nan"))
                )

            population = updated_population

        # -------------------------------------------------------------------
        # Step 4: JSD matrix
        # -------------------------------------------------------------------
        jsd_matrix = compute_jsd_matrix(agent_outputs)
        m_jsd = mean_jsd(jsd_matrix)
        log.info("gen=%d mean_JSD=%.4f", generation, m_jsd)

        # Capture all agent data before deaths so dying agents are still
        # included in phylogeny records with their measured fitness.
        _generation_agents: dict[str, dict] = {
            aid: {
                "parent_id": s.parent_id,
                "fitness": float(s.metrics.get("fitness", float("nan"))),
            }
            for aid, s in population.items()
        }

        # -------------------------------------------------------------------
        # Step 4.5: Z-score mechanics (Phase 1b)
        # -------------------------------------------------------------------
        pop_mech_params = _get_population_mechanics_params(config)
        zscores, mu_f, sigma_f = _compute_population_zscores(
            population, pop_mech_params["sigma_min"]
        )
        death_probs_map = zscore_death_probs(population, pop_mech_params, zscores)
        mean_p_death = float(np.mean(list(death_probs_map.values()))) if death_probs_map else 0.0
        log.info(
            "gen=%d zscore — mu_f=%.4f sigma_f=%.4f mean_p_death=%.4f",
            generation, mu_f, sigma_f, mean_p_death,
        )

        # -------------------------------------------------------------------
        # Step 5: Deaths
        # -------------------------------------------------------------------
        dying_ids = select_candidates_for_death_zscore(
            population, gen_rng, pop_mech_params, zscores
        )
        population = apply_deaths(population, dying_ids)
        log.info(
            "gen=%d deaths=%d remaining=%d", generation, len(dying_ids), len(population)
        )

        # -------------------------------------------------------------------
        # Step 6: Reproduction — train offspring from selected parents
        # -------------------------------------------------------------------
        births: list[str] = []
        reproducing_ids = select_candidates_for_reproduction_zscore(
            population, gen_rng, pop_mech_params, zscores
        )

        for _ in reproducing_ids:
            if not population:
                break
            parent_id = select_parent(population, gen_rng)
            parent_state = population[parent_id]

            offspring_id = build_agent_id(_PHASE1_ARCHETYPE, generation, agent_counter)
            agent_counter += 1

            offspring_doc_slice = sample_biome_documents(
                corpus_manifest_path=corpus_manifest_path,
                biome_config=biome_cfg,
                n_documents=n_documents_per_agent,
                generation=generation,
                rng=gen_rng,
                _corpus_cache=corpus_cache,
            )

            offspring_gen_dir = str(Path(adapters_dir) / f"gen{generation:03d}_offspring")

            offspring_metadata = AdapterMetadata(
                generation=generation,
                parent_id=parent_id,
                biome=biome_name,
                archetype=_PHASE1_ARCHETYPE,
                fitness_score=0.0,
                creation_timestamp=now_utc_iso(),
            )

            if offspring_id in completed:
                log.info("Skipping agent %s — already completed", offspring_id)
                offspring_adapter_path = str(completed[offspring_id]["adapter_path"])
                offspring_metrics = dict(completed[offspring_id]["metrics"])
            else:
                offspring_adapter_path, offspring_metrics = _run_agent_subprocess(
                    agent_id=offspring_id,
                    base_model_name=base_model_name,
                    parent_adapter_path=parent_state.adapter_path,
                    documents=offspring_doc_slice,
                    seed_output=seed_output,
                    config=config,
                    config_path=config_path,
                    output_dir=offspring_gen_dir,
                    metadata=offspring_metadata,
                    tmp_dir=tmp_dir,
                )
                save_agent_checkpoint(
                    output_dir=output_dir,
                    biome_name=biome_name,
                    generation=generation,
                    agent_id=offspring_id,
                    metrics=offspring_metrics,
                    adapter_path=offspring_adapter_path,
                )
            offspring_text = str(offspring_metrics.pop("output_text"))

            population = register_offspring(
                population=population,
                parent_id=parent_id,
                new_agent_id=offspring_id,
                new_adapter_path=offspring_adapter_path,
                new_metrics=offspring_metrics,
                generation=generation,
            )
            births.append(offspring_id)
            # Record offspring in phylogeny snapshot (born after the pre-death
            # capture, so not yet present in _generation_agents).
            _generation_agents[offspring_id] = {
                "parent_id": population[offspring_id].parent_id,
                "fitness": float(
                    population[offspring_id].metrics.get("fitness", float("nan"))
                ),
            }
            log.info(
                "gen=%d offspring=%s parent=%s fitness=%.4f",
                generation,
                offspring_id,
                parent_id,
                offspring_metrics.get("fitness", float("nan")),
            )

        log.info("gen=%d births=%d", generation, len(births))

        # -------------------------------------------------------------------
        # Step 7: Cannibalism if population > k_max
        # -------------------------------------------------------------------
        if check_carrying_capacity(population, k_max):
            log.info(
                "gen=%d population=%d > k_max=%d — applying cannibalism",
                generation,
                len(population),
                k_max,
            )
            population = apply_cannibalism(
                population=population,
                k_max=k_max,
                adapters_dir=adapters_dir,
                rng=gen_rng,
            )
            log.info("gen=%d post-cannibalism population=%d", generation, len(population))

        # -------------------------------------------------------------------
        # Step 8: Checkpoint
        # -------------------------------------------------------------------
        gen_log = build_generation_log(
            generation=generation,
            biome_name=biome_name,
            population=population,
            jsd_matrix=jsd_matrix,
            deaths=dying_ids,
            births=births,
            pop_mean_fitness=mu_f,
            pop_sigma_fitness=sigma_f,
            mean_p_death=mean_p_death,
        )
        save_generation_checkpoint(
            output_dir=output_dir,
            biome_name=biome_name,
            generation=generation,
            population=population,
            generation_log=gen_log,
        )
        delete_partial_checkpoint(
            output_dir=output_dir,
            biome_name=biome_name,
            generation=generation,
        )

        # -------------------------------------------------------------------
        # Phylogeny tracking — append per-agent records and update graph
        # -------------------------------------------------------------------
        append_phylogeny_log(
            output_dir=output_dir,
            biome_name=biome_name,
            generation=generation,
            generation_agents=_generation_agents,
            dying_ids=dying_ids,
        )
        update_phylogeny_graph(
            output_dir=output_dir,
            biome_name=biome_name,
            generation=generation,
            generation_agents=_generation_agents,
        )

        # -------------------------------------------------------------------
        # Step 9: Log generation summary
        # -------------------------------------------------------------------
        log.info(
            "=== gen=%d END — pop=%d mean_fitness=%.4f std=%.4f mean_JSD=%.4f "
            "deaths=%d births=%d ===",
            generation,
            gen_log["population_size"],
            gen_log["mean_fitness"],
            gen_log["std_fitness"],
            gen_log["mean_jsd"],
            len(dying_ids),
            len(births),
        )

    log.info(
        "run_biome COMPLETE — biome=%s, generations=%d-%d, final_pop=%d",
        biome_name,
        start_gen,
        end_gen - 1,
        len(population),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _agent_doc_slice(
    pool: list[str], agent_idx: int, n_per_agent: int
) -> list[str]:
    """Extract a non-overlapping document slice for one agent.

    If the pool is exhausted, wraps around using modular indexing so every
    agent always receives exactly *n_per_agent* documents.

    Parameters
    ----------
    pool : list[str]
        Generation-level document pool.
    agent_idx : int
        Zero-based agent index within the current generation.
    n_per_agent : int
        Number of documents per agent.

    Returns
    -------
    list[str]
        Slice of *n_per_agent* content strings.
    """
    if not pool:
        return []
    start = (agent_idx * n_per_agent) % len(pool)
    end = start + n_per_agent
    if end <= len(pool):
        return pool[start:end]
    # Wrap around
    return pool[start:] + pool[: end - len(pool)]


def _next_agent_index(population: Population) -> int:
    """Return the next available agent index to avoid ID collisions on resume.

    Parses the trailing numeric component from all agent IDs in the population
    and returns ``max_index + 1``.

    Parameters
    ----------
    population : Population
        Existing population loaded from a checkpoint.

    Returns
    -------
    int
        Safe starting index for new agents.
    """
    max_idx = -1
    for agent_id in population:
        parts = agent_id.rsplit("_", 1)
        if len(parts) == 2:
            try:
                max_idx = max(max_idx, int(parts[1]))
            except ValueError:
                pass
    return max_idx + 1
