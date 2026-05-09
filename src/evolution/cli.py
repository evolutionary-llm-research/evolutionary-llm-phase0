"""Command-line interface for Phase 1 biome experiments.

Provides a single entry point to run, resume, and inspect evolutionary
experiments per biome.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.evolution.biome_runner import run_biome

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: str = "config/phase1_single_model.yaml"
DEFAULT_OUTPUT_DIR: str = "experiments/phase1/"
DEFAULT_CORPUS_MANIFEST: str = "data/v2/corpus_manifest_v3.json"
DEFAULT_SEED: int = 42

_BIOME_ALIASES: dict[str, str] = {
    "plains": "plain",
}


@dataclass(frozen=True)
class BiomeStatus:
    """Summary of the latest checkpoint state for one biome run."""

    biome: str
    completed_generations: int
    last_generation: int
    population_size: int
    last_mean_fitness: float | None
    last_fitness_values: list[float]


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load YAML config file and return a mapping.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML mapping.
    """
    with open(config_path, encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path!r}")
    return payload


def _extract_seed(config: dict[str, Any]) -> int:
    """Return project seed from config, falling back to DEFAULT_SEED."""
    project = config.get("project", {})
    if isinstance(project, dict) and "seed" in project:
        return int(project["seed"])
    return DEFAULT_SEED


def _extract_corpus_manifest(config: dict[str, Any], override: str | None) -> str:
    """Return corpus manifest path from override or config defaults."""
    if override:
        return override

    corpus_cfg = config.get("corpus", {})
    if isinstance(corpus_cfg, dict):
        manifest = corpus_cfg.get("manifest")
        if isinstance(manifest, str) and manifest.strip():
            return manifest

    return DEFAULT_CORPUS_MANIFEST


def _resolve_biome_name(requested_biome: str, config: dict[str, Any]) -> str:
    """Validate and resolve biome name against config.

    Supports alias mapping (for example ``plains`` to ``plain``) while keeping
    validation based on config-defined biome keys.
    """
    biomes = config.get("biomes", {})
    if not isinstance(biomes, dict) or not biomes:
        raise ValueError("Config is missing non-empty 'biomes' mapping")

    available = {str(name).lower() for name in biomes.keys()}
    candidate = requested_biome.lower()

    if candidate not in available:
        mapped = _BIOME_ALIASES.get(candidate)
        if mapped is not None:
            candidate = mapped

    if candidate not in available:
        valid_display = ", ".join(sorted(available | set(_BIOME_ALIASES.keys())))
        raise ValueError(
            f"Invalid biome: {requested_biome!r}. Valid values: {valid_display}."
        )

    return candidate


def _configure_run_logging(output_dir: str, biome_name: str) -> Path:
    """Configure console + file logging for run/resume commands."""
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = logs_dir / f"phase1_{biome_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def _configure_status_logging() -> None:
    """Configure console logging for status command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _parse_generation_index(path: Path) -> int | None:
    """Extract integer generation index from generation directory name."""
    prefix = "generation_"
    if not path.name.startswith(prefix):
        return None
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _collect_generation_dirs(biome_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted generation directories for one biome directory."""
    generations: list[tuple[int, Path]] = []
    for child in biome_dir.iterdir():
        if not child.is_dir():
            continue
        idx = _parse_generation_index(child)
        if idx is not None:
            generations.append((idx, child))
    generations.sort(key=lambda item: item[0])
    return generations


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _collect_biome_status(output_dir: str) -> list[BiomeStatus]:
    """Build status records for all biome directories found in output_dir."""
    root = Path(output_dir)
    if not root.exists():
        return []

    statuses: list[BiomeStatus] = []
    for biome_dir in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not biome_dir.is_dir() or biome_dir.name == "logs":
            continue

        generation_dirs = _collect_generation_dirs(biome_dir)
        if not generation_dirs:
            continue

        last_generation, last_dir = generation_dirs[-1]
        generation_log = _read_json(last_dir / "generation_log.json")

        mean_fitness_raw = generation_log.get("mean_fitness")
        mean_fitness = (
            float(mean_fitness_raw)
            if isinstance(mean_fitness_raw, (int, float))
            else None
        )

        population_size_raw = generation_log.get("population_size", 0)
        population_size = int(population_size_raw)

        fitness_values: list[float] = []
        agents_raw = generation_log.get("agents", [])
        if isinstance(agents_raw, list):
            for agent_item in agents_raw:
                if not isinstance(agent_item, dict):
                    continue
                fitness_raw = agent_item.get("fitness")
                if isinstance(fitness_raw, (int, float)):
                    fitness_values.append(float(fitness_raw))

        statuses.append(
            BiomeStatus(
                biome=biome_dir.name,
                completed_generations=len(generation_dirs),
                last_generation=last_generation,
                population_size=population_size,
                last_mean_fitness=mean_fitness,
                last_fitness_values=fitness_values,
            )
        )

    return statuses


def _format_fitness_values(values: list[float], max_values: int = 5) -> str:
    """Format a compact list of last fitness values for table output."""
    if not values:
        return "-"
    head = values[:max_values]
    text = ", ".join(f"{item:.4f}" for item in head)
    if len(values) > max_values:
        text += ", ..."
    return text


def _render_status_table(items: list[BiomeStatus]) -> str:
    """Render a plain-text status table."""
    headers = [
        "Biome",
        "Completed",
        "Last Gen",
        "Population",
        "Last Mean Fitness",
        "Last Fitness Values",
    ]

    rows = [
        [
            item.biome,
            str(item.completed_generations),
            str(item.last_generation),
            str(item.population_size),
            f"{item.last_mean_fitness:.4f}" if item.last_mean_fitness is not None else "-",
            _format_fitness_values(item.last_fitness_values),
        ]
        for item in items
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _line(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_line(headers), separator]
    lines.extend(_line(row) for row in rows)
    return "\n".join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the run subcommand."""
    config = _load_yaml_config(args.config)
    biome_name = _resolve_biome_name(args.biome, config)
    log_path = _configure_run_logging(args.output_dir, biome_name)

    seed = _extract_seed(config)
    corpus_manifest_path = _extract_corpus_manifest(config, args.corpus_manifest)

    log.info("run command started")
    log.info("biome=%s generations=%d agents=%d docs_per_agent=%d", biome_name, args.generations, args.agents, args.docs_per_agent)
    log.info("config=%s output_dir=%s corpus_manifest=%s seed=%d", args.config, args.output_dir, corpus_manifest_path, seed)
    log.info("file log=%s", log_path)

    run_biome(
        biome_name=biome_name,
        config_path=args.config,
        corpus_manifest_path=corpus_manifest_path,
        output_dir=args.output_dir,
        n_generations=args.generations,
        n_agents=args.agents,
        n_documents_per_agent=args.docs_per_agent,
        resume_from_generation=0,
        seed=seed,
    )
    log.info("run command completed")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    """Execute the resume subcommand."""
    if args.from_generation <= 0:
        raise ValueError("--from-generation must be > 0 for resume")

    config = _load_yaml_config(args.config)
    biome_name = _resolve_biome_name(args.biome, config)
    log_path = _configure_run_logging(args.output_dir, biome_name)

    seed = _extract_seed(config)
    corpus_manifest_path = _extract_corpus_manifest(config, args.corpus_manifest)

    log.info("resume command started")
    log.info(
        "biome=%s from_generation=%d generations=%d agents=%d docs_per_agent=%d",
        biome_name,
        args.from_generation,
        args.generations,
        args.agents,
        args.docs_per_agent,
    )
    log.info("config=%s output_dir=%s corpus_manifest=%s seed=%d", args.config, args.output_dir, corpus_manifest_path, seed)
    log.info("file log=%s", log_path)

    run_biome(
        biome_name=biome_name,
        config_path=args.config,
        corpus_manifest_path=corpus_manifest_path,
        output_dir=args.output_dir,
        n_generations=args.generations,
        n_agents=args.agents,
        n_documents_per_agent=args.docs_per_agent,
        resume_from_generation=args.from_generation,
        seed=seed,
    )
    log.info("resume command completed")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Execute the status subcommand."""
    _configure_status_logging()
    statuses = _collect_biome_status(args.output_dir)

    if not statuses:
        log.warning("No biome runs found in output directory: %s", args.output_dir)
        return 0

    log.info("Phase 1 run status for output directory: %s", args.output_dir)
    log.info("\n%s", _render_status_table(statuses))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build argparse parser with run/resume/status subcommands."""
    parser = argparse.ArgumentParser(
        description="CLI for Phase 1 biome evolutionary experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Start a fresh biome run.",
    )
    run_parser.add_argument("--biome", type=str, required=True, help="Biome name (savanna, desert, plains/plain).")
    run_parser.add_argument("--generations", type=int, default=35, help="Number of generations to run.")
    run_parser.add_argument("--agents", type=int, default=10, help="Initial number of agents.")
    run_parser.add_argument("--docs-per-agent", type=int, default=30, help="Training documents per agent per generation.")
    run_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to Phase 1 YAML config.")
    run_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root output directory for Phase 1 runs.")
    run_parser.add_argument("--corpus-manifest", type=str, default=None, help="Override corpus manifest path.")
    run_parser.set_defaults(func=_cmd_run)

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an existing biome run.",
    )
    resume_parser.add_argument("--biome", type=str, required=True, help="Biome name (savanna, desert, plains/plain).")
    resume_parser.add_argument("--from-generation", type=int, required=True, help="Generation index to resume from (loads checkpoint N-1).")
    resume_parser.add_argument("--generations", type=int, default=35, help="Number of generations to run after resuming.")
    resume_parser.add_argument("--agents", type=int, default=10, help="Fallback agent count (used only if generation 0 is reached).")
    resume_parser.add_argument("--docs-per-agent", type=int, default=30, help="Training documents per agent per generation.")
    resume_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to Phase 1 YAML config.")
    resume_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root output directory for Phase 1 runs.")
    resume_parser.add_argument("--corpus-manifest", type=str, default=None, help="Override corpus manifest path.")
    resume_parser.set_defaults(func=_cmd_resume)

    status_parser = subparsers.add_parser(
        "status",
        help="Show checkpoint status across biome runs.",
    )
    status_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Root output directory for Phase 1 runs.")
    status_parser.set_defaults(func=_cmd_status)

    return parser


def main() -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        log.exception("CLI command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
