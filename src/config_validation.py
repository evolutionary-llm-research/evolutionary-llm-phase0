"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML configuration.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_config(config: dict[str, Any]) -> None:
    """Validate project configuration against phase constraints.

    Parameters
    ----------
    config : dict[str, Any]
        Parsed configuration object.

    Raises
    ------
    ValueError
        Raised when a required key or phase rule is violated.
    """
    project = config.get("project", {})
    phase = project.get("phase")
    if phase not in {0, 1, 2, 3, 4}:
        raise ValueError("project.phase must be an integer in [0, 1, 2, 3, 4].")

    constraints = config.get("constraints", {})
    if constraints.get("pass_content_type_tags_to_model") is not False:
        raise ValueError("constraints.pass_content_type_tags_to_model must be false.")

    biomes = config.get("biomes", {})
    for biome_name, biome in biomes.items():
        total = float(biome.get("food", 0.0)) + float(biome.get("toxin", 0.0)) + float(biome.get("noise", 0.0))
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Biome '{biome_name}' ratios must sum to 1.0, got {total}.")

    mechanics = config.get("mechanics", {})
    phase_expected = {
        0: (False, False, False),
        1: (False, False, False),
        2: (True, False, False),
        3: (True, True, True),
        4: (True, True, True),
    }
    expected_repro, expected_hgt, expected_cannibalism = phase_expected[phase]

    got_repro = bool(mechanics.get("reproduction_enabled", False))
    got_hgt = bool(mechanics.get("hgt_enabled", False))
    got_cannibalism = bool(mechanics.get("cannibalism_enabled", False))

    if (got_repro, got_hgt, got_cannibalism) != (expected_repro, expected_hgt, expected_cannibalism):
        raise ValueError(
            "mechanics section violates current phase isolation rules. "
            f"Expected {(expected_repro, expected_hgt, expected_cannibalism)}, "
            f"got {(got_repro, got_hgt, got_cannibalism)}."
        )
