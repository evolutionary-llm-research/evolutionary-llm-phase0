import pytest

from src.config_validation import validate_config


def _base_config() -> dict:
    return {
        "project": {"name": "evolutionary-llm-research", "phase": 0, "seed": 42},
        "constraints": {
            "pass_content_type_tags_to_model": False,
            "phase_mechanics_isolated": True,
        },
        "biomes": {
            "savanna": {"food": 0.6, "toxin": 0.2, "noise": 0.2, "k_max": 30},
        },
        "mechanics": {
            "reproduction_enabled": False,
            "hgt_enabled": False,
            "cannibalism_enabled": False,
        },
    }


def test_validate_config_phase0_ok() -> None:
    validate_config(_base_config())


def test_validate_config_phase0_rejects_reproduction() -> None:
    config = _base_config()
    config["mechanics"]["reproduction_enabled"] = True
    with pytest.raises(ValueError):
        validate_config(config)
