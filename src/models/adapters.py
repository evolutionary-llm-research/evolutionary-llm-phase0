"""LoRA adapter metadata structures and naming utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AdapterMetadata:
    """Container for LoRA adapter metadata required by the project."""

    generation: int
    parent_id: str
    biome: str
    archetype: str
    fitness_score: float
    creation_timestamp: str


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_agent_id(archetype: str, generation: int, index: int) -> str:
    """Build agent ID using project naming convention.

    Example: ego_gen02_007
    """
    return f"{archetype.lower()}_gen{generation:02d}_{index:03d}"


def adapter_filename(agent_id: str) -> str:
    """Build adapter filename using project naming convention."""
    return f"adapter_{agent_id}.safetensors"
