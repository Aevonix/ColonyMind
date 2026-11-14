"""ColonyMind Vector Store — query and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorResult:
    """A single result from vector search."""

    id: str
    score: float  # cosine similarity in [0, 1]
    text: str  # original embedded text
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorItem:
    """An item to insert into the vector store."""

    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
