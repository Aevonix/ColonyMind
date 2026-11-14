"""Declared context durability per initiative type.

Durable contexts are snapshots. Volatile contexts can change while queued and
retain their source observation time in ``context_captured_at``. The host refresh
endpoint reports durability and the stored timestamp; it does not rebuild context
or enforce a freshness TTL.
"""
from __future__ import annotations

from typing import Dict

DURABLE = "durable"
VOLATILE = "volatile"

# Durability by InitiativeType value. Unlisted types default to durable.
CONTEXT_DURABILITY: Dict[str, str] = {
    # Existing types
    "follow_up": DURABLE,
    "relationship": DURABLE,
    "introduction": DURABLE,
    "health": VOLATILE,
    "scheduling": DURABLE,
    "coding": VOLATILE,
    "subsystem_health": VOLATILE,
    "data_quality": VOLATILE,
    "operational": DURABLE,
    "capability_gap": DURABLE,
    "knowledge_acquisition": DURABLE,
    "behavioral_correction": DURABLE,
    "agent_action": VOLATILE,
    # Autonomous work domains (v0.16.0)
    "commitment": DURABLE,
    "calendar": VOLATILE,
    "research": DURABLE,
    "task": DURABLE,
    "project": DURABLE,
    "system": VOLATILE,
}

def durability_for(type_value: str) -> str:
    """Declared durability for an initiative type ('durable'/'volatile')."""
    return CONTEXT_DURABILITY.get(type_value, DURABLE)
