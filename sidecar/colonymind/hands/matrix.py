"""Static routing: each kind of work goes to one role. No effort is learned."""


ROLES = {"coding": "coding", "deep": "deep", "research": "deep", "ops": "coding",
         "media": "media", "plan": "deep", "synthesize": "deep", "verify": "verify",
         "integrate": "coding"}


def route(kind, config):
    """Return the configured role at effort 2, regardless of an effort hint."""
    cell = config.get("matrix", {}).get(kind, {})
    role = cell.get("2", cell.get(2, ROLES[kind])) if isinstance(cell, dict) else cell
    if isinstance(role, dict):
        role = role.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("invalid role matrix")
    return role, 2
