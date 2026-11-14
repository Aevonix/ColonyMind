"""Turn ingress checks shared with the thin gateway adapter."""
import sys


def owner_started_turn(session_id, turn_id, info, inbound):
    """Require this turn's sender and ingress evidence, failing closed on bridge errors."""
    if (not session_id or not turn_id or info is None or turn_id != getattr(info, "turn_id", "")
            or not getattr(info, "sender_id", "") or getattr(info, "parent_session_id", "")
            or info.platform in {"", "cli", "tui", "owner", "api", "internal", "system", "worker",
                                 "cron", "cron-delivery", "subagent", "notification", "relay"}):
        return False
    bridge = sys.modules.get("colonymind_hands_front")
    try:
        if bridge is None:
            return inbound(info) is True
        return bridge.owner_started_turn(session_id, turn_id) is True
    except Exception:
        return False
