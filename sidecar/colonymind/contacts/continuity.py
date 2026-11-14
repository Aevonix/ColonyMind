"""Small, handle-free views of the existing communication ledger."""
from datetime import timedelta

from colonymind.identity import get_owner_name, get_persona_name
from colonymind.redact import redact_sensitive_text
from colonymind.util import temporal


_DISPLAY = {
    "whatsapp": "WhatsApp", "companion": "Companion", "telegram": "Telegram",
    "signal": "Signal", "sms": "SMS", "rcs": "RCS", "email": "Email",
    "discord": "Discord", "slack": "Slack", "imessage": "iMessage",
    "matrix": "Matrix", "cli": "CLI", "tui": "TUI", "voice": "Voice",
}
_AUTOMATED = {"system", "automated", "cron", "worker", "internal", "api", "unknown", ""}
PLATFORMS = frozenset(_DISPLAY)          # the platforms a channel names that are known to be no handle


def platform(channel: str) -> str:
    return str(channel or "").split(":", 1)[0].strip().lower()


def platform_name(channel: str) -> str:
    # An unrecognized value might itself be a handle. Never echo it.
    return _DISPLAY.get(platform(channel), "Other")


def recorded_channels(per_channel: dict) -> str:
    latest = {}
    for channel, row in per_channel.items():
        name = platform_name(channel)
        date = str(row["ts"])[:10]
        latest[name] = max(latest.get(name, ""), date)
    return ", ".join(f"{name} {date}" for name, date in sorted(latest.items()))


def recent_elsewhere(rows: list[dict], current_channel: str, zone: str, *, now=None, handles=()) -> str:
    """Newest other platform, at most six exchanges, oldest first, within 72 hours."""
    now = now or temporal.now_utc()
    current = platform(current_channel)
    if not current:
        return ""  # Without a turn platform we cannot establish 'elsewhere'.
    eligible = []
    handles = (set(handles) | {str(row.get("channel", "")).partition(":")[2] for row in rows}) - {""}
    for row in rows:
        at = temporal.parse_iso(row.get("ts"))
        other = platform(row.get("channel"))
        if (at is None or not now - timedelta(hours=72) <= at <= now
                or other == current or other in _AUTOMATED
                or row.get("direction") not in {"in", "out"} or not row.get("summary")):
            continue
        eligible.append((at, other, row))
    # Stable ordering retains the ledger's newest-first ordering for equal stamps.
    eligible.sort(key=lambda item: item[0], reverse=True)
    if not eligible:
        return ""
    chosen = eligible[0][1]
    exchanges = [item for item in eligible if item[1] == chosen][:6]
    lines = []
    for at, other, row in reversed(exchanges):
        summary = " ".join(str(row["summary"]).split())
        for handle in sorted(handles, key=len, reverse=True):
            summary = summary.replace(handle, "[handle]")
        summary = redact_sensitive_text(summary, contact_handles=True)
        summary = summary if len(summary) <= 280 else summary[:279] + "…"
        local = temporal.to_zone(at, zone)
        clock = local.strftime("%I:%M %p").lstrip("0")
        speaker = get_owner_name() if row["direction"] == "in" else get_persona_name()
        lines.append(f"{platform_name(other)}, {local.strftime('%a %b %d')} {clock}: {speaker}: {summary}")
    return "\n".join(lines)
