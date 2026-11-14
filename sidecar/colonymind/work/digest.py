"""The digest caller supplies one line from the same read-only work summary."""
from pathlib import Path
from .store import read_summary


def summary_line(state_dir, *, contact_id, since, until):
    summary = read_summary(Path(state_dir) / 'colonymind-work.db', contact_id=contact_id, since=since, until=until)
    if not summary['total']:
        return None
    counts, overlays = summary['counts'], summary['overlays']
    return (f"Work for you: {counts['proven']} proven, {counts['partial']} partial, "
            f"{counts['unproven']} unproven, {counts['failed']} failed; "
            f"{overlays['confirmed']} owner-confirmed, {overlays['corrected']} owner-corrected.")
