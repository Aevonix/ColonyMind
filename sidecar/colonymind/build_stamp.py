"""The build stamp: the commit this package was built from.

``scripts/write_build_stamp.py`` writes ``build_stamp.json`` beside this module; the file is never tracked. It holds
the full commit id, the branch name (empty when the tree was detached) and whether the tree had uncommitted
changes. Nothing reads the file when ``colonymind`` is imported: :func:`read` does, each time it is called.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

#: The one file a stamp is read from.
PATH = Path(__file__).with_name("build_stamp.json")

_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def read() -> Optional[dict]:
    """The stamp as ``{"commit": str, "branch": str, "dirty": bool}``, or ``None`` when there is none to show.

    ``None`` stands for a file that is absent, cannot be read, is not JSON, does not hold exactly these three
    facts, or holds a value of another type, a commit that is not a full id, or a branch name that has more than
    255 characters or one that does not print. Never raises.
    """
    try:
        stamp = json.loads(PATH.read_bytes())
    except Exception:  # noqa: BLE001 - absent, unreadable or damaged all read as no stamp
        return None
    if not isinstance(stamp, dict) or set(stamp) != {"commit", "branch", "dirty"}:
        return None
    commit, branch, dirty = stamp["commit"], stamp["branch"], stamp["dirty"]
    if not (isinstance(commit, str) and isinstance(branch, str) and isinstance(dirty, bool)):
        return None
    if not _COMMIT.fullmatch(commit):
        return None
    if not branch.isprintable() or len(branch) > 255:  # the script writes no such name; an empty one: detached
        return None
    return stamp
