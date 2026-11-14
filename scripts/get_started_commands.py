"""Print the commands of the README's "Get started" block, one per line, as a reader copies them.

Usage: ``python scripts/get_started_commands.py [README]``. The block is the first ``bash`` code block of the
``## Get started`` section; a README without one exits 1. CI's fresh-machine job (``scripts/ci_get_started.sh``)
runs exactly these lines, so what it checks is what the page shows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(text: str) -> list[str]:
    """The non-empty, non-comment lines of the first ``bash`` block under ``## Get started``."""
    section = re.search(r"^## Get started\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if section is None:
        raise ValueError('no "## Get started" section')
    block = re.search(r"^```bash\n(.*?)^```", section.group(1), flags=re.MULTILINE | re.DOTALL)
    if block is None:
        raise ValueError('no bash block in "Get started"')
    found = [line.strip() for line in block.group(1).splitlines()
             if line.strip() and not line.strip().startswith("#")]
    if not found:
        raise ValueError('the "Get started" block is empty')
    return found


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else ROOT / "README.md"
    try:
        found = commands(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    print("\n".join(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
