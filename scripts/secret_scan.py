"""Scan the tracked files for values that look like secrets (the capture detector, docs/SECRETS.md).

Usage: ``python scripts/secret_scan.py [--root DIR] [PATH ...]``. With no paths it scans ``git ls-files``. A finding
prints ``path:line: rule (label)`` and never the value; any finding exits 1.

The scan runs the detector's rules that name a credential by its form (vendor tokens, key blocks, JWTs, webhook
and credential-bearing URLs, bot tokens, card numbers). The rules that read a label or an assignment are left to
conversation capture: in source code they match type annotations, keyword arguments and test doubles.

Synthetic fixtures are allowed where they are marked: a file whose first ten lines carry the marker
``secret-scan: synthetic fixtures`` is skipped, and so is a single line carrying ``secret-scan: synthetic``. A
real value is never marked; it is removed and rotated.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE_MARKER = "secret-scan: synthetic fixtures"
LINE_MARKER = "secret-scan: synthetic"
MAX_BYTES = 2 * 1024 * 1024
CI_RULES = ("prefix", "jwt", "private_key", "webhook", "url_userinfo", "url_token", "telegram", "card")
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".svg", ".mp3", ".mp4", ".wav", ".ogg",
                   ".zip", ".gz", ".tar", ".whl", ".db", ".sqlite", ".sqlite3", ".woff", ".woff2", ".ttf"}


def detector():
    """The sidecar's capture detector, loaded from this repository's file (it needs nothing but the standard
    library), whatever tree is scanned."""
    path = ROOT / "sidecar/colonymind/secrets/detect.py"
    spec = importlib.util.spec_from_file_location("colonymind_secret_scan_detect", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tracked(root: Path) -> list[str]:
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


def scan_text(text: str, find) -> list[tuple[int, str, str]]:
    """``(line, rule, label)`` of every finding outside a marked line; none in a marked file."""
    if FILE_MARKER in "\n".join(text.splitlines()[:10]):
        return []
    findings = []
    lines = text.splitlines()
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    for hit in find(text, rules=CI_RULES):
        number = next(index for index in range(len(starts) - 1, -1, -1) if starts[index] <= hit.start)
        if LINE_MARKER in lines[number]:
            continue
        findings.append((number + 1, hit.rule, hit.label))
    return findings


def scan(root: Path, paths: list[str]) -> list[tuple[str, int, str, str]]:
    find = detector().find_secrets
    results = []
    for name in paths:
        path = root / name
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        results += [(name, line, rule, label) for line, rule, label in scan_text(text, find)]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    findings = scan(root, args.paths or tracked(root))
    for name, line, rule, label in findings:
        print(f"{name}:{line}: {rule} ({label})")
    print(f"secret scan: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
