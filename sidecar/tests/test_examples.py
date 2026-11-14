"""The repository's examples run on the current interfaces, and the docs name only examples that exist."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Interfaces the 1.9 line had and later releases removed: the per-principal key file and the adapter's
# transport-scope table.
RETIRED = ("api-keyring.json", "_TRANSPORT_SCOPES")


def test_no_example_reads_a_retired_interface():
    for path in sorted((REPO / "examples").glob("**/*.py")):
        text = path.read_text(encoding="utf-8")
        assert not [name for name in RETIRED if name in text], path


def test_the_docs_name_only_examples_that_exist():
    documents = [REPO / "README.md", REPO / "sidecar" / "README.md", *sorted((REPO / "docs").glob("**/*.md"))]
    for document in documents:
        for name in re.findall(r"\bexamples/[\w./-]+\.py\b", document.read_text(encoding="utf-8")):
            assert (REPO / name).is_file(), f"{document.relative_to(REPO)} names {name}"
