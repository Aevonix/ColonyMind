"""The documentation's shape: user guides at the top of ``docs/``, build notes and historical studies under
``docs/internal/``, and an index that reaches every page."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from test_turn_source_evidence import envelope, source_app  # noqa: F401  (source_app is a fixture)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
INTERNAL = DOCS / "internal"
TROUBLESHOOTING = DOCS / "TROUBLESHOOTING.md"
LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)")

# Build notes, slice records and the older qualification studies: not guides for running ColonyMind.
MOVED = (
    "AGENT-BENCHMARK.md", "AGENT-BENCHMARK-AUTHORITY.md", "AGENT-BENCHMARK-NATIVE-MEMORY.md",
    "EXPECTED-REPLY-FORECASTS.md", "HARNESS_INTEGRATION.md", "SITUATION-EXPECTATIONS-P6.md",
    "TEMPORAL-FORECASTS.md", "TOM-SOURCE-LINEAGE.md", "WORKING-PERSPECTIVE.md", "phase-c-durable-scheduler.md",
)


def _linked(page: Path) -> set[Path]:
    """The files a page links to, resolved."""
    return {(page.parent / target).resolve() for target in LINK.findall(page.read_text(encoding="utf-8"))
            if "://" not in target and not target.startswith("mailto:")}


def test_build_notes_and_historical_studies_live_under_docs_internal():
    qualification = sorted(path.name for path in INTERNAL.glob("qualification-*.md"))
    assert len(qualification) == 11, qualification
    assert not list(DOCS.glob("qualification-*.md"))
    for name in MOVED:
        assert not (DOCS / name).exists(), name
        assert (INTERNAL / name).is_file(), name


def test_the_internal_index_reaches_every_internal_page():
    index = INTERNAL / "README.md"
    linked = _linked(index)
    missing = [path.name for path in sorted(INTERNAL.glob("*.md")) if path != index and path.resolve() not in linked]
    assert missing == []


INDEX_SECTIONS = ["Start here", "Using it", "Memory", "Media", "Security and privacy", "Models and execution",
                  "Measurement", "Design and history"]


def test_the_docs_index_is_organized_by_task():
    text = (DOCS / "README.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (.+)$", text, flags=re.MULTILINE) == INDEX_SECTIONS


def test_every_page_under_docs_is_reachable_from_the_index():
    """A page no index path leads to is a page nobody finds."""
    reached, queue = set(), [(DOCS / "README.md").resolve()]
    while queue:
        page = queue.pop()
        if page in reached:
            continue
        reached.add(page)
        queue += [target for target in _linked(page)
                  if target.suffix == ".md" and target.is_file() and DOCS.resolve() in target.parents]
    missing = sorted(str(path.relative_to(DOCS)) for path in DOCS.rglob("*.md") if path.resolve() not in reached)
    assert missing == []


def test_the_readme_links_the_docs_index_and_troubleshooting():
    linked = _linked(ROOT / "README.md")
    assert (DOCS / "README.md").resolve() in linked
    assert TROUBLESHOOTING.resolve() in linked


SETTING = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*[A-Z0-9])=", re.MULTILINE)
CODE_FILES = ("*.py", "*.yaml", "*.sh")


def test_no_example_configuration_advertises_a_setting_nothing_reads():
    """An example environment file is read as the list of what can be set: a name no code reads misleads."""
    read = {word for base in (ROOT / "sidecar/colonymind", ROOT / "plugins") for pattern in CODE_FILES
            for path in base.rglob(pattern)
            for word in re.findall(r"\b[A-Z][A-Z0-9_]+\b", path.read_text(encoding="utf-8", errors="replace"))}
    unread = []
    for example in sorted(set(ROOT.glob("**/*.env.example"))):
        if any(part.startswith(".") for part in example.relative_to(ROOT).parts[:-1]):
            continue
        unread += [f"{example.relative_to(ROOT)}: {name}"
                   for name in SETTING.findall(example.read_text(encoding="utf-8")) if name not in read]
    assert unread == []


FRONT_PAGES = ("README.md", "CONTRIBUTING.md", "docs/README.md", "docs/TROUBLESHOOTING.md")
PATH_SPAN = re.compile(r"`(\.?[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]*)*)`")


def test_the_front_pages_name_only_paths_the_repository_has():
    """A path in code on a front page that starts at the repository root (a top-level name, or a dotfile) exists."""
    missing = []
    for name in FRONT_PAGES:
        text = re.sub(r"^```.*?^```", "", (ROOT / name).read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)
        for span in PATH_SPAN.findall(text):
            first = span.split("/")[0]
            if "/" not in span and "." not in span.lstrip("."):
                continue
            if (ROOT / first).exists() or span.startswith("."):
                if not (ROOT / span).exists():
                    missing.append(f"{name}: {span}")
    assert missing == []


SYMPTOMS = (
    "No memory is recalled",
    "The owner's turns are not recorded",
    "Recall matches keywords only",
    "Recall stopped abstaining after a reranker change",
    "Gateway prefetch timeouts",
    "The mind does not act",
    "An answered ask is not taken",
)


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    return {parts[index].strip(): parts[index + 1] for index in range(1, len(parts), 2)}


def test_troubleshooting_gives_each_symptom_a_check_and_a_fix():
    sections = _sections(TROUBLESHOOTING.read_text(encoding="utf-8"))
    assert [name for name in SYMPTOMS if name not in sections] == []
    for name, body in sections.items():
        assert "**Check.**" in body and "**Fix.**" in body, name


def _commands(text: str) -> list[str]:
    """Every ``colonymind ...`` command the page shows, in code spans and code blocks."""
    found = [span for span in re.findall(r"`([^`\n]+)`", text) if span.startswith("colonymind ")]
    for block in re.findall(r"^```[^\n]*\n(.*?)^```", text, flags=re.MULTILINE | re.DOTALL):
        found += [line.split("  #")[0].strip() for line in block.splitlines() if line.startswith("colonymind ")]
    return found


class _Parsed(Exception):
    """The command line parsed; nothing ran."""


def test_every_colonymind_command_on_the_troubleshooting_page_exists(monkeypatch, capsys):
    """Each command parses with the installed CLI: its subcommands, options and choices. Parsing stops before the
    command runs."""
    from colonymind import cli
    parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        parse(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    commands = _commands(TROUBLESHOOTING.read_text(encoding="utf-8"))
    assert commands
    refused = []
    for command in commands:
        argv = [re.sub(r"^<[^>]+>$", "X", word) for word in shlex.split(command)[1:]]
        monkeypatch.setattr("sys.argv", ["colonymind", *argv])
        try:
            cli.main()
        except _Parsed:
            continue
        except SystemExit:
            pass
        refused.append(command)
    capsys.readouterr()
    assert refused == []


def _curl_calls(text: str) -> list[tuple[str, dict]]:
    """``(route, JSON body)`` of every ``curl ... -d '<json>' <url>`` example in the page's code blocks."""
    calls = []
    for block in re.findall(r"^ *```[^\n]*\n(.*?)^ *```", text, flags=re.MULTILINE | re.DOTALL):
        joined = block.replace("\\\n", " ")
        for match in re.finditer(r"\bcurl\b.*?-d '(.*?)'.*?https?://[^/\s]+(/\S*)", joined, flags=re.DOTALL):
            calls.append((match.group(2), json.loads(match.group(1))))
    return calls


def test_every_route_the_troubleshooting_page_calls_takes_the_fields_it_sends(monkeypatch):
    """Where no command does the job the page shows the route: it exists, takes every field the example sends,
    and is sent every field it requires."""
    monkeypatch.setenv("COLONYMIND_API_KEY", "test-secret-key")
    from colonymind.server import create_app
    spec = create_app().openapi()
    calls = _curl_calls(TROUBLESHOOTING.read_text(encoding="utf-8"))
    assert calls
    problems = []
    for route, body in calls:
        operation = spec["paths"].get(route, {}).get("post")
        if operation is None:
            problems.append(f"{route}: no such route")
            continue
        name = operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
        schema = spec["components"]["schemas"][name]
        problems += [f"{route}: takes no {field}" for field in body if field not in schema["properties"]]
        problems += [f"{route}: needs {field}" for field in schema.get("required", []) if field not in body]
    assert problems == []


def test_a_handle_on_the_wrong_record_is_corrected_alone_and_a_merge_is_for_the_owners_own_records():
    """One of the owner's handles on another person's record is fixed by moving that handle, never by folding the
    record: a merge gives the owner every handle and the whole history of the other record. The page keeps the
    merge for a record that is the owner's too, checked first."""
    section = _sections(TROUBLESHOOTING.read_text(encoding="utf-8"))["The owner's turns are not recorded"]
    fix = section.split("**Fix.**", 1)[1]
    assert "/v1/host/social/contacts/correct-identity" in fix
    merges = [item for item in re.split(r"\n(?=- )", fix) if "people merge" in item]
    assert merges
    assert all("every handle" in item and "people inspect" in item for item in merges)


async def test_the_correction_moves_one_handle_and_a_merge_moves_them_all(tmp_path):
    """What the page says of the two: the correction leaves the other person's own handle with them; the merge makes
    it the owner's, so their messages from it would be recorded as the owner's."""
    from colonymind.contacts.config import ContactsConfig
    from colonymind.contacts.store import SQLiteContactStore
    held = {}
    for way in ("correct", "merge"):
        store = SQLiteContactStore(ContactsConfig(sqlite_path=str(tmp_path / f"{way}.db")))
        await store.connect()
        try:
            owner, sam = await store.create(display_name="Owner"), await store.create(display_name="Sam")
            await store.add_handle(sam.contact_id, "telegram", "1001", source="auto:sender")  # the owner's
            await store.add_handle(sam.contact_id, "sms", "+15550000123", source="auto:sender")  # Sam's own
            if way == "correct":
                await store.correct_handle_identity(
                    operation_id="handle-fix-1", performed_by=owner.contact_id, gateway="telegram", address="1001",
                    expected_contact_id=sam.contact_id, contact_id=owner.contact_id,
                    evidence_refs=["this handle is mine"])
            else:
                await store.merge(owner.contact_id, sam.contact_id, performed_by=owner.contact_id)
            names = {owner.contact_id: "owner", sam.contact_id: "sam"}
            held[way] = {gateway: names[(await store.resolve_messaging_handle(gateway, address)).contact_id]
                         for gateway, address in (("telegram", "1001"), ("sms", "+15550000123"))}
        finally:
            await store.close()
    assert held["correct"] == {"telegram": "owner", "sms": "sam"}
    assert held["merge"] == {"telegram": "owner", "sms": "owner"}


def test_an_empty_search_is_not_read_as_a_turn_that_was_not_recorded():
    """The search ranks as recall does, so finding nothing leaves capture open: the page settles it with the
    unranked read before it sends anyone to the capture or credential fixes."""
    section = _sections(TROUBLESHOOTING.read_text(encoding="utf-8"))["No memory is recalled"]
    check = section.split("**Check.**", 1)[1].split("**Fix.**", 1)[0]
    searches = [item for item in re.split(r"\n(?=- )", check) if "colonymind_memory_search" in item]
    assert searches
    assert not [item for item in searches if re.search(r"not found: it was not recorded", item, flags=re.IGNORECASE)]
    assert "/v1/host/memory/recent" in check


class _SameScore:
    """A reranker that gives every candidate one score."""

    def __init__(self, score: float):
        self.score = score

    def calibration_metadata(self) -> dict:
        return {"model": "neutral-fixture", "format_version": "v1"}

    async def rerank(self, query, documents, top_k):
        return [{"index": index, "score": self.score} for index in range(len(documents))]


@pytest.mark.parametrize("score", [0.99, 0.01])
async def test_the_unranked_read_finds_a_recorded_fact_the_search_leaves_out(source_app, monkeypatch, score):  # noqa: F811
    """Under a calibrated cutoff a recorded fact that scores below it is missing from the search, as from recall;
    the unranked read lists it whatever it scores."""
    from httpx import ASGITransport, AsyncClient
    from onekey import KEY
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import host
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    reranker = _SameScore(score)
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK", "on")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_MIN_SCORE", ".8")
    monkeypatch.setenv("COLONYMIND_RECALL_RERANK_CALIBRATION",
                       calibration_fingerprint(provider_calibration_metadata(reranker)))
    monkeypatch.setattr(host, "_reranker", reranker)
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    turn = envelope("told-once")
    turn["context"]["metadata"] = {"occurred_at": "2026-01-02T03:04:05+00:00"}
    async with AsyncClient(transport=ASGITransport(app=source_app), base_url="http://test",
                           headers={"Authorization": "Bearer " + KEY}) as client:
        assert (await client.put("/v2/host/turns/told-once", json=turn)).status_code == 201
        search = await client.post("/v1/host/memory/search", json={
            "identity": {"host_id": "hermes"}, "person_id": "contact-a", "session_id": "session-a",
            "query": "hydrofoil departure"})
        recent = await client.post("/v1/host/memory/recent", json={
            "identity": {"host_id": "hermes"}, "person_id": "contact-a", "session_id": "another-session",
            "platform": "test"})
    assert search.status_code == 200 and recent.status_code == 200, search.text + recent.text
    assert search.json()["count"] == (1 if score > 0.8 else 0)
    assert [entry["source_id"] for entry in recent.json()["entries"]] == ["told-once"]


# The sidecar's package page (``sidecar/README.md``, the ``readme`` of ``sidecar/pyproject.toml``) states the same
# facts as the packaging and the project README. A package index renders no relative link.
PACKAGE_PAGE = ROOT / "sidecar/README.md"


def _project() -> dict:
    return tomllib.loads((ROOT / "sidecar/pyproject.toml").read_text(encoding="utf-8"))["project"]


def _python_range(spec: str) -> str:
    """``>=3.11,<3.14`` as ``3.11 to 3.13``."""
    low = re.search(r">=\s*3\.(\d+)", spec)
    high = re.search(r"<\s*3\.(\d+)", spec)
    return f"3.{low.group(1)} to 3.{int(high.group(1)) - 1}"


def test_the_package_page_is_the_sidecar_readme():
    assert _project()["readme"] == "README.md"


def test_the_package_page_and_the_readme_state_the_python_range_the_package_accepts():
    stated = f"Python {_python_range(_project()['requires-python'])}"
    assert stated in PACKAGE_PAGE.read_text(encoding="utf-8")
    assert stated in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_package_page_never_calls_a_required_dependency_optional():
    required = " ".join(_project()["dependencies"]).lower()
    page = PACKAGE_PAGE.read_text(encoding="utf-8")
    sentences = re.split(r"(?<=[.;:])\s+", " ".join(page.split()))
    if "lancedb" in required:
        assert "LanceDB" in page
        assert [sentence for sentence in sentences if "vector" in sentence.lower()
                and "optional" in sentence.lower() and "extra" not in sentence.lower()] == []


def _tagline(text: str) -> str:
    return re.search(r"^\*\*(.+)\*\*$", text, flags=re.MULTILINE).group(1)


def _first_bash_block(text: str, after: str = "") -> str:
    start = text.index(after) if after else 0
    return re.search(r"^```bash\n(.*?)^```", text[start:], flags=re.MULTILINE | re.DOTALL).group(1)


def test_the_package_page_carries_the_tagline_and_the_install_commands_of_the_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = PACKAGE_PAGE.read_text(encoding="utf-8")
    assert _tagline(page) == _tagline(readme)
    assert _first_bash_block(page) == _first_bash_block(readme, "## Get started")


def test_the_package_page_links_only_by_absolute_url():
    targets = re.findall(r"\]\(([^)\s]+)", PACKAGE_PAGE.read_text(encoding="utf-8"))
    assert targets
    assert [target for target in targets if not re.match(r"https?://", target)] == []
