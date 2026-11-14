"""The front pages hold to the code: what a Hermes user reads first says what the release does.

The front pages are the project README, the two package pages (``sidecar/README.md`` for ``colonymind`` and
``docs/HERMES-ADAPTER.md`` for ``colonymind-hermes``), the plugin READMEs, the memory provider's skill text, the docs
index, troubleshooting and the pages this release rewrote. Here the README's demo transcript is what
``colonymind demo`` prints, every command the pages show parses, every route they name is served, every tool they
name is one the plugins register, every router setting they name is the configuration's, the comparison with
Hermes' other memory options is by category, the package pages carry the one tagline and link only by absolute
URL, forgetting is what the memory provider's page says, and CI checks the links and runs the README's "Get
started" commands on a fresh machine (``test_front_pages_init.py`` runs its local embedder).
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
ADAPTER_PAGE = DOCS / "HERMES-ADAPTER.md"
SIDECAR_PAGE = ROOT / "sidecar" / "README.md"
SKILL = ROOT / "plugins" / "colonymind-memory" / "SKILL.md"
PLUGIN_READMES = tuple(sorted((ROOT / "plugins").glob("*/README.md")))
# The pages a Hermes user reads first, written for this release as a whole.
FRONT_PAGES = (README, SIDECAR_PAGE, ADAPTER_PAGE, *PLUGIN_READMES, SKILL, DOCS / "README.md",
               DOCS / "TROUBLESHOOTING.md", DOCS / "INSTALL.md", DOCS / "SOURCE-ANNOTATIONS.md")
# The pages whose every command is checked against the CLI (the long guides name commands with elisions).
COMMAND_PAGES = (README, SIDECAR_PAGE, *PLUGIN_READMES, DOCS / "README.md")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of the ``## heading`` section, up to the next second-level heading."""
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


def _code_spans(text: str) -> list[str]:
    """Every code span and every line of every code block."""
    blocks = re.findall(r"^ *```[^\n]*\n(.*?)^ *```", text, flags=re.MULTILINE | re.DOTALL)
    prose = re.sub(r"^ *```[^\n]*\n.*?^ *```", "", text, flags=re.MULTILINE | re.DOTALL)
    return re.findall(r"`([^`\n]+)`", prose) + [line for block in blocks for line in block.splitlines()]


def _tagline(text: str) -> str:
    return re.search(r"^\*\*(.+)\*\*$", text, flags=re.MULTILINE).group(1)


# -- the demo ---------------------------------------------------------------------------------------------------------

def _demo_block() -> list[str]:
    """The README's demo transcript: the first text block of "See it work"."""
    section = _section(_text(README), "See it work")
    return re.search(r"^```text\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL).group(1).splitlines()


def _demo_patterns() -> list[re.Pattern]:
    """One pattern per transcript line the demo prints: a line whose text starts with "..." is the page's own
    elision, and ``<name>`` stands for a value that differs per run (an id, a digest, a duration)."""
    patterns = []
    for line in _demo_block():
        if not line.strip() or line.strip().startswith("..."):
            continue
        patterns.append(re.compile(re.sub(r"<[a-z][a-z -]*>", r"\\S+", re.escape(line.rstrip()))))
    return patterns


def _typed_answer() -> str:
    """What the page shows the owner typing in reply to the ask: the demo is given exactly that."""
    typed = [line.split("Owner: ", 1)[1] for line in _demo_block() if re.match(r"\[session \d+\] Owner: ", line)]
    return typed[-1]


@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """One run of ``colonymind demo`` answered as the page shows (the demo's own test drives it the same way)."""
    from test_demo import PROMPT, DemoRun
    run = DemoRun(tmp_path_factory.mktemp("front-page-demo"))
    try:
        run.wait_for(PROMPT)
        run.type(_typed_answer())
        assert run.finish() == 0, run.text()
        return list(run.lines)
    finally:
        run.kill()


def test_the_readme_demo_is_what_the_demo_prints(demo_output):
    """Every line of the README's transcript is a line the demo prints, in the order it prints them."""
    patterns = _demo_patterns()
    assert len(patterns) >= 10
    lines = iter(line.rstrip() for line in demo_output)
    missing = [pattern.pattern for pattern in patterns if not any(pattern.fullmatch(line) for line in lines)]
    assert missing == [], "\n".join(demo_output)


def test_the_readme_demo_shows_the_agent_acknowledge_the_approved_ask(demo_output):
    """After the owner's yes the demo's agent acknowledges it (it is handed what the sidecar says the answer did, as
    Hermes is): the page shows that reply, the demo's own last one, so the loop visibly closes."""
    block = [line.rstrip() for line in _demo_block() if line.strip()]
    answered = max(index for index, line in enumerate(block) if re.match(r"\[session \d+\] Owner: ", line))
    session = block[answered].split("]", 1)[0] + "]"
    printed = [line.rstrip() for line in demo_output if line.startswith(f"{session} Agent: ")]
    assert printed and printed[-1] in block[answered + 1:], "\n".join(demo_output)


# -- commands and tools -----------------------------------------------------------------------------------------------

def _commands(text: str) -> list[str]:
    """Every ``colonymind ...`` command a page shows; ``[...]`` marks an optional part, shown here as given."""
    found = []
    for span in _code_spans(text):
        command = span.split("  #")[0].strip()
        if command.startswith("colonymind "):
            found.append(command.replace("[", "").replace("]", ""))
    return found


class _Parsed(Exception):
    """The command line parsed; nothing ran."""


def test_every_command_the_front_pages_show_parses(monkeypatch, capsys):
    """A command on a front page is one the installed CLI takes: its subcommands, options and choices. Parsing
    stops before the command runs."""
    from colonymind import cli
    parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        parse(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    refused = []
    for page in COMMAND_PAGES:
        commands = _commands(_text(page))
        for command in commands:
            argv = [re.sub(r"^<[^>]+>$", "X", word) for word in shlex.split(command)[1:]]
            monkeypatch.setattr("sys.argv", ["colonymind", *argv])
            try:
                cli.main()
            except _Parsed:
                continue
            except SystemExit:
                pass
            refused.append(f"{page.relative_to(ROOT)}: {command}")
    capsys.readouterr()
    assert refused == []


# Routes a page names on another server: the decision model Instinct's fast tier asks (``instinct.providers.local``).
OTHER_SERVERS = {("POST", "/v1/decide")}


def test_every_route_the_front_pages_name_is_one_the_sidecar_serves(monkeypatch):
    """A ``METHOD /path`` a page names is a route of the sidecar's app: a path parameter stands for any value, a
    query is left out, and ``a|b`` in the last segment names both routes."""
    from starlette.routing import Match
    monkeypatch.setenv("COLONYMIND_API_KEY", "test-secret-key")
    from colonymind.server import create_app
    app = create_app()

    def served(method: str, path: str) -> bool:
        scope = {"type": "http", "method": method, "path": path, "root_path": "", "query_string": b"",
                 "headers": []}
        return any(route.matches(scope)[0] is Match.FULL for route in app.routes)

    named = [(page, method, route) for page in FRONT_PAGES
             for method, route in re.findall(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`\s]*)", _text(page))]
    assert named
    unserved = []
    for page, method, route in named:
        path = route.split("?")[0]
        if (method, path) in OTHER_SERVERS:
            continue
        stem, _, last = path.rpartition("/")
        unserved += [f"{page.relative_to(ROOT)}: {method} {route}" for alternative in last.split("|")
                     if not served(method, re.sub(r"\{[^}]+\}|<[^>]+>", "x", f"{stem}/{alternative}"))]
    assert unserved == []


def _registered_tools() -> set[str]:
    """The tool names the Hermes plugins register, read from their schemas."""
    names = set()
    for path in (ROOT / "plugins").glob("*/*.py"):
        names |= set(re.findall(r"""(?:["']name["']:\s*|_TOOL\s*=\s*)["'](colonymind_[a-z_]+)["']""", _text(path)))
    return names


def test_the_front_pages_name_only_tools_the_plugins_register():
    """A tool a page names is one a Hermes model can call today; the package names are not tools."""
    tools = _registered_tools()
    assert {"colonymind_self", "colonymind_memory_search", "colonymind_resolve_commitment"} <= tools
    tomlfile = tomllib.loads(_text(ROOT / "pyproject.toml"))
    packages = set(tomlfile["tool"]["setuptools"]["packages"])
    named = {f"{page.relative_to(ROOT)}: {name}" for page in FRONT_PAGES for span in _code_spans(_text(page))
             for name in re.findall(r"\bcolonymind_[a-z_]+\b", span) if name not in tools | packages}
    assert named == set()


# Design records and plans name tools as they were planned or built then; every other page names only live ones.
RECORDS = (DOCS / "internal", DOCS / "proto-agi")


def test_no_guide_names_a_tool_the_plugins_no_longer_register():
    """1.10 made the Hermes adapter thin: the tools that read a source, coordinated a commitment's work or accepted
    a local draft are gone. A guide that names one tells a reader to call what no Hermes model can; a state file
    that shares the prefix (``colonymind_world_model.db``) is not a tool."""
    tools = _registered_tools()
    packages = set(tomllib.loads(_text(ROOT / "pyproject.toml"))["tool"]["setuptools"]["packages"])
    pages = [page for page in sorted(DOCS.rglob("*.md")) if not any(record in page.parents for record in RECORDS)]
    named = sorted({f"{page.relative_to(ROOT)}: {name}" for page in (*FRONT_PAGES, *pages)
                    for name in re.findall(r"\bcolonymind_[a-z_]+\b(?!\.db)", _text(page))
                    if name not in tools | packages})
    assert named == []


def _local_embedder_facts():
    """What a reader choosing the local embedder needs, as the code has it: the default model's download, the least
    memory a model is offered on, the transformers release its models need."""
    from colonymind.vector import local_embedder
    default = local_embedder.PINNED[local_embedder.CEILING]
    least = next(gb for gb in range(1, 65) if _chosen(local_embedder, gb))
    floor = next(oldest for name, oldest in default.requires if name == "transformers")
    return default.download_gb, least, floor.rsplit(".0", 1)[0] if floor.endswith(".0") else floor


def _chosen(local_embedder, gb):
    try:
        local_embedder.choose(0, gb)
        return True
    except local_embedder.LocalEmbedderError:
        return False


@pytest.mark.parametrize("page", [README, DOCS / "INSTALL.md", DOCS / "TROUBLESHOOTING.md"],
                         ids=lambda page: page.name)
def test_the_pages_that_offer_the_local_embedder_state_its_costs(page):
    """The README, the install guide and troubleshooting each say what the local embedder downloads, that a machine
    under the memory it needs is offered none (and how to name one anyway), that its models need a recent
    transformers, and that its files stay in the instance after a switch to an endpoint or an uninstall."""
    size, least, transformers = _local_embedder_facts()
    text = " ".join(_text(page).split())
    assert f"about {size:g} GB" in text
    assert f"under {least} GB" in text and "--embed-model" in text
    assert f"transformers {transformers} or later" in text
    assert "<instance>/models" in text and "uninstall" in text and "stay" in text


# The 1.11 answer to an ask, or a way to approve one from the console: neither exists any more.
OLD_ANSWERS = (r"\b(?:opens|starts?) with the answer and the code\b", r"(?<!colonymind mind )\byes <code>",
               r"`yes [A-Z0-9]{3}`",
               r"\bwith the code\b", r"\bpasses your answer on\b", r"with `colonymind mind yes")


def test_no_front_page_tells_the_owner_to_answer_with_a_code_or_approve_at_the_console():
    """The owner answers in their own chat, in their own words; the console only declines or shows an ask again."""
    found = [f"{page.relative_to(ROOT)}: {pattern}" for page in FRONT_PAGES for pattern in OLD_ANSWERS
             if re.search(pattern, " ".join(_text(page).split()))]
    assert found == []


# -- configuration -----------------------------------------------------------------------------------------------------

def _configure_example() -> dict:
    """The install guide's example ``colonymind.yaml``."""
    section = _section(_text(DOCS / "INSTALL.md"), "Configure")
    return yaml.safe_load(re.search(r"^```yaml\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL).group(1))


def test_the_router_settings_the_pages_name_are_the_configurations():
    """A ``router.<key>`` a page names, and every key of the install guide's example ``router``, is one the
    configuration has: no page tells the owner to set a key the sidecar would not read."""
    from colonymind.config import DEFAULTS
    keys = set(DEFAULTS["router"])
    unknown = {f"{page.relative_to(ROOT)}: router.{key}" for page in FRONT_PAGES
               for key in re.findall(r"\brouter\.([a-z][a-z_]*[a-z])\b", _text(page)) if key not in keys}
    assert unknown == set()
    assert set(_configure_example()["router"]) == keys


# -- the comparison ---------------------------------------------------------------------------------------------------

COLUMNS = ["", "Hermes built-in memory", "A memory-only provider", "ColonyMind"]
ROWS = ["Hermes slot", "Recall computed for the current message, before the reply", "What recall returns",
        "Local by default", "Tracks promises and deadlines", "Acts unprompted, within limits the owner sets",
        "Unprompted actions ask the owner first", "Forgetting follows lineage to derived data",
        "Measured against stock Hermes, paired"]
# What is true of a memory-only provider by its definition; anything else would describe one product.
CATEGORY_CELLS = {"the one external provider", "varies", "no", "no unprompted actions"}


def _comparison() -> list[list[str]]:
    section = _section(_text(README), "ColonyMind and other Hermes memory options")
    rows = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip().strip("|").split("|")] for row in rows if not set(row) <= set("|- ")]


def test_the_comparison_is_by_category_with_the_rows_of_the_plan():
    table = _comparison()
    assert table[0] == COLUMNS
    assert [row[0] for row in table[1:]] == ROWS
    assert {row[2] for row in table[1:]} <= CATEGORY_CELLS
    unsupported = [row[0] for row in table[1:] if row[3].startswith("yes") and "](" not in row[3]]
    assert unsupported == [], "a ColonyMind cell that claims something cites the doc that shows it"


# -- ready to publish -------------------------------------------------------------------------------------------------

def test_the_front_pages_carry_no_draft_leftovers():
    leftovers = [f"{page.relative_to(ROOT)}: {mark}" for page in FRONT_PAGES
                 for mark in ("<!--", "TODO(", "Review notes", "FIXME") if mark in _text(page)]
    assert leftovers == []


def test_the_readme_badges_state_facts_and_track_nothing():
    """Version, Python range, Hermes range, CI and licence: no download, star or trending counts, no pixel."""
    header = _text(README).split("## What it is", 1)[0]
    images = re.findall(r"""(?:!\[[^\]]*\]\(|src(?:set)?=["'])([^)"'\s]+)""", header)
    assert images
    assert [url for url in images if re.search(r"download|/d[mwt]/|stars|trending|hits|visitor|pixel|analytics",
                                              url, flags=re.IGNORECASE)] == []
    python = tomllib.loads(_text(ROOT / "sidecar" / "pyproject.toml"))["project"]["requires-python"]
    low, high = (int(value) for value in re.findall(r"3\.(\d+)", python))
    assert "%20%7C%20".join(f"3.{minor}" for minor in range(low, high)) in header


def test_the_install_comes_within_the_first_400_words():
    """The target of the presentation plan (its assumption, not a measured rule): prose words before the install
    section, code blocks, markup and the header's badges left out."""
    text = _text(README)
    before = text[text.index("## What it is"):text.index("## Get started")]
    before = re.sub(r"^```.*?^```", "", before, flags=re.MULTILINE | re.DOTALL)
    before = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", before)
    words = [word for word in re.split(r"\s+", before) if re.search(r"\w", word)]
    assert len(words) <= 400, len(words)


def test_the_package_summaries_carry_the_tagline():
    """One tagline everywhere a package is listed: the README, both package pages and both package summaries."""
    tagline = _tagline(_text(README))
    assert _tagline(_text(SIDECAR_PAGE)) == tagline and _tagline(_text(ADAPTER_PAGE)) == tagline
    for pyproject in (ROOT / "pyproject.toml", ROOT / "sidecar" / "pyproject.toml"):
        summary = tomllib.loads(_text(pyproject))["project"]["description"]
        assert tagline.rstrip(".") in summary, pyproject


def test_the_adapter_package_page_links_only_by_absolute_url():
    """``docs/HERMES-ADAPTER.md`` is the ``colonymind-hermes`` page on the package index, which renders no
    relative link."""
    assert tomllib.loads(_text(ROOT / "pyproject.toml"))["project"]["readme"] == "docs/HERMES-ADAPTER.md"
    targets = re.findall(r"\]\(([^)\s]+)", _text(ADAPTER_PAGE))
    assert targets
    assert [target for target in targets if not re.match(r"https?://", target)] == []


def test_the_skill_text_is_durable():
    """The provider's skill is read by the agent: instructions only, with no version, date, count or status."""
    body = _text(SKILL).split("---", 2)[2]
    assert not re.search(r"\d", body)
    assert not re.search(r"\b(?:version|release|currently|now ships|new in)\b", body, flags=re.IGNORECASE)


# -- corrections ------------------------------------------------------------------------------------------------------

def _annotation_intro() -> str:
    return _text(DOCS / "SOURCE-ANNOTATIONS.md").split("\n## ", 1)[0]


def test_the_corrections_page_says_who_may_annotate_as_the_credentials_decide():
    """The annotate route is on the owner's side of the API (``manage``), or a 1.9.0 credential's
    ``memory:write``; no native tool annotates."""
    from colonymind.api.credentials import route_operation
    from colonymind.api.legacy_scopes import required_scope
    route = ("POST", "/v1/host/memory/sources/annotations")
    intro = " ".join(_annotation_intro().split())
    assert f"`{route_operation(*route)}`" in intro
    assert f"`{required_scope(*route)}`" in intro


def test_the_corrections_example_is_a_request_the_route_takes():
    from colonymind.api.routers.host import SourceAnnotationRequest
    import json
    example = json.loads(re.search(r"^```json\n(.*?)^```", _annotation_intro(),
                                   flags=re.MULTILINE | re.DOTALL).group(1))
    assert set(example) == set(SourceAnnotationRequest.model_fields)


# -- CI ---------------------------------------------------------------------------------------------------------------

def _ci() -> dict:
    return yaml.safe_load(_text(ROOT / ".github" / "workflows" / "ci.yml"))


def _runs(job: dict) -> list[str]:
    return [step.get("run", "") for step in job.get("steps", [])]


def test_ci_checks_every_link_in_the_docs():
    assert any("python scripts/check_doc_links.py" in run for job in _ci()["jobs"].values() for run in _runs(job))


def test_ci_runs_the_get_started_commands_on_a_fresh_machine():
    """A job runs the README's commands as the page shows them, against the qualified distributions and stock
    Hermes."""
    jobs = [job for job in _ci()["jobs"].values()
            if any("bash scripts/ci_get_started.sh" in run for run in _runs(job))]
    assert len(jobs) == 1
    job = jobs[0]
    assert "artifacts" in job["needs"]
    step = next(step for step in job["steps"] if "ci_get_started.sh" in step.get("run", ""))
    assert {"HERMES_SRC", "DIST_DIR"} <= set(step["env"])


def test_the_get_started_commands_are_the_readmes_own():
    """The CI script takes its commands from the page, so the job checks what a reader copies."""
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "get_started_commands.py"), str(README)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    section = _section(_text(README), "Get started")
    block = re.search(r"^```bash\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL).group(1)
    assert done.stdout.splitlines() == [line for line in block.splitlines() if line.strip()]
    assert done.stdout.splitlines()[-1] == "hermes gateway restart"


@pytest.mark.asyncio
async def test_forgetting_is_what_the_memory_provider_page_says(tmp_path, monkeypatch):
    """A removal from Hermes' memory files reaches ColonyMind as the removed text with the current session (the
    provider's hook), and forgets a source only when that text is a whole message of exactly one source of that
    session: a fact told in an earlier session, part of a message, or a message a compression checkpoint also holds
    stays recallable. Forgetting by ``source_ids`` erases it, and the answer is the receipt."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.routers import host
    from colonymind.turns import get_turn_idempotency_ledger
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    ledger = get_turn_idempotency_ledger(tmp_path)
    from test_curated_memory import PERSON
    told, today = "My sister Robin moved to Lisbon last spring.", "I start at the bakery on Monday."
    ledger.record_source("told", contact_id=PERSON, session_id="earlier", messages=[{"role": "user", "content": told}])
    for turn in ("today", "today-checkpoint"):
        ledger.record_source(turn, contact_id=PERSON, session_id="now", messages=[{"role": "user", "content": today}])
    app = FastAPI()
    from test_curated_memory import app_for, PERSON
    from onekey import AUTH
    app = app_for()
    FORGET = '/v1/host/memory/curated/sources/forget'

    def recallable(word: str) -> set[str]:
        from colonymind.memory import curated
        return {row['turn_id'] for row in curated.current_candidates(ledger,
            ledger.search_sources(word, contact_id=PERSON, session_id='now'), contact_id=PERSON)}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers=AUTH) as client:
        async def removed(text: str, session: str) -> int:
            """What the provider sends when the agent's memory tool removes *text*."""
            answer = await client.post(FORGET, json={"person_id": PERSON, "session_id": session, "old_text": text})
            return answer.status_code

        assert await removed(told, "now") == 422                        # told in an earlier session
        assert await removed("Robin moved to Lisbon", "earlier") == 422   # part of a message
        assert await removed(today, "now") == 409                       # the message and its checkpoint copy
        assert recallable("Lisbon") == {"told"} and recallable("bakery")
        receipt = await client.post(FORGET, json={"person_id": PERSON, "source_ids": ["told"]})
    assert receipt.status_code == 200 and receipt.json()["source_erased"] is True
    assert recallable("Lisbon") == set()
