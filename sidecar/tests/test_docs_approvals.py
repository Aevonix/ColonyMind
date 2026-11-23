"""The approval docs say what the answer path does, as the code has it (docs/MIND.md, Asks).

Asks are answered from the owner's own message in their chat: bound to one ask by structure, read by fixed tables,
never by a code and never by a credential alone. The docs that describe it (``MIND.md``, ``OWNER-GATE.md``,
``CREDENTIALS.md``, ``HERMES-ADAPTER.md``, ``DECISIONS.md``, ``KNOWN-GAPS.md``) are checked against the code here:
every section the code cites exists, no page tells the owner to answer with a code, every route, command, hook,
refusal and setting of the answer path is named with the value the code has, and what the asks report counts but
nothing records yet is a known gap.
"""

from __future__ import annotations

import argparse
import re
import shlex
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from colonymind import doctor
from colonymind.mind import answers, asks_report, digest, gate, outbox, tick
from colonymind.mind.outbox import AskSettings
from test_ask_answers_binding import CHAT, PLATFORM, asked, deliver, evidence, fx, sent  # noqa: F401
from test_ask_answers_threats import CHAT as THREAT_CHAT, PLATFORM as THREAT_PLATFORM, post, world  # noqa: F401
from test_ask_answers_threats import body as answer_body
from test_ask_pause import asked as pause_asked, conversation, its_message, ready_ids
from test_instinct_cutover import _imports_instinct
from test_mind_cadence import CADENCE_ITEM, CadenceRouter, cadence_turns
from test_mind_cadence import CONTACT as CADENCE_CONTACT, owner_turn as cadence_turn
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (the pinned wall clock a mind fixture needs)
from test_mind_social import contact, make  # noqa: F401  (a mind with a contact store)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SIDECAR = ROOT / "sidecar" / "colonymind"
PLUGIN = ROOT / "plugins" / "hermes-plugin"
APPROVAL_DOCS = ("MIND.md", "OWNER-GATE.md", "CREDENTIALS.md", "HERMES-ADAPTER.md", "DECISIONS.md", "KNOWN-GAPS.md")


def _read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _sections(text: str, level: int = 2) -> dict[str, str]:
    """``{heading: body}`` for the headings of one level; a body runs to the next heading of that level or higher."""
    marks = "#" * level
    parts = re.split(rf"^{marks} (.+)$", text, flags=re.MULTILINE)
    found = {}
    for index in range(1, len(parts), 2):
        body = parts[index + 1]
        higher = re.search(rf"^#{{1,{level - 1}}} ", body, flags=re.MULTILINE) if level > 1 else None
        found[parts[index].strip()] = body[:higher.start()] if higher else body
    return found


def _asks() -> str:
    return _sections(_read("MIND.md"))["Asks"]


def _unwrapped(path: Path) -> str:
    """A source file with its comment and docstring lines joined, so a reference that wraps reads whole."""
    return re.sub(r"\s*\n\s*(?:#:?\s*)?", " ", path.read_text(encoding="utf-8"))


def _sources(*bases: Path) -> list[Path]:
    return [path for base in bases for path in sorted(base.rglob("*.py")) if "__pycache__" not in path.parts]


# -- MIND.md: the sections the code cites --------------------------------------------------------------------------

def test_every_asks_section_the_code_cites_exists():
    """``docs/MIND.md, Asks: <part>`` in a docstring or comment names a part of the Asks section a reader can find."""
    cited = set()
    for path in _sources(SIDECAR, PLUGIN, ROOT / "sidecar" / "tests", ROOT / "tests"):
        for match in re.finditer(r"docs/MIND\.md, Asks: ([A-Za-z][A-Za-z ]*?)(?=[).;,]|\s-|$)", _unwrapped(path)):
            cited.add(match.group(1).strip().lower())
    assert {"console and api answers", "who may answer"} <= cited
    parts = {name.lower() for name in _sections(_asks(), level=3)}
    assert sorted(cited - parts) == []


def test_every_mind_section_the_code_cites_for_approvals_exists():
    headings = set(_sections(_read("MIND.md")))
    subheadings = {name for body in _sections(_read("MIND.md")).values() for name in _sections(body, level=3)}
    for name in ("Asks", "Questions", "The daily digest", "Standing grants", "The floor"):
        assert name in headings, name
    assert "Declared actions" in subheadings


# -- no code instructions --------------------------------------------------------------------------------------------

# How the 1.11 answer was spelled: a code typed after a yes (the console's refused ``colonymind mind yes <code>`` may
# be named), a reply "with the code", a "tap" where no tap exists.
CODE_INSTRUCTIONS = (r"(?<!colonymind mind )\byes <code>", r"\byes\|no <code>", r"\bwith the code\b", r"\bone[- ]tap\b",
                     r"\btypes? (?:its|the) code\b", r"\btyping the code\b", r"\btyped code\b")


def test_no_approval_doc_tells_the_owner_to_answer_with_a_code():
    """Asks carry no code the owner reads; a code typed in chat approves nothing. The console's ``no <code>`` and
    ``show <code>`` stay: codes are the asks' ids there."""
    found = []
    for name in APPROVAL_DOCS:
        text = _read(name)
        for pattern in CODE_INSTRUCTIONS:
            found += [f"{name}: {match.group(0)!r}" for match in re.finditer(pattern, text, flags=re.IGNORECASE)]
        if digest.NEEDS_OK_1_11 in text:
            found.append(f"{name}: the 1.11 label {digest.NEEDS_OK_1_11!r}")
    assert found == []


def test_the_digest_section_names_the_parts_the_digest_writes():
    section = _sections(_read("MIND.md"))["The daily digest"]
    for label in (digest.WAITING, digest.DONE):
        assert f"**{label}**" in section, label


# -- settings and windows --------------------------------------------------------------------------------------------

def _configuration() -> dict:
    block = _sections(_read("MIND.md"))["Configuration"].split("```yaml", 1)[1].split("```", 1)[0]
    return yaml.safe_load(block)


def test_the_asks_settings_are_in_the_configuration_with_their_defaults():
    settings = AskSettings()
    documented = _configuration()["mind"].get("asks")
    assert documented == {"messages_per_day": settings.messages_per_day, "remind_hours": settings.remind_hours,
                          "quiet_minutes": settings.quiet_minutes}


def _span(value: float, unit: str) -> str:
    return f"{value:g} {unit}{'' if value == 1 else 's'}"


def test_the_answer_windows_are_documented_as_the_code_has_them():
    """Each window the answer path keeps is a default chosen for this release; the Asks section names it with the
    value the code has, so a changed default shows in the docs."""
    section = _asks()
    windows = {
        "an ask delivered this close before a message is never what it answered": _span(answers.SETTLE_SECONDS,
                                                                                          "second"),
        "two sends this close in no known order bind nothing": _span(answers.ORDER_SECONDS, "second"),
        "a floor or widening ask binds by follow-up this long": _span(answers.FOLLOW_UP_HOURS, "hour"),
        "a list's numbers mean it this long": _span(answers.LIST_HOURS, "hour"),
        "one plain question per ask in this long": _span(answers.CLARIFY_HOURS, "hour"),
        "an approval can be undone this long": _span(answers.UNDO_MINUTES, "minute"),
        "the reminder waits for a digest this long after the ask": _span(
            outbox.REMIND_AFTER_DIGEST.total_seconds() / 3600, "hour"),
        "a list shows at most": f"{outbox.LIST_ITEMS} asks",
        "what a yes authorizes is shown in its message up to": f"{outbox.MANDATORY_CHARS:,} characters",
    }
    assert {what: span for what, span in windows.items() if span not in section} == {}


# -- the API table and the CLI block ---------------------------------------------------------------------------------

ANSWER_ROUTES = re.compile(r"^/v1/mind/(?:asks|answers|grants|allows|tripwire|decide)\b")
# Path parameters the route checks against a closed set: the table spells each value.
ENUMERATED = {"answer": ("yes", "no")}


def _documented_routes(text: str) -> set[tuple[str, str]]:
    found = set()
    for method, path in re.findall(r"`(GET|POST) (/[^`?\s,]*)", text):
        found.add((method, re.sub(r"\{[^}]+\}", "{}", path)))
    return found


def test_the_api_table_lists_every_route_of_the_answer_path():
    from colonymind.api.routers import mind as mind_router
    table = _sections(_read("MIND.md"))["The API (`/v1/mind`, one bearer key)"]
    documented = _documented_routes(table)
    missing = []
    for route in mind_router.router.routes:
        if not ANSWER_ROUTES.match(route.path):
            continue
        path = route.path[len("/v1/mind"):]
        variants = [path]
        for name, values in ENUMERATED.items():
            if "{" + name + "}" in path:
                variants = [path.replace("{" + name + "}", value) for value in values]
        for method in route.methods:
            missing += [f"{method} {variant}" for variant in variants
                        if (method, re.sub(r"\{[^}]+\}", "{}", variant)) not in documented]
    assert sorted(missing) == []
    assert "approve_required" in table


def _served_routes() -> set[tuple[str, str]]:
    """``(method, path)`` for every route of the mind router, relative to ``/v1/mind``, a closed path parameter
    spelled out per value and the others as ``{}``."""
    from colonymind.api.routers import mind as mind_router
    served = set()
    for route in mind_router.router.routes:
        path = route.path[len("/v1/mind"):]
        variants = [path]
        for name, values in ENUMERATED.items():
            if "{" + name + "}" in path:
                variants = [path.replace("{" + name + "}", value) for value in values]
        served |= {(method, re.sub(r"\{[^}]+\}", "{}", variant)) for method in route.methods for variant in variants}
    return served


def test_the_api_table_names_no_answer_route_the_sidecar_does_not_serve():
    """A route of the answer path that is removed (the 1.11 chat route) leaves the table with it."""
    table = _sections(_read("MIND.md"))["The API (`/v1/mind`, one bearer key)"]
    documented = {(method, path) for method, path in _documented_routes(table)
                  if ANSWER_ROUTES.match(f"/v1/mind{path}")}
    assert documented and sorted(documented - _served_routes()) == []


ASK_COMMANDS = ("asks", "asks-report", "show", "yes", "no", "grants")
# Commands of the answer path the CLI has only once their unit ships: listed as soon as the CLI has them.
ASK_COMMANDS_WHEN_BUILT = ("allows",)


def _mind_commands() -> set[str]:
    from colonymind.mind import cli as mind_cli
    parser = argparse.ArgumentParser()
    mind_cli.add_parser(parser.add_subparsers())
    mind = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction)).choices["mind"]
    return set(next(action for action in mind._actions if isinstance(action, argparse._SubParsersAction)).choices)


class _Parsed(Exception):
    """The command line parsed; nothing ran."""


def _cli_lines() -> list[str]:
    block = _sections(_read("MIND.md"))["The CLI"].split("```", 1)[1].split("```", 1)[0]
    return [re.split(r"\s{2,}", line.strip(), maxsplit=1)[0] for line in block.splitlines()
            if line.startswith("colonymind mind ")]


def test_the_cli_block_lists_every_ask_command_and_each_parses(monkeypatch, capsys):
    from colonymind import cli
    lines = _cli_lines()
    listed = {line.split()[2] for line in lines}
    built = [command for command in ASK_COMMANDS_WHEN_BUILT if command in _mind_commands()]
    assert [command for command in (*ASK_COMMANDS, *built) if command not in listed] == []
    parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        parse(self, args, namespace)
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    refused = []
    for line in lines:
        if line.split()[2] not in (*ASK_COMMANDS, *ASK_COMMANDS_WHEN_BUILT):
            continue
        words = [re.sub(r"^<[^>]+>$", "X", word) for word in shlex.split(re.sub(r"\[[^\]]*\]", "", line))[1:]]
        monkeypatch.setattr("sys.argv", ["colonymind", *words])
        try:
            cli.main()
        except _Parsed:
            continue
        except SystemExit:
            pass
        refused.append(line)
    capsys.readouterr()
    assert refused == []


# -- OWNER-GATE.md ---------------------------------------------------------------------------------------------------

def test_the_owner_gate_doc_names_every_prompted_type_and_the_pause():
    text = _read("OWNER-GATE.md")
    named = set(re.findall(r"`([a-z_*]+)`", text))
    missing = [kind for kind in sorted(gate.PROMPTED_TYPES)
               if kind not in named and not (kind.startswith("commitment_") and "commitment_*" in named)]
    assert missing == []
    assert set(gate.PAUSED_TYPES) <= named
    review = _sections(text)["The review"]
    assert "pause" in review and "quiet_minutes" in review
    configuration = _sections(text)["Configuration"].split("```yaml", 1)[1].split("```", 1)[0]
    assert yaml.safe_load(configuration)["mind"]["asks"]["quiet_minutes"] == AskSettings().quiet_minutes


# -- CREDENTIALS.md --------------------------------------------------------------------------------------------------

# A refusal of the answer path's routes: the status, and a file that raises it.
REFUSALS = {"approve_required": (403, SIDECAR / "api" / "routers" / "mind.py"),
            "needs_owner_ask": (409, SIDECAR / "api" / "routers" / "people.py")}


def test_the_credentials_doc_names_each_refusal_the_approval_routes_give():
    rows = [line for line in _read("CREDENTIALS.md").splitlines() if line.startswith("| ")]
    for code, (status, source) in REFUSALS.items():
        assert f'"code": "{code}"' in source.read_text(encoding="utf-8"), code
        assert any(line.startswith(f"| {status} | `{code}` |") for line in rows), code


def test_the_doctors_approval_warnings_are_named_in_the_docs():
    """A WARN line of ``colonymind doctor`` names a check; the docs say what it means and how to fix it."""
    results = doctor.owner_lane_results(
        {"backend": "local", "lanes": {"whatsapp": ["terminal"], "cli": ["terminal"]},
         "whatsapp": {"enabled": True, "bot": False}},
        {"api-key": {"whatsapp", "cli"}})
    warnings = sorted(result.name for result in results if result.status == doctor.WARN)
    assert len(warnings) == 3, warnings
    for name in warnings:
        assert f"`{name}`" in _asks(), name
    credentials = _read("CREDENTIALS.md")
    assert "`cli-owner-platform`" in credentials and "`cli`" in credentials


# -- HERMES-ADAPTER.md -----------------------------------------------------------------------------------------------

def _registered_hooks() -> list[str]:
    return sorted(set(re.findall(r'register_hook\("(\w+)"', (PLUGIN / "__init__.py").read_text(encoding="utf-8"))))


def test_the_adapter_doc_lists_every_hook_the_plugin_registers():
    seams = _sections(_read("HERMES-ADAPTER.md"))["Seams used"]
    hooks = _registered_hooks()
    assert {"pre_llm_call", "pre_gateway_dispatch", "gateway_platform_event"} <= set(hooks)
    assert [hook for hook in hooks if f"`{hook}`" not in seams] == []


def test_the_adapter_doc_names_only_routes_the_plugin_calls_for_asks():
    section = _sections(_read("HERMES-ADAPTER.md"))["Asks"]
    plugin = "\n".join(path.read_text(encoding="utf-8") for path in _sources(PLUGIN))
    calls = {re.sub(r"\{[^}]*\}", "{}", route) for route in re.findall(r'"(/v1/mind/[^"\s]*)"', plugin)}
    named = {re.sub(r"\{[^}]*\}", "{}", route) for route in re.findall(r"(/v1/mind/[A-Za-z0-9_/{}-]*)", section)}
    assert {"/v1/mind/answers", "/v1/mind/asks/{}/show"} <= named
    assert sorted(named - calls) == []


# -- DECISIONS.md ----------------------------------------------------------------------------------------------------

ANSWER_READERS = (SIDECAR / "mind" / "answers.py", SIDECAR / "mind" / "answer_turns.py", PLUGIN / "answers.py")


def test_the_answer_reader_never_imports_instinct_and_the_decisions_doc_says_so():
    """No model reading approves an ask: the reader that settles one never imports Instinct."""
    for path in ANSWER_READERS:
        assert _imports_instinct(path) == [], path
    intro = _read("DECISIONS.md").split("\n## ", 1)[0]
    paragraph = next(part for part in intro.split("\n\n") if "never import Instinct" in part)
    assert "`mind/answers.py`" in paragraph


# -- KNOWN-GAPS.md ---------------------------------------------------------------------------------------------------

# What the asks report counts, and the words KNOWN-GAPS.md uses for it while nothing records it.
UNBUILT = {asks_report.ASK_GRACE_CANCELLED: "floor grace", asks_report.ASK_REVIVED: "revived",
           asks_report.INVENTED_REFERENCE: "invented code lines"}


def _recorded(event: str) -> bool:
    """Whether the sidecar records ``event`` anywhere but the report itself: a history action written by its value or
    by a constant bound to it (``action=ASK_SHOWN``), or, for a message-level event, any use of it."""
    names = sorted({name for module in (asks_report, outbox) for name, value in vars(module).items()
                    if name.isupper() and value == event}) or ["(?!)"]
    if event.startswith("ask_"):
        pattern = re.compile(r"action\s*=\s*(?:(?:\w+\.)?(?:" + "|".join(names) + r")\b|[\"']"
                             + re.escape(event) + r"[\"'])")
    else:
        pattern = re.compile("|".join([*(rf"\b{name}\b" for name in names), re.escape(event)]))
    return any(pattern.search(path.read_text(encoding="utf-8")) for path in _sources(SIDECAR)
               if path.name != "asks_report.py")


def test_what_the_asks_report_counts_but_nothing_records_is_a_known_gap():
    events = [event for event in asks_report.EVENT_NAMES if event.startswith("ask_")]
    events.append(asks_report.INVENTED_REFERENCE)
    unrecorded = [event for event in events if not _recorded(event)]
    assert set(unrecorded) <= set(UNBUILT), unrecorded
    gaps = _read("KNOWN-GAPS.md")
    assert [event for event in unrecorded if UNBUILT[event] not in gaps] == []


def test_the_1_11_chat_route_is_a_known_gap_exactly_while_it_serves():
    serves = ("POST", "/decide") in _served_routes()
    assert ("`/v1/mind/decide`" in _read("KNOWN-GAPS.md")) == serves


def _always_offered() -> list[str]:
    """Where an ask offers "always": a call of the reader, outside it, that passes ``offers_always`` other than
    ``False``. The reader reads "always" only where the ask offers it."""
    call = re.compile(r"\boffers_always\s*=\s*(?!False\b)")
    reader = SIDECAR / "mind" / "answers.py"
    return [path.name for path in _sources(SIDECAR) if path != reader and call.search(path.read_text(encoding="utf-8"))]


def test_the_docs_say_whether_an_ask_offers_always():
    """While no ask offers "always", the Asks section and the known gaps say so; once one does, neither may."""
    offered = bool(_always_offered())
    reading = _flat(_sections(_asks(), level=3)["Reading the words"])
    gaps = _flat(_read("KNOWN-GAPS.md"))
    said = {"MIND.md": "no ask offers it yet" in reading, "KNOWN-GAPS.md": 'nothing offers "always"' in gaps}
    assert said == {"MIND.md": not offered, "KNOWN-GAPS.md": not offered}


def test_the_known_gaps_page_states_the_forged_turn_exposure():
    gaps = _read("KNOWN-GAPS.md")
    assert "answer_forgery_attempt" in gaps and "colonymind doctor" in gaps


# -- what reaches the owner's chat -----------------------------------------------------------------------------------

def _flat(text: str) -> str:
    return " ".join(text.split())


async def test_the_docs_say_which_approvals_reach_the_owners_chat(fx):
    """A reaction gets one line in the owner's chat (``ask_ack``); a text answer on ``/v1/mind/answers`` tells only
    its caller, whose model acknowledges it in its reply. A forged text answer therefore approves with nothing in the
    owner's chat: no page promises that every approval is told to the owner, and both say which are."""
    tapped, typed = asked(fx, "K7F"), asked(fx, "Q2Z")
    deliver(fx, tapped, message_id="m-1")
    own = deliver(fx, typed, message_id="m-2")
    fx.shift(minutes=1)
    tap = await fx.mind.read_answer(evidence(fx, reaction="\N{THUMBS UP SIGN}", reply_to_message_id="m-1",
                                             message_id="r-1", ledger=None))
    text = await fx.mind.read_answer(evidence(fx, "yes", message_id="in-2", ledger=[sent(own)]))
    assert tap["approved"] == [tapped.id] and text["approved"] == [typed.id]
    acks = fx.store.intentions(kind=["message"], types=[tick.ASK_ACK], limit=10)
    assert tap["acknowledged"] and text["acknowledged"] is None and [row.id for row in acks] == [tap["acknowledged"]]
    for name, part in (("KNOWN-GAPS.md", _read("KNOWN-GAPS.md")), ("MIND.md", _asks())):
        flat = _flat(part)
        assert "Every approval is told to the owner" not in flat, name
        assert "a forged text answer" in flat and "`ask_ack`" in flat, name


def _undo_bullet() -> str:
    return _flat(next(part for part in _sections(_asks(), level=3)["What an answer does"].split("\n- ")
                      if part.startswith("**Undo.**")))


async def test_the_undo_bullet_names_what_an_undo_cannot_take_back(make):
    """An undo takes back work not started, a standing grant, and what a people change set; a yes that already did
    anything else stays. A check-in matched by name only keeps the cadence its yes set, and an identity link stays:
    the bullet promises no rollback of either and names both."""
    named = {**CADENCE_ITEM, "metadata": {**CADENCE_ITEM["metadata"], "recipient": "Sam"}, "counterpart": "Sam"}
    fx = make([contact(CADENCE_CONTACT, may_contact="auto", name="Sam")], router=CadenceRouter(named))
    from colonymind.commitments.extract import CommitmentExtractor
    fx.mind.capture = CommitmentExtractor(fx.ledger, lambda: fx.commitments)
    cadence_turn(fx, "turn-1", cadence_turns()[0].replace(CADENCE_CONTACT, "Sam"))
    formed, = (await fx.tick())["formed"]
    cadence = fx.store.get(formed["id"])
    link, _ = fx.store.create_intention(
        kind="note", type="link_proposal", title="Is this handle Sam?", drive="duty", cls="owner", decision="ask",
        decision_reason="test", status="asked", dedup_key="test:link", ask_code="L1K", source_id="candidate-1",
        context={"notice": True}, hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
    undone = {}
    for number, row in enumerate((cadence, link)):
        own = deliver(fx, row, message_id=f"m-{number}")
        fx.shift(timedelta(minutes=1))
        yes = await fx.mind.read_answer(evidence(fx, "yes", message_id=f"yes-{number}", ledger=[sent(own)]))
        assert yes["approved"] == [row.id], row.type
        fx.shift(timedelta(minutes=1))
        undo = await fx.mind.read_answer(evidence(fx, "undo", message_id=f"undo-{number}", ledger=[]))
        undone[row.type] = [item["undone"] for item in undo["undone"]]
    assert undone == {"cadence_confirm": ["too_late"], "link_proposal": ["too_late"]}
    assert fx.contacts.records[CADENCE_CONTACT]["cadence_minutes"] == CADENCE_ITEM["metadata"]["cadence_minutes"]
    bullet = _undo_bullet()
    assert "a raised permission or check-in cadence is set back" not in bullet
    assert "a people change set is set back" in bullet
    assert "a check-in matched by name only keeps the cadence" in bullet and "an identity link" in bullet


def test_the_asks_report_paragraph_says_what_its_json_carries(fx):
    """The report's text is counts and times; its JSON (``--json``, ``/v1/mind/asks/stats``) also lists each ask
    formed in the window by its id and code. The paragraph promises no more than each form keeps back."""
    row = asked(fx)
    deliver(fx, row, message_id="m-1")
    value = asks_report.report(fx.store, since=fx.now - timedelta(days=7), now=fx.now, events=[])
    record, = value["asks"]
    assert record["id"] == row.id and record["code"] == row.ask_code
    text = asks_report.render(value)
    assert row.ask_code not in text and row.id not in text and row.description not in repr(value)
    paragraph = _flat(_sections(_asks(), level=3)["The asks report"])
    assert "Counts and times only, never an ask's words or code." not in paragraph
    assert "`asks`" in paragraph and "`id`" in paragraph and "`code`" in paragraph


def test_the_invented_code_line_count_the_running_mind_reports_is_documented_as_it_reads(fx):
    """The plugin removes invented code lines and counts them in its own log; nothing in the sidecar records them.
    The running mind hands the report its answer log all the same, so the line reads 0, not "not recorded": the
    known gap says that 0 is not a measurement, for as long as nothing records the event."""
    asked(fx)
    value = asks_report.report(fx.store, since=fx.now - timedelta(days=7), now=fx.now,
                               events=fx.mind.answer_events(fx.now - timedelta(days=7)))
    line = asks_report.render(value).splitlines()[-1]
    gap = _flat(next(part for part in _read("KNOWN-GAPS.md").split("\n  - ") if "invented code lines" in part))
    if value["invented_references"] is None:
        assert "Invented code lines removed: not recorded yet." in line and "not recorded" in gap
    else:
        assert not _recorded(asks_report.INVENTED_REFERENCE) and value["invented_references"] == 0
        assert "Invented code lines removed: 0." in line
        assert "not recorded" not in gap and "reads 0" in gap and "not a measurement" in gap


def _notice_1_11(fx, rows, message_id: str) -> None:
    """A 1.11 ask notice delivered to the owner's chat: one message listing the open asks by code."""
    listed = "\n".join([f"{digest.NEEDS_OK_1_11}:", *(f"- {row.ask_code}: {row.description}." for row in rows),
                        "Without an answer they expire and nothing is done."])
    notice = fx.mind.outbox.notice(type="ask_notice", title=f"{len(rows)} open ask(s)", text=listed,
                                   dedup_key=f"notice:{message_id}")
    fx.mind.outbox.sending(notice.id, target=f"{PLATFORM}:{CHAT}")
    fx.mind.outbox.sent(notice.id, result="sent", hermes_ref=message_id)


async def test_the_1_11_notice_paragraph_says_what_binds_on_such_a_notice(fx):
    """On a 1.11 notice listing several asks, a quoted reply binds as a list for a day and a reaction binds nothing,
    as on any list; on one listing a single ask, a reaction answers it. The paragraph says each."""
    one, first, second, old_a, old_b = (asked(fx, code) for code in ("K7F", "Q2Z", "M4P", "R8T", "W3X"))
    _notice_1_11(fx, [old_a, old_b], "m-old")
    fx.shift(hours=answers.LIST_HOURS + 1)
    _notice_1_11(fx, [one], "m-1")
    _notice_1_11(fx, [first, second], "m-2")
    fx.shift(minutes=1)
    thumbs = "\N{THUMBS UP SIGN}"
    single = await fx.mind.read_answer(evidence(fx, reaction=thumbs, reply_to_message_id="m-1", message_id="r-1",
                                                ledger=None))
    tapped = await fx.mind.read_answer(evidence(fx, reaction=thumbs, reply_to_message_id="m-2", message_id="r-2",
                                                ledger=None))
    quoted = await fx.mind.read_answer(evidence(fx, "2", reply_to_message_id="m-2", message_id="in-3", ledger=None))
    late = await fx.mind.read_answer(evidence(fx, "both", reply_to_message_id="m-old", message_id="in-4",
                                              ledger=None))
    assert single["path"] == "reaction" and single["approved"] == [one.id]
    assert tapped["path"] is None and tapped["settled"] == []
    assert quoted["path"] == "list" and quoted["approved"] == [second.id]
    assert late["path"] is None and late["settled"] == []
    paragraph = _flat(_sections(_asks(), level=3)["Asks from before 1.12"])
    assert "binds a quoted reply or a reaction as a list" not in paragraph
    assert f"a quoted reply as a list for {_span(answers.LIST_HOURS, 'hour')}" in paragraph
    assert "a reaction on it is ignored" in paragraph and "listed one ask is that ask's own message" in paragraph


# -- the API table's shown route -------------------------------------------------------------------------------------

def _api_row(route: str) -> str:
    table = _sections(_read("MIND.md"))["The API (`/v1/mind`, one bearer key)"]
    return _flat(next(line for line in table.splitlines() if line.startswith(f"| `{route}` |")))


async def test_the_shown_route_row_says_a_reaction_binds_only_on_a_message_of_one_ask(world):
    """A message a host reports through ``/asks/{id}/shown`` binds as an ask's own message or a list does
    (``answers.bind``): on one ask, a reaction or a quoted reply answers it; on several, a quoted reply binds as that
    list for a day and a reaction binds nothing. The route's row in the API table says each."""
    fx = world.fx
    one, first, second, old_a, old_b = (asked(fx, code) for code in ("K7F", "Q2Z", "M4P", "R8T", "W3X"))

    async def show(rows, message_id):
        response = await world.client.post(f"/v1/mind/asks/{rows[0].id}/shown", json={
            "platform": THREAT_PLATFORM, "chat_id": THREAT_CHAT, "message_id": message_id,
            "asks": [row.id for row in rows]})
        assert response.status_code == 200, response.text

    async def answer(text, message_id, **extra):
        return (await post(world, answer_body(fx, text, message_id, None, **extra))).json()

    await show([old_a, old_b], "m-old")
    fx.shift(hours=answers.LIST_HOURS + 1)
    await show([one], "m-1")
    await show([first, second], "m-2")
    fx.shift(minutes=1)
    thumbs = "\N{THUMBS UP SIGN}"
    tapped = await answer("", "r-2", reaction=thumbs, reply_to_message_id="m-2")
    single = await answer("", "r-1", reaction=thumbs, reply_to_message_id="m-1")
    quoted = await answer("2", "in-3", reply_to_message_id="m-2")
    late = await answer("both", "in-4", reply_to_message_id="m-old")
    assert tapped["path"] is None and tapped["settled"] == []
    assert single["path"] == "reaction" and single["approved"] == [one.id]
    assert quoted["path"] == "list" and quoted["approved"] == [second.id]
    assert late["path"] is None and late["settled"] == []
    assert fx.store.get(first.id).status == "asked"
    row = _api_row("POST /asks/{id}/shown")
    assert "a reaction or a quoted reply on that message now binds" not in row
    assert "on one ask, a reaction or a quoted reply answers it" in row
    assert f"a quoted reply binds as that list for {_span(answers.LIST_HOURS, 'hour')}" in row
    assert "a reaction on it is ignored" in row


# -- the pause and the gate's longest wait ---------------------------------------------------------------------------

@pytest.mark.parametrize("page", ["MIND.md", "OWNER-GATE.md"])
def test_the_pause_docs_explain_the_owner_started_exception_and_the_remaining_protection(page):
    """The owner's request starts the work, so its reply need not postpone the permission it needs. Other recent
    sends still protect a bare yes, and a background ask keeps the existing pause."""
    if page == "MIND.md":
        paragraph = _asks().split("**A pause first.**", 1)[1].split("\n- ", 1)[0]
    else:
        paragraph = _read(page).split("**The pause.**", 1)[1].split("\n\n", 1)[0]
    paragraph = _flat(paragraph)
    assert "`delivery_ref`" in paragraph and "owner's request" in paragraph
    assert "stored note has origin `hand`" in paragraph
    assert all(term in paragraph for term in ("`held_from`", "`origin_ref`", "`owner_turn`", "owner contact"))
    assert all(platform in paragraph for platform in ("WhatsApp", "SMS", "RCS", "API"))
    assert "reply to that request" in paragraph and "without the conversational pause" in paragraph
    assert "another ask or message the gate let go" in paragraph
    assert "Background asks keep the full pause and bounds" in paragraph
    assert "`answer_turn`" in paragraph


def test_the_after_reply_docs_cover_ask_notices_and_the_bounded_chat_hold():
    from colonymind_hermes import answers as host_answers

    paragraph = _flat(_read("HERMES-ADAPTER.md").split("**After the reply.**", 1)[1].split("\n- ", 1)[0])
    assert all(f"`{kind}`" in paragraph for kind in ("ask_notice", "ask_show", "ask_clarify"))
    assert "target chat" in paragraph
    assert f"{host_answers.AFTER_REPLY_SECONDS:g} s after" in paragraph
    assert f"no reply after {host_answers.TURN_OPEN_SECONDS / 60:g} minutes" in paragraph
    assert "no turn is open" in paragraph


@pytest.mark.parametrize("quiet, longest", [(10, 60), (240, 60), (30, 5)])
async def test_the_pause_docs_bound_an_ordinary_ask_by_the_longer_of_the_pause_and_the_gates_longest_wait(
        tmp_path, monkeypatch, quiet, longest):
    """Through a conversation that never stops, a floor ask waits ``mind.asks.quiet_minutes`` and any other ask the
    longer of those minutes and the gate's ``max_wait_minutes``, each from its first wait: a pause set longer than
    the gate's longest wait holds an ordinary ask for the whole pause. Both pages say the longer of the two."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, config={"faculties": {"owner_gate": True}, "quiet_hours": "",
                                   "asks": {"quiet_minutes": quiet}, "owner_gate": {"max_wait_minutes": longest}})
    try:
        floor = await its_message(fx, pause_asked(fx, "K7F", "Let this run once: delete old-builds", cls="floor",
                                                  type_="floor_effect",
                                                  context={"effect": {"cls": "irreversible_deletion"}}))
        ordinary = await its_message(fx, pause_asked(fx, "Q2Z", "Check the venue hours before Friday"))
        went = {}
        for minute in range(0, max(quiet, longest) + 4):   # a turn every four minutes, never answered by an ask
            if minute % 4 == 0:
                conversation(fx, f"t-{minute}")
            for ident in await ready_ids(fx):
                went.setdefault(ident, minute)
                fx.mind.outbox.sending(ident, target="telegram:owner-chat")
                fx.mind.outbox.sent(ident)
            fx.shift(minutes=1)
    finally:
        fx.store.close()
    assert (went.get(floor.id), went.get(ordinary.id)) == (quiet, max(quiet, longest))
    gate_pause = _flat(_read("OWNER-GATE.md").split("**The pause.**", 1)[1].split("\n\n", 1)[0])
    mind_pause = _flat(_asks().split("**A pause first.**", 1)[1].split("\n- ", 1)[0])
    assert "any other ask at most `max_wait_minutes`" not in gate_pause
    assert "any other ask at most the longer of `quiet_minutes` and `max_wait_minutes`" in gate_pause
    assert "any other at most the gate's `max_wait_minutes`" not in mind_pause
    assert "any other at most the longer of those minutes and the gate's `max_wait_minutes`" in mind_pause
