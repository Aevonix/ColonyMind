"""PrecisionMemBench: 77 retrieval cases over a seed corpus of 35 stored beliefs, and 2 session scenarios.

Source: three fixture files of the repository ``tenurehq/precisionMemBench`` at a pinned commit (MIT; paper arXiv
2605.11325). Scoring reads belief ids only: for each query, which beliefs must, may or must not surface in each
output tier (``personaPrelude``, ``pinnedFacts``, ``relevantBeliefs``, ``openQuestions``); a session turn adds
``noiseCheck`` (beliefs that must not surface) and ``topicRetrieval``.

How it maps onto turn capture (checked against the pinned fixtures):

- The seed beliefs are stored records (type, canonical name, aliases, scope, status, pinned, supersession), not
  conversation. There are no source turns: a replay would have to state each belief's ``content`` as an owner
  message, which measures claim formation as well as retrieval, and ``pinned``, ``epistemic_status`` and
  ``superseded_by`` reach memory only as far as the words carry them.
- The four output tiers have no one-to-one counterpart in a recall packet. Only "this belief surfaces, that one
  does not" maps, through the turn each belief was replayed from.
- The two session scenarios are real user and assistant turns and map directly onto turn capture. Their
  mid-session belief creation and alias update are harness actions with no turn behind them.
- The one belief of another user maps onto person scope: it would be captured under a different person.

Its README's category table lists 8 fuzzy-matching cases; the pinned fixture holds 7, which matches the README's
own total of 89 scored cases (77 single-turn and 12 session turns).

Finding, by name: ``unread_conditions`` (a case or session turn states a condition in a shape the adapter does not
read; the raw expectation stays in ``detail``). A key starting with ``_`` is a note and is not a condition.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from . import pinned
from .records import DatasetError, verify_counts

NAME = "precisionmembench"
_BASE = ("https://raw.githubusercontent.com/tenurehq/precisionMemBench/"
         "61a712a1754e7d27b6dee42da32a2c214b003ece/fixtures")
PINS = {
    "beliefs": pinned.Pin(name="precisionmembench-beliefs.seed.json", url=f"{_BASE}/beliefs.seed.json",
                          sha256="35659b3b9a581b644fc6746e1b4e902c3e7b56c0973b546bf7b0816a69e3de98", size=33303),
    "cases": pinned.Pin(name="precisionmembench-retrieval.cases.json", url=f"{_BASE}/retrieval.cases.json",
                        sha256="09e99f0852280538abe34b03570f65583fdb10df3c554b7150f0f665d5db709c", size=35940),
    "sessions": pinned.Pin(name="precisionmembench-session-retrieval.cases.json",
                           url=f"{_BASE}/session-retrieval.cases.json",
                           sha256="f24a359cdb2bf50237e1ab463af938cd54a81917f64c8d97304de318aed4cc1c", size=37967),
}
PINNED = tuple(PINS.values())
LICENSE = "MIT"
EXPECTED_CATEGORIES = {"Alias resolution": 23, "Scope disambiguation": 12, "Fuzzy matching and prefix guards": 7,
                       "Type routing and open questions": 6, "Design boundary cases": 6,
                       "Budget eviction and capacity": 5, "Persona prelude content": 4, "Relation expansion": 4,
                       "Supersession chain exclusion": 3, "Ranking stability": 3, "Counter-signal retrieval": 2,
                       "Cross-user isolation": 1, "Cold start behavior": 1}
EXPECTED_BELIEFS = 35
EXPECTED_SESSIONS = 2
EXPECTED_SCORED_TURNS = 12
DEFAULT_USER = "test-user"   # the user the seed corpus belongs to; a case names another with ``userId``
# Upstream condition name -> Expectation field. Lists name belief ids, except ``contains`` and
# ``mustNotContain`` (text) and ``shouldRetrieveTurns`` (turn indexes).
CONDITIONS = {"mustInclude": "must_include", "mustExclude": "must_exclude", "shouldOnlyInclude": "only",
              "contains": "contains", "mustNotContain": "must_not_contain", "nonEmpty": "non_empty",
              "isNull": "is_null", "mustNotSurface": "must_not_surface", "shouldRetrieveTurns": "retrieve_turns"}
BELIEF_LISTS = ("must_include", "must_exclude", "only", "must_not_surface")


@dataclass(frozen=True)
class Belief:
    id: str
    user: str
    type: str
    name: str
    aliases: Tuple[str, ...]
    content: str
    scope: Tuple[str, ...]
    status: str
    pinned: bool
    superseded_by: Optional[str]
    created_at: datetime
    updated_at: datetime
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Expectation:
    """One output tier's conditions; ``None`` where the case states none (``only == ()`` means nothing may
    surface)."""
    must_include: Optional[Tuple[str, ...]] = None
    must_exclude: Optional[Tuple[str, ...]] = None
    only: Optional[Tuple[str, ...]] = None
    contains: Optional[Tuple[str, ...]] = None
    must_not_contain: Optional[Tuple[str, ...]] = None
    non_empty: Optional[bool] = None
    is_null: Optional[bool] = None
    must_not_surface: Optional[Tuple[str, ...]] = None
    retrieve_turns: Optional[Tuple[Any, ...]] = None


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    query: str
    scope: Tuple[str, ...]
    user: str
    budget: Optional[Mapping[str, Any]]
    expect: Mapping[str, Expectation]
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionTurn:
    index: int
    label: str
    scope: Tuple[str, ...]
    user_message: str
    assistant_message: str
    expect: Mapping[str, Expectation]
    creates: Optional[Belief]                 # a belief the harness adds after this turn
    updates: Optional[Mapping[str, Any]]      # an alias update the harness applies after this turn
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionCase:
    id: str
    description: str
    turns: Tuple[SessionTurn, ...]


@dataclass(frozen=True)
class Corpus:
    name: str
    sha256: Mapping[str, str]                # per fixture file
    beliefs: Tuple[Belief, ...]
    cases: Tuple[Case, ...]
    sessions: Tuple[SessionCase, ...]
    findings: Mapping[str, Tuple[str, ...]]

    def counts(self) -> dict:
        return dict(Counter(item.category for item in self.cases))

    def abstention_ids(self) -> Tuple[str, ...]:
        return ()

    def scored_turns(self) -> int:
        return sum(1 for session in self.sessions for turn in session.turns if turn.expect)


def _time(text: str, where: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        raise DatasetError(f"{where}: {text!r} is not an ISO time") from None
    if moment.tzinfo is None:
        raise DatasetError(f"{where}: {text!r} has no zone")
    return moment


def _belief(raw: dict) -> Belief:
    bid = raw["_id"]
    known = {"_id", "user_id", "type", "canonical_name", "aliases", "content", "scope", "epistemic_status", "pinned",
             "superseded_by", "created_at", "updated_at"}
    return Belief(id=bid, user=raw["user_id"], type=raw["type"], name=raw["canonical_name"],
                  aliases=tuple(raw.get("aliases") or ()), content=raw["content"],
                  scope=tuple(raw.get("scope") or ()), status=raw["epistemic_status"], pinned=bool(raw.get("pinned")),
                  superseded_by=raw.get("superseded_by"),
                  created_at=_time(raw["created_at"], bid), updated_at=_time(raw["updated_at"], bid),
                  detail={key: value for key, value in raw.items() if key not in known})


def _expectations(raw: Mapping, where: str, unread: list) -> dict:
    tiers, irregular = {}, False
    for tier, conditions in raw.items():
        if tier.startswith("_"):
            continue
        if not isinstance(conditions, dict):
            irregular = True
            continue
        values = {}
        for name, value in conditions.items():
            if name.startswith("_"):
                continue
            if name not in CONDITIONS or isinstance(value, dict):
                irregular = True
                continue
            values[CONDITIONS[name]] = value if isinstance(value, bool) else tuple(value)
        tiers[tier] = Expectation(**values)
    if irregular:
        unread.append(where)
    return tiers


def _check_ids(expect: Mapping[str, Expectation], known: set, where: str) -> None:
    for tier, expectation in expect.items():
        for name in BELIEF_LISTS:
            missing = [bid for bid in getattr(expectation, name) or () if bid not in known]
            if missing:
                raise DatasetError(f"{where}: {tier} names unknown beliefs {missing}")


def parse(paths: Mapping[str, Path], sha256: Optional[Mapping[str, str]] = None) -> Corpus:
    """The corpus in the three fixture files, keyed as ``PINS`` is."""
    raw = {key: json.loads(Path(paths[key]).read_text(encoding="utf-8")) for key in PINS}
    beliefs = tuple(_belief(item) for item in raw["beliefs"])
    created = [turn["createBeliefAtTurn"] for session in raw["sessions"] for turn in session["turns"]
               if turn.get("createBeliefAtTurn")]
    known = {item.id for item in beliefs} | {item["_id"] for item in created}
    if len(known) != len(beliefs) + len(created):
        raise DatasetError(f"{NAME}: two beliefs share an id")
    unread, cases = [], []
    for item in raw["cases"]:
        where = item["caseId"]
        expect = _expectations(item["expect"], where, unread)
        _check_ids(expect, known, where)
        cases.append(Case(id=where, category=item["category"], query=item["query"],
                          scope=tuple(item.get("scope") or ()), user=item.get("userId", DEFAULT_USER),
                          budget=item.get("budget"), expect=expect,
                          detail={key: value for key, value in item.items()
                                  if key not in ("caseId", "category", "query", "scope", "userId", "budget")}))
    sessions = []
    for session in raw["sessions"]:
        turns = []
        for turn in session["turns"]:
            where = f"{session['caseId']}:{turn['turnIndex']}"
            expect = _expectations(turn.get("expect") or {}, where, unread)
            _check_ids(expect, known, where)
            creates = turn.get("createBeliefAtTurn")
            turns.append(SessionTurn(
                index=int(turn["turnIndex"]), label=turn.get("label", ""), scope=tuple(turn.get("scope") or ()),
                user_message=turn["userMessage"], assistant_message=turn["assistantMessage"], expect=expect,
                creates=_belief(creates) if creates else None, updates=turn.get("updateBeliefAtTurn"),
                detail={key: value for key, value in turn.items() if key in ("topics", "turnSignal", "expect")}))
        sessions.append(SessionCase(id=session["caseId"], description=session.get("description", ""),
                                    turns=tuple(turns)))
    ids = [item.id for item in cases] + [item.id for item in sessions]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{NAME}: two cases share an id")
    return Corpus(name=NAME, sha256=dict(sha256 or {key: pinned.sha256_file(Path(paths[key])) for key in PINS}),
                  beliefs=beliefs, cases=tuple(cases), sessions=tuple(sessions),
                  findings={"unread_conditions": tuple(unread)})


def load(directory=None, *, download: bool = True) -> Corpus:
    """The pinned fixtures, fetched into the cache if needed, parsed, and checked against their counts."""
    paths = {key: pinned.fetch(pin, directory, download=download) for key, pin in PINS.items()}
    corpus = parse(paths, sha256={key: pin.sha256 for key, pin in PINS.items()})
    verify_counts(corpus, EXPECTED_CATEGORIES)
    if (len(corpus.beliefs), len(corpus.sessions), corpus.scored_turns()) != (
            EXPECTED_BELIEFS, EXPECTED_SESSIONS, EXPECTED_SCORED_TURNS):
        raise DatasetError(f"{NAME}: {len(corpus.beliefs)} beliefs, {len(corpus.sessions)} sessions and "
                           f"{corpus.scored_turns()} scored turns; the pinned fixtures hold {EXPECTED_BELIEFS}, "
                           f"{EXPECTED_SESSIONS} and {EXPECTED_SCORED_TURNS}")
    return corpus
