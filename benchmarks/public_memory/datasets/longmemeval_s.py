"""LongMemEval-S (cleaned): 500 questions, each over its own history of about 48 dated sessions.

Source: ``longmemeval_s_cleaned.json`` from the dataset ``xiaowu0162/longmemeval-cleaned`` at a pinned revision
(MIT; Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory", ICLR 2025).

Each question is its own conversation. Its sessions are dated ``YYYY/MM/DD (Day) HH:MM``; the weekday is checked
against the date. Gold sessions are the question's ``answer_session_ids``; gold turns are the turns the file marks
``has_answer``. A question whose id ends in ``_abs`` is an abstention question.

What the adapter finds in the upstream file, by finding name:

- ``reordered``: the file lists the question's sessions out of time order; they replay in time order (a tie keeps
  the file's order).
- ``tied_sessions``: two of its sessions carry the same minute.
- ``asked_after_last_session``: the question is stamped earlier than a session of the same day. The question is
  asked one minute after that session, still on its own day (at 23:59:59 at the latest); ``date_text`` keeps the
  stamp the dataset gives the reader. A session on a later day than its question is refused.
- ``repeated_sessions``: the history lists one session id more than once; each occurrence is kept, the second as
  ``<id>#2``.
- ``unmarked_answer_sessions``: a gold session marks no turn as evidence (every abstention question in the pinned
  file, and some others), so turn-level reach can only be checked at the session level there.

The upstream judge prompts, the abstention prompt among them, are in the LongMemEval repository (MIT):
``JUDGE_PIN`` pins ``src/evaluation/evaluate_qa.py``; ``get_anscheck_prompt`` holds one prompt per question type
and, in its ``abstention=True`` branch (line 41 at the pinned commit), the abstention prompt, which the evaluator
chooses when the question id contains ``_abs`` (line 101).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import pinned, records
from .records import UTC, Conversation, DatasetError, Evidence, Question, Session, Turn

NAME = "longmemeval_s"
PIN = pinned.Pin(
    name="longmemeval_s_cleaned.json",
    url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
        "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
    sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    size=277383467,
)
JUDGE_PIN = pinned.Pin(
    name="longmemeval-evaluate_qa.py",
    url="https://raw.githubusercontent.com/xiaowu0162/LongMemEval/"
        "9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py",
    sha256="ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    size=7436,
)
PINNED = (PIN, JUDGE_PIN)
LICENSE = "MIT"
# The published question counts (the paper's statistics); ``load`` refuses a file that parses to others.
EXPECTED_TYPES = {"single-session-user": 70, "single-session-assistant": 56, "single-session-preference": 30,
                  "multi-session": 133, "temporal-reasoning": 133, "knowledge-update": 78}
EXPECTED_ABSTENTION = 30
ABSTENTION_SUFFIX = "_abs"
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
STAMP = re.compile(r"(\d{4})/(\d{2})/(\d{2}) \((\w{3})\) (\d{2}):(\d{2})")
FINDINGS = ("reordered", "tied_sessions", "asked_after_last_session", "repeated_sessions",
            "unmarked_answer_sessions")
ENTRY_KEYS = frozenset({"question_id", "question_type", "question", "answer", "question_date",
                        "haystack_session_ids", "haystack_dates", "haystack_sessions", "answer_session_ids"})


def parse_stamp(text: str, where: str) -> datetime:
    """``2023/05/20 (Sat) 02:21`` as UTC; the weekday must be the date's."""
    match = STAMP.fullmatch(text)
    if not match:
        raise DatasetError(f"{where}: {text!r} is not a YYYY/MM/DD (Day) HH:MM stamp")
    year, month, day, weekday, hour, minute = match.groups()
    try:
        moment = datetime(int(year), int(month), int(day), int(hour), int(minute), tzinfo=UTC)
    except ValueError as error:
        raise DatasetError(f"{where}: {text!r} is not a date ({error})") from None
    if WEEKDAYS[moment.weekday()] != weekday:
        raise DatasetError(f"{where}: {text!r} names the wrong weekday for its date")
    return moment


def _question(item: dict, findings: dict) -> tuple:
    qid = str(item["question_id"])
    stamped = parse_stamp(item["question_date"], qid)
    ids, dates, histories = item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    if not len(ids) == len(dates) == len(histories):
        raise DatasetError(f"{qid}: session ids, dates and sessions differ in number")
    listed, marked, seen = [], {}, {}
    for position, (source_id, date, turns) in enumerate(zip(ids, dates, histories)):
        seen[source_id] = seen.get(source_id, 0) + 1
        key = source_id if seen[source_id] == 1 else f"{source_id}#{seen[source_id]}"
        where = f"{qid} session {key}"
        listed.append(Session(key=key, source_id=source_id, position=position, at=parse_stamp(date, where),
                              turns=tuple(Turn(index=index, role=records.check_role(entry["role"], where),
                                               content=entry["content"]) for index, entry in enumerate(turns))))
        marked[key] = tuple(index for index, entry in enumerate(turns) if entry.get("has_answer") is True)
    sessions = records.in_time_order(listed)
    if [session.key for session in sessions] != [session.key for session in listed]:
        findings["reordered"].append(qid)
    if any(earlier.at == later.at for earlier, later in zip(sessions, sessions[1:])):
        findings["tied_sessions"].append(qid)
    if any(count > 1 for count in seen.values()):
        findings["repeated_sessions"].append(qid)
    asked_at, last = stamped, sessions[-1]
    if last.at > stamped:
        if last.at.date() != stamped.date():
            raise DatasetError(f"{qid}: session {last.key} on {last.at.date()} falls after the question date "
                               f"{stamped.date()}")
        end_of_day = stamped.replace(hour=23, minute=59, second=59)
        asked_at = min(last.at + timedelta(minutes=1), end_of_day)
        findings["asked_after_last_session"].append(qid)
    answers = set(item["answer_session_ids"])
    missing = answers - set(ids)
    if missing:
        raise DatasetError(f"{qid}: answer sessions {sorted(missing)} are not in its history")
    gold = tuple(session.key for session in sessions if session.source_id in answers)
    if any(not marked[key] for key in gold):
        findings["unmarked_answer_sessions"].append(qid)
    evidence = tuple(Evidence(session=session.key, turn=index) for session in sessions for index in marked[session.key])
    question = Question(id=qid, conversation=qid, type=str(item["question_type"]), question=item["question"],
                        answer=str(item["answer"]), asked_at=asked_at, date_text=item["question_date"],
                        abstention=qid.endswith(ABSTENTION_SUFFIX), gold_sessions=gold, evidence=evidence)
    return Conversation(id=qid, sessions=sessions), question


def load_records(path: Path, *, encoding: Optional[str] = "utf-8"):
    """Raw JSON values; each caller applies its own validation and date rules."""
    return json.loads(Path(path).read_text(encoding=encoding))


def parse(path: Path, sha256: Optional[str] = None) -> records.Dataset:
    """The dataset in ``path`` (LongMemEval's JSON list of questions)."""
    items = load_records(path)
    if (not isinstance(items, list) or not items
            or any(not isinstance(item, dict) or not ENTRY_KEYS <= set(item) for item in items)):
        raise DatasetError("A LongMemEval file is a JSON list of questions with their haystack sessions")
    if len({str(item["question_id"]) for item in items}) != len(items):
        raise DatasetError("Question ids must be distinct")
    findings = {name: [] for name in FINDINGS}
    conversations, questions = {}, []
    for item in items:
        conversation, question = _question(item, findings)
        conversations[conversation.id] = conversation
        questions.append(question)
    return records.check(records.Dataset(
        name=NAME, sha256=sha256 or pinned.sha256_file(Path(path)), conversations=conversations,
        questions=tuple(questions), findings={name: tuple(ids) for name, ids in findings.items()}))


def load(directory=None, *, download: bool = True) -> records.Dataset:
    """The pinned file, fetched into the cache if needed, parsed, and checked against the published counts."""
    dataset = parse(pinned.fetch(PIN, directory, download=download), sha256=PIN.sha256)
    records.verify_counts(dataset, EXPECTED_TYPES, EXPECTED_ABSTENTION)
    return dataset
