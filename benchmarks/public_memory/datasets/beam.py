"""BEAM, the 100K split: 20 conversations of 3 to 5 sessions each and 400 probing questions, 20 per conversation.

Source: the ``100K`` split of the dataset ``Mohammadta/BEAM`` at a pinned revision, one parquet file (Tavakoli et
al., "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs", arXiv 2510.27246).

Licence: CC BY-SA 4.0, as the dataset card at the pinned revision states. Publishing anything derived from the data
(per-question rows, quoted questions or answers) needs the attribution and the same licence for that material.

Each row is one conversation. Its ``chat`` is a list of sessions; the first turn of each session carries the
session's day as a time anchor (``March-15-2024``) and every turn a chat id that runs on across the conversation.
The day has no time of day: a session is placed at 12:00 UTC on it (the adapter's choice), which keeps the
calendar day in any zone from UTC-11 to UTC+11. BEAM gives no question date; its questions are asked about the
whole conversation, so each is asked at 12:00 UTC the day after the conversation's last session.

The probing questions are a Python literal keyed by ability, 40 questions per ability over the split. Every
ability but abstention carries turn-level evidence: ``source_chat_ids`` names chat ids as a list, as nested lists,
or as a mapping whose keys give each id's role (``original_info`` and ``updated_info`` for a knowledge update,
``first_statement`` and ``second_statement`` for a contradiction, ``first_event`` and ``second_event`` for
temporal reasoning); the key becomes the evidence label. A question id is ``<conversation>:<ability>:<n>``.

Findings, by name: ``reordered`` (a conversation lists its sessions out of time order; they replay in time order)
and ``no_turn_evidence`` (a question other than an abstention question names no evidence turn).
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import pinned, records
from .records import UTC, Conversation, DatasetError, Evidence, Question, Session, Turn

NAME = "beam"
SPLIT = "100K"
PIN = pinned.Pin(
    name="beam-100K.parquet",
    url="https://huggingface.co/datasets/Mohammadta/BEAM/resolve/"
        "3205395e897e7318c7b094ef4e6047b9b82dbb03/data/100K-00000-of-00001.parquet",
    sha256="c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace",
    size=5429768,
)
PINNED = (PIN,)
LICENSE = "CC BY-SA 4.0"
ABILITIES = ("abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning", "preference_following",
             "summarization", "temporal_reasoning")
# The published split: 20 conversations, 2 questions per ability in each, so 40 per ability.
EXPECTED_TYPES = {name: 40 for name in ABILITIES}
EXPECTED_CONVERSATIONS = 20
ABSTENTION = "abstention"
# The field that holds the expected answer, by ability: the first one present.
ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
ANCHOR = re.compile(r"([A-Za-z]+)-(\d{1,2})-(\d{4})")
SESSION_HOUR = 12
FINDINGS = ("reordered", "no_turn_evidence")


def parse_anchor(text: Optional[str], where: str) -> datetime:
    """``March-15-2024`` as 12:00 UTC that day (month names in English, whatever the locale)."""
    match = ANCHOR.fullmatch(text or "")
    if not match or match.group(1) not in MONTHS:
        raise DatasetError(f"{where}: {text!r} is not a Month-DD-YYYY time anchor")
    month, day, year = MONTHS.index(match.group(1)) + 1, int(match.group(2)), int(match.group(3))
    try:
        return datetime(year, month, day, SESSION_HOUR, tzinfo=UTC)
    except ValueError as error:
        raise DatasetError(f"{where}: {text!r} is not a date ({error})") from None


def _chat_ids(value, label: Optional[str] = None):
    """``(chat id, label)`` for every id in any of the upstream forms."""
    if value is None:
        return
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _chat_ids(inner, str(key))
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _chat_ids(inner, label)
    elif isinstance(value, int) and not isinstance(value, bool):
        yield value, label
    else:
        raise DatasetError(f"{value!r} is not a chat id")


def _conversation(row: dict, findings: dict) -> tuple:
    cid = str(row["conversation_id"])
    listed, turns_by_id = [], {}
    for position, raw_turns in enumerate(row["chat"]):
        key = f"s{position + 1}"
        where = f"conversation {cid} session {key}"
        anchors = [entry["time_anchor"] for entry in raw_turns if entry.get("time_anchor")]
        if not anchors:
            raise DatasetError(f"{where}: no turn carries a time anchor")
        turns = []
        for index, entry in enumerate(raw_turns):
            chat_id = int(entry["id"])
            if chat_id in turns_by_id:
                raise DatasetError(f"{where}: chat id {chat_id} is used twice")
            turns_by_id[chat_id] = (key, index)
            turns.append(Turn(index=index, role=records.check_role(entry["role"], where), content=entry["content"],
                              ref=str(chat_id)))
        listed.append(Session(key=key, source_id=key, position=position, at=parse_anchor(anchors[0], where),
                              turns=tuple(turns)))
    sessions = records.in_time_order(listed)
    if [session.key for session in sessions] != [session.key for session in listed]:
        findings["reordered"].append(cid)
    return Conversation(id=cid, sessions=sessions), turns_by_id


def _question(cid: str, ability: str, number: int, item: dict, conversation: Conversation, turns_by_id: dict,
              findings: dict) -> Question:
    qid = f"{cid}:{ability}:{number}"
    order = {session.key: rank for rank, session in enumerate(conversation.sessions)}
    found = set()
    for chat_id, label in _chat_ids(item.get("source_chat_ids")):
        if chat_id not in turns_by_id:
            raise DatasetError(f"{qid}: evidence chat id {chat_id} is not a turn of conversation {cid}")
        session, index = turns_by_id[chat_id]
        found.add(Evidence(session=session, turn=index, label=label))
    evidence = tuple(sorted(found, key=lambda item: (order[item.session], item.turn, item.label or "")))
    abstention = ability == ABSTENTION
    if not evidence and not abstention:
        findings["no_turn_evidence"].append(qid)
    answer = next((item[name] for name in ANSWER_FIELDS if item.get(name) is not None), None)
    if answer is None:
        raise DatasetError(f"{qid}: no answer field ({', '.join(ANSWER_FIELDS)})")
    gold = tuple(session.key for session in conversation.sessions
                 if any(entry.session == session.key for entry in evidence))
    asked_at = (conversation.sessions[-1].at + timedelta(days=1)).replace(hour=SESSION_HOUR, minute=0, second=0)
    return Question(id=qid, conversation=cid, type=ability, question=item["question"], answer=str(answer),
                    asked_at=asked_at, date_text=None, abstention=abstention, gold_sessions=gold, evidence=evidence,
                    detail={key: value for key, value in item.items() if key not in ("question", "source_chat_ids")})


def parse(path: Path, sha256: Optional[str] = None) -> records.Dataset:
    """The dataset in ``path`` (one BEAM split as parquet)."""
    import pyarrow.parquet as pq

    findings = {name: [] for name in FINDINGS}
    conversations, questions = {}, []
    for row in pq.read_table(path, columns=["conversation_id", "chat", "probing_questions"]).to_pylist():
        conversation, turns_by_id = _conversation(row, findings)
        if conversation.id in conversations:
            raise DatasetError(f"conversation {conversation.id} appears twice")
        conversations[conversation.id] = conversation
        probing = ast.literal_eval(row["probing_questions"]) if row["probing_questions"] else {}
        if not isinstance(probing, dict):
            raise DatasetError(f"conversation {conversation.id}: the probing questions are not keyed by ability")
        for ability, items in probing.items():
            for number, item in enumerate(items):
                questions.append(_question(conversation.id, str(ability), number, item, conversation, turns_by_id,
                                           findings))
    return records.check(records.Dataset(
        name=NAME, sha256=sha256 or pinned.sha256_file(Path(path)), conversations=conversations,
        questions=tuple(questions), findings={name: tuple(ids) for name, ids in findings.items()}))


def load(directory=None, *, download: bool = True) -> records.Dataset:
    """The pinned split, fetched into the cache if needed, parsed, and checked against the published counts."""
    dataset = parse(pinned.fetch(PIN, directory, download=download), sha256=PIN.sha256)
    records.verify_counts(dataset, EXPECTED_TYPES, abstention=EXPECTED_TYPES[ABSTENTION])
    if len(dataset.conversations) != EXPECTED_CONVERSATIONS:
        raise DatasetError(f"{NAME}: {len(dataset.conversations)} conversations, the published count is "
                           f"{EXPECTED_CONVERSATIONS}")
    return dataset
