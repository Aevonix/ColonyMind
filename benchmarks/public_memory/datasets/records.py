"""The records every public dataset adapter produces.

A dataset is a set of conversations and the questions asked about them. A conversation is its sessions in the
order they happened (``at``, then the file's own order for a tie). A question names its conversation, the time it
is asked (``asked_at``, never before any of its sessions), the gold sessions and the gold turns (``evidence``) the
dataset marks, and whether it is an abstention question. Times are timezone-aware UTC: the datasets give local
wall times with no zone, and one fixed zone keeps a replay reproducible.

``findings`` lists, by name, the questions where the adapter found something in the upstream file worth knowing
(sessions listed out of time order, a question stamped before a session of its own day); each adapter documents
its names. The upstream data is never changed beyond what those findings record.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Tuple

UTC = timezone.utc
ROLES = ("user", "assistant")


class DatasetError(ValueError):
    """The upstream file is not what the adapter reads, or breaks an invariant a replay relies on."""


@dataclass(frozen=True)
class Turn:
    index: int                    # position in its session, from 0
    role: str                     # "user" or "assistant"
    content: str
    ref: Optional[str] = None     # the dataset's own id for the turn, where it has one


@dataclass(frozen=True)
class Session:
    key: str                      # unique within its conversation
    source_id: str                # the dataset's id for the session (a repeated session shares it)
    position: int                 # position in the dataset's own listing
    at: datetime                  # when the session happened, UTC
    turns: Tuple[Turn, ...]


@dataclass(frozen=True)
class Evidence:
    session: str                  # a Session.key
    turn: int                     # a Turn.index in that session
    label: Optional[str] = None   # the dataset's role for this evidence, where it gives one


@dataclass(frozen=True)
class Conversation:
    id: str
    sessions: Tuple[Session, ...]  # in time order


@dataclass(frozen=True)
class Question:
    id: str
    conversation: str
    type: str
    question: str
    answer: str
    asked_at: datetime             # UTC, never before a session of its conversation
    date_text: Optional[str]       # the question date exactly as the dataset gives it, if it gives one
    abstention: bool
    gold_sessions: Tuple[str, ...]  # Session.key values, in time order
    evidence: Tuple[Evidence, ...]  # in time order
    detail: Mapping[str, Any] = field(default_factory=dict)   # the dataset's other fields for the question


@dataclass(frozen=True)
class Dataset:
    name: str
    sha256: str
    conversations: Mapping[str, Conversation]
    questions: Tuple[Question, ...]
    findings: Mapping[str, Tuple[str, ...]]

    def counts(self) -> dict:
        return dict(Counter(item.type for item in self.questions))

    def abstention_ids(self) -> Tuple[str, ...]:
        return tuple(item.id for item in self.questions if item.abstention)


def in_time_order(sessions: Iterable[Session]) -> Tuple[Session, ...]:
    return tuple(sorted(sessions, key=lambda item: (item.at, item.position)))


def check_role(role: str, where: str) -> str:
    if role not in ROLES:
        raise DatasetError(f"{where}: role {role!r} is neither user nor assistant")
    return role


def check(dataset: Dataset) -> Dataset:
    """Refuse a dataset that breaks what a replay relies on; return it unchanged otherwise."""
    for conversation in dataset.conversations.values():
        keys = [item.key for item in conversation.sessions]
        if len(set(keys)) != len(keys):
            raise DatasetError(f"{conversation.id}: two sessions share a key")
        for earlier, later in zip(conversation.sessions, conversation.sessions[1:]):
            if later.at < earlier.at:
                raise DatasetError(f"{conversation.id}: session {later.key} is out of time order")
    ids = [item.id for item in dataset.questions]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{dataset.name}: two questions share an id")
    for item in dataset.questions:
        if item.asked_at.tzinfo is None:
            raise DatasetError(f"{item.id}: the question time has no zone")
        conversation = dataset.conversations.get(item.conversation)
        if conversation is None:
            raise DatasetError(f"{item.id}: no conversation {item.conversation}")
        sessions = {session.key: session for session in conversation.sessions}
        late = [session.key for session in conversation.sessions if session.at > item.asked_at]
        if late:
            raise DatasetError(f"{item.id}: session {late[0]} falls after the question is asked")
        for key in item.gold_sessions:
            if key not in sessions:
                raise DatasetError(f"{item.id}: gold session {key} is not in its conversation")
        for evidence in item.evidence:
            session = sessions.get(evidence.session)
            if session is None or not 0 <= evidence.turn < len(session.turns):
                raise DatasetError(f"{item.id}: evidence {evidence.session}:{evidence.turn} is not a turn")
    return dataset


def verify_counts(dataset: Dataset, types: Mapping[str, int], abstention: Optional[int] = None) -> None:
    """Refuse a dataset whose questions per type (and abstention questions) are not the published counts."""
    counts = dataset.counts()
    for name in sorted(set(counts) | set(types)):
        if counts.get(name, 0) != types.get(name, 0):
            raise DatasetError(f"{dataset.name}: {counts.get(name, 0)} {name} questions, "
                               f"the published count is {types.get(name, 0)}")
    if abstention is not None and len(dataset.abstention_ids()) != abstention:
        raise DatasetError(f"{dataset.name}: {len(dataset.abstention_ids())} abstention questions, "
                           f"the published count is {abstention}")
