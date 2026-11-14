"""Private SQLite versions, reusing MindState for the current pointer."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path

from colonymind.mind.concerns import MindState, open_mind_db
from colonymind.util.temporal import now_utc
from .builder import BuiltCard, CardInputs


def _content_hash(owner_id: str, text: str) -> str:
    return hashlib.sha256((owner_id + "\0" + text).encode()).hexdigest()


@dataclass(frozen=True)
class Card:
    owner_id: str
    version: int
    text: str
    token_count: int
    input_fingerprint: str
    local_day: str
    built_at: str
    checked_at: str
    source_refs: tuple[str, ...]
    erasure_watermark: int
    stale: bool = False

    @property
    def etag(self) -> str:
        # Include owner binding: a rebinding cannot collide with version one.
        digest = _content_hash(self.owner_id, self.text)
        # Weak validator: card bytes are equivalent while checked_at/staleness
        # metadata in the JSON representation can advance independently.
        return f'W/"context-card-{self.version}-{digest}"'


class CardStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn = open_mind_db(path)
        self.conn.execute("PRAGMA secure_delete=ON")
        os.chmod(path, 0o600)
        self.state = MindState(self.conn)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS context_card_versions (
            owner_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(owner_id, version))""")
        self.conn.commit()
        self.current: Card | None = None
        # Prune historical copies left by the previous eight-version contract.
        pointer = self.state.get("context_card.current") or {}
        active = json.loads(pointer.get("text") or "{}")
        if active.get("active", True):
            self.conn.execute("DELETE FROM context_card_versions WHERE owner_id != ? OR version != ?",
                              (active.get("owner_id", ""), active.get("version", 0)))
        else:
            self.conn.execute("DELETE FROM context_card_versions")
        self.conn.commit()

    def binding(self) -> tuple[str, int]:
        pointer = self.state.get("context_card.current") or {}
        value = json.loads(pointer.get("text") or "{}")
        return value.get("owner_id", ""), value.get("metadata", {}).get("erasure_watermark", 0)

    def load(self, owner_id: str) -> Card | None:
        pointer = self.state.get("context_card.current")
        if not pointer:
            return None
        active = json.loads(pointer["text"])
        if not active.get("active", True) or active["owner_id"] != owner_id:
            return None
        row = self.conn.execute("SELECT payload FROM context_card_versions WHERE owner_id=? AND version=?",
                                (owner_id, active["version"])).fetchone()
        if row:
            payload = json.loads(row[0])
            payload["source_refs"] = tuple(payload["source_refs"])
            metadata = active.get("metadata", {})
            if "source_refs" in metadata:
                metadata["source_refs"] = tuple(metadata["source_refs"])
            self.current = replace(Card(**payload), **metadata)
        return self.current

    def publish(self, built: BuiltCard, inputs: CardInputs) -> Card:
        current = self.current
        stamp = now_utc().isoformat()
        pointer = self.state.get("context_card.current") or {}
        previous = json.loads(pointer.get("text") or "{}")
        sequence = previous.get("version", 0)
        version = int(sequence) + 1
        # Even after privacy cleanup, an unrelated erasure must not churn
        # unchanged prefix bytes. The pointer retains only their digest.
        if previous.get("content_hash") == _content_hash(inputs.owner_id, built.text):
            version = int(sequence)
        if current and current.owner_id == inputs.owner_id and current.text == built.text:
            version = current.version
        card = Card(inputs.owner_id, version, built.text, built.token_count, built.input_fingerprint,
                    inputs.local_day, stamp, stamp, built.source_refs, inputs.erasure_watermark)
        # Same bytes preserve their immutable published row and version. The
        # live cache carries the latest reconciliation metadata separately.
        try:
            if current and current.owner_id == inputs.owner_id and current.text == card.text:
                card = replace(card, built_at=current.built_at)
            else:
                self.conn.execute("INSERT INTO context_card_versions VALUES (?,?,?)",
                                  (card.owner_id, card.version, json.dumps(card.__dict__)))
                self.conn.execute("DELETE FROM context_card_versions WHERE owner_id != ? OR version != ?",
                                  (card.owner_id, card.version))
            # MindState.set commits the inserted version and current pointer
            # together. A failed publication keeps the prior version intact.
            self.checked(card)
        except BaseException:
            self.conn.rollback()
            raise
        return card

    def checked(self, card: Card) -> None:
        self.state.set("context_card.current", text=json.dumps({"owner_id": card.owner_id, "version": card.version,
            "active": True,
            "content_hash": _content_hash(card.owner_id, card.text),
            "metadata": {"input_fingerprint": card.input_fingerprint, "checked_at": card.checked_at,
                         "token_count": card.token_count, "erasure_watermark": card.erasure_watermark,
                         "source_refs": card.source_refs}}))
        self.current = card

    def purge(self) -> None:
        """Erase derived copies as well as withholding the active card."""
        self.conn.execute("DELETE FROM context_card_versions")
        pointer = self.state.get("context_card.current") or {}
        value = json.loads(pointer.get("text") or "{}")
        self.state.set("context_card.current", text=json.dumps({**value, "active": False}))
        self.current = None

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def delete(path: Path) -> None:
        """Remove this feature's derived database and its SQLite side files."""
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(path) + suffix).unlink(missing_ok=True)
