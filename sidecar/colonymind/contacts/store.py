"""ColonyMind Contacts — SQLite-backed ContactStore."""

from __future__ import annotations

import json
import asyncio
import hashlib
import inspect
import logging
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite

from .config import ContactsConfig
from .models import (
    Contact,
    ContactHandle,
    MAY_CONTACT,
    ScopeMember,
    TrustScope,
    TRUST_TIERS,
    regular_or_above,
    tier_rank,
)

logger = logging.getLogger("colonymind.contacts.store")

_SCHEMA_FILE = Path(__file__).parent / "migrations" / "001_contacts_schema.sql"
_SCHEMA_FILE_002 = Path(__file__).parent / "migrations" / "002_trust_scopes.sql"
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_PHONE_DIGITS = re.compile(r'\D')


def _gen_id(prefix: str) -> str:
    import secrets
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(6)  # 12 hex chars, 48 bits of CSPRNG entropy
    return f"{prefix}-{ts}-{rand}"


def _now_iso() -> str:
    """UTC now from ``time.time``: the clock the mind compares contact stamps with. ``datetime.now``
    reads the system clock directly, so a body that shifts ``time.time`` would split the two."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_phone(phone: str) -> str:
    """Strip non-digit characters; preserve leading +."""
    phone = phone.strip()
    if phone.startswith("+"):
        return "+" + _PHONE_DIGITS.sub("", phone[1:])
    return _PHONE_DIGITS.sub("", phone)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _phone_key(address: str) -> str:
    """Canonical phone digits, retaining international country codes.

    Only a bare ten-digit string (a legacy row stored without its ``+1``) is read as NANP. A phone
    JID (``<number>@s.whatsapp.net``) always carries its country code, so ``6595550117@...`` is
    +65 0000 0100, never +1 659 555 0117; an E.164 address carries its own."""
    text = (address or "").strip()
    digits = _PHONE_DIGITS.sub("", text.split('@', 1)[0])
    return '1' + digits if len(digits) == 10 and not text.startswith('+') and '@' not in text else digits


# C1: a phone number is ONE identity on every gateway. The rule is derived from the handle's
# format, never from a channel name: an address written as an international number (``+`` and 7
# to 15 digits, separators allowed) or a messaging id whose host says its local part is a phone
# number (``<number>@s.whatsapp.net``, ``<number>@c.us``) matches on ``phone_key`` whatever
# gateway it arrived on. A bare digit string is not a phone number by its format (numeric user
# ids are common, and ``<number>@lid`` is an opaque id): it matches exactly, on its own gateway.
_E164 = re.compile(r"^\+[1-9]\d{6,14}$")
_PHONE_JID = re.compile(r"^[1-9]\d{6,14}@(s\.whatsapp\.net|c\.us)$", re.IGNORECASE)
_BARE_DIGITS = re.compile(r"^\d{7,15}$")
_PHONE_SEPARATORS = re.compile(r"[\s().\-]")
_CONVERSATION_GAP = timedelta(minutes=30)   # C3: a longer silence starts a new conversation


def is_e164(address: Optional[str]) -> bool:
    """True when ``address`` is written as a phone number: E.164 with its ``+``, or a phone JID."""
    text = (address or "").strip()
    if "@" in text:
        return bool(_PHONE_JID.match(text))
    return bool(_E164.match(_PHONE_SEPARATORS.sub("", text)))


def canonical_handle(gateway: Optional[str], address: Optional[str]) -> Tuple[str, str]:
    """The identity a handle matches on: ``("email", lower)``, ``("phone", "+<digits>")`` for an
    address written as a phone number on any gateway, else the exact ``(gateway, address)``."""
    g, a = (gateway or "").strip().lower(), (address or "").strip()
    if g == "email":
        return "email", _normalize_email(a)
    if is_e164(a):
        return "phone", "+" + _phone_key(a)
    return g, a


def stored_handle(gateway: Optional[str], address: Optional[str]) -> Tuple[str, str]:
    """The form a handle is stored and looked up exactly in: the transport gateway (needed for
    sending) with a lower-cased email, a separator-free number, or the address as given (a
    ``number@host`` id keeps its transport form)."""
    kind, key = canonical_handle(gateway, address)
    g, a = (gateway or "").strip().lower(), (address or "").strip()
    if kind == "email":
        return g, key
    if (kind == "phone" and "@" not in a) or _BARE_DIGITS.match(_PHONE_SEPARATORS.sub("", a)):
        return g, _normalize_phone(a)
    return g, a


def _cadence(minutes: Any) -> Optional[int]:
    """A cadence is a positive whole number of minutes, or None for no cadence."""
    if minutes is None:
        return None
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float, str)):
        raise ValueError("cadence_minutes must be a positive number of minutes or null")
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        raise ValueError("cadence_minutes must be a positive number of minutes or null") from None
    if value <= 0 or value > 60 * 24 * 366:
        raise ValueError("cadence_minutes must be a positive number of minutes or null")
    return value


def _name_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Simple character bigram similarity in [0, 1]."""
    if not a or not b:
        return 0.0
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0

    def bigrams(s):
        return {s[i:i+2] for i in range(len(s) - 1)}

    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    return 2.0 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))


# ── Abstract interface ────────────────────────────────────────────────────────

class ContactChanged(ValueError):
    """A write made against what a contact was when read (``expected``) found it changed since: nothing
    was written. The caller decides again on the record as it is now."""


_UNREAD = object()   # no expectation: the write applies to the record as it is
# The handle set a compare-and-set also holds a contact to (``expected_handles``, ``gateway:address``): the
# same count, and none that is not one of them. One statement with the write, so nothing lands in between.
_SAME_HANDLES = ("(SELECT COUNT(*) FROM contact_handles WHERE contact_id = ?) = ? AND NOT EXISTS (SELECT 1 FROM "
                 "contact_handles WHERE contact_id = ? AND gateway || ':' || address NOT IN "
                 "(SELECT value FROM json_each(?)))")


def _same_handles(contact_id: str, expected_handles: Optional[Sequence[str]]) -> tuple[str, tuple]:
    """The SQL condition (and its parameters) that the contact holds exactly ``expected_handles``; empty
    without an expectation."""
    if expected_handles is None:
        return "", ()
    handles = sorted({str(handle) for handle in expected_handles})
    return " AND " + _SAME_HANDLES, (contact_id, len(handles), contact_id, json.dumps(handles))


class ContactStore(ABC):
    """Primary read/write interface for the contact store."""

    @abstractmethod
    async def get(self, contact_id: str) -> Optional[Contact]:
        """Fetch a contact by canonical ID. Returns None if not found or deleted."""

    @abstractmethod
    async def resolve_handle(self, gateway: str, address: str) -> Optional[Contact]:
        """Resolve a gateway handle to a Contact."""

    async def resolve_verified_handles(self, gateway: str, addresses: List[str]) -> Optional[Contact]:
        """Resolve exact transport aliases only when they identify one verified contact."""
        raise NotImplementedError

    @abstractmethod
    async def create(
        self,
        display_name: Optional[str] = None,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        organization: Optional[str] = None,
        trust_tier: str = "unknown",
        may_contact: str = "ask",
        cadence_minutes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        privacy_level: str = "private",
        import_source: str = "manual",
        notes: Optional[str] = None,
    ) -> Contact:
        """Create a new contact record. A tier implies no permission: ``may_contact`` starts at ``ask``."""

    @abstractmethod
    async def add_handle(
        self,
        contact_id: str,
        gateway: str,
        address: str,
        is_primary: bool = False,
        confidence: float = 1.0,
        source: str = "manual",
        verified: bool = False,
    ) -> ContactHandle:
        """Add a gateway handle to an existing contact."""

    @abstractmethod
    async def get_handles(self, contact_id: str) -> List[ContactHandle]:
        """Return all handles for a contact."""

    @abstractmethod
    async def update_tier(
        self,
        contact_id: str,
        new_tier: str,
        reason: Optional[str] = None,
        performed_by: str = "operator",
    ) -> None:
        """Update a contact's trust tier and record the change in audit."""

    @abstractmethod
    async def set_may_contact(self, contact_id: str, value: str, *, by: str, reason: str = "",
                              expected: Any = _UNREAD, expected_handles: Optional[Sequence[str]] = None) -> Contact:
        """The owner's path: move ``may_contact`` in any direction, audited; with ``expected`` only from that
        stored value, and with ``expected_handles`` only while the contact holds exactly those handles
        (``ContactChanged`` otherwise)."""

    @abstractmethod
    async def lower_may_contact(self, contact_id: str, *, reason: str, source_ref: str) -> Optional[Contact]:
        """A contact's opt-out: to ``never`` only; None when already there."""

    @abstractmethod
    async def soft_delete(
        self, contact_id: str, reason: Optional[str] = None, performed_by: str = "operator"
    ) -> None:
        """Soft-delete a contact."""

    @abstractmethod
    async def hard_delete(self, contact_id: str, performed_by: str = "system") -> None:
        """Permanently delete a contact record."""

    @abstractmethod
    async def list(
        self,
        trust_tier: Optional[str] = None,
        may_contact: Optional[str] = None,
        tag: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Contact]:
        """List contacts with optional filtering."""

    @abstractmethod
    async def find_by_name(self, name: str, threshold: float = 0.5) -> List[Contact]:
        """Find contacts whose display_name is similar to name."""

    @abstractmethod
    async def merge(self, keep_id: str, drop_id: str, *, performed_by: str, reattribute=None,
                    sources_of=None, expected: Optional[Dict[str, Any]] = None) -> Contact:
        """C2: fold ``drop_id`` into ``keep_id`` through the identity-correction path (handles and
        their sources), fold recency, permission, cadence and digest, re-attribute the other
        stores through ``reattribute`` hooks, soft-delete the dropped record. Audited both sides."""

    @abstractmethod
    async def list_handle_proposals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Pending handle-link proposals (from scoped-name attribution) for
        owner review: {contact_id, display_name, gateway, address, at}."""

    @abstractmethod
    async def update(self, contact_id: str, **fields) -> Optional[Contact]:
        """Update arbitrary fields on a contact."""

    @abstractmethod
    async def record_audit(
        self,
        contact_id: str,
        action: str,
        detail: Optional[Dict[str, Any]] = None,
        performed_by: str = "system",
    ) -> None:
        """Write an audit record."""

    @abstractmethod
    async def connect(self) -> None:
        """Open the database connection."""

    @abstractmethod
    async def close(self) -> None:
        """Close the database connection."""

    async def __aenter__(self) -> "ContactStore":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


# ── SQLite implementation ─────────────────────────────────────────────────────

class SQLiteContactStore(ContactStore):
    """SQLite-backed implementation of ContactStore."""

    def __init__(
        self,
        config: Optional[ContactsConfig] = None,
        *,
        sources_of: Optional[Callable[[str], Awaitable[Iterable[str]]]] = None,
        reattribute: Iterable[Callable[[str, str], Any]] = (),
    ) -> None:
        self._config = config or ContactsConfig()
        self._db: Optional[aiosqlite.Connection] = None
        self._location_lock = asyncio.Lock()
        # Injected by the server: the source ids a contact holds in the ledger (moved with a merge
        # through the identity-correction receipts) and the other stores' ``reattribute(old, new)``.
        self.sources_of = sources_of
        self.reattribute = list(reattribute)

    async def connect(self) -> None:
        path = self._config.sqlite_path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(path)
        self._db.row_factory = aiosqlite.Row
        # In-query phone identity key so messaging resolution matches a number to its contact even
        # when stored handles are formatted differently / under a different gateway -- works without
        # a data migration (the stored side is keyed at query time).
        await self._db.create_function(
            "phone_key", 1, lambda v: _phone_key(v) if v else "")
        await self._db.create_function("is_phone", 1, lambda v: 1 if is_e164(v) else 0)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        from colonymind.migrations import run_migrations
        await run_migrations(self._db, _MIGRATIONS_DIR)
        await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        """Idempotent additive migrations for DBs created before a column existed.

        SQLite has no ADD COLUMN IF NOT EXISTS, so we introspect first.
        """
        db = self._db
        assert db is not None
        async with db.execute("PRAGMA table_info(contacts)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if "timezone" not in cols:  # v0.21.0 — per-contact timezone
            await db.execute("ALTER TABLE contacts ADD COLUMN timezone TEXT")
            await db.commit()
        # Introduction provenance (social-graph autonomy): who introduced this
        # contact + how/where the agent met them. First-class on the contact so
        # it is always available even when no world-model Person node exists yet.
        if "introduced_by" not in cols:
            await db.execute("ALTER TABLE contacts ADD COLUMN introduced_by TEXT")
            await db.commit()
        if "met_via_json" not in cols:
            await db.execute("ALTER TABLE contacts ADD COLUMN met_via_json TEXT")
            await db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if not self._db:
            raise RuntimeError("ContactStore not connected. Use async with or call connect().")
        return self._db

    async def _open_durable_connection(self) -> aiosqlite.Connection:
        """Open one dedicated durable connection for a provisioning command.

        Provisioning promises crash-safe idempotency.  A process-local memory
        database cannot provide that promise, and sharing ``self._db`` would
        allow an unrelated coroutine's commit to persist a partial operation.
        """

        raw_path = str(self._config.sqlite_path or "").strip()
        if (
            not raw_path
            or raw_path == ":memory:"
            or raw_path.startswith("file:")
        ):
            raise RuntimeError(
                "contact provisioning requires a durable file-backed store"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(
                "contact provisioning durable store is unavailable"
            )
        db = await aiosqlite.connect(str(path), timeout=5.0)
        db.row_factory = aiosqlite.Row
        await db.create_function(
            "phone_key", 1, lambda value: _phone_key(value) if value else ""
        )
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=5000")
        return db

    # ── Read ops ──────────────────────────────────────────────────────────────

    async def propose_handle_link(self, contact_id, gateway, address, *, evidence_refs=(), source='auto:scoped-name'):
        from .identity_links import propose
        return await propose(self, contact_id=contact_id, gateway=gateway, address=address,
                             evidence_refs=evidence_refs, source=source)

    async def correct_handle_identity(self, *, operation_id, performed_by, gateway, address,
                                      expected_contact_id, contact_id, evidence_refs,
                                      affected_source_ids=()):
        from .identity_links import correct, normalized_handle, _json, _refs
        given_address = address
        gateway, address = normalized_handle(gateway, address)
        kind, key = canonical_handle(gateway, address)
        if kind != 'phone' or contact_id is None:
            return await correct(self, operation_id=operation_id, performed_by=performed_by,
                gateway=gateway, address=given_address, expected_contact_id=expected_contact_id,
                contact_id=contact_id, evidence_refs=evidence_refs, affected_source_ids=affected_source_ids)

        # Keep the exact correction's preimage, receipt and retry contract, but move the whole
        # phone in its transaction. Calling correct once per handle could commit a split number.
        refs, sources = _refs(evidence_refs), _refs(affected_source_ids)
        if not refs:
            raise ValueError('identity_correction_requires_evidence')
        for value in (operation_id, performed_by):
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError('invalid_identity_correction')
        for value in (expected_contact_id, contact_id):
            if value is not None and (not isinstance(value, str) or not value.startswith('cid-') or len(value) > 256):
                raise ValueError('invalid_identity_contact')
        request_hash = hashlib.sha256(_json([operation_id, performed_by, gateway, address,
            expected_contact_id, contact_id, refs, sources]).encode()).hexdigest()
        db = await self._open_durable_connection()
        try:
            await db.execute('BEGIN IMMEDIATE')
            async with db.execute('SELECT * FROM contact_identity_operations WHERE operation_id=?', (operation_id,)) as cur:
                prior = await cur.fetchone()
            if prior:
                if prior['request_sha256'] != request_hash:
                    raise ValueError('identity_operation_conflict')
                await db.rollback()
                return json.loads(prior['result_json'])
            async with db.execute('SELECT * FROM contact_handles WHERE gateway=? AND address=?', (gateway, address)) as cur:
                handle = await cur.fetchone()
            old_id = handle['contact_id'] if handle else None
            if old_id != expected_contact_id:
                raise ValueError('identity_preimage_changed')
            async with db.execute('SELECT contact_id FROM contacts WHERE contact_id=? AND deleted_at IS NULL', (contact_id,)) as cur:
                if await cur.fetchone() is None:
                    raise ValueError('identity_contact_not_found')
            await db.create_function('is_phone', 1, lambda value: int(is_e164(value)))
            await db.create_function('is_bare_phone', 1,
                lambda value: bool(_BARE_DIGITS.fullmatch(_PHONE_SEPARATORS.sub('', value or ''))))
            phone = 'phone_key(address)=? AND (is_phone(address) OR (gateway=? AND is_bare_phone(address)))'
            params = (key[1:], gateway)
            async with db.execute('SELECT handle_id,contact_id FROM contact_handles WHERE ' + phone, params) as cur:
                moved = [dict(row) for row in await cur.fetchall()]
            now = _now_iso()
            await db.execute('''UPDATE contact_handles SET contact_id=?, verified=1,
                confidence=1,source='owner_operator',is_primary=CASE WHEN contact_id=? THEN is_primary ELSE 0 END
                WHERE ''' + phone, (contact_id, contact_id, *params))
            handle_id = handle['handle_id'] if handle else _gen_id('hdl')
            if handle is None:
                await db.execute('''INSERT INTO contact_handles
                    (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
                    VALUES (?,?,?,?,0,1,1,'owner_operator',?)''', (handle_id, contact_id, gateway, address, now))
            async with db.execute('SELECT DISTINCT gateway FROM contact_handles WHERE phone_key(address)=?',
                                  (key[1:],)) as cur:
                gateways = [row['gateway'] for row in await cur.fetchall()]
            conflicts = set()
            for channel in gateways:
                if channel.strip().lower() == 'email':
                    continue
                matches = await self._match_canonical(channel, key, all_matches=True, conn=db)
                if len({row['contact_id'] for row in matches}) > 1:
                    conflicts.update(f"{row['gateway']}:{row['handle_id']}:{row['contact_id']}" for row in matches
                                     if row['contact_id'] != contact_id)
            if conflicts:
                raise ValueError('identity_phone_ambiguous ' + ', '.join(sorted(conflicts)))
            await db.execute('''UPDATE contact_identity_candidates SET status=CASE WHEN contact_id=?
                THEN 'confirmed' ELSE 'rejected' END,resolved_at=? WHERE status='pending' AND ''' + phone,
                (contact_id, now, *params))
            result = {'schema': 'IdentityCorrectionV1', 'operation_id': operation_id,
                'old_contact_id': old_id, 'contact_id': contact_id, 'handle_id': handle_id,
                'gateway': gateway, 'address_sha256': hashlib.sha256(address.encode()).hexdigest(),
                'evidence_refs': refs, 'affected_source_ids': sources, 'recorded_at': now,
                'performed_by': performed_by, 'authority_granted': False,
                'source_reconciliation_required': bool(sources), 'moved_handles': moved}
            for cid in sorted({contact_id, *(row['contact_id'] for row in moved)}):
                await db.execute('''INSERT INTO contact_audit
                    (id,contact_id,action,detail,performed_by,created_at) VALUES (?,?,?,?,?,?)''',
                    (_gen_id('aud'), cid, 'identity_corrected', _json(result), performed_by, now))
            await db.execute('INSERT INTO contact_identity_operations VALUES (?,?,?,?)',
                (operation_id, request_hash, _json(result), now))
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()
        return result

    async def pending_identity_reconciliations(self, *, limit=100):
        from .identity_links import pending_reconciliations
        return await pending_reconciliations(self, limit=limit)

    async def mark_sources_reconciled(self, operation_id, source_result):
        from .identity_links import mark_sources_reconciled
        return await mark_sources_reconciled(self, operation_id=operation_id, source_result=source_result)

    async def mark_sources_conflicted(self, operation_id, code):
        from .identity_links import mark_sources_conflicted
        return await mark_sources_conflicted(self, operation_id=operation_id, code=code)

    async def identity_evidence(self, contact_id):
        from .identity_links import evidence
        return await evidence(self, contact_id)

    async def get(self, contact_id: str) -> Optional[Contact]:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM contacts WHERE contact_id = ? AND deleted_at IS NULL",
            (contact_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return Contact.from_row(dict(row))

    async def resolve_handle(self, gateway: str, address: str) -> Optional[Contact]:
        """The one live contact behind a handle's canonical identity (C1): an E.164 number on any
        gateway matches on ``phone_key``, an email on its lower-cased form, anything else on the
        exact gateway and address. Ambiguity is None, never a guess."""
        return await self._match_canonical(gateway, address)

    async def resolve_messaging_handle(self, gateway: str, address: str) -> Optional[Contact]:
        """Resolve phone and email identities canonically; opaque aliases match their transport.

        Explicit owner bindings take precedence, as on other owner channels.
        Otherwise a number held by two live contacts is ambiguous on every gateway.
        This does not grant a contact any authority.
        """
        g, stored = stored_handle(gateway, address)
        if not stored:
            return None
        from colonymind.identity import get_owner_contact_id
        owner_id = get_owner_contact_id()
        if g in {"sms", "rcs"}:
            from colonymind.config import ConfigError, owner_phone_handle
            try:
                configured = owner_phone_handle(g)
            except ConfigError:
                configured = ""
            # Local configuration is an explicit transport binding, never a durable phone alias.
            if owner_id and configured and (address or "").strip() == configured:
                return await self.get(owner_id)
        # Preserve an explicit stored owner binding even in an older split store.
        if owner_id and g:
            async with self._require_db().execute(
                "SELECT c.* FROM contacts c JOIN contact_handles h ON h.contact_id = c.contact_id "
                "WHERE c.deleted_at IS NULL AND (h.verified=1 OR h.source!='auto:scoped-name') "
                "AND c.contact_id = ? AND h.gateway = ? AND h.address = ?", (owner_id, g, stored),
            ) as cur:
                owner = await cur.fetchone()
            if owner is not None:
                return Contact.from_row(dict(owner))
        if g and canonical_handle(g, address)[0] not in {"phone", "email"}:
            exact = await self.resolve_verified_handles(g, [stored])
            if exact is None:
                exact = await self._match_exact(g, stored)
            if exact is not None:
                return exact
        inferred = await self._match_canonical(g, address)
        # Another platform's number alone does not configure an owner phone channel.
        if g in {"sms", "rcs"} and inferred is not None and inferred.contact_id == owner_id:
            return None
        return inferred

    async def _match_exact(self, gateway: str, address: str) -> Optional[Contact]:
        """The live contact holding this transport handle (never a name guess)."""
        if canonical_handle(gateway, address)[0] in {"phone", "email"}:
            return await self._match_canonical(gateway, address)
        async with self._require_db().execute(
            "SELECT c.* FROM contacts c JOIN contact_handles h ON h.contact_id = c.contact_id "
            "WHERE c.deleted_at IS NULL AND (h.verified=1 OR h.source!='auto:scoped-name') "
            "AND h.gateway = ? AND h.address = ?", (gateway, address),
        ) as cur:
            row = await cur.fetchone()
        return Contact.from_row(dict(row)) if row else None

    async def _match_canonical(self, gateway: str, address: str, *, all_matches: bool = False,
                               conn: Optional[aiosqlite.Connection] = None) -> Optional[Contact] | list[aiosqlite.Row]:
        db = conn if conn is not None else self._require_db()
        kind, key = canonical_handle(gateway, address)
        if not key:
            return [] if all_matches else None
        base = ("SELECT c.* FROM contacts c JOIN contact_handles h ON h.contact_id = c.contact_id "
                "WHERE c.deleted_at IS NULL AND (h.verified=1 OR h.source!='auto:scoped-name') AND ")
        g = (gateway or "").strip().lower()
        if kind == "email":
            sql, params = base + "h.gateway = 'email' AND lower(h.address) = ?", (key,)
        elif kind == "phone":
            # Every handle written as a phone number, and a legacy bare-digit row on the sender's
            # own gateway; never a numeric id on another gateway that happens to share the digits.
            sql, params = base + "phone_key(h.address) = ? AND (is_phone(h.address) OR h.gateway = ?)", (key[1:], g)
        elif _BARE_DIGITS.match(_PHONE_SEPARATORS.sub("", key)):
            sql, params = base + "h.gateway = ? AND phone_key(h.address) = ?", (g, _phone_key(key))
        else:
            sql, params = base + "h.gateway = ? AND h.address = ?", stored_handle(gateway, address)
        if all_matches:
            sql = sql.replace("SELECT c.*", "SELECT c.*, h.gateway, h.handle_id", 1)
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        if all_matches:
            return rows
        # Multiple matching contacts are ambiguous, even on one phone number.
        row = rows[0] if len({r['contact_id'] for r in rows}) == 1 else None
        return Contact.from_row(dict(row)) if row else None

    async def is_ambiguous_handle(self, gateway: str, address: str) -> bool:
        rows = await self._match_canonical(gateway, address, all_matches=True)
        return len({r['contact_id'] for r in rows}) > 1

    async def resolve_verified_handles(self, gateway: str, addresses: List[str]) -> Optional[Contact]:
        """Exact transport aliases only, when they identify one verified contact."""
        values = list(dict.fromkeys(stored_handle(gateway, a)[1] for a in addresses if isinstance(a, str)))
        if not 1 <= len(values) <= 5 or any(not 1 <= len(v) <= 512 for v in values):
            raise ValueError('bounded_exact_transport_aliases_required')
        db = self._require_db()
        async with db.execute('SELECT DISTINCT c.* FROM contacts c JOIN contact_handles h '
            'ON h.contact_id=c.contact_id WHERE c.deleted_at IS NULL AND h.verified=1 '
            'AND h.gateway=? AND h.address IN (' + ','.join('?' for _ in values) + ')',
            [(gateway or "").strip().lower(), *values]) as cursor:
            rows = await cursor.fetchall()
        if len(rows) != 1:
            return None
        for address in values:
            if canonical_handle(gateway, address)[0] in {"phone", "email"}:
                canonical = await self._match_canonical(gateway, address)
                if canonical is None or canonical.contact_id != rows[0]['contact_id']:
                    return None
        return Contact.from_row(dict(rows[0]))

    async def get_handles(self, contact_id: str) -> List[ContactHandle]:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM contact_handles WHERE contact_id = ? ORDER BY is_primary DESC, created_at",
            (contact_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [ContactHandle.from_row(dict(r)) for r in rows]

    async def list(
        self,
        trust_tier: Optional[str] = None,
        may_contact: Optional[str] = None,
        tag: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Contact]:
        db = self._require_db()
        clauses = []
        params: List[Any] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if trust_tier:
            clauses.append("trust_tier = ?")
            params.append(trust_tier)
        if may_contact is not None:
            clauses.append("may_contact = ?")
            params.append(may_contact)
        if tag:
            # SQL-02: escape LIKE wildcards to prevent contact enumeration
            safe_tag = tag.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
            clauses.append("tags_json LIKE ? ESCAPE '\\'")
            params.append(f'%"{safe_tag}"%')
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        async with db.execute(
            f"SELECT * FROM contacts {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [Contact.from_row(dict(r)) for r in rows]

    async def search(self, query: str, *, limit: int = 20) -> List[Contact]:
        """``who``: the contact the text names (``resolve_reference``) first, then live contacts whose
        name, organization or a handle address contains it, the most recently talked to first. A
        phone number, a text with an at sign or a ``gateway:address`` of a number or an email answers
        that contact alone when it names one, and nobody when two contacts hold the number or the
        email; when it names nobody (``@example.com``, the first digits of a number) it is searched
        for like any other text."""
        text = (query or "").strip()
        limit = max(1, min(int(limit), 100))
        if not text:
            return await self.list(limit=limit)
        found: List[Contact] = []
        exact = await self.resolve_reference(text)
        gateway, sep, address = text.partition(":")
        if (is_e164(text) or ("@" in text and not sep)
                or (sep and canonical_handle(gateway, address)[0] in {"phone", "email"})):
            if exact is not None:
                return [exact]
            identity = (gateway, address) if sep else ("" if is_e164(text) else "email", text)
            if await self.is_ambiguous_handle(*identity):
                return []
        if exact is not None:
            found.append(exact)
        pattern = "%" + text.lower().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_") + "%"
        db = self._require_db()
        async with db.execute(
            "SELECT DISTINCT c.* FROM contacts c LEFT JOIN contact_handles h ON h.contact_id = c.contact_id "
            "WHERE c.deleted_at IS NULL AND (lower(coalesce(c.display_name, '')) LIKE ? ESCAPE '\\' "
            "OR lower(coalesce(c.given_name, '')) LIKE ? ESCAPE '\\' OR lower(coalesce(c.family_name, '')) LIKE ? "
            "ESCAPE '\\' OR lower(coalesce(c.organization, '')) LIKE ? ESCAPE '\\' "
            "OR lower(coalesce(h.address, '')) LIKE ? ESCAPE '\\') "
            "ORDER BY c.last_interaction_at IS NULL, c.last_interaction_at DESC, c.created_at DESC LIMIT ?",
            (pattern, pattern, pattern, pattern, pattern, limit),
        ) as cur:
            rows = await cur.fetchall()
        seen = {c.contact_id for c in found}
        found += [Contact.from_row(dict(r)) for r in rows if r["contact_id"] not in seen]
        return found[:limit]

    async def find_by_name(self, name: str, threshold: float = 0.5) -> List[Contact]:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM contacts WHERE deleted_at IS NULL",
        ) as cur:
            rows = await cur.fetchall()
        results = []
        for row in rows:
            c = Contact.from_row(dict(row))
            sim = _name_similarity(name, c.display_name)
            if sim >= threshold:
                results.append((sim, c))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results]

    # ── Write ops ─────────────────────────────────────────────────────────────

    async def create(
        self,
        display_name: Optional[str] = None,
        given_name: Optional[str] = None,
        family_name: Optional[str] = None,
        organization: Optional[str] = None,
        trust_tier: str = "unknown",
        may_contact: str = "ask",
        cadence_minutes: Optional[int] = None,
        tags: Optional[List[str]] = None,
        privacy_level: str = "private",
        import_source: str = "manual",
        notes: Optional[str] = None,
        introduced_by: Optional[str] = None,
        met_via: Optional[Dict[str, Any]] = None,
    ) -> Contact:
        db = self._require_db()
        if trust_tier not in TRUST_TIERS:
            raise ValueError(f"Invalid trust_tier: {trust_tier}")
        if may_contact not in MAY_CONTACT:
            raise ValueError(f"Invalid may_contact: {may_contact!r} (never|ask|auto)")
        cadence_minutes = _cadence(cadence_minutes)
        contact_id = _gen_id("cid")
        now = _now_iso()
        dn = display_name
        if not dn and (given_name or family_name):
            dn = " ".join(p for p in [given_name, family_name] if p)
        await db.execute(
            """
            INSERT INTO contacts
              (contact_id, display_name, given_name, family_name, organization,
               trust_tier, may_contact, cadence_minutes, tags_json, privacy_level,
               import_source, notes, introduced_by, met_via_json,
               first_seen_at, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                contact_id, dn, given_name, family_name, organization,
                trust_tier, may_contact, cadence_minutes,
                json.dumps(tags or []), privacy_level,
                import_source, notes, introduced_by,
                json.dumps(met_via) if met_via else None,
                now, now, now,
            ),
        )
        await db.commit()
        audit = {"import_source": import_source}
        if introduced_by:
            audit["introduced_by"] = introduced_by
        await self.record_audit(contact_id, "created", audit)
        contact = await self.get(contact_id)
        assert contact is not None
        return contact

    async def record_introduction(
        self,
        contact_id: str,
        introduced_by: Optional[str] = None,
        met_via: Optional[Dict[str, Any]] = None,
    ) -> Optional[Contact]:
        """Annotate an EXISTING contact with introduction provenance.

        Used when an intro names someone ColonyMind already knows: we record who
        introduced them / how they were met without duplicating the contact, and
        do NOT touch their trust_tier or may_contact (an intro never
        grants standing). Only fills blanks — an existing introduced_by/met_via
        is preserved (first introduction wins).
        """
        db = self._require_db()
        existing = await self.get(contact_id)
        if existing is None:
            return None
        set_parts, params, details = [], [], {}
        if introduced_by and not existing.introduced_by:
            set_parts.append("introduced_by = ?")
            params.append(introduced_by)
            details["introduced_by"] = introduced_by
        if met_via and not existing.met_via:
            set_parts.append("met_via_json = ?")
            params.append(json.dumps(met_via))
            details["met_via"] = met_via
        if not set_parts:
            return existing
        set_parts.append("updated_at = ?")
        params.append(_now_iso())
        params.append(contact_id)
        await db.execute(
            f"UPDATE contacts SET {', '.join(set_parts)} WHERE contact_id = ?",
            params,
        )
        await db.commit()
        await self.record_audit(contact_id, "introduction_recorded", details)
        return await self.get(contact_id)

    async def add_handle(
        self,
        contact_id: str,
        gateway: str,
        address: str,
        is_primary: bool = False,
        confidence: float = 1.0,
        source: str = "manual",
        verified: bool = False,
    ) -> ContactHandle:
        db = self._require_db()
        gateway, address = stored_handle(gateway, address)
        if not gateway or not address:
            raise ValueError("a handle needs a gateway and an address")

        phone = canonical_handle(gateway, address)[0] == "phone"
        if not phone and _BARE_DIGITS.match(address):
            # Bare digits are a phone number where a live contact holds the same number, written as
            # one, on this gateway. Anywhere else they are a numeric id and match exactly.
            held = await self._match_canonical("", "+" + _phone_key(address), all_matches=True)
            phone = any(row["gateway"] == gateway for row in held)
        if phone:
            canonical = await self._match_canonical(gateway, address)
            if canonical is not None and canonical.contact_id != contact_id:
                raise ValueError(f"Phone identity is already assigned to contact {canonical.contact_id}")
            if source == "auto:sender" and canonical is None and await self.is_ambiguous_handle(gateway, address):
                raise ValueError("Phone identity is ambiguous; correct the whole number first")

        # An exact handle also belongs to one live contact.
        async with db.execute(
            "SELECT h.handle_id, h.contact_id, c.deleted_at FROM contact_handles h "
            "JOIN contacts c ON c.contact_id = h.contact_id WHERE h.gateway = ? AND h.address = ?",
            (gateway, address),
        ) as cur:
            existing = await cur.fetchone()
        if existing is not None and existing["contact_id"] != contact_id and existing["deleted_at"] is None:
            raise ValueError(
                f"Handle ({gateway}, {address}) is already assigned to contact {existing['contact_id']}"
            )
        if existing is not None and existing["contact_id"] == contact_id:
            async with db.execute("SELECT * FROM contact_handles WHERE handle_id = ?",
                                  (existing["handle_id"],)) as cur:
                row = await cur.fetchone()
            return ContactHandle.from_row(dict(row))

        now = _now_iso()
        if existing is not None:
            # Held only by a deleted record: the sender is back, so the handle is theirs again.
            handle_id = existing["handle_id"]
            await db.execute(
                "UPDATE contact_handles SET contact_id = ?, is_primary = ?, verified = ?, confidence = ?, "
                "source = ?, created_at = ? WHERE handle_id = ?",
                (contact_id, 1 if is_primary else 0, 1 if verified else 0, confidence, source, now, handle_id),
            )
        else:
            handle_id = _gen_id("hdl")
            await db.execute(
                """
                INSERT INTO contact_handles
                  (handle_id, contact_id, gateway, address, is_primary, verified, confidence, source, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (handle_id, contact_id, gateway, address, 1 if is_primary else 0,
                 1 if verified else 0, confidence, source, now),
            )
        await db.commit()
        audit = {"gateway": gateway, "address": address, "source": source}
        if existing is not None:
            audit["released_from"] = existing["contact_id"]
        await self.record_audit(contact_id, "handle_added", audit)
        async with db.execute(
            "SELECT * FROM contact_handles WHERE handle_id = ?", (handle_id,)
        ) as cur:
            row = await cur.fetchone()
        return ContactHandle.from_row(dict(row))

    async def update_tier(
        self,
        contact_id: str,
        new_tier: str,
        reason: Optional[str] = None,
        performed_by: str = "operator",
    ) -> None:
        db = self._require_db()
        if new_tier not in TRUST_TIERS:
            raise ValueError(f"Invalid trust_tier: {new_tier}")
        async with db.execute(
            "SELECT trust_tier FROM contacts WHERE contact_id = ?", (contact_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise ValueError(f"Contact not found: {contact_id}")
        old_tier = row["trust_tier"]
        await db.execute(
            "UPDATE contacts SET trust_tier = ?, updated_at = ? WHERE contact_id = ?",
            (new_tier, _now_iso(), contact_id),
        )
        await db.commit()
        await self.record_audit(
            contact_id, "tier_changed",
            {"old_tier": old_tier, "new_tier": new_tier, "reason": reason},
            performed_by=performed_by,
        )

    async def _permission_row(self, contact_id: str) -> aiosqlite.Row:
        db = self._require_db()
        async with db.execute(
            "SELECT may_contact, cadence_minutes FROM contacts WHERE contact_id = ? AND deleted_at IS NULL",
            (contact_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            raise ValueError(f"Contact not found: {contact_id}")
        return row

    async def set_may_contact(self, contact_id: str, value: str, *, by: str, reason: str = "",
                              expected: Any = _UNREAD, expected_handles: Optional[Sequence[str]] = None) -> Contact:
        """The owner's path (7.4): ``may_contact`` moves in any direction, audited as ``may_contact_set``.
        With ``expected`` the write is a compare-and-set against that stored value, and with
        ``expected_handles`` against the contact's handle set too, in the one statement that writes: a
        contact whose permission moved since it was read (an opt-out meanwhile), or that gained or lost a
        handle, raises ``ContactChanged`` and is left as it is. Nothing is rolled back: the statement wrote
        nothing, and the connection is shared (a rollback would undo another task's opt-out in flight)."""
        if value not in MAY_CONTACT:
            raise ValueError(f"Invalid may_contact: {value!r} (never|ask|auto)")
        current = str((await self._permission_row(contact_id))["may_contact"])
        db = self._require_db()
        if expected is _UNREAD and expected_handles is None:
            await db.execute(
                "UPDATE contacts SET may_contact = ?, updated_at = ? WHERE contact_id = ?",
                (value, _now_iso(), contact_id),
            )
        else:
            held, params = _same_handles(contact_id, expected_handles)
            same = "" if expected is _UNREAD else " AND may_contact IS ?"
            written = await db.execute(
                "UPDATE contacts SET may_contact = ?, updated_at = ? WHERE contact_id = ? AND deleted_at IS NULL"
                + same + held, (value, _now_iso(), contact_id, *(() if expected is _UNREAD else (expected,)), *params))
            if not written.rowcount:
                raise ContactChanged(f"{contact_id}: may_contact or handles changed since they were read")
            current = current if expected is _UNREAD else str(expected)
        await db.commit()
        await self.record_audit(contact_id, "may_contact_set",
                                {"from": current, "to": value, "reason": reason or ""}, performed_by=by)
        contact = await self.get(contact_id)
        assert contact is not None
        return contact

    async def lower_may_contact(self, contact_id: str, *, reason: str, source_ref: str) -> Optional[Contact]:
        """A contact's opt-out (4.7 item 9): to ``never`` only. None when unknown or already never."""
        try:
            current = str((await self._permission_row(contact_id))["may_contact"])
        except ValueError:
            return None
        if current == "never":
            return None
        db = self._require_db()
        await db.execute(
            "UPDATE contacts SET may_contact = 'never', updated_at = ? WHERE contact_id = ?",
            (_now_iso(), contact_id),
        )
        await db.commit()
        await self.record_audit(contact_id, "opt_out",
                                {"from": current, "to": "never", "reason": reason, "source_ref": source_ref},
                                performed_by="contact")
        return await self.get(contact_id)

    async def set_cadence(self, contact_id: str, minutes: Optional[int], *, by: str, expected: Any = _UNREAD,
                          expected_handles: Optional[Sequence[str]] = None) -> Contact:
        """The owner-set cadence in minutes (None clears it), audited as ``cadence_set``; with ``expected`` and
        ``expected_handles`` a compare-and-set as ``set_may_contact`` makes (``ContactChanged`` when the
        cadence or the handle set moved since, nothing written and nothing rolled back)."""
        minutes = _cadence(minutes)
        row = await self._permission_row(contact_id)
        current = int(row["cadence_minutes"]) if row["cadence_minutes"] is not None else None
        db = self._require_db()
        if expected is _UNREAD and expected_handles is None:
            await db.execute(
                "UPDATE contacts SET cadence_minutes = ?, updated_at = ? WHERE contact_id = ?",
                (minutes, _now_iso(), contact_id),
            )
        else:
            held, params = _same_handles(contact_id, expected_handles)
            same = "" if expected is _UNREAD else " AND cadence_minutes IS ?"
            written = await db.execute(
                "UPDATE contacts SET cadence_minutes = ?, updated_at = ? WHERE contact_id = ? AND deleted_at IS NULL"
                + same + held, (minutes, _now_iso(), contact_id, *(() if expected is _UNREAD else (expected,)), *params))
            if not written.rowcount:
                raise ContactChanged(f"{contact_id}: cadence or handles changed since they were read")
            current = current if expected is _UNREAD else expected
        await db.commit()
        await self.record_audit(contact_id, "cadence_set", {"from": current, "to": minutes}, performed_by=by)
        contact = await self.get(contact_id)
        assert contact is not None
        return contact

    async def set_digest(self, contact_id: str, text: str, sources: List[str], *,
                         writer: Optional[str] = None) -> None:
        """Write the per-contact digest, the claim ids it was built from and its writer (``template`` or
        ``night``); an empty text clears it."""
        db = self._require_db()
        await db.execute(
            "UPDATE contacts SET digest = ?, digest_sources = ?, digest_writer = ?, digest_withheld_at = NULL, "
            "updated_at = ? WHERE contact_id = ?",
            (text or None, json.dumps([str(s) for s in sources or []]) if text else "[]",
             writer if text else None, _now_iso(), contact_id),
        )
        await db.commit()

    async def mark_digest_stale(self, contact_id: str, *, changed_at: str) -> None:
        """A fact changed: drop this contact's summary and restart its quiet period."""
        db = self._require_db()
        await db.execute(
            "UPDATE contacts SET digest = NULL, digest_sources = '[]', digest_withheld_at = ?, updated_at = ? "
            "WHERE contact_id = ? AND deleted_at IS NULL",
            (changed_at, _now_iso(), contact_id),
        )
        await db.commit()

    async def withhold_digest(self, contact_id: str, *, sources: List[str]) -> bool:
        """Withhold the digest a reader found resting on a claim that is no longer live: its text and sources
        go, its writer stays, and ``digest_withheld_at`` marks it for a rebuild. Only the digest that was
        checked (the same sources) is withheld, so one written again meanwhile stays for its own check.
        True when this call withheld it."""
        db = self._require_db()
        written = await db.execute(
            "UPDATE contacts SET digest = NULL, digest_sources = '[]', digest_withheld_at = ?, updated_at = ? "
            "WHERE contact_id = ? AND digest IS NOT NULL AND digest_sources = ?",
            (_now_iso(), _now_iso(), contact_id, json.dumps([str(s) for s in sources or []])),
        )
        await db.commit()
        return bool(written.rowcount)

    async def withheld_digests(self, *, limit: int = 200, offset: int = 0) -> List[Contact]:
        """The live contacts whose digest is withheld and waits for a rebuild, longest waiting first."""
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM contacts WHERE digest_withheld_at IS NOT NULL AND deleted_at IS NULL "
            "ORDER BY digest_withheld_at, contact_id LIMIT ? OFFSET ?",
            (max(1, int(limit)), max(0, int(offset))),
        ) as cur:
            rows = await cur.fetchall()
        return [Contact.from_row(dict(row)) for row in rows]

    async def count_withheld_digests(self) -> int:
        db = self._require_db()
        async with db.execute("SELECT count(*) FROM contacts WHERE digest_withheld_at IS NOT NULL "
                              "AND deleted_at IS NULL") as cur:
            return int((await cur.fetchone())[0])

    async def compute_cadence_overdue(
        self,
        *,
        now_iso: Optional[str] = None,
        default_cadence_days: float = 7.0,
        factor: float = 1.5,
        min_silence_days: float = 2.0,
        overdue_only: bool = True,
        limit: int = 20,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Per-contact rhythm and silence.

        An owner-set ``cadence_minutes`` is the cadence when present and is honoured exactly.
        Otherwise the cadence is estimated from the contact's own conversations (C3: active span /
        conversations, a conversation being turns less than 30 minutes apart), with the floor and
        ``factor`` applied, so a daily contact is overdue after a few days while a monthly one is
        not for weeks. ``never`` contacts are not listed.
        """
        from colonymind.util import temporal as _t
        db = self._require_db()
        now = _t.parse_iso(now_iso) or _t.now_utc()
        exclude = set(exclude_ids or [])
        async with db.execute(
            "SELECT contact_id, display_name, given_name, first_seen_at, last_interaction_at, "
            "interaction_count, timezone, may_contact, cadence_minutes "
            "FROM contacts WHERE deleted_at IS NULL AND may_contact != 'never' "
            "AND last_interaction_at IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            cid = r["contact_id"]
            if cid in exclude:
                continue
            last = _t.parse_iso(r["last_interaction_at"])
            if last is None:
                continue
            first = _t.parse_iso(r["first_seen_at"])
            count = int(r["interaction_count"] or 0)
            days_since = (now - last).total_seconds() / 86400.0
            minutes = r["cadence_minutes"]
            if minutes is not None:
                cadence = int(minutes) / 1440.0
                threshold = cadence
                is_overdue = days_since >= threshold
                source = "owner"
            else:
                if count >= 2 and first is not None and last > first:
                    span = (last - first).total_seconds() / 86400.0
                    cadence = span / max(count - 1, 1)
                else:
                    cadence = default_cadence_days
                cadence = max(0.5, min(cadence, 90.0))
                threshold = max(min_silence_days, cadence * factor)
                is_overdue = days_since > threshold
                source = "estimate"
            if overdue_only and not is_overdue:
                continue
            out.append({
                "contact_id": cid,
                "name": r["display_name"] or r["given_name"] or cid,
                "timezone": r["timezone"],
                "may_contact": r["may_contact"],
                "cadence_minutes": int(minutes) if minutes is not None else None,
                "cadence_source": source,
                "first_seen_at": r["first_seen_at"],
                "last_interaction_at": r["last_interaction_at"],
                "interaction_count": count,
                "days_since": round(days_since, 2),
                "cadence_days": round(cadence, 2),
                "overdue": is_overdue,
                "overdue_ratio": round(days_since / max(cadence, 1e-6), 2),
            })

        out.sort(key=lambda x: x["overdue_ratio"], reverse=True)
        return out[:limit]

    async def record_interaction(self, contact_id: str, at_iso: Optional[str] = None) -> bool:
        """An inbound turn from the contact (C3).

        ``last_interaction_at`` only moves forward; ``interaction_count`` counts CONVERSATIONS: a
        turn more than 30 minutes after the previous one starts a new conversation, a turn inside
        that window belongs to the current one. Returns True when the contact exists.
        """
        from colonymind.util.temporal import parse_iso
        db = self._require_db()
        ts = at_iso or _now_iso()
        async with db.execute(
            "SELECT last_interaction_at, first_seen_at FROM contacts WHERE contact_id = ? AND deleted_at IS NULL",
            (contact_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return False
        last, first = row["last_interaction_at"], row["first_seen_at"]
        new_dt, last_dt, first_dt = parse_iso(ts), parse_iso(last), parse_iso(first)
        if last is None or last_dt is None or new_dt is None:
            conversation, latest = True, ts
        else:
            conversation = (new_dt - last_dt) > _CONVERSATION_GAP
            latest = ts if new_dt >= last_dt else last
        first_seen = ts if (first_dt is None or (new_dt is not None and new_dt < first_dt)) else first
        await db.execute(
            "UPDATE contacts SET last_interaction_at = ?, first_seen_at = ?, "
            "interaction_count = interaction_count + ?, updated_at = ? "
            "WHERE contact_id = ? AND deleted_at IS NULL",
            (latest, first_seen, 1 if conversation else 0, _now_iso(), contact_id),
        )
        await db.commit()
        return True

    async def social_candidates(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        """The contacts the social drive may consider (architecture 4.5, 4.7): not the owner,
        permission is not ``never`` and the owner set a cadence or the tier is ``regular`` or
        above. Shadow and group-only contacts are never listed, so a group chat cannot become
        check-in asks."""
        from colonymind.identity import get_owner_contact_id
        db = self._require_db()
        tiers = [t for t in TRUST_TIERS if regular_or_above(t)]
        placeholders = ",".join("?" for _ in tiers)
        async with db.execute(
            "SELECT contact_id, display_name, trust_tier, may_contact, cadence_minutes, first_seen_at, "
            "last_interaction_at, interaction_count, timezone FROM contacts "
            "WHERE deleted_at IS NULL AND may_contact != 'never' AND contact_id != ? "
            f"AND (cadence_minutes IS NOT NULL OR trust_tier IN ({placeholders})) "
            "ORDER BY last_interaction_at IS NULL, last_interaction_at, created_at LIMIT ?",
            (get_owner_contact_id() or "", *tiers, max(1, min(int(limit), 1000))),
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            out.append({
                "contact_id": r["contact_id"], "display_name": r["display_name"], "trust_tier": r["trust_tier"],
                "may_contact": r["may_contact"],
                "cadence_minutes": int(r["cadence_minutes"]) if r["cadence_minutes"] is not None else None,
                "first_seen_at": r["first_seen_at"], "last_interaction_at": r["last_interaction_at"],
                "interaction_count": int(r["interaction_count"] or 0), "timezone": r["timezone"],
            })
        return out

    async def resolve_reference(self, reference: str, *, exact: bool = False) -> Optional[Contact]:
        """One contact for the way the owner names people: a contact id; a phone number or an email,
        alone or as ``gateway:address``, by the canonical rule (one that two contacts hold is nobody,
        whoever holds the text as given); a handle address only one contact holds; ``gateway:address``;
        and then (unless ``exact``) a unique display or given name, or a unique first word of a
        display name. A text with an at sign that no ``email`` handle matches (an iMessage address, a
        ``@name``) is such a handle address and never a name. A name is a guess:
        ``exact`` refuses it, so an owner's grant never sends to someone matched by name alone.
        A shadow's handle address is a guess too: the sender chose it (a username can be anyone's
        name), so only ``gateway:address``, a number or an email identifies a shadow exactly.
        Ambiguity or an unknown reference is None."""
        ref = (reference or "").strip()
        if not ref:
            return None
        if ref.startswith("cid-"):
            return await self.get(ref)
        if is_e164(ref):
            return await self._match_canonical("", ref)
        at_sign = "@" in ref and ":" not in ref
        if at_sign:
            found = await self._match_canonical("email", ref)
            if found is not None or await self.is_ambiguous_handle("email", ref):
                return found
        if ":" in ref:
            gateway, _, address = ref.partition(":")
            if canonical_handle(gateway, address)[0] in {"phone", "email"}:
                return await self._match_canonical(gateway, address)
        db = self._require_db()
        async with db.execute(
            "SELECT DISTINCT c.* FROM contacts c JOIN contact_handles h ON h.contact_id = c.contact_id "
            "WHERE h.address = ? AND c.deleted_at IS NULL AND (h.verified=1 OR h.source!='auto:scoped-name')",
            (ref,),
        ) as cur:
            holders = [Contact.from_row(dict(row)) for row in await cur.fetchall()]
        if len(holders) == 1 and not holders[0].is_shadow:
            return holders[0]
        if ":" in ref:
            gateway, _, address = ref.partition(":")
            if gateway.strip() and address.strip():
                found = await self.resolve_messaging_handle(gateway, address)
                if found is not None:
                    return found
        if exact or at_sign or len(holders) > 1:
            return None
        wanted = ref.lower()
        async with db.execute(
            "SELECT * FROM contacts WHERE deleted_at IS NULL AND (lower(display_name) = ? OR lower(given_name) = ?)",
            (wanted, wanted),
        ) as cur:
            rows = await cur.fetchall()
        if not rows and not holders:
            async with db.execute("SELECT * FROM contacts WHERE deleted_at IS NULL AND display_name IS NOT NULL") as cur:
                rows = [r for r in await cur.fetchall()
                        if str(r["display_name"] or "").strip().lower().split(" ")[0] == wanted]
        # A guess answers only when everyone the reference could mean is one person.
        guesses = {c.contact_id: c for c in [*holders, *(Contact.from_row(dict(r)) for r in rows)]}
        return next(iter(guesses.values())) if len(guesses) == 1 else None

    async def set_timezone(
        self, contact_id: str, timezone: Optional[str], performed_by: str = "operator",
        *, _connection: Optional[aiosqlite.Connection] = None,
    ) -> None:
        """Set (or clear, with None) a contact's IANA timezone. v0.21.0."""
        from colonymind.util.temporal import is_valid_timezone
        if timezone is not None and not is_valid_timezone(timezone):
            raise ValueError(f"Invalid IANA timezone: {timezone!r}")
        db = _connection if _connection is not None else self._require_db()
        async with db.execute(
            "SELECT timezone FROM contacts WHERE contact_id = ?", (contact_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            raise ValueError(f"Contact not found: {contact_id}")
        old_tz = row["timezone"]
        await db.execute(
            "UPDATE contacts SET timezone = ?, updated_at = ? WHERE contact_id = ?",
            (timezone, _now_iso(), contact_id),
        )
        if _connection is None:
            await db.commit()
        await self.record_audit(
            contact_id, "timezone_changed",
            {"old_timezone": old_tz, "new_timezone": timezone},
            performed_by=performed_by,
            _connection=_connection,
        )

    async def set_location_report(
        self, contact_id: str, *, timezone: str, city: Optional[str], region: Optional[str],
        country_code: Optional[str], reported_at: str,
    ) -> tuple[bool, Contact]:
        """Apply only a newer phone report, with its timezone and audits in one transaction.

        Durable stores use a dedicated connection so an unrelated contact write cannot
        commit half a report. BEGIN IMMEDIATE also orders writers across store instances.
        """
        from colonymind.util.temporal import parse_iso

        stamp = parse_iso(reported_at)
        if stamp is None:
            raise ValueError("Invalid location report timestamp")
        async with self._location_lock:
            durable = self._config.sqlite_path != ":memory:"
            db = await self._open_durable_connection() if durable else self._require_db()
            try:
                await db.execute("BEGIN IMMEDIATE")
                async with db.execute(
                    "SELECT * FROM contacts WHERE contact_id = ? AND deleted_at IS NULL", (contact_id,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    raise ValueError(f"Contact not found: {contact_id}")
                contact = Contact.from_row(dict(row))
                previous = parse_iso(contact.location_reported_at)
                if previous is not None and stamp <= previous:
                    await db.rollback()
                    return False, contact
                await self.set_timezone(contact_id, timezone, performed_by="companion-location", _connection=db)
                await db.execute(
                    "UPDATE contacts SET location_city = ?, location_region = ?, "
                    "location_country_code = ?, location_reported_at = ? WHERE contact_id = ?",
                    (city, region, country_code, reported_at, contact_id),
                )
                await self.record_audit(contact_id, "location_reported", {
                    "city": city, "region": region, "country_code": country_code, "reported_at": reported_at,
                }, performed_by="companion-location", _connection=db)
                async with db.execute("SELECT * FROM contacts WHERE contact_id = ?", (contact_id,)) as cur:
                    contact = Contact.from_row(dict(await cur.fetchone()))
                await db.commit()
                return True, contact
            except BaseException:
                await db.rollback()
                raise
            finally:
                if durable:
                    await db.close()

    async def soft_delete(
        self, contact_id: str, reason: Optional[str] = None, performed_by: str = "operator"
    ) -> None:
        db = self._require_db()
        now = _now_iso()
        await db.execute(
            "UPDATE contacts SET deleted_at = ?, updated_at = ? WHERE contact_id = ?",
            (now, now, contact_id),
        )
        await db.commit()
        await self.record_audit(
            contact_id, "soft_deleted", {"reason": reason}, performed_by=performed_by
        )

    async def hard_delete(self, contact_id: str, performed_by: str = "system") -> None:
        db = self._require_db()
        await self.record_audit(
            contact_id, "hard_deleted", {}, performed_by=performed_by
        )
        await db.execute("DELETE FROM contacts WHERE contact_id = ?", (contact_id,))
        await db.commit()

    async def merge(self, keep_id: str, drop_id: str, *, performed_by: str, reattribute=None,
                    sources_of=None, expected: Optional[Dict[str, Any]] = None) -> Contact:
        """C2: fold ``drop_id`` into ``keep_id``.

        Every handle of ``drop`` moves through ``identity_links.correct`` (one durable receipt per
        handle, ``operation_id`` ``merge:<drop>:<handle_id>``); the dropped contact's sources
        (``sources_of(drop_id)``) ride on receipts of their own (``merge:<drop>:sources:<n>``),
        which the host's existing reconciliation of ``pending_identity_reconciliations`` moves in
        the ledger. Every ``reattribute`` hook is then awaited as ``hook(drop_id, keep_id)`` (the
        comms log and the affect store). Last, in one commit: ``last_interaction_at`` is the max,
        ``first_seen_at`` the min, ``interaction_count`` the sum, the cadence the keeper's else the
        dropped one's, ``may_contact`` ``never`` if either was (an opt-out survives a merge;
        nothing else changes it), the tier the keeper's, the keeper's digest (the dropped one's
        when the keeper has none; never two joined), tags and notes appended, group memberships
        and identity candidates moved and ``drop`` soft-deleted, guarded so that a merge of the
        same pair that finished meanwhile is not folded twice. Both records are audited. A merge
        that stopped half way can be run again: moved handles and recorded sources are skipped.

        ``reattribute`` and ``sources_of`` default to the store's own (set once by the server). With
        ``expected`` (``contacts.changes.snapshot`` of both records when the merge was decided) a pair that
        changed since (permission, cadence or handles) raises ``ContactChanged`` before anything moves. The
        fold reads both records' permission and cadence as they are at that moment, so an opt-out that lands
        while the handles move still makes the result ``never``. With ``expected``, only the dropped record's
        handles it lists move, and a handle attached to the dropped record meanwhile (inbound attribution does
        not wait for the owner) stops the merge before the fold with ``ContactChanged``: the handles the owner
        saw are the keeper's as they approved, the new one stays where it landed, and the rest is theirs to ask
        again.
        """
        from .identity_links import correct, move_sources
        from colonymind.util.temporal import parse_iso
        if keep_id == drop_id:
            raise ValueError("a contact cannot be merged into itself")
        keep, drop = await self.get(keep_id), await self.get(drop_id)
        if keep is None or drop is None:
            raise ValueError("both contacts must exist to merge")
        if expected is not None:
            from .changes import snapshot
            if await snapshot(self, [keep, drop]) != expected:
                raise ContactChanged(f"{keep_id} or {drop_id} changed since the merge was decided")
        evidence = [f"merge:{keep_id}:{drop_id}"]
        operations = []
        seen = set((expected or {}).get(drop_id, {}).get("handles") or []) if expected is not None else None
        for handle in await self.get_handles(drop_id):
            if seen is not None and f"{handle.gateway}:{handle.address}" not in seen:
                continue           # attached after the merge was decided: not the owner's to fold
            receipt = await correct(
                self, operation_id=f"merge:{drop_id}:{handle.handle_id}", performed_by=performed_by,
                gateway=handle.gateway, address=handle.address, expected_contact_id=drop_id,
                contact_id=keep_id, evidence_refs=evidence)
            operations.append(receipt["operation_id"])
        if seen is not None and await self.get_handles(drop_id):
            raise ContactChanged(f"{drop_id} gained a handle while the merge was applied; nothing more is folded")
        sources_of = sources_of if sources_of is not None else self.sources_of
        sources = [str(s) for s in (await sources_of(drop_id) if sources_of else [])]
        moves = await move_sources(self, operation_prefix=f"merge:{drop_id}:sources:", performed_by=performed_by,
                                   old_contact_id=drop_id, contact_id=keep_id, evidence_refs=evidence,
                                   source_ids=sources)
        operations += [receipt["operation_id"] for receipt in moves]
        for hook in (list(reattribute) if reattribute is not None else list(self.reattribute)):
            result = hook(drop_id, keep_id)
            if inspect.isawaitable(result):
                await result

        def _pick(values, choose):
            stamped = [(parse_iso(v), v) for v in values if v]
            stamped = [(dt, v) for dt, v in stamped if dt is not None]
            return choose(stamped)[1] if stamped else None

        now = _now_iso()
        db = self._require_db()
        # The keeper's digest, else the dropped one's; with neither shown, one that waits for a rebuild.
        digest = keep if keep.digest else drop if drop.digest else keep if keep.digest_withheld_at else drop
        # The fold and the soft delete are one write guarded on the dropped record still being live:
        # a merge of the same pair that finished meanwhile leaves this one nothing to fold twice.
        folded = await db.execute(
            "UPDATE contacts SET last_interaction_at = ?, first_seen_at = ?, interaction_count = ?, "
            "cadence_minutes = COALESCE(cadence_minutes, (SELECT d.cadence_minutes FROM contacts d "
            "WHERE d.contact_id = ?)), "
            "may_contact = CASE WHEN may_contact = 'never' OR (SELECT d.may_contact FROM contacts d "
            "WHERE d.contact_id = ?) = 'never' THEN 'never' ELSE may_contact END, "
            "digest = ?, digest_sources = ?, digest_writer = ?, digest_withheld_at = ?, tags_json = ?, notes = ?, "
            "updated_at = ? WHERE contact_id = ? AND EXISTS (SELECT 1 FROM contacts d WHERE d.contact_id = ? "
            "AND d.deleted_at IS NULL)",
            (_pick([keep.last_interaction_at, drop.last_interaction_at], max),
             _pick([keep.first_seen_at, drop.first_seen_at], min) or keep.first_seen_at,
             int(keep.interaction_count) + int(drop.interaction_count),
             drop_id, drop_id,
             digest.digest or None, json.dumps(list(digest.digest_sources) if digest.digest else []),
             digest.digest_writer if digest.digest or digest.digest_withheld_at else None, digest.digest_withheld_at,
             json.dumps(list(keep.tags) + [t for t in drop.tags if t not in keep.tags]),
             "\n".join(n for n in (keep.notes, drop.notes) if n) or None, now, keep_id, drop_id),
        )
        if not folded.rowcount:
            # Nothing written: no rollback on the shared connection (it would undo another task's write).
            merged = await self.get(keep_id)
            assert merged is not None
            return merged
        # Group memberships follow the person; one both records held stays one (current if either was).
        await db.execute("UPDATE scope_members SET left_at = NULL WHERE contact_id = ? AND left_at IS NOT NULL "
                         "AND scope_id IN (SELECT scope_id FROM scope_members WHERE contact_id = ? AND left_at IS NULL)",
                         (keep_id, drop_id))
        await db.execute("UPDATE OR IGNORE scope_members SET contact_id = ? WHERE contact_id = ?", (keep_id, drop_id))
        await db.execute("DELETE FROM scope_members WHERE contact_id = ?", (drop_id,))
        async with db.execute("SELECT candidate_id FROM contact_identity_candidates WHERE contact_id = ?",
                              (drop_id,)) as cur:
            candidates = [row["candidate_id"] for row in await cur.fetchall()]
        for candidate_id in candidates:
            try:
                await db.execute("UPDATE contact_identity_candidates SET contact_id = ? WHERE candidate_id = ?",
                                 (keep_id, candidate_id))
            except sqlite3.IntegrityError:  # the keeper already holds the same candidate
                await db.execute("DELETE FROM contact_identity_candidates WHERE candidate_id = ?", (candidate_id,))
        await db.execute("UPDATE contacts SET deleted_at = ?, updated_at = ? WHERE contact_id = ?",
                         (now, now, drop_id))
        await db.commit()
        await self.record_audit(
            keep_id, "merged_in",
            {"merged_contact_id": drop_id, "merged_display_name": drop.display_name,
             "operations": operations, "sources": sum(len(r["affected_source_ids"]) for r in moves)},
            performed_by=performed_by)
        await self.record_audit(drop_id, "merged_into", {"kept_contact_id": keep_id, "reason": f"merged into {keep_id}"},
                                performed_by=performed_by)
        merged = await self.get(keep_id)
        assert merged is not None
        return merged

    async def _candidate(self, candidate_id: str) -> Dict[str, Any]:
        db = self._require_db()
        async with db.execute("SELECT * FROM contact_identity_candidates WHERE candidate_id = ?",
                              (candidate_id,)) as cur:
            row = await cur.fetchone()
        if row is None or row["status"] != "pending":
            raise ValueError("identity_candidate_not_pending")
        value = dict(row)
        value["evidence_refs"] = json.loads(value.pop("evidence_refs_json") or "[]")
        return value

    async def _handle_holder(self, gateway: str, address: str) -> Optional[str]:
        if canonical_handle(gateway, address)[0] in {"phone", "email"}:
            canonical = await self._match_canonical(gateway, address)
            return canonical.contact_id if canonical is not None else None
        db = self._require_db()
        async with db.execute("SELECT contact_id FROM contact_handles WHERE gateway = ? AND address = ?",
                              (gateway, address)) as cur:
            row = await cur.fetchone()
        return row["contact_id"] if row else None

    async def confirm_link(self, candidate_id: str, *, performed_by: str, reattribute=None,
                           sources_of=None) -> Dict[str, Any]:
        """The owner's yes to a name-only link proposal (A7).

        The handle nobody holds is attached, verified, through ``identity_links.correct``. When a
        shadow contact holds it (the sender kept a separate identity while the name only suggested
        the link), confirming the identity folds that shadow into the proposed contact through
        ``merge``, so its history follows the person. When an established contact holds it, nothing
        moves: the proposal is closed and ``identity_handle_held`` raised.
        """
        from .identity_links import correct
        candidate = await self._candidate(candidate_id)
        target, gateway, address = candidate["contact_id"], candidate["gateway"], candidate["address"]
        if await self.get(target) is None:
            raise ValueError("identity_contact_not_found")
        if await self.is_ambiguous_handle(gateway, address):
            raise ValueError("Handle identity is ambiguous; correct the whole number first")
        holder = await self._handle_holder(gateway, address)
        evidence = list(candidate["evidence_refs"]) or [f"candidate:{candidate_id}"]
        held_by = await self.get(holder) if holder is not None and holder != target else None
        if held_by is not None and not held_by.is_shadow:
            # An established, different person holds the handle (a number reassigned, a proposal
            # anyone may file): a yes to the link never deletes them. The proposal is closed and the
            # owner merges the two explicitly if they are one person.
            db = self._require_db()
            await db.execute("UPDATE contact_identity_candidates SET status = 'rejected', resolved_at = ? "
                             "WHERE candidate_id = ? AND status = 'pending'", (_now_iso(), candidate_id))
            await db.commit()
            await self.record_audit(target, "link_refused", {"candidate_id": candidate_id, "gateway": gateway,
                                                             "held_by": holder}, performed_by=performed_by)
            raise ValueError("identity_handle_held")
        if held_by is not None:
            await self.merge(target, holder, performed_by=performed_by, reattribute=reattribute, sources_of=sources_of)
            receipt = {"operation_id": f"merge:{holder}", "old_contact_id": holder, "contact_id": target, "merged": True}
        else:
            # The correction acts on the exact row, so it expects that row's holder: the contact itself,
            # or nobody where the contact holds the number only in another form.
            own = any((h.gateway, h.address) == (gateway, address) for h in await self.get_handles(target))
            receipt = await correct(
                self, operation_id=f"link:{candidate_id}", performed_by=performed_by, gateway=gateway,
                address=address, expected_contact_id=target if own else None, contact_id=target,
                evidence_refs=evidence)
            receipt = dict(receipt, merged=False)
        db = self._require_db()
        await db.execute("UPDATE contact_identity_candidates SET status = 'confirmed', resolved_at = ? "
                         "WHERE candidate_id = ? AND status = 'pending'", (_now_iso(), candidate_id))
        await db.commit()
        return {"candidate_id": candidate_id, "status": "confirmed", **receipt}

    async def reject_link(self, candidate_id: str, *, performed_by: str) -> Dict[str, Any]:
        """The owner's no: the candidate is closed; whoever holds the handle keeps it."""
        candidate = await self._candidate(candidate_id)
        db = self._require_db()
        now = _now_iso()
        await db.execute("UPDATE contact_identity_candidates SET status = 'rejected', resolved_at = ? "
                         "WHERE candidate_id = ?", (now, candidate_id))
        await db.commit()
        await self.record_audit(
            candidate["contact_id"], "link_rejected",
            {"candidate_id": candidate_id, "gateway": candidate["gateway"],
             "address_sha256": hashlib.sha256(candidate["address"].encode()).hexdigest()},
            performed_by=performed_by)
        return {"candidate_id": candidate_id, "contact_id": candidate["contact_id"], "status": "rejected",
                "resolved_at": now}

    async def list_handle_proposals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Pending candidate associations, never installed identity links."""
        db = self._require_db()
        async with db.execute(
            "SELECT p.*, c.display_name FROM contact_identity_candidates p "
            "JOIN contacts c ON c.contact_id=p.contact_id WHERE p.status='pending' "
            "AND c.deleted_at IS NULL ORDER BY p.created_at DESC LIMIT ?", (max(1, min(100, limit)),)) as cur:
            rows = [dict(row) for row in await cur.fetchall()]
        for row in rows:
            row['at'] = row['created_at']
            row['evidence_refs'] = json.loads(row.pop('evidence_refs_json'))
        return rows

    async def update(self, contact_id: str, **fields) -> Optional[Contact]:
        db = self._require_db()
        allowed_fields = {
            "display_name", "given_name", "family_name", "organization",
            "notes", "person_node_id", "privacy_level",
            "last_interaction_at", "interaction_count",
            "enrichment_source", "enrichment_last_at",
            "cadence_minutes", "digest", "digest_sources",
        }
        set_parts = []
        params = []
        for k, v in fields.items():
            if k not in allowed_fields:
                continue
            if k in ("enrichment_source", "digest_sources") and isinstance(v, list):
                v = json.dumps(v)
            if k == "cadence_minutes":
                v = _cadence(v)
            # SQL-01: column name is validated against allowed_fields; double-quote the
            # identifier so SQLite treats it safely even if allowed_fields is later extended.
            set_parts.append(f'"{k}" = ?')
            params.append(v)
        if not set_parts:
            return await self.get(contact_id)
        set_parts.append("updated_at = ?")
        params.append(_now_iso())
        params.append(contact_id)
        await db.execute(
            f"UPDATE contacts SET {', '.join(set_parts)} WHERE contact_id = ?",
            params,
        )
        await db.commit()
        return await self.get(contact_id)

    async def record_audit(
        self,
        contact_id: str,
        action: str,
        detail: Optional[Dict[str, Any]] = None,
        performed_by: str = "system",
        *, _connection: Optional[aiosqlite.Connection] = None,
    ) -> None:
        db = _connection if _connection is not None else self._require_db()
        audit_id = _gen_id("cau")
        now = _now_iso()
        await db.execute(
            "INSERT INTO contact_audit (id, contact_id, action, detail, performed_by, created_at) VALUES (?,?,?,?,?,?)",
            (audit_id, contact_id, action, json.dumps(detail or {}), performed_by, now),
        )
        if _connection is None:
            await db.commit()

    async def audit_since(self, actions: List[str], since: str, *, limit: int = 100) -> List[Dict[str, Any]]:
        """Audit rows of ``actions`` recorded at or after ``since`` (ISO), oldest first, each with the
        contact's display name and its detail parsed: the opt-outs the owner's digest lists."""
        if not actions:
            return []
        db = self._require_db()
        marks = ",".join("?" for _ in actions)
        async with db.execute(
            f"SELECT a.contact_id, a.action, a.detail, a.performed_by, a.created_at, c.display_name "  # noqa: S608
            f"FROM contact_audit a LEFT JOIN contacts c ON c.contact_id = a.contact_id "
            f"WHERE a.action IN ({marks}) AND a.created_at >= ? ORDER BY a.created_at LIMIT ?",
            (*actions, since, limit),
        ) as cur:
            rows = [dict(row) for row in await cur.fetchall()]
        for row in rows:
            try:
                row["detail"] = json.loads(row["detail"] or "{}")
            except (TypeError, ValueError):
                row["detail"] = {}
        return rows

    async def get_audit_log(self, contact_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        db = self._require_db()
        async with db.execute(
            "SELECT * FROM contact_audit WHERE contact_id = ? ORDER BY created_at DESC LIMIT ?",
            (contact_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Trust scopes (context-scoped trust) ────────────────────────────────────

    async def create_scope(
        self,
        *,
        scope_type: str = "group",
        platform: Optional[str] = None,
        external_id: Optional[str] = None,
        label: Optional[str] = None,
        granted_tier: str = "group_guest",
        created_by: str = "agent",
    ) -> TrustScope:
        """Create a trust scope, or return the existing active one for
        (platform, external_id) if it already exists (idempotent upsert)."""
        if granted_tier not in TRUST_TIERS:
            raise ValueError(f"invalid granted_tier: {granted_tier}")
        db = self._require_db()
        if platform is not None and external_id is not None:
            existing = await self.get_scope(platform=platform, external_id=external_id)
            if existing is not None:
                return existing
        scope_id = _gen_id("ts")
        now = _now_iso()
        await db.execute(
            "INSERT INTO trust_scopes (scope_id, scope_type, platform, external_id, label, "
            "granted_tier, created_by, active, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (scope_id, scope_type, platform, external_id, label, granted_tier, created_by, now, now),
        )
        await db.commit()
        return TrustScope(
            scope_id=scope_id, scope_type=scope_type, platform=platform, external_id=external_id,
            label=label, granted_tier=granted_tier, created_by=created_by, active=True,
            created_at=now, updated_at=now,
        )

    async def get_scope(
        self,
        *,
        scope_id: Optional[str] = None,
        platform: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> Optional[TrustScope]:
        """Fetch a scope by scope_id, or by (platform, external_id)."""
        db = self._require_db()
        if scope_id is not None:
            sql, params = "SELECT * FROM trust_scopes WHERE scope_id = ?", (scope_id,)
        elif platform is not None and external_id is not None:
            sql = "SELECT * FROM trust_scopes WHERE platform = ? AND external_id = ?"
            params = (platform, external_id)
        else:
            raise ValueError("get_scope needs scope_id or (platform, external_id)")
        async with db.execute(sql, params) as cur:
            row = await cur.fetchone()
        return TrustScope.from_row(dict(row)) if row else None

    async def add_scope_member(
        self, scope_id: str, contact_id: str, role: str = "member"
    ) -> None:
        """Add (or re-activate) a contact's membership in a scope."""
        db = self._require_db()
        await db.execute(
            "INSERT INTO scope_members (scope_id, contact_id, role, joined_at, left_at) "
            "VALUES (?, ?, ?, ?, NULL) "
            "ON CONFLICT(scope_id, contact_id) DO UPDATE SET role = excluded.role, left_at = NULL",
            (scope_id, contact_id, role, _now_iso()),
        )
        await db.commit()
        await self.record_audit(contact_id, "scope_member_added",
                                {"scope_id": scope_id, "role": role}, performed_by="agent")

    async def remove_scope_member(self, scope_id: str, contact_id: str) -> None:
        """Mark a member as having left the scope (soft; preserves history)."""
        db = self._require_db()
        await db.execute(
            "UPDATE scope_members SET left_at = ? WHERE scope_id = ? AND contact_id = ? AND left_at IS NULL",
            (_now_iso(), scope_id, contact_id),
        )
        await db.commit()
        await self.record_audit(contact_id, "scope_member_removed",
                                {"scope_id": scope_id}, performed_by="agent")

    async def scope_members(self, scope_id: str, *, current_only: bool = True) -> List[ScopeMember]:
        db = self._require_db()
        sql = "SELECT * FROM scope_members WHERE scope_id = ?"
        if current_only:
            sql += " AND left_at IS NULL"
        async with db.execute(sql, (scope_id,)) as cur:
            rows = await cur.fetchall()
        return [ScopeMember.from_row(dict(r)) for r in rows]

    async def scopes_for_contact(self, contact_id: str, *, active_only: bool = True) -> List[TrustScope]:
        """All scopes a contact is a current member of (optionally active scopes only)."""
        db = self._require_db()
        sql = (
            "SELECT s.* FROM trust_scopes s "
            "JOIN scope_members m ON m.scope_id = s.scope_id "
            "WHERE m.contact_id = ? AND m.left_at IS NULL"
        )
        if active_only:
            sql += " AND s.active = 1"
        async with db.execute(sql, (contact_id,)) as cur:
            rows = await cur.fetchall()
        return [TrustScope.from_row(dict(r)) for r in rows]

    async def is_authorized_in_scope(self, contact_id: str, scope_id: str) -> bool:
        """True iff the contact is a current member of the (active) scope.
        This is group-scoped authorization — it says nothing about 1:1 rights."""
        db = self._require_db()
        async with db.execute(
            "SELECT 1 FROM scope_members m JOIN trust_scopes s ON s.scope_id = m.scope_id "
            "WHERE m.scope_id = ? AND m.contact_id = ? AND m.left_at IS NULL AND s.active = 1",
            (scope_id, contact_id),
        ) as cur:
            return await cur.fetchone() is not None

    async def deactivate_scope(self, scope_id: str) -> None:
        """Deactivate a scope (revokes group-trust for all members at once)."""
        db = self._require_db()
        members = await self.scope_members(scope_id, current_only=True)
        await db.execute(
            "UPDATE trust_scopes SET active = 0, updated_at = ? WHERE scope_id = ?",
            (_now_iso(), scope_id),
        )
        await db.commit()
        for m in members:
            await self.record_audit(m.contact_id, "scope_deactivated",
                                    {"scope_id": scope_id}, performed_by="agent")

    async def group_promotion_candidates(
        self, *, min_interactions: int = 5, limit: int = 50
    ) -> List["Contact"]:
        """Current members of an ACTIVE scope with sustained conversations whose tier is still below
        ``regular``: the people the owner could promote. Group membership alone never promotes,
        and a tier never grants permission (``may_contact`` is the owner's alone)."""
        db = self._require_db()
        below = [t for t in TRUST_TIERS if not regular_or_above(t)]
        placeholders = ",".join("?" for _ in below)
        async with db.execute(
            "SELECT DISTINCT c.* FROM contacts c "
            "JOIN scope_members m ON m.contact_id = c.contact_id AND m.left_at IS NULL "
            "JOIN trust_scopes s ON s.scope_id = m.scope_id AND s.active = 1 "
            f"WHERE c.deleted_at IS NULL AND c.trust_tier IN ({placeholders}) "
            "AND c.interaction_count >= ? ORDER BY c.interaction_count DESC LIMIT ?",
            (*below, int(min_interactions), int(limit)),
        ) as cur:
            rows = await cur.fetchall()
        return [Contact.from_row(dict(r)) for r in rows]

    async def promote_scope_member(
        self, contact_id: str, *, to_tier: str = "regular", performed_by: str = "agent"
    ) -> bool:
        """Raise a group-scope member's tier to ``to_tier``. Only ever raises the tier, never
        ``may_contact``; returns True iff something changed."""
        c = await self.get(contact_id)
        if c is None:
            return False
        if tier_rank(c.trust_tier) >= tier_rank(to_tier):
            return False
        await self.update_tier(contact_id, to_tier, reason="promoted from group scope", performed_by=performed_by)
        await self.record_audit(contact_id, "scope_promoted", {"to_tier": to_tier}, performed_by=performed_by)
        return True
