"""Ingress fingerprints and one-use sealing attestations; never retain message text.

The dispatch wrapper records ingress only. The secret sealer alone calls
``seal_rewrite`` around its own transform. Each observer retains the original
immutable receipt, and consumes its own ingress ledger row when binding a turn.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import hmac
import secrets
from threading import RLock
from weakref import WeakKeyDictionary, WeakSet, ref


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True, eq=False)
class TextReceipt:
    event_id: int
    source_id: int
    coordinates: str
    before: str
    key: bytes = field(repr=False)
    identity: object = field(repr=False)


@dataclass(frozen=True)
class _EventIdentity:
    # Owned by the event, never by evidence. This supports event stand-ins which
    # cannot themselves be weak-referenced. Evidence cannot extend their life.
    event: object
    source: object


@dataclass(frozen=True)
class _Sealed:
    after: tuple[str, str]


# Identity keys reject copies, including dataclass copies with identical fields.
# Weak keys release evidence when the event and its observing ledgers release it.
_sealed = WeakKeyDictionary()
_observers = WeakKeyDictionary()
_revoked = WeakSet()
_lock = RLock()


def _digest(key, text):
    return hmac.new(key, text.encode('utf-8'), hashlib.sha256).hexdigest()


def _coordinates(event):
    source = getattr(event, 'source', None)
    platform = getattr(source, 'platform', '')
    values = (getattr(platform, 'value', platform),
              *(getattr(source, name, '') for name in
                ('chat_id', 'chat_type', 'user_id', 'thread_id', 'profile', 'profile_name')),
              getattr(event, 'message_id', ''), getattr(event, 'user_id', ''))
    return repr(tuple(str(value or '') for value in values))


def _bound(receipt, event):
    identity = receipt.identity() if isinstance(receipt, TextReceipt) else None
    return (identity is not None and identity.event is event
            and identity.source is getattr(event, 'source', None)
            and receipt in _sealed
            and receipt.event_id == id(event)
            and receipt.source_id == id(getattr(event, 'source', None))
            and receipt.coordinates == _digest(receipt.key, _coordinates(event)))


def begin_rewrite(event):
    """Snapshot raw ingress, without granting authority to any dispatch output."""
    with _lock:
        existing = getattr(event, '_colonymind_text_receipt', None)
        if existing is not None:
            return existing if _bound(existing, event) else None
        text = getattr(event, 'text', None)
        if not isinstance(text, str):
            return None
        key = secrets.token_bytes(32)
        identity = _EventIdentity(event, getattr(event, 'source', None))
        event._colonymind_ingress_identity = identity
        receipt = TextReceipt(id(event), id(getattr(event, 'source', None)),
                              _digest(key, _coordinates(event)), _digest(key, text), key, ref(identity))
        _sealed[receipt] = None
        _observers[receipt] = set()
        event._colonymind_text_receipt = receipt
        return receipt


def bind_rewrite(event, *, observer=None):
    """Observe this event's original ingress or its exact attested sealed text."""
    with _lock:
        receipt = begin_rewrite(event)
        if receipt is None or receipt in _revoked:
            return None
        digest = _digest(receipt.key, event.text)
        sealed = _sealed[receipt]
        if digest != receipt.before and not (sealed and digest == sealed.after[0]):
            return None
        if observer is not None:
            observers = _observers[receipt]
            if id(observer) in observers:
                return None
            observers.add(id(observer))
        return receipt


def seal_rewrite(event, transform):
    """Run the trusted sealer; attest only an unchanged original event, once.

    Contaminated input still needs secret removal, but cannot acquire ingress
    authority. The result is never inferred from the remainder of a hook chain.
    """
    with _lock:
        receipt = begin_rewrite(event)
        valid = (receipt is not None and _sealed[receipt] is None
                 and receipt.before == _digest(receipt.key, event.text))
        # Reserve the single use before invoking code, including recursive calls.
        if valid:
            _sealed[receipt] = _Sealed(('', ''))
        result = transform(event.text)
        if valid and _bound(receipt, event) and isinstance(result, str):
            _sealed[receipt] = _Sealed((_digest(receipt.key, result), _digest(receipt.key, result.strip())))
        return result


def valid_ingress(receipt):
    """A wrapper can revoke ingress, but can never issue a sealing attestation."""
    with _lock:
        return receipt is None or (isinstance(receipt, TextReceipt)
                                   and receipt in _sealed and receipt not in _revoked)


def finish_ingress(receipt, event):
    """Revoke a dispatch output which differs from raw ingress and trusted sealing.

    This is a refusal check, never a source of accepted after-fingerprints. It
    also prevents dispatch prefixes from using the later native-note matcher.
    """
    with _lock:
        if not isinstance(receipt, TextReceipt) or receipt not in _sealed:
            return
        sealed = _sealed[receipt]
        text = getattr(event, 'text', None)
        if not isinstance(text, str):
            _revoked.add(receipt)
            return
        digest = _digest(receipt.key, text)
        if (not _bound(receipt, event) or
                (digest != receipt.before and not (sealed and digest == sealed.after[0]))):
            _revoked.add(receipt)


def matches_rewrite(receipt, text, *, strip=False):
    with _lock:
        sealed = _sealed.get(receipt) if isinstance(receipt, TextReceipt) and valid_ingress(receipt) else None
        return bool(sealed and sealed.after[bool(strip)] ==
                    _digest(receipt.key, text.strip() if strip else text))


def composition(history, text, message_id):
    """Only the current native user row, anchored to this turn's last platform id."""
    current = history[-1] if isinstance(history, (list, tuple)) and history else {}
    if not isinstance(current, Mapping) or current.get('role') != 'user':
        return None
    metadata = current.get('display_metadata') or {}
    value = metadata.get('midturn_composition') if isinstance(metadata, Mapping) else None
    if not isinstance(value, Mapping) or value.get('version') != 1:
        return None
    envelope, members = value.get('text'), value.get('messages')
    if not isinstance(envelope, str) or not (text == envelope or text.endswith('\n\n' + envelope)):
        return None
    if current.get('content') != text or current.get('display_kind') == 'internal_notification':
        return None
    if not isinstance(members, (list, tuple)) or len(members) < 2:
        return None
    if any(not isinstance(row, Mapping) or not isinstance(row.get('text'), str)
           or not row['text'].strip() or not isinstance(row.get('message_id'), str)
           or not row['message_id'] for row in members):
        return None
    ids = [row['message_id'] for row in members]
    if len(set(ids)) != len(ids) or ids[-1] != message_id:
        return None
    return tuple((row['message_id'], row['text']) for row in members)


def member_turn_id(turn_id, message_id):
    return turn_id + ':message:' + text_hash(message_id)
