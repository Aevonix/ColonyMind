"""Sealing: detected secrets move into the vault and a reference stands where each value was.

``redact_sensitive_messages`` seals a list of source messages (the ledger's ``record_source`` runs it for every
writer before its digest, insert, index and queues) and ``redact_sensitive_turn`` seals a whole turn envelope
(turn capture runs it before the envelope digest and the reservation, so the summary, topics, communications log,
preference learner, opt-out, the mind's owner turn and every event see references only). Sealing is idempotent
(a reference is never detected) and deterministic (the same value in the same scope gets the same reference), so
a replay or a re-import stores the same bytes and digests.

When the vault cannot take a value, the value is withheld (``detect.WITHHELD``) rather than stored, except in
``strict`` sealing (the migration), which raises and changes nothing.

A source's sealing is staged (``Staged``): the references are worked out without storing anything, and the values
are stored only when the source they came in is admitted (``commit``). A replay of an erased source, a conflicting
one or one whose digest check fails leaves the vault as it was, so a value the owner forgot stays forgotten.

Every text a message carries is sealed, not only its content: a tool observation's executed arguments and the
reason it was kept, a message's metadata. Structure is left alone only where it is structure: a message's own role
and source references (``STRUCTURAL_KEYS``), and in its provenance a native observation's identity and the digest
of its arguments, which the observation writer takes again over the sealed words. A string under a mapping key is
also read with its key (``"password": "..."``), as it read in the serialized text, so a value named only by its
field is found too.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .config import SecretsSettings, current_settings
from .detect import REFERENCE, WITHHELD, Hit, find_secrets, replace_secrets
from .refs import reference_text, references_in
from .vault import Vault, VaultUnavailable, get_vault

logger = logging.getLogger(__name__)

TEXT_BLOCKS = frozenset({"text", "input_text", "output_text"})
# A message's own fields that are structure, never words (at its top level only; the same names deeper are words).
STRUCTURAL_KEYS = frozenset({"role", "_supplied_sources", "_observation_sources", "_supplied_inputs",
                             "_native_tool_observation", "_native_runtime_observation", "_source_message_hash",
                             "_observed_conversation"})
# In a message's provenance: a native observation's identity, and in its input the digest of its arguments.
PROVENANCE_STRUCTURE = frozenset({"native"})
INPUT_STRUCTURE = frozenset({"arguments_sha256"})


@dataclass(frozen=True)
class Scope:
    """Whose value it is (the person whose message carried it), and where it came from."""
    contact_id: str
    session_id: str = ""
    source_turn: str = ""


@dataclass
class SealCount:
    sealed: int = 0
    withheld: int = 0
    rules: Dict[str, int] = field(default_factory=dict)

    def add(self, other: "SealCount") -> None:
        self.sealed += other.sealed
        self.withheld += other.withheld
        for rule, count in other.rules.items():
            self.rules[rule] = self.rules.get(rule, 0) + count


def _settings(settings: Optional[SecretsSettings]) -> SecretsSettings:
    return settings if settings is not None else current_settings()


class Staged:
    """Values a source's sealing planned and has not stored: stored by ``commit``, once the source is admitted.
    Until then nothing is written, so a rejected source leaves the vault exactly as it was."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        # A value this stage's sealing could not plan (the vault unavailable) and withheld instead.
        self.withheld = False

    def __bool__(self) -> bool:
        return bool(self._entries)

    def value(self, ref: str) -> Optional[str]:
        """The value planned under ``ref`` (in memory only, for an erasure's identity check), or None."""
        entry = self._entries.get(ref)
        return None if entry is None else entry["value"]

    def add(self, ref: str, value: str, *, vault: Vault, scope: Scope, kind: str, label: str) -> None:
        self._entries.setdefault(ref, {"value": value, "vault": vault, "scope": scope, "kind": kind, "label": label})

    def merge(self, other: Optional["Staged"]) -> "Staged":
        if other is not None and other is not self:
            for ref, entry in other._entries.items():
                self._entries.setdefault(ref, entry)
            self.withheld = self.withheld or other.withheld
        return self

    def commit(self, *, only: Optional[Callable[[str], bool]] = None) -> None:
        """Store every planned value (``only``: those whose reference it accepts, the ones a retained message
        carries) under the reference it was planned under. Raises ``VaultUnavailable`` (the caller's source is
        then not stored either) when the vault cannot take one or changed since the plan (the owner forgot or
        released the value meanwhile): the vault checks before it writes, so nothing is stored for it."""
        for ref in [ref for ref in self._entries if only is None or only(ref)]:
            entry = self._entries[ref]
            scope = entry["scope"]
            stored = entry["vault"].seal(entry["value"], contact_id=scope.contact_id, session_id=scope.session_id,
                                         kind=entry["kind"], label=entry["label"], source_turn=scope.source_turn,
                                         expect=ref)
            if stored != ref:
                raise VaultUnavailable("the sealed store changed while this source was sealed; send it again")
            del self._entries[ref]


def seal_text(text: Any, *, scope: Scope, vault: Optional[Vault] = None,
              settings: Optional[SecretsSettings] = None, count: Optional[SealCount] = None,
              existing_only: bool = False, stage: Optional[Staged] = None, strict: bool = False,
              released: bool = False) -> Any:
    """``text`` with every detected value sealed; anything that is not a string comes back as it is.

    ``stage``: the references are planned and the values added to the stage, nothing is stored. ``strict``: a
    value the vault cannot take raises ``VaultUnavailable`` instead of being withheld. ``existing_only``: a value
    takes the reference it already has and nothing new is stored (a value never sealed stays as it is), the form
    a stored copy of ``text`` has; with ``released`` a released value takes its reference too (the form it had
    before the owner released it)."""
    if not isinstance(text, str) or not text:
        return text
    active = _settings(settings)
    if not active.capture_enabled:
        return text
    store = vault
    tally = count if count is not None else SealCount()

    def replacement(hit: Hit) -> str:
        nonlocal store
        try:
            store = store if store is not None else get_vault()
            if existing_only:
                found = store.find(hit.value, contact_id=scope.contact_id, released=released)
                return reference_text(found, hit.label) if found else hit.value
            if stage is not None:
                ref, state = store.plan(hit.value, contact_id=scope.contact_id)
                if state == "released":
                    ref = None
                else:
                    stage.add(ref, hit.value, vault=store, scope=scope, kind=hit.rule, label=hit.label)
            else:
                ref = store.seal(hit.value, contact_id=scope.contact_id, session_id=scope.session_id, kind=hit.rule,
                                 label=hit.label, source_turn=scope.source_turn)
        except Exception as error:   # the value is never stored as it came, whatever failed
            if strict:
                raise VaultUnavailable(f"a {hit.rule} value could not be sealed ({type(error).__name__})") from None
            reason = error if isinstance(error, VaultUnavailable) else type(error).__name__
            logger.warning("a %s value was withheld: the vault could not take it (%s)", hit.rule, reason)
            tally.withheld += 1
            if stage is not None:
                stage.withheld = True
            return WITHHELD
        if ref is None:
            return hit.value  # the owner released this very value: it stays as it came
        tally.sealed += 1
        tally.rules[hit.rule] = tally.rules.get(hit.rule, 0) + 1
        return reference_text(ref, hit.label)

    sealed, _ = replace_secrets(text, replacement, rules=active.rules)
    return sealed


def _map_content(content: Any, change: Callable[[str], str]) -> Any:
    """``content`` (a string or content blocks) with ``change`` applied to every text it carries: text blocks,
    audio transcript segments and an observed conversation's words (its pieces, their context and the device's
    summary and action items). Other blocks are kept as they are."""
    if isinstance(content, str):
        return change(content)
    if not isinstance(content, list):
        return content
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append(block)
            continue
        kind = block.get("type")
        if kind in TEXT_BLOCKS and isinstance(block.get("text"), str):
            block = {**block, "text": change(block["text"])}
        elif kind == "audio_transcript" and isinstance(block.get("segments"), list):
            block = {**block, "segments": [
                {**segment, "text": change(segment["text"])}
                if isinstance(segment, dict) and isinstance(segment.get("text"), str) else segment
                for segment in block["segments"]]}
        elif kind == "observed_conversation":
            block = dict(block)
            for name in ("pieces", "context_before"):
                if isinstance(block.get(name), list):
                    block[name] = [{**piece, "text": change(piece["text"])}
                                   if isinstance(piece, dict) and isinstance(piece.get("text"), str) else piece
                                   for piece in block[name]]
            derived = block.get("derived")
            if isinstance(derived, dict):
                derived = dict(derived)
                if isinstance(derived.get("summary"), str):
                    derived["summary"] = change(derived["summary"])
                if isinstance(derived.get("action_items"), list):
                    derived["action_items"] = [change(item) if isinstance(item, str) else item
                                               for item in derived["action_items"]]
                block["derived"] = derived
        blocks.append(block)
    return blocks


def seal_content(content: Any, *, scope: Scope, vault: Optional[Vault] = None,
                 settings: Optional[SecretsSettings] = None, count: Optional[SealCount] = None,
                 stage: Optional[Staged] = None, strict: bool = False) -> Any:
    return _map_content(content, lambda text: seal_text(text, scope=scope, vault=vault, settings=settings,
                                                        count=count, stage=stage, strict=strict))


def _field(key: Any, text: str, change: Callable[[str], str]) -> str:
    """``text``, the string a mapping holds under ``key``, with ``change`` applied: to the string alone, and when
    that finds nothing, to the string read with its key as the serialized mapping showed them
    (``"password": "..."``), where a value named only by its field is found. What the second reading changed is
    taken only when it is the whole string, replaced by one reference (or a dry run's count mark)."""
    changed = change(text)
    if changed != text or not isinstance(key, str) or not key or len(key) > 80 or '"' in key or '"' in text:
        return changed
    prefix = f'"{key}": "'
    field = change(prefix + text + '"')
    if field.endswith('"\x00'):          # a dry run counts a hit by marking the text
        return text + "\x00"
    inner = field[len(prefix):-1] if field.startswith(prefix) and field.endswith('"') else ""
    if inner != text and (REFERENCE.fullmatch(inner) or inner == WITHHELD):
        return inner
    return text


def map_leaves(value: Any, change: Callable[[str], str], *, skip: Iterable[str] = (), key: Any = None) -> Any:
    """``value`` with ``change`` applied to every string it holds, at any depth (a string under a mapping key read
    with its key too: ``_field``); a mapping key in ``skip`` keeps its value as it is."""
    skipped = frozenset(skip)
    if isinstance(value, str):
        return _field(key, value, change) if key is not None else change(value)
    if isinstance(value, list):
        return [map_leaves(item, change, skip=skipped) for item in value]
    if isinstance(value, Mapping):
        return {name: item if name in skipped else map_leaves(item, change, skip=skipped, key=name)
                for name, item in value.items()}
    return value


def _map_provenance(provenance: Any, change: Callable[[str], str]) -> Any:
    """A message's provenance with its words changed: everything but a native observation's identity and the
    digest of its arguments (the observation writer takes both again over the sealed words)."""
    if not isinstance(provenance, Mapping):
        return map_leaves(provenance, change)
    mapped: Dict[str, Any] = {}
    for key, value in provenance.items():
        if key in PROVENANCE_STRUCTURE:
            mapped[key] = value
        elif key == "input" and isinstance(value, Mapping):
            mapped[key] = {name: item if name in INPUT_STRUCTURE else map_leaves(item, change, key=name)
                           for name, item in value.items()}
        else:
            mapped[key] = map_leaves(value, change, key=key)
    return mapped


def map_message(message: Mapping[str, Any], change: Callable[[str], str]) -> Dict[str, Any]:
    """A copy of ``message`` with ``change`` applied to its words: the content's text (text blocks, transcript
    segments) and every string anywhere else in it. Structure is left as it is only where it is structure: the
    message's own ``STRUCTURAL_KEYS`` and, in its provenance, a native observation's identity and arguments
    digest; the same names deeper (metadata's ``content`` or ``native``) are words like any other."""
    mapped: Dict[str, Any] = {}
    for key, value in copy.deepcopy(dict(message)).items():
        if key in STRUCTURAL_KEYS:
            mapped[key] = value
        elif key == "content":
            mapped[key] = _map_content(value, change)
        elif key == "provenance":
            mapped[key] = _map_provenance(value, change)
        else:
            mapped[key] = map_leaves(value, change, key=key)
    return mapped


def redact_sensitive_messages(messages: List[Mapping[str, Any]], *, scope: Scope, vault: Optional[Vault] = None,
                              settings: Optional[SecretsSettings] = None,
                              count: Optional[SealCount] = None, stage: Optional[Staged] = None,
                              strict: bool = False) -> List[Dict[str, Any]]:
    """Copies of ``messages`` with every detected value sealed, in their content and everywhere else they carry
    words (``map_message``); the caller's list is untouched. See ``seal_text`` for ``stage`` and ``strict``."""
    def change(text: str) -> str:
        return seal_text(text, scope=scope, vault=vault, settings=settings, count=count, stage=stage, strict=strict)
    return [map_message(message, change) if isinstance(message, Mapping) else message for message in messages]


@dataclass
class TurnSeal:
    """What sealing changed in one envelope: the counts, the values staged until the turn is admitted, and each
    changed message's raw form, kept in memory only so the ledger's erasure check can test the raw form's
    identity too (never stored)."""
    count: SealCount = field(default_factory=SealCount)
    raw_forms: List[Tuple[str, Any, Any]] = field(default_factory=list)
    stage: Staged = field(default_factory=Staged)

    def commit(self) -> None:
        """Store the staged values: the turn was admitted (``Staged.commit``)."""
        self.stage.commit()

    def raw_hashes(self, session_id: str) -> Dict[str, str]:
        """``{sealed identity hash: raw identity hash}`` of every changed message in ``session_id``."""
        from colonymind.turns.idempotency import source_message_hash
        return {source_message_hash(session_id, {"role": role, "content": sealed}):
                source_message_hash(session_id, {"role": role, "content": raw})
                for role, raw, sealed in self.raw_forms}

    def merge(self, other: "TurnSeal") -> "TurnSeal":
        self.count.add(other.count)
        self.raw_forms.extend(other.raw_forms)
        self.stage.merge(other.stage)
        return self


def redact_sensitive_turn(body: Any, *, vault: Optional[Vault] = None, settings: Optional[SecretsSettings] = None,
                          contact_id: Optional[str] = None, stage: Optional[Staged] = None) -> TurnSeal:
    """Seal a turn envelope in place: its messages (user, assistant, checkpoint), summary, topics, entities, pending
    tasks, tools used and a transport caption. The scope is ``contact_id`` (the person whose message carried the
    values, as the caller resolved them), else the envelope's contact, and its session. The values are staged
    (``TurnSeal.stage``, or ``stage``): stored when the turn is admitted."""
    result = TurnSeal(stage=stage if stage is not None else Staged())
    active = _settings(settings)
    if not active.capture_enabled or body is None:
        return result
    context = getattr(body, "context", None)
    scope = Scope(contact_id=str(contact_id or getattr(context, "contact_id", "") or ""),
                  session_id=str(getattr(context, "session_id", "") or ""),
                  source_turn=str(getattr(context, "turn_id", "") or ""))

    def text(value: Any) -> Any:
        return seal_text(value, scope=scope, vault=vault, settings=active, count=result.count, stage=result.stage)

    for message in [getattr(body, "user_message", None), getattr(body, "assistant_message", None),
                    *(getattr(body, "checkpoint_messages", None) or [])]:
        if message is None:
            continue
        raw = message.content
        sealed = _map_content(copy.deepcopy(raw), text)
        if sealed != raw:
            result.raw_forms.append((message.role, raw, sealed))
            message.content = sealed
    if isinstance(getattr(body, "summary", None), str):
        body.summary = text(body.summary)
    for name in ("topics", "entities", "pending_tasks", "tools_used"):
        values = getattr(body, name, None)
        if isinstance(values, list):
            setattr(body, name, [text(value) for value in values])
    work = getattr(body, 'work', None)
    if work is not None:
        # Only sealed facts enter the immutable request digest and work ledger.
        # Hashes already refer to sealed content; source refs are structural.
        values = map_leaves(work.model_dump(exclude_none=True), text,
                            skip=('source_refs', 'intended_sha256', 'actual_sha256'))
        body.work = type(work).model_validate(values)
    return result


def unseal_text(text: str, values: Mapping[str, str]) -> str:
    """``text`` with each reference whose id is in ``values`` replaced by its value (a release, a fill)."""
    if not isinstance(text, str) or not values:
        return text
    parts, cursor = [], 0
    for item in references_in(text):
        if item.ref_id in values:
            parts += [text[cursor:item.start], values[item.ref_id]]
            cursor = item.end
    parts.append(text[cursor:])
    return "".join(parts)


def unseal_content(content: Any, values: Mapping[str, str]) -> Any:
    return _map_content(content, lambda text: unseal_text(text, values))


def erasure_forms(content: Any, *, contact_id: str, vault: Optional[Vault] = None,
                  settings: Optional[SecretsSettings] = None, stage: Optional[Staged] = None) -> List[Any]:
    """The other forms a stored message's ``content`` has had, for an erasure's identity check (computed in
    memory, never stored, never returned to anyone):

    * as it arrived, every reference in it resolved (``stage``'s planned values first, then the vault): an erasure
      written before sealing existed names that form, and a gateway that sealed the value first never showed it
      to the sidecar;
    * as it was sealed at any earlier time: each value taking each reference it has had in ``contact_id``'s scope
      (``Vault.identities``: a released one, one the owner forgot and the value then took anew), so an erasure
      written before a release or before a forget names that form.

    A reference the store no longer holds (forgotten, unknown, sealed under a key the instance lost) stays as it
    is. Raises ``VaultUnavailable`` when a form cannot be worked out because the key cannot be read now: the
    caller must not admit the message on a check it could not make."""
    texts: List[str] = []
    _map_content(content, lambda text: texts.append(text) or text)
    refs = list(dict.fromkeys(item.ref_id for text in texts for item in references_in(text)))
    active = _settings(settings)
    if not refs and not any(find_secrets(text, rules=active.rules) for text in texts):
        return []
    store = vault if vault is not None else get_vault()
    if not store.available:
        raise VaultUnavailable("an erasure cannot be checked while the vault key is unavailable")
    from .vault import SecretForgotten, UnknownReference
    values: Dict[str, str] = {}
    for ref in refs:
        planned = stage.value(ref) if stage is not None else None
        if planned is not None:
            values[ref] = planned
            continue
        try:
            values[ref] = store.resolve(ref)
        except (SecretForgotten, UnknownReference):
            continue
        except VaultUnavailable:
            if not store.available:
                raise
            continue   # sealed under a key this instance no longer has: that form cannot be rebuilt
    raw = unseal_content(content, values) if values else content
    forms: List[Any] = [raw] if raw != content else []
    identities: Dict[str, List[str]] = {}

    def collect(text: str) -> str:
        for hit in find_secrets(text, rules=active.rules):
            if hit.value not in identities:
                identities[hit.value] = store.identities(hit.value, contact_id=contact_id)
        return text
    _map_content(raw, collect)
    generations = max((len(found) for found in identities.values()), default=0)
    for generation in range(generations):
        def sealed_as(text: str, generation: int = generation) -> str:
            def replacement(hit: Hit) -> str:
                found = identities.get(hit.value) or []
                return reference_text(found[min(generation, len(found) - 1)], hit.label) if found else hit.value
            return replace_secrets(text, replacement, rules=active.rules)[0]
        form = _map_content(copy.deepcopy(raw), sealed_as)
        if form != content and form not in forms:
            forms.append(form)
    return forms


__all__ = ["INPUT_STRUCTURE", "PROVENANCE_STRUCTURE", "STRUCTURAL_KEYS", "Scope", "SealCount", "Staged", "TurnSeal",
           "erasure_forms", "map_leaves", "map_message", "redact_sensitive_messages", "redact_sensitive_turn",
           "seal_content", "seal_text", "unseal_content", "unseal_text"]
