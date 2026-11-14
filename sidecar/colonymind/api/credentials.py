"""Who each API credential may write as (docs/CREDENTIALS.md).

An instance has one key of its own, ``api.key``: the Hermes adapter's credential, which ``colonymind
init`` writes and the adapter reads. ``colonymind credentials add`` issues more, one per integration.
``colonymind.yaml`` binds every credential to the contacts and channels it may write as:

    credentials:
      api-key:                            # api.key
        owner: [whatsapp, sms, rcs, voice]
        contacts: all
      game-bot:
        sha256: <digest of its secret>
        owner: []
        contacts: own

``owner`` lists the platforms a credential may attribute a turn to the owner from (``any``: every
platform); a turn's platform is its sender's, else the part of the channel it names before the first
colon. ``contacts: all`` lets it write as anyone else the sidecar resolves or the body names;
``contacts: own`` binds it to its own identity, ``credential:<name>``; a list of contact ids binds it to
exactly those persons (``viewer``: the one a request that names nobody acts as; ``senders: all``: a
sender may still resolve to anyone). A turn written under a credential's identity is kept as session
evidence: it never becomes a claim, an appraisal, a commitment or a question to the owner. ``access``
names what it may do: ``write`` turns, ``read`` context and memory, ``manage`` (every other route, the
owner's side of the API), ``note`` (leave notes through the MCP endpoint, always under its own identity;
``write`` includes it); by default it writes and reads, and manages only when granted the owner. A
credential carried from a 1.9.0 principal keeps that principal's ``scopes`` instead, and reaches exactly
the routes 1.9.0's rules gave them (``colonymind.api.legacy_scopes``).

``answer`` is the one exception to "no credential approves": a credential issued to an owner channel's own
relay (``companion-hub: {owner: [companion], access: [answer]}``) reads the owner's hands permit asks and records
the owner's explicit answer to one (docs/hands-permit-answer-contract.md). It does nothing else: ``answer`` stands
alone in its access, its owner platforms are explicit-answer channels named one by one (never ``any``), and the
instance key never has it, so nothing that holds ``api.key`` (the Hermes adapter, an agent's shell) answers that way.
Default and 1.9.0 grants never include it.
The same private relay serves explicit app password reveals (docs/secret-reveal.md).

``observe`` is a hub's least privilege for outside observations
(``sensor-hub: {owner: [companion], access: [observe], observe_sources: [sensor]}``): granted the owner, it posts
the observations of the sources it lists, reads and sets their daily caps, and records the conversations observed
from them; it reaches nothing else, no turns, no reads of memory and no other owner route. Like ``answer`` it stands
alone in its access, names its owner platforms one by one and is never the instance key's. ``manage`` with the owner
grant reaches the same routes for every source.

``tool-origin`` is an external MCP consumer's narrow grant: only claim a gateway's tool-origin record and
request an external-write owner action. It stands alone, has no owner platforms, uses ``contacts: own`` and
is never the instance key's. The instance key alone registers origins; neither grant can answer these asks.

Outside Companion owner-goal actions (docs/CREDENTIALS.md), no credential approves anything by itself:
a yes to an ask, a standing grant and a change that widens who gets
messaged come only from the owner's own message (docs/MIND.md, Asks), or, for a hands permit ask only, from an
``answer`` credential's explicit answer as the owner. There is no ``approve`` operation: an agent with a shell could
mint any credential the owner can. A credential granted the owner on the terminal (``cli``, or
every platform) lets anything that can start a Hermes chat on the machine write as the owner, and so answer the
owner's asks; ``describe`` says so, and ``colonymind doctor`` warns when the agent can.

A credential may carry several secrets while one rotates (``secrets``: each a digest with an optional
``accept_until``), and ``accept_until`` on the credential bounds them all; a secret is refused at
authentication from its time on. The instance key's ``secrets`` are the ones it accepts besides
``api.key``, which never expires.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping

from colonymind.util.temporal import now_utc

#: The instance key's entry: its secret is ``api.key`` (or ``COLONYMIND_API_KEY``), never a digest here.
INSTANCE_CREDENTIAL = "api-key"
#: The principal of a keyless loopback request (``colonymind.api.auth``): never a credential's name.
RESERVED_NAMES = frozenset({"anonymous-dev"})
#: The principal of development mode (no key): a credential of that name would share its stamps.
DEVELOPMENT_PRINCIPAL = "anonymous-dev"
IDENTITY_PREFIX = "credential:"
DEFAULT_OWNER_PLATFORMS = ("whatsapp", "sms", "rcs", "voice")
#: Hermes' own terminal: a turn from it is whatever a local ``hermes chat`` says, whoever started it.
TERMINAL_PLATFORM = "cli"
#: ``owner: any``: the owner from every platform (a 1.9.0 principal granted the owner that declared none).
ANY_OWNER = "any"
#: The owner platforms of a credential granted the owner from every platform (and of an in-process call).
ANY_PLATFORM = "*"
CONTACT_SCOPES = ("all", "own")
#: What a credential may do: write turns, read context and memory, reach every other route, and leave notes through
#: the MCP endpoint under its own identity (``colonymind.api.mcp``; a credential that writes turns may note too). No
#: route is a ``note`` route: that access alone reaches none. ``answer``: read the hands permit asks and answer one
#: as the owner from an explicit-answer channel (``ANSWER_ROUTES``); it stands alone (``validate_credentials``).
#: ``observe``: post the outside observations of the sources a credential lists, and read and set their caps and
#: observed conversations (``observe_route``); it stands alone too, and ``manage`` with the owner reaches the same.
OPERATIONS = ("read", "write", "manage", "note", "answer", "observe", "tool-origin", "hands")
HANDS = "hands"


def hands_route(path: str) -> bool:
    return path.startswith("/v1/hands/")
ANSWER = "answer"
OBSERVE = "observe"
#: External MCP consumers only: claim a gateway origin or ask for one write's owner action.
TOOL_ORIGIN = "tool-origin"
TOOL_ORIGIN_ROUTES = frozenset({("POST", "/v1/mind/tool-origins/claim"),
                                ("POST", "/v1/mind/owner-actions")})
#: The owner channels whose relay answers a hands permit ask by its id (``answer``): named in an ``answer``
#: credential's ``owner`` one by one. A messaging channel answers by the owner's own message instead.
ANSWER_PLATFORMS = ("companion",)
_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
_PLATFORM = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,31}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_KEYS = frozenset({"sha256", "secrets", "accept_until", "owner", "contacts", "viewer", "senders", "access",
                   "scopes", "unscoped_api", "observe_sources"})
#: An outside observation source, as ``colonymind.mind.observations`` takes one: at most 128 opaque characters.
_SOURCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_PERSON = re.compile(r"^[^\s\x00-\x1f]{1,256}$")
_SECRET_KEYS = frozenset({"sha256", "accept_until"})

#: The routes of each operation: writing turns and reading context and memory under the contacts a
#: credential is granted. Everything else (the mind, people, commitments, erasure, configuration) is the
#: owner's side of the API, which ``manage`` reaches.
WRITE_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", "/v1/host/turns/sync"),
    ("PUT", "/v2/host/turns/"),
)
READ_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", "/v1/host/context/card"),
    ("POST", "/v1/host/context/assemble"),
    ("POST", "/v1/host/memory/read"),
    ("POST", "/v1/host/memory/search"),
    ("POST", "/v1/host/memory/voice-recall"),
    ("POST", "/v1/host/memory/voice-affect"),
    ("POST", "/v1/host/memory/recent"),
    # The memory inventory and a claim's history: a credential bound to a person lists that person only.
    ("GET", "/v1/host/memory/inventory"),
    ("GET", "/v1/host/memory/inventory/claims/"),
    ("GET", "/v1/host/memory/curated/"),
)
INTEGRATION_ROUTES: tuple[tuple[str, str], ...] = WRITE_ROUTES + READ_ROUTES
#: The permit asks read and answered by id through the private owner relay.
ANSWER_PREFIX = "/v1/mind/permit-asks"
# The same private owner relay also serves a single app-initiated password reveal.
SECRET_REVEAL_PATH = "/v1/secrets/saved-password/reveal"
#: The ``observe`` routes (``colonymind.api.routers.mind``, ``colonymind.api.routers.host``), each with the source it
#: is for when the path names one: a source's observations posted, its daily cap read and set, and a conversation
#: observed from a source (named in the body, ``colonymind.api.auth.observe_source_granted``) recorded and read.
OBSERVE_ROUTES: tuple[tuple[frozenset[str], re.Pattern[str]], ...] = (
    (frozenset({"POST"}), re.compile(r"^/v1/mind/observations/sources/(?P<source>[^/]+)$")),
    (frozenset({"GET", "PUT"}), re.compile(r"^/v1/mind/observations/sources/(?P<source>[^/]+)/cap$")),
    (frozenset({"GET", "PUT"}), re.compile(r"^/v2/host/conversations/observed/[^/]+$")),
)


def default_credentials() -> dict[str, dict[str, Any]]:
    return {INSTANCE_CREDENTIAL: {"owner": list(DEFAULT_OWNER_PLATFORMS), "contacts": "all"}}


def identity_of(name: str) -> str:
    return IDENTITY_PREFIX + name


def is_credential_identity(contact_id: str | None) -> bool:
    """A contact id that names a credential's own identity rather than a person."""
    return str(contact_id or "").startswith(IDENTITY_PREFIX)


def secret_digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def platform_of(sender_platform: str | None, channel_id: str | None) -> str:
    """The platform a turn arrives from: its sender's, else its named channel's prefix; "" when neither."""
    value = (sender_platform or "").strip() or (channel_id or "").strip().split(":", 1)[0]
    return value.strip().lower()


def _config_error(message: str) -> Exception:
    from colonymind.config import ConfigError
    return ConfigError(message)


def parse_until(value: Any, where: str) -> datetime:
    """An ``accept_until`` time (ISO 8601; without a zone it is UTC), in UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            raise _config_error(f"{where} must be an ISO 8601 time, for example 2030-01-01T00:00:00+00:00") from None
    elif isinstance(value, date):
        raise _config_error(f"{where} must be a time, not only a date (for example {value.isoformat()}T00:00:00Z)")
    else:
        raise _config_error(f"{where} must be an ISO 8601 time, for example 2030-01-01T00:00:00+00:00")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _digest(value: Any, where: str) -> str:
    digest = str(value or "").strip().lower()
    if not _DIGEST.match(digest):
        raise _config_error(f"{where} must be the 64-character SHA-256 hex digest of its secret "
                            "('colonymind credentials add' writes it)")
    return digest


def _validated_secrets(value: Any, where: str) -> list[dict[str, str]]:
    """``secrets``: a non-empty list of ``{sha256, accept_until?}``, each checked and normalized."""
    if not isinstance(value, list) or not value:
        raise _config_error(f"{where} must be a non-empty list of {{sha256, accept_until}} entries")
    checked = []
    for index, item in enumerate(value):
        at = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise _config_error(f"{at} must be a mapping with sha256 and an optional accept_until")
        unknown = sorted(set(item) - _SECRET_KEYS)
        if unknown:
            raise _config_error(f"{at}: unknown key {unknown[0]} (sha256 or accept_until)")
        secret = {"sha256": _digest(item.get("sha256"), f"{at}.sha256")}
        if item.get("accept_until") is not None:
            secret["accept_until"] = parse_until(item["accept_until"], f"{at}.accept_until").isoformat()
        checked.append(secret)
    return checked


def validate_credentials(value: Any) -> dict[str, dict[str, Any]]:
    """``credentials`` in ``colonymind.yaml``, checked and normalized; the instance key's entry always exists."""
    if value is None:
        return default_credentials()
    if not isinstance(value, dict):
        raise _config_error("credentials must be a mapping of credential names to their grants")
    checked: dict[str, dict[str, Any]] = {}
    seen: dict[str, str] = {}
    for raw_name, entry in value.items():
        name = str(raw_name)
        where = f"credentials.{name}"
        if not _NAME.match(name):
            raise _config_error(f"{where}: a name is lowercase letters, digits, '.', '_' or '-' (at most 63)")
        if name in RESERVED_NAMES:
            raise _config_error(f"{where}: {name} is the principal of development mode (no key); choose another name")
        if name == DEVELOPMENT_PRINCIPAL:
            raise _config_error(f"{where}: {DEVELOPMENT_PRINCIPAL} names development mode's caller; choose another")
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            raise _config_error(f"{where} must be a mapping with sha256, owner and contacts")
        unknown = sorted(set(entry) - _KEYS)
        if unknown:
            raise _config_error(f"{where}: unknown key {unknown[0]} (sha256, secrets, accept_until, owner or "
                                "contacts)")
        owner = entry.get("owner", list(DEFAULT_OWNER_PLATFORMS) if name == INSTANCE_CREDENTIAL else [])
        if owner is None:
            owner = []
        if isinstance(owner, str) and owner.strip().lower() == ANY_OWNER:
            owner = [ANY_PLATFORM]
        if not isinstance(owner, list) or any(not isinstance(item, str) for item in owner):
            raise _config_error(f"{where}.owner must be a list of platform names, or any (every platform)")
        platforms: list[str] = []
        for item in owner if owner != [ANY_PLATFORM] else []:
            platform = item.strip().lower()
            if not _PLATFORM.match(platform):
                raise _config_error(f"{where}.owner: {item!r} is not a platform name "
                                    "(lowercase letters, digits, '.', '_' or '-')")
            if platform not in platforms:
                platforms.append(platform)
        raw_contacts = entry.get("contacts")
        if isinstance(raw_contacts, list):
            contacts: Any = []
            for item in raw_contacts:
                person = str(item).strip() if isinstance(item, str) else ""
                if not _PERSON.match(person):
                    raise _config_error(f"{where}.contacts: {item!r} is not a contact id")
                if person.startswith(IDENTITY_PREFIX):
                    raise _config_error(f"{where}.contacts: {person} is a credential's identity, not a person")
                if person not in contacts:
                    contacts.append(person)
        else:
            contacts = str(raw_contacts or ("all" if name == INSTANCE_CREDENTIAL else "own")).strip().lower()
            if contacts not in CONTACT_SCOPES:
                raise _config_error(f"{where}.contacts must be all (anyone the sidecar resolves), own (only "
                                    f"{identity_of(name)}) or a list of contact ids")
        listed = isinstance(contacts, list)
        viewer = entry.get("viewer")
        if viewer is not None and (not listed or not isinstance(viewer, str) or not _PERSON.match(viewer.strip())):
            raise _config_error(f"{where}.viewer is the contact id a request that names nobody acts as; it "
                                "takes a list of contacts")
        senders = entry.get("senders")
        if senders is not None and (not listed or str(senders).strip().lower() not in {"all", "listed"}):
            raise _config_error(f"{where}.senders is all (a sender may resolve to anyone) or listed; it takes a "
                                "list of contacts")
        access = entry.get("access")
        if access is not None:
            if not isinstance(access, list) or any(str(item).strip().lower() not in OPERATIONS for item in access):
                raise _config_error(f"{where}.access must list what it may do: read, write, manage, note, answer, "
                                    "observe, tool-origin, hands")
            access = [op for op in OPERATIONS if op in {str(item).strip().lower() for item in access}]
        if access is not None and TOOL_ORIGIN in access:
            if (name == INSTANCE_CREDENTIAL or access != [TOOL_ORIGIN] or owner
                    or contacts != "own" or viewer is not None or senders is not None):
                raise _config_error(f"{where}: tool-origin stands alone, with no owner grant and contacts: own")
        if access is not None and HANDS in access:
            if (name == INSTANCE_CREDENTIAL or access != [HANDS] or owner == [ANY_PLATFORM]
                    or platforms not in ([], ['companion'])
                    or contacts != "own" or viewer is not None or senders is not None):
                raise _config_error(f"{where}: hands stands alone, with only optional companion owner platform and contacts: own")
        answers = access is not None and ANSWER in access
        if answers:
            _check_answer_credential(name, where, access, owner, platforms, entry.get("contacts"))
        observing = access is not None and OBSERVE in access
        sources = entry.get("observe_sources")
        if observing:
            sources = _check_observe_credential(name, where, access, owner, platforms, entry.get("contacts"), sources)
        elif sources is not None:
            raise _config_error(f"{where}.observe_sources lists the sources an observe credential posts for; it "
                                "takes access: [observe]")
        scopes = entry.get("scopes")
        if scopes is not None:
            from colonymind.api.legacy_scopes import is_scope
            if access is not None:
                raise _config_error(f"{where}: give access or a 1.9.0 principal's scopes, not both")
            if not isinstance(scopes, list) or any(not isinstance(item, str) or not is_scope(item.strip())
                                                   for item in scopes):
                raise _config_error(f"{where}.scopes must list 1.9.0 scopes (* or area:action)")
            scopes = sorted(dict.fromkeys(item.strip() for item in scopes))
        unscoped_api = entry.get("unscoped_api")
        if unscoped_api is not None and (scopes is None or not isinstance(unscoped_api, bool)):
            raise _config_error(f"{where}.unscoped_api is true or false, and takes scopes")
        if (contacts == "own" and (platforms or owner == [ANY_PLATFORM]) and not answers and not observing
                and access != [HANDS]):
            raise _config_error(f"{where}: a credential with contacts: own writes only as {identity_of(name)}; "
                                "it cannot also write as the owner (empty its owner list or set contacts: all)")
        result: dict[str, Any] = {}
        if "sha256" in entry and "secrets" in entry:
            raise _config_error(f"{where}: give sha256 (one secret) or secrets (several, while one rotates), "
                                "not both")
        if name == INSTANCE_CREDENTIAL:
            if "sha256" in entry:
                raise _config_error(f"{where}: api-key is the instance key, whose secret is api.key; "
                                    "it takes no sha256 (secrets lists the ones it also accepts)")
            if entry.get("accept_until") is not None:
                raise _config_error(f"{where}: api.key never expires; a secret in secrets may carry "
                                    "accept_until")
            if contacts != "all":
                raise _config_error(f"{where}.contacts: the instance key is the Hermes adapter's and writes "
                                    "as the contacts it resolves (contacts: all)")
            if entry.get("secrets") is not None:
                result["secrets"] = _validated_secrets(entry["secrets"], f"{where}.secrets")
        elif "secrets" in entry:
            result["secrets"] = _validated_secrets(entry["secrets"], f"{where}.secrets")
        else:
            result["sha256"] = _digest(entry.get("sha256"), f"{where}.sha256")
        digests = [result["sha256"]] if "sha256" in result else [item["sha256"] for item in result.get("secrets", [])]
        for digest in digests:
            if digest in seen:
                other = "itself" if seen[digest] == name else f"credentials.{seen[digest]}"
                raise _config_error(f"{where} has the same secret as {other}")
            seen[digest] = name
        if entry.get("accept_until") is not None:
            result["accept_until"] = parse_until(entry["accept_until"], f"{where}.accept_until").isoformat()
        if owner == [ANY_PLATFORM]:
            platforms = [ANY_PLATFORM]
        result["owner"] = ANY_OWNER if platforms == [ANY_PLATFORM] else platforms
        result["contacts"] = contacts
        if viewer is not None:
            result["viewer"] = viewer.strip()
        if senders is not None and str(senders).strip().lower() == "all":
            result["senders"] = "all"
        if access is not None:
            result["access"] = access
        if observing:
            result["observe_sources"] = sources
        if scopes is not None:
            result["scopes"] = scopes
            if unscoped_api is False:
                result["unscoped_api"] = False
        checked[name] = result
    if INSTANCE_CREDENTIAL not in checked:
        checked = {**default_credentials(), **checked}
    return checked


def _check_answer_credential(name: str, where: str, access: list[str], owner: Any, platforms: list[str],
                             contacts: Any) -> None:
    """An ``answer`` credential: an owner channel's relay that answers hands permit asks and does nothing else. Not
    the instance key (anything that can read ``api.key`` would approve held calls), never ``owner: any``, its owner
    platforms the explicit-answer channels it answers from, named one by one, and it writes as no one."""
    if name == INSTANCE_CREDENTIAL:
        raise _config_error(f"{where}.access: the instance key never answers asks (anything that can read api.key "
                            "would approve held calls); issue the channel's relay its own credential")
    if access != [ANSWER]:
        raise _config_error(f"{where}.access: answer stands alone; a credential that answers asks does nothing else")
    if owner == [ANY_PLATFORM]:
        raise _config_error(f"{where}.owner: a credential that answers asks names its channels one by one, never any")
    if not platforms or any(platform not in ANSWER_PLATFORMS for platform in platforms):
        raise _config_error(f"{where}.owner: a credential that answers asks lists the channels it answers from, "
                            f"of {', '.join(ANSWER_PLATFORMS)}")
    if contacts is not None and str(contacts).strip().lower() != "own":
        raise _config_error(f"{where}.contacts: a credential that answers asks writes as no one (leave contacts "
                            "out, or own)")


def _check_observe_credential(name: str, where: str, access: list[str], owner: Any, platforms: list[str],
                              contacts: Any, sources: Any) -> list[str]:
    """An ``observe`` credential: a hub that posts the owner's outside observations for the sources it lists and does
    nothing else. Not the instance key, granted the owner on platforms named one by one (never ``any``), writing as no
    one, with at least one source. Returns the sources, checked and in order."""
    if name == INSTANCE_CREDENTIAL:
        raise _config_error(f"{where}.access: the instance key never has observe; issue the hub its own credential")
    if access != [OBSERVE]:
        raise _config_error(f"{where}.access: observe stands alone; a credential that posts observations does "
                            "nothing else")
    if owner == [ANY_PLATFORM]:
        raise _config_error(f"{where}.owner: a credential that posts observations names its platforms one by one, "
                            "never any")
    if not platforms:
        raise _config_error(f"{where}.owner: a credential that posts observations is granted the owner (the "
                            "observations are the owner's); list the platform it posts from")
    if contacts is not None and str(contacts).strip().lower() != "own":
        raise _config_error(f"{where}.contacts: a credential that posts observations writes as no one (leave "
                            "contacts out, or own)")
    if not isinstance(sources, list) or not sources:
        raise _config_error(f"{where}.observe_sources must list the sources it posts observations for")
    checked: list[str] = []
    for item in sources:
        if not isinstance(item, str) or not _SOURCE.match(item.strip()):
            raise _config_error(f"{where}.observe_sources: {item!r} is not a source identifier (letters, digits, "
                                "'_', '.', ':', '/' or '-', at most 128)")
        if item.strip() not in checked:
            checked.append(item.strip())
    return checked


def observes(grant: "Grant") -> bool:
    """Whether ``grant`` is an ``observe`` credential (never the instance key, never ``owner: any``)."""
    return (OBSERVE in grant.access and not grant.instance and grant.scopes is None
            and ANY_PLATFORM not in grant.owner)


def answers_asks(grant: "Grant") -> bool:
    """Whether ``grant`` is an ``answer`` credential (never the instance key, never ``owner: any``)."""
    return (ANSWER in grant.access and not grant.instance and grant.scopes is None
            and ANY_PLATFORM not in grant.owner)


def _live(until: datetime | None, now: datetime) -> bool:
    return until is None or now < until


@dataclass(frozen=True)
class Grant:
    """One credential's binding: the platforms it may write the owner from, in the configured order, and
    the secrets it accepts (a digest and its own expiry each), all bounded by ``accept_until``."""

    name: str
    owner: tuple[str, ...]
    #: ``all``, ``own`` or ``list`` (exactly ``listed``).
    contacts: str
    secrets: tuple[tuple[str, datetime | None], ...] = ()
    accept_until: datetime | None = None
    listed: tuple[str, ...] = ()
    viewer: str | None = None
    #: A sender may resolve to anyone (a list credential with ``senders: all``; always for ``all``).
    senders: bool = False
    access: frozenset[str] = frozenset({"read", "write"})
    #: A 1.9.0 principal's scopes, which decide its routes in place of ``access`` (``legacy_scopes``).
    scopes: frozenset[str] | None = None
    unscoped_api: bool = True
    #: The sources an ``observe`` credential posts observations for.
    observe_sources: tuple[str, ...] = ()

    def accepts(self, digest: str, now: datetime) -> bool:
        """Whether a secret with this digest is accepted at ``now``."""
        matched = False
        for candidate, until in self.secrets:
            if hmac.compare_digest(candidate, digest) and _live(until, now):
                matched = True
        return matched and _live(self.accept_until, now)

    @property
    def owner_platforms(self) -> frozenset[str]:
        return frozenset(self.owner)

    @property
    def identity(self) -> str:
        return identity_of(self.name)

    @property
    def instance(self) -> bool:
        return self.name == INSTANCE_CREDENTIAL


def _join(items: Iterable[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def lifetime_note(grant: Grant, now: datetime | None = None) -> str:
    """When a credential's secrets stop being accepted, as a clause to append ("" when never)."""
    return _lifetime(grant, now or now_utc())


def _lifetime(grant: Grant, now: datetime) -> str:
    """When a credential's secrets stop being accepted, for ``describe``."""
    notes = []
    if grant.accept_until is not None:
        at = grant.accept_until.isoformat()
        notes.append(f"accepted until {at}" if _live(grant.accept_until, now) else f"expired at {at}")
    ends = ", ".join(("until " if _live(until, now) else "expired at ") + until.isoformat() if until else
                     "no expiry" for _, until in grant.secrets)
    count = len(grant.secrets)
    if grant.instance and count:
        notes.append(f"also accepts {count} other secret{'s' if count > 1 else ''} ({ends})")
    elif not grant.instance and (count > 1 or any(until for _, until in grant.secrets)):
        notes.append(f"{count} secret{'s' if count > 1 else ''} ({ends})")
    return "".join("; " + note for note in notes)


def _owner_phrase(owner: Iterable[str]) -> str:
    owner = tuple(owner)
    return "the owner from " + ("any platform" if ANY_PLATFORM in owner else _join(owner))


def terminal_owner(grant: Grant) -> bool:
    """Whether ``grant`` writes the owner from the terminal (``cli``, or every platform)."""
    return TERMINAL_PLATFORM in grant.owner or ANY_PLATFORM in grant.owner


#: What a grant on the terminal means, appended to ``describe`` and said by ``colonymind credentials``.
TERMINAL_NOTE = ("from the terminal (cli), anything that can start a Hermes chat on this machine writes as the owner, "
                 "an agent's own shell included, and can answer the owner's asks")


def _default_access(grant: Grant) -> frozenset[str]:
    return frozenset({"read", "write", *(["manage"] if grant.owner else [])})


def _access_note(access: frozenset[str], default: frozenset[str]) -> str:
    if access == default:
        return ""
    doing = [verb for op, verb in (("write", "writes turns"), ("read", "reads context and memory"),
                                   ("manage", "reaches the rest of the API"), ("note", "leaves notes"),
                                   (ANSWER, "answers the owner's hands permit asks"),
                                   (OBSERVE, "posts outside observations"))
             if op in access]
    return "; it " + (_join(doing) if doing else "does nothing") + " and nothing else"


def _scopes_note(scopes: frozenset[str], unscoped_api: bool) -> str:
    listed = ", ".join(sorted(scopes)) if scopes else "none"
    return (f"; it reaches the routes its 1.9.0 scopes reached ({listed})"
            + ("" if unscoped_api else ", the unscoped API not among them"))


def describe(grant: Grant, now: datetime | None = None) -> str:
    """What a credential may write as, and until when, in one sentence (the upgrade report, the sidecar
    log and ``credentials list``)."""
    now = now or now_utc()
    if answers_asks(grant):
        return (f"may only read and answer the owner's hands permit asks, as the owner from {_join(grant.owner)}; "
                f"it writes and reads nothing else{_lifetime(grant, now)}")
    if observes(grant):
        return (f"may only post the owner's outside observations from {_join(grant.observe_sources)}, read and set "
                f"those sources' daily caps and record the conversations observed from them, as the owner from "
                f"{_join(grant.owner)}; it writes and reads nothing else{_lifetime(grant, now)}")
    if TOOL_ORIGIN in grant.access:
        return ("may only claim tool origins and request external-write owner actions; "
                f"it has no owner authority{_lifetime(grant, now)}")
    if HANDS in grant.access:
        start = 'may start owner tasks' if grant.owner else 'tasks await an owner start'
        questions = '; may answer Hand questions' if grant.owner else '; cannot answer Hand questions'
        return f"may only manage hands; {start}{questions}; cannot answer permit asks{_lifetime(grant, now)}"
    note = (_scopes_note(grant.scopes, grant.unscoped_api) if grant.scopes is not None
            else _access_note(grant.access, _default_access(grant))) + _lifetime(grant, now)
    if terminal_owner(grant) and grant.contacts != "own":
        note += "; " + TERMINAL_NOTE
    if grant.contacts == "own":
        return f"may write only as {grant.identity}{note}"
    if grant.contacts == "list":
        persons = ([_owner_phrase(grant.owner)] if grant.owner else []) + list(grant.listed)
        head = f"may write and read only as {_join(persons)}" if persons else "may write and read as no one"
        if grant.senders:
            head += ", and as anyone else a sender resolves to"
        head += (f"; a request that names no one is {grant.viewer}'s" if grant.viewer
                 else "; a request that names no one is refused")
        return head + note
    if grant.owner:
        head = f"may write as {_owner_phrase(grant.owner)}, and as anyone else the sidecar resolves"
    else:
        head = "may write as anyone the sidecar resolves except the owner"
    return f"{head}; any other turn only as {grant.identity}{note}"


@dataclass(frozen=True)
class Authority:
    """What a principal may do, in the terms 1.9.0 and this release share (the upgrade compares them):
    the persons it may name (``*``: anyone the sidecar resolves; the owner, shared and global persons
    appear only when granted), the one a request that names nobody acts as, whether a sender may resolve
    to anyone, its operations, and the platforms it may write the owner from (``*``: every one)."""

    persons: frozenset[str]
    default: str | None
    senders: bool
    operations: frozenset[str]
    owner: frozenset[str]
    #: A 1.9.0 principal's scopes (and ``allow_unscoped_api``), which decide its routes in place of
    #: ``operations``; ``None`` for a credential of this release's kind.
    scopes: frozenset[str] | None = None
    unscoped_api: bool = True


def authority_of(grant: Grant, owner_id: str, shared_id: str = "shared", global_id: str = "global") -> Authority:
    """A credential's :class:`Authority`, as the sidecar enforces it."""
    owner = frozenset(grant.owner)
    lanes = {owner_id, shared_id, global_id}
    if grant.contacts == "own":
        persons: frozenset[str] = frozenset({grant.identity})
        default: str | None = grant.identity
    elif grant.contacts == "all":
        persons = frozenset({"*", *(lanes if owner else ())})
        default = None
    else:
        persons = frozenset({*(set(grant.listed) - {owner_id}), *([owner_id] if owner else [])})
        default = grant.viewer if grant.viewer in persons else None
    return Authority(persons=persons, default=default, senders=grant.senders, operations=grant.access,
                     owner=owner if owner_id in persons or "*" in persons else frozenset(),
                     scopes=grant.scopes, unscoped_api=grant.unscoped_api)


def authority_text(authority: Authority, owner_id: str) -> str:
    """An :class:`Authority` in words, the same for a 1.9.0 principal and a credential (the upgrade report)."""
    if authority.scopes is not None:
        def has(scope: str) -> bool:
            return "*" in authority.scopes or scope in authority.scopes
        ops = [op for op, granted in (("write", has("turns:write")),
                                      ("read", any(has(scope) for scope in ("context:read", "memory:read",
                                                                           "memory:search")))) if granted]
    else:
        ops = [op for op in ("write", "read") if op in authority.operations]
    others = sorted(authority.persons - {owner_id, "*"})
    persons = ([_owner_phrase(sorted(authority.owner))] if authority.owner else []) + (
        ["anyone the sidecar resolves"] if "*" in authority.persons else others)
    named = _join(persons) if persons else ("no one it names" if authority.senders else "no one")
    text = f"may {' and '.join(ops)} as {named}" if ops else f"may neither write nor read (its persons: {named})"
    if authority.senders and "*" not in authority.persons:
        text += ", and as anyone a sender resolves to"
    if "*" not in authority.persons:
        text += f"; a request that names no one is {authority.default}'s" if authority.default else \
            "; a request that names no one is refused"
    if authority.scopes is not None:
        return text + _scopes_note(authority.scopes, authority.unscoped_api)
    if OBSERVE in authority.operations and "manage" not in authority.operations:
        text += "; it posts outside observations for the sources it lists"
    text += "; it reaches the rest of the API" if "manage" in authority.operations else \
        "; it does not reach the rest of the API"
    return text


class CredentialTable:
    """Every credential of an instance, from ``colonymind.yaml``."""

    def __init__(self, grants: Mapping[str, Grant]):
        self._grants = dict(grants)

    @classmethod
    def from_config(cls, credentials: Any) -> "CredentialTable":
        checked = validate_credentials(credentials)
        grants = {}
        for name, entry in checked.items():
            secrets = ([(entry["sha256"], None)] if "sha256" in entry else
                       [(item["sha256"], parse_until(item["accept_until"], name) if "accept_until" in item else None)
                        for item in entry.get("secrets", [])])
            until = parse_until(entry["accept_until"], name) if "accept_until" in entry else None
            owner = cls._owner(entry["owner"])
            listed = isinstance(entry["contacts"], list)
            access = entry.get("access")
            if "scopes" in entry:
                access = []
            elif access is None:
                access = ["read", "write", *(["manage"] if owner else [])]
            grants[name] = Grant(name=name, owner=owner, contacts="list" if listed else entry["contacts"],
                                 secrets=tuple(secrets), accept_until=until,
                                 listed=tuple(entry["contacts"]) if listed else (), viewer=entry.get("viewer"),
                                 senders=entry["contacts"] == "all" or entry.get("senders") == "all",
                                 access=frozenset(access),
                                 scopes=frozenset(entry["scopes"]) if "scopes" in entry else None,
                                 unscoped_api=entry.get("unscoped_api", True),
                                 observe_sources=tuple(entry.get("observe_sources", ())))
        return cls(grants)

    @staticmethod
    def _owner(value: Any) -> tuple[str, ...]:
        return (ANY_PLATFORM,) if value == ANY_OWNER else tuple(value)

    @classmethod
    def load(cls, home: Any = None) -> "CredentialTable":
        """The instance's table; without a ``colonymind.yaml`` the default grants."""
        from colonymind.config import load_config
        return cls.from_config(load_config(home).get("credentials"))

    @property
    def instance(self) -> Grant:
        return self._grants[INSTANCE_CREDENTIAL]

    def grants(self) -> list[Grant]:
        return list(self._grants.values())

    def get(self, name: str) -> Grant | None:
        return self._grants.get(name)

    def match(self, token: str | None, now: datetime | None = None) -> Grant | None:
        """The credential that accepts ``token`` at ``now`` (default: the current time): an issued credential,
        or the instance key for a secret it accepts besides ``api.key`` (checked on its own). An expired
        secret, or any secret of an expired credential, matches nothing."""
        if not token:
            return None
        digest, now = secret_digest(token), now or now_utc()
        found = None
        for grant in self._grants.values():
            if grant.accepts(digest, now):
                found = grant
        return found


def _matches(routes: tuple[tuple[str, str], ...], method: str, path: str) -> bool:
    method = method.upper()
    for allowed_method, allowed in routes:
        if method != allowed_method:
            continue
        if path == allowed or (allowed.endswith("/") and path.startswith(allowed) and len(path) > len(allowed)):
            return True
    return False


def answer_route(method: str, path: str) -> bool:
    """Routes restricted to the issued owner relay: permit asks and the private app password reveal.

    All methods are restricted so unsupported methods cannot bypass that boundary.
    """
    return (path == ANSWER_PREFIX or path.startswith(ANSWER_PREFIX + "/")
            or path.rstrip("/") == SECRET_REVEAL_PATH)


def observe_route(method: str, path: str) -> bool:
    """Whether a route is an ``observe`` route (``OBSERVE_ROUTES``): ``POST /v1/mind/observations/sources/{source}``,
    ``GET`` and ``PUT /v1/mind/observations/sources/{source}/cap`` and ``/v2/host/conversations/observed/{id}``."""
    method = method.upper()
    return any(method in methods and pattern.match(path) for methods, pattern in OBSERVE_ROUTES)


def observed_source(method: str, path: str) -> str | None:
    """The source an ``observe`` route's path names, or None (not an observe route, or the source is in the body)."""
    method = method.upper()
    for methods, pattern in OBSERVE_ROUTES:
        matched = pattern.match(path) if method in methods else None
        if matched:
            return matched.groupdict().get("source")
    return None


def route_operation(method: str, path: str) -> str:
    """The operation a route is: ``write`` (turns), ``read`` (context and memory), ``answer`` (the hands permit
    asks, read and answered by id), ``observe`` (outside observations and their caps; ``manage`` reaches them too)
    ``tool-origin`` (claim origins and request external-write owner actions), or ``manage`` (the rest)."""
    if (method.upper(), path) in TOOL_ORIGIN_ROUTES:
        return TOOL_ORIGIN
    if answer_route(method, path):
        return ANSWER
    if observe_route(method, path):
        return OBSERVE
    if _matches(WRITE_ROUTES, method, path):
        return "write"
    return "read" if _matches(READ_ROUTES, method, path) else "manage"


def route_granted(method: str, path: str) -> bool:
    """A route a credential that may never write as the owner reaches."""
    return route_operation(method, path) in {"write", "read"}


__all__ = ["HANDS", "hands_route", "ANSWER", "ANSWER_PLATFORMS", "ANY_OWNER", "ANY_PLATFORM", "Authority", "answer_route", "answers_asks",
           "CONTACT_SCOPES", "OBSERVE", "OBSERVE_ROUTES", "OPERATIONS", "authority_of", "observe_route",
           "observed_source", "observes", "TOOL_ORIGIN", "TOOL_ORIGIN_ROUTES",
           "authority_text", "lifetime_note", "route_operation", "CredentialTable", "DEFAULT_OWNER_PLATFORMS", "DEVELOPMENT_PRINCIPAL", "Grant",
           "IDENTITY_PREFIX",
           "INSTANCE_CREDENTIAL", "INTEGRATION_ROUTES", "default_credentials", "describe", "identity_of",
           "is_credential_identity", "parse_until", "platform_of", "route_granted", "secret_digest",
           "TERMINAL_NOTE", "TERMINAL_PLATFORM", "terminal_owner", "validate_credentials"]
