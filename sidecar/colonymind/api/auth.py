"""Bearer authentication for the sidecar API.

The instance has one key of its own (``api.key``, or ``COLONYMIND_API_KEY``): the
Hermes adapter's credential. ``colonymind credentials add`` issues more. Each is
bound to the contacts and channels it may write as (``colonymind.api.credentials``):
the person a request acts for comes from the request body, as the contact identity
the plugin resolved, and the credential decides whether it may name that person.
Only a credential granted the owner may act as the owner, and a turn attributed to
the owner must arrive from one of that credential's owner platforms.

Without a key the API serves loopback callers only (a development convenience,
see ``middleware.is_local_request``), and ``colonymind start`` refuses to bind a
non-loopback interface.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from dataclasses import dataclass

from fastapi import HTTPException
from starlette.requests import Request

from colonymind.api.credentials import (
    ANSWER,
    ANSWER_PLATFORMS,
    ANY_PLATFORM,
    DEVELOPMENT_PRINCIPAL,
    INSTANCE_CREDENTIAL,
    OBSERVE,
    OPERATIONS,
    Grant,
    identity_of,
)

AUDIENCES = frozenset({"viewer", "owner", "shared", "global"})
KEY_PRINCIPAL = INSTANCE_CREDENTIAL
ANONYMOUS_PRINCIPAL = DEVELOPMENT_PRINCIPAL


@dataclass(frozen=True)
class RequestAuthority:
    """Authority attached to ``request.state`` by the auth middleware."""

    principal_id: str
    authenticated: bool
    anonymous: bool = False
    credential_id: str | None = None
    viewer_person_id: str | None = None
    person_ids: frozenset[str] = frozenset()
    audiences: frozenset[str] = frozenset()
    #: The platforms this credential may attribute a turn to the owner from; empty: never the owner.
    owner_platforms: frozenset[str] = frozenset()
    #: ``all``: anyone the sidecar resolves or the body names; ``own``: only ``identity``; ``list``: exactly
    #: ``person_ids`` (the owner among them only with owner platforms).
    contacts: str = "all"
    #: The credential's own identity, ``credential:<name>``: where a turn that names nobody is written.
    identity: str = ""
    #: A sender may resolve to anyone (the owner, shared and global persons only as granted).
    senders_any: bool = True
    #: What it may do (``colonymind.api.credentials.OPERATIONS``): write turns, read, reach the rest.
    access: frozenset[str] = frozenset(OPERATIONS)
    #: A credential carried from 1.9.0: its scopes decide its routes (``colonymind.api.legacy_scopes``).
    scopes: frozenset[str] | None = None
    unscoped_api: bool = True
    #: The sources an ``observe`` credential posts observations for (``observe_source_granted``).
    observe_sources: frozenset[str] = frozenset()


def owner_person_id() -> str:
    return (
        os.environ.get("COLONYMIND_OWNER_PERSON_ID", "").strip()
        or os.environ.get("COLONYMIND_OWNER_CONTACT_ID", "").strip()
        or "owner"
    )


def audience_person_ids() -> dict[str, str]:
    return {
        "owner": owner_person_id(),
        "shared": os.environ.get("COLONYMIND_SHARED_PERSON_ID", "shared").strip() or "shared",
        "global": os.environ.get("COLONYMIND_GLOBAL_PERSON_ID", "global").strip() or "global",
    }


def key_authority(grant: Grant | None = None) -> RequestAuthority:
    """The authority of a request that presented the instance key, bound by its ``grant``.

    Without a grant (a direct in-process call) every platform may carry the owner.
    """
    owner = owner_person_id()
    return RequestAuthority(
        principal_id=KEY_PRINCIPAL,
        authenticated=True,
        credential_id="api.key",
        viewer_person_id=owner,
        person_ids=frozenset({owner}),
        audiences=AUDIENCES,
        owner_platforms=grant.owner_platforms if grant is not None else frozenset({ANY_PLATFORM}),
        contacts="all",
        identity=identity_of(INSTANCE_CREDENTIAL),
        access=grant.access if grant is not None else frozenset(OPERATIONS),
    )


def credential_authority(grant: Grant) -> RequestAuthority:
    """The authority of a request that presented an issued credential."""
    if grant.contacts == "own":
        viewer, persons, audiences = grant.identity, frozenset({grant.identity}), frozenset({"viewer"})
    elif grant.contacts == "list":
        lanes = {person: lane for lane, person in audience_person_ids().items()}
        owner = owner_person_id()
        persons = frozenset({*(set(grant.listed) - {owner}), *([owner] if grant.owner_platforms else [])})
        viewer = grant.viewer if grant.viewer in persons else None
        audiences = frozenset({"viewer", *(lanes[person] for person in persons if person in lanes)})
    elif grant.owner_platforms:
        owner = owner_person_id()
        viewer, persons, audiences = owner, frozenset({owner}), AUDIENCES
    else:
        viewer, persons, audiences = None, frozenset(), frozenset({"viewer"})
    return RequestAuthority(
        principal_id=grant.name,
        authenticated=True,
        credential_id=grant.name,
        viewer_person_id=viewer,
        person_ids=persons,
        audiences=audiences,
        owner_platforms=grant.owner_platforms,
        contacts=grant.contacts,
        identity=grant.identity,
        senders_any=grant.senders,
        access=grant.access,
        scopes=grant.scopes,
        unscoped_api=grant.unscoped_api,
        observe_sources=frozenset(grant.observe_sources),
    )


def anonymous_authority() -> RequestAuthority:
    """Loopback development mode without a configured key: a principal of its own, granted no owner
    platform, so it never writes or reads as the owner, shared or global person (a sender who is the
    owner included) and changes nothing on the owner's side of the API."""
    return RequestAuthority(principal_id=ANONYMOUS_PRINCIPAL, authenticated=False, anonymous=True,
                            owner_platforms=frozenset(), contacts="all",
                            identity=identity_of(DEVELOPMENT_PRINCIPAL), access=frozenset({"read", "write"}))


def request_authority(request: Request | None) -> RequestAuthority:
    if request is None:
        # Direct in-process router calls are trusted like other internal calls.
        return key_authority()
    value = getattr(request.state, "colonymind_authority", None)
    if isinstance(value, RequestAuthority):
        return value
    # Routers mounted without the middleware in focused unit tests behave like
    # loopback development mode, never like the key.
    return anonymous_authority()


def _authority_error(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=403, detail={"code": code, "message": message})


def reserved_person_ids() -> frozenset[str]:
    """The owner, shared and global persons: only a credential granted the owner acts as them."""
    return frozenset(audience_person_ids().values())


def has_owner_authority(authority: RequestAuthority) -> bool:
    """Whether ``authority`` speaks for the owner at all: a credential granted the owner on at least one
    platform, or an in-process call. Only it acts as the owner, shared or global person, sees the
    unscoped owner view when it names nobody, and reaches the owner's side of the API."""
    return bool(authority.owner_platforms)


def speaks_for_owner(request: Request | None) -> bool:
    """Whether a request may see the owner's view: an authenticated credential granted the owner (an in-process
    call is). Decided by the request's credential alone; a ``viewer``, ``contact_id`` or ``by`` it sends may
    narrow a view to a guest's, never widen one to the owner's. Development mode never speaks for the owner."""
    authority = request_authority(request)
    return authority.authenticated and not authority.anonymous and has_owner_authority(authority)


def owner_platform_granted(authority: RequestAuthority, platform: str) -> bool:
    """Whether ``authority`` may attribute a turn from ``platform`` to the owner."""
    platforms = authority.owner_platforms
    return ANY_PLATFORM in platforms or (bool(platform) and platform in platforms)


def answer_credential(authority: RequestAuthority, platform: str | None = None) -> bool:
    """Whether ``authority`` is an ``answer`` credential (docs/hands-permit-answer-contract.md): an authenticated,
    issued credential (never the instance key, its rotation secrets included, never development mode, never a 1.9.0
    principal) with ``answer`` in its access and an explicit-answer channel among its owner platforms by name (every
    platform, ``*``, is no name). With ``platform``, that channel."""
    if not authority.authenticated or authority.anonymous or authority.scopes is not None:
        return False
    if authority.principal_id == KEY_PRINCIPAL or authority.credential_id == "api.key":
        return False
    if ANSWER not in authority.access:
        return False
    named = authority.owner_platforms - {ANY_PLATFORM}
    if platform is not None:
        return platform in ANSWER_PLATFORMS and platform in named
    return any(item in named for item in ANSWER_PLATFORMS)


def observe_credential(authority: RequestAuthority) -> bool:
    """Whether ``authority`` is an ``observe`` credential: an authenticated, issued credential (never the instance key,
    never development mode, never a 1.9.0 principal) with ``observe`` in its access and not ``manage``, so bound to
    the sources it lists."""
    if not authority.authenticated or authority.anonymous or authority.scopes is not None:
        return False
    if authority.principal_id == KEY_PRINCIPAL or authority.credential_id == "api.key":
        return False
    return OBSERVE in authority.access and "manage" not in authority.access


def observe_source_granted(request: Request | None, source: str) -> bool:
    """Whether the request may post or read the owner's outside observations from ``source``: it speaks for the owner,
    and an ``observe`` credential lists the source (``manage`` with the owner reaches every source). A route that
    takes its source in the body (an observed conversation) asks this before it acts."""
    if not speaks_for_owner(request):
        return False
    authority = request_authority(request)
    return not observe_credential(authority) or str(source or "") in authority.observe_sources


def source_not_granted(authority: RequestAuthority, source: str) -> HTTPException:
    granted = ", ".join(sorted(authority.observe_sources)) or "none"
    return _authority_error(
        "source_not_granted",
        f"credential {authority.principal_id} may not post or read observations from {source}: its sources are "
        f"{granted}")


def answer_credential_required(authority: RequestAuthority) -> HTTPException:
    return _authority_error(
        "answer_credential_required",
        f"credential {authority.principal_id} may not read or answer the owner's hands permit asks: that takes the "
        f"owner channel relay's own credential (access: [answer], owner: [{', '.join(ANSWER_PLATFORMS)}]); the "
        "instance key never answers this way")


def owner_not_granted(authority: RequestAuthority, platform: str | None = None) -> HTTPException:
    where = ""
    if platform is not None:
        where = f" from {platform}" if platform else " from a turn that names no platform"
    granted = sorted(authority.owner_platforms - {ANY_PLATFORM})
    allowed = (f"it may write as the owner only from {', '.join(granted)}" if granted
               else "it never writes as the owner")
    if authority.anonymous:
        return _authority_error(
            "owner_not_granted",
            f"development mode ({authority.principal_id}, no key) may not write as the owner{where}: {allowed}; "
            "write api.key in the instance directory to serve the owner's channels",
        )
    own = f"; a turn that names no contact is written as {authority.identity}" if authority.identity else ""
    return _authority_error(
        "owner_not_granted",
        f"credential {authority.principal_id} may not write as the owner{where}: {allowed}{own}",
    )


def contact_not_granted(authority: RequestAuthority, target: str) -> HTTPException:
    if authority.contacts == "own":
        message = f"credential {authority.principal_id} writes and reads only as {authority.identity}"
    elif authority.contacts == "list":
        granted = ", ".join(sorted(authority.person_ids)) or "no one"
        message = f"credential {authority.principal_id} may not act as {target}: it acts only as {granted}"
    else:
        message = f"credential {authority.principal_id} may not write as {target}, another credential's identity"
    return _authority_error("contact_not_granted", message)


def _granted_person(authority: RequestAuthority, target: str | None) -> str | None:
    """``target`` when the credential may act for it; when it names nobody, the unscoped owner view
    (``None``) for a credential granted the owner and the credential's own identity for any other; else
    the refusal that says what the credential may use. Never another person."""
    reserved = reserved_person_ids()
    if authority.contacts == "own":
        if not target or target == authority.identity:
            return authority.identity
        if target in reserved:
            raise owner_not_granted(authority)
        raise contact_not_granted(authority, target)
    if authority.contacts == "list":
        target = target or authority.viewer_person_id
        if not target:
            raise _authority_error("person_binding_required",
                                   f"credential {authority.principal_id} has no default person: name one of "
                                   f"{', '.join(sorted(authority.person_ids)) or 'no one'}")
        if not person_granted(authority, target):
            if target == owner_person_id():
                raise owner_not_granted(authority)
            raise contact_not_granted(authority, target)
        return target
    if not target:
        return None if has_owner_authority(authority) else authority.identity
    if target in reserved and not has_owner_authority(authority):
        raise owner_not_granted(authority)
    return target


def needs_owner_platform(authority: RequestAuthority, person: str | None) -> bool:
    """Whether writing as ``person`` needs one of the credential's owner platforms: the owner always; the
    shared and global persons too, except for a credential that lists them by name."""
    if authority.contacts == "list":
        return person == owner_person_id()
    return person in reserved_person_ids()


def person_granted(authority: RequestAuthority, person: str | None) -> bool:
    """Whether ``authority`` may act as ``person`` (a sender's contact included), the owner's platforms aside."""
    person = str(person or "").strip()
    if not person:
        return True
    if authority.contacts == "own":
        return person == authority.identity
    if person == authority.identity or person.startswith("credential:"):
        return person == authority.identity and authority.contacts != "list"
    if authority.contacts == "list":
        return person in authority.person_ids
    return person not in reserved_person_ids() or has_owner_authority(authority)


def resolve_request_person(
    request: Request | None,
    *,
    claimed_person_id: str | None = None,
    context_person_id: str | None = None,
    audience: str | None = None,
) -> str | None:
    """Resolve the person a request acts for.

    With a credential the body selects the person within the credential's
    grant: the owner, shared and global persons need a credential granted the
    owner, and a credential bound to its own identity acts only as that
    identity. A body that names nobody is the unscoped owner view (``None``,
    no filter) only for a credential granted the owner; any other credential
    acts as its own identity then. The
    ``viewer`` audience is that same body person (the credential's viewer when
    the body names nobody), which is how the memory provider marks a guest
    prefetch; the owner, shared and global lanes are fixed persons. In
    development mode a default person applies and the authenticated lanes need
    the key.
    """
    authority = request_authority(request)
    claimed = (claimed_person_id or "").strip() or None
    context = (context_person_id or "").strip() or None
    if claimed and context and claimed != context:
        raise _authority_error(
            "person_scope_conflict",
            "person_id and context.contact_id must identify the same person",
        )
    audience = (audience or "").strip().lower() or None
    if audience and audience not in AUDIENCES:
        raise _authority_error("invalid_audience", "unknown audience lane")

    mapped = audience_person_ids()
    lane_person: str | None = None
    if audience:
        if authority.anonymous:
            raise _authority_error(
                "audience_not_granted",
                "development mode cannot select authenticated audience lanes",
            )
        if audience == "viewer":
            lane_person = claimed or context or authority.viewer_person_id
        else:
            lane_person = mapped[audience]
        if not lane_person:
            raise _authority_error("audience_unbound", "the selected audience has no person binding")
        body_person = claimed or context
        if body_person and body_person != lane_person:
            raise _authority_error(
                "person_scope_conflict",
                "body person does not match the selected audience lane",
            )

    target = lane_person or claimed or context
    if authority.authenticated:
        return _granted_person(authority, target)

    target = target or (os.environ.get("COLONYMIND_DEV_PERSON_ID", "dev-anonymous").strip() or "dev-anonymous")
    if target in set(mapped.values()):
        raise _authority_error(
            "reserved_authority_required",
            "owner, shared, and global memory require the API key",
        )
    return target


def resolve_turn_person(
    request: Request | None,
    *,
    context_person_id: str | None,
    has_sender: bool,
) -> str | None:
    """The initial contact of a turn; a structured sender is resolved server-side later.

    A body that names nobody but carries a sender leaves the person to the sender (``None``): a
    credential's own identity is only where a turn that names no one at all is written. A credential
    bound to a list of persons whose senders may be anyone (a 1.9.0 principal that resolved its senders)
    starts a sender's turn from its viewer, whatever the body names, as 1.9.0 did: the sender decides.
    """
    authority = request_authority(request)
    if has_sender and authority.authenticated and authority.contacts == "list" and authority.senders_any:
        if not authority.viewer_person_id:
            raise _authority_error("person_binding_required",
                                   f"credential {authority.principal_id} resolves senders but has no viewer")
        return authority.viewer_person_id
    if (has_sender and authority.authenticated and authority.contacts != "own"
            and not (context_person_id or "").strip()):
        return None
    return resolve_request_person(request, context_person_id=context_person_id)


def query_person_is_granted(authority: RequestAuthority, person_id: str) -> bool:
    """Validate a query-string person selector without trusting it."""
    target = (person_id or "").strip()
    if not target:
        return True
    if authority.contacts in {"own", "list"}:
        return person_granted(authority, target)
    if authority.authenticated:
        return bool(authority.owner_platforms) or target not in reserved_person_ids()
    return target not in reserved_person_ids()


# ---------------------------------------------------------------------------
# Key material and bind safety
# ---------------------------------------------------------------------------

def configured_api_key() -> str | None:
    """``COLONYMIND_API_KEY`` when set, else the instance's ``api.key``."""
    from colonymind.config import read_api_key
    return read_api_key()


def token_matches(token: str | None, expected: str | None) -> bool:
    if not token or not expected:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


def bearer_token(request: Request) -> str:
    """The credential a request presents: ``Authorization: Bearer`` or ``X-API-Key``."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-API-Key", "").strip()


def is_loopback_host(host: str | None) -> bool:
    """True when a bind address only accepts local connections."""
    value = (host or "").strip().lower()
    if value in {"", "localhost"}:
        return True
    try:
        address = ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback
