"""``/v1/host/memory/inventory``: what memory holds about one person, as claim cards (docs/SOURCE-ANNOTATIONS.md,
The memory inventory).

``GET /v1/host/memory/inventory?person_id=<id>&page=<token>&limit=<n>`` lists the person's source claims, newest
first, each with its state, its quotation and the exact source revision the annotate and forget routes act on
(``colonymind.memory.inventory``). ``GET /v1/host/memory/inventory/claims/{claim_id}/history?person_id=<id>`` walks
one claim's own chain: what it superseded or retracted and what superseded or retracted it, the owner's answers about
it and the corrections attached to its quoted message, each step with its source reference; an erased step is a
content-free marker. Both are memory reads: they write nothing, and changes still go through annotate and forget.

Who may list whom: a credential that manages the instance (the owner's side of the API), anyone it may name; any
other credential only the person it is bound to (its viewer, its listed persons, or its own identity). Development
mode (no key) lists no one. A history that names no person is read in the claim's own person's scope, under the same
rule; a claim the credential may not read is not found.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path, Query, Request

from colonymind.memory import inventory
from colonymind.api.public_spec import PUBLIC

router = APIRouter(prefix="/v1/host/memory/inventory", tags=["memory"])


def _not_granted() -> HTTPException:
    return HTTPException(status_code=403, detail={
        "code": "inventory_not_granted",
        "message": "a credential that does not manage the instance lists only the person it is bound to"})


def granted_person(request: Request, named: Optional[str]) -> str:
    """The person a request may list: resolved within the credential's grant (as every memory read is), and, for a
    credential that does not manage the instance, one it is bound to."""
    from colonymind.api.auth import request_authority, resolve_request_person
    from colonymind.api.routers import host
    person = resolve_request_person(request, claimed_person_id=named)
    host._require_person_authority(request, person)
    if host._manages(request):
        return person
    authority = request_authority(request)
    own = {authority.viewer_person_id, authority.identity}
    if authority.contacts in ("own", "list"):
        own |= set(authority.person_ids)
    if person not in own - {None, ""}:
        raise _not_granted()
    return person


def _ledger():
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    return get_turn_idempotency_ledger(get_state_dir())


@router.get("", tags=[PUBLIC])
async def memory_inventory(request: Request,
                           person_id: str = Query(min_length=1, max_length=256),
                           page: Optional[str] = Query(default=None, max_length=128),
                           limit: int = Query(default=inventory.PAGE_DEFAULT, ge=1, le=inventory.PAGE_MAX)
                           ) -> Dict[str, Any]:
    person = granted_person(request, person_id)
    try:
        return await asyncio.to_thread(inventory.page, _ledger(), person, token=page, limit=limit)
    except inventory.InvalidPage:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_page", "message": "page takes the next_page value a listing returned"}) from None
    except inventory.InventoryChanged:
        raise HTTPException(status_code=409, detail={
            "code": "inventory_changed_restart",
            "message": "the person's memory lost or moved a claim since this listing began (an erasure, an "
                       "attribution change): list again from the first page and drop the earlier pages"}) from None


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail={
        "code": "claim_not_found", "message": "no readable claim with this id in the person's memory"})


@router.get("/claims/{claim_id}/history", tags=[PUBLIC])
async def claim_history(request: Request,
                        claim_id: str = Path(min_length=1, max_length=256),
                        person_id: Optional[str] = Query(default=None, min_length=1, max_length=256)
                        ) -> Dict[str, Any]:
    from colonymind.api.auth import request_authority
    from colonymind.api.routers import host
    ledger = _ledger()
    if person_id is not None:
        person = granted_person(request, person_id)
    else:
        authority = request_authority(request)
        if not authority.authenticated or authority.anonymous:
            host._require_person_authority(request, None)
        owner_of = await asyncio.to_thread(inventory.claim_person, ledger, claim_id)
        if owner_of is None:
            raise _not_found()
        try:
            person = granted_person(request, owner_of)
        except HTTPException as exc:
            if exc.status_code != 403:
                raise
            # The caller named no one: whether another person's claim exists is not its to learn.
            raise _not_found() from None
    found = await asyncio.to_thread(inventory.history, ledger, person, claim_id)
    if found is None:
        raise _not_found()
    return found


__all__ = ["granted_person", "router"]
