"""``/v1/host/claims``: what a message would become as memory, for the owner, with nothing written.

``POST /v1/host/claims/dry-run`` runs claim extraction on supplied text in a named person's scope: the prior
records and claim keys that person's memory would offer, the extraction call, every validation step, the review,
and the commit's own checks in a transaction that is rolled back (``SourceClaimProjection.rehearse``). It returns
the accepted proposals (each ``new``, or ``restates`` a live claim, with what it would retract, supersede or hold
for the owner), the rejected ones with their stage and reason, and the tokens the calls reported. ``prompt_only``
returns the prompts extraction would send instead, with no model call. The text is admitted as ingestion admits a
message first: sealed, so no model sees a detected secret, and refused (``status: source_erased``) when it is an
exact copy of an erased message of that session.

It reads that person's records, so it is the owner's: a credential that manages the instance (never development
mode, never a credential carried from 1.9.0), and the person resolved within the credential's grant.

Body: ``{person_id, session_id?, text, occurred_at?, timezone?, prompt_only?}``. ``occurred_at`` (ISO 8601 with
a zone) is when the text would have been said, the clock relative dates count from (default: now); ``timezone``
defaults to the person's.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from colonymind.api.public_spec import PUBLIC

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/host/claims", tags=["claims"])


class DryRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    person_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(default="", max_length=256)
    text: str = Field(min_length=1, max_length=12000)
    occurred_at: Optional[str] = Field(default=None, max_length=64)
    timezone: Optional[str] = Field(default=None, max_length=64)
    prompt_only: bool = False


def _refused() -> HTTPException:
    return HTTPException(status_code=403, detail={
        "code": "dry_run_not_granted",
        "message": "a claim dry run reads a person's memory: it is for a credential that manages the instance"})


def _occurred_at(value: Optional[str]) -> str:
    if value is None:
        from colonymind.util.temporal import now_utc
        return now_utc().isoformat()
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_occurred_at", "message": "occurred_at must be an ISO 8601 time with a zone"})
    return parsed.astimezone(timezone.utc).isoformat()


async def _timezone(person: str, requested: Optional[str]) -> str:
    from colonymind.api.routers import host
    from colonymind.util.temporal import is_valid_timezone, resolve_communication_timezone
    if requested is not None and not is_valid_timezone(requested):
        raise HTTPException(status_code=422, detail={
            "code": "invalid_timezone", "message": "timezone must be an IANA time zone name"})
    contact_tz = None
    if requested is None and host._contacts_store is not None:
        try:
            contact_tz = getattr(await host._contacts_store.get(person), "timezone", None)
        except Exception:
            logger.debug("contact timezone unavailable for a claim dry run", exc_info=True)
    return resolve_communication_timezone(contact_tz, requested)


@router.post("/dry-run", tags=[PUBLIC])
async def claims_dry_run(body: DryRunRequest, request: Request) -> Dict[str, Any]:
    from colonymind import get_state_dir
    from colonymind.api.auth import resolve_request_person
    from colonymind.api.routers import host
    if not host._manages(request):
        raise _refused()
    person = resolve_request_person(request, claimed_person_id=body.person_id)
    host._require_person_authority(request, person)
    occurred_at = _occurred_at(body.occurred_at)
    zone = await _timezone(person, body.timezone)
    model_router = host._llm_router
    if model_router is None and not body.prompt_only:
        raise HTTPException(status_code=503, detail={
            "code": "extraction_unavailable", "message": "no model is configured for claim extraction"})
    from colonymind.beliefs.source_projection import SourceClaimProjection
    from colonymind.secrets.vault import VaultUnavailable
    from colonymind.turns import get_turn_idempotency_ledger
    from colonymind.turns.source_vectors import SourceVectors
    from colonymind.vector import get_pipeline, get_store
    ledger = get_turn_idempotency_ledger(get_state_dir())
    store, pipeline = get_store(), get_pipeline()
    vectors = SourceVectors(ledger, store, pipeline) if store is not None and pipeline is not None else None
    try:
        result = await SourceClaimProjection(ledger, vectors=vectors).rehearse(
            model_router, contact_id=person, session_id=body.session_id, text=body.text,
            occurred_at=occurred_at, timezone_name=zone, prompt_only=body.prompt_only)
    except VaultUnavailable:
        raise host._vault_unavailable() from None
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "extraction_timeout", "message": "claim extraction did not finish in its deadline"}) from None
    except Exception as exc:
        logger.warning("claim dry run failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail={
            "code": "extraction_failed", "message": "claim extraction failed (" + type(exc).__name__ + ")"}) from None
    if result.get("status") == "local_extraction_role_unavailable":
        raise HTTPException(status_code=503, detail={
            "code": "extraction_unavailable", "message": "no local extraction role is configured"})
    return {"person_id": person, "session_id": body.session_id, "occurred_at": occurred_at, "timezone": zone,
            **result}
