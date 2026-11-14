"""Owner-only cached fetch; publication happens in the lifecycle worker."""
from fastapi import APIRouter, HTTPException, Request, Response

from colonymind.api.auth import owner_not_granted, request_authority, speaks_for_owner
from colonymind.api.schemas.context_card import ContextCardResponse
from colonymind.context_card.builder import TOKENIZER

router = APIRouter(prefix="/v1/host/context", tags=["context"])


@router.get("/card", response_model=ContextCardResponse, responses={
    304: {"description": "Card bytes and version are unchanged; no response body."},
    403: {"description": "Credential has no owner authority or no read access."},
    503: {"description": "Publication disabled, unavailable, or awaiting background reconciliation."},
})
async def context_card(request: Request, response: Response):
    # Check before cache lookup or conditional GET, including guessed ETags.
    if not speaks_for_owner(request):
        raise owner_not_granted(request_authority(request))
    service = getattr(request.app.state, "context_card", None)
    card = service.fetch() if service is not None else None
    if card is None:
        raise HTTPException(503, detail={"code": "context_card_pending", "message": "No reconciled card is ready"},
                            headers={"Cache-Control": "no-store", "Retry-After": "5"})
    headers = {"ETag": card.etag, "X-Context-Card-Version": str(card.version),
               "Cache-Control": "private, no-cache", "Vary": "Authorization",
               "X-Context-Card-Stale": str(card.stale).lower()}
    etags = [value.strip().removeprefix("W/") for value in request.headers.get("if-none-match", "").split(",")]
    if card.etag.removeprefix("W/") in etags or "*" in etags:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return ContextCardResponse(owner_id=card.owner_id, version=card.version, text=card.text,
        token_count=card.token_count, token_estimator=TOKENIZER, local_day=card.local_day,
        built_at=card.built_at, checked_at=card.checked_at, stale=card.stale)
