"""Resolve the authenticated viewer of person-scoped API data."""
import os

from fastapi import HTTPException, Request

from colonymind.api.auth import request_authority, resolve_request_person


def authorized_viewer(request: Request, contact_id: str) -> tuple[str, bool]:
    authority = request_authority(request)
    if not authority.authenticated or authority.anonymous:
        raise HTTPException(403, detail={"code": "scoped_execution_authority_required"})
    person = resolve_request_person(request, claimed_person_id=contact_id)
    owner = os.environ.get("COLONYMIND_OWNER_PERSON_ID", "").strip() or os.environ.get("COLONYMIND_OWNER_CONTACT_ID", "").strip()
    return person, bool(owner and person == owner)
