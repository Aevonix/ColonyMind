"""Adapter activity ingestion and reconnect snapshot on the existing host API."""
import asyncio
import json
import sqlite3

from fastapi import Depends, HTTPException, Request, Response
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from colonymind import get_state_dir
from colonymind.api.auth import request_authority
from colonymind.events.activity import ActivityStore, ProducerSnapshot
from colonymind.turns import get_turn_idempotency_ledger


def authenticated(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    authority = request_authority(request)
    if not authority.authenticated or authority.anonymous:
        raise HTTPException(403, detail={"code": "activity_auth_required"},
                            headers={"Cache-Control": "no-store"})


def store():
    return ActivityStore(get_turn_idempotency_ledger(get_state_dir()))


def call(method, *args):
    try:
        return getattr(store(), method)(*args)
    except ValueError:
        raise HTTPException(409, detail={"code": "activity_revision_conflict"},
                            headers={"Cache-Control": "no-store"}) from None
    except (sqlite3.Error, OSError):
        raise HTTPException(503, detail={"code": "activity_unavailable"},
                            headers={"Cache-Control": "no-store"}) from None


async def snapshot():
    return await asyncio.to_thread(call, "snapshot")


async def ingest(request: Request):
    try:
        body = ProducerSnapshot.model_validate(await request.json())
    except ClientDisconnect:
        # The adapter gives up on a snapshot after its own short limit and delivers it again later: the hang-up of a
        # sender while its body is read is expected, not a server error. Nothing was read, so nothing changes.
        return Response(status_code=499, headers={"Cache-Control": "no-store"})
    except (ValidationError, ValueError, json.JSONDecodeError):
        raise HTTPException(422, detail={"code": "invalid_activity_snapshot"},
                            headers={"Cache-Control": "no-store"}) from None
    return await asyncio.to_thread(call, "ingest", body)


def register(router):
    # Keep the established host route table flat for existing route consumers.
    router.add_api_route("/activity", snapshot, methods=["GET"], dependencies=[Depends(authenticated)])
    router.add_api_route("/activity/snapshot", ingest, methods=["POST"], dependencies=[Depends(authenticated)])
