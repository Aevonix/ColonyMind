"""Owner-only local mailbox projection. No provider, webhook or send route."""
from pathlib import Path
import sqlite3
import threading

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from colonymind import get_state_dir
from colonymind.agent_mailbox import IdentityUpdate, LocalIdentity, MailboxError, MailboxStore
from colonymind.api.auth import KEY_PRINCIPAL, request_authority, speaks_for_owner


def require_owner(request: Request, response: Response):
    authority = request_authority(request)
    if (authority.principal_id == KEY_PRINCIPAL or authority.credential_id in (None, "api.key")
            or authority.scopes is not None or not speaks_for_owner(request)
            or "companion" not in authority.owner_platforms or "manage" not in authority.access):
        raise HTTPException(403, detail={"code": "mailbox_owner_required"},
                            headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "no-store"


router = APIRouter(prefix="/v1/agent-mail", tags=["agent-mail"], dependencies=[Depends(require_owner)])
_stores: dict[Path, MailboxStore] = {}
_stores_lock = threading.Lock()


def get_store():
    path = (get_state_dir() / "agent-mailbox.db").resolve()
    with _stores_lock:
        if path not in _stores:
            _stores[path] = MailboxStore(path)
        return _stores[path]


def _call(store, method, *args, **kwargs):
    try:
        return getattr(store, method)(*args, **kwargs)
    except MailboxError as error:
        raise HTTPException(error.status, detail={"code": error.code},
                            headers={"Cache-Control": "no-store"}) from None
    except sqlite3.OperationalError as error:
        if getattr(error, "sqlite_errorcode", 0) & 0xff not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise
        raise HTTPException(503, detail={"code": "mailbox_busy"},
                            headers={"Cache-Control": "no-store"}) from None


class ReadBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    through_message_id: str = Field(min_length=1, max_length=128)


@router.get("/mailboxes")
def identities(store=Depends(get_store)):
    return _call(store, "identities")


@router.post("/mailboxes", status_code=201)
def create(body: LocalIdentity, store=Depends(get_store)):
    return _call(store, "create", body)


@router.get("/mailboxes/{mailbox_id}")
def identity(mailbox_id: str, store=Depends(get_store)):
    return _call(store, "identity", mailbox_id)


@router.patch("/mailboxes/{mailbox_id}")
def update(mailbox_id: str, body: IdentityUpdate, store=Depends(get_store)):
    return _call(store, "update", mailbox_id, body)


@router.get("/mailboxes/{mailbox_id}/threads")
def threads(mailbox_id: str, limit: int = Query(50, ge=1, le=100),
            cursor: str | None = Query(None, max_length=1024), store=Depends(get_store)):
    return _call(store, "threads", mailbox_id, limit=limit, cursor=cursor)


@router.get("/mailboxes/{mailbox_id}/threads/{thread_id}")
def thread(mailbox_id: str, thread_id: str, limit: int = Query(50, ge=1, le=100),
           cursor: str | None = Query(None, max_length=1024), store=Depends(get_store)):
    return _call(store, "thread", mailbox_id, thread_id, limit=limit, cursor=cursor)


@router.post("/mailboxes/{mailbox_id}/threads/{thread_id}/read")
def read(mailbox_id: str, thread_id: str, body: ReadBody, store=Depends(get_store)):
    return _call(store, "read", mailbox_id, thread_id, body.through_message_id)


@router.delete("/mailboxes/{mailbox_id}/threads/{thread_id}")
def delete(mailbox_id: str, thread_id: str, store=Depends(get_store)):
    return _call(store, "delete", mailbox_id, thread_id)


def require_agent(request: Request, response: Response):
    authority = request_authority(request)
    if not (authority.authenticated and not authority.anonymous and authority.scopes is None
            and authority.principal_id == KEY_PRINCIPAL):
        raise HTTPException(403, detail={"code": "mailbox_agent_required"},
                            headers={"Cache-Control": "no-store"})
    response.headers["Cache-Control"] = "no-store"


# The one route of the agent's own runtime (the instance key): what arrived since a marker. Read only.
agent_router = APIRouter(prefix="/v1/agent-mail", tags=["agent-mail"], dependencies=[Depends(require_agent)])


@agent_router.get("/new")
def new_mail(since: int | None = Query(None, ge=0), limit: int = Query(20, ge=1, le=100),
             store=Depends(get_store)):
    return _call(store, "new_since", since, limit=limit)
