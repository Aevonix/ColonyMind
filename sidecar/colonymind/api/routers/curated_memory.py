"""Versioned curated memory, using host authentication and inventory person authority."""
from __future__ import annotations

import asyncio
import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from colonymind.api.public_spec import PUBLIC
from colonymind.api.routers.inventory import granted_person
from colonymind.hands.sessions import refuse_hand_session
from colonymind.memory import curated

router = APIRouter(prefix='/v1/host/memory/curated', tags=['memory', PUBLIC])
Nonblank = Annotated[str, StringConstraints(min_length=1, pattern=r'\S')]
Person = Annotated[Nonblank, Field(max_length=256)]
Content = Annotated[Nonblank, Field(max_length=8192)]
RequestId = Annotated[Nonblank, Field(max_length=128)]
MemoryId = Annotated[str, Path(min_length=1, max_length=256)]
Revision = Annotated[int, Field(strict=True, ge=1)]


class CuratedMutation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    person_id: Person
    request_id: RequestId


class CuratedRemember(CuratedMutation):
    id: str = Field(pattern=r'^mem_[A-Za-z0-9_-]{1,96}$')
    content: Content
    kind: Literal['fact', 'preference'] = 'fact'


class CuratedForget(CuratedMutation):
    expected_revision: Revision


class CuratedCorrect(CuratedForget):
    content: Content


class CuratedSourceForget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    person_id: Person
    source_ids: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(default_factory=list, max_length=100)
    old_text: str | None = Field(default=None, min_length=1, max_length=131072)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)


class CuratedSourceRef(BaseModel):
    source_id: str
    source_version: str
    session_id: str


class CuratedRelationshipRef(BaseModel):
    kind: Literal['contact', 'entity']
    id: str
    relation: Literal['participant', 'subject']
    label: str | None


class CuratedMemory(BaseModel):
    id: str
    person_id: str
    revision: int
    content: str
    kind: Literal['fact', 'preference', 'claim']
    state: Literal['active']
    verification: Literal['owner_asserted', 'unverified', 'owner_confirmed']
    applied_at: str | None
    source_refs: list[CuratedSourceRef]
    relationship_refs: list[CuratedRelationshipRef]


class CuratedReceipt(BaseModel):
    id: str
    revision: int
    applied_at: str
    operation: Literal['remember', 'correct', 'forget']
    request_id: str


class CuratedSearchPage(BaseModel):
    items: list[CuratedMemory]
    next_cursor: str | None
    read_revision: str
    status: Literal['empty', 'ready']


class CuratedSourceReceipt(BaseModel):
    source_erased: Literal[True]
    source_ids: list[str]
    receipt: CuratedReceipt
    scope: Literal['curated_source_tombstone']
    note: str


class CuratedRelationships(BaseModel):
    id: str
    revision: int
    relationship_refs: list[CuratedRelationshipRef]
    status: Literal['empty', 'ready']


_active_ledger = None


def bind_ledger(app, ledger):
    """Only startup supplies this handle; requests must never run ledger migrations."""
    global _active_ledger
    app.state.curated_ledger = _active_ledger = ledger


def is_available():
    try:
        return curated.available(_active_ledger)
    except Exception:
        return False


async def _run(request, operation, *args, **kwargs):
    ledger = getattr(request.app.state, 'curated_ledger', None)
    try:
        if ledger is None or not curated.available(ledger):
            raise sqlite3.OperationalError('curated_backend_unavailable')
        return await asyncio.to_thread(operation, ledger, *args, **kwargs)
    except curated.Refused as exc:
        raise HTTPException(status_code=exc.status, detail={'code': exc.code}) from None
    except Exception as exc:
        if ledger is not None:
            curated.disable(ledger, exc)
        # Never expose private SQL parameters, text, or backend exception messages.
        raise HTTPException(status_code=503, detail={'code': 'curated_backend_unavailable'}) from None


@router.get('/search', response_model=CuratedSearchPage)
async def search(request: Request, person_id: Person,
                 query: str = Query(default='', max_length=4096),
                 cursor: str | None = Query(default=None, max_length=128),
                 limit: int = Query(default=20, ge=1, le=100)):
    person = granted_person(request, person_id)
    return await _run(request, curated.search, person, query=query, cursor=cursor, limit=limit)


@router.post('', response_model=CuratedReceipt, dependencies=[Depends(refuse_hand_session)])
async def remember(body: CuratedRemember, request: Request):
    person = granted_person(request, body.person_id)
    return await _run(request, curated.mutate, person, body.id, 'remember', request_id=body.request_id,
                      content=body.content, kind=body.kind)


@router.post('/sources/forget', response_model=CuratedSourceReceipt, dependencies=[Depends(refuse_hand_session)])
async def forget_sources(body: CuratedSourceForget, request: Request):
    return await _run(request, curated.forget_sources, granted_person(request, body.person_id),
                      source_ids=body.source_ids, old_text=body.old_text, session_id=body.session_id)


@router.get('/{memory_id}', response_model=CuratedMemory)
async def read(request: Request, memory_id: MemoryId, person_id: Person,
               revision: int | None = Query(default=None, ge=1)):
    return await _run(request, curated.read, granted_person(request, person_id), memory_id, revision)


@router.get('/{memory_id}/relationships', response_model=CuratedRelationships)
async def relationships(request: Request, memory_id: MemoryId, person_id: Person,
                        revision: int | None = Query(default=None, ge=1)):
    item = await _run(request, curated.read, granted_person(request, person_id), memory_id, revision)
    refs = item['relationship_refs']
    return {'id': item['id'], 'revision': item['revision'], 'relationship_refs': refs,
            'status': 'ready' if refs else 'empty'}


@router.post('/{memory_id}/correct', response_model=CuratedReceipt, dependencies=[Depends(refuse_hand_session)])
async def correct(body: CuratedCorrect, request: Request, memory_id: MemoryId):
    return await _run(request, curated.mutate, granted_person(request, body.person_id), memory_id, 'correct',
                      request_id=body.request_id, expected_revision=body.expected_revision, content=body.content)


@router.post('/{memory_id}/forget', response_model=CuratedReceipt, dependencies=[Depends(refuse_hand_session)])
async def forget(body: CuratedForget, request: Request, memory_id: MemoryId):
    return await _run(request, curated.mutate, granted_person(request, body.person_id), memory_id, 'forget',
                      request_id=body.request_id, expected_revision=body.expected_revision)
