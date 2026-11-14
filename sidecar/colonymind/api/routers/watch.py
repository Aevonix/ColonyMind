"""Owner-only read access to findings from the independent watcher."""
import asyncio
import sqlite3

from fastapi import APIRouter, HTTPException, Request

from colonymind.api.auth import owner_not_granted, request_authority, speaks_for_owner
from colonymind.config import instance_home
from colonymind.jobwatch.store import read_findings
from colonymind.work.hermes_jobs import failed

router = APIRouter(prefix='/v1/host/watch', tags=['watch'])


@router.get('/findings', responses={403: {'description': 'Owner authority required'},
                                    503: {'description': 'Findings store unavailable'}})
async def findings(request: Request):
    if not speaks_for_owner(request):
        raise owner_not_granted(request_authority(request))
    try:
        return await asyncio.to_thread(read_findings, instance_home() / 'jobwatch.db')
    except (OSError, sqlite3.Error, ValueError) as error:
        failed(error)
        raise HTTPException(503, detail='watch_findings_unavailable') from None
