"""The recorder for callers in another process: ``/v1/recorder``.

The four routes are the recorder's ``begin``, ``finish``, ``once`` and
``mark``. A caller sends with a short timeout and ignores every failure, so
a route never answers bad input with an error: a field that is missing or
is not a string counts as empty, a body that is not a JSON object counts as
an empty one, and the answer is 200 either way. The instance's API key
authorises the calls, like the other routes a local plugin drives.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request

from colonymind import recorder

router = APIRouter(prefix="/v1/recorder", tags=["recorder"])


async def _fields(request: Request) -> dict[str, Any]:
    """The body's JSON object, or an empty one."""
    try:
        body = json.loads(await request.body() or b"{}")
    except Exception:  # noqa: BLE001 - a body that cannot be read counts as empty
        return {}
    return body if isinstance(body, dict) else {}


def _text(fields: dict[str, Any], *names: str) -> dict[str, str]:
    """The named fields as strings: one that is missing or is not a string is empty."""
    return {name: fields[name] if isinstance(fields.get(name), str) else "" for name in names}


@router.post("/begin")
async def begin(request: Request) -> dict[str, str]:
    """Open a line: ``{"kind", "source", "ref"?}`` gives ``{"id"}``."""
    labels = _text(await _fields(request), "kind", "source", "ref")
    return {"id": await asyncio.to_thread(recorder.begin, **labels)}


@router.post("/finish")
async def finish(request: Request) -> dict[str, bool]:
    """Close a line: ``{"id", "result", "choice"?, "chooser"?, "detail"?}`` gives ``{"ok": true}``."""
    fields = await _fields(request)
    labels = _text(fields, "result", "choice", "chooser", "detail")
    await asyncio.to_thread(lambda: recorder.finish(_text(fields, "id")["id"], **labels))
    return {"ok": True}


@router.post("/once")
async def once(request: Request) -> dict[str, str]:
    """Record a settled line in one step: the fields of ``begin`` and ``finish`` and ``took_ms`` give ``{"id"}``."""
    fields = await _fields(request)
    labels = _text(fields, "kind", "source", "result", "ref", "choice", "chooser", "detail")
    took = fields.get("took_ms")
    took_ms = took if isinstance(took, (int, float)) and not isinstance(took, bool) else None
    return {"id": await asyncio.to_thread(lambda: recorder.once(
        labels.pop("kind"), labels.pop("source"), took_ms=took_ms, **labels))}


@router.post("/mark")
async def mark(request: Request) -> dict[str, int]:
    """Add a later label to the lines of one ``ref``: ``{"ref", "label"}`` gives ``{"count"}``."""
    labels = _text(await _fields(request), "ref", "label")
    return {"count": await asyncio.to_thread(recorder.mark, **labels)}
