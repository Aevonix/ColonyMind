"""``/v1/mcp``: a read-first MCP endpoint bound to the credential that calls it (docs/MCP.md).

An MCP client (a coding agent, a desktop assistant) reaches this instance's memory and mind through the Model Context
Protocol over streamable HTTP. The endpoint keeps no session: every POST carries one JSON-RPC message (or, as the
2025-03-26 transport allows, a batch of them) and its credential, and gets a JSON answer. A GET opens no stream and a
DELETE ends no session (405).

Who the tools act for comes from the credential alone, never from an argument: no tool takes a person. A credential
granted the owner acts as the owner (their memory, the mind's owner view); one bound to a list of persons as its
viewer; any other as its own identity, ``credential:<name>``. Each read calls, in process and with the request's own
authority, the route that serves it, so that route decides the view as for any caller and the endpoint adds no read
path of its own. The mind's and the commitments' reads need only ``read`` here, where their API routes need
``manage``, which also reaches every change:

- ``recall``: the memory search (``POST /v1/host/memory/search``), and on ``explain`` its trace;
- ``open_commitments``: the open commitments, overdue first, every one for the owner, else those the person's own
  words show (by the rule a guest's context uses, up to the tool's limit);
- ``mind_status``, ``why``, ``asks``: the mind's state, one intention's audit entry and the open asks, in the view the
  credential has;
- ``note``: a turn written as ``credential:<name>``, kept as session evidence that never becomes a claim, an appraisal,
  a commitment or a question to the owner, whoever the credential speaks for.

Nothing answers an ask, changes a grant or the autonomy level, or erases, and there is no model behind the endpoint.
The credential's ``access`` decides what it lists: ``read`` the five reads, ``note`` (or ``write``) the note. The
instance key is the Hermes adapter's and is never accepted here; development mode (no key) serves nothing here.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse, Response

from colonymind.api.schemas.host import MEMORY_SEARCH_MAX_LIMIT as RECALL_MAX

logger = logging.getLogger(__name__)

PATH = "/v1/mcp"
#: The protocol versions this endpoint speaks, newest first. ``initialize`` answers with the client's version when it
#: is one of them, else with the newest; a later request naming another in ``MCP-Protocol-Version`` is refused (400).
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
SERVER_NAME = "colonymind"
#: What the memory routes record as the calling host.
HOST_ID = "mcp"
#: The longest note, in characters (my choice: a paragraph or a short list; a document is not a note).
NOTE_MAX_CHARS = 8000
#: The most commitments one call lists (the commitments route's own maximum).
COMMITMENTS_MAX = 200
QUERY_MAX_CHARS = 4096

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602

INSTRUCTIONS = (
    "ColonyMind's memory and mind, read through the credential this client was given: recall searches what memory "
    "holds for the person the credential acts for, open_commitments lists what is still owed, mind_status, why and "
    "asks show the mind's state, the reason for something it did and its open questions. note leaves a note under "
    "the credential's own identity: it is kept as evidence and never becomes anyone's claim. Nothing here answers "
    "an ask; the owner answers in their own chat.")

INSTANCE_KEY_MESSAGE = (
    "the instance key (api.key) is the Hermes adapter's and is not accepted here: issue this client a credential "
    "of its own, for example 'colonymind credentials add <name> --owner-platform mcp --access read --access note' "
    "(docs/MCP.md)")

router = APIRouter()


# -- who may call -----------------------------------------------------------------------------------------------------

def _refusal(status: int, code: str, message: str, *, challenge: bool = False) -> JSONResponse:
    headers = {"WWW-Authenticate": 'Bearer realm="colonymind"'} if challenge else None
    return JSONResponse(status_code=status, content={"detail": {"code": code, "message": message}}, headers=headers)


def origin_allowed(origin: Optional[str]) -> bool:
    """Whether a request's ``Origin`` may reach the endpoint: none (a program, not a browser page), or a loopback one. A
    page that rebinds a name to this machine sends its own origin, and is refused."""
    if origin is None:
        return True
    from colonymind.api.auth import is_loopback_host
    try:
        host = urlsplit(origin.strip()).hostname
    except ValueError:
        return False
    return bool(host) and is_loopback_host(host)


def operations(authority: Any) -> frozenset[str]:
    """The operations a credential's MCP tools may use: its ``access``, where ``write`` includes ``note``. A credential
    carried from 1.9.0 keeps its scopes, which name no MCP tool."""
    if getattr(authority, "scopes", None) is not None:
        return frozenset()
    granted = set(authority.access)
    if "write" in granted:
        granted.add("note")
    return frozenset(granted)


def authenticate(request: Request, *, api_key: Optional[str], credentials: Any) -> Optional[Response]:
    """The auth middleware's check for :data:`PATH`: ``None`` when the request carries an issued credential that is
    granted at least one tool (its authority is then on the request), else the refusal. The instance key and
    development mode never pass: there is no credential of the client's own to bind the tools to."""
    from colonymind.api.auth import bearer_token, credential_authority, token_matches
    if not origin_allowed(request.headers.get("origin")):
        return _refusal(403, "origin_not_allowed", "the MCP endpoint serves programs and pages from this machine "
                        "only: a request from another origin is refused")
    if not api_key:
        return _refusal(401, "credential_required", "this instance has no key (development mode), so it has no "
                        "credential to accept: run 'colonymind init', then issue this client a credential with "
                        "'colonymind credentials add'", challenge=True)
    token = bearer_token(request)
    if token_matches(token, api_key):
        return _refusal(403, "instance_key_refused", INSTANCE_KEY_MESSAGE)
    grant = credentials.match(token)
    if grant is None:
        return _refusal(401, "credential_required", "an issued credential is required ('Authorization: Bearer "
                        "<secret>'); the secret matches none, or its time is up", challenge=True)
    if grant.instance:
        return _refusal(403, "instance_key_refused", INSTANCE_KEY_MESSAGE)
    authority = credential_authority(grant)
    if not granted_tools(authority):
        return _refusal(403, "mcp_not_granted", f"credential {grant.name} may use no MCP tool: the reads need read "
                        "in its access and a note needs note or write ('colonymind credentials set "
                        f"{grant.name} --access read --access note')")
    request.state.colonymind_authority = authority
    return None


# -- the tools --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """One tool: what it lists (``listing``), the operation a credential needs for it, and what it runs."""

    name: str
    title: str
    description: str
    run: Callable[[Request, dict], Awaitable[dict]]
    properties: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    operation: str = "read"

    def listing(self) -> dict:
        schema: dict[str, Any] = {"type": "object", "properties": {name: dict(spec) for name, spec in
                                                                   self.properties.items()},
                                  "additionalProperties": False}
        if self.required:
            schema["required"] = list(self.required)
        reads = self.operation == "read"
        return {"name": self.name, "title": self.title, "description": self.description, "inputSchema": schema,
                "annotations": {"title": self.title, "readOnlyHint": reads, "destructiveHint": False,
                                "idempotentHint": reads, "openWorldHint": False}}


def _person(request: Request) -> str:
    """Who the credential acts for: the owner for a credential granted the owner that names nobody, its viewer for one
    bound to a list of persons (refused without one), else its own identity."""
    from colonymind.api.auth import owner_person_id, resolve_request_person
    return resolve_request_person(request) or owner_person_id()


def _viewer(person: str) -> Optional[str]:
    """The ``viewer`` the mind's reads take: none for the owner (the owner's view, which only a credential granted the
    owner gets), else the person, whose view it narrows to."""
    from colonymind.api.auth import owner_person_id
    return None if person == owner_person_id() else person


def _notes_session(request: Request) -> str:
    """The session a credential's notes are kept in: its recall admits that session's evidence, so a credential bound
    to its own identity reads its notes back."""
    from colonymind.api.auth import request_authority
    return f"{HOST_ID}:{request_authority(request).principal_id}"


async def _recall(request: Request, args: dict) -> dict:
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import HostIdentity, MemorySearchRequest
    person = _person(request)
    body = MemorySearchRequest(identity=HostIdentity(host_id=HOST_ID), person_id=person,
                               session_id=_notes_session(request), query=args["query"],
                               limit=args.get("limit", 5), trace=args.get("explain", False))
    value = (await host.memory_search(body, request)).model_dump(mode="json")
    if args.get("explain"):
        from colonymind.recall_cli import explain
        value["text"] = "\n".join([value.get("content") or "(nothing recalled)", "", *explain(value, person=person)])
    return value


def _commitment(row: Mapping[str, Any]) -> dict:
    return {name: row.get(name) for name in ("id", "person_id", "description", "status", "due_at", "priority",
                                             "created_at")}


def _open_rows(store: Any, person: Optional[str], shown: Callable[[list], list], limit: int) -> list:
    """Up to ``limit`` open commitments of ``person`` (everyone's for None) that ``shown`` keeps: the overdue ones
    first, then the other open ones, each newest first. The order comes before the limit, so an overdue commitment
    older than every pending one is still listed first; the store is read a page at a time until the limit is met,
    so the rows a view leaves out never shorten the list."""
    from colonymind.commitments.store import OPEN_STATUSES
    found: list = []
    for selection in ({"overdue_only": True}, {"status": [name for name in OPEN_STATUSES if name != "overdue"]}):
        offset = 0
        while len(found) < limit:
            listed = store.list(person_id=person, limit=COMMITMENTS_MAX, offset=offset, **selection)
            page = listed.get("commitments", []) if isinstance(listed, dict) else list(listed)
            found.extend(shown(page))
            if len(page) < COMMITMENTS_MAX:
                break
            offset += len(page)
    return found[:limit]


async def _open_commitments(request: Request, args: dict) -> dict:
    from colonymind.api.routers import host
    store = host._commitment_store
    if store is None:
        raise HTTPException(status_code=503, detail={"code": "commitments_not_running",
                                                     "message": "commitment tracking is not running"})
    person = _person(request)
    limit = args.get("limit", 50)
    owner_view = not host._viewer_is_guest(request, person)
    if owner_view:
        rows = await asyncio.to_thread(_open_rows, store, None, list, limit)
    else:
        def shown(page: list) -> list:
            # A guest sees what their own words already say, by their context's rule; never the owner's wording.
            # The rule is asked one row at a time: it also stops a context at five lines, a bound on a context and
            # not on this list.
            return [row for row in page if host._canonical_shared_commitments([row], person)]
        rows = await asyncio.to_thread(_open_rows, store, person, shown, limit)
    items = [_commitment(row) for row in rows]
    lines = [f"- {item['description']} ({item['status']}" + (f", due {item['due_at']}" if item["due_at"] else "")
             + (f", for {item['person_id']}" if owner_view else "") + ")" for item in items]
    return {"commitments": items, "text": "\n".join(lines) if lines else "(no open commitments)"}


async def _mind_status(request: Request, args: dict) -> dict:
    from colonymind.api.routers import mind
    return await mind.status(request, viewer=_viewer(_person(request)))


async def _why(request: Request, args: dict) -> dict:
    from colonymind.api.routers import mind
    return await mind.why(args["intention_id"], request, viewer=_viewer(_person(request)))


async def _asks(request: Request, args: dict) -> dict:
    from colonymind.api.routers import mind
    return await mind.asks(request, viewer=_viewer(_person(request)))


def _as_own_identity(request: Request) -> Request:
    """The request as the credential bound to its own identity: a note is written there, whoever it speaks for, so a
    model's words never reach the owner's or anyone else's memory as theirs."""
    from colonymind.api.auth import request_authority
    authority = request_authority(request)
    own = dataclasses.replace(authority, viewer_person_id=authority.identity,
                              person_ids=frozenset({authority.identity}), audiences=frozenset({"viewer"}),
                              owner_platforms=frozenset(), contacts="own", senders_any=False,
                              access=frozenset({"write"}))
    return Request({"type": "http", "method": "POST", "path": PATH, "headers": [], "query_string": b"",
                    "state": {"colonymind_authority": own}})


async def _note(request: Request, args: dict) -> dict:
    from colonymind.api.auth import request_authority
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import TurnSyncRequest
    identity = request_authority(request).identity
    session = _notes_session(request)
    source = f"{HOST_ID}-note-{uuid.uuid4().hex}"
    body = TurnSyncRequest.model_validate({
        "identity": {"host_id": HOST_ID},
        "context": {"session_id": session, "contact_id": identity, "turn_id": source},
        "user_message": {"role": "user", "content": args["text"]}})
    result = await host.turns_sync(body, _as_own_identity(request), None)
    if not result.accepted or not result.source_recorded:
        raise HTTPException(status_code=409, detail={
            "code": result.skipped_reason or "note_not_kept",
            "message": f"the note was not kept ({result.skipped_reason or 'not recorded'})"})
    # The source it is kept as: what the owner's forget route takes to erase it.
    return {"kept": True, "identity": identity, "session_id": session, "source_id": source,
            "text": f"Noted under {identity}, as evidence only: it never becomes anyone's claim."}


_QUERY = {"type": "string", "minLength": 1, "maxLength": QUERY_MAX_CHARS, "description": "What to look for."}

TOOLS: tuple[Tool, ...] = (
    Tool("recall", "Recall from memory",
         "Search what memory holds for the person this credential acts for, and return the matching evidence with "
         "its sources. explain adds why each candidate was kept or left out.",
         _recall, {"query": _QUERY,
                   "limit": {"type": "integer", "minimum": 1, "maximum": RECALL_MAX, "default": 5,
                             "description": "The most records to return."},
                   "explain": {"type": "boolean", "default": False,
                               "description": "Also explain each candidate's fate."}},
         ("query",)),
    Tool("open_commitments", "Open commitments",
         "What is still owed and open (pending or overdue), overdue first: every open commitment for the owner, "
         "else those the person's own words show.",
         _open_commitments, {"limit": {"type": "integer", "minimum": 1, "maximum": COMMITMENTS_MAX, "default": 50,
                                       "description": "The most commitments to list."}}),
    Tool("mind_status", "The mind's status",
         "Whether the mind is on and its autonomy level; the owner's view adds what it is doing, its open asks, "
         "drives, budgets and faculties.",
         _mind_status),
    Tool("why", "Why the mind acted",
         "The audit entry for one of the mind's intentions (an id from mind_status or asks): what it did or asked, "
         "why, and how it ended.",
         _why, {"intention_id": {"type": "string", "minLength": 1, "maxLength": 256,
                                 "description": "The intention's id."}},
         ("intention_id",)),
    Tool("asks", "Open asks",
         "The mind's open questions to the owner, each with its code, reason and expiry. Read only: an ask is "
         "answered in the owner's own chat.",
         _asks),
    Tool("note", "Leave a note",
         "Leave a note in memory under this credential's own identity. It is kept as evidence, recallable by a "
         "credential bound to its own identity, and never becomes a claim about anyone.",
         _note, {"text": {"type": "string", "minLength": 1, "maxLength": NOTE_MAX_CHARS,
                          "description": "The note, in plain words."}},
         ("text",), operation="note"),
)
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def granted_tools(authority: Any) -> list[Tool]:
    granted = operations(authority)
    return [tool for tool in TOOLS if tool.operation in granted]


def _invalid(tool: Tool, arguments: Any) -> Optional[str]:
    """Why ``arguments`` do not fit the tool's input schema, or None. The schemas are flat: strings, integers and
    booleans, with bounds."""
    if not isinstance(arguments, dict):
        return f"{tool.name} takes an object of arguments"
    unknown = sorted(set(arguments) - set(tool.properties))
    if unknown:
        takes = ", ".join(tool.properties) or "no arguments"
        return (f"{tool.name} takes no argument {unknown[0]!r} (it takes {takes}): who a tool acts for comes from "
                "the credential, never from an argument")
    for name in tool.required:
        if name not in arguments:
            return f"{tool.name} needs {name}"
    for name, value in arguments.items():
        spec = tool.properties[name]
        kind = spec["type"]
        if kind == "string":
            if not isinstance(value, str) or not value.strip():
                return f"{name} is a string with words in it"
            if len(value) > spec.get("maxLength", len(value)):
                return f"{name} is at most {spec['maxLength']} characters"
        elif kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{name} is an integer"
            if not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                return f"{name} is from {spec['minimum']} to {spec['maximum']}"
        elif kind == "boolean" and not isinstance(value, bool):
            return f"{name} is true or false"
    return None


def _tool_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


def _detail_text(detail: Any) -> str:
    if isinstance(detail, Mapping):
        return str(detail.get("message") or detail.get("code") or "refused")
    return str(detail)


def _text(value: Mapping[str, Any]) -> str:
    """A tool result as text: its own rendering (``text``, a recall's ``content``) when it has one, else its JSON."""
    for key in ("text", "content"):
        if isinstance(value.get(key), str):
            return value[key] or "(nothing)"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


# -- the protocol -----------------------------------------------------------------------------------------------------

class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _error(ident: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def _version() -> str:
    from colonymind import __version__
    return __version__


async def _initialize(request: Request, params: dict) -> dict:
    requested = params.get("protocolVersion")
    version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
    return {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "ColonyMind", "version": _version()},
            "instructions": INSTRUCTIONS}


async def _ping(request: Request, params: dict) -> dict:
    return {}


async def _list_tools(request: Request, params: dict) -> dict:
    from colonymind.api.auth import request_authority
    return {"tools": [tool.listing() for tool in granted_tools(request_authority(request))]}


async def _call_tool(request: Request, params: dict) -> dict:
    from colonymind.api.auth import request_authority
    name = params.get("name")
    tool = TOOLS_BY_NAME.get(name) if isinstance(name, str) else None
    if tool is None:
        raise _RpcError(INVALID_PARAMS, f"unknown tool: {name!r} (the tools are {', '.join(TOOLS_BY_NAME)})")
    authority = request_authority(request)
    if tool not in granted_tools(authority):
        needs = "note or write" if tool.operation == "note" else tool.operation
        return _tool_error(f"credential {authority.principal_id} may not use {tool.name}: that needs {needs} in "
                           "its access")
    arguments = params.get("arguments")
    arguments = {} if arguments is None else arguments
    if isinstance(arguments, dict):
        # A model often sends null for an argument it means to leave out: absent, and a required one stays missing.
        arguments = {name: value for name, value in arguments.items() if value is not None}
    problem = _invalid(tool, arguments)
    if problem is not None:
        return _tool_error(problem)
    try:
        value = jsonable_encoder(await tool.run(request, dict(arguments)))
    except HTTPException as exc:
        return _tool_error(_detail_text(exc.detail))
    except Exception:
        logger.warning("the %s tool failed", tool.name, exc_info=True)
        return _tool_error(f"{tool.name} failed inside the sidecar; its log has the reason")
    return {"content": [{"type": "text", "text": _text(value)}], "structuredContent": value, "isError": False}


METHODS: dict[str, Callable[[Request, dict], Awaitable[dict]]] = {
    "initialize": _initialize, "ping": _ping, "tools/list": _list_tools, "tools/call": _call_tool}


def _id_of(message: Any) -> Any:
    ident = message.get("id") if isinstance(message, dict) else None
    return ident if isinstance(ident, (str, int)) and not isinstance(ident, bool) else None


async def _answer(message: Any, request: Request) -> tuple[Optional[dict], bool]:
    """One message's answer (None for a notification or a response), and whether the message was malformed."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(_id_of(message), INVALID_REQUEST, "not a JSON-RPC 2.0 message"), True
    method = message.get("method")
    if method is None:
        if "id" in message and ("result" in message or "error" in message):
            return None, False          # a response to a request this endpoint never sends
        return _error(_id_of(message), INVALID_REQUEST, "a message names a method, or is a response"), True
    if not isinstance(method, str) or not method:
        return _error(_id_of(message), INVALID_REQUEST, "method is a string"), True
    if "id" not in message:
        return None, False              # a notification: taken, and nothing to answer
    ident = message["id"]
    if ident is None or isinstance(ident, bool) or not isinstance(ident, (str, int)):
        return _error(None, INVALID_REQUEST, "id is a string or an integer"), True
    params = message.get("params")
    params = {} if params is None else params
    if not isinstance(params, dict):
        return _error(ident, INVALID_PARAMS, "params is an object"), False
    handler = METHODS.get(method)
    if handler is None:
        return _error(ident, METHOD_NOT_FOUND, f"method not found: {method}"), False
    try:
        return {"jsonrpc": "2.0", "id": ident, "result": await handler(request, params)}, False
    except _RpcError as exc:
        return _error(ident, exc.code, exc.message), False


def _issued_credential(request: Request) -> bool:
    from colonymind.api.auth import KEY_PRINCIPAL, request_authority
    authority = request_authority(request)
    return (authority.authenticated and not authority.anonymous and authority.principal_id != KEY_PRINCIPAL
            and authority.credential_id == authority.principal_id)


@router.post(PATH, summary="The MCP endpoint (docs/MCP.md)")
async def endpoint(request: Request) -> Response:
    """The Model Context Protocol over streamable HTTP, for an issued credential: one JSON-RPC message or a batch,
    answered with JSON; notifications and responses alone get 202. Its tools are listed by ``tools/list``."""
    if not _issued_credential(request):
        # Only the auth middleware puts an issued credential's authority on the request; anything else (the router
        # mounted without it) has no one to act for, and is never served as development mode's caller.
        return _refusal(401, "credential_required", "an issued credential is required", challenge=True)
    version = request.headers.get("mcp-protocol-version")
    if version is not None and version.strip() not in PROTOCOL_VERSIONS:
        return JSONResponse(status_code=400, content=_error(None, INVALID_REQUEST, (
            f"unsupported MCP-Protocol-Version {version.strip()!r}: this endpoint speaks "
            + ", ".join(PROTOCOL_VERSIONS))))
    try:
        payload = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=400, content=_error(None, PARSE_ERROR, "the body is not JSON"))
    batch = isinstance(payload, list)
    if batch and not payload:
        return JSONResponse(status_code=400, content=_error(None, INVALID_REQUEST, "an empty batch"))
    answered = [await _answer(message, request) for message in (payload if batch else [payload])]
    answers = [answer for answer, _ in answered if answer is not None]
    if not answers:
        return Response(status_code=202)
    if batch:
        return JSONResponse(content=answers)
    return JSONResponse(status_code=400 if answered[0][1] else 200, content=answers[0])


@router.get(PATH, include_in_schema=False)
@router.delete(PATH, include_in_schema=False)
async def no_stream() -> Response:
    """No server-sent stream to open and no session to end: every answer comes back on its POST."""
    return Response(status_code=405, headers={"Allow": "POST"})


__all__ = ["PATH", "PROTOCOL_VERSIONS", "TOOLS", "Tool", "authenticate", "granted_tools", "operations",
           "origin_allowed", "router"]
