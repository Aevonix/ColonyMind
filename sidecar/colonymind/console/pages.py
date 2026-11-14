"""The owner console's pages: plain HTML with one inline stylesheet pinned by its hash, no script, nothing fetched.

Each page shows what a terminal command shows, through that command's own renderer where it has one: the reports'
documents (``colonymind.reports``), the recall trace in words (``colonymind recall --explain``), a claim's history
(``colonymind memory history``), and the mind's own text for concerns, goals and asks. Every value a page shows
passes through ``shown``: escaped, and with any value the capture detector would seal replaced by the withheld
marker (a reference to a sealed value stays: it names the value, never holds it). A claim is also read as its card
shows it, its predicate beside its value (``wifi password: ...``), so a value its own label calls a credential is
withheld wherever the page holds it, its quotation included (``colonymind.secrets.claims``, which the terminal's
``colonymind memory`` commands read the same way). Nothing here writes; the off switch is the app's
(``app.Console``).

The recall check's question is the one input a person types freely, so it may hold a value the detector would seal.
It is posted (``POSTS``), never put in an address a browser keeps in its history, and every such value is withheld
from it before the search, so the sidecar and any embedder or reranker behind it never receive it.
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

from colonymind.atlas.render import CSS as ATLAS_CSS
from colonymind.atlas.render import esc
from colonymind.reports.text import Document
from colonymind.secrets.claims import withheld_claims
from colonymind.secrets.detect import HIDDEN, replace_secrets

from .source import Source, Unavailable

CSS = ATLAS_CSS + (
    "pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:.6rem 0;padding:.6rem .8rem;border:1px solid var(--rule);"
    "font-size:.86rem}form{margin:1rem 0}label{margin-right:.4rem}input{font:inherit;padding:.2rem .4rem;"
    "margin:0 .6rem .4rem 0}button{font:inherit;padding:.25rem .9rem}form.off button{font-weight:600}"
    ".notice{border-left:3px solid var(--acc);padding:.35rem .8rem}nav a{margin-right:.15rem}")
STYLE_HASH = "sha256-" + base64.b64encode(hashlib.sha256(CSS.encode()).digest()).decode()
#: The page policy: nothing loads but the one stylesheet above, no script runs, forms go only to the console itself,
#: and no other page may frame it.
CSP = f"default-src 'none'; style-src '{STYLE_HASH}'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
#: Every claim state the inventory gives (``colonymind.memory.inventory.STATES``), in words.
STATE_WORDS = {"current": "current", "in_conflict": "in conflict", "upcoming": "upcoming", "expired": "expired",
               "superseded": "superseded", "retracted": "retracted"}
#: An id the console puts in a sidecar path (an intention, a claim): no separator and no dot segment.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._@+-]{0,255}$")
_PAGE_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
#: Log entries on a page by default (a builder's choice: a screen or two), and at most (the route's own bound).
LOG_DEFAULT, LOG_MAX = 50, 500


class BadInput(ValueError):
    """A query the page cannot take; its message says what to change."""


@dataclass
class View:
    status: int
    title: str
    body: str


@dataclass
class Context:
    """What one page request has: the source, its fields (the query of a GET, the checked form of a POST), the
    session's form token and any one-shot notice."""

    source: Source
    query: Dict[str, List[str]]
    base: str = "/"
    form_token: str = ""
    notice: Optional[str] = None
    method: str = "GET"

    def href(self, path: str, **query: Any) -> str:
        return href(self.base, path, **query)

    def link(self, path: str, text: Any, **query: Any) -> str:
        return link(self.base, path, text, **query)

    def param(self, name: str, *, limit: int = 256, required: bool = False, what: str = "") -> Optional[str]:
        values = self.query.get(name) or []
        value = values[0].strip() if values else ""
        if not value:
            if required:
                raise BadInput(what or f"this page needs {name}")
            return None
        if len(value) > limit or _CONTROL.search(value):
            raise BadInput(f"{what or name} takes at most {limit} characters, with no control characters")
        return value


# -- text and the page shell ------------------------------------------------------------------------------------------

def shown(value: Any) -> str:
    """``value`` as a page shows it: escaped, with any value the capture detector would seal withheld."""
    text, _ = replace_secrets("" if value is None else str(value), lambda hit: HIDDEN)
    return esc(text)


def para(text: Any, cls: str = "") -> str:
    attributes = f' class="{cls}"' if cls else ""
    return f"<p{attributes}>{shown(text)}</p>"


def links(*anchors: str) -> str:
    """A paragraph of links built with ``link`` (already escaped)."""
    return "<p>" + " · ".join(anchors) + "</p>"


def pre(text: Any) -> str:
    return f"<pre>{shown(text)}</pre>"


def bullets(items: List[Any]) -> str:
    return "<ul>" + "".join(f"<li>{shown(item)}</li>" for item in items) + "</ul>" if items else ""


def href(base: str, path: str, **query: Any) -> str:
    """A link to a console page (``path``, as ``PAGES`` names it) under the session's own ``base``, query encoded."""
    values = {name: value for name, value in query.items() if value is not None}
    return esc(base + path.lstrip("/") + ("?" + urlencode(values) if values else ""))


def link(base: str, path: str, text: Any, **query: Any) -> str:
    return f'<a href="{href(base, path, **query)}">{shown(text)}</a>'


def document_body(document: Document) -> str:
    """A report's blocks, as ``colonymind report --format html`` renders them, under the page's own title."""
    parts = []
    for block in document.blocks:
        if block.kind == "heading":
            level = min(6, block.level + 1)
            parts.append(f"<h{level}>{shown(block.text)}</h{level}>")
        elif block.kind == "paragraph":
            parts.append(para(block.text))
        else:
            parts.append(bullets(block.items))
    return "".join(parts)


def page(title: str, body: str, *, base: Optional[str] = None) -> bytes:
    """A whole page; with the session's ``base``, the menu of the console's pages above it."""
    menu = ""
    if base:
        menu = "<nav>" + " · ".join(link(base, path, label) for path, (label, _, listed) in PAGES.items()
                                    if listed) + "</nav>"
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="robots" content="noindex, nofollow">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(title)} · Owner console</title><style>{CSS}</style></head>'
            f'<body>{menu}<main><h1>{esc(title)}</h1>{body}</main></body></html>\n').encode()


def failed(title: str, error: Unavailable) -> View:
    """A page whose source did not answer: the sidecar's own words, and the status that says whose it was."""
    status = 404 if error.status == 404 else 400 if error.status == 422 else 502
    return View(status, title, para(error.message[:1].upper() + error.message[1:] + "."))


def page_form(action: str, fields: List[tuple], button: str, *, token: Optional[str] = None) -> str:
    """A form back to the console (``action``, a ``Context.href``): ``(name, label, value, size)`` per field, each
    value shown as ``shown`` shows it. With the session's form ``token`` it is posted (its fields stay out of the
    address); without, it is a GET."""
    inputs = "".join(f'<label>{esc(label)} <input type="text" name="{esc(name)}" value="{shown(value or "")}" '
                     f'size="{size}"></label>' for name, label, value, size in fields)
    if token is None:
        return f'<form method="get" action="{action}">{inputs}<button type="submit">{esc(button)}</button></form>'
    return (f'<form method="post" action="{action}"><input type="hidden" name="token" value="{esc(token)}">'
            f'{inputs}<button type="submit">{esc(button)}</button></form>')


# -- the mind ---------------------------------------------------------------------------------------------------------

def _state_lines(state: Dict[str, Any]) -> List[str]:
    asks = state.get("open_asks")
    if asks is None:
        asks = len(state["asks"]) if isinstance(state.get("asks"), list) else state.get("asks")
    lines = [f"Autonomy: {state.get('autonomy')}.",
             f"Open asks: {asks}; queued tasks: {state.get('queued')}; messages waiting to send: "
             f"{state.get('outbox')}; with the body: {state.get('dispatched')}.",
             f"Ticks: {state.get('ticks')}; the last at {state.get('last_tick') or 'none yet'}; the body last took "
             f"work at {state.get('last_pull') or 'never'}" + (" (the body is stale)." if state.get("body_stale")
                                                                else ".")]
    breaker = state.get("breaker") or []
    if breaker:
        lines.append("Breaker: " + ", ".join(f"{item.get('cls')}: {'tripped' if item.get('tripped') else 'ok'}"
                                             for item in breaker) + ".")
    return lines


def overview(context: Context) -> View:
    source = context.source
    parts = [para(context.notice, "notice")] if context.notice else []
    parts.append("<h2>The mind</h2>")
    try:
        state: Optional[Dict[str, Any]] = source.state()
    except Unavailable as error:
        state, problem = None, error.message
    marked = source.marker.exists()
    if state is not None:
        on: Optional[bool] = state.get("enabled") is not False
        reason = state.get("off_reason")
        parts.append(para("The mind is on." if on else f"The mind is off{f' ({reason})' if reason else ''}."))
        if on and marked:
            parts.append(para("Its off marker is set, so it is off from its next start."))
        parts.append(bullets(_state_lines(state)))
    else:
        # Only the sidecar can say a running mind is off: a marker takes effect when the mind next starts.
        on = None
        parts.append(para(f"The mind's state is unknown here: {problem}."))
        if marked:
            parts.append(para("Its off marker is set, so the mind is off from its next start. A mind running now has "
                              "not confirmed that it stopped: the switch below asks it again."))
    if on is False:
        parts.append(para("It stays off until someone turns it back on from a terminal (colonymind mind on); this "
                          "console cannot."))
    else:
        parts.append(f'<form class="off" method="post" action="{context.href("/off")}">'
                     f'<input type="hidden" name="token" value="{esc(context.form_token)}">'
                     '<button type="submit">Turn the mind off</button></form>')
        parts.append(para("No further effects, at once: unsent messages are cancelled and nothing new is dispatched "
                          "(with the sidecar away, its off marker keeps it off from its next start). It stays off "
                          "until someone runs colonymind mind on in a terminal; this console cannot turn it back on.",
                          "meta"))
    parts.append("<h2>Health</h2>")
    try:
        health = source.health()
        parts.append(para(f"Health: {health.get('status')}."))
        parts.append(bullets(list(health.get("problems") or [])))
    except Unavailable as error:
        parts.append(para(f"Health: unknown ({error.message})."))
    parts.append("<h2>Pages</h2>")
    parts.append("<ul>" + "".join(f"<li>{context.link(path, label)}: {esc(what)}</li>" for path, (label, what, listed)
                                  in PAGES.items() if listed and path != "/") + "</ul>")
    return View(200, "Owner console", "".join(parts))


def concerns(context: Context) -> View:
    try:
        value = context.source.concerns()
    except Unavailable as error:
        return failed("Concerns", error)
    drives = []
    for name, item in (value.get("drives") or {}).items():
        item = item if isinstance(item, dict) else {}
        drives.append(f"{name} {float(item.get('effective') or 0):g} (level {float(item.get('level') or 0):g})")
    body = para("Drives: " + (", ".join(drives) or "none") + ".") + pre(value.get("text") or "(nothing on my mind)")
    return View(200, "Concerns", body + para("A star marks the broadcast set: the concerns carried into the "
                                             "owner's turns and into recall.", "meta"))


def goals(context: Context) -> View:
    try:
        value = context.source.goals()
    except Unavailable as error:
        return failed("Goals", error)
    return View(200, "Goals", pre(value.get("text") or "(no open goals)"))


def log(context: Context) -> View:
    raw = context.param("limit", limit=3)
    if raw is not None and not (raw.isdigit() and 1 <= int(raw) <= LOG_MAX):
        raise BadInput(f"limit takes a number from 1 to {LOG_MAX}")
    try:
        value = context.source.log(int(raw) if raw else LOG_DEFAULT)
    except Unavailable as error:
        return failed("Log", error)
    items = []
    for item in value.get("entries") or []:
        when = (item.get("created_at") or "")[:16].replace("T", " ")
        code = f" [{item['ask_code']}]" if item.get("ask_code") and item.get("status") == "asked" else ""
        ident = str(item.get("id") or "")
        name = context.link("/why", ident, id=ident) if _ID.match(ident) else shown(ident)
        items.append(f"<li>{name} {shown(when)} {shown(item.get('kind'))} {shown(item.get('decision'))} "
                     f"{shown(item.get('outcome') or item.get('status'))}: {shown(item.get('drive'))}/"
                     f"{shown(item.get('type'))}: {shown(item.get('title'))}{shown(code)}</li>")
    body = "<ul>" + "".join(items) + "</ul>" if items else para("(no intentions yet)")
    return View(200, "Log", para("Newest first; an id opens why it was done.", "meta") + body)


def why(context: Context) -> View:
    ident = context.param("id", required=True, what="an intention id (open one from the log)")
    if not _ID.match(ident):
        raise BadInput("an intention id is letters, digits and - . : _ @ +")
    try:
        value = context.source.why(ident)
    except Unavailable as error:
        return failed("Why", error)
    history = [f"{item.get('at')}: {item.get('action')}" for item in value.get("history") or []]
    body = para(value.get("sentence") or value.get("text") or "") + para(f"Intention {ident}.", "meta")
    return View(200, "Why", body + ("<h2>History</h2>" + bullets(history) if history else ""))


def asks(context: Context) -> View:
    try:
        value = context.source.asks()
    except Unavailable as error:
        return failed("Asks", error)
    return View(200, "Asks", pre(value.get("text") or "(no open asks)") + para(
        "The owner answers an ask in their own chat, where it was sent. This console shows the open asks and never "
        "answers one.", "meta"))


def health(context: Context) -> View:
    try:
        value = context.source.health()
    except Unavailable as error:
        return failed("Health", error)
    parts = [para(f"Status: {value.get('status')}.")]
    if value.get("problems"):
        parts += ["<h2>Problems</h2>", bullets(list(value["problems"]))]
    notes = value.get("notes") or {}
    if notes:
        parts += ["<h2>Notes</h2>", bullets([f"{key}: {note}" for key, note in sorted(notes.items())])]
    capabilities = value.get("capabilities") or []
    parts.append(para(f"Capabilities: {', '.join(map(str, capabilities)) or 'none'}.", "meta"))
    return View(200, "Health", "".join(parts))


# -- the reports ------------------------------------------------------------------------------------------------------

def _report(context: Context, action: str, read: Callable[[str], Document]) -> View:
    """A report over a window (``since``, as ``colonymind report`` takes it), with a form to change the window."""
    since = context.param("since", limit=32) or "7d"
    try:
        document = read(since)
    except ValueError as error:
        raise BadInput(str(error)) from None
    form = page_form(context.href(action), [("since", "Since", since, 10)], "Show")
    return View(200, document.title, form + document_body(document))


def faculties(context: Context) -> View:
    return _report(context, "/faculties", context.source.faculties)


def instinct(context: Context) -> View:
    return _report(context, "/instinct", context.source.instinct)


def usage(context: Context) -> View:
    from colonymind.usage import Unreadable
    try:
        return _report(context, "/usage", context.source.usage)
    except Unreadable as error:             # locked or damaged: its usage is unknown, never zero
        return View(503, "Model usage", para(f"Model usage is unknown here: {error}."))


def bench(context: Context) -> View:
    document = context.source.bench()
    if document is None:
        return View(200, "Paired benchmark results", para(
            "No results directory was given. Start the console with --bench DIR (a paired benchmark results "
            "directory) to read it here."))
    return View(200, document.title, document_body(document))


# -- memory -----------------------------------------------------------------------------------------------------------

def recall(context: Context) -> View:
    """The recall check. Its question comes only from the posted form; a value the capture detector would seal is
    withheld from it before the search, and the page says so."""
    from colonymind.recall_cli import explain
    person = context.param("as", what="whose memory (a contact id)")
    posted = context.method == "POST"
    question = context.param("q", limit=4096, what="the question") if posted else None
    withheld = False
    if question:
        question, hits = replace_secrets(question, lambda hit: HIDDEN)
        withheld = bool(hits)
    form = page_form(context.href("/recall"), [("as", "Whose memory", person, 18), ("q", "Question", question, 48)],
                     "Explain", token=context.form_token)
    if not posted and context.query.get("q"):
        return View(400, "Recall check", form + para(
            "A question in the page's address is not read: ask it with the form, which keeps it out of the address "
            "and the browser's history."))
    if not person or not question:
        return View(200, "Recall check", form + para(
            "What memory recalls for a question, and why each candidate is in the packet or not: colonymind recall "
            "--explain, as the owner's credential.", "meta"))
    notice = para("A value the capture detector would seal was withheld from the question: the search ran "
                  "without it.", "notice") if withheld else ""
    try:
        value = context.source.recall(person, question)
    except Unavailable as error:
        view = failed("Recall check", error)
        return View(view.status, view.title, form + notice + view.body)
    body = pre("\n".join(explain(value, person=person))) + "<h2>The packet</h2>" + pre(
        value.get("content") or "(nothing recalled)")
    return View(200, "Recall check", form + notice + body)


def _card(context: Context, card: Dict[str, Any], person: str) -> str:
    source = card.get("source") or {}
    label = "owner confirmed" if card.get("verification") == "owner_confirmed" else "unverified"
    state = str(card.get("state"))
    corrections = int(card.get("corrections") or 0)
    extra = f", {corrections} correction{'' if corrections == 1 else 's'}" if corrections else ""
    ident = str(card.get("claim_id") or "")
    claim = context.link("/memory/history", ident, claim=ident, person=person) if _ID.match(ident) else shown(ident)
    return (f'<li class="line"><span class="state">{shown(STATE_WORDS.get(state, state))}</span> '
            f'<span class="fact">{shown(card.get("predicate"))}: <b>{shown(card.get("value"))}</b></span> '
            f'<span class="meta">({esc(label)})</span><q>{shown(card.get("quote"))}</q>'
            f'<div class="meta">claim {claim}, source {shown(source.get("source_id"))} version '
            f'{shown(str(source.get("source_version") or "")[:12])} (session {shown(source.get("session_id"))}), '
            f'said {shown(card.get("said_at"))}{shown(extra)}</div></li>')


def memory(context: Context) -> View:
    person = context.param("person", what="whose memory (a contact id)")
    token = context.param("page", limit=128)
    if token is not None and not _PAGE_TOKEN.match(token):
        raise BadInput("page takes the next page a listing gave")
    form = page_form(context.href("/memory"), [("person", "Whose memory", person, 24)], "List")
    if not person:
        return View(200, "Memory", form + para(
            "What memory holds about a person, newest first: each claim with its state, its exact words and its "
            "source; a claim opens its history. Corrections go through annotate and removals through forget.",
            "meta"))
    try:
        value = context.source.inventory(person, page=token)
    except Unavailable as error:
        if error.code == "inventory_changed_restart":
            return View(409, "Memory", form + para(
                "The memory changed since this listing began (an erasure or an attribution change): list again "
                "from the first page.") + links(context.link("/memory", "The first page", person=person)))
        view = failed("Memory", error)
        return View(view.status, view.title, form + view.body)
    cards = withheld_claims(value.get("claims") or [])
    parts = [para(f"Memory about {value.get('person_id') or person}: {len(cards)} claim{'' if len(cards) == 1 else 's'}"
                  " on this page, newest first.")]
    parts.append('<ul>' + "".join(_card(context, card, person) for card in cards) + "</ul>" if cards else "")
    if value.get("next_page"):
        parts.append(links(context.link("/memory", "The next page", person=person, page=value["next_page"])))
    else:
        parts.append(para("(end of the listing)", "meta"))
    return View(200, "Memory", form + "".join(parts))


def history(context: Context) -> View:
    from colonymind.memory_cli import render_history
    claim = context.param("claim", required=True, what="a claim id (open one from the memory page)")
    person = context.param("person", what="whose memory (a contact id)")
    if not _ID.match(claim):
        raise BadInput("a claim id is letters, digits and - . : _ @ +")
    try:
        value = context.source.history(claim, person=person)
    except Unavailable as error:
        return failed("Claim history", error)
    owner = value.get("person_id") or person
    back = links(context.link("/memory", f"The memory of {owner}", person=owner)) if owner else ""
    steps = withheld_claims(value.get("steps") or [])
    return View(200, "Claim history", pre("\n".join(render_history({**value, "steps": steps}))) + back)


#: The console's pages: path to (its name, what it shows, whether the menu lists it). The order is the menu's.
PAGES: Dict[str, tuple] = {
    "/": ("Overview", "the mind on or off, its level and queues, health, and the off switch", True),
    "/concerns": ("Concerns", "what is on the mind, and the drives", True),
    "/goals": ("Goals", "the goals the mind adopted that are open", True),
    "/log": ("Log", "the audit log, newest first", True),
    "/why": ("Why", "one intention: drive, evidence, decision, outcome and its history", False),
    "/asks": ("Asks", "the open asks, read only", True),
    "/health": ("Health", "the sidecar's health, its problems in words", True),
    "/faculties": ("Faculties", "what each faculty did over a window, with the benchmark's evidence", True),
    "/instinct": ("Instinct", "what each decision point did per tier over a window", True),
    "/usage": ("Usage", "model calls by task, role and endpoint, in counts", True),
    "/bench": ("Bench", "the paired benchmark's results in plain words", True),
    "/recall": ("Recall check", "what memory recalls for a question and why", True),
    "/memory": ("Memory", "what memory holds about a person", True),
    "/memory/history": ("Claim history", "one claim's own chain", False),
}
#: The pages that also take a posted form (a read whose input stays out of the address), under the off switch's own
#: checks (``app.Console``).
POSTS = frozenset({"/recall"})
HANDLERS: Dict[str, Callable[[Context], View]] = {
    "/": overview, "/concerns": concerns, "/goals": goals, "/log": log, "/why": why, "/asks": asks,
    "/health": health, "/faculties": faculties, "/instinct": instinct, "/usage": usage, "/bench": bench,
    "/recall": recall, "/memory": memory, "/memory/history": history,
}

__all__ = ["BadInput", "CSP", "CSS", "Context", "HANDLERS", "PAGES", "POSTS", "STATE_WORDS", "View", "page", "page_form",
           "shown"]
