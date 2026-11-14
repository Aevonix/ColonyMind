"""The owner console's gate and routes: who may see a page, and the one control (docs/CONSOLE.md).

Every request is checked in this order, and the first failure ends it:

1. Host. The request must name the console's own address, ``127.0.0.1:<port>``; anything else is refused (421)
   before any other check, so a page elsewhere that points a name of its own at this machine (DNS rebinding) is
   never answered, and a refused request spends nothing.
2. The session. ``/start?token=<one-time token>`` (the link the console printed) opens one browser session: the
   token is spent, the session's pages are served under a random path of their own (``/<base>/``), and the browser
   receives a random session cookie scoped to that path (``Path=/<base>/``, ``HttpOnly``, ``SameSite=Strict``).
   Every page requires both. Browsers do not keep cookies apart by port, so a cookie for all of 127.0.0.1 would
   also go to any other local service the browser opens; scoped to a path only this session knows, it goes
   nowhere else. A second use of the link opens nothing. Tokens are compared in constant time.
3. The page. Pages are read with GET. A posted form (the one control, ``POST /<base>/off``, the mind's off switch;
   and the recall check's question, ``POST /<base>/recall``, a read whose words stay out of the address) also
   requires the console's own origin (``Origin`` exactly ``http://127.0.0.1:<port>``; ``Sec-Fetch-Site``, when the
   browser sends it, ``same-origin``), a form body, and the session's form token, which only the console's own
   page holds. The off switch is idempotent: a mind already off is left as it is.

Every response, refusals included, carries the page policy (``pages.CSP``: nothing loads but the pinned
stylesheet, no script runs, forms go only to the console, no framing), ``nosniff``, ``no-store`` and a same-origin
referrer policy (a form posted under ``no-referrer`` would carry ``Origin: null`` and could not be told from a
cross-site one). The console never answers an ask and never changes the level or a grant: its source calls only
reads and the off switch (``source.ROUTES``).
"""
from __future__ import annotations

import hmac
import logging
import secrets
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from . import pages
from .pages import CSP, PAGES, BadInput, Context, View
from .source import Source

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
#: The most a request body may hold (a builder's choice: 64 KiB holds the largest form, the recall check's, whose
#: question of at most 4096 characters takes up to 12 bytes a character percent-encoded, with its contact id and the
#: form token).
MAX_BODY = 1 << 16
#: Random bytes in the one-time token, the session cookie and the form token, and in the session's path (a builder's
#: choice: 256 bits for what grants access; 128 for the path, a second lock beside the cookie).
TOKEN_BYTES, PATH_BYTES = 32, 16
FORM = "application/x-www-form-urlencoded"
HEADERS: Tuple[Tuple[str, str], ...] = (
    ("Content-Type", "text/html; charset=utf-8"),
    ("Content-Security-Policy", CSP),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "same-origin"),
    ("Cache-Control", "no-store"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
)


@dataclass
class Response:
    status: int
    body: bytes = b""
    headers: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def all_headers(self) -> List[Tuple[str, str]]:
        return [*HEADERS, *self.headers]


def refusal(status: int, title: str, text: str, *, headers: Iterable[Tuple[str, str]] = ()) -> Response:
    """A page that shows nothing of the instance: no menu, one sentence."""
    return Response(status, pages.page(title, pages.para(text)), list(headers))


def refused(status: int) -> Response:
    """A refusal the HTTP layer makes before the console sees the request (a malformed one, an unknown method)."""
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:
        phrase = "Refused"
    return refusal(status, phrase, "The console did not take this request.")


class Headers:
    """A request's headers by lowercase name. ``one`` is a header sent exactly once (a repeated one is not taken)."""

    def __init__(self, pairs: Iterable[Tuple[str, str]]):
        self._values: Dict[str, List[str]] = {}
        for name, value in pairs:
            self._values.setdefault(name.lower(), []).append(value)

    def one(self, name: str) -> Optional[str]:
        values = self._values.get(name.lower()) or []
        return values[0] if len(values) == 1 else None

    def present(self, name: str) -> bool:
        return bool(self._values.get(name.lower()))

    def all(self, name: str) -> List[str]:
        return list(self._values.get(name.lower()) or [])


def _same(given: str, expected: Optional[str]) -> bool:
    return expected is not None and hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


#: What a marker cannot say: a running mind reads it only when it starts.
UNCONFIRMED = ("A mind running now has not confirmed that it stopped: press the switch again once the sidecar "
               "answers.")


def off_words(result: Dict[str, object]) -> str:
    """What the overview says once, after the off switch."""
    how = result.get("how")
    if how == "now":
        count = int(result.get("cancelled") or 0)
        cancelled = ("No unsent message was waiting." if count == 0 else "1 unsent message was cancelled."
                     if count == 1 else f"{count} unsent messages were cancelled.")
        return (f"The mind is off. {cancelled} Unstarted mind tasks are archived at the body's next tick; running "
                "workers end at their runtime limit.")
    if how == "marker":
        return (f"The sidecar did not take the off switch ({result.get('note')}), so the off marker is written: the "
                f"mind is off from its next start. {UNCONFIRMED}")
    if how == "marker_set":
        return (f"The sidecar still did not take the off switch ({result.get('note')}). The off marker was already "
                f"set, so the mind is off from its next start. {UNCONFIRMED}")
    return "The mind was already off; nothing changed."


class Console:
    """One console: its address, its one-time link, the session that link opens, and the pages."""

    def __init__(self, source: Source, *, port: int):
        self.source = source
        self.port = int(port)
        self.origin = f"http://{HOST}:{self.port}"
        self.hosts = {f"{HOST}:{self.port}"} | ({HOST} if self.port == 80 else set())
        self.cookie = f"colonymind_console_{self.port}"
        self._launch: Optional[str] = secrets.token_urlsafe(TOKEN_BYTES)
        self.url = f"{self.origin}/start?token={self._launch}"
        self._session: Optional[str] = None
        self._base: Optional[str] = None
        self._form: Optional[str] = None
        self._notice: Optional[str] = None
        self._lock = threading.Lock()
        self._off_lock = threading.Lock()

    # -- the gate ---------------------------------------------------------------------------------------------------

    def _has_session(self, headers: Headers) -> bool:
        from http.cookies import CookieError, SimpleCookie
        jar: SimpleCookie = SimpleCookie()
        try:
            jar.load("; ".join(headers.all("cookie")))
        except CookieError:
            return False
        held = jar.get(self.cookie)
        return held is not None and _same(held.value, self._session)

    def handle(self, method: str, target: str, pairs: Iterable[Tuple[str, str]], body: bytes = b"") -> Response:
        headers = Headers(pairs)
        if headers.one("host") not in self.hosts:
            return refusal(421, "Not this address", f"Open the console at the address it printed ({self.origin}).")
        if not target.startswith("/") or target.startswith("//"):
            return refusal(400, "Bad request", "The console takes only its own paths.")
        split = urlsplit(target)
        try:
            query = parse_qs(split.query, max_num_fields=16)
        except ValueError:
            return refusal(400, "Bad request", "The console takes at most a few query fields.")
        if split.path == "/start":
            if method != "GET":
                return refusal(405, "Not allowed", "The link is opened with GET.", headers=[("Allow", "GET")])
            return self._start(query)
        base = self._base
        if base is None or not split.path.startswith(base) or not self._has_session(headers):
            return refusal(403, "No console session", "No console session in this browser. Open the link the "
                           "console printed (it works once); if you just opened it, reload this page.")
        path = "/" + split.path[len(base):]
        if path == "/off":
            if method != "POST":
                return refusal(405, "Not allowed", "The off switch is the button on the overview.",
                               headers=[("Allow", "POST")])
            return self._off(headers, body)
        handler = pages.HANDLERS.get(path)
        if method == "POST" and path in pages.POSTS and handler is not None:
            form = self._posted(headers, body, what=f"The {PAGES[path][0].lower()}")
            if isinstance(form, Response):
                return form
            return self._render(path, handler, form, base, method="POST")
        if method != "GET":
            allow = "GET, POST" if path in pages.POSTS else "GET"
            return refusal(405, "Not allowed", "Pages are read-only.", headers=[("Allow", allow)])
        if handler is None:
            return Response(404, pages.page("Not here", pages.para("The console has no such page."), base=base))
        return self._render(path, handler, query, base)

    def _start(self, query: Dict[str, List[str]]) -> Response:
        given = (query.get("token") or [""])[0]
        with self._lock:
            if self._launch is not None and _same(given, self._launch):
                self._launch = None
                self._session, self._form = secrets.token_urlsafe(TOKEN_BYTES), secrets.token_urlsafe(TOKEN_BYTES)
                self._base = f"/{secrets.token_urlsafe(PATH_BYTES)}/"
                return Response(303, headers=[
                    ("Location", self._base),
                    ("Set-Cookie", f"{self.cookie}={self._session}; Path={self._base}; HttpOnly; SameSite=Strict")])
        return refusal(403, "This link is not valid", "The console's link works once, in one browser. To open "
                       "the console somewhere else, stop it and start it again for a new link.")

    def _posted(self, headers: Headers, body: bytes, *, what: str) -> "Response | Dict[str, List[str]]":
        """A posted form's fields (its token taken out), or the refusal: the console's own origin, a form body and
        the session's form token, as the module's step 3 says. ``what`` names the form in a refusal."""
        if headers.one("origin") != self.origin:
            return refusal(403, "Refused", f"{what} takes only a request from this console's own page.")
        site = headers.one("sec-fetch-site")
        if headers.present("sec-fetch-site") and site != "same-origin":
            return refusal(403, "Refused", f"{what} takes only a request from this console's own page.")
        kind = (headers.one("content-type") or "").split(";", 1)[0].strip().lower()
        if kind != FORM:
            return refusal(415, "Refused", f"{what} takes its page's form.")
        if len(body) > MAX_BODY:
            return refusal(413, "Refused", f"{what} takes its page's form.")
        try:
            form = parse_qs(body.decode("ascii"), max_num_fields=4)
        except (UnicodeDecodeError, ValueError):
            return refusal(400, "Refused", f"{what} takes its page's form.")
        if not _same((form.pop("token", None) or [""])[0], self._form):
            return refusal(403, "Refused", f"{what} takes the form of this browser's console session.")
        return form

    def _off(self, headers: Headers, body: bytes) -> Response:
        form = self._posted(headers, body, what="The off switch")
        if isinstance(form, Response):
            return form
        with self._off_lock:            # one at a time: a second press finds the mind already off
            result = self.source.off()
        with self._lock:
            self._notice = off_words(result)
        return Response(303, headers=[("Location", self._base or "/")])

    # -- the pages --------------------------------------------------------------------------------------------------

    def _render(self, path: str, handler, query: Dict[str, List[str]], base: str, *, method: str = "GET") -> Response:
        notice = None
        if path == "/":
            with self._lock:
                notice, self._notice = self._notice, None
        context = Context(source=self.source, query=query, base=base, form_token=self._form or "", notice=notice,
                          method=method)
        try:
            view = handler(context)
        except BadInput as error:
            view = View(400, PAGES[path][0], pages.para(str(error)[:1].upper() + str(error)[1:] + "."))
        except Exception as error:      # a page that cannot be built says so; the console keeps serving
            logger.warning("console page %s failed: %s", path, type(error).__name__)
            view = View(500, PAGES[path][0], pages.para(
                f"The console could not build this page ({type(error).__name__}); the terminal command for it shows "
                "the same thing."))
        return Response(view.status, pages.page(view.title, view.body, base=base))


__all__ = ["Console", "HEADERS", "HOST", "Headers", "MAX_BODY", "PAGES", "Response", "UNCONFIRMED", "off_words",
           "refusal", "refused"]
