"""``colonymind console``: the owner console on loopback (docs/CONSOLE.md).

A local web page over what the terminal already shows: the mind's status, concerns, goals, log, why and asks,
health, the reports, the recall check and the memory inventory. It binds 127.0.0.1 on a random port and prints a
link carrying a one-time token; the link opens one browser session. Every page is read-only, sent under a content
policy that loads nothing and runs no script. Its one control is the mind's off switch: a POST carrying the
session's form token, from the console's own origin, to the console's own host. It never answers an ask and never
changes the level or a grant.

The tests drive a real server on a real socket, with a browser's requests written out by hand (so a wrong Host or a
missing Origin can be sent), and read what a person would see. Every secret-shaped value below is made up and
assembled at run time.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from html import escape as esc_html
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote, quote_plus, urlencode, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient

from onekey import KEY
from colonymind.mind.tick import OFF_MARKER
from report_fixtures import write_bench_dir, write_instance
from test_memory_cli import sidecar  # noqa: F401 (fixture: the CLI's instance beside the inventory's state)
from test_memory_inventory import id_of, make_app, office_chain, say, state  # noqa: F401 (fixture)

DOC = Path(__file__).resolve().parents[2] / "docs" / "CONSOLE.md"
WIFI = "Hunter" + "22!" + "wifi"
RAW = f"the wifi password is {WIFI}"
REFERENCE = "[secret S-" + "ABCDEF23" + ": wifi password]"
ENTRY_ID = "5f0c2a1e-7b1d-4c8e-9a51-0c9d2b7e4a10"


# -- a browser, written out by hand ------------------------------------------------------------------------------------

class Parsed(HTMLParser):
    """What a page holds: its visible text, its tags and their attributes, its links, forms and stylesheets."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.words, self.tags, self.hrefs, self.forms, self.inputs, self.styles = [], [], [], [], [], []
        self._style = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag == "a":
            self.hrefs.append(attributes.get("href") or "")
        elif tag == "form":
            self.forms.append(attributes)
        elif tag == "input":
            self.inputs.append(attributes)
        self._style = tag == "style"

    def handle_endtag(self, tag):
        if tag == "style":
            self._style = False

    def handle_data(self, data):
        (self.styles if self._style else self.words).append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.words).split())


class Page:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body
        self.parsed = Parsed()
        self.parsed.feed(body)

    @property
    def text(self) -> str:
        return self.parsed.text

    def form_token(self) -> str:
        return next(field["value"] for field in self.parsed.inputs if field.get("name") == "token")


class Browser:
    """One browser's requests to a console: the Host it names, the session cookie it holds (sent, as a browser
    sends it, only to paths under the cookie's own path), the headers it sends. Once the session is open, a page
    named as ``PAGES`` names it (``/concerns``) is asked for under the session's own path."""

    def __init__(self, server):
        self.server = server
        self.port = server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.origin = f"http://127.0.0.1:{self.port}"
        self.cookie = None
        self.cookie_path = "/"
        self.base = None

    def path(self, target: str) -> str:
        if self.base and target.startswith("/") and not target.startswith((self.base, "/start")):
            return self.base + target[1:]
        return target

    def request(self, method, target, *, host=True, headers=None, body=b"", cookie=True, raw=False) -> Page:
        target = target if raw else self.path(target)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            connection.putrequest(method, target, skip_host=True, skip_accept_encoding=True)
            if host:
                connection.putheader("Host", self.host if host is True else host)
            if self.cookie and (cookie == "always" or (cookie and target.startswith(self.cookie_path))):
                connection.putheader("Cookie", self.cookie)
            for name, value in (headers or {}).items():
                connection.putheader(name, value)
            if body:
                connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body or None)
            response = connection.getresponse()
            return Page(response.status, response.headers, response.read().decode("utf-8", "replace"))
        finally:
            connection.close()

    def link(self) -> str:
        parts = urlsplit(self.server.url)
        return f"{parts.path}?{parts.query}"

    def open(self) -> Page:
        """Open the link the console printed, and keep the session it gives."""
        page = self.request("GET", self.link())
        assert page.status == 303, page.body
        cookie = page.headers["Set-Cookie"]
        self.cookie = cookie.split(";", 1)[0]
        self.cookie_path = next(part.strip()[5:] for part in cookie.split(";")
                                if part.strip().lower().startswith("path="))
        self.base = page.headers["Location"]
        return page

    def get(self, target, **options) -> Page:
        return self.request("GET", target, **options)

    def post(self, target, fields, *, origin=True, fetch_site="same-origin", host=True, cookie=True,
             kind="application/x-www-form-urlencoded") -> Page:
        """A form posted as a browser posts it from the console's own page (each part can be made wrong)."""
        headers = {"Content-Type": kind}
        if origin:
            headers["Origin"] = self.origin if origin is True else origin
        if fetch_site:
            headers["Sec-Fetch-Site"] = fetch_site
        body = urlencode({name: value for name, value in fields.items() if value is not None}).encode()
        return self.request("POST", target, headers=headers, body=body, host=host, cookie=cookie)

    def off(self, token, **options) -> Page:
        return self.post("/off", {"token": token}, **options)

    def recall(self, token, person, question, **options) -> Page:
        return self.post("/recall", {"token": token, "as": person, "q": question}, **options)


@contextmanager
def running(**options):
    from colonymind.console.server import ConsoleServer
    from colonymind.console.source import Source
    server = ConsoleServer(Source(**options), port=0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield Browser(server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


# -- a sidecar, canned -------------------------------------------------------------------------------------------------

def entry(**extra):
    return {"id": ENTRY_ID, "created_at": "2026-09-20T10:00:00+00:00", "kind": "task", "type": "research",
            "drive": "curiosity", "cls": "internal", "title": f"Look into the tide tables for Sam ({RAW})",
            "decision": "act", "decision_reason": "the curiosity drive", "status": "done", "outcome": "done",
            "ask_code": None, "recipient": None, "message": None, **extra}


CARD = {"claim_id": "claim:lake", "state": "upcoming", "subject": "speaker", "predicate": "office location",
        "value": "Lake", "quote": "Starting 2026-10-01, my office moves to Lake.", "verification": "unverified",
        "source": {"source_id": "o-2", "source_version": "c" * 64, "session_id": "s1", "message_hash": "d" * 64},
        "said_at": "2026-09-02T09:00:00+00:00", "recorded_at": "2026-09-02T09:00:01+00:00",
        "valid_from": "2026-10-01T00:00:00+00:00", "valid_to": None, "operation": "change",
        "prior_claim_id": "claim:river", "superseded_by": None, "retracted_by": None, "corrections": 1}
HISTORY = {
    "person_id": "contact-a", "claim_id": "claim:lake", "complete": False, "limit_reached": True,
    "gaps": [{"claim_id": "claim:gone", "side": "before"}],
    "steps": [
        {"event": "stated", "claim_id": "claim:lake", "claim": CARD},
        {"event": "corrected", "claim_id": "claim:lake", "corrects": "o-2",
         "by": {"correction": "It is Lake Street, not Lake.", "author": "contact-a", "source": {"source_id": "o-9"}}},
    ]}


# A claim whose own label says what its value is: neither field alone looks like a credential, the two together do,
# and its quotation holds the value without the label. A passphrase of several words is one value too.
PASSPHRASE = "blue heron" + " at dawn 42"
LABELLED = [
    {**CARD, "claim_id": "claim:wifi", "state": "current", "predicate": "wifi password", "value": WIFI,
     "quote": f"The wifi at the lake house is {WIFI}, on the router's sticker.", "corrections": 0},
    {**CARD, "claim_id": "claim:boathouse", "state": "current", "predicate": "wifi password", "value": PASSPHRASE,
     "quote": f"At the boathouse the network wants {PASSPHRASE} now.", "corrections": 0},
]


class FakeSidecar:
    """The sidecar routes the console reads, canned, and the mind's off switch; every request is recorded.
    ``down`` makes it unreachable."""

    def __init__(self):
        self.enabled = True
        self.down = False
        self.calls = []

    def state(self):
        return {"enabled": self.enabled, "autonomy": "standard", "off_reason": None if self.enabled else "console",
                "queued": 1, "outbox": 0, "dispatched": 2, "open_asks": 1, "ticks": 42,
                "asks": [{"ask_code": "K7F", "title": "Message Sam"}],
                "last_tick": "2026-09-29T08:00:00+00:00", "last_pull": "2026-09-29T08:00:05+00:00",
                "body_stale": False, "breaker": [{"cls": "owner", "tripped": False}, {"cls": "internal",
                                                                                     "tripped": True}]}

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path, method = request.url.path, request.method
        self.calls.append((method, path, body, dict(request.url.params)))
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        if request.headers.get("authorization") != f"Bearer {KEY}":
            return httpx.Response(401, json={"detail": {"code": "unauthorized", "message": "a key is required"}})
        answer = self.answer(method, path, body, request.url.params)
        if isinstance(answer, httpx.Response):
            return answer
        if answer is None:
            return httpx.Response(404, json={"detail": {"code": "not_found", "message": "no such route"}})
        return httpx.Response(200, json=answer)

    def answer(self, method, path, body, params):
        if (method, path) == ("GET", "/v1/mind/state"):
            return self.state()
        if (method, path) == ("POST", "/v1/mind/off"):
            self.enabled = False
            return {"enabled": False, "reason": body.get("reason"), "cancelled_messages": 2}
        if (method, path) == ("GET", "/v1/mind/concerns"):
            return {"concerns": [], "broadcast": [], "drives": {"duty": {"effective": 0.6, "level": 0.5}},
                    "text": f"* 0.80 duty/commitment: Follow up with Sam about the boat ({RAW})"}
        if (method, path) == ("GET", "/v1/mind/goals"):
            return {"goals": [], "open_goals": 3, "text": "g1  Learn the tide tables (1/3 steps, until 2026-10-01)"}
        if (method, path) == ("GET", "/v1/mind/log"):
            return {"entries": [entry()], "text": "rendered by the sidecar"}
        if method == "GET" and path.startswith("/v1/mind/why/"):
            if path.rsplit("/", 1)[1] != ENTRY_ID:
                return httpx.Response(404, json={"detail": {"code": "unknown_intention",
                                                            "message": "no intention exists with this id"}})
            return {**entry(), "sentence": f"Look into the tide tables: curiosity drive, outcome done ({RAW}).",
                    "history": [{"at": "2026-09-20T10:00:00+00:00", "action": "created", "details": {}},
                                {"at": "2026-09-20T11:00:00+00:00", "action": "done", "details": {}}]}
        if (method, path) == ("GET", "/v1/mind/asks"):
            return {"asks": [{"ask_code": "K7F", "title": "Message Sam"}],
                    "text": f"[K7F] Message Sam (someone new; expires 2026-09-30T10:00:00+00:00)\n"
                            f"  would send to contact-sam: \"Hi Sam, {RAW} {REFERENCE}\""}
        if (method, path) == ("GET", "/v1/host/health"):
            return {"status": "degraded", "api_version": "1.0.0", "capabilities": ["memory"],
                    "problems": ["semantic recall is off: the embedder is not answering"],
                    "notes": {"memory": "Canonical source ledger readable", "secrets": f"sealed store wired ({RAW})"}}
        if (method, path) == ("POST", "/v1/host/memory/search"):
            return {"content": f"Sam said the tide turns at six. {RAW}", "count": 1, "trace": {
                "packet": {"limit": 5, "max_chars": 4000}, "lanes": {"lexical": 2},
                "rerank": {"outcome": "off"}, "calibration": {"status": "none"},
                "rows": [{"turn_id": "t-1", "message_hash": "ab" * 32, "form": "message", "candidate": 1,
                          "fused_rank": 1, "exit": None, "lanes": {"lexical": 1}},
                         {"turn_id": "t-2", "message_hash": "cd" * 32, "form": "message", "candidate": 2,
                          "fused_rank": 2, "exit": "over_packet_limit", "lanes": {"lexical": 2}}]}}
        if (method, path) == ("GET", "/v1/host/memory/inventory"):
            if params.get("page") == "stale":
                return httpx.Response(409, json={"detail": {"code": "inventory_changed_restart",
                                                            "message": "list again from the first page"}})
            if params.get("person_id") == "contact-b":
                return {"person_id": "contact-b", "claims": LABELLED, "read_revision": "e" * 64,
                        "erasure_watermark": 0, "limit": 50, "next_page": None, "complete": True}
            return {"person_id": params.get("person_id"), "claims": [CARD], "read_revision": "e" * 64,
                    "erasure_watermark": 0, "limit": 50, "next_page": "1.1." + "e" * 64, "complete": False}
        if method == "GET" and path == f"/v1/host/memory/inventory/claims/{quote('claim:lake', safe=':')}/history":
            return HISTORY
        for card in LABELLED:
            if method == "GET" and path == f"/v1/host/memory/inventory/claims/{quote(card['claim_id'], safe=':')}/history":
                return {"person_id": "contact-b", "claim_id": card["claim_id"], "complete": True, "gaps": [],
                        "limit_reached": False, "steps": [
                            {"event": "stated", "claim_id": card["claim_id"], "claim": card},
                            {"event": "corrected", "claim_id": card["claim_id"], "corrects": "o-2",
                             "by": {"correction": f"Still {card['value']}, the sticker is right.",
                                    "author": "contact-b", "source": {"source_id": "o-5"}}}]}
        return None


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An instance a week of a running mind left, with its key; nothing of the test's own beside it."""
    instance = write_instance(tmp_path / "instance",
                              datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=1))
    (instance / "api.key").write_text(KEY + "\n")
    monkeypatch.setenv("COLONYMIND_HOME", str(instance))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(instance))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    return instance


@pytest.fixture
def fake(home, monkeypatch):
    canned = FakeSidecar()
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real(transport=httpx.MockTransport(canned.handle),
                                                               timeout=kwargs.get("timeout", 5)))
    return canned


READS = ["/", "/concerns", "/goals", "/log", f"/why?id={ENTRY_ID}", "/asks", "/health", "/faculties",
         "/faculties?since=2d", "/instinct", "/usage", "/bench", "/recall", "/recall?as=contact-a",
         "/memory", "/memory?person=contact-a", "/memory/history?claim=claim%3Alake&person=contact-a"]


def crawl(browser, start=READS):
    """Every page: the ones named, and every link any page shows, once each."""
    seen, queue, pages = set(), list(start), {}
    while queue:
        target = queue.pop(0)
        if target in seen:
            continue
        seen.add(target)
        page = browser.get(target)
        pages[target] = page
        for href in page.parsed.hrefs:
            assert href.startswith("/") and not href.startswith("//"), (target, href)
            if href not in seen:
                queue.append(href)
    return pages


def files(root: Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


# -- 1. it binds loopback only ----------------------------------------------------------------------------------------

def test_it_binds_loopback_only_and_prints_a_link_with_a_one_time_token(home, fake, monkeypatch, capsys):
    from colonymind import cli
    from colonymind.console.server import ConsoleServer
    seen = {}

    def serve_forever(self, poll_interval=0.5):
        seen["address"] = self.server_address

    monkeypatch.setattr(ConsoleServer, "serve_forever", serve_forever)
    monkeypatch.setattr("sys.argv", ["colonymind", "console"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    host, port = seen["address"]
    assert host == "127.0.0.1" and port > 0
    link = re.search(r"http://127\.0\.0\.1:(\d+)/start\?token=([A-Za-z0-9_-]{40,})", out)  # secret-scan: synthetic
    assert link and int(link.group(1)) == port
    assert KEY not in out


def test_there_is_no_option_to_bind_anywhere_else(home, monkeypatch):
    from colonymind import cli
    for argv in (["console", "--host", "0.0.0.0"], ["console", "--bind", "0.0.0.0"]):
        monkeypatch.setattr("sys.argv", ["colonymind", *argv])
        with pytest.raises(SystemExit) as exit_:
            cli.main()
        assert exit_.value.code == 2
    monkeypatch.setenv("COLONYMIND_SIDECAR_HOST", "0.0.0.0")
    with running() as browser:
        assert browser.server.server_address[0] == "127.0.0.1"
        assert browser.server.url.startswith(f"http://127.0.0.1:{browser.port}/start?token=")


# -- 2. a wrong Host header and a missing token are rejected ----------------------------------------------------------

def test_a_wrong_host_is_rejected_before_anything_else(fake):
    with running() as browser:
        link = browser.link()
        for host in ("attacker.example", f"attacker.example:{browser.port}", f"localhost.attacker.example:"
                     f"{browser.port}", "127.0.0.1", f"127.0.0.1:{browser.port + 1}", f"0.0.0.0:{browser.port}"):
            page = browser.get(link, host=host)
            assert page.status == 421, host
            assert "Set-Cookie" not in page.headers
        assert browser.get(link, host=False).status == 421
        # The link was not spent by the refused requests: it still opens the session.
        browser.open()
        assert browser.get("/").status == 200
        for host in ("attacker.example", f"attacker.example:{browser.port}"):
            page = browser.get("/", host=host)
            assert page.status == 421 and "Autonomy" not in page.text and "The mind" not in page.text
        assert not [call for call in fake.calls if call[1] != "/v1/mind/state" and call[1] != "/v1/host/health"]


def test_no_page_opens_without_the_session_the_one_time_link_gives(fake):
    with running() as browser:
        link = browser.link()
        token = parse_qs(urlsplit(browser.server.url).query)["token"][0]
        for target in READS:
            page = browser.get(target)
            assert page.status == 403, target
            assert "Autonomy" not in page.text and "Sam" not in page.text
        for wrong in ("/start", "/start?token=", "/start?token=" + "A" * len(token), f"/start?token={token}x"):
            assert browser.get(wrong).status == 403, wrong
        assert fake.calls == []
        page = browser.open()
        cookie = page.headers["Set-Cookie"]
        assert {part.strip() for part in cookie.split(";")[1:]} == {f"Path={browser.base}", "HttpOnly",
                                                                     "SameSite=Strict"}
        assert token not in cookie and token not in browser.base
        assert browser.get("/").status == 200
        # The link works once: neither another browser nor this one opens it again.
        other = Browser(browser.server)
        assert other.get(link).status == 403 and browser.get(link).status == 403
        assert other.get("/").status == 403
        other.base, other.cookie_path = browser.base, browser.base
        other.cookie = browser.cookie.split("=", 1)[0] + "=" + "B" * 43
        assert other.get("/").status == 403
        for page in crawl(browser).values():
            assert token not in page.body


def test_the_session_lives_under_a_path_only_it_knows(fake):
    """Browsers keep no cookies apart by port: a session cookie for all of 127.0.0.1 would go to every other local
    service the browser opens. The session's cookie is scoped to its own random path, and the pages live there."""
    with running() as browser:
        browser.open()
        base = browser.base
        assert re.fullmatch(r"/[A-Za-z0-9_-]{20,}/", base)
        assert browser.get("/").status == 200 and browser.get(base).status == 200
        for target in ("/", "/concerns", "/off", "/" + "A" * (len(base) - 2) + "/", base[:-2] + "/"):
            assert browser.request("GET", target, raw=True, cookie="always").status == 403, target
        assert browser.request("GET", base, raw=True, cookie=False).status == 403
        for href in browser.get("/").parsed.hrefs:
            assert href.startswith(base), href


# -- 3. the content policy ---------------------------------------------------------------------------------------------

def test_every_response_carries_the_content_policy(fake):
    with running() as browser:
        refused = [browser.get("/", host="attacker.example"), browser.get("/"), browser.get("/start?token=x")]
        browser.open()
        pages = list(crawl(browser).values())
        others = [browser.get("/nowhere"), browser.request("PUT", "/"), browser.request("DELETE", "/off"),
                  browser.get("/off"), browser.request("POST", "/"), browser.get("/why"),
                  browser.get("/why?id=unknown-id")]
        for page in refused + pages + others:
            policy = page.headers.get("Content-Security-Policy") or ""
            directives = {part.strip().split(" ", 1)[0]: part.strip() for part in policy.split(";") if part.strip()}
            assert directives.get("default-src") == "default-src 'none'", page.status
            assert directives.get("frame-ancestors") == "frame-ancestors 'none'"
            assert directives.get("form-action") == "form-action 'self'"
            assert directives.get("base-uri") == "base-uri 'none'"
            assert "script-src" not in directives
            assert page.headers.get("X-Content-Type-Options") == "nosniff"
            assert page.headers.get("Cache-Control") == "no-store"
            assert page.headers.get("Referrer-Policy") == "same-origin"
            assert (page.headers.get("Content-Type") or "").startswith("text/html")
            for style in page.parsed.styles:
                digest = base64.b64encode(hashlib.sha256(style.encode()).digest()).decode()
                assert f"'sha256-{digest}'" in directives.get("style-src", "")
        assert {page.status for page in others} >= {404, 405, 400}


# -- 4. no script in any page ------------------------------------------------------------------------------------------

LOADS = {"script", "img", "link", "iframe", "frame", "object", "embed", "base", "audio", "video", "source", "svg",
         "math", "portal", "meta"}


def test_no_page_holds_a_script_or_loads_anything(fake, tmp_path):
    with running(bench=write_bench_dir(tmp_path / "bench")) as browser:
        browser.open()
        pages = crawl(browser)
    assert len(pages) >= len(READS)
    for target, page in pages.items():
        assert "<script" not in page.body.lower(), target
        assert "javascript:" not in page.body.lower(), target
        for tag, attributes in page.parsed.tags:
            if tag == "meta":
                assert set(attributes) <= {"charset", "name", "content"}, (target, attributes)
                assert attributes.get("name") in (None, "viewport", "robots"), (target, attributes)
                continue
            assert tag not in LOADS, (target, tag)
            for name, value in attributes.items():
                assert not name.startswith("on"), (target, tag, name)
                assert name not in {"src", "srcset", "action", "formaction", "style", "background", "ping"} \
                    or (tag == "form" and name == "action" and value.startswith("/")), (target, tag, name)
        for form in page.parsed.forms:
            assert form.get("action", "").startswith("/") and not form["action"].startswith("//")


# -- 5. off works and is idempotent ------------------------------------------------------------------------------------

def test_off_turns_the_mind_off_once_and_again_changes_nothing(fake):
    with running() as browser:
        browser.open()
        overview = browser.get("/")
        assert "The mind is on" in overview.text
        assert [form.get("method", "").lower() for form in overview.parsed.forms
                if form.get("action") == browser.path("/off")] == ["post"]
        token = overview.form_token()
        page = browser.off(token)
        assert page.status == 303 and page.headers["Location"] == browser.base
        offs = [call for call in fake.calls if call[:2] == ("POST", "/v1/mind/off")]
        assert len(offs) == 1 and offs[0][2]["by"] == "console"
        after = browser.get("/")
        assert "The mind is off" in after.text and "2 unsent messages were cancelled" in after.text
        assert not [form for form in after.parsed.forms if form.get("action") == browser.path("/off")]
        again = browser.off(token)
        assert again.status == 303
        assert len([call for call in fake.calls if call[:2] == ("POST", "/v1/mind/off")]) == 1
        assert "The mind is off" in browser.get("/").text


def test_off_with_the_sidecar_down_writes_the_marker_once(fake, home):
    fake.down = True
    with running() as browser:
        browser.open()
        overview = browser.get("/")
        assert overview.status == 200
        assert "not answering" in overview.text and "unknown" in overview.text
        token = overview.form_token()
        assert browser.off(token).status == 303
        marker = home / OFF_MARKER
        assert marker.is_file()
        written = marker.read_bytes()
        page = browser.get("/")
        assert "next start" in page.text and "not confirmed" in page.text
        assert browser.off(token).status == 303
        assert marker.read_bytes() == written


def test_an_off_marker_never_says_a_running_mind_stopped(fake, home):
    """A sidecar can be running and still not answer the console (busy, or refusing the call). The marker the off
    switch then writes takes effect when the mind next starts, and a running mind reads it only then. So the
    overview never calls the mind off on the marker's word: it says the marker is set, that a running mind has not
    confirmed it stopped, and keeps the switch; a second press asks the sidecar again, never "already off"."""
    fake.down = True                    # the mind runs on (fake.enabled): only the console's calls fail
    with running() as browser:
        browser.open()
        token = browser.get("/").form_token()
        assert browser.off(token).status == 303
        written = (home / OFF_MARKER).read_bytes()
        for page in (browser.get("/"), browser.get("/")):     # the notice, then the overview alone
            assert "The mind is off" not in page.text and "already off" not in page.text
            assert "next start" in page.text and "not confirmed" in page.text
            assert [form for form in page.parsed.forms if form.get("action") == browser.path("/off")]
        assert browser.off(token).status == 303
        again = browser.get("/")
        assert "already off" not in again.text and "not confirmed" in again.text
        assert (home / OFF_MARKER).read_bytes() == written
        assert fake.enabled
        # The sidecar answers again: the mind is on, off from its next start; the same switch now turns it off.
        fake.down = False
        back = browser.get("/")
        assert "The mind is on" in back.text and "off marker is set" in back.text and "next start" in back.text
        assert browser.off(token).status == 303
        done = browser.get("/")
        assert "The mind is off" in done.text and "2 unsent messages were cancelled" in done.text
        assert not [form for form in done.parsed.forms if form.get("action") == browser.path("/off")]
    assert not fake.enabled


def test_off_takes_only_the_sessions_own_form_from_the_consoles_own_page(fake):
    with running() as browser:
        browser.open()
        token = browser.get("/").form_token()
        refused = [
            browser.off(token, origin=False),
            browser.off(token, origin="null"),
            browser.off(token, origin="http://attacker.example"),
            browser.off(token, origin=f"http://localhost:{browser.port}"),
            browser.off(token, fetch_site="cross-site"),
            browser.off(token, fetch_site="same-site"),
            browser.off(None),
            browser.off("A" * len(token)),
            browser.off(token, cookie=False),
            browser.off(token, host="attacker.example"),
            browser.off(token, kind="text/plain"),
            browser.off(token, kind="multipart/form-data; boundary=x"),
            browser.request("POST", "/off", headers={"Origin": browser.origin, "Content-Type":
                                                     "application/x-www-form-urlencoded"},
                            body=b"token=" + token.encode() + b"&x=" + b"y" * 70000),
            browser.get(f"/off?token={token}"),
        ]
        for page in refused:
            assert page.status in (403, 405, 413, 415, 421), page.status
        assert not [call for call in fake.calls if call[:2] == ("POST", "/v1/mind/off")]
        assert "The mind is on" in browser.get("/").text


# -- the console never answers, never changes the level or a grant -----------------------------------------------------

def test_the_console_calls_only_reads_and_the_off_switch(fake):
    with running() as browser:
        browser.open()
        pages = crawl(browser)
        assert browser.recall(pages["/recall"].form_token(), "contact-a", "tide times").status == 200
        browser.off(pages["/"].form_token())
    posts = {(method, path) for method, path, _, _ in fake.calls if method != "GET"}
    assert posts == {("POST", "/v1/host/memory/search"), ("POST", "/v1/mind/off")}
    paths = {path for _, path, _, _ in fake.calls}
    assert not [path for path in paths if path.startswith(("/v1/secrets", "/v1/mind/level", "/v1/mind/grants",
                                                            "/v1/mind/allows", "/v1/mind/answers", "/v1/mind/on",
                                                            "/v1/mind/asks/"))]
    # The asks page reads the asks and offers nothing to answer them with.
    asks = pages["/asks"]
    assert "[K7F] Message Sam" in asks.text
    assert asks.parsed.forms == [] and not [field for field in asks.parsed.inputs]


def test_the_source_refuses_any_route_it_is_not_given(fake):
    """Refused before any request is made: nothing reaches even the canned sidecar."""
    from colonymind.console.source import Source, Unavailable
    source = Source()
    for method, path in (("POST", "/v1/mind/asks/K7F/yes"), ("POST", "/v1/mind/level"), ("POST", "/v1/mind/on"),
                         ("GET", "/v1/secrets/S-ABCDEF23"), ("POST", "/v1/mind/grants"),
                         ("GET", "/v1/mind/why/../asks"), ("GET", "/v1/mind/why/..%2Fasks"),
                         ("GET", "/v1/mind/why/%2E%2E"), ("GET", "/v1/mind/why/"), ("GET", "/v1//mind/state"),
                         ("GET", "/v1/host/memory/inventory/claims/claim:x"),
                         ("GET", "/v1/host/memory/inventory/claims/a/b/history")):
        with pytest.raises(Unavailable) as refused:
            source.call(method, path)
        assert refused.value.code == "not_a_console_route", path
    assert fake.calls == []


# -- 6. no sealed secret is ever rendered ------------------------------------------------------------------------------

def test_no_secret_value_the_capture_would_seal_is_ever_shown(fake):
    with running() as browser:
        browser.open()
        session = browser.cookie.split("=", 1)[1]
        pages = crawl(browser)
        pages["after off"] = browser.get("/") if browser.off(pages["/"].form_token()).status == 303 else None
    for target, page in pages.items():
        assert WIFI not in page.body, target
        assert KEY not in page.body and session not in page.body, target
    # A reference names a sealed value as the stored words hold it; the value itself never leaves the vault.
    assert REFERENCE in pages["/asks"].text
    assert "(a private value, withheld)" in pages["/log"].text


def test_a_claim_card_never_shows_a_value_its_own_label_calls_a_credential(fake):
    """A claim's predicate can say what its value is ("wifi password") though neither field alone looks like a
    credential, and its quotation (or a later correction) can hold the value without the label. The card is read as
    it shows the two together, and the value is withheld wherever the page holds it: in the listing and in the
    claim's history. No form of the value is anywhere in the page, markup included."""
    with running() as browser:
        browser.open()
        listing = browser.get("/memory?person=contact-b")
        assert listing.status == 200 and "wifi password" in listing.text
        assert "The wifi at the lake house is (a private value, withheld), on the router's sticker." in listing.text
        assert "At the boathouse the network wants (a private value, withheld) now." in listing.text
        histories = [browser.get(f"/memory/history?{urlencode({'claim': card['claim_id'], 'person': 'contact-b'})}")
                     for card in LABELLED]
        for page in [listing, *histories]:
            assert page.status == 200
            for value in (WIFI, PASSPHRASE):
                assert value not in page.body and esc_html(value) not in page.body
                assert quote(value) not in page.body and quote_plus(value) not in page.body
        for page in histories:
            assert "wifi password: (a private value, withheld)" in page.text
            assert "Still (a private value, withheld), the sticker is right." in page.text


def test_a_secret_in_the_recall_question_never_leaves_the_console(fake):
    """A question can hold a value the capture detector would seal (pasted, or asked about). The value is withheld
    before the search, so neither the sidecar nor an embedder or reranker behind it receives it; the page never shows
    it back; and the question travels in a posted form of the session's own page, never in an address a browser
    keeps in its history."""
    with running() as browser:
        browser.open()
        empty = browser.get("/recall")
        forms = [form for form in empty.parsed.forms if form.get("action") == browser.path("/recall")]
        assert [form.get("method", "get").lower() for form in forms] == ["post"]
        token = empty.form_token()
        page = browser.recall(token, "contact-a", f"did I say {RAW}?")
        assert page.status == 200 and "Sam said the tide turns at six." in page.text
        assert WIFI not in page.body and quote(WIFI) not in page.body
        assert "withheld from the question" in page.text
        searches = [body for method, path, body, _ in fake.calls if path == "/v1/host/memory/search"]
        assert len(searches) == 1 and WIFI not in json.dumps(searches[0])
        assert searches[0]["query"] == "did I say the wifi password is (a private value, withheld)"
        # A question in an address is not read: nothing is searched and nothing of it is shown back.
        page = browser.get("/recall?" + urlencode({"as": "contact-a", "q": f"did I say {RAW}?"}))
        assert page.status == 400 and WIFI not in page.body and "form" in page.text
        # The recall form takes only the session's own posted form, from the console's own page.
        for refused in (browser.recall("A" * len(token), "contact-a", "tide"),
                        browser.recall(None, "contact-a", "tide"),
                        browser.recall(token, "contact-a", "tide", origin="http://attacker.example"),
                        browser.recall(token, "contact-a", "tide", origin=False),
                        browser.recall(token, "contact-a", "tide", fetch_site="cross-site"),
                        browser.recall(token, "contact-a", "tide", kind="text/plain"),
                        browser.recall(token, "contact-a", "tide", cookie=False)):
            assert refused.status in (403, 415), refused.status
    assert len([call for call in fake.calls if call[1] == "/v1/host/memory/search"]) == 1


@pytest.mark.asyncio
async def test_a_sealed_message_never_reaches_the_memory_pages(sidecar, state, monkeypatch):
    await say(state, "n-1", f"Remember {RAW} at home.")
    await say(state, "n-2", "My office is in River.")
    app = make_app()
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: TestClient(app))
    with running() as browser:
        browser.open()
        page = browser.get("/memory?person=contact-a")
        assert page.status == 200 and "office location" in page.text
        assert WIFI not in page.body


# -- what it shows ----------------------------------------------------------------------------------------------------

def test_the_mind_pages_show_what_the_terminal_shows(fake):
    with running() as browser:
        browser.open()
        overview = browser.get("/")
        assert "The mind is on" in overview.text and "standard" in overview.text
        assert "internal: tripped" in overview.text and "Health: degraded" in overview.text
        log = browser.get("/log")
        assert browser.path(f"/why?id={ENTRY_ID}") in log.parsed.hrefs and "curiosity/research" in log.text
        why = browser.get(f"/why?id={ENTRY_ID}")
        assert why.status == 200 and "Look into the tide tables: curiosity drive" in why.text
        assert "created" in why.text and "done" in why.text
        assert browser.get("/why?id=unknown-id").status == 404
        assert browser.get("/why?id=../asks").status == 400
        assert "Follow up with Sam about the boat" in browser.get("/concerns").text
        assert "Learn the tide tables" in browser.get("/goals").text
        health = browser.get("/health")
        assert "degraded" in health.text and "semantic recall is off" in health.text
    assert ("GET", "/v1/mind/log", None, {"limit": "50"}) in fake.calls


def test_the_recall_check_explains_a_search_stage_by_stage(fake):
    with running() as browser:
        browser.open()
        form = browser.get("/recall")
        assert [form.get("method", "get").lower() for form in form.parsed.forms] == ["post"]
        token = form.form_token()
        page = browser.recall(token, "contact-a", "when does the tide turn")
        assert page.status == 200
        assert "Recall for contact-a: 1 of at most 5 records in the packet" in page.text
        assert "in the packet" in page.text and "left out: the packet was full (5 records)" in page.text
        assert "Sam said the tide turns at six." in page.text and "withheld from the question" not in page.text
        assert browser.recall(token, "contact-a", "x" * 5000).status == 400
        # The longest question the page takes fits the form's body, however it is encoded.
        assert browser.recall(token, "contact-a", "\U0001F30A" * 4096).status == 200
    search = [body for method, path, body, _ in fake.calls if path == "/v1/host/memory/search"]
    assert search[0] == {"identity": {"host_id": "colonymind-console"}, "person_id": "contact-a",
                         "query": "when does the tide turn", "limit": 5, "trace": True}
    assert len(search) == 2


def test_the_inventory_and_history_pages_read_the_shapes_the_routes_give(fake):
    """Every state the inventory gives (``upcoming`` among them) is shown by name; a history shows what each
    correction was recorded against, and says so when the walk stopped at its bound or an erasure broke the chain."""
    from colonymind.memory.inventory import STATES
    from colonymind.console import pages
    assert set(pages.STATE_WORDS) == set(STATES)
    with running() as browser:
        browser.open()
        listing = browser.get("/memory?person=contact-a")
        assert "upcoming" in listing.text and "Starting 2026-10-01, my office moves to Lake." in listing.text
        assert "o-2" in listing.text and "c" * 12 in listing.text and "1 correction" in listing.text
        assert browser.path("/memory/history?claim=claim%3Alake&person=contact-a") in listing.parsed.hrefs
        next_page = [href for href in listing.parsed.hrefs if "page=" in href]
        assert next_page and "person=contact-a" in next_page[0]
        history = browser.get("/memory/history?claim=claim%3Alake&person=contact-a")
        assert history.status == 200
        assert "recorded against source o-2" in history.text
        assert "(what came before claim:gone was erased with it: the chain may go on)" in history.text
        assert "(the walk stopped at its bound: the chain goes on)" in history.text
        stale = browser.get("/memory?person=contact-a&page=stale")
        assert stale.status == 409 and "list again" in stale.text.lower()
        assert browser.path("/memory?person=contact-a") in stale.parsed.hrefs


def test_the_reports_are_the_reports_the_terminal_prints(fake, tmp_path):
    with running(bench=write_bench_dir(tmp_path / "bench")) as browser:
        browser.open()
        faculties = browser.get("/faculties")
        assert faculties.status == 200 and "Faculty report" in faculties.text
        assert "Owner gate: on" in faculties.text and "Self-narrative: off" in faculties.text
        assert browser.get("/faculties?since=soon").status == 400
        assert "Paired benchmark results" in browser.get("/bench").text
        assert "Instinct report" in browser.get("/instinct").text
        assert "Model usage" in browser.get("/usage").text
    with running() as browser:
        browser.open()
        bench = browser.get("/bench")
        assert bench.status == 200 and "--bench" in bench.text


def test_a_usage_ledger_that_cannot_be_read_is_unknown_never_zero(fake, monkeypatch):
    from colonymind import usage
    from colonymind.reports import usage as report

    def locked(home, since="7d", now=None):
        raise usage.Unreadable("database is locked")

    monkeypatch.setattr(report, "collect", locked)
    with running() as browser:
        browser.open()
        page = browser.get("/usage")
        assert page.status == 503
        assert "unknown" in page.text and "database is locked" in page.text


def test_reading_every_page_writes_nothing(fake, home, tmp_path):
    before = files(home)
    with running(bench=write_bench_dir(tmp_path / "bench")) as browser:
        browser.open()
        crawl(browser)
    assert files(home) == before


def test_a_sidecar_that_is_down_is_said_in_words_and_the_reports_still_read(fake):
    fake.down = True
    with running() as browser:
        browser.open()
        for target in ("/concerns", "/log", "/asks", "/health", "/memory?person=contact-a"):
            page = browser.get(target)
            assert page.status == 502 and "not answering" in page.text, target
        assert browser.get("/faculties").status == 200


# -- the command and its documentation --------------------------------------------------------------------------------

def test_the_colonymind_command_registers_the_console(monkeypatch):
    from colonymind import cli
    from colonymind.console import cli as console_cli
    seen = []
    monkeypatch.setattr(console_cli, "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr("sys.argv", ["colonymind", "console", "--port", "8765", "--bench", "results"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0
    assert seen[0].port == 8765 and seen[0].bench == "results"


def test_the_doc_lists_every_page_the_console_serves():
    from colonymind.console import app
    listed = re.findall(r"^\| `(/[^`]*)` \|", DOC.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert listed == list(app.PAGES)


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_the_memory_pages_walk_a_real_chain(sidecar, state, monkeypatch):
    river, lake, hill = await office_chain(state)
    app = make_app()
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: TestClient(app))
    with running() as browser:
        browser.open()
        listing = browser.get('/memory?person=contact-a')
        assert listing.status == 200
        text = listing.text
        assert text.index('Hill') < text.index('Lake') < text.index('River')
        assert 'current' in text and 'retracted' in text and ('superseded' in text)
        assert 'Correction: my office is in Hill.' in text
        history = browser.get(f"/memory/history?{urlencode({'claim': river, 'person': 'contact-a'})}")
        assert history.status == 200
        assert history.text.index('stated') < history.text.index('superseded') < history.text.index('retracted')
        from colonymind.turns import get_turn_idempotency_ledger
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(get_turn_idempotency_ledger(state))
        curated.forget_sources(get_turn_idempotency_ledger(state), 'contact-a', source_ids=['o-2', 'o-3'])
        assert historical_bytes(get_turn_idempotency_ledger(state)) == _native_before_withdrawal
        broken = browser.get(f"/memory/history?{urlencode({'claim': river, 'person': 'contact-a'})}")
        assert f'(what came after {lake} was erased with it: the chain may go on)' in broken.text
        assert 'now in Lake' not in broken.text
        assert browser.get('/memory/history?claim=claim%3A' + '0' * 64).status == 404
