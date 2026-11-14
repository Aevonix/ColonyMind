"""Fixtures: a fake sidecar, a temporary Hermes home and subprocess probes.

Every probe runs stock Hermes (installed in this environment) in a fresh
interpreter with ``HERMES_HOME`` pointing at a temporary profile whose config
enables the adapter the way ``colonymind init`` writes it. The sidecar is a
small HTTP server in the test process that records every request.
"""

from __future__ import annotations

import atexit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import yaml

from scripts.check_sidecar_skips import SKIP_REASONS
from scripts.hardware_latency import (configure as configure_hardware_latency, hardware_latency,
    pytest_addoption, pytest_sessionfinish, pytest_terminal_summary)

ROOT = Path(__file__).resolve().parents[2]
# The sidecar and Hermes find their default directories (~/.colonymind, ~/.hermes) through HOME, and the
# sidecar fixes one path under it when its module is imported. Tests run code of both in this process and
# in the probes it starts, so HOME is a directory of this run before any test module is imported: nothing
# reads or changes the home of the user who runs the suite. A test that sets its own HOME keeps it.
# Outside a virtual environment the interpreter has packages under the home (its user site): the processes
# a test starts keep finding them where this interpreter has them.
_TEST_HOME_DIR = tempfile.mkdtemp(prefix="colonymind-pytest-home-")
atexit.register(shutil.rmtree, _TEST_HOME_DIR, ignore_errors=True)
if site.ENABLE_USER_SITE:
    os.environ.setdefault("PYTHONUSERBASE", site.getuserbase())
os.environ["HOME"] = _TEST_HOME_DIR
for variable in ("HERMES_HOME", "COLONYMIND_HOME", "COLONYMIND_INSTANCE_SELECTED"):
    os.environ.pop(variable, None)
for variable in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
    os.environ.pop(variable, None)  # their defaults lie under HOME
# Select the core for this interpreter and every child probe, including tests
# which historically read HERMES_SOURCE_ROOT directly.
if os.environ.get("HERMES_SRC"):
    selected = str(Path(os.environ["HERMES_SRC"]).resolve())
    source_root = os.environ.get("HERMES_SOURCE_ROOT")
    if source_root and str(Path(source_root).resolve()) != selected:
        raise ValueError("HERMES_SRC and HERMES_SOURCE_ROOT select different Hermes trees")
    os.environ["HERMES_SOURCE_ROOT"] = selected
    sys.path.insert(0, selected)
    inherited_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (selected, inherited_path)))
API_KEY = "test-key-0123456789abcdef"
OWNER = "p-01"
CANARY = "OWNER-ONLY-CANARY-7f3a"

LIVE_PREREQUISITES = {
    "test_cancel_releases_live_engine_and_recorded_process_integrator": "HANDS_CANCEL_RECEIPT",
    "test_socket_adapter_integrator": "HANDS_RUN_ADAPTER_CHECK",
    # test_session_archive_pull.py. The five probe tests run under the suite's own interpreter everywhere; only
    # their parameter for a second interpreter with Hermes installed carries the marker and needs that variable.
    # The last test needs a sidecar-only environment.
    "test_reviewed_profile_unload_and_discovery_reproductions": "COLONYMIND_TEST_HERMES_PYTHON",
    "test_routing_drift_compensates_without_positive_receipt": "COLONYMIND_TEST_HERMES_PYTHON",
    "test_actual_gateway_sigkill_stops_unit_and_recovers": "COLONYMIND_TEST_HERMES_PYTHON",
    "test_stalled_engine_does_not_join_at_interpreter_shutdown": "COLONYMIND_TEST_HERMES_PYTHON",
    "test_scheduler_claim_during_stalled_archive_exits_and_recovers": "COLONYMIND_TEST_HERMES_PYTHON",
    "test_separate_interpreters_sidecar_has_no_hermes_and_worker_completes_archive": "COLONYMIND_TEST_SIDECAR_PYTHON",
}


def pytest_collection_modifyitems(items):
    sidecar_absent = importlib.util.find_spec("colonymind") is None
    for item in items:
        item.user_properties.append(("guard_nodeid", item.nodeid))
        prerequisite = LIVE_PREREQUISITES.get(item.name)
        if prerequisite:
            item.add_marker(pytest.mark.requires_live_qualification)
        live = item.get_closest_marker("requires_live_qualification")
        sidecar = item.get_closest_marker("requires_sidecar")
        if sidecar and sidecar_absent:
            missing = True
            marker = "requires_sidecar"
        elif live:
            # A parameter that carries the marker itself has its test's entry, under the function's name.
            prerequisite = prerequisite or LIVE_PREREQUISITES.get(getattr(item, "originalname", None))
            assert prerequisite, f"missing live prerequisite mapping: {item.nodeid}"
            value = os.environ.get(prerequisite)
            missing = value != "1" if prerequisite == "HANDS_RUN_ADAPTER_CHECK" else not value
            marker = "requires_live_qualification"
        elif sidecar:
            missing = False
            marker = "requires_sidecar"
        else:
            continue
        reason = SKIP_REASONS[marker] if missing else ""
        item.user_properties.extend((("guard_marker", marker), ("guard_expected_skip", reason)))
        if missing:
            # This takes precedence over a module's opt-in skipif.
            item.add_marker(pytest.mark.skipif(True, reason=reason), append=False)


def pytest_configure(config):
    configure_hardware_latency(config)
    # These probes use isolated local sockets and fake transports, so a selected reviewed source is sufficient.
    if os.environ.get("HERMES_SOURCE_ROOT"):
        os.environ.setdefault("COLONYMIND_HERMES_INTEGRATION", "1")
        os.environ.setdefault("COLONYMIND_TEST_HANDS_SOCKETS", "1")


CONTACTS = {
    ("telegram", "1001"): {"contact_id": OWNER, "display_name": "Owner", "may_contact": "auto",
                           "trust_tier": "GENESIS"},
    ("telegram", "2002"): {"contact_id": "p-02", "display_name": "Never", "may_contact": "never",
                           "trust_tier": "unknown"},
    ("telegram", "2003"): {"contact_id": "p-03", "display_name": "Friend", "may_contact": "ask",
                           "trust_tier": "REGULAR", "cadence_minutes": None,
                           "digest": "Friend: known since spring; " + CANARY},
}
PUBLIC_PERSON = ("contact_id", "display_name", "trust_tier")


class FakeMind:
    """The ``/v1/mind`` contract the adapter's body, tools and guard rely on.

    Intentions live in ``intentions`` keyed by id with a lifecycle ``status``
    (``asked``, ``approved``, ``dispatched``, ``done``...). ``GET dispatch``
    offers the approved tasks; ``bound`` moves them to ``dispatched`` unless
    ``lose_bound`` simulates a lost ack. Outbox messages live in ``outbox``
    with ``state`` ``ready | sending | sent | failed | uncertain``;
    ``relist_sending`` makes the fake re-offer a message already in
    ``sending``, the sloppy sidecar the body must tolerate.
    """

    def __init__(self):
        self.enabled = True
        self.autonomy = "standard"
        self.narrative: dict = {"enabled": False, "text": "", "sections": {}, "cites": [], "updated_at": None}
        self.intentions: dict[str, dict] = {}
        self.outbox: dict[str, dict] = {}
        self.guard_verdict: dict = {"allow": True, "reason": ""}
        self.lose_bound = False
        self.relist_sending = False
        self.bound: list[dict] = []
        self.outcomes: list[dict] = []
        self.observations: list[dict] = []
        self.rates: list[dict] = []
        # ``POST /answers``: what the plugin handed over, and the sidecar's reading of it (a dict, or a function
        # of the request body); ``POST /asks/{id}/shown``: the messages the plugin showed asks in.
        self.answers: list[dict] = []
        self.answer_reply = None
        self.shown: list[dict] = []
        # ``POST /asks/{id}/show``: the sidecar puts an open ask in front of the owner again as an ``ask_show``
        # outbox message in its own words (``show_words`` by ask id: every term a yes grants, in full), unless
        # ``show_route`` is off (a sidecar without the route).
        self.show_route = True
        self.show_words: dict[str, str] = {}
        self.shows: list[str] = []
        self.sending: list[dict] = []
        self.sent: list[dict] = []
        self.pulls = 0
        self.last_pull_at = None
        # ColonyMind's skills (M9): the generation the sidecar bumps on every skill change, read from a JSON
        # file a probe can rewrite as the sidecar would ({"generation": n}); and the loads reported to it.
        self.skills_state: Path | None = None
        self.skill_loads: list[dict] = []

    def skills(self) -> dict | None:
        if self.skills_state is None or not self.skills_state.exists():
            return None
        return {"enabled": True, "owned": [], **json.loads(self.skills_state.read_text())}

    def handle(self, method, path, body, query=None):
        import time as _time
        query = query or {}
        parts = path.split("/")[3:]  # after /v1/mind
        head = parts[0] if parts else ""
        if head == "state" and method == "GET":
            asks = [{"id": i["id"], "code": i.get("ask_code"), "ask_code": i.get("ask_code"), "title": i.get("title"),
                     "kind": i.get("kind", "task"), "message": i.get("message"), "expires_at": i.get("expires_at")}
                    for i in self.intentions.values() if i.get("status") == "asked"]
            state = {"enabled": self.enabled, "autonomy": self.autonomy, "level": self.autonomy,
                     "queued": len([i for i in self.intentions.values() if i.get("status") == "approved"]),
                     "asks": asks, "open_asks": len(asks), "last_tick_at": None, "body": {"last_pull_at": self.last_pull_at}}
            if self.skills() is not None:
                state["skills"] = self.skills()
            return 200, state
        if head == "skills" and parts[1:] == ["used"] and method == "POST":
            self.skill_loads.append(dict(body or {}))
            counted = str((body or {}).get("skill") or "").startswith("colonymind-")
            return 200, {"ok": True, "counted": counted, "loads": len(self.skill_loads)}
        if head == "dispatch" and method == "GET":
            self.pulls += 1
            self.last_pull_at = _time.time()
            if not self.enabled:
                return 200, {"intentions": []}
            offered = [dict(i) for i in self.intentions.values()
                       if i.get("status") == "approved" and i.get("kind", "task") == "task"]
            for item in offered:
                item.setdefault("dedup_key", "mind:" + item["id"])
                item.setdefault("assignee", "colonymind-act")
            return 200, {"intentions": offered}
        if head == "dispatch" and len(parts) == 3 and parts[2] == "bound" and method == "POST":
            intention = self.intentions.get(parts[1])
            if intention is None:
                return 404, {"detail": "unknown intention"}
            self.bound.append({"id": parts[1], **(body or {})})
            if not self.lose_bound:
                intention["status"] = "dispatched"
                intention["hermes_ref"] = (body or {}).get("hermes_ref")
            return 200, {"ok": True, "id": parts[1], "status": intention["status"]}
        if head == "outbox" and len(parts) == 1 and method == "GET":
            if not self.enabled:
                return 200, {"messages": []}
            states = {"ready", "sending"} if self.relist_sending else {"ready"}
            return 200, {"messages": [dict(m) for m in self.outbox.values() if m.get("state", "ready") in states]}
        if head == "outbox" and len(parts) == 3 and method == "POST":
            message = self.outbox.get(parts[1])
            if message is None:
                return 404, {"detail": "unknown message"}
            if parts[2] == "sending":
                self.sending.append({"id": parts[1], **(body or {})})
                if message.get("state", "ready") != "ready":
                    return 409, {"detail": f"message is {message['state']}"}
                message["state"] = "sending"
                return 200, {"ok": True, "state": "sending"}
            if parts[2] == "sent":
                self.sent.append({"id": parts[1], **(body or {})})
                message["state"] = (body or {}).get("result") or "uncertain"
                return 200, {"ok": True, "state": message["state"]}
        if head == "outcome" and method == "POST":
            intention = self.intentions.get(str((body or {}).get("id")))
            if intention is None:
                return 404, {"detail": "unknown intention"}
            self.outcomes.append(dict(body))
            intention["outcome"] = body.get("outcome")
            if body.get("final"):
                intention["status"] = body.get("outcome")
            return 200, {"ok": True}
        if head == "observations" and method == "POST":
            self.observations.append(dict(body or {}))
            return 200, {"ok": True}
        if head == "guard" and method == "POST":
            return 200, dict(self.guard_verdict)
        if head == "answers" and method == "POST":
            self.answers.append(dict(body or {}))
            reply = self.answer_reply(dict(body or {})) if callable(self.answer_reply) else self.answer_reply
            return 200, dict(reply or {"read": True, "path": None, "settled": [], "context": ""})
        if head == "asks" and len(parts) == 3 and parts[2] == "shown" and method == "POST":
            intention = self.intentions.get(parts[1])
            if intention is None or intention.get("status") != "asked":
                return 404, {"detail": {"code": "unknown_ask", "message": f"no open ask {parts[1]}"}}
            self.shown.append({"id": parts[1], **(body or {})})
            return 200, {"ok": True, "asks": [parts[1]], "message_id": (body or {}).get("message_id")}
        if head == "asks" and len(parts) == 3 and parts[2] == "show" and method == "POST" and self.show_route:
            intention = self.intentions.get(parts[1])
            if intention is None or intention.get("status") != "asked":
                return 404, {"detail": {"code": "unknown_ask", "message": f"no open ask {parts[1]}"}}
            self.shows.append(parts[1])
            words = self.show_words.get(parts[1]) or f"{intention.get('title')}. Reply yes or no."
            ident = f"show-{len(self.shows)}"
            self.outbox[ident] = {"id": ident, "kind": "notice", "type": "ask_show", "recipient_is_owner": True,
                                  "text": words, "target": "telegram:1001", "asks": [parts[1]]}
            return 200, {"ok": True, "id": parts[1], "message": ident, "text": f"sent to the owner's chat again: {words}"}
        if head == "rate" and method == "POST":
            self.rates.append(dict(body or {}))
            return 200, {"ok": True, **(body or {})}
        if head == "log" and method == "GET":
            statuses = {s for s in query.get("status", "").split(",") if s}
            kinds = {k for k in query.get("kind", "").split(",") if k}
            rows = [i for i in self.intentions.values()
                    if (not statuses or i.get("status") in statuses) and (not kinds or i.get("kind", "task") in kinds)
                    and (not query.get("recipient") or i.get("recipient") == query["recipient"])]
            entries = [{"id": i["id"], "kind": i.get("kind", "task"), "type": i.get("type"), "status": i.get("status"),
                        "title": i.get("title"), "decision": i.get("decision"), "recipient": i.get("recipient")}
                       for i in rows]
            if query.get("split") == "true":     # the agent's own actions apart from the rest (audit.is_action)
                return 200, {"actions": [e for e in entries if e["kind"] != "note"],
                             "notes": [e for e in entries if e["kind"] == "note"], "text": ""}
            return 200, {"entries": entries}
        if head == "narrative" and method == "GET":
            return (200, dict(self.narrative)) if self.narrative is not None else (500, {"detail": "narrative failed"})
        if head == "why" and len(parts) == 2 and method == "GET":
            intention = self.intentions.get(parts[1])
            return (200, dict(intention)) if intention else (404, {"detail": {
                "code": "unknown_intention", "message": f"no intention {parts[1]} exists in the audit log"}})
        if head == "off" and method == "POST":
            self.enabled = False
            for message in self.outbox.values():
                if message.get("state", "ready") == "ready":
                    message["state"] = "cancelled"
            return 200, {"ok": True, "enabled": False}
        if head == "on" and method == "POST":
            self.enabled = True
            return 200, {"ok": True, "enabled": True}
        if head == "tick" and method == "POST":
            return 200, {"ok": True}
        return 404, {"detail": "not found"}


class FakeSidecar:
    """Just enough of ``/v1/host`` (and optionally ``/v1/mind``) for the adapter."""

    def __init__(self):
        self.requests: list[dict] = []
        self.unauthorized: list[str] = []
        self.mind_routes = False
        self.mind = FakeMind()
        self.contacts = {key: dict(value) for key, value in CONTACTS.items()}
        self.proposals: list[dict] = []
        self.people_owner = OWNER   # whom the sidecar knows as the owner (its own check)
        self.people_asks: list[dict] = []   # the one-tap asks ``/v1/mind/people/changes`` filed
        self.told: dict[str, list[str]] = {}   # what each contact said in synced turns, recalled by contact
        self.delays: dict = {}   # path -> seconds the answer is held, or a function of the request (a slow route)
        self.drips: dict[str, float] = {}   # path -> seconds over which its body is sent, in small slow chunks
        self.assemble = None                   # body -> {"sections": [...]}: a context the test builds itself
        # The platforms the key may write the owner's turns from (docs/CREDENTIALS.md); None: not checked.
        self.owner_platforms: set[str] | None = None
        self.lock = threading.Lock()
        sidecar = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def _reply(self, status, body, drip=0.0):
                data = json.dumps(body).encode()
                pieces = [data]
                if drip:  # trailing whitespace keeps the JSON; each piece comes well within a per-read timeout
                    data += b" " * 1024
                    pieces = [data[i:i + len(data) // 12 + 1] for i in range(0, len(data), len(data) // 12 + 1)]
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                for piece in pieces:
                    self.wfile.write(piece)
                    if drip:
                        self.wfile.flush()
                        time.sleep(drip / len(pieces))

            def _route(self):
                parts = urlsplit(self.path)
                query = {k: v[0] for k, v in parse_qs(parts.query).items()}
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                record = {"method": self.command, "path": parts.path, "query": query, "json": body,
                          "authorization": self.headers.get("Authorization", "")}
                with sidecar.lock:
                    sidecar.requests.append(record)
                if record["authorization"] != f"Bearer {API_KEY}":
                    with sidecar.lock:
                        sidecar.unauthorized.append(parts.path)
                    return self._reply(401, {"detail": "unauthorized"})
                status, reply = sidecar.dispatch(self.command, parts.path, query, body)
                delay = sidecar.delays.get(parts.path)
                if callable(delay):
                    delay = delay(record)
                if delay:
                    time.sleep(delay)
                self._reply(status, reply, sidecar.drips.get(parts.path, 0.0))

            do_GET = do_POST = do_PUT = do_PATCH = _route

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a probe that exits mid-request (the crash tests) resets its socket

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def calls(self, path: str, method: str | None = None) -> list[dict]:
        with self.lock:
            return [r for r in self.requests if r["path"] == path and (method is None or r["method"] == method)]

    @property
    def mind_enabled(self) -> bool:
        return self.mind.enabled

    @mind_enabled.setter
    def mind_enabled(self, value: bool) -> None:
        self.mind.enabled = bool(value)

    @property
    def guard_verdict(self) -> dict:
        return self.mind.guard_verdict

    @guard_verdict.setter
    def guard_verdict(self, value: dict) -> None:
        self.mind.guard_verdict = dict(value)

    def dispatch(self, method, path, query, body):
        if method == "POST" and path == "/v1/host/activity/snapshot":
            return 200, {"accepted": True, "duplicate": False, "superseded": False,
                         "revision": body["revision"]}
        if path == "/v1/mind/people" or path.startswith("/v1/mind/people/"):
            return self._people(method, path, query, body)
        if path.startswith("/v1/mind/"):
            return self._mind(method, path, body, query)
        if path == "/v1/host/health":
            return 200, {"status": "ok", "capabilities": ["memory"]}
        if path == "/v1/host/contacts/resolve":
            contact = self.contacts.get((query.get("gateway", ""), query.get("address", "")))
            if contact is None and query.get("create") == "true" and query.get("address"):
                contact = {"contact_id": "p-" + query["address"][-2:], "display_name": query["address"],
                           "may_contact": "ask", "trust_tier": "unknown"}
                self.contacts[(query.get("gateway", ""), query["address"])] = contact
            return (200, contact) if contact else (404, {"detail": "No contact for that handle"})
        if path == "/v1/host/contacts":
            return 200, {"contacts": list(self.contacts.values()), "total": len(self.contacts)}
        if path.startswith("/v1/host/contacts/"):
            wanted = path.rsplit("/", 1)[1]
            found = next((c for c in self.contacts.values() if c["contact_id"] == wanted), None)
            return (200, found) if found else (404, {"detail": "unknown contact"})
        if path == "/v1/host/turns/sync":
            if not isinstance(body, dict) or not isinstance(body.get("context"), dict):
                return 422, {"detail": "context required"}
            context, sender = body["context"], body.get("sender") or {}
            contact = str(context.get("contact_id") or "")
            if self.owner_platforms is not None:
                # The sidecar's attribution check: the owner (named, or a sender who is the owner) only
                # from a granted platform; a turn naming nobody is the key's own identity.
                platform = str(sender.get("platform") or str(context.get("channel_id") or "").split(":", 1)[0])
                known = self.contacts.get((platform, str(sender.get("user_id") or "")), {}).get("contact_id")
                if OWNER in (contact, known) and platform not in self.owner_platforms:
                    return 403, {"detail": {"code": "owner_not_granted",
                                            "message": f"credential api-key may not write as the owner from {platform}"}}
                if not contact and not sender:
                    contact = "credential:api-key"
            if not contact:
                return 422, {"detail": "contact_id required"}
            self.told.setdefault(contact, []).append(str((body.get("user_message") or {}).get("content") or ""))
            return 200, {"accepted": True, "continuity_updated": True, "source_recorded": True,
                         **({"work_recorded": True} if body.get("work") is not None else {})}
        if path == "/v1/host/context/assemble" and self.assemble is not None:
            return 200, self.assemble(body)
        if path == "/v1/host/context/assemble":
            audience = body.get("audience") if isinstance(body, dict) else None
            shared = [{"id": "shared", "title": "Shared", "body": "shared facts", "priority": 50}]
            told = self.told.get(str((body.get("context") or {}).get("contact_id") or ""), [])
            if told:   # recall by contact, whatever session or channel the words arrived on
                shared.append({"id": "colonymind-memory", "title": "Recalled", "body": "\n".join(told), "priority": 90})
            if audience == "viewer":
                return 200, {"sections": shared}
            return 200, {"sections": [{"id": "private", "title": "Owner notes", "body": CANARY, "priority": 90},
                                      *shared]}
        if path == "/v1/host/context/temporal":
            return 200, {"title": "Current Time", "body": "Sunday, 10:00 AM UTC"}
        if path == "/v1/host/memory/search":
            return 200, {"content": f"excerpt for {body.get('person_id')}", "count": 1,
                         "source_refs": [{"source_id": "src-1", "source_version": "a" * 64}],
                         "watermark": 0, "retrieval": {"semantic": "ready", "contact_facts": "ready"},
                         "annotation_checks": []}
        if path == "/v1/host/memory/curated/sources/forget":
            return 200, {"source_erased": True, "watermark": 1}
        if path == "/v1/host/memory/sources/deadline":
            from datetime import datetime, timedelta, timezone
            deadline = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
            return 200, {"status": "current", "deadline_at": deadline, "subject": "the report",
                         "predicate": "is due", "value": "in two hours", "timezone_name": "UTC",
                         "timezone_basis": "caller_override", "claim_id": body.get("claim_id"),
                         "root_claim_id": body.get("claim_id"), "source_refs": []}
        return 404, {"detail": "not found"}

    def _mind(self, method, path, body, query=None):
        if not self.mind_routes:
            return 404, {"detail": "not found"}
        with self.lock:
            return self.mind.handle(method, path, body, query)

    def _people(self, method, path, query, body):
        """``/v1/mind/people`` as the sidecar serves it: a named non-owner viewer sees only who someone
        is, and a mutation needs ``contact_id`` == the owner (or ``by: cli``)."""
        if not self.mind_routes:
            return 404, {"detail": "not found"}
        body = body or {}
        rest = [unquote(part) for part in path[len("/v1/mind/people"):].split("/") if part]
        with self.lock:
            people = {c["contact_id"]: c for c in self.contacts.values()}
            viewer = query.get("contact_id")

            def view(contact):
                return dict(contact) if viewer is None or viewer == self.people_owner else \
                    {key: contact[key] for key in PUBLIC_PERSON}

            def find(reference):
                return people.get(reference) or next(
                    (c for c in people.values() if c["display_name"].lower() == reference.lower()), None)

            if method == "GET" and not rest:
                wanted = query.get("q", "").lower()
                return 200, {"contacts": [view(c) for c in people.values()
                                          if not wanted or wanted in c["display_name"].lower() or wanted == c["contact_id"]]}
            if method == "GET" and rest == ["proposals"]:
                return 200, {"proposals": list(self.proposals)}
            if method == "POST" and rest == ["link"]:
                candidate = {"candidate_id": f"identity-candidate:{len(self.proposals) + 1}", "status": "pending",
                             **{key: body.get(key) for key in ("contact_id", "gateway", "address", "by")}}
                self.proposals.append(candidate)
                return 200, {"ok": True, **candidate, "text": "proposed; the owner confirms"}
            if method == "GET" and len(rest) == 1:
                contact = find(rest[0])
                if contact is None:
                    return 404, {"detail": {"code": "unknown_contact", "message": f"no single contact matches {rest[0]!r}"}}
                return 200, {"contact": view(contact)}
            if method != "POST":
                return 404, {"detail": "not found"}
            if rest == ["changes"]:
                return self._people_change(body, find)
            if body.get("by") != "cli" and body.get("contact_id") != self.people_owner:
                return 403, {"detail": {"code": "not_owner", "message": "only the owner can change this"}}
            if rest == ["merge"]:
                keep, drop = find(body.get("keep") or ""), find(body.get("drop") or "")
                if keep is None or drop is None:
                    return 404, {"detail": {"code": "unknown_contact", "message": "no single contact matches"}}
                self.contacts = {k: v for k, v in self.contacts.items() if v["contact_id"] != drop["contact_id"]}
                return 200, {"ok": True, "dropped": drop["contact_id"], "contact": keep, "text": "merged"}
            contact = find(rest[0]) if len(rest) == 2 else None
            if contact is None:
                return 404, {"detail": {"code": "unknown_contact", "message": "no single contact matches"}}
            if rest[1] == "permission":
                contact["may_contact"] = body.get("may_contact")
                return 200, {"ok": True, "contact_id": contact["contact_id"], "may_contact": contact["may_contact"]}
            if rest[1] == "cadence":
                contact["cadence_minutes"] = body.get("minutes")
                return 200, {"ok": True, "contact_id": contact["contact_id"], "cadence_minutes": body.get("minutes")}
            return 404, {"detail": "not found"}

    def _people_change(self, body, find):
        """``POST /v1/mind/people/changes`` as the sidecar decides it, simplified: the viewer must be the
        owner; a narrowing applies; a raise (every merge here) is the owner's ask with a code, whatever the
        turn said (the body has no field for words)."""
        if body.get("viewer") != self.people_owner:
            return 403, {"detail": {"code": "not_owner",
                                    "message": "only the owner can change who may be contacted, cadences or merges"}}
        operation = body.get("operation")
        contact = find(body.get("contact_id") or "")
        drop = find(body.get("drop") or "") if operation == "merge" else None
        if contact is None or (operation == "merge" and drop is None):
            return 404, {"detail": {"code": "unknown_contact", "message": "no single contact matches"}}
        rank = {"never": 0, "ask": 1, "auto": 2}
        if operation == "set_permission":
            widens = rank.get(body.get("value"), 0) > rank.get(contact.get("may_contact"), 0)
        elif operation == "set_cadence":
            minutes, current = body.get("minutes"), contact.get("cadence_minutes")
            widens = minutes is not None and (current is None or minutes < current)
        else:
            widens = True
        if set(body) - {"operation", "contact_id", "value", "minutes", "drop", "viewer", "by"}:
            return 422, {"detail": [{"msg": "extra fields not permitted"}]}
        if widens:
            code = f"PC{len(self.people_asks) + 1}"
            self.people_asks.append({"code": code, **body})
            return 200, {"ok": True, "status": "asked", "code": code,
                         "text": f"not changed yet: ask {code} is open; the owner's own reply 'yes {code}' applies it"}
        if operation == "merge":
            self.contacts = {k: v for k, v in self.contacts.items() if v["contact_id"] != drop["contact_id"]}
            return 200, {"ok": True, "status": "applied", "dropped": drop["contact_id"], "text": "merged"}
        if operation == "set_permission":
            contact["may_contact"] = body.get("value")
            return 200, {"ok": True, "status": "applied", "may_contact": contact["may_contact"], "text": "set"}
        contact["cadence_minutes"] = body.get("minutes")
        return 200, {"ok": True, "status": "applied", "cadence_minutes": body.get("minutes"), "text": "set"}


@pytest.fixture
def sidecar():
    server = FakeSidecar().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def home(tmp_path, sidecar):
    """A Hermes profile configured the way ``colonymind init`` writes it."""
    return build_home(tmp_path, sidecar.server.server_address[1], sidecar.url)


def build_home(tmp_path, port, url):
    hermes, instance = tmp_path / "hermes", tmp_path / "colonymind"
    hermes.mkdir()
    instance.mkdir()
    key_file = instance / "api.key"
    key_file.write_text(API_KEY + "\n")
    key_file.chmod(0o600)
    (instance / "colonymind.yaml").write_text(yaml.safe_dump({
        "sidecar": {"host": "127.0.0.1", "port": port},
        "mind": {"enabled": True, "autonomy": "standard",
                 "deny": {"commands": ["rm -rf *"], "tools": ["terminal"], "text": ["secret-project-x"]}},
    }))
    (instance / "identity.yaml").write_text(yaml.safe_dump({
        "owner": {"name": "Owner", "contact_id": OWNER, "handles": {"telegram": ["1001"]}},
        "agent": {"name": "Agent", "values": ["care"], "boundaries": ["never send money"]},
    }))
    config = {
        "plugins": {"enabled": ["colonymind"], "hook_callback_timeout": 0,
                    "colonymind": {"sidecar_url": url, "key_file": str(key_file)}},
        "memory": {"provider": "colonymind-memory", "config": {"contact_id": OWNER}},
        "kanban": {"dispatch_in_gateway": True},
    }
    (hermes / "config.yaml").write_text(yaml.safe_dump(config))
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def write_config(**changes):
        merged = json.loads(json.dumps(config))
        for key, value in changes.items():
            merged[key] = value
        (hermes / "config.yaml").write_text(yaml.safe_dump(merged))

    def write_mind(**mind):
        current = yaml.safe_load((instance / "colonymind.yaml").read_text())
        current["mind"].update(mind)
        (instance / "colonymind.yaml").write_text(yaml.safe_dump(current))

    return SimpleNamespace(root=tmp_path, hermes=hermes, instance=instance, key_file=key_file,
                           workspace=workspace, config=config, write_config=write_config,
                           write_mind=write_mind)


PRELUDE = r'''
import json, os, sys, threading, time
def emit(**values):
    print("@@RESULT@@" + json.dumps(values, default=str), flush=True)
from hermes_cli.plugins import get_plugin_manager, _reset_plugin_managers_for_tests, VALID_HOOKS
from hermes_cli.lifecycle import invoke_hook
_reset_plugin_managers_for_tests()
manager = get_plugin_manager()
manager.discover_and_load(force=True)
loaded = manager._plugins["colonymind"]
assert loaded.enabled, loaded.error
import colonymind_hermes
'''

# The body tests drive ticks by hand: the registered body thread stays parked so
# its own ticks cannot race the assertions.
MIND_PRELUDE = PRELUDE.replace("manager.discover_and_load(force=True)", '''
import colonymind_hermes.body as _body_module
_body_module.Body.start = lambda self: None
manager.discover_and_load(force=True)''') + r'''
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
body = colonymind_hermes._BODY
def tick():
    body.on_dispatch_tick(board="default", outcome="idle")  # the gateway dispatcher's heartbeat
    return body.run_once()
def tasks():
    conn = connect()
    try:
        return {t.idempotency_key or t.id: {"id": t.id, "status": t.status, "assignee": t.assignee, "title": t.title,
                "body": t.body, "goal_mode": t.goal_mode, "max_runtime_seconds": t.max_runtime_seconds,
                "max_retries": t.max_retries, "created_by": t.created_by, "workspace_kind": t.workspace_kind}
                for t in kb.list_tasks(conn, include_archived=True)}
    finally:
        conn.close()
SENDS = []
def fake_send(args, **kw):
    SENDS.append(dict(args))
    return json.dumps({"success": True, "platform": "telegram", "message_id": "m-%d" % len(SENDS)})
import tools.send_message_tool as _smt
'''

WORKER_ENV_KEYS = ("HERMES_KANBAN_TASK", "HERMES_PROFILE", "HERMES_KANBAN_WORKSPACE", "TERMINAL_CWD")


def worker_env(home, task_id="task-01", profile="colonymind-act") -> dict[str, str]:
    """Simulate a mind dispatch's durable creation record as well as its worker variables."""
    if profile == "colonymind-act":
        from colonymind_hermes.standing import STANDING_FILE, StandingStore
        StandingStore(home.instance / STANDING_FILE).put("kanban", task_id, "mind", "worker")
    return {"HERMES_KANBAN_TASK": task_id, "HERMES_PROFILE": profile,
            "HERMES_KANBAN_WORKSPACE": str(home.workspace), "TERMINAL_CWD": str(home.workspace)}


def probe(code: str, home, *, env: dict | None = None, timeout: float = 240, prelude: str = PRELUDE) -> dict:
    """Run ``code`` after the prelude in a fresh interpreter; return the emitted result."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("HERMES_", "COLONYMIND_", "TERMINAL_"))}
    environment.update({"HERMES_HOME": str(home.hermes), "COLONYMIND_TEST_HOME": str(home.root),
                        "COLONYMIND_HOME": str(home.instance), "COLONYMIND_SKIP_DOTENV": "1",
                        "HERMES_DISABLE_LAZY_INSTALLS": "1", "PYTHONUNBUFFERED": "1",
                        "PYTHON_DOTENV_DISABLED": "1"})
    if os.environ.get("HERMES_SOURCE_ROOT"):
        environment["HERMES_SOURCE_ROOT"] = os.environ["HERMES_SOURCE_ROOT"]
    environment.update(env or {})
    result = subprocess.run([sys.executable, "-c", prelude + code], env=environment, text=True,
                            capture_output=True, timeout=timeout, cwd=home.root, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stdout[-4000:] + "\n---\n" + result.stderr[-4000:]
    lines = [line for line in result.stdout.splitlines() if line.startswith("@@RESULT@@")]
    assert lines, result.stdout[-4000:] + "\n---\n" + result.stderr[-4000:]
    return json.loads(lines[-1][len("@@RESULT@@"):])


def refused(answer: dict) -> str:
    """What a refused tool call says: the reason of the one final answer (``client.final_answer``), else the
    error of an argument the model can correct."""
    final = answer.get("unavailable") is True and answer.get("retry") is False
    return str(answer.get("reason") if final else answer.get("error") or "")


def run_python(*args, cwd, env=None):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, text=True,
                            capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """The built wheel and sdist (or the prepared ones CI hands over)."""
    output = tmp_path_factory.mktemp("adapter-artifacts")
    prepared = os.environ.get("COLONYMIND_DISTRIBUTIONS_DIR")
    distributions = Path(prepared).resolve() if prepared else output
    if not prepared:
        run_python("-m", "build", "--no-isolation", "--outdir", output, cwd=ROOT)
    wheel, = distributions.glob("colonymind_hermes-*.whl")
    source, = distributions.glob("colonymind_hermes-*.tar.gz")
    return output, wheel, source
