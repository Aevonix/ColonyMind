"""What the owner console reads (docs/CONSOLE.md).

The mind, health, the recall check and the memory inventory come from the running sidecar, asked as the instance
key (the owner's credential), exactly as ``colonymind mind``, ``colonymind recall`` and ``colonymind memory`` ask it.
The reports read the instance's stores read-only, as ``colonymind report`` does, so they work with the sidecar
stopped.

Every sidecar route the console may call is listed in ``ROUTES``: reads, the recall check's search (a read) and the
mind's off switch. ``Source.call`` refuses any other, so no page can answer an ask, change the level or a grant,
turn the mind back on or resolve a sealed value, whatever it is asked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, unquote

from colonymind.util.temporal import now_utc

#: ``(method, path)``: a path ending in ``/`` names the route family under it (one id after it, or the history of
#: a claim), any other is exact.
ROUTES: Tuple[Tuple[str, str], ...] = (
    ("GET", "/v1/mind/state"), ("GET", "/v1/mind/concerns"), ("GET", "/v1/mind/goals"), ("GET", "/v1/mind/log"),
    ("GET", "/v1/mind/why/"), ("GET", "/v1/mind/asks"), ("GET", "/v1/host/health"),
    ("POST", "/v1/host/memory/search"), ("GET", "/v1/host/memory/inventory"),
    ("GET", "/v1/host/memory/inventory/claims/"),
    ("POST", "/v1/mind/off"),
)
#: Who the off switch's record names, and why.
OFF_BY, OFF_REASON = "console", "the owner console"


class Unavailable(Exception):
    """The sidecar did not give an answer: ``status`` is its HTTP status (None when it did not answer at all),
    ``code`` its error code, ``message`` what a person reads."""

    def __init__(self, status: Optional[int], code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def allowed(method: str, path: str) -> bool:
    """Whether the console may call ``path``: a route in ``ROUTES`` (an id family: exactly one id segment, or one
    claim id and ``/history``), with no empty or dot segment and no encoded separator that could lead elsewhere."""
    decoded = unquote(path)
    if decoded.count("/") != path.count("/") or "\\" in decoded:
        return False
    segments = decoded.split("/")
    if segments[0] != "" or "" in segments[1:] or any(part in (".", "..") for part in segments):
        return False
    for verb, route in ROUTES:
        if verb != method:
            continue
        if not route.endswith("/"):
            if path == route:
                return True
            continue
        rest = path[len(route):] if path.startswith(route) else ""
        if route.endswith("/claims/"):
            claim, _, tail = rest.partition("/")
            if claim and tail == "history":
                return True
        elif rest and "/" not in rest:
            return True
    return False


class Source:
    """The console's one way to the sidecar and the instance. ``home``: the instance directory (default the selected
    one); ``bench``: a paired benchmark results directory for the bench report and the faculties' evidence."""

    def __init__(self, home: Optional[str] = None, *, bench: Optional[str | Path] = None):
        from colonymind.config import instance_home
        from colonymind.mind.cli import Sidecar
        sidecar = Sidecar(home)
        self.home = instance_home(home)
        self.url, self._key = sidecar.url, sidecar.key
        self.bench_dir = Path(bench).expanduser() if bench else None

    # -- the sidecar ------------------------------------------------------------------------------------------------

    def call(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json_body: Any = None,
             timeout: float = 30.0) -> Any:
        if not allowed(method, path):
            raise Unavailable(None, "not_a_console_route", f"the console does not call {method} {path}")
        import httpx
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.request(method, self.url + path, headers=headers, params=params, json=json_body)
        except httpx.HTTPError as error:
            raise Unavailable(None, "unreachable",
                              f"the sidecar is not answering at {self.url} ({type(error).__name__})") from None
        if response.is_success:
            try:
                return response.json()
            except ValueError:
                raise Unavailable(response.status_code, "not_json", "the sidecar's answer was not JSON") from None
        detail: Any = None
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            pass
        if isinstance(detail, dict):
            code, message = str(detail.get("code") or response.status_code), str(detail.get("message") or
                                                                                 detail.get("code") or "")
        else:
            code, message = str(response.status_code), str(detail or response.reason_phrase)
        raise Unavailable(response.status_code, code, f"the sidecar answered HTTP {response.status_code}: {message}")

    def state(self) -> Dict[str, Any]:
        return self.call("GET", "/v1/mind/state")

    def concerns(self) -> Dict[str, Any]:
        return self.call("GET", "/v1/mind/concerns", params={"limit": 50})

    def goals(self) -> Dict[str, Any]:
        return self.call("GET", "/v1/mind/goals")

    def log(self, limit: int = 50) -> Dict[str, Any]:
        return self.call("GET", "/v1/mind/log", params={"limit": limit})

    def why(self, ident: str) -> Dict[str, Any]:
        return self.call("GET", f"/v1/mind/why/{quote(ident, safe='')}")

    def asks(self) -> Dict[str, Any]:
        return self.call("GET", "/v1/mind/asks")

    def health(self) -> Dict[str, Any]:
        return self.call("GET", "/v1/host/health")

    def recall(self, person: str, question: str, *, limit: int = 5) -> Dict[str, Any]:
        """The search ``colonymind recall --explain`` makes, traced. It names no session, so it admits person-scoped
        evidence only and supplies nothing to any turn."""
        return self.call("POST", "/v1/host/memory/search", json_body={
            "identity": {"host_id": "colonymind-console"}, "person_id": person, "query": question, "limit": limit,
            "trace": True})

    def inventory(self, person: str, *, page: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        params: Dict[str, Any] = {"person_id": person, "limit": limit}
        if page:
            params["page"] = page
        return self.call("GET", "/v1/host/memory/inventory", params=params)

    def history(self, claim: str, *, person: Optional[str] = None) -> Dict[str, Any]:
        return self.call("GET", f"/v1/host/memory/inventory/claims/{quote(claim, safe=':')}/history",
                         params={"person_id": person} if person else None)

    # -- the off switch ---------------------------------------------------------------------------------------------

    @property
    def marker(self) -> Path:
        from colonymind import get_state_dir
        from colonymind.mind.tick import OFF_MARKER
        return get_state_dir() / OFF_MARKER

    def off(self) -> Dict[str, Any]:
        """The mind off, as ``colonymind mind off`` turns it off: through the sidecar, or, when it does not take it,
        by the marker file the mind reads when it starts. Idempotent: a mind the sidecar reports off, and a marker
        already written, are left as they are. ``how``: ``now`` (the sidecar took it), ``already`` (the sidecar
        reports the mind off), ``marker`` (the sidecar did not take it: the marker is written) or ``marker_set`` (it
        still did not, and the marker was already there). A marker takes effect at the mind's next start, and a
        running mind reads it only then: neither marker answer says a running mind stopped, and every press asks
        the sidecar again."""
        try:
            if self.state().get("enabled") is False:
                return {"how": "already"}
        except Unavailable:
            pass
        try:
            value = self.call("POST", "/v1/mind/off", json_body={"reason": OFF_REASON, "by": OFF_BY})
            return {"how": "now", "cancelled": int(value.get("cancelled_messages") or 0)}
        except Unavailable as error:
            marker = self.marker
            if marker.exists():
                return {"how": "marker_set", "note": error.message}
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{OFF_REASON} at {now_utc().isoformat()}\n", encoding="utf-8")
            return {"how": "marker", "note": error.message}

    # -- the reports, read from the instance's stores ---------------------------------------------------------------

    def faculties(self, since: str):
        """``colonymind report faculties``, with the bench directory's evidence beside each faculty when one is given."""
        from colonymind.reports import bench, faculties
        loaded = bench.load(self.bench_dir) if self.bench_dir else None
        return faculties.document(faculties.collect(self.home, since=since, bench=loaded))

    def instinct(self, since: str):
        from colonymind.reports import instinct
        return instinct.document(instinct.collect(self.home, since=since))

    def usage(self, since: str):
        from colonymind.reports import usage
        return usage.document(usage.collect(self.home, since=since))

    def bench(self):
        """``colonymind report bench DIR``, or None when the console was started without a results directory."""
        from colonymind.reports import bench
        return bench.document(bench.load(self.bench_dir)) if self.bench_dir else None


__all__ = ["OFF_BY", "OFF_REASON", "ROUTES", "Source", "Unavailable", "allowed"]
