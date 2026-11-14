"""The cron wake gate: a job's pre-run script that skips a fire with nothing new for the job.

Stock Hermes runs a job's ``script`` before the agent and skips the run when the script's last output line is
``{"wakeAgent": false}``; whatever else it prints is put before the job's prompt. A gated job's script is a two-line
stub in the Hermes scripts directory (``python -m colonymind_gates.cron_gate stub <job>`` prints it)::

    from colonymind_gates.cron_gate import main
    raise SystemExit(main("<job>", __file__))

On each fire the gate:

1. runs the job's own script first, when it has one, where Hermes would have run it (the job's workdir, else the
   script's own directory), and keeps its answer: a failure, an empty output (Hermes skips the run) or its own
   ``{"wakeAgent": false}`` stands as it is, and the gate asks nothing;
2. reads the job's declared sources (files, a loopback URL, a command's output) and hashes them: every source
   unchanged since the last fire is the rules tier's answer, and no model is asked; a change that brings no new item
   (something removed or reordered) wakes the job without asking;
3. otherwise cuts what is new into items (blocks between blank lines, each short enough that the question's whole
   state fits the fast tier's 1,200 characters) and asks the job's point once per item, all in one request;
4. skips only when the sidecar says to apply a ``no`` for every item (the point is live), and never more than
   ``force_wake_after`` fires in a row; any error, a 2 s budget spent, a missing sidecar or a bad configuration
   wakes (the budget covers reading the sources: a read still blocked when it is spent is abandoned). In shadow it
   always wakes, and the sidecar logs what it would have done;
5. when it wakes, prints the job's own output and then one fixed line, ``WAKE_LINE``, so an operator can qualify a
   job with that line before the gate ever skips;
6. binds the fire to the run Hermes is executing (its execution id in Hermes' own ledger) and, on the next fire,
   reads how that run ended from its ledger row (an answer delivered, or nothing to deliver) and reports it as that
   fire's decisions' outcome; never from a run record's words, whose prompt carries whatever a script printed.

Its own state (``state/colonymind-gates/cron-<job>.json`` in the Hermes home) holds hashes and ids, never text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets as _random
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from . import config, windows

logger = logging.getLogger(__name__)

WAKE_LINE = "[colonymind-gate] this run proceeds"
SKIP_LINE = '{"wakeAgent": false}'
STUB = 'from colonymind_gates.cron_gate import main\nraise SystemExit(main("{name}", __file__))\n'
#: A source larger than this is not read (my choice): the job wakes.
MAX_SOURCE_BYTES = 1_000_000
#: At most this many item hashes are kept per source (my choice).
KEEP_ITEMS = 4096
#: Of the gate's budget, at most this long goes to reporting the previous run's outcome (my choice).
OUTCOME_DEADLINE_S = 0.5
_BLOCKS = re.compile(r"\n\s*\n")
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EXECUTION_ID = re.compile(r"^[0-9a-f]{32}$")
#: How Hermes' ledger records a completed run that had an answer to deliver: delivered, queued for later, with no
#: target resolved, or a delivery that failed. ``suppressed`` is a run with nothing to deliver.
_ANSWERED = frozenset({"delivered", "queued", "not_configured", "failed"})


class SourceError(RuntimeError):
    """A source could not be read: the job wakes."""


def _hermes_home() -> Path:
    from colonymind_hermes.client import hermes_home
    return hermes_home()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]


def _last_line_skips(output: str) -> bool:
    """Stock Hermes' reading of a script's output: the last non-empty line is ``{"wakeAgent": false}``."""
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return False
    try:
        gate = json.loads(lines[-1].strip())
    except ValueError:
        return False
    return isinstance(gate, dict) and gate.get("wakeAgent", True) is False


# -- the job's own script ------------------------------------------------------------------------------------------

def _script(name: str) -> tuple[list[str], Path]:
    scripts = (_hermes_home() / "scripts").resolve()
    path = (scripts / Path(name).expanduser()).resolve()
    try:
        path.relative_to(scripts)
    except ValueError:
        raise SourceError("the job's script must be in the Hermes scripts directory") from None
    if not path.is_file():
        raise SourceError("the job's script is missing")
    argv = ["bash", str(path)] if path.suffix.lower() in {".sh", ".bash"} else [sys.executable, str(path)]
    return argv, path


def own_workdir(script: Path, stub: str | None) -> str:
    """Where Hermes would have run the job's own script: the job's ``workdir`` when it has one (a directory that
    exists), else the script's own directory. Hermes started the stub the same way (in the workdir, or in the stub's
    directory), so without the job's record the stub's own directory tells which."""
    record = job_record(stub)
    if record:
        workdir = str(record.get("workdir") or "").strip()
        return workdir if workdir and Path(workdir).is_dir() else str(script.parent)
    here = Path.cwd().resolve()
    return str(script.parent) if stub and here == Path(stub).resolve().parent else str(here)


def _run_own_script(name: str, stub: str | None = None) -> tuple[int, str, str]:
    try:
        argv, path = _script(name)
    except SourceError as error:
        return 1, "", str(error)
    done = subprocess.run(argv, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL,
                          cwd=own_workdir(path, stub))
    return done.returncode, done.stdout, done.stderr


# -- the sources ---------------------------------------------------------------------------------------------------

def _read(source: Mapping[str, Any], deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SourceError("the gate's budget is spent")
    if "file" in source:
        path = Path(source["file"]).expanduser()
        path = path if path.is_absolute() else _hermes_home() / path
        try:
            status = path.stat()
            if not stat.S_ISREG(status.st_mode):        # a pipe or a device can wait for a writer forever
                raise SourceError("a source file is not a regular file")
            if status.st_size > MAX_SOURCE_BYTES:
                raise SourceError("a source is too large to read")
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raise SourceError("a source file could not be read") from None
    if "url" in source:
        import httpx
        body = bytearray()
        try:
            # The timeout bounds each network wait; the loop bounds the whole response by the budget and the size.
            with httpx.stream("GET", source["url"], timeout=remaining, trust_env=False,
                              follow_redirects=False) as response:
                if response.status_code != 200:
                    raise SourceError("a source URL answered no content")
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_SOURCE_BYTES or time.monotonic() > deadline:
                        raise SourceError("a source URL did not answer within the budget")
                encoding = response.encoding or "utf-8"
        except httpx.HTTPError:
            raise SourceError("a source URL could not be read") from None
        return bytes(body).decode(encoding, errors="replace")
    try:
        done = subprocess.run(source["command"], capture_output=True, text=True, errors="replace",
                              timeout=remaining, cwd=_hermes_home() / "scripts", stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        raise SourceError("a source command did not run") from None
    if done.returncode != 0 or len(done.stdout) > MAX_SOURCE_BYTES:
        raise SourceError("a source command failed")
    return done.stdout


def collect(sources: tuple[dict, ...], deadline: float) -> list[str]:
    """Every source's text, within the gate's budget. The reads run on a worker thread, and the gate stops waiting
    the moment the budget is spent, whatever a read is still blocked on (a file system, a stream, a command's
    children); a read that is still running is abandoned and the job wakes."""
    from .client import bounded
    texts = bounded(lambda: [_read(source, deadline) for source in sources], deadline - time.monotonic())
    if texts is None:
        raise SourceError("the sources were not read within the gate's budget")
    return texts


def items_of(text: str) -> list[tuple[str, str]]:
    """The items of a source's text, as ``(item, sent)``: its blocks between blank lines, each cut into pieces of at
    most ``config.ITEM_CHARS`` (a piece is asked alone; any yes wakes the job), so that the question's whole state,
    its own words and the longest watch included, fits the fast tier. Secrets are found over the whole text before
    it is cut, so ``sent`` hides a value even when its label and the value fall in different pieces."""
    text = text.replace("\r\n", "\n")
    spans = windows.hidden_spans(text)
    out, start = [], 0
    for separator in [*_BLOCKS.finditer(text), None]:
        end = separator.start() if separator is not None else len(text)
        block = text[start:end]
        first = start + len(block) - len(block.lstrip())
        last = start + len(block.rstrip())
        out += [(text[piece.start:piece.end], piece.sent)
                for piece in windows.pieces(text, config.ITEM_CHARS, spans, start=first, end=last)]
        start = separator.end() if separator is not None else len(text)
    return [(item, sent) for item, sent in out if item.strip()]


# -- Hermes' own records (read only) ---------------------------------------------------------------------------------

def job_record(stub: str | None) -> dict[str, Any]:
    """The Hermes job whose script is this stub, from Hermes' own job list; {} when there is not exactly one."""
    if not stub:
        return {}
    home = _hermes_home()
    try:
        data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    jobs = data.get("jobs", []) if isinstance(data, dict) else data
    if isinstance(jobs, dict):
        jobs = [dict(job, id=job.get("id", key)) for key, job in jobs.items() if isinstance(job, dict)]
    scripts, mine = home / "scripts", Path(stub).resolve()
    found = []
    for job in jobs if isinstance(jobs, list) else []:
        script = job.get("script") if isinstance(job, dict) else None
        if not isinstance(script, str) or not script.strip():
            continue
        path = Path(script).expanduser()
        if (path if path.is_absolute() else scripts / path).resolve() == mine:
            found.append(job)
    return found[0] if len(found) == 1 and _JOB_ID.match(str(found[0].get("id") or "")) else {}


def job_id(stub: str | None) -> str:
    """The id of the Hermes job whose script is this stub; "" when there is not exactly one."""
    return str(job_record(stub).get("id") or "")


def delivers_locally(job: Mapping[str, Any]) -> bool:
    """Whether the job's answers stay local (Hermes' default): then its ledger records every run as suppressed,
    whatever it said, and no outcome can be read from it."""
    deliver = job.get("deliver")
    if isinstance(deliver, (list, tuple)):
        deliver = ",".join(str(part).strip() for part in deliver if str(part).strip())
    return str(deliver or "local").strip().lower() == "local"


def _ledger() -> sqlite3.Connection:
    """Hermes' cron execution ledger, opened read only."""
    path = (_hermes_home() / "cron" / "executions.db").resolve()
    if not path.is_file():
        raise sqlite3.OperationalError("no execution ledger")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.2)


def running_execution(job: str) -> str:
    """The execution Hermes is running for ``job`` now, the one that ran this script, from Hermes' own execution
    ledger (read only); "" when there is not exactly one."""
    if not job:
        return ""
    try:
        with closing(_ledger()) as conn:
            rows = conn.execute("SELECT id FROM executions WHERE job_id = ? AND status = 'running' LIMIT 2",
                                (job,)).fetchall()
    except sqlite3.Error:
        return ""
    return str(rows[0][0]) if len(rows) == 1 and _EXECUTION_ID.match(str(rows[0][0])) else ""


def run_outcome(execution: str) -> str:
    """How the run ``execution`` ended, from its own row in Hermes' execution ledger: ``delivered`` when it
    completed with an answer to deliver, ``silent`` when it completed with none (a silence marker, no response or
    nothing new); "" when it failed, has not ended or is gone. Never read from a run record's words: its prompt
    carries whatever a script printed."""
    if not execution:
        return ""
    try:
        with closing(_ledger()) as conn:
            row = conn.execute("SELECT status, delivery_outcome FROM executions WHERE id = ?",
                               (execution,)).fetchone()
    except sqlite3.Error:
        return ""
    if row is None or row[0] != "completed":
        return ""
    return "delivered" if row[1] in _ANSWERED else "silent" if row[1] == "suppressed" else ""


# -- the gate ------------------------------------------------------------------------------------------------------

class Gate:
    def __init__(self, job: config.CronJob, stub: str | None, client: Any = None):
        self.job, self.stub, self._client = job, stub, client
        self.path = _hermes_home() / "state" / "colonymind-gates" / f"cron-{job.name}.json"

    @property
    def client(self) -> Any:
        if self._client is None:
            from .client import GateClient
            self._client = GateClient()
        return self._client

    def _state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save(self, state: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True)
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def fire(self) -> bool:
        """True to skip this fire. Raises nothing it can help: an error wakes."""
        deadline = time.monotonic() + config.CRON_DEADLINE_S
        state = self._state()
        previous = state.get("last") if isinstance(state.get("last"), dict) else {}
        record = job_record(self.stub)
        job = str(record.get("id") or "")
        known = state.get("sources") if isinstance(state.get("sources"), dict) else {}
        sources: dict[str, dict[str, Any]] = {}
        fresh: list[str] = []
        for index, (source, text) in enumerate(zip(self.job.sources, collect(self.job.sources, deadline))):
            key = f"{index}:{_digest(json.dumps(source, sort_keys=True))}"
            items = items_of(text)
            hashes = [_digest(item) for item, _ in items]
            seen = set((known.get(key) or {}).get("items") or [])
            fresh += [sent for (_, sent), digest in zip(items, hashes) if digest not in seen]
            sources[key] = {"hash": _digest(text), "items": hashes[-KEEP_ITEMS:]}
        # Unchanged means every source hashes as it did last time: only then is it the rules' skip. A change with no
        # new item (something removed or reordered) cannot be asked about, so it wakes without asking.
        unchanged = set(known) == set(sources) and all(
            isinstance(known[key], dict) and known[key].get("hash") == value["hash"] for key, value in sources.items())
        fire = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{_random.token_hex(3)}"
        refs = {"cron_job": job or self.job.name, "cron_fire": fire}
        decisions = None
        if unchanged:
            requests = [{"point": self.job.point, "unchanged": True, "refs": refs}]
        elif not fresh:
            requests = []          # changed, but nothing new to ask about: wake without asking
        elif len(fresh) <= config.MAX_ITEMS:
            requests = [{"point": self.job.point, "fields": {"item": item},
                         "refs": {**refs, "item": str(index)}} for index, item in enumerate(fresh)]
        else:
            requests = []          # too much is new to call it nothing: wake without asking
        if requests:
            decisions = self.client.decide({"event": "cron_fire", "requests": requests},
                                           deadline_s=deadline - time.monotonic())
        skips = int(state.get("skips") or 0) if isinstance(state.get("skips"), int) else 0
        # Only an applied no from a live gate, for every item: anything else (shadow, unsure, a yes) wakes.
        skip = bool(decisions) and all(item.get("apply") is True and item.get("state") == "live"
                                       and item.get("label") == "no" for item in decisions) \
            and skips < self.job.force_wake_after
        ids = [str(item.get("decision_id")) for item in decisions or [] if item.get("decision_id")]
        self._report(previous, deadline)
        # The run this fire belongs to, bound by Hermes' own execution id: the next fire reads how it ended. A job
        # that delivers locally is recorded as suppressed whatever it said, so it binds none.
        execution = "" if skip or delivers_locally(record) else running_execution(job)
        self._save({"sources": sources, "skips": skips + 1 if skip else 0,
                    "last": {"decisions": ids, "at": time.time(), "woke": not skip, "execution": execution}})
        return skip

    def _report(self, previous: Mapping[str, Any], deadline: float) -> None:
        """The previous fire's run, when it woke: how it ended, read from that run's own execution record, as its
        decisions' outcome."""
        ids = [str(item) for item in previous.get("decisions") or [] if isinstance(item, str)]
        execution = previous.get("execution")
        if not ids or not previous.get("woke") or not isinstance(execution, str) or not execution:
            return
        outcome = run_outcome(execution)
        remaining = min(OUTCOME_DEADLINE_S, deadline - time.monotonic())
        if outcome and remaining > 0:
            self.client.outcome(ids, outcome, deadline_s=remaining)


def main(name: str, stub: str | None = None) -> int:
    """One fire of the gated job ``name``; prints what Hermes reads and returns the exit status."""
    try:
        job = config.cron_job(config.section(), name)
    except Exception as error:               # a missing or bad configuration wakes the job
        logger.warning("cron gate %s: %s; the job wakes", name, error)
        print(WAKE_LINE)
        return 0
    own = ""
    if job.script:
        code, stdout, stderr = _run_own_script(job.script, stub)
        if code != 0 or not stdout.strip() or _last_line_skips(stdout):
            # The job's own answer stands as it was: its failure, its empty output (Hermes skips the run) or its
            # own skip. The gate asks nothing.
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)
            return code
        own = stdout.rstrip("\n")
    try:
        skip = Gate(job, stub).fire()
    except Exception as error:
        logger.warning("cron gate %s failed (%s); the job wakes", name, type(error).__name__)
        skip = False
    if skip:
        print(SKIP_LINE)
    else:
        if own:
            print(own)
        print(WAKE_LINE)
    return 0


def cli(argv: list[str]) -> int:
    """``stub <job>``: print the job's two-line stub script."""
    if len(argv) == 2 and argv[0] == "stub" and re.match(r"^[a-z][a-z0-9_]{0,40}$", argv[1]):
        sys.stdout.write(STUB.format(name=argv[1]))
        return 0
    print("usage: python -m colonymind_gates.cron_gate stub <job>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))


__all__ = ["SKIP_LINE", "STUB", "WAKE_LINE", "Gate", "cli", "collect", "delivers_locally", "items_of", "job_id",
           "job_record", "main", "own_workdir", "run_outcome", "running_execution"]
