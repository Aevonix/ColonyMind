"""``colonymind demo``: a disposable instance, a scripted model, no network and no keys.

The demo tells a fact in one session and recalls it in another (with ``recall --explain``), runs one mind tick
on a shifted clock that follows up an overdue promise with its quoted reason, shows an ask in the owner's
notice format and waits for the person to type the answer, then runs ``mind why`` on the result. These tests
drive it as a person would, from a terminal: they read what it prints and type into it. They check that it
finishes within its bound with no network and no keys, leaves no file outside its temporary directory and no
process behind, and that an ask stays open until an answer is typed (the demo never answers for the owner).
"""
from __future__ import annotations

import hashlib
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import psutil
import pytest
import yaml

# The demo's bound in CI (a chosen bound, not a measured rule): three minutes.
BOUND_SECONDS = 180
PROMPT = "Type your reply as the owner"

# Loaded into the demo and every process it starts (PYTHONPATH): a connection or a name lookup that is not
# loopback is written down and refused, so "no network" is observed, not assumed.
NETWORK_GUARD = r'''
import ipaddress, os, socket

_LOG = os.environ.get("DEMO_TEST_NETWORK_LOG") or ""


def _local(host):
    if host in (None, "", "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _refuse(what):
    if _LOG:
        with open(_LOG, "a", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()} {what!r}\n")
    raise OSError(f"network use refused by the demo test: {what!r}")


def _guard(method):
    original = getattr(socket.socket, method)

    def guarded(self, address, *args):
        target = address if method != "sendto" else (args[-1] if args else None)
        if self.family in (socket.AF_INET, socket.AF_INET6) and isinstance(target, tuple) \
                and not _local(target[0]):
            _refuse((method, target))
        return original(self, address, *args)
    setattr(socket.socket, method, guarded)


if _LOG:
    for _method in ("connect", "connect_ex", "sendto"):
        _guard(_method)
    _getaddrinfo = socket.getaddrinfo

    def _lookup(host, *args, **kwargs):
        if not _local(host.decode() if isinstance(host, bytes) else host):
            _refuse(("getaddrinfo", host))
        return _getaddrinfo(host, *args, **kwargs)
    socket.getaddrinfo = _lookup
'''


def _tree(root: Path) -> dict[str, str]:
    """Every file and directory under ``root`` with its mode and content digest."""
    found = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            found[relative] = "link:" + os.readlink(path)
        elif path.is_dir():
            found[relative] = f"dir:{path.stat().st_mode:o}"
        else:
            found[relative] = f"{path.stat().st_mode:o}:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def _processes_mentioning(text: str) -> list[str]:
    """Live processes whose command line mentions ``text``, read from the platform's process table (Linux and
    macOS alike; macOS has no ``/proc``). A process that ends or hides its command line meanwhile is skipped."""
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        command = " ".join(process.info["cmdline"] or [])
        if text in command:
            found.append(f"{process.info['pid']}: {command[:200]}")
    return found


class DemoRun:
    """``colonymind demo`` in a terminal-like pipe, with a fake home that stands for the user's own instance."""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "home"
        (self.home / ".colonymind").mkdir(parents=True)
        (self.home / ".hermes").mkdir()
        # The user's own instance and Hermes home: the demo must neither read them as its own nor write to them.
        (self.home / ".colonymind" / "colonymind.yaml").write_text(yaml.safe_dump({
            "sidecar": {"host": "127.0.0.1", "port": 7777}, "owner": {"contact_id": "cid-users-own-owner"}}))
        (self.home / ".colonymind" / "api.key").write_text("users-own-key\n")
        (self.home / ".colonymind" / "api.key").chmod(0o600)
        (self.home / ".hermes" / "config.yaml").write_text("model:\n  provider: custom\n  default: users-model\n")
        (self.home / "notes.txt").write_text("the user's own file\n")
        self.tmp = tmp_path / "tmp"
        self.tmp.mkdir()
        guard = tmp_path / "guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(NETWORK_GUARD)
        self.network_log = tmp_path / "network.log"
        env = {key: value for key, value in os.environ.items()
               if not any(mark in key.upper() for mark in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
               and not key.startswith(("COLONYMIND_", "HERMES_"))}
        env.update({
            "HOME": str(self.home), "TMPDIR": str(self.tmp),
            "XDG_CONFIG_HOME": str(self.home / ".config"), "XDG_DATA_HOME": str(self.home / ".local" / "share"),
            "XDG_CACHE_HOME": str(self.home / ".cache"),
            "COLONYMIND_HOME": str(self.home / ".colonymind"), "HERMES_HOME": str(self.home / ".hermes"),
            # What the user's shell might carry: the demo must not take any of it for its own instance.
            "COLONYMIND_API_KEY": "users-own-key", "COLONYMIND_SIDECAR_PORT": "7777",
            "COLONYMIND_EMBED_BASE_URL": "http://192.0.2.10/v1", "COLONYMIND_RERANKER_BASE_URL": "http://192.0.2.11/v1",
            "https_proxy": "http://192.0.2.12:3128", "HTTP_PROXY": "http://192.0.2.12:3128",
            "PYTHONPATH": os.pathsep.join([str(guard), *filter(None, [os.environ.get("PYTHONPATH")])]),
            "DEMO_TEST_NETWORK_LOG": str(self.network_log),
            "PYTHONUNBUFFERED": "1",
        })
        self.env = env
        self.before = _tree(self.home)
        self.lines: list[str] = []
        self._queue: queue.Queue = queue.Queue()
        self.started = time.monotonic()
        self.process = subprocess.Popen([sys.executable, "-m", "colonymind", "demo"], env=env, cwd=tmp_path,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        self.root: Path | None = None

    def _read(self) -> None:
        for line in self.process.stdout:
            self._queue.put(line.rstrip("\n"))
        self._queue.put(None)

    def wait_for(self, needle: str, *, timeout: float = BOUND_SECONDS) -> str:
        deadline = self.started + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AssertionError(f"no line with {needle!r} in time; output so far:\n" + "\n".join(self.lines))
            try:
                line = self._queue.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                raise AssertionError(f"the demo ended before {needle!r}; output:\n" + "\n".join(self.lines))
            self.lines.append(line)
            if line.startswith("Working in "):
                self.root = Path(line.split("Working in ", 1)[1].split(" ", 1)[0])
            if needle in line:
                return line

    def type(self, text: str) -> None:
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def close_input(self) -> None:
        self.process.stdin.close()

    def finish(self) -> int:
        left = max(1.0, self.started + BOUND_SECONDS - time.monotonic())
        try:
            code = self.process.wait(timeout=left)
        except subprocess.TimeoutExpired:
            self.process.send_signal(signal.SIGTERM)
            self.process.wait(timeout=30)
            raise AssertionError("the demo did not finish within its bound; output:\n" + "\n".join(self.lines))
        while True:
            line = self._queue.get(timeout=10)
            if line is None:
                break
            self.lines.append(line)
        self.elapsed = time.monotonic() - self.started
        return code

    def sidecar(self) -> httpx.Client:
        """The demo's own sidecar, found through the instance it printed."""
        assert self.root is not None, "the demo did not say where it works"
        instance = self.root / "home" / ".colonymind"
        port = yaml.safe_load((instance / "colonymind.yaml").read_text())["sidecar"]["port"]
        key = (instance / "api.key").read_text().strip()
        return httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": f"Bearer {key}"},
                            timeout=30, trust_env=False)

    def text(self) -> str:
        return "\n".join(self.lines)

    def kill(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()


@pytest.fixture(scope="module")
def answered(tmp_path_factory):
    """One full run: the person waits a moment, types a line that is no answer, then the answer the notice asks
    for. Each test below reads its own part of this run."""
    run = DemoRun(tmp_path_factory.mktemp("demo-answered"))
    try:
        run.wait_for(PROMPT)
        with run.sidecar() as sidecar:
            open_asks = sidecar.get("/v1/mind/asks").json()["asks"]
            run.open_before = [(ask["id"], ask["status"]) for ask in open_asks]
            ask = open_asks[0]
            time.sleep(2)
            sidecar.post("/v1/mind/tick")
            run.after_waiting = sidecar.get(f"/v1/mind/why/{ask['id']}").json()["status"]
            run.type("maybe later")
            run.wait_for("not an answer")
            run.after_no_answer = sidecar.get(f"/v1/mind/why/{ask['id']}").json()["status"]
        run.ask = ask
        # The person answers in their own words, with no code: the reply goes to the ask's own message.
        notice = "\n".join(line for line in run.lines if line.startswith("  | "))
        run.notice = notice
        run.type("sure, go ahead")
        run.code = run.finish()
        yield run
    finally:
        run.kill()


def test_the_demo_completes_in_its_bound_with_no_network_and_no_keys(answered):
    text = answered.text()
    assert answered.code == 0, text
    assert answered.elapsed < BOUND_SECONDS, f"{answered.elapsed:.0f} s"
    assert not answered.network_log.exists(), answered.network_log.read_text()
    for step in ("== 1.", "== 2.", "== 3.", "== 4."):
        assert step in text, text
    # 1: the fact from session 1 reaches session 2, and recall --explain says why.
    assert "[session 2] Agent: You told me earlier: the spare key is under the blue flower pot." in text, text
    assert "$ colonymind recall --explain" in text and "Recall for " in text and "found by keyword" in text, text
    # 2: one mind tick on the shifted clock follows up the overdue promise, quoting it.
    assert "COLONYMIND_MIND_CLOCK_OFFSET_SECONDS=7200" in text, text
    assert "Formed: Overdue: Send them the weekly report" in text, text
    assert "\"I'll send you the weekly report within the hour.\"" in text, text
    # 4: mind why on the approved action.
    assert "$ colonymind mind why " + answered.ask["id"] in text, text
    assert "outcome approved" in text, text


def test_the_demo_leaves_no_file_outside_its_directory_and_no_process(answered):
    assert _tree(answered.home) == answered.before
    assert list(answered.tmp.iterdir()) == []
    assert answered.root is not None and str(answered.root).startswith(str(answered.tmp))
    assert not answered.root.exists()
    assert _processes_mentioning(str(answered.root)) == []


def test_an_ask_stays_open_until_the_answer_is_typed(answered):
    assert answered.open_before == [(answered.ask["id"], "asked")]
    assert answered.after_waiting == "asked"      # waiting, and a tick meanwhile, answer nothing
    assert answered.after_no_answer == "asked"    # a typed line that is no answer leaves it open
    # The notice is the sidecar's own text, shown verbatim, and names what it asks about; a plain "sure, go
    # ahead" in reply approves it, no code typed.
    assert answered.ask["title"] in answered.notice
    assert "[session 3] Owner: sure, go ahead" in answered.text() and "-> approved" in answered.text()
    # The agent replies the way the host runs its turn: with what the sidecar says the answer did appended to the
    # owner's message, so it acknowledges the approval instead of saying it has nothing on record.
    after = answered.text().split("[session 3] Owner: sure, go ahead", 1)[1]
    reply = next(line for line in after.splitlines() if line.startswith("[session 3] Agent: "))
    assert reply == "[session 3] Agent: Understood: it is under way.", after


def test_without_an_answer_the_ask_stays_open_and_nothing_is_approved(tmp_path):
    run = DemoRun(tmp_path)
    try:
        run.wait_for(PROMPT)
        with run.sidecar() as sidecar:
            ask = sidecar.get("/v1/mind/asks").json()["asks"][0]
        run.close_input()
        code = run.finish()
    finally:
        run.kill()
    text = run.text()
    assert code == 0, text
    assert "No answer was typed: the ask stays open and nothing was approved." in text, text
    assert f"waiting for the owner (code {ask['ask_code']})" in text or "outcome asked" in text, text
    assert "-> approved" not in text and "outcome approved" not in text, text
    assert _tree(run.home) == run.before
    assert list(run.tmp.iterdir()) == []



def test_a_stop_while_it_waits_leaves_nothing_behind(tmp_path):
    """A terminal closed or a stop sent while the ask waits: the sidecar is stopped and the directory removed."""
    run = DemoRun(tmp_path)
    try:
        run.wait_for(PROMPT)
        run.process.send_signal(signal.SIGTERM)
        code = run.finish()
    finally:
        run.kill()
    assert code == 128 + signal.SIGTERM, run.text()
    assert "The disposable instance was removed." in run.text()
    assert run.root is not None and not run.root.exists()
    assert _processes_mentioning(str(run.root)) == []
    assert _tree(run.home) == run.before
    assert list(run.tmp.iterdir()) == []


def test_the_process_check_finds_a_live_process_on_a_host_with_no_proc(monkeypatch, tmp_path):
    """The leftover-process check reads the platform's process table, so it works where there is no ``/proc``
    (macOS), and it finds a process that does mention the directory: its empty answers above mean something."""
    marker = str(tmp_path / "demo-root")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", marker])
    listing = Path.iterdir

    def no_proc(self):
        if str(self) == "/proc":
            raise FileNotFoundError(2, "No such file or directory", "/proc")
        return listing(self)
    try:
        monkeypatch.setattr(Path, "iterdir", no_proc)
        found = _processes_mentioning(marker)
    finally:
        child.kill()
        child.wait(timeout=30)
    assert [line.split(":", 1)[0] for line in found] == [str(child.pid)], found
    assert _processes_mentioning(marker) == []


def test_the_demo_environment_is_its_own(tmp_path):
    base = {"HOME": "/home/someone", "PATH": "/usr/bin", "COLONYMIND_HOME": "/home/someone/.colonymind",
            "COLONYMIND_API_KEY": "k", "COLONYMIND_EMBED_BASE_URL": "http://192.0.2.1/v1", "HERMES_HOME": "/h",
            "OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "HTTPS_PROXY": "http://192.0.2.2:3128",
            "all_proxy": "socks5://192.0.2.3:1080", "PYTHONPATH": "/extra",
            "LITELLM_MODEL_COST_MAP_URL": "http://192.0.2.4/prices.json"}
    from colonymind import demo
    env = demo.demo_environment(tmp_path, base)
    assert env["PATH"] == "/usr/bin" and env["PYTHONPATH"] == "/extra"
    for name in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
                 "COLONYMIND_HOME", "COLONYMIND_STATE_DIR", "HERMES_HOME"):
        assert Path(env[name]).is_relative_to(tmp_path), name
    for name in ("COLONYMIND_API_KEY", "COLONYMIND_EMBED_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "HTTPS_PROXY", "all_proxy", "LITELLM_MODEL_COST_MAP_URL"):
        assert name not in env, name
    assert env["PYTHON_KEYRING_BACKEND"] == "keyring.backends.fail.Keyring"
    # The model library fetches a price list from the internet on import unless told to use its own copy.
    assert env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_the_demo_is_a_registered_command():
    result = subprocess.run([sys.executable, "-m", "colonymind", "demo", "--help"], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr
    assert "disposable" in result.stdout and "no network" in result.stdout.lower()
