"""``colonymind demo``: first value in a few minutes, with no model, no keys and no network.

A disposable instance in a new temporary directory runs a real sidecar against the scripted model
(``colonymind.devtools.fake_model``, on a loopback port served by this process). The demo plays the host the way
the Hermes adapter does (recall before each turn, the turn captured after it, the mind's messages delivered to
the owner) and shows, in plain text:

1. a fact told in session 1 and recalled in session 2, with ``colonymind recall --explain``;
2. one mind tick on a shifted clock (the ``COLONYMIND_MIND_CLOCK_OFFSET_SECONDS`` seam): an overdue promise
   followed up, with the promise quoted as its reason;
3. an ask in the owner's notice format, read from the message the mind queued. The person types the answer and
   the demo never answers for the owner: what is typed goes in as the owner's own message, a reply to the ask's
   message, the way the host hands it to the sidecar (``POST /v1/mind/answers``) before its model replies, and
   the sidecar alone decides whether it answers the ask (no code is needed);
4. ``colonymind mind why`` on the resulting action.

It never touches the user's own instance or Hermes home: this process and every process it starts get an
environment of their own (``demo_environment``) whose home, instance, Hermes home and temporary directories are
inside the demo's directory, with none of the user's ``COLONYMIND_*`` or ``HERMES_*`` settings, keys or proxies.
At the end it stops the sidecar and the model and removes the directory (``--keep`` leaves the directory).
"""
from __future__ import annotations

import argparse
import copy
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from colonymind.util.temporal import now_utc

# Made-up names; nothing here is anyone's.
OWNER, AGENT = "Owner", "Agent"
FACT = "Please remember this for me: the spare key is under the blue flower pot."
QUESTION = "Where did I leave the spare key?"
PROMISE = "Send me the weekly report within the hour."
FLOOR_WORK = "Clean up the old build folders within the hour."
SESSIONS = {"session 1": "demo-session-1", "session 2": "demo-session-2", "session 3": "demo-session-3"}
# How far ahead of the wall clock the mind's clock runs once the sidecar restarts (my choice): past the hour
# both promises are due in, so each is overdue by about an hour when the mind next looks.
CLOCK_OFFSET_SECONDS = 2 * 3600
# The longest the demo waits for any one thing the sidecar does (my choice); the whole run takes well under a
# minute on a laptop.
WAIT_SECONDS = 120.0
# The owner speaks on the local terminal, as with `hermes chat`; the demo's key is granted the owner there.
PLATFORM = "cli"
CHAT = "demo-chat"
# The executor toolset that makes every task of the worker floor work (it asks at every level).
SHELL = "terminal"
# The outbox messages that put an ask in front of the owner: the notice, an ask shown again, and the one plain
# question about an unclear answer (each binds the owner's reply like the ask's own message).
ASK_MESSAGE_TYPES = ("ask_notice", "ask_show", "ask_clarify")

_SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_DROPPED_PREFIXES = ("COLONYMIND_", "HERMES_", "LITELLM_")
_PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "FTP_PROXY")


class DemoError(RuntimeError):
    """A step the demo could not complete; the message says which and why."""


def demo_environment(root: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment of the demo and of every process it starts: ``base`` without the user's instance, Hermes
    home, keys or proxies, and with home, instance, Hermes home and temporary directories inside ``root``."""
    env = {name: value for name, value in base.items()
           if not name.upper().startswith(_DROPPED_PREFIXES)
           and not any(mark in name.upper() for mark in _SECRET_MARKS)
           and name.upper() not in _PROXIES}
    home = root / "home"
    env.update({
        "HOME": str(home),
        "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
        "XDG_CONFIG_HOME": str(home / ".config"), "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_CACHE_HOME": str(home / ".cache"), "XDG_STATE_HOME": str(home / ".local" / "state"),
        "COLONYMIND_HOME": str(home / ".colonymind"), "COLONYMIND_STATE_DIR": str(home / ".colonymind"),
        "HERMES_HOME": str(home / ".hermes"),
        # Nothing of the demo may reach the operating system's keyring.
        "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        # The model library would fetch its price list from the internet on import; the copy it ships is used.
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    })
    return env


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "demo", help="A disposable demo with a scripted model: memory across sessions, a mind tick and an ask "
                     "you answer (no network, no keys)",
        description="Runs a disposable instance with a scripted model in a new temporary directory: no network, "
                    "no keys, and nothing of your own instance or Hermes home is read or changed. It shows a "
                    "fact recalled in a later session (with recall --explain), one mind tick following up an "
                    "overdue promise, an ask you answer yourself, and mind why on the result.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the disposable directory at the end (the sidecar is still stopped)")


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    saved = dict(os.environ)
    root = Path(tempfile.mkdtemp(prefix="colonymind-demo-")).resolve()
    demo = Demo(root)
    previous = _on_signals(_stop)
    code = 1
    try:
        code = demo.run()
    except DemoError as error:
        print(f"\ncolonymind demo stopped: {error}", flush=True)
        demo.show_log()
    except KeyboardInterrupt:
        print("\ncolonymind demo interrupted.", flush=True)
        code = 130
    finally:
        demo.close()
        os.environ.clear()
        os.environ.update(saved)
        _on_signals(previous)
        if getattr(args, "keep", False):
            ending = f"The disposable directory is kept: {root}"
        else:
            shutil.rmtree(root, ignore_errors=True)
            ending = "The disposable instance was removed."
        print(f"\nDone in {time.monotonic() - started:.0f} s. {ending}", flush=True)
    return code


def _stop(signum: int, _frame: Any) -> None:
    """A terminal closed or a stop requested: unwind, so the sidecar is stopped and the directory removed."""
    raise SystemExit(128 + signum)


def _on_signals(handler: Any) -> Any:
    """Install ``handler`` (a callable, or the mapping this returned before) for the signals that end a terminal
    session; returns what was installed before, so a stop always runs the cleanup."""
    names = [name for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)]
    before = {name: signal.getsignal(getattr(signal, name)) for name in names}
    for name in names:
        signal.signal(getattr(signal, name), handler[name] if isinstance(handler, dict) else handler)
    return before


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _clock(value: Any) -> str:
    """An ISO time as ``HH:MM UTC``."""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    return moment.astimezone(timezone.utc).strftime("%H:%M UTC")


class Demo:
    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.instance = self.home / ".colonymind"
        self.env = demo_environment(root, os.environ)
        self.port = 0
        self.owner = ""
        self.model: Any = None
        self.model_url = ""
        self.client: Any = None
        self.sidecar: Optional[subprocess.Popen] = None
        self.log: Any = None
        self.said: dict[str, tuple[str, str]] = {}      # turn id -> (the owner's words, the agent's reply)
        self.turns = 0
        # The owner's chat as the host's plugin sees it: what reached it and when, on the mind's clock
        # (``/v1/mind/answers`` judges a reply at its arrival), and the delivery id of the last ask message.
        self.offset = 0
        self.ledger: list[dict] = []
        self.ask_message = ""
        self.sent = 0
        self.received = 0

    # -- the run ----------------------------------------------------------------------------------

    def run(self) -> int:
        for path in (self.home, self.root / "tmp", self.home / ".hermes"):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        # This process too: nothing below may read the user's own instance or Hermes home.
        os.environ.clear()
        os.environ.update(self.env)
        print("ColonyMind demo: a disposable instance and a scripted model on this machine. No network, no keys.")
        print(f"Working in {self.root} (removed at the end); your own instance and Hermes home are not touched.",
              flush=True)
        self._start_model()
        self._make_instance()
        self._start_sidecar(offset=0)
        self.memory_across_sessions()
        self.promise_followed_up()
        ask, settled = self.an_ask()
        self.why(ask, settled)
        return 0

    def memory_across_sessions(self) -> None:
        print("\n== 1. A fact told in one session, recalled in another ==")
        self.owner_says("session 1", FACT)
        print("\nWhat memory recalls for the next session's question, and why:")
        print(f"$ colonymind recall --explain --as {self.owner} \"{QUESTION}\"")
        self.cli("recall", QUESTION, "--as", self.owner, "--explain")
        print()
        self.owner_says("session 2", QUESTION)

    def promise_followed_up(self) -> None:
        print("\n== 2. A promise the mind follows up on its own ==")
        turn = self.owner_says("session 2", PROMISE)
        commitment = self._commitment_from(turn)
        print(f"Recorded: the agent owes you \"{commitment['description']}\", due {_clock(commitment['due_at'])}.")
        print("The sidecar restarts with the mind's clock two hours ahead of the wall clock "
              f"(COLONYMIND_MIND_CLOCK_OFFSET_SECONDS={CLOCK_OFFSET_SECONDS}); its mind ticks once.", flush=True)
        self._start_sidecar(offset=CLOCK_OFFSET_SECONDS)
        row = self._formed_for(commitment)
        self._show_formed(row, turn)
        if row.get("decision") == "act":
            print("  The mind acts without asking: this is ordinary work, a task for the host's worker.")

    def an_ask(self) -> tuple[dict, Optional[dict]]:
        print("\n== 3. An ask: floor work waits for your yes ==")
        print(f"The agent's worker now gets a shell (mind.worker_toolsets gains {SHELL}). A task that can run "
              "commands is floor work: it asks at every autonomy level.")
        turn = self.owner_says("session 3", FLOOR_WORK)
        commitment = self._commitment_from(turn)
        print(f"Recorded: the agent owes you \"{commitment['description']}\", due {_clock(commitment['due_at'])}.")
        self._give_the_worker_a_shell()
        print("The sidecar restarts with the new toolsets, the mind's clock still two hours ahead; its mind ticks "
              "once.", flush=True)
        self._start_sidecar(offset=CLOCK_OFFSET_SECONDS)
        row = self._formed_for(commitment)
        self._show_formed(row, turn)
        if row.get("status") != "asked":
            raise DemoError(f"the follow-up did not become an ask (status {row.get('status')})")
        notices = self._deliver_to_owner()
        print("\nThe mind's message to you, as the host delivers it to your chat:")
        self._show(notices)
        print(f"\nType your reply as the owner, in your own words; it goes as a reply to that message. The demo never "
              "answers for you; end the input (Ctrl-D) to leave the ask open.", flush=True)
        while True:
            if sys.stdin.isatty():
                print("> ", end="", flush=True)
            line = sys.stdin.readline()
            if not line:
                print("No answer was typed: the ask stays open and nothing was approved.", flush=True)
                return row, None
            text = line.strip()
            if not text:
                continue
            settled, why = self.owner_answers("session 3", text, row)
            if settled is not None:
                return row, settled
            print(f"That is not an answer yet ({why}): the ask stays open and nothing was approved.", flush=True)

    def why(self, ask: dict, settled: Optional[dict]) -> None:
        print("\n== 4. Why the mind did it ==")
        print(f"$ colonymind mind why {ask['id']}")
        self.cli("mind", "why", ask["id"])
        if settled is not None and settled.get("status") in {"approved", "dispatched"}:
            print("A running Hermes takes the approved task from the mind's dispatch queue and works it on its "
                  "kanban board; the demo stops here.")

    # -- the host's part --------------------------------------------------------------------------

    def owner_says(self, session: str, text: str) -> str:
        """One owner turn as the host runs it: recall for the message, the model's reply, the turn captured."""
        memory = self._recall(session, text)
        content = text + (f"\n\n<memory-context>\n{memory}\n</memory-context>" if memory else "")
        reply = self.chat(content)
        self.ledger.append({"at": self._mind_now(), "kind": "reply"})
        turn = self._capture(session, text, reply)
        print(f"[{session}] {OWNER}: {text}")
        print(f"[{session}] {AGENT}: {reply}", flush=True)
        return turn

    def owner_answers(self, session: str, text: str, ask: dict) -> tuple[Optional[dict], str]:
        """The owner's typed reply, handed to the sidecar the way the host adapter hands over the owner's own
        message before its model replies (``POST /v1/mind/answers``): a reply to the ask's own message, with what
        the chat showed and when. The demo does not read the words: the sidecar binds the message to the ask by
        what it replies to and reads it with its own tables, and says what the model is told. Then the model
        replies, told that as Hermes composes a turn (after the owner's words), and the turn is captured with the
        owner's words alone. ``(the settled ask or None, how the sidecar read the words)``."""
        self.received += 1
        turn = self._next_turn()
        value = self.api("POST", "/v1/mind/answers", {
            "platform": PLATFORM, "contact_id": self.owner, "chat_id": CHAT, "chat_type": "dm", "turn_id": turn,
            "message_id": f"demo-in-{self.received}", "arrived_at": self._mind_now(), "text": text,
            "reply_to_message_id": self.ask_message or None, "ledger": list(self.ledger)})
        # What the sidecar says the message did reaches the model the way the adapter's hook hands it to Hermes,
        # appended to the owner's message after a blank line, so the agent acknowledges an approval it caused.
        told = str(value.get("context") or "").strip()
        reply = self.chat(text + (f"\n\n{told}" if told else ""))
        self.ledger.append({"at": self._mind_now(), "kind": "reply"})
        self._capture(session, text, reply, turn=turn)
        print(f"[{session}] {OWNER}: {text}")
        settled = next((item for item in value.get("settled") or [] if item.get("id") == ask["id"]), None)
        reading = value.get("reading") or {}
        if settled is not None:
            print(f"  the sidecar: {reading.get('verdict')}, {reading.get('reason')}, as a reply to the ask's message: "
                  f"{ask['title']} -> {settled.get('status')}")
        if value.get("context"):
            print(f"  what the agent is told before it replies: {value['context']}")
        print(f"[{session}] {AGENT}: {reply}", flush=True)
        followups = self._deliver_to_owner(wait=False)
        if followups:
            print("\nThe mind's message to you:")
            self._show(followups)
        why = str(reading.get("reason") or value.get("reason") or "no answer word")
        return settled, why

    def chat(self, content: str) -> str:
        import httpx
        body = {"model": "scripted-model", "messages": [
            {"role": "system", "content": f"You are {AGENT}, {OWNER}'s assistant."},
            {"role": "user", "content": content}]}
        response = httpx.post(self.model_url + "/chat/completions", json=body, timeout=30, trust_env=False)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"].get("content") or "")

    def _recall(self, session: str, text: str) -> str:
        """What the sidecar recalls for this message, as the memory provider renders it into the turn."""
        value = self.api("POST", "/v1/host/context/assemble", {
            "identity": {"host_id": "hermes"},
            "context": {"session_id": SESSIONS[session], "contact_id": self.owner},
            "incoming_message": {"role": "user", "content": text}, "include_initiatives": True})
        sections = [section for section in value.get("sections") or [] if isinstance(section, dict)]
        return "\n\n".join(f"## {section.get('title') or section.get('id') or 'context'}\n{section.get('body', '')}"
                           for section in sections if str(section.get("body") or "").strip())

    @staticmethod
    def _show(texts: list[str]) -> None:
        for text in texts:
            for line in text.splitlines():
                print("  | " + line)

    def _next_turn(self) -> str:
        self.turns += 1
        return f"demo-{self.turns}"

    def _mind_now(self) -> str:
        """Now on the mind's clock (the sidecar may run shifted ahead, ``_start_sidecar``)."""
        return (now_utc() + timedelta(seconds=self.offset)).isoformat()

    def _capture(self, session: str, said: str, replied: str, *, turn: str = "") -> str:
        """``POST /v1/host/turns/sync`` as the adapter's capture sends a turn on the owner's own platform."""
        turn = turn or self._next_turn()
        value = self.api("POST", "/v1/host/turns/sync", {
            "identity": {"host_id": "hermes"},
            "context": {"session_id": SESSIONS[session], "contact_id": self.owner, "turn_id": turn,
                        "channel_id": PLATFORM,
                        "metadata": {"occurred_at": now_utc().isoformat()}},
            "user_message": {"role": "user", "content": said},
            "assistant_message": {"role": "assistant", "content": replied}, "model": "scripted-model"})
        if value.get("accepted") is False:
            raise DemoError(f"the sidecar did not capture the turn ({value.get('skipped_reason')})")
        self.said[turn] = (said, replied)
        return turn

    def _deliver_to_owner(self, *, wait: bool = True) -> list[str]:
        """The mind's queued messages to the owner, delivered as the host's body does (claimed, then reported
        sent with the delivery id the chat gave them, and noted in the chat's ledger); returns the texts that
        carry an ask, and names the others. ``wait``: until a message carrying the ask is queued."""
        deadline = time.monotonic() + (WAIT_SECONDS if wait else 0)
        while True:
            messages = self.api("GET", "/v1/mind/outbox")
            messages = messages.get("messages", []) if isinstance(messages, dict) else messages
            asks = [m for m in messages if m.get("type") in ASK_MESSAGE_TYPES]
            if asks or time.monotonic() > deadline:
                break
            time.sleep(0.5)
        if wait and not asks:
            raise DemoError("the mind queued no message carrying the ask")
        others = []
        for message in messages:
            now = now_utc().isoformat()
            self.sent += 1
            delivered = f"demo-out-{self.sent}"
            self.api("POST", f"/v1/mind/outbox/{message['id']}/sending", {"target": f"{PLATFORM}:{CHAT}", "at": now})
            self.api("POST", f"/v1/mind/outbox/{message['id']}/sent", {"result": "sent", "at": now,
                                                                         "hermes_ref": delivered})
            self.ledger.append({"at": self._mind_now(), "outbox_id": message["id"]})
            if message in asks:
                self.ask_message = delivered
            else:
                others.append(str(message.get("title") or message.get("type")))
        if others:
            print("(Also delivered to you: " + "; ".join(others) + ".)")
        return [str(message.get("text") or "") for message in asks]

    # -- what the mind did ------------------------------------------------------------------------

    def _commitment_from(self, turn: str) -> dict:
        """The commitment the sidecar's audit recorded for this turn (the scripted model answers the audit)."""
        def found() -> Optional[dict]:
            value = self.api("GET", "/v1/host/commitments?limit=100")
            rows = (value.get("commitments") or []) if isinstance(value, dict) else value
            return next((row for row in rows if (row.get("metadata") or {}).get("source_turn") == turn), None)
        return self._wait(found, "the promise to be recorded as a commitment")

    def _formed_for(self, commitment: dict) -> dict:
        """The intention the mind formed for this overdue commitment: the restarted sidecar's first tick forms
        it; if that tick has not come in a few seconds, one is run."""
        evidence = f"commitment:{commitment['id']}"
        ticked = time.monotonic() + 10

        def found() -> Optional[dict]:
            nonlocal ticked
            entries = self.api("GET", "/v1/mind/log?limit=200").get("entries") or []
            row = next((entry for entry in entries if entry.get("type") == "commitment_overdue"
                        and evidence in (entry.get("evidence") or [])), None)
            if row is None and ticked and time.monotonic() > ticked:
                self.api("POST", "/v1/mind/tick")
                ticked = 0.0
            return row
        return self._wait(found, "the mind to follow up the overdue promise")

    def _show_formed(self, row: dict, turn: str) -> None:
        evidence = [item for item in row.get("evidence") or [] if not str(item).startswith("commitment:")]
        print(f"Formed: {row['title']}")
        print(f"  decided {row.get('decision')} ({row.get('decision_reason')})")
        print(f"  reason: {row.get('concern')}" + (f" ({', '.join(evidence)})" if evidence else ""))
        print(f"  the promise, from this conversation: \"{self.said[turn][1]}\"", flush=True)

    # -- plumbing ---------------------------------------------------------------------------------

    def api(self, method: str, path: str, body: Any = None) -> Any:
        import httpx
        try:
            response = self.client.request(method, path, json=body)
        except httpx.HTTPError as error:
            raise DemoError(f"the sidecar did not answer {method} {path} ({type(error).__name__})") from None
        if not response.is_success:
            raise DemoError(f"{method} {path} answered HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def cli(self, *args: str) -> None:
        """A ``colonymind`` command against the demo's instance, printed as it prints."""
        result = subprocess.run([sys.executable, "-m", "colonymind", "--instance", str(self.instance), *args],
                                env=self.env, cwd=self.root, capture_output=True, text=True, timeout=WAIT_SECONDS,
                                stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise DemoError(f"colonymind {args[0]} failed: {(result.stderr or result.stdout).strip()[-500:]}")
        print(result.stdout.rstrip(), flush=True)

    def _wait(self, check: Callable[[], Optional[dict]], what: str) -> dict:
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            value = check()
            if value is not None:
                return value
            if time.monotonic() > deadline:
                raise DemoError(f"waited {WAIT_SECONDS:.0f} s for {what}")
            time.sleep(0.5)

    def _start_model(self) -> None:
        from colonymind.devtools import fake_model
        self.model = fake_model.serve(0, log=str(self.root / "model.log"))
        self.model_url = f"http://127.0.0.1:{self.model.server_address[1]}/v1"

    def _make_instance(self) -> None:
        """What ``colonymind init`` writes, without Hermes: identity, key, configuration, the owner contact and
        the router pointed at the scripted model."""
        from colonymind import config as configuration
        from colonymind.api.credentials import INSTANCE_CREDENTIAL
        from colonymind.init import ensure_owner_contact, write_llm_config
        self.instance.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.port = _free_port()
        identity = {"owner": {"name": OWNER, "handles": []},
                    "agent": {"name": AGENT, "values": [], "boundaries": [], "timezone": "UTC", "quiet_hours": ""}}
        data = copy.deepcopy(configuration.load_config(self.instance).data)
        data["sidecar"].update(host="127.0.0.1", port=self.port)
        data["hermes"]["home"] = str(self.home / ".hermes")
        data["router"].update(base_url=self.model_url, model="scripted-model")
        # The level most installs run at, and no quiet hours, so the run does not depend on the time of day; no pause
        # before an ask enters the conversation (mind.asks.quiet_minutes), since the demo's owner and its other
        # messages speak seconds before the ask forms, and a run waits for nobody.
        data["mind"].update(autonomy="standard", quiet_hours="", asks={"quiet_minutes": 0})
        data["credentials"][INSTANCE_CREDENTIAL]["owner"] = [PLATFORM]
        configuration.save_identity(identity, self.instance)
        configuration.write_api_key(secrets.token_urlsafe(32), self.instance)
        configuration.save_config(data, self.instance)
        self.owner, _ = ensure_owner_contact(self.instance, configuration.load_config(self.instance), identity)
        data["owner"]["contact_id"] = self.owner
        configuration.save_config(data, self.instance)
        write_llm_config(self.instance, base_url=self.model_url, model="scripted-model", api_key="local-no-key")

    def _give_the_worker_a_shell(self) -> None:
        from colonymind import config as configuration
        data = copy.deepcopy(configuration.load_config(self.instance).data)
        toolsets = list(data["mind"].get("worker_toolsets") or [])
        data["mind"]["worker_toolsets"] = toolsets + ([SHELL] if SHELL not in toolsets else [])
        configuration.save_config(data, self.instance)

    def _start_sidecar(self, *, offset: int) -> None:
        import httpx
        from colonymind.config import read_api_key
        self._stop_sidecar()
        env = dict(self.env)
        self.offset = offset
        if offset:
            env["COLONYMIND_MIND_CLOCK_OFFSET_SECONDS"] = str(offset)
        if self.log is None:
            self.log = open(self.root / "sidecar.log", "a", encoding="utf-8")
        self.sidecar = subprocess.Popen(
            [sys.executable, "-m", "colonymind", "--instance", str(self.instance), "start", "--port", str(self.port)],
            env=env, cwd=self.root, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix")
        key = read_api_key(self.instance, environ={}) or ""
        if self.client is None:
            self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=60, trust_env=False,
                                       headers={"Authorization": f"Bearer {key}"})
        deadline = time.monotonic() + WAIT_SECONDS
        while True:
            if self.sidecar.poll() is not None:
                raise DemoError(f"the sidecar exited while starting (code {self.sidecar.returncode})")
            try:
                if self.client.get("/v1/mind/state", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise DemoError(f"the sidecar did not come up in {WAIT_SECONDS:.0f} s")
            time.sleep(0.25)

    def _stop_sidecar(self) -> None:
        process, self.sidecar = self.sidecar, None
        if process is None or process.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, 30), (getattr(signal, "SIGKILL", signal.SIGTERM), 10)):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                else:
                    process.send_signal(sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)      # anything the sidecar itself started
            except (ProcessLookupError, PermissionError):
                pass

    def show_log(self) -> None:
        path = self.root / "sidecar.log"
        if path.is_file():
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("The sidecar's last log lines:\n  " + "\n  ".join(tail), flush=True)

    def close(self) -> None:
        """Stop what the demo started; safe to call at any point of the run."""
        try:
            self._stop_sidecar()
        finally:
            if self.client is not None:
                self.client.close()
                self.client = None
            if self.model is not None:
                self.model.shutdown()
                self.model.server_close()
                self.model = None
            if self.log is not None:
                self.log.close()
                self.log = None
