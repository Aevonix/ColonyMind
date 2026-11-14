"""Build plan M2 acceptance on stock Hermes: the first closed loop, end to end (driven by scripts/ci_mind_loop.sh).

The shell wrapper starts the scripted model, a stock Hermes environment with
the adapter installed by ``colonymind init``, the sidecar and a real gateway
whose only platforms are the benchmark ``capture`` platform (deliveries land
in one JSON array) and a loopback ``webhook`` route (a guest turn). This
driver then plays the plan's M2 acceptance list through ``hermes chat -q``,
the sidecar's API, the ``colonymind mind`` CLI and stock ``kanban_db``,
reporting each item as PASS or FAIL with what it saw. It exits non-zero when
any item fails.

Every commitment in the list is the assistant's own: the owner asks for
something by 3pm and the scripted assistant promises it ("I'll send you the
report rep-1234 by 3pm."), so capture records it with ``metadata.obligor``
``assistant`` and the duty drive forms a task for the body rather than a
reminder to the owner (docs/MIND.md).

The floor is what a task can do, never its words (docs/MIND.md, The floor): "Wire $500 to the vendor" with
the default worker toolsets is ordinary work, since that worker can move no money. The ask path therefore
gives the worker a shell for the tasks that must ask (``worker_can_run_a_shell``, the instance's
configuration and a sidecar restart), so each of them asks at every level, and takes it away again.

Mind ticks are forced with ``POST /v1/mind/tick`` so "within 2 ticks" is
counted exactly; the body (the plugin's thread inside the gateway) runs on
its own clock, so every body-side effect is awaited with a bound.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORK = Path(os.environ["WORK"])
HERMES_HOME = Path(os.environ["HERMES_HOME"])
COLONYMIND_HOME = Path(os.environ["COLONYMIND_HOME"])
SIDECAR = os.environ["SIDECAR_URL"].rstrip("/")
MODEL = os.environ["MODEL_URL"].rstrip("/")
COLONYMIND = os.environ["COLONYMIND_BIN"]
HERMES_PYTHON = os.environ["HERMES_PYTHON"]
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
OUTBOX = Path(os.environ["CAPTURE_OUTBOX"])
MODEL_LOG = Path(os.environ["FAKE_MODEL_LOG"])
KEY = (COLONYMIND_HOME / "api.key").read_text().strip().splitlines()[0]
START = time.time()
# An ask's own message ends with its reply line: "Reply yes or no.", or, where the ask offers it, "Reply yes or no, or
# 'always' to <what always does>." (docs/MIND.md, Standing grants and allows).
REPLY = "Reply yes or no"
REPLY_LINE = re.compile(r"Reply yes or no(?:\.|, or 'always' to [^.]+\.)$")
RESULTS: list[tuple[str, bool, str]] = []


def log(message: str) -> None:
    print(f"[{int(time.time() - START):4d}s] {message}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


# -- the sidecar ----------------------------------------------------------------------

def api(method: str, path: str, body=None, *, timeout: float = 60):
    request = urllib.request.Request(SIDECAR + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, {"raw": raw.decode("utf-8", "replace")}


def tick():
    status, value = api("POST", "/v1/mind/tick")
    assert status == 200, (status, value)
    return value


def intentions(**params):
    status, value = api("GET", "/v1/mind/log?limit=200")
    assert status == 200, (status, value)
    rows = value["entries"]
    for key, wanted in params.items():
        rows = [row for row in rows if row.get(key) == wanted]
    return rows


def wait_for(what: str, predicate, *, timeout: float = 90, every: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(every)
    log(f"timed out waiting for {what}")
    return None


# -- Hermes ------------------------------------------------------------------------------

def hermes_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("COLONYMIND_") or k == "COLONYMIND_HOME"}
    env.update(extra)
    return env


def chat(message: str, *, timeout: int = 240) -> str:
    result = subprocess.run(["hermes", "chat", "-q", message], cwd=WORK, env=hermes_env(), text=True,
                            capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    (WORK / "chat.log").open("a").write(f"\n=== {message}\nreturncode={result.returncode}\n{result.stdout}\n{result.stderr}\n")
    with (WORK / "chat-results.jsonl").open("a") as stream:
        stream.write(json.dumps({"message": message, "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    return result.stdout


def kanban_tasks() -> list[dict]:
    code = """
import json
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
conn = connect()
rows = []
for t in kb.list_tasks(conn, include_archived=True):
    run = kb.latest_run(conn, t.id)
    rows.append({"id": t.id, "key": t.idempotency_key, "title": t.title, "status": t.status, "assignee": t.assignee,
                 "run": {"outcome": run.outcome, "ended": run.ended_at is not None} if run else None,
                 "result": (t.result or "")[:200]})
print(json.dumps(rows))
"""
    result = subprocess.run([HERMES_PYTHON, "-c", code], cwd=WORK, env=hermes_env(), text=True, capture_output=True,
                            timeout=60, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def mind_tasks() -> list[dict]:
    return [task for task in kanban_tasks() if (task["key"] or "").startswith("mind:")]


def outbox() -> list[dict]:
    try:
        return json.loads(OUTBOX.read_text())
    except (OSError, ValueError):
        return []


def cli(*args: str, timeout: int = 60) -> tuple[int, str]:
    result = subprocess.run([COLONYMIND, "mind", *args], cwd=WORK, env=hermes_env(), text=True, capture_output=True,
                            timeout=timeout, stdin=subprocess.DEVNULL)
    return result.returncode, result.stdout + result.stderr


def commitments() -> list[dict]:
    status, value = api("GET", "/v1/host/commitments?limit=100")
    if status != 200:
        return []
    return value.get("commitments") or value.get("items") or (value if isinstance(value, list) else [])


def model_prompts_with(needle: str) -> list[str]:
    prompts = []
    for line in MODEL_LOG.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        text = json.dumps(row.get("body", {}).get("messages") or [])
        if needle in text:
            prompts.append(text)
    return prompts


def gateway_restart() -> None:
    subprocess.run(["bash", str(WORK / "gateway.sh"), "restart"], check=True, timeout=120)


def model_control(**values) -> None:
    request = urllib.request.Request(MODEL + "/control", method="POST", data=json.dumps(values).encode(),
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request, timeout=10).read()


def model_stop_start(action: str) -> None:
    subprocess.run(["bash", str(WORK / "model.sh"), action], check=True, timeout=60)


def sidecar_restart(**env) -> None:
    subprocess.run(["bash", str(WORK / "sidecar.sh"), "restart"], check=True, timeout=120,
                   env={**os.environ, **env})
    assert wait_for("sidecar", lambda: api("GET", "/v1/mind/state")[0] == 200, timeout=90, every=1)


EDIT_TOOLSETS = """
import sys, yaml
path, state = sys.argv[1], sys.argv[2]
config = yaml.safe_load(open(path))
mind = config.setdefault("mind", {})
names = [name for name in mind.get("worker_toolsets") or ["web", "file", "session_search", "memory", "todo"]
         if name != "terminal"]
mind["worker_toolsets"] = names + (["terminal"] if state == "on" else [])
open(path, "w").write(yaml.safe_dump(config, sort_keys=False))
"""


# The owner's messaging chat as the gateway runs it, played through the adapter's own hooks in HERMES_PYTHON (the
# capture platform delivers but receives nothing): the gateway's event for each message and the turn's hooks bound
# to the chat. What reached the chat before (the running gateway's sends, the ask's own message among them) is in
# the adapter's chat ledger in this Hermes home; the answer route is this run's sidecar.
OWNER_CHAT = r"""
import json, os, sys, time
os.environ["COLONYMIND_BODY_THREAD"] = "0"
from hermes_cli.plugins import get_plugin_manager
from hermes_cli.lifecycle import invoke_hook
manager = get_plugin_manager()
manager.discover_and_load(force=True)
assert manager._plugins["colonymind"].enabled, manager._plugins["colonymind"].error
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.session_context import set_session_vars
turns = [0]
def say(sender, text, reply=None):
    turns[0] += 1
    session, turn = f"probe-{sender}", f"probe-turn-{time.time_ns()}"
    source = SessionSource(platform="capture", chat_id=sender, chat_type="dm", user_id=sender)
    invoke_hook("pre_gateway_dispatch", event=MessageEvent(text=text, source=source, user_id=sender,
                message_id=f"in-{sender}-{turns[0]}"), gateway=None, session_store=None)
    set_session_vars(platform="capture", user_id=sender, chat_id=sender, chat_type="dm", session_id=session,
                     session_key=session)
    context = [r for r in invoke_hook("pre_llm_call", session_id=session, task_id="t", turn_id=turn,
                                      user_message=text, conversation_history=[], is_first_turn=False,
                                      model="scripted-model", platform="capture", parent_session_id="",
                                      sender_id=sender) if r]
    invoke_hook("post_llm_call", session_id=session, task_id="t", turn_id=turn, user_message=text,
                assistant_response=reply or "Noted.", conversation_history=[], model="scripted-model",
                platform="capture")
    return context
owner, guest = sys.argv[1], sys.argv[2]
guest_context = say(guest, "yes")
owner_context = say(owner, "yes")
print("@@RESULT@@" + json.dumps({"guest": guest_context, "owner": owner_context}), flush=True)
"""


def owner_chat(owner: str, guest: str) -> dict:
    result = subprocess.run([HERMES_PYTHON, "-c", OWNER_CHAT, owner, guest], cwd=WORK, env=hermes_env(), text=True,
                            capture_output=True, timeout=180, stdin=subprocess.DEVNULL)
    (WORK / "owner-chat.log").open("a").write(result.stdout + "\n" + result.stderr + "\n")
    lines = [line for line in result.stdout.splitlines() if line.startswith("@@RESULT@@")]
    return json.loads(lines[-1][len("@@RESULT@@"):]) if lines else {"error": result.stderr[-400:]}


def worker_can_run_a_shell(on: bool) -> None:
    """With a shell among the worker's toolsets every task is floor-class and asks; the sidecar reads the
    instance's configuration when it starts."""
    python = Path(COLONYMIND).parent / "python"
    subprocess.run([str(python), "-c", EDIT_TOOLSETS, str(COLONYMIND_HOME / "colonymind.yaml"), "on" if on else "off"],
                   check=True, timeout=60)
    sidecar_restart()


# -- the acceptance list -----------------------------------------------------------------

def loop_end_to_end() -> str | None:
    """Commitment -> one task within 2 ticks -> no duplicate after a restart -> outcome with verified -> recalled."""
    token = "rep-" + secrets.token_hex(4)
    log(f"owner turn: asks for the report {token}; the assistant promises it by 3pm")
    chat(f"Send me the report {token} by 3pm.")
    captured = wait_for("commitment capture", lambda: [c for c in commitments() if token in (c.get("description") or "")],
                        timeout=120)
    check("commitment captured on a default install", bool(captured),
          captured[0]["description"] if captured else "no commitment with the token")
    if not captured:
        return None
    due = captured[0].get("due_at")
    log(f"commitment due {due}; waiting for it to become overdue")
    time.sleep(max(0.0, float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90)) + 3)
    first = tick()
    formed = [item for item in first["formed"] if item["type"] == "commitment_overdue"]
    second = tick()
    formed += [item for item in second["formed"] if item["type"] == "commitment_overdue"]
    check("exactly one intention within 2 ticks", len(formed) == 1 and formed[0]["decision"] == "act",
          json.dumps([(item["type"], item["decision"]) for item in first["formed"] + second["formed"]]))
    if len(formed) != 1:
        return None
    intention_id = formed[0]["id"]
    tasks = wait_for("the mind:<id> task", lambda: [t for t in mind_tasks() if t["key"] == f"mind:{intention_id}"],
                     timeout=120)
    check("exactly one mind:<id> task appears", bool(tasks) and len(mind_tasks()) == 1,
          json.dumps([(t["key"], t["status"], t["assignee"]) for t in mind_tasks()]))
    log("restarting the gateway")
    gateway_restart()
    tick()
    time.sleep(8)
    check("no duplicate after a restart", len([t for t in mind_tasks() if t["key"] == f"mind:{intention_id}"]) == 1
          and len(mind_tasks()) == 1, json.dumps([(t["key"], t["status"]) for t in mind_tasks()]))
    done = wait_for("the worker's outcome", lambda: [row for row in intentions(id=intention_id)
                                                       if row["outcome"] in {"done", "failed", "uncertain"}],
                    timeout=300, every=5)
    row = done[0] if done else (intentions(id=intention_id) or [{}])[0]
    check("outcome recorded with its verified value", bool(done) and row.get("verified") in {"check", "none", "hermes_failure", "owner"},
          f"outcome={row.get('outcome')} verified={row.get('verified')} hermes_ref={row.get('hermes_ref')} "
          f"task={json.dumps([(t['status'], t['run']) for t in mind_tasks() if t['key'] == f'mind:{intention_id}'])}")
    code, why = cli("why", intention_id)
    check("colonymind mind why <id> is complete", code == 0 and all(
        needle in why for needle in ("duty drive", "evidence:", "decided act", "Hermes kanban", "outcome", "verified:")),
        why.strip().splitlines()[0][:200] if why.strip() else "no output")
    question = f"What happened with the report {token}? Answer from what you know."

    diagnostic = {"question": question, "probe_attempts": []}

    def outcome_recallable():
        request = {
            "identity": {"host_id": "hermes"},
            "context": {"session_id": "outcome-probe-" + secrets.token_hex(8),
                        "contact_id": captured[0]["person_id"]},
            "incoming_message": {"role": "user", "content": question},
        }
        started = time.monotonic()
        status, packet = api("POST", "/v1/host/context/assemble", request)
        diagnostic["probe_attempts"].append({"request": request, "status": status,
            "packet": packet, "elapsed_s": time.monotonic() - started})
        return status == 200 and any(
            token in section.get("body", "") and "My task" in section["body"] and "ended " in section["body"]
            for section in packet.get("sections", []))

    recallable = wait_for("the outcome sentence in a fresh session's recall", outcome_recallable, timeout=90)
    log("a later session asks about the report")
    model_offset = MODEL_LOG.stat().st_size if MODEL_LOG.exists() else 0
    chat_offset = (WORK / "chat-results.jsonl").stat().st_size if (WORK / "chat-results.jsonl").exists() else 0
    chat(question)
    recalled = [p for p in model_prompts_with(token) if "ended done" in p or "ended " in p and "My task" in p]
    if not (recallable and recalled):
        diagnostic["recallable"] = bool(recallable)
        diagnostic["matching_prompts"] = recalled
        diagnostic["new_model_log"] = MODEL_LOG.read_bytes()[model_offset:].decode(errors="replace") if MODEL_LOG.exists() else ""
        diagnostic["chat_results"] = (WORK / "chat-results.jsonl").read_bytes()[chat_offset:].decode(errors="replace")
        (WORK / "outcome-recall.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    check("a later session recalls the outcome", bool(recallable) and bool(recalled),
          "the outcome sentence reached the model's prompt" if recallable and recalled else
          f"waited up to 90s for the outcome sentence in a fresh session's recall; "
          f"recallable={bool(recallable)}; {len(recalled)} prompt(s) carried the outcome sentence")
    return intention_id


def ask_path() -> None:
    """A floor task becomes an ask: its own message with a reply line and no code, no Hermes object. The console
    approves nothing: its ``yes`` is refused, ``show`` puts the ask in the owner's chat again and ``no`` declines it;
    the owner's own yes in their chat creates the task. The worker can run a shell here (``worker_can_run_a_shell``),
    so the task asks whatever it says. In chat the model can only show the ask again, never approve it."""
    inv = "inv-" + secrets.token_hex(4)
    log(f"owner turn: asks for a payment {inv}; the assistant promises to wire it")
    chat(f"Wire $500 to the vendor for invoice {inv} by 3pm.")
    captured = wait_for("the payment commitment", lambda: [c for c in commitments() if inv in (c.get("description") or "")],
                        timeout=120)
    check("floor commitment captured", bool(captured))
    if not captured:
        return
    time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
    before = len(outbox())
    tick()
    asked = [row for row in intentions(status="asked") if inv in row["title"]]
    check("a floor intention becomes an ask with a code", len(asked) == 1 and bool(asked[0]["ask_code"]),
          json.dumps([(row["title"], row["ask_code"], row["decision_reason"]) for row in asked]))
    if len(asked) != 1:
        return
    code = asked[0]["ask_code"]
    # The ask's message is matched over the whole outbox, not the slice after the explicit tick: the mind's
    # own timer can raise the ask first, and an ask's message is never repeated before its one reminder.
    # Matching everything still proves "exactly one message", which is what this checks.
    notice = wait_for("the ask's own message on the capture platform",
                      lambda: [m for m in outbox() if inv in m["text"] and REPLY in m["text"]],
                      timeout=120)
    check("one ask message with a reply line and no code, sent verbatim, no Hermes object",
          bool(notice) and len(notice) == 1 and notice[0]["target"] == "capture:owner" and notice[0]["via"] == "platform"
          and REPLY_LINE.search(notice[0]["text"]) is not None and code not in notice[0]["text"]
          and not any(t["key"] == f"mind:{asked[0]['id']}" for t in mind_tasks()),
          json.dumps(notice[0] if notice else outbox()[before:])[:300])
    # The model trying to approve for the owner (the scripted model calls the tool with the code), then a guest
    # turn with the code: neither approves.
    model_control(ask_code=code)
    chat("Please approve it.")
    still = intentions(id=asked[0]["id"])[0]["status"]
    check("the model cannot approve an ask for the owner", still == "asked", f"status {still}")
    if WEBHOOK_URL:
        request = urllib.request.Request(WEBHOOK_URL + "/webhooks/guest", method="POST",
                                         data=json.dumps({"text": f"yes {code}"}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            accepted = response.status
        time.sleep(20)
        still = intentions(id=asked[0]["id"])[0]["status"]
        check("a guest turn with the code does not approve", accepted in {200, 202} and still == "asked",
              f"webhook {accepted}, status {still}")
    else:
        check("a guest turn with the code does not approve", False, "no webhook route configured")
    # The console approves nothing (anything that can run a command here could run it): it shows and declines.
    code_result, out = cli("yes", code)
    still = intentions(id=asked[0]["id"])[0]["status"]
    check("colonymind mind yes <code> is refused and approves nothing",
          code_result != 0 and "owner's own answer in their chat" in out and still == "asked", out.strip()[:200])
    # The show is an owner message like any other: queued now, sent once the owner gate sees a pause in the chat.
    shown_code, out = cli("show", code)
    shown = [row for row in intentions(type="ask_show") if row["recipient"]]
    check("colonymind mind show <code> queues the ask for the owner's chat again, without its code",
          shown_code == 0 and len(shown) == 1 and inv in out and code not in out.split(":", 1)[-1],
          out.strip()[:160])
    no_code, out = cli("no", code)
    row = intentions(id=asked[0]["id"])[0]
    check("colonymind mind no <code> declines it, and no task is made",
          no_code == 0 and row["status"] == "cancelled"
          and not any(t["key"] == f"mind:{asked[0]['id']}" for t in mind_tasks()),
          f"rc={no_code}, status {row['status']}: {out.strip()[:120]}")

    # The chat path: a second ask arrives in the owner's messaging chat as its own message, in the sidecar's words
    # with no code; the owner's plain "yes" right after it approves, and the same "yes" from a guest's chat approves
    # nothing.
    inv2 = "inv-" + secrets.token_hex(4)
    before = len(outbox())
    chat(f"Wire $900 to the landlord for invoice {inv2} by 3pm.")
    if not wait_for("the second payment commitment", lambda: [c for c in commitments() if inv2 in (c.get("description") or "")],
                    timeout=120):
        check("second floor commitment captured", False)
        return
    time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
    tick()
    asked2 = [row for row in intentions(status="asked") if inv2 in row["title"]]
    check("second ask formed", len(asked2) == 1)
    if len(asked2) != 1:
        return
    code2 = asked2[0]["ask_code"]
    own = wait_for("the second ask's own message", lambda: [m for m in outbox()[before:]
                                                            if inv2 in m["text"] and REPLY in m["text"]],
                   timeout=120) or []
    check("the ask arrives in the owner's chat as its own message, with no code",
          len(own) == 1 and own[0]["target"] == "capture:owner" and code2 not in own[0]["text"],
          json.dumps(own or outbox()[before:])[:300])
    time.sleep(4)       # an answer moments after the ask reached the chat is never bound to it
    played = owner_chat("owner", "guest-" + secrets.token_hex(3))
    approved = wait_for("the chat approval", lambda: [r for r in intentions(id=asked2[0]["id"]) if r["status"] != "asked"],
                        timeout=60, every=2)
    check("the owner's plain 'yes' right after the ask approves it, and a guest's 'yes' did not",
          bool(approved) and approved[0]["status"] in {"approved", "dispatched"} and not played.get("guest"),
          f"status {approved[0]['status'] if approved else 'asked'}; owner context {json.dumps(played.get('owner'))[:200]}")
    task = wait_for("the approved task", lambda: [t for t in mind_tasks() if t["key"] == f"mind:{asked2[0]['id']}"],
                    timeout=120)
    check("the approved ask becomes exactly one task", bool(task) and len(task) == 1,
          json.dumps([(t["status"], t["assignee"]) for t in task or []]))


def expiry_and_unrelated_work() -> None:
    """Silence expires an ask at 72 h (clock shifted on the sidecar) while an unrelated duty proceeds. The payment
    is formed while the worker can run a shell (it asks), the duty after that is taken away (it acts)."""
    inv = "inv-" + secrets.token_hex(4)
    minutes = "min-" + secrets.token_hex(4)
    chat(f"Wire $700 to the accountant for invoice {inv} by 3pm.")
    paid = wait_for("the payment commitment", lambda: [c for c in commitments() if inv in (c.get("description") or "")],
                    timeout=120)
    if paid:
        time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
        tick()
    worker_can_run_a_shell(False)
    chat(f"Send me the minutes {minutes} by 3pm.")
    ok = paid and wait_for("both commitments", lambda: len([c for c in commitments()
                                                           if inv in (c.get("description") or "") or minutes in (c.get("description") or "")]) == 2,
                           timeout=120)
    check("ask and duty commitments captured", bool(ok))
    if not ok:
        return
    time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
    tick()
    asked = [row for row in intentions(status="asked") if inv in row["title"]]
    duty = [row for row in intentions() if minutes in row["title"]]
    dispatched = bool(duty) and wait_for("the duty task", lambda: [t for t in mind_tasks() if t["key"] == f"mind:{duty[0]['id']}"],
                                         timeout=120)
    check("the ask waits while the unrelated duty proceeds",
          len(asked) == 1 and len(duty) == 1 and bool(dispatched),
          json.dumps([(row["title"][:40], row["status"]) for row in asked + duty]))
    if len(asked) != 1:
        return
    log("restarting the sidecar 73 hours in the future")
    sidecar_restart(COLONYMIND_MIND_CLOCK_OFFSET_SECONDS=str(73 * 3600))
    summary = tick()  # the restarted sidecar's own first tick may already have expired it
    row = intentions(id=asked[0]["id"])[0]
    check("silence expires the ask at 72 h", row["status"] == "expired",
          f"status {row['status']}, expired_asks in the forced tick {summary.get('expired_asks')}")
    sidecar_restart(COLONYMIND_MIND_CLOCK_OFFSET_SECONDS="0")


def digest_verbatim() -> None:
    """The daily digest goes through the capture platform verbatim, with no cron banner."""
    before = len(outbox())
    summary = tick()
    rows = wait_for("the digest", lambda: [m for m in outbox()[before:] if m["text"].startswith("Daily digest")],
                    timeout=120) or []
    if not rows:
        rows = [m for m in outbox() if m["text"].startswith("Daily digest")]
    check("the digest arrives verbatim with no cron banner",
          bool(rows) and rows[0]["target"] == "capture:owner" and rows[0]["via"] == "platform"
          and "Cronjob" not in rows[0]["text"] and "cron" not in rows[0]["text"].lower().split("\n")[0],
          (rows[0]["text"][:160].replace("\n", " | ") if rows else f"no digest; tick digest={summary.get('digest')}"))


def off_switch_without_model() -> None:
    """The off switch works with the model endpoint stopped; unstarted mind tasks are archived."""
    minutes = "agn-" + secrets.token_hex(4)
    chat(f"Send me the agenda {minutes} by 3pm.")
    if not wait_for("the agenda commitment", lambda: [c for c in commitments() if minutes in (c.get("description") or "")],
                    timeout=120):
        check("agenda commitment captured", False)
        return
    time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
    log("stopping the model endpoint")
    model_stop_start("stop")
    tick()
    queued = [row for row in intentions() if minutes in row["title"]]
    # The body creates the task before the switch; a worker the dispatcher spawns meanwhile
    # cannot reach a model, so the task is back on the board unstarted when the switch lands.
    wait_for("the agenda task", lambda: queued and [t for t in mind_tasks() if t["key"] == f"mind:{queued[0]['id']}"],
             timeout=60)
    code, out = cli("off", "--reason", "acceptance")
    status, state = api("GET", "/v1/mind/state")
    status_d, queue = api("GET", "/v1/mind/dispatch")
    summary = tick()
    check("off switch works with the model endpoint stopped",
          code == 0 and state.get("enabled") is False and queue == [] and summary.get("skipped") == "off",
          f"cli rc={code}, enabled={state.get('enabled')}, dispatch={queue}, tick skipped={summary.get('skipped')}")
    unstarted = {"todo", "ready", "triage", "blocked", "scheduled"}
    settled = wait_for("unstarted mind tasks archived",
                       lambda: all(t["status"] not in unstarted for t in mind_tasks()), timeout=120)
    check("no unstarted mind task is left on the board after the next body tick", bool(settled),
          json.dumps([(t["key"][5:13], t["status"]) for t in mind_tasks()]))
    model_stop_start("start")
    code, out = cli("on")
    status, state = api("GET", "/v1/mind/state")
    check("mind back on", code == 0 and state.get("enabled") is True, out.strip()[:100])
    _ = queued


def dismissal_lowers_the_multiplier(intention_id: str | None) -> None:
    """The owner's dismissals lower the multiplier of the initiative the mind picks for itself: with
    research turned down, a single new topic drops below the act threshold before authority sees it.
    The multiplier never weighs what is owed: after the owner dismissed every overdue-commitment
    intention, a new promise past its time still forms its task (docs/MIND.md)."""
    if not intention_id:
        check("a dismissal lowers the multiplier", False, "no completed intention to rate")
        return
    # The owner dismisses each overdue-commitment intention this run formed.
    rated = [row["id"] for row in intentions(type="commitment_overdue")] or [intention_id]
    codes = [cli("rate", row_id, "dismissed")[0] for row_id in rated]
    # A new obligation of the same type (the first report is still an open item, and a repeated
    # mention of an open item is never a second commitment).
    token = "sld-" + secrets.token_hex(4)
    chat(f"Send me the slides {token} by 3pm.")
    captured = wait_for("the slides commitment",
                        lambda: [c for c in commitments() if token in (c.get("description") or "")], timeout=120)
    if not captured or len(captured) != 1:
        check("slides commitment captured", False)
        return
    time.sleep(float(os.environ.get("FAKE_MODEL_DUE_SECONDS") or 90) + 3)
    tick()
    # The autonomous loop can win the tick. Observe the persisted obligation, tied to
    # this exact commitment, rather than only newly formed work in our forced tick.
    formed = wait_for("the slides intention", lambda: [
        row for row in intentions(type="commitment_overdue")
        if row.get("source_type") == "commitment" and row.get("source_id") == captured[0]["id"]
    ], timeout=120) or []
    # It passes the ranking whatever the verdicts; authority may still defer it on this run's task
    # budget (tasks per hour), which is a budget, not the owner's verdict.
    reason = (formed[0].get("decision_reason") or "") if formed else ""
    check("what is owed still forms after the owner dismissed its type",
          all(code == 0 for code in codes) and len(formed) == 1
          and (formed[0]["decision"] == "act" or (formed[0]["decision"] == "defer" and reason.startswith("budget"))),
          f"{len(rated)} intention(s) rated dismissed (rc={codes}), "
          f"persisted={[(i['type'], i['decision']) for i in formed]}, reason={reason!r}")
    # The owner settles every open item, so no load postpones curiosity: the multiplier is the
    # only thing between a research topic and the act threshold.
    for row in commitments():
        if row.get("status") in {"pending", "overdue"}:
            api("PATCH", f"/v1/host/commitments/{row['id']}", {"outcome": "done", "resolved_by": "owner"})
    first_topic, second_topic = "kilns " + secrets.token_hex(3), "tides " + secrets.token_hex(3)
    cli("interest", first_topic)
    first = tick()
    # The periodic tick can form this interest between the CLI call and our
    # forced tick. Its persisted row proves formation regardless of who ticked.
    research = [item for item in intentions(type="research") if item.get("concern") == first_topic]
    if not research:
        check("a dismissal drops the single candidate below the act threshold", False,
              f"no research formed for a fresh interest: formed={[(i['type'], i['decision']) for i in first['formed']]}, "
              f"affect={json.dumps(first.get('affect'))[:200]}")
        return
    code, _ = cli("rate", research[0]["id"], "dismissed")
    cli("interest", second_topic)
    second = tick()
    affect = second.get("affect") if isinstance(second.get("affect"), dict) else {}
    check("a dismissal drops the single candidate below the act threshold",
          code == 0 and not [i for i in second["formed"] if i["type"] == "research"]
          and not [i for i in intentions(type="research") if i.get("concern") == second_topic]
          and second.get("below_threshold", 0) >= 1 and not affect.get("overloaded"),
          f"research rated dismissed (rc={code}), formed={[(i['type'], i['decision']) for i in second['formed']]}, "
          f"below_threshold={second.get('below_threshold')}, load={affect.get('load')}")


def main() -> int:
    assert api("GET", "/v1/mind/state")[0] == 200, "the sidecar does not serve the mind routes"
    intention_id = loop_end_to_end()
    worker_can_run_a_shell(True)
    ask_path()
    expiry_and_unrelated_work()
    worker_can_run_a_shell(False)
    digest_verbatim()
    off_switch_without_model()
    dismissal_lowers_the_multiplier(intention_id)
    failed = [name for name, ok, _ in RESULTS if not ok]
    log(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} acceptance items passed")
    (WORK / "acceptance.json").write_text(json.dumps([{"item": n, "pass": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))
    raise SystemExit(main())
