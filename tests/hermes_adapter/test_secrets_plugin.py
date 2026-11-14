"""The Hermes half of the sealed store, through stock Hermes seams (docs/SECRETS.md).

A gateway message carrying a secret is sealed before Hermes sees it (``pre_gateway_dispatch``), and so are the
words that come with it but are not the sender's (a quoted message, a channel's backfill); without the sidecar the
value is withheld. In the owner's own session a reference is filled into a call only when it was sealed from their
own words of their latest message; any other becomes a one-tap ask about that exact call, put to them directly, and
the call waits. A terminal command or Python code never carries a value in its text: it reads
``$COLONYMIND_SECRET_<id>`` (or the private file ``$COLONYMIND_SECRET_<id>_FILE`` names), set for that one call. In a
mind run a worker's message to anyone but the owner hides every value and reference. ``colonymind_secret`` is the
owner's alone and a reveal goes to their chat (or a private file), not to the model. The body fills a
reference only where the sidecar allows it, just before the send, and holds the send while it cannot; the turn
outbox keeps no delivered words and keeps a rejected turn's.

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from conftest import MIND_PRELUDE, OWNER, probe, worker_env

KEY = "sk-" + "proj" + "Qz7Wx3Er9Ty1Ui5Op2As8Df4"
WIFI = "Hunter" + "22!" + "wifi"
HIDDEN = "(a private value, withheld)"
WITHHELD = "[secret withheld: the store was unavailable; send it again]"

CODE = '''
from types import SimpleNamespace
from tools.registry import registry
from hermes_cli.middleware import apply_tool_request_middleware
from hermes_cli.plugins import get_pre_tool_call_directive
def dispatch(text, sender, platform="telegram"):
    source = SimpleNamespace(platform=SimpleNamespace(value=platform), user_id=sender, chat_id="c")
    return [r for r in invoke_hook("pre_gateway_dispatch", event=SimpleNamespace(text=text, source=source),
                                   gateway=None, session_store=None) if r]
def turn(session, sender, message, platform="telegram", chat=None):
    # chat: the chat the gateway binds for the turn (the one their message came in); None leaves the binding alone
    from gateway.session_context import set_session_vars, clear_session_vars
    tokens = set_session_vars(platform=platform, user_id=sender, chat_id=chat, chat_type="dm") if chat else None
    invoke_hook("pre_llm_call", session_id=session, task_id="t", turn_id="turn", user_message=message,
                conversation_history=[], is_first_turn=True, model="m", platform=platform,
                parent_session_id="", sender_id=sender)
    if tokens is not None:
        clear_session_vars(tokens)
    return session
def request(tool, args, session):
    ids = dict(session_id=session, task_id="t", tool_call_id="c", turn_id="turn", api_request_id="r")
    changed = apply_tool_request_middleware(tool, args, **ids).payload
    action, message = get_pre_tool_call_directive(tool, changed, **ids)
    return {"args": changed, "action": action, "message": message}
def call(tool, args, session):
    return json.loads(registry.dispatch(tool, args, task_id="t", session_id=session))
'''


@pytest.fixture
def vault(sidecar):
    """The sidecar's ``/v1/secrets`` routes as the plugin relies on them, over the plugin's own detector copy."""
    from colonymind_hermes.secrets import find_secrets
    state = SimpleNamespace(values={}, scopes={}, labels={}, down=False, asks=[], released=set(), actions=[],
                            unavailable=False, permitted=set(), dropped=set())

    def ref_for(value, scope, label):
        for ref, known in state.values.items():
            if known == value and state.scopes[ref] == scope:
                return ref
        ref = "S-" + "ABCDEFGHJKLMNPQR"[len(state.values) % 16] * 8
        state.values[ref], state.scopes[ref], state.labels[ref] = value, scope, label
        return ref

    def secrets(method, path, query, body):
        if state.down:
            return 503, {"detail": "unavailable"}
        if path == "/v1/secrets/seal":
            scope = OWNER if body.get("owner") else body.get("contact_id")
            parts, released, cursor = [], [], 0
            for hit in find_secrets(body["text"]):
                parts.append(body["text"][cursor:hit.start])
                if hit.value in state.released:
                    start = sum(map(len, parts))
                    released.append([start, start + len(hit.value)])
                    parts.append(hit.value)
                else:
                    parts.append("[secret %s: %s]" % (ref_for(hit.value, scope, hit.label), hit.label))
                cursor = hit.end
            text = "".join(parts) + body["text"][cursor:]
            return 200, {"text": text, "sealed": len(find_secrets(body["text"])), "withheld": 0,
                         "released": released}
        if path == "/v1/secrets/resolve":
            if len(body["refs"]) > 32:                        # the route's own limit
                return 422, {"detail": "at most 32 references"}
            if state.unavailable:
                return 200, {"values": {}, "refused": {}, "asks": {},
                             "unavailable": {ref: "the key is locked" for ref in body["refs"]}}
            values, refused, asks = {}, {}, {}
            if body.get("action") is not None:
                state.actions.append(body["action"])
            for ref in body["refs"]:
                if ref in state.dropped:
                    continue
                if state.scopes.get(ref) != OWNER:
                    refused[ref] = "another person's private value is never used"
                elif body["purpose"] == "send":
                    values[ref] = state.values[ref]
                elif body.get("lane") != "owner":
                    refused[ref] = "only the owner's own conversation uses a private value"
                elif (body["purpose"] == "fill" and ref not in (body.get("own_refs") or [])
                      and ref not in state.permitted):
                    asks[ref] = {"code": "K7F2", "text": f"Use {state.labels[ref]} in {body['action']['tool']}, "
                                 f"to: {body.get('destination')}? yes K7F2"}
                    state.asks.append(ref)
                else:
                    values[ref] = state.values[ref]
            return 200, {"values": values, "refused": refused, "asks": asks, "unavailable": {}}
        if path == "/v1/secrets" and method == "GET":
            return 200, {"entries": [{"ref": ref, "label": state.labels[ref], "scope_contact": state.scopes[ref],
                                      "state": "sealed", "created_at": "2030-01-01T00:00:00+00:00", "uses": 0}
                                     for ref in state.values]}
        return 404, {"detail": "not found"}

    original = sidecar.dispatch

    def dispatch(method, path, query, body):
        if path.startswith("/v1/secrets"):
            return secrets(method, path, query, body)
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    return state


def test_a_gateway_message_is_sealed_before_hermes_sees_it(home, sidecar, vault):
    result = probe(CODE + f'''
emit(owner=dispatch("my key is {KEY}", "1001"), friend=dispatch("the wifi password is {WIFI}", "2003"),
     stranger=dispatch("my key is {KEY}", "9999"), plain=dispatch("see you at 1234567890123", "1001"),
     mind=dispatch("/mind off", "2002"))
''', home)
    for name in ("owner", "friend", "stranger"):
        rewrite, = result[name]
        assert rewrite["action"] == "rewrite" and "[secret S-" in rewrite["text"], name
        assert KEY not in rewrite["text"] and WIFI not in rewrite["text"]
    assert result["plain"] == []                                   # nothing detected: no sidecar call either
    assert [r["action"] for r in result["mind"]] == ["skip"]       # the /mind gate still runs first
    seals = [r["json"] for r in sidecar.calls("/v1/secrets/seal")]
    assert len(seals) == 3
    assert seals[0]["owner"] is True
    assert seals[1] == {"text": f"the wifi password is {WIFI}", "contact_id": "p-03"}
    assert seals[2]["contact_id"] == "handle:telegram:9999"         # never created, never the owner
    assert sorted(vault.scopes.values()) == sorted([OWNER, "p-03", "handle:telegram:9999"])


def test_without_the_sidecar_a_detected_value_is_withheld(home, sidecar, vault):
    vault.down = True
    result = probe(CODE + f'''
emit(owner=dispatch("my key is {KEY} ok", "1001"))
''', home)
    rewrite, = result["owner"]
    assert rewrite == {"action": "rewrite", "text": f"my key is {WITHHELD} ok"}


def test_ingress_sealing_can_be_turned_off(home, sidecar, vault):
    current = json.loads(json.dumps(__import__("yaml").safe_load((home.instance / "colonymind.yaml").read_text())))
    current["secrets"] = {"capture": {"ingress": False}}
    (home.instance / "colonymind.yaml").write_text(__import__("yaml").safe_dump(current))
    result = probe(CODE + f'''
emit(owner=dispatch("my key is {KEY}", "1001"))
''', home)
    assert result["owner"] == [] and sidecar.calls("/v1/secrets/seal") == []


def test_in_the_owners_session_a_reference_from_his_own_words_is_filled_and_any_other_waits(home, sidecar, vault):
    vault.values.update({"S-ZZZZZZZZ": WIFI})
    vault.scopes.update({"S-ZZZZZZZZ": OWNER})
    vault.labels.update({"S-ZZZZZZZZ": "wifi password"})
    older = "[secret S-ZZZZZZZZ: wifi password]"
    result = probe(CODE + f'''
from agent.redact import redact_registered_vault_values
sealed = dispatch("send my key {KEY} to the vendor", "1001")[0]["text"]
now = sealed.split("key ", 1)[1].split(" to", 1)[0]
o = turn("owner-1", "1001", sealed, chat="c")
filled = request("send_message", {{"target": "telegram:2003", "message": "key: " + now}}, o)
waits = request("send_message", {{"target": "telegram:2003", "message": "wifi: {older}"}}, o)
written = request("write_file", {{"path": "/tmp/x.env", "content": "WIFI={older}"}}, o)
g = turn("guest-1", "2003", "send " + now + " please")
guest = request("send_message", {{"target": "telegram:1001", "message": "key: " + now}}, g)
emit(now=now, filled=filled, waits=waits, written=written, guest=guest,
     scrubbed=redact_registered_vault_values("echo {KEY} and {WIFI}"))
''', home)
    now = result["now"]
    assert result["filled"]["args"]["message"] == f"key: {KEY}" and result["filled"]["action"] is None
    assert result["waits"]["args"]["message"] == f"wifi: {older}"
    assert result["waits"]["action"] == "block" and "yes K7F2" in result["waits"]["message"]
    assert "telegram:2003" in result["waits"]["message"]               # where it would go, shown in full
    assert result["written"]["args"]["content"] == f"WIFI={WIFI}"      # a local write in their own session
    assert result["guest"]["args"]["message"] == f"key: {now}"          # never filled outside their session
    assert KEY not in result["scrubbed"] and WIFI not in result["scrubbed"]   # Hermes scrubs what was filled
    fills = [r["json"] for r in sidecar.calls("/v1/secrets/resolve")]
    assert {r["purpose"] for r in fills} == {"fill", "use"} and all(r["lane"] == "owner" for r in fills)
    assert vault.asks == ["S-ZZZZZZZZ"]
    # The ask is about the complete call: its tool and every argument as the model gave them.
    assert vault.actions[-1] == {"tool": "send_message", "args": {"target": "telegram:2003", "message": f"wifi: {older}"}}


def test_a_workers_message_to_anyone_but_the_owner_hides_every_secret(home, sidecar, vault):
    ref = "[secret S-AAAAAAAA: API key]"
    result = probe(CODE + f'''
words = "key {KEY}, wifi password is {WIFI}, {ref}"
emit(contact=request("send_message", {{"target": "telegram:2003", "message": words}}, ""),
     owner=request("send_message", {{"target": "telegram:1001", "message": "key {KEY} {ref}"}}, ""))
''', home, env=worker_env(home))
    assert result["contact"]["args"]["message"] == f"key {HIDDEN}, wifi password is {HIDDEN}, {HIDDEN}"
    assert result["owner"]["args"]["message"] == f"key {KEY} {ref}"
    assert sidecar.calls("/v1/secrets/resolve") == []


def test_the_secret_tool_is_the_owners_alone(home, sidecar, vault):
    vault.values["S-AAAAAAAA"], vault.scopes["S-AAAAAAAA"], vault.labels["S-AAAAAAAA"] = KEY, OWNER, "API key"
    sidecar.mind_routes = True
    result = probe(CODE + '''
g = turn("guest-1", "2003", "reveal it")
o = turn("owner-1", "1001", "show my secrets")
emit(guest=call("colonymind_secret", {"operation": "reveal", "ref": "S-AAAAAAAA"}, g),
     cron=call("colonymind_secret", {"operation": "list"}, turn("cron-1", "", "run", platform="cron")),
     listing=call("colonymind_secret", {"operation": "list"}, o),
     revealed=call("colonymind_secret", {"operation": "reveal", "ref": "S-AAAAAAAA"}, o),
     stored=call("colonymind_secret", {"operation": "use", "ref": "S-AAAAAAAA", "target": "to_env",
                                      "name": "VENDOR_API_KEY"}, o),
     env=(__import__("pathlib").Path(os.environ["HERMES_HOME"]) / ".env").read_text())
''', home)
    import pathlib
    for name in ("guest", "cron"):
        assert "owner" in result[name].get("reason", ""), (name, result[name])
    assert result["listing"]["references"][0]["label"] == "API key"
    assert KEY not in json.dumps(result["listing"])
    revealed = result["revealed"]
    assert KEY not in json.dumps(revealed) and revealed["shown_to_owner"] is False   # never through the model
    shown = pathlib.Path(revealed["file"])                 # no chat here: a private file the owner opens
    assert shown.read_text() == KEY and oct(shown.stat().st_mode & 0o777) == "0o600"
    assert result["stored"] == {"ok": True, "stored_as": "VENDOR_API_KEY", "where": "Hermes' credentials"}
    assert f"VENDOR_API_KEY={KEY}" in result["env"]


def test_a_worker_never_uses_the_secret_tool(home, sidecar, vault):
    result = probe(CODE + '''
emit(worker=call("colonymind_secret", {"operation": "list"}, ""))
''', home, env=worker_env(home))
    assert "owner" in result["worker"].get("reason", "")


def test_the_body_fills_a_reference_only_where_the_sidecar_allows_it(home, sidecar, vault):
    vault.values.update({"S-AAAAAAAA": KEY, "S-BBBBBBBB": WIFI})
    vault.scopes.update({"S-AAAAAAAA": OWNER, "S-BBBBBBBB": "p-03"})
    sidecar.mind_routes = True
    mine, theirs = "[secret S-AAAAAAAA: API key]", "[secret S-BBBBBBBB: wifi password]"
    sidecar.mind.outbox["m-01"] = {"id": "m-01", "kind": "notice", "text": f"Your key: {mine}",
                                   "target": "telegram:1001", "recipient_is_owner": True}
    sidecar.mind.outbox["m-02"] = {"id": "m-02", "kind": "message", "text": f"Their wifi: {theirs}",
                                   "recipient": {"platform": "telegram", "chat_id": "2003", "contact_id": "p-03"}}
    result = probe('''
_smt.send_message_tool = fake_send
tick()
emit(sends=SENDS, ledger=body.ledger.send("m-01"))
''', home, prelude=MIND_PRELUDE)
    assert [s["message"] for s in result["sends"]] == [f"Your key: {KEY}", f"Their wifi: {HIDDEN}"]
    assert [r["json"]["purpose"] for r in sidecar.calls("/v1/secrets/resolve")] == ["send", "send"]
    assert KEY not in json.dumps(result["ledger"])


def test_the_turn_outbox_keeps_no_delivered_words(home, sidecar):
    result = probe('''
from colonymind_hermes.capture import TurnOutbox
import tempfile, pathlib
path = pathlib.Path(tempfile.mkdtemp()) / "outbox.sqlite3"
box = TurnOutbox(path)
box.enqueue("turn-1", {"user_message": {"role": "user", "content": "words that stay private"}})
box.enqueue("turn-2", {"user_message": {"role": "user", "content": "still waiting"}})
delivered = box.drain(lambda payload: "private" in json.dumps(payload), limit=1)
emit(delivered=delivered, rows=box.rows(), path=str(path))
''', home)
    assert result["delivered"] == 1
    states = {row["turn_id"]: (row["state"], row["payload"]) for row in result["rows"]}
    assert states["turn-1"] == ("delivered", {})
    assert states["turn-2"][0] == "pending" and states["turn-2"][1]["user_message"]["content"] == "still waiting"
    with sqlite3.connect(result["path"]) as conn:
        assert "words that stay private" not in "\n".join(conn.iterdump())


# -- round 1: where a value goes, whose words stand behind it, and what waits -------------------------------------

# A password whose bytes are shell and Python syntax: it must arrive as those bytes and never run.
PW = 'x"; touch PWNED; echo "$(touch PWNED2)`touch PWNED3`Z9!'
EVENT = '''
def event(text, sender, **extra):
    source = SimpleNamespace(platform=SimpleNamespace(value="telegram"), user_id=sender, chat_id="c")
    return SimpleNamespace(text=text, source=source, **extra)
'''


def test_words_the_sender_did_not_write_are_sealed_in_place_and_never_stand_behind_a_fill(home, sidecar, vault):
    """A quoted message is sealed in its author's scope, a channel's backfill in nobody's; a reference that came in
    a quote (even of the owner's own message) is not their words now: using it asks."""
    result = probe(CODE + EVENT + f'''
quoted = event("use that", "1001", reply_to_text="my key is {KEY}", reply_to_message_id="m1", reply_to_author_id="2003")
backfill = event("what do you think", "1001", channel_context="[Friend] the wifi password is {WIFI}")
mine = event("send it to the vendor", "1001", reply_to_text="my key is {KEY}", reply_to_message_id="m2",
             reply_to_author_id="1001")
answers = [[r for r in invoke_hook("pre_gateway_dispatch", event=e, gateway=None, session_store=None) if r]
           for e in (quoted, backfill, mine)]
ref = mine.reply_to_text.split("key is ", 1)[1]
o = turn("owner-q", "1001", '[Replying to: "' + mine.reply_to_text + '"]\\n\\nsend it to the vendor')
use = request("send_message", {{"target": "telegram:2003", "message": "key: " + ref}}, o)
emit(answers=answers, quoted=quoted.reply_to_text, backfill=backfill.channel_context, ref=ref, use=use)
''', home)
    assert result["answers"] == [[], [], []]          # nothing in the sender's own text; the rest sealed in place
    assert KEY not in result["quoted"] and "[secret S-" in result["quoted"]
    assert WIFI not in result["backfill"] and "[secret S-" in result["backfill"]
    assert sorted(set(vault.scopes.values())) == sorted({OWNER, "p-03", "handle:telegram:channel:c"})
    assert result["use"]["action"] == "block" and "yes K7F2" in result["use"]["message"]
    fill, = [r["json"] for r in sidecar.calls("/v1/secrets/resolve")]
    assert fill["own_refs"] == [] and fill["purpose"] == "fill"


def test_a_command_or_code_gets_a_value_only_from_its_environment(home, sidecar, vault):
    """Nothing is ever put into a command's or code's text: the plugin sets ``$COLONYMIND_SECRET_<id>`` (the value)
    and ``$COLONYMIND_SECRET_<id>_FILE`` (a private file holding it) for that one call, and the program reads them
    as data, its exact bytes. A reference written into the command or the code is refused, naming the variable."""
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = PW + "\n", OWNER, "password"
    vault.permitted.add("S-NNNNNNNN")                    # the owner's yes to that exact call
    result = probe(CODE + EVENT + f'''
import pathlib, subprocess, tempfile
from colonymind_hermes.secrets import REFERENCE
sealed = invoke_hook("pre_gateway_dispatch", event=event("the db password is '" + {PW!r} + "'", "1001"),
                     gateway=None, session_store=None)[-1]["text"]
ref = REFERENCE.search(sealed).group(0)
name = "COLONYMIND_SECRET_" + ref[len("[secret S-"):len("[secret S-") + 8]
o = turn("owner-t", "1001", sealed, chat="c")
calls = {{
    "shell": request("terminal", {{"command": 'printf %s "$' + name + '" > out1; cat "$' + name + '_FILE" > out2'}}, o),
    "nested": request("terminal", {{"command": "python3 -c 'import os, sys; sys.stdout.write(os.environ[\\"" + name
                                    + "\\"])' > out3"}}, o),
    "newline": request("terminal", {{"command": 'printf %s "$COLONYMIND_SECRET_NNNNNNNN" > out4'}}, o),
    "code": request("execute_code", {{"code": "import os\\nv = os.environ[" + repr(name) + "]\\nprint(len(v), v == "
                                     + repr({PW!r}) + ", open(os.environ[" + repr(name + "_FILE") + "]).read() == v)"}}, o),
    "stdin": request("process_manage", {{"action": "submit", "session_id": "proc_1", "data": ref}}, o),
    "in_command": request("terminal", {{"command": "login --password '" + ref + "'"}}, o),
    "in_code": request("execute_code", {{"code": 'key = "' + ref + '"'}}, o),
    "mixed": request("process_manage", {{"action": "submit", "session_id": "proc_1", "data": "pw " + ref}}, o),
    "target": request("send_message", {{"target": ref, "message": "hi"}}, o),
}}
work = pathlib.Path(tempfile.mkdtemp())
for key in ("shell", "nested", "newline"):
    subprocess.run(["bash", "-c", calls[key]["args"]["command"]], cwd=work, check=True)
ran = subprocess.run([sys.executable, "-c", calls["code"]["args"]["code"] + "\\nimport os\\nprint(sorted(k for k in "
                      "os.environ if k.startswith('COLONYMIND_SECRET')))"], capture_output=True, text=True, check=True)
bindings = pathlib.Path({str(home.instance)!r}) / "private" / "secret-bindings"
invoke_hook("post_tool_call", tool_name="terminal", args={{}}, result="", task_id="t", session_id=o,
            tool_call_id="c", turn_id="turn", api_request_id="r", duration_ms=0, status="ok", error_type=None,
            error_message=None, middleware_trace=[])
emit(calls=calls, out=[(work / f"out{{n}}").read_text() for n in (1, 2, 3, 4)], ran=ran.stdout.splitlines(),
     planted=sorted(p.name for p in work.iterdir() if p.name.startswith("PWNED")),
     left=sorted(p.name for p in bindings.iterdir()), mode=oct(bindings.stat().st_mode & 0o777))
''', home)
    calls = result["calls"]
    assert result["out"] == [PW, PW, PW, PW + "\n"]                     # the exact bytes, a last newline included
    assert result["ran"] == [f"{len(PW)} True True", "[]"]              # and none left in the interpreter after
    assert result["planted"] == []                                      # nothing in the value ever ran
    for name in ("shell", "nested", "newline", "code"):
        assert PW not in json.dumps(calls[name]["args"]) and calls[name]["action"] is None, name
    assert calls["stdin"]["args"]["data"] == PW                         # a prompt's whole answer is data
    for name in ("in_command", "in_code"):
        assert calls[name]["action"] == "block" and "$COLONYMIND_SECRET_" in calls[name]["message"], name
        assert PW not in json.dumps(calls[name]), name
    for name in ("mixed", "target"):
        assert calls[name]["action"] == "block" and "cannot be placed" in calls[name]["message"], name
    assert result["left"] == [] and result["mode"] == "0o700"


def test_through_hermes_own_terminal_and_kernel_a_value_lives_for_its_call_only(home, sidecar, vault):
    """Hermes runs a command in its session shell (a snapshot of the environment kept between commands) and code
    in a kernel kept for the session: the value reaches the call exactly, and neither the next command, the
    snapshot, the next cell nor the kernel's names keep any of it; what the code itself keeps stays."""
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = PW + "\n", OWNER, "password"
    vault.permitted.add("S-NNNNNNNN")
    result = probe(CODE + '''
import glob, pathlib, tempfile
import tools.terminal_tool, tools.code_execution_tool  # noqa: F401 (they register the tools)
o = turn("owner-e", "1001", "use it")
work = pathlib.Path(tempfile.mkdtemp())
shell = request("terminal", {"command": "cd " + str(work) + ' && printf %s "$COLONYMIND_SECRET_NNNNNNNN" > out'
                                        ' && export LEAK="$COLONYMIND_SECRET_NNNNNNNN"'}, o)
first = call("terminal", shell["args"], o)
second = call("terminal", {"command": 'printf "%s|%s" "${COLONYMIND_SECRET_NNNNNNNN:-unset}" "${LEAK:-unset}"'}, o)
cell = request("execute_code", {"code": "import os\\nv = os.environ['COLONYMIND_SECRET_NNNNNNNN']\\nprint(len(v))"
                                        "\\nkept = 1"}, o)
ran = call("execute_code", cell["args"], o)
after = call("execute_code", {"code": "import os\\nprint(sorted(k for k in os.environ if k.startswith('COLONYMIND')),"
                                      " kept, [n for n in globals() if n.startswith('__colonymind')])"}, o)
# Hermes keeps its session snapshot in the first of TERMINAL_TEMP_DIR, TMPDIR/TMP/TEMP, HERMES_HOME/cache/terminal,
# /tmp that it may use: look in every one, so the leak check below is never vacuous.
places = [os.environ.get(n) for n in ("TERMINAL_TEMP_DIR", "TMPDIR", "TMP", "TEMP")]
places += [os.path.join(os.environ.get("HERMES_HOME", ""), "cache", "terminal"), "/tmp", tempfile.gettempdir()]
snapshots = sorted({p for d in places if d for p in glob.glob(os.path.join(d, "hermes-snap-*.sh"))})
leaked = any("Z9!" in pathlib.Path(p).read_text(errors="replace") for p in snapshots if os.path.exists(p))
emit(first=first.get("exit_code"), out=(work / "out").read_text(), second=second.get("output"),
     ran=ran.get("output"), after=after.get("output"), snapshots=len(snapshots), leaked=leaked)
''', home)
    assert result["first"] == 0 and result["out"] == PW + "\n"
    assert result["second"] == "unset|unset"
    assert result["snapshots"] >= 1 and result["leaked"] is False
    assert result["ran"].strip() == str(len(PW) + 1)
    assert result["after"].strip() == "[] 1 []"


def test_a_value_file_stays_while_the_program_that_reads_it_runs_and_no_longer(home, sidecar, vault):
    """A background command may read its file long after Hermes reports the call over: the file goes when the
    program that took it ends or, if that program was killed, at the next sweep; never by age alone."""
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = WIFI, OWNER, "password"
    vault.permitted.add("S-NNNNNNNN")
    result = probe(CODE + f'''
import pathlib, signal, subprocess, tempfile
command = 'while [ ! -e go ]; do sleep 0.05; done; cat "$COLONYMIND_SECRET_NNNNNNNN_FILE" > out'
o = turn("owner-b", "1001", "run it")
bindings = pathlib.Path({str(home.instance)!r}) / "private" / "secret-bindings"
def over():
    invoke_hook("post_tool_call", tool_name="terminal", args={{}}, result="", task_id="t", session_id=o,
                tool_call_id="c", turn_id="turn", api_request_id="r", duration_ms=0, status="ok", error_type=None,
                error_message=None, middleware_trace=[])
def start():
    call = request("terminal", {{"command": command, "background": True}}, o)
    work = pathlib.Path(tempfile.mkdtemp())
    proc = subprocess.Popen(["bash", "-c", call["args"]["command"]], cwd=work, start_new_session=True)
    deadline = time.time() + 20
    while not list(bindings.glob("*.pid")) and time.time() < deadline:
        time.sleep(0.05)
    return proc, work
proc, work = start()
over()                                          # Hermes reports a background call over at once
for path in bindings.glob("*"):
    os.utime(path, (time.time() - 7200,) * 2)  # and two hours pass
over()
kept = sorted(p.suffix for p in bindings.glob("*"))
(work / "go").touch()
proc.wait(20)
out = (work / "out").read_text()
after = sorted(p.name for p in bindings.glob("*"))
proc, work = start()                            # this one is killed before it ends
os.killpg(proc.pid, signal.SIGKILL)
proc.wait(20)
orphaned = len(list(bindings.glob("*.value")))
over()
emit(kept=kept, out=out, after=after, orphaned=orphaned, swept=sorted(p.name for p in bindings.glob("*")))
''', home)
    assert result["kept"] == [".pid", ".value"]
    assert result["out"] == WIFI
    assert result["after"] == []                  # gone when its program ended
    assert result["orphaned"] == 1 and result["swept"] == []


def test_on_a_backend_elsewhere_a_command_waits_and_says_how_to_use_the_value(home, sidecar, vault):
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = WIFI, OWNER, "password"
    vault.permitted.add("S-NNNNNNNN")
    result = probe(CODE + '''
o = turn("owner-d", "1001", "log in")
emit(call=request("terminal", {"command": 'login --password "$COLONYMIND_SECRET_NNNNNNNN"'}, o))
''', home, env={"TERMINAL_ENV": "docker"})
    assert result["call"]["action"] == "block" and "to_env" in result["call"]["message"]
    assert WIFI not in json.dumps(result["call"])


def test_only_his_latest_message_in_that_same_chat_stands_behind_a_fill(home, sidecar, vault):
    """A value they gave in one chat does not fill a call in another chat of theirs without the ask."""
    result = probe(CODE + f'''
from gateway.session_context import set_session_vars, clear_session_vars
sealed = dispatch("my key {KEY}", "1001")[0]["text"]     # their message in chat "c"
ref = sealed.split("key ", 1)[1]
tokens = set_session_vars(platform="telegram", user_id="1001", chat_id="elsewhere", chat_type="dm")
other = turn("owner-elsewhere", "1001", "use it")
elsewhere = request("send_message", {{"target": "telegram:2003", "message": "key: " + ref}}, other)
clear_session_vars(tokens)
tokens = set_session_vars(platform="telegram", user_id="1001", chat_id="c", chat_type="dm")
same = turn("owner-here", "1001", sealed)
here = request("send_message", {{"target": "telegram:2003", "message": "key: " + ref}}, same)
clear_session_vars(tokens)
emit(elsewhere=elsewhere, here=here)
''', home)
    assert result["elsewhere"]["action"] == "block" and "yes K7F2" in result["elsewhere"]["message"]
    assert result["here"]["args"]["message"] == f"key: {KEY}" and result["here"]["action"] is None


def test_a_write_or_patch_that_still_carries_a_reference_waits(home, sidecar, vault):
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = WIFI, OWNER, "password"
    ref = "[secret S-NNNNNNNN: password]"
    vault.values["S-PPPPPPPP"], vault.scopes["S-PPPPPPPP"], vault.labels["S-PPPPPPPP"] = KEY, "p-03", "API key"
    theirs = "[secret S-PPPPPPPP: API key]"
    result = probe(CODE + f'''
o = turn("owner-p", "1001", "fix the config")
emit(matched=request("patch", {{"path": "/tmp/x.conf", "old_string": "psk={ref}", "new_string": "psk=new"}}, o),
     filled=request("patch", {{"path": "/tmp/x.conf", "old_string": "psk=old", "new_string": "psk={ref}"}}, o),
     noted=request("write_file", {{"path": "/tmp/notes.md", "content": "they sent {theirs}; mine is {ref}"}}, o))
''', home)
    assert result["matched"]["action"] == "block" and "cannot be placed" in result["matched"]["message"]
    assert result["filled"]["action"] is None and result["filled"]["args"]["new_string"] == f"psk={WIFI}"
    # A reference whose value is never used (another person's) is written as the reference it is; their own is used.
    assert result["noted"]["action"] is None
    assert result["noted"]["args"]["content"] == f"they sent {theirs}; mine is {WIFI}"


def test_a_released_value_passes_the_gateway_as_it_came(home, sidecar, vault):
    vault.released.add(KEY)
    result = probe(CODE + f'''
emit(released=dispatch("the build id is {KEY}", "1001"), other=dispatch("the wifi password is {WIFI}", "1001"))
''', home)
    assert result["released"] == [{"action": "rewrite", "text": f"the build id is {KEY}"}]
    assert WIFI not in result["other"][0]["text"] and "[secret S-" in result["other"][0]["text"]


def test_a_reveal_goes_to_the_owners_chat_and_never_to_the_model(home, sidecar, vault):
    vault.values["S-AAAAAAAA"], vault.scopes["S-AAAAAAAA"], vault.labels["S-AAAAAAAA"] = KEY, OWNER, "API key"
    result = probe(CODE + '''
import tools.send_message_tool as _smt
SENT = []
_smt.send_message_tool = lambda args, **kw: SENT.append(dict(args)) or json.dumps({"success": True})
from gateway.session_context import set_session_vars, clear_session_vars
tokens = set_session_vars(platform="telegram", user_id="1001", chat_id="1001", chat_type="dm", session_key="dm-1001")
o = turn("owner-r", "1001", "show me my key")
shown = call("colonymind_secret", {"operation": "reveal", "ref": "S-AAAAAAAA"}, o)
clear_session_vars(tokens)
emit(shown=shown, sent=SENT)
''', home)
    assert KEY not in json.dumps(result["shown"]) and result["shown"]["shown_to_owner"] is True
    sent, = result["sent"]
    assert sent["target"] == "telegram:1001" and KEY in sent["message"] and "history" in sent["message"]


def test_the_body_holds_a_send_whose_reference_cannot_be_resolved_now(home, sidecar, vault):
    """The key is locked: nothing is claimed or sent, and the message leaves once the value can be read."""
    vault.values["S-AAAAAAAA"], vault.scopes["S-AAAAAAAA"] = KEY, OWNER
    vault.unavailable = True
    sidecar.mind_routes = True
    sidecar.mind.outbox["m-01"] = {"id": "m-01", "kind": "notice", "text": "Your key: [secret S-AAAAAAAA: API key]",
                                   "target": "telegram:1001", "recipient_is_owner": True}
    result = probe('''
_smt.send_message_tool = fake_send
tick()
held = list(SENDS)
emit(held=held, ledger=body.ledger.send("m-01"))
''', home, prelude=MIND_PRELUDE)
    assert result["held"] == [] and result["ledger"] is None
    assert sidecar.calls("/v1/mind/outbox/m-01/sending") == []
    vault.unavailable = False
    result = probe('''
_smt.send_message_tool = fake_send
tick()
emit(sends=SENDS)
''', home, prelude=MIND_PRELUDE)
    assert [s["message"] for s in result["sends"]] == [f"Your key: {KEY}"]


def test_the_body_resolves_every_reference_of_a_message_or_holds_it(home, sidecar, vault):
    """More references than one request carries are asked in turns; a reference the sidecar gave no answer for
    holds the message, which never leaves with a reference it could not settle."""
    refs = {f"S-{chr(65 + n // 8)}{chr(65 + n % 8)}AAAAAA": f"Value{n:02d}x!" for n in range(33)}
    for ref, value in refs.items():
        vault.values[ref], vault.scopes[ref] = value, OWNER
    sidecar.mind_routes = True
    words = " ".join(f"[secret {ref}: password]" for ref in refs)
    sidecar.mind.outbox["m-01"] = {"id": "m-01", "kind": "notice", "text": "Yours: " + words,
                                   "target": "telegram:1001", "recipient_is_owner": True}
    result = probe('''
_smt.send_message_tool = fake_send
tick()
emit(sends=SENDS)
''', home, prelude=MIND_PRELUDE)
    sent, = [s["message"] for s in result["sends"]]
    assert sent == "Yours: " + " ".join(refs.values())
    del sidecar.mind.outbox["m-01"]
    vault.dropped.add(next(iter(refs)))
    sidecar.mind.outbox["m-02"] = {"id": "m-02", "kind": "notice", "text": "Again: " + words,
                                   "target": "telegram:1001", "recipient_is_owner": True}
    result = probe('''
_smt.send_message_tool = fake_send
tick()
emit(sends=SENDS, ledger=body.ledger.send("m-02"))
''', home, prelude=MIND_PRELUDE)
    assert [s for s in result["sends"] if s["message"].startswith("Again")] == []
    assert result["ledger"] is None


def test_a_rejected_turn_keeps_its_payload_for_repair(home, sidecar):
    result = probe('''
from colonymind_hermes.capture import TurnOutbox, Undeliverable
import tempfile, pathlib
path = pathlib.Path(tempfile.mkdtemp()) / "outbox.sqlite3"
box = TurnOutbox(path)
box.enqueue("turn-r", {"user_message": {"role": "user", "content": "remind me to call the bank"}})
def reject(payload):
    raise Undeliverable("HTTP 422")
box.drain(reject, limit=1)
box.prune(keep_delivered=0)
emit(rows=TurnOutbox(path).rows(), pending=box.pending_count())
''', home)
    row, = result["rows"]
    assert row["state"] == "delivered" and row["last_error"] == "rejected: HTTP 422" and result["pending"] == 0
    assert row["payload"]["user_message"]["content"] == "remind me to call the bank"


SECTIONS = [None, {}, {"capture": {"rules": ["prefx"]}}, {"capture": {"rules": ["prefix", "jwt"]}},
            {"capture": {"rules": {"card": "off"}}}, {"capture": {"rules": {"card": "maybe"}}},
            {"capture": {"ingress": False}}, {"capture": {"ingress": "sometimes"}}, {"capture": "on"},
            {"vault": {"key_store": "cloud"}, "capture": {"ingress": False}}, {"capture": {"rules": "prefix"}},
            {"migrate": {"on_start": "later"}, "capture": {"enabled": False}}, "yes"]


@pytest.mark.parametrize("section", SECTIONS, ids=[json.dumps(s) for s in SECTIONS])
@pytest.mark.requires_sidecar
def test_the_plugin_reads_the_secrets_section_as_the_sidecar_does(tmp_path, section):
    """A malformed section never weakens gateway sealing: the plugin falls back to the defaults exactly where the
    sidecar does (a misspelled rule list once left the gateway with no rule at all)."""
    import yaml
    from colonymind.secrets.config import SecretsSettings, parse_settings
    from colonymind_hermes.client import Settings
    from colonymind_hermes.secrets import capture_settings
    (tmp_path / "colonymind.yaml").write_text(yaml.safe_dump({} if section is None else {"secrets": section}))
    settings = Settings(sidecar_url="http://127.0.0.1:9", key_file=tmp_path / "key", api_key="k", home=tmp_path,
                        hermes_home=tmp_path, outbox_path=tmp_path / "outbox.sqlite3")
    try:
        expected = parse_settings(section)
    except ValueError:
        expected = SecretsSettings()
    assert capture_settings(settings) == {"enabled": expected.capture_enabled, "ingress": expected.ingress,
                                          "rules": expected.rules}


def test_a_revealed_value_file_is_removed_on_time(home, sidecar, vault):
    """No chat to send it to: the file waits for the owner a limited time, and goes even if nothing else happens."""
    vault.values["S-AAAAAAAA"], vault.scopes["S-AAAAAAAA"], vault.labels["S-AAAAAAAA"] = KEY, OWNER, "API key"
    result = probe(CODE + '''
import pathlib
import colonymind_hermes.secrets as secrets_module
secrets_module.REVEAL_SECONDS = 0.5
o = turn("owner-r", "1001", "show me my key")
shown = call("colonymind_secret", {"operation": "reveal", "ref": "S-AAAAAAAA"}, o)
there = pathlib.Path(shown["file"]).exists()
time.sleep(2)
emit(there=there, later=pathlib.Path(shown["file"]).exists())
''', home)
    assert result == {"there": True, "later": False}


def test_hermes_still_reads_the_words_of_a_command_given_a_value(home, sidecar, vault):
    """The command follows the lines that set its values as it was written, so what Hermes does with a command's
    own words still happens: a ``sudo`` in it is served as in any other command."""
    vault.values["S-NNNNNNNN"], vault.scopes["S-NNNNNNNN"], vault.labels["S-NNNNNNNN"] = WIFI, OWNER, "password"
    vault.permitted.add("S-NNNNNNNN")
    result = probe(CODE + '''
from tools.terminal_tool_sudo import _count_real_sudo_invocations
o = turn("owner-s", "1001", "update it")
command = 'sudo -E env TOKEN="$COLONYMIND_SECRET_NNNNNNNN" /usr/bin/true'
bound = request("terminal", {"command": command}, o)
emit(action=bound["action"], sudo=_count_real_sudo_invocations(bound["args"]["command"]),
     plain=_count_real_sudo_invocations(command))
''', home)
    assert result == {"action": None, "sudo": 1, "plain": 1}
