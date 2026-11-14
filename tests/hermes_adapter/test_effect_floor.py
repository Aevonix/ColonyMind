"""The floor by effect at the plugin: what a mind run's call does, classified by the table the sidecar shares.

In a mind run the plugin classifies each effectful call (``effect_of``: the tool and its target, and for a command
the directory it runs in) and sends that reading with the guard request; the sidecar decides and holds a floor
effect as the owner's ask. Nothing is parsed to set a command free: a command is never read as the worker's own.
Nothing in a mind run raises Hermes' approval gate. A cron run whose job a mind run made is a mind run too; the
owner's own session and the owner's own cron jobs are never gated. Standing grants are the owner's:
``colonymind_self grant`` and ``revoke`` answer only the owner's own conversation. A declared action
(``colonymind_action``) runs the owner's fixed command with checked arguments.
"""

import json
import sys

import pytest

from conftest import probe, refused, worker_env
from test_tools_commands import TOOL_CODE
from test_worker_boundary import WORKER_CODE

HELD = {"allow": False, "action": "block", "held": "K7F",
        "reason": "held for the owner's yes (ask K7F): irreversible deletion; stop and report that it waits"}


def test_a_mind_run_sends_its_calls_effect_and_a_hold_is_a_block_never_the_gate(home, sidecar):
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    rm = "r" + "m -rf"
    result = probe(WORKER_CODE + '''
emit(outside=check("terminal", {"command": "%(rm)s ../shared"}),
     own=check("terminal", {"command": "%(rm)s ./build"}),
     push=check("terminal", {"command": "git push --force origin main"}))
''' % {"rm": rm}, home, env=worker_env(home))
    for name in ("outside", "own", "push"):      # the sidecar's verdict decides; its hold is a block
        verdict = result[name]
        assert verdict["action"] == "block" and "ask K7F" in verdict["message"], (name, verdict)
    sent = {call["json"]["args"]["command"]: call["json"] for call in sidecar.calls("/v1/mind/guard", "POST")}
    outside, own, push = sent[f"{rm} ../shared"], sent[f"{rm} ./build"], sent["git push --force origin main"]
    assert outside["run"] == "mind" and outside["task_id"] == "task-01" and "workspace" not in outside
    assert outside["cwd"] == str(home.workspace)
    assert outside["effect"]["cls"] == "irreversible_deletion" and outside["effect"]["digest"]
    assert own["effect"]["cls"] == "irreversible_deletion"      # never read as the worker's own: the owner decides
    assert push["effect"]["cls"] == "irreversible_deletion"


def test_a_floor_effect_without_the_mind_routes_is_blocked_and_the_rest_is_unchanged(home, sidecar):
    """An older sidecar has no one to take the owner's yes: the floor effect is refused in words the worker can
    report; an unbounded command it cannot hold goes to the rules as before (a 404 allows)."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    result = probe(WORKER_CODE + '''
emit(renew=check("terminal", {"command": "certbot renew"}), plain=check("terminal", {"command": "make site"}),
     vault=check("browser_vault_fill", {"handle": "h-1"}))
''', home, env=worker_env(home))
    assert result["renew"]["action"] == "block" and "credential change" in result["renew"]["message"]
    assert "waits for the owner's yes" in result["renew"]["message"]
    assert result["vault"]["action"] == "block" and "money movement" in result["vault"]["message"]  # kind unread
    assert result["plain"]["action"] is None


def test_the_owners_own_session_is_never_gated(home, sidecar):
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    rm = "r" + "m -rf"
    result = probe(WORKER_CODE + '''
from hermes_cli.lifecycle import invoke_hook
invoke_hook("pre_llm_call", session_id="owner-1", task_id="t", turn_id="turn", user_message="clean up",
            conversation_history=[], is_first_turn=True, model="m", platform="telegram", parent_session_id="",
            sender_id="1001")
from hermes_cli.plugins import get_pre_tool_call_directive
def owner_check(tool, args):
    action, message = get_pre_tool_call_directive(tool, args, session_id="owner-1", task_id="t", tool_call_id="c",
                                                  turn_id="turn", api_request_id="r")
    return {"action": action, "message": message}
emit(delete=owner_check("execute_code", {"code": "os.system('%(rm)s /srv/old')"}),
     pay=owner_check("execute_code", {"code": "requests.post('https://api.stripe.com/v1/charges')"}),
     send=owner_check("send_message", {"target": "telegram:2003", "message": "hi"}))
''' % {"rm": rm}, home)
    assert all(verdict["action"] is None for verdict in result.values()), result
    assert sidecar.calls("/v1/mind/guard", "POST") == []


CRON_CODE = '''
from uuid import uuid4
from cron import jobs
from hermes_cli.plugins import get_pre_tool_call_directive
def marker():
    return "marker-" + str(uuid4())
call_id, turn_id, request_id = marker(), marker(), marker()
def cron_check(job, tool, args, session=None, task=marker()):
    session = session or "cron_%s_20300101_120000" % job["id"]
    action, message = get_pre_tool_call_directive(tool, args, session_id=session, task_id=task, tool_call_id=call_id,
                                                  turn_id=turn_id, api_request_id=request_id)
    return {"action": action, "message": message}

def observe_cron(job):
    from hermes_cli.lifecycle import invoke_hook
    session = "cron_%s_20300101_120000" % job["id"]
    invoke_hook("pre_llm_call", session_id=session, task_id="cron:%s:%s" % (job["id"], marker()),
                turn_id=turn_id, user_message="go", conversation_history=[], is_first_turn=True,
                model="stub", platform="cron", parent_session_id="", sender_id="")
    return session

def delegate_cron(job):
    from hermes_cli.lifecycle import invoke_hook
    parent, child = observe_cron(job), marker()
    invoke_hook("subagent_start", parent_session_id=parent, child_session_id=child)
    invoke_hook("pre_llm_call", session_id=child, task_id=child, turn_id=turn_id, user_message="go",
                conversation_history=[], is_first_turn=True, model="stub", platform="subagent",
                parent_session_id=parent, sender_id="")
    return child
'''
MAKE_JOBS = '''
import json
mine = jobs.create_job(prompt="check the build and tidy up", schedule="every 1h")
owners = jobs.create_job(prompt="send me the morning summary", schedule="every 1h")
mine["id"], owners["id"] = marker(), marker()
jobs.save_jobs([mine, owners])
emit(mine=mine["id"], owners=owners["id"])
'''


def cron_origins(sidecar, mind_jobs, *, down=(), gone=()):
    """The sidecar's ``GET /v1/mind/cron-origin``: the jobs a mind run made, by id; a job in ``down`` is answered
    503 once (the sidecar could not say), one in ``gone`` 404 once (something else answered)."""
    original, asked, failed = sidecar.dispatch, [], set()

    def dispatch(method, path, query, body):
        if path == "/v1/mind/cron-origin" and method == "GET":
            job = query.get("job_id")
            asked.append(job)
            if (job in down or job in gone) and job not in failed:
                failed.add(job)
                return (503, {"detail": "starting"}) if job in down else (404, {"detail": "not found"})
            mind = job in mind_jobs
            return 200, {"origin": "mind" if mind else "owner", "mind": mind, "task_id": "task-09" if mind else "",
                         "intention_id": "i-9" if mind else None}
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    return asked


def cron_jobs(home):
    """Retained synthetic jobs; retention alone supplies no creator authority."""
    return probe(CRON_CODE + MAKE_JOBS, home)


def owner_cron_record(home, job):
    """Fixture for positive host-private owner creation evidence, separate from the legacy route."""
    from colonymind_hermes.standing import STANDING_FILE, StandingStore
    StandingStore(home.instance / STANDING_FILE).put("cron", job, "owner", "telegram")


@pytest.mark.parametrize("owner_record", [False, True])
def test_a_cron_job_a_mind_run_made_runs_as_a_mind_run_and_the_owners_does_not(home, sidecar, owner_record):
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    ids = cron_jobs(home)
    if owner_record:
        owner_cron_record(home, ids["owners"])
    asked = cron_origins(sidecar, {ids["mine"]})
    result = probe(CRON_CODE + '''
mine, owners = {"id": %(mine)r}, {"id": %(owners)r}
emit(mind_push=cron_check(mine, "terminal", {"command": "git push --force"}),
     mind_write=cron_check(mine, "write_file", {"path": "notes.md", "content": "x"}),
     owner_push=cron_check(owners, "terminal", {"command": "git push --force"}),
     owner_write=cron_check(owners, "write_file", {"path": "notes.md", "content": "x"}))
''' % ids, home)
    assert result["mind_push"]["action"] == "block" and "ask K7F" in result["mind_push"]["message"]
    assert result["mind_write"]["action"] == "block" and "workspace" in result["mind_write"]["message"]
    for name in ("owner_push", "owner_write"):
        assert result[name]["action"] == (None if owner_record else "block"), (name, result[name])
        if not owner_record:
            assert "retry: false" in result[name]["message"]
    guarded = sidecar.calls("/v1/mind/guard", "POST")
    assert [call["json"]["run"] for call in guarded] == ["mind"]
    assert guarded[0]["json"]["task_id"] == "task-09" and guarded[0]["json"]["effect"]["cls"] == "irreversible_deletion"
    assert sorted(asked) == sorted([ids["mine"]] + ([] if owner_record else [ids["owners"]] * 2))


@pytest.mark.parametrize("owner_record", [False, True])
def test_a_cron_run_is_known_by_its_job_through_compression_and_delegation(home, sidecar, owner_record):
    """A compressed cron run has a new session id but keeps its task id (``cron:<job>:<run>``); a delegated child
    has its own session and task, and its parent session is the cron run's."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    ids = cron_jobs(home)
    if owner_record:
        owner_cron_record(home, ids["owners"])
    cron_origins(sidecar, {ids["mine"]})
    result = probe(CRON_CODE + '''
mine, owners = {"id": %(mine)r}, {"id": %(owners)r}
child, owner_child = delegate_cron(mine), delegate_cron(owners)
push = {"command": "git push --force"}
emit(compressed=cron_check(mine, "terminal", push, session=marker(), task="cron:%(mine)s:" + marker()),
     owner_compressed=cron_check(owners, "terminal", push, session=marker(), task="cron:%(owners)s:" + marker()),
     child=cron_check(mine, "terminal", push, session=child, task=child),
     owner_child=cron_check(owners, "terminal", push, session=owner_child, task=owner_child))
''' % ids, home)
    for name in ("compressed", "child"):
        assert result[name]["action"] == "block" and "ask K7F" in result[name]["message"], (name, result[name])
    for name in ("owner_compressed", "owner_child"):
        assert result[name]["action"] == (None if owner_record else "block"), (name, result[name])
        if not owner_record:
            assert "retry: false" in result[name]["message"]


@pytest.mark.parametrize("owner_record", [False, True])
def test_a_cron_runs_origin_that_cannot_be_read_is_not_the_owners(home, sidecar, owner_record):
    """Neither an unavailable route nor its later default-owner answer grants authority. A positive creator
    record independently permits owner effects; a cron context with no job to name stays unknown."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    ids = cron_jobs(home)
    if owner_record:
        owner_cron_record(home, ids["owners"])
    asked = cron_origins(sidecar, {ids["mine"]}, down={ids["owners"]})
    result = probe(CRON_CODE + '''
owners = {"id": %(owners)r}
first = cron_check(owners, "write_file", {"path": "notes.md", "content": "x"})
second = cron_check(owners, "write_file", {"path": "notes.md", "content": "x"})
emit(first=first, second=second)
''' % ids, home)
    for name in ("first", "second"):
        assert result[name]["action"] == (None if owner_record else "block"), (name, result[name])
        if not owner_record:
            assert "retry: false" in result[name]["message"]
    assert asked.count(ids["owners"]) == (0 if owner_record else 2)
    result = probe(CRON_CODE + '''
emit(nameless=cron_check({"id": marker()}, "terminal", {"command": "git push --force"},
                         session=marker(), task=marker()))
''', home, env={"HERMES_CRON_SESSION": "1"})
    assert result["nameless"]["action"] == "block"


@pytest.mark.parametrize("owner_record", [False, True])
def test_an_answer_that_says_nothing_about_a_cron_runs_job_is_not_kept(home, sidecar, owner_record):
    """A 404 from the origin route, or a session whose ancestry cannot be read, says nothing about whose job a
    run is for: the run is gated meanwhile and asked about again at its next call, so a mind job's run is gated
    once the answer comes. A child needs its parent's creation snapshot; later knowledge cannot upgrade an
    explicit unknown child. Fresh children inherit positive mind/owner evidence only."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = HELD
    ids = cron_jobs(home)
    if owner_record:
        owner_cron_record(home, ids["owners"])
    asked = cron_origins(sidecar, {ids["mine"]}, gone={ids["mine"]})
    result = probe(CRON_CODE + '''
mine = {"id": %(mine)r}
push = {"command": "git push --force"}
emit(first=cron_check(mine, "terminal", push), second=cron_check(mine, "terminal", push))
''' % ids, home)
    assert result["first"]["action"] == "block" and "retry: false" in result["first"]["message"]
    assert result["second"]["action"] == "block" and "ask K7F" in result["second"]["message"]
    assert asked.count(ids["mine"]) == 2                   # the 404 was not kept as the owner's answer
    result = probe(CRON_CODE + '''
import hermes_state_registry
from hermes_cli.lifecycle import invoke_hook
mine = {"id": %(mine)r}
push = {"command": "git push --force"}
child = marker()
real = hermes_state_registry.acquire
def unavailable(*args, **kwargs):
    raise RuntimeError("the session store is busy")
hermes_state_registry.acquire = unavailable
first = cron_check(mine, "terminal", push, session=child, task=child)
hermes_state_registry.acquire = real
invoke_hook("pre_llm_call", session_id=child, task_id=child, turn_id=turn_id, user_message="go",
            conversation_history=[], is_first_turn=True, model="stub", platform="subagent",
            parent_session_id="cron_%(mine)s_20300101_120000", sender_id="")
fresh = delegate_cron(mine)
owners = {"id": %(owners)r}
owner_child = delegate_cron(owners)
emit(first=first, second=cron_check(mine, "terminal", push, session=child, task=child),
     fresh=cron_check(mine, "terminal", push, session=fresh, task=fresh),
     owner_child=cron_check(owners, "terminal", push, session=owner_child, task=owner_child))
''' % ids, home)
    for name in ("first", "second"):
        assert result[name]["action"] == "block" and "retry: false" in result[name]["message"], (name, result[name])
    assert result["fresh"]["action"] == "block" and "ask K7F" in result["fresh"]["message"]
    assert result["owner_child"]["action"] == (None if owner_record else "block")
    if not owner_record:
        assert "retry: false" in result["owner_child"]["message"]


def test_what_a_mind_run_made_is_reported_after_it_was_made(home, sidecar):
    """A cron job a mind run created (any form: the id stock cron gave it) and a kanban task it created are told to
    the sidecar once the call returned; a failed create is told as failed; the owner's own session tells nothing."""
    sidecar.mind_routes = True
    code = '''
import json
from hermes_cli.lifecycle import invoke_hook
# The real worker observes its session before the model runs, independently of
# the missing pre-tool hook exercised below.
invoke_hook("pre_llm_call", session_id="worker-session", task_id="task-01", turn_id="turn",
            user_message="", conversation_history=[], is_first_turn=True, model="m", platform="cli",
            parent_session_id="", sender_id="")
def done(tool, args, result, call="c-9"):
    invoke_hook("post_tool_call", tool_name=tool, args=args, result=json.dumps(result), task_id="task-01",
                session_id="worker-session", tool_call_id=call, turn_id="turn", api_request_id="r", duration_ms=1,
                status="ok", error_type=None, error_message=None, middleware_trace=[])
done("cronjob_manage", {"action": "create", "skills": ["daily-check"], "schedule": "every 1h"},
     {"success": True, "job_id": "job-77"})
done("cronjob_manage", {"action": "create", "prompt": "p", "schedule": "every 1h"},
     {"success": False, "error": "bad schedule"}, call="c-10")
done("kanban_create", {"title": "t", "assignee": "colonymind-act"}, {"ok": True, "task_id": "task-02"}, call="c-11")
emit(ok=True)
'''
    probe(WORKER_CODE + code, home, env=worker_env(home))
    told = [call["json"] for call in sidecar.calls("/v1/mind/cron-origin", "POST")]
    assert told == [{"job_id": "job-77", "task_id": "task-01", "tool_call_id": "c-9", "action": "create", "ok": True},
                    {"job_id": "", "task_id": "task-01", "tool_call_id": "c-10", "action": "create", "ok": False}]
    assert [call["json"] for call in sidecar.calls("/v1/mind/lineage", "POST")] == [
        {"child": "task-02", "parent": "task-01"}]
    before = len(sidecar.requests)
    probe(WORKER_CODE + code, home)
    assert not [r for r in sidecar.requests[before:] if r["path"] in {"/v1/mind/cron-origin", "/v1/mind/lineage"}]


def test_a_terminal_calls_directory_is_the_one_hermes_runs_it_in(home, sidecar):
    """After a ``cd`` Hermes runs the session's next command in the directory it recorded, not ``TERMINAL_CWD``;
    an explicit ``workdir`` wins."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    elsewhere = home.root / "elsewhere"
    elsewhere.mkdir()
    rm = "r" + "m -rf"
    probe(WORKER_CODE + '''
from tools.terminal_tool import record_session_cwd
record_session_cwd("task-01", %(elsewhere)r)
emit(moved=check("terminal", {"command": "%(rm)s ./build"}),
     pinned=check("terminal", {"command": "%(rm)s ./dist", "workdir": %(ws)r}))
''' % {"elsewhere": str(elsewhere), "ws": str(home.workspace), "rm": rm}, home, env=worker_env(home))
    sent = {call["json"]["args"]["command"]: call["json"] for call in sidecar.calls("/v1/mind/guard", "POST")}
    moved, pinned = sent[f"{rm} ./build"], sent[f"{rm} ./dist"]
    assert moved["cwd"] == str(elsewhere) and moved["effect"]["cls"] == "irreversible_deletion"
    assert pinned["cwd"] == str(home.workspace) and pinned["effect"]["cls"] == "irreversible_deletion"
    assert moved["effect"]["digest"] != pinned["effect"]["digest"]


VAULT_CODE = WORKER_CODE + '''
import json
from contextlib import nullcontext
from types import SimpleNamespace
import agent.vault_backends as vault_backends
import tools.browser_vault_tool as vault_tool
from tools.registry import registry
ORIGIN, ELSEWHERE = "https://shop.example.org", "https://elsewhere.example.org"
meta = SimpleNamespace(kind="payment", label="Main card", origin=ORIGIN, has_otp=False,
                       allowed_origins=())  # Vault metadata supports multiple origins.
class Backend:
    name, display_name, needs_unlock = "local", "Hermes vault", False
    def is_unlocked(self):
        return True
    def get_meta(self, handle):
        return meta if handle == "vault_card" else None
    def fill_context(self):
        return nullcontext()
vault_backends.backend_for_handle = lambda handle: Backend()
def page(origin):
    vault_tool._focus_bound_origin = lambda *args, **kwargs: origin
    vault_tool._current_page_origin = lambda *args, **kwargs: origin
# The source says which tool carries a fill: one tool per action (``browser_vault_fill``) or ONE tool with an
# ``action`` (``browser_vault``). The guard is asked about that call and the fill runs through that tool's handler.
registered = sorted(name for name in registry.get_all_tool_names() if name.startswith("browser_vault"))
ONE_TOOL = "browser_vault" in registered
assert ONE_TOOL != ("browser_vault_fill" in registered), registered
TOOL = "browser_vault" if ONE_TOOL else "browser_vault_fill"
def call(handle="vault_card"):
    return {"action": "fill", "handle": handle} if ONE_TOOL else {"handle": handle}
def fill_as(handle, task):
    return json.loads(registry.get_entry(TOOL).handler(call(handle), task_id=task))
def fill():
    return fill_as("vault_card", "task-01")
# A source with one tool per action asks its consent first and looks at the page after it: the page stands
# elsewhere, so a fill that is past the consent ends at ``origin_mismatch``. A source with the one tool checks the
# page first: the page is the item's own, and what follows the consent there is a stand-in that names itself.
page(ORIGIN if ONE_TOOL else ELSEWHERE)
PAST_CONSENT = "page_inspection" if ONE_TOOL else "origin_mismatch"
vault_tool.snapshot_refusal = lambda *args, **kwargs: None
vault_tool._fill_bound_page = lambda *args, **kwargs: json.dumps({"success": False, "error_type": PAST_CONSENT})
asked = []      # what Hermes' own consent was asked: [label, origin]
stock_confirm = vault_tool._confirm_payment_fill
def confirm(label, origin):
    asked.append([label, origin])
    return stock_confirm(label, origin)
vault_tool._confirm_payment_fill = confirm
shape = {"registered": registered, "tool": TOOL, "call": call(), "past_consent": PAST_CONSENT}
'''


def _fill_calls(sidecar, shape):
    """The guard requests for the fill: each sent as the source has the call, and read as the older tool."""
    sent = [call["json"] for call in sidecar.calls("/v1/mind/guard", "POST")]
    assert sent and all(call["tool"] == shape["tool"] and call["args"] == shape["call"] for call in sent), sent
    assert shape["tool"] in shape["registered"] and shape["tool"] in ("browser_vault", "browser_vault_fill")
    assert shape["call"] == ({"action": "fill", "handle": "vault_card"} if shape["tool"] == "browser_vault"
                             else {"handle": "vault_card"})
    for call in sent:
        assert (call["effect"]["cls"], call["effect"]["tool"]) == ("money_movement", "browser_vault_fill"), call
        assert call["context"] == {"vault_kind": "payment"}
    return sent


def test_the_owners_yes_to_a_payment_fill_is_hermes_consent_for_that_fill_once(home, sidecar):
    """Stock ``browser_vault_fill`` asks its own consent before a card is written into a page, which a headless
    worker cannot give. The owner's exact yes (a permit) or their standing grant is that consent, for that item,
    once; the owner's approval of the task alone is not. The fill is the tool the source registers: one tool per
    action, or the one vault tool with ``action: "fill"``. The yes ends with the fill it was given for."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = {"allow": True, "action": "allow", "permit": "i-9",
                             "reason": "the owner approved exactly this call"}
    result = probe(VAULT_CODE + '''
before = fill()
allowed = check(TOOL, call())
first, second = fill(), fill()
emit(shape=shape, before=before, allowed=allowed, first=first, second=second, asked=asked)
''', home, env=worker_env(home))
    shape = result["shape"]
    assert result["before"]["error_type"] == "payment_declined"        # no yes: Hermes' consent refuses
    assert result["allowed"]["action"] is None
    assert result["first"]["error_type"] == shape["past_consent"]      # past the consent, on to what follows it
    assert result["second"]["error_type"] == "payment_declined"        # once
    assert len(result["asked"]) == 2 and len(_fill_calls(sidecar, shape)) == 1    # before and second: Hermes' own
    # A fill that Hermes refuses at the page takes the yes with it. A source with the one tool refuses a page on
    # another origin BEFORE it asks its consent (nothing is asked for it); the yes given for that fill is gone all
    # the same when the page is the item's own again.
    result = probe(VAULT_CODE + '''
page(ELSEWHERE)
unasked = fill()
count = len(asked)
allowed = check(TOOL, call())
refused = fill()
during = len(asked) - count
page(ORIGIN if ONE_TOOL else ELSEWHERE)
after = fill()
emit(shape=shape, unasked=unasked, allowed=allowed, refused=refused, during=during, after=after, asked=asked)
''', home, env=worker_env(home))
    assert result["shape"] == shape and result["allowed"]["action"] is None
    assert result["refused"]["error_type"] == "origin_mismatch" and result["during"] == 0
    assert result["after"]["error_type"] == "payment_declined"         # the yes went with the refused fill
    if shape["tool"] == "browser_vault":
        assert result["unasked"]["error_type"] == "origin_mismatch"    # refused before the consent, yes or no yes
        assert result["asked"] == [["Main card", "https://shop.example.org"]]      # asked once, by the last fill
    else:
        assert result["unasked"]["error_type"] == "payment_declined"   # the consent comes first there
        assert len(result["asked"]) == 2                               # the fill with no yes, and the last one
    sidecar.guard_verdict = {"allow": True, "action": "allow", "reason": "the owner approved this task"}
    result = probe(VAULT_CODE + '''
allowed = check(TOOL, call())
emit(allowed=allowed, filled=fill())
''', home, env=worker_env(home))
    assert result["allowed"]["action"] is None and result["filled"]["error_type"] == "payment_declined"


def test_the_owners_yes_to_one_fill_is_no_consent_for_another_item_or_task(home, sidecar):
    """The consent the owner's yes gives is for the fill they said yes to: that item, in that task, once. Another
    payment item with the same label and site, or the same item filled from another task, asks Hermes' own
    consent as before (which a worker cannot give)."""
    home.write_mind(deny={"tools": [], "text": [], "commands": []})
    sidecar.mind_routes = True
    sidecar.guard_verdict = {"allow": True, "action": "allow", "permit": "i-9",
                             "reason": "the owner approved exactly this call"}
    result = probe(VAULT_CODE + '''
Backend.get_meta = lambda self, handle: meta if handle in ("vault_card", "vault_card_2") else None
allowed = check(TOOL, call())
emit(shape=shape, allowed=allowed, twin=fill_as("vault_card_2", "task-01"),
     other_task=fill_as("vault_card", "task-02"), own=fill_as("vault_card", "task-01"),
     again=fill_as("vault_card", "task-01"), asked=asked)
''', home, env=worker_env(home))
    shape = result["shape"]
    assert result["allowed"]["action"] is None and len(_fill_calls(sidecar, shape)) == 1
    assert result["twin"]["error_type"] == "payment_declined"            # same label and site, another item
    assert result["other_task"]["error_type"] == "payment_declined"      # the same item from another task
    assert result["own"]["error_type"] == shape["past_consent"]          # past the consent, on to what follows it
    assert result["again"]["error_type"] == "payment_declined"           # once
    assert len(result["asked"]) == 3                                     # the twin, the other task, the second fill


ACTION_CODE = TOOL_CODE + '''
def run(args, session):
    return json.loads(registry.dispatch("colonymind_action", args, task_id="t", session_id=session))
'''


def test_a_declared_action_runs_its_fixed_command_with_checked_arguments(home, sidecar):
    """The worked example: ``mind.actions`` declares renewing one site's certificate as a fixed command whose only
    free word is a checked argument. The plugin offers ``colonymind_action`` for it; a call runs exactly that command
    (no shell), a value outside the declaration is refused before anything runs, a guest cannot run it, and in a mind
    run the guard sends the action's effect (its class, its name as the tool) for the sidecar's grant or hold. The
    plugin runs the declaration it reads now, so a sidecar that allowed a different reading of the call (one still
    on an older declaration) lets nothing run: the call waits for the owner."""
    marker = home.root / "renewed.txt"
    script = ("import sys, pathlib; pathlib.Path(sys.argv[1]).write_text(' '.join(sys.argv[2:])); "
              "print('renewed ' + sys.argv[-1])")
    home.write_mind(deny={"tools": [], "text": [], "commands": []}, actions={"renew_certificates": {
        "description": "Renew the TLS certificate of one site", "class": "credential_change",
        "command": [sys.executable, "-c", script, str(marker), "renew", "--cert-name", "{site}"],
        "args": {"site": {"type": "string", "enum": ["example.org", "www.example.org"]}}, "target": "site"}})
    result = probe(ACTION_CODE + '''
o = owner("owner-1", message="renew the certificate of example.org")
g = guest("guest-1", message="renew it")
emit(ran=run({"action": "renew_certificates", "args": {"site": "example.org"}}, o),
     hook=run({"action": "renew_certificates", "args": {"site": "--deploy-hook=/bin/true"}}, o),
     other=run({"action": "renew_certificates", "args": {"site": "evil.example.org"}}, o),
     extra=run({"action": "renew_certificates", "args": {"site": "example.org", "post_hook": "x"}}, o),
     unknown=run({"action": "wipe"}, o),
     guest=run({"action": "renew_certificates", "args": {"site": "example.org"}}, g),
     schema=registry.get_schema("colonymind_action"))
''', home)
    assert result["ran"]["ok"] is True and result["ran"]["exit_code"] == 0
    assert "renewed example.org" in result["ran"]["stdout"]
    assert marker.read_text() == "renew --cert-name example.org"
    for name in ("hook", "other", "extra", "unknown"):
        assert "error" in result[name], (name, result[name])
    assert "owner" in refused(result["guest"])
    described = result["schema"]["description"]
    assert "renew_certificates" in described and "Renew the TLS certificate" in described and "example.org" in described
    assert marker.read_text() == "renew --cert-name example.org"          # nothing else ran
    sidecar.mind_routes = True
    original, stale = sidecar.dispatch, {"on": True}

    def dispatch(method, path, query, body):
        if path == "/v1/mind/guard" and method == "POST" and body.get("tool") == "colonymind_action":
            if stale["on"]:     # a sidecar still on a declaration where the action's class was none
                return 200, {"allow": True, "action": "allow", "reason": "allowed"}
            return 200, {"allow": True, "action": "allow", "grant": "renew-certs", "effect": body["effect"],
                         "reason": "standing grant renew-certs (renew TLS certificates)"}
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    request = '''
emit(verdict=check("colonymind_action", {"action": "renew_certificates", "args": {"site": "www.example.org"}}))
'''
    held = probe(WORKER_CODE + request, home, env=worker_env(home))
    assert held["verdict"]["action"] == "block" and "waits for the owner" in held["verdict"]["message"]
    stale["on"] = False
    worker = probe(WORKER_CODE + request, home, env=worker_env(home))
    assert worker["verdict"]["action"] is None
    sent = [call["json"] for call in sidecar.calls("/v1/mind/guard", "POST") if call["json"]["tool"] == "colonymind_action"]
    assert sent[-1]["effect"]["cls"] == "credential_change" and sent[-1]["effect"]["tool"] == "renew_certificates"
    assert sent[-1]["effect"]["target"] == "www.example.org"


def test_no_action_is_offered_until_the_owner_declares_one(home, sidecar):
    result = probe(TOOL_CODE + '''
emit(offered="colonymind_action" in registry.get_all_tool_names())
''', home)
    assert result["offered"] is False


def test_standing_grants_are_the_owners_alone(home, sidecar):
    sidecar.mind_routes = True
    original = sidecar.dispatch

    def dispatch(method, path, query, body):
        if path == "/v1/mind/grants" and method == "POST":
            if body["grant"].get("class") not in {"credential_change"}:
                return 422, {"detail": {"code": "invalid_grant", "message": "class is one of money_movement, ..."}}
            return 200, {"ok": True, "status": "asked", "code": "Q4M", "text": "reply 'yes Q4M'"}
        if path.startswith("/v1/mind/grants/") and path.endswith("/revoke"):
            return 200, {"ok": True, "status": "revoked", "id": path.split("/")[4]}
        return original(method, path, query, body)
    sidecar.dispatch = dispatch
    grant = {"id": "renew-certs", "class": "credential_change", "scope": "renew TLS certificates",
             "tools": ["renew_certificates"], "per_day": 2}
    words = "you can renew the certificates on your own"
    result = probe(TOOL_CODE + '''
grant, words = %r, %r
o = owner("owner-1", message=words)
g = guest("guest-1", message=words)
c = guest("cron_job1_20300101_120000", "", message=words, platform="cron")
emit(asked=call("colonymind_self", {"operation": "grant", "grant": grant, "reason": words}, o),
     malformed=call("colonymind_self", {"operation": "grant", "grant": dict(grant, **{"class": "all"}), "reason": words}, o),
     no_reason=call("colonymind_self", {"operation": "grant", "grant": grant}, o),
     no_grant=call("colonymind_self", {"operation": "grant", "reason": words}, o),
     guest=call("colonymind_self", {"operation": "grant", "grant": grant, "reason": words}, g),
     cron=call("colonymind_self", {"operation": "grant", "grant": grant, "reason": words}, c),
     revoked=call("colonymind_self", {"operation": "revoke", "id": "renew-certs"}, o),
     guest_revoke=call("colonymind_self", {"operation": "revoke", "id": "renew-certs"}, g))
''' % (grant, words), home)
    # Filed as the owner's ask: its code is never the model's to show, the owner answers it in their own words.
    assert result["asked"]["status"] == "asked" and "Q4M" not in json.dumps(result["asked"])
    assert "their own yes" in result["asked"]["text"]
    assert "class is one of" in result["malformed"]["error"]          # an argument the model can correct
    assert "reason is required" in result["no_reason"]["error"] and "grant is required" in result["no_grant"]["error"]
    for name in ("guest", "cron", "guest_revoke"):
        assert "only the owner" in refused(result[name]), name
    assert result["revoked"]["status"] == "revoked"
    posted = [call["json"] for call in sidecar.calls("/v1/mind/grants", "POST")]
    assert len(posted) == 2 and posted[0]["grant"] == grant and posted[0]["reason"] == words
    assert posted[0]["by"] == "owner" and posted[0]["message"] == words and posted[0]["session_id"] == "owner-1"
    assert [call["json"] for call in sidecar.calls("/v1/mind/grants/renew-certs/revoke", "POST")] == [
        {"by": "owner", "contact_id": posted[0]["contact_id"]}]
    worker = probe(TOOL_CODE + '''
o = owner("owner-1", message=%r)
emit(grant=call("colonymind_self", {"operation": "grant", "grant": %r, "reason": "x"}, o),
     revoke=call("colonymind_self", {"operation": "revoke", "id": "renew-certs"}, o))
''' % (words, grant), home, env=worker_env(home))
    assert "only the owner" in refused(worker["grant"]) and "only the owner" in refused(worker["revoke"])
    assert len(sidecar.calls("/v1/mind/grants", "POST")) == 2
    assert json.dumps(result["asked"])
