"""The answer tripwire through the stock ``pre_tool_call`` dispatch (docs/MIND.md, Asks: console and API answers).

In every session (the owner's own lane, a guest's, a mind run), a tool call that reads the instance key or names
the routes that record the owner's answers is reported to the sidecar as ``answer_forgery_attempt``, which tells the
owner at most once a day. In the owner's own lane it blocks nothing (an approval never stands in the owner's way);
in a guest's session and a mind run the call is also refused, and there a read of any file in the key's directory
counts too. A shell spells a name in more ways than any rule sees, so this is no boundary on its own: a guest has no
shell (docs/HERMES-ADAPTER.md, Owner and guest), a mind run's toolsets have none, and ``colonymind doctor`` names the
lanes that do. One report per signal and session in ten minutes, however often the call repeats.
"""

from conftest import probe, worker_env

TRIPWIRE = "/v1/mind/tripwire"
CHECKS = '''
from hermes_cli.plugins import get_pre_tool_call_directive
def session(name, sender):
    invoke_hook("pre_llm_call", session_id=name, task_id="t", turn_id="turn", user_message="hi",
                conversation_history=[], is_first_turn=True, model="m", platform="telegram",
                parent_session_id="", sender_id=sender)
    return name
def check(tool, args, name=""):
    action, message = get_pre_tool_call_directive(tool, args, session_id=name, task_id="t",
                                                  tool_call_id="c", turn_id="turn", api_request_id="r")
    return {"action": action, "message": message}
'''


def _reports(sidecar) -> list[dict]:
    return [call["json"] for call in sidecar.calls(TRIPWIRE, "POST")]


def test_the_owner_lane_reading_the_key_or_naming_the_answer_routes_is_reported_and_runs(home, sidecar):
    sidecar.mind_routes = True
    result = probe(CHECKS + '''
o, spelled, routed = session("owner-1", "1001"), session("owner-2", "1001"), session("owner-3", "1001")
explicit = session("owner-4", "1001")
emit(read=check("read_file", {"path": "~/.colonymind/api.key"}, o),
     again=check("read_file", {"path": "/home/me/.colonymind/api.key"}, o),
     spelled=check("terminal", {"command": "cat ~/.colonymind/api'.'key"}, spelled),
     env=check("execute_code", {"code": "import os; print(os.environ['COLONYMIND_API_KEY'])"}, spelled),
     routed=check("terminal", {"command": "curl -X POST http://127.0.0.1:7850/v1/mind/answers -d @reply.json"},
                  routed),
     yes=check("terminal", {"command": "curl -X POST localhost:7850/v1/mind/asks/K7F/yes"}, routed),
     explicit=check("terminal", {"command": "curl -X POST localhost:7850/v1/mind/permit-asks/a-1/answer"},
                    explicit),
     plain=check("terminal", {"command": "ls ~/projects"}, o),
     other=check("read_file", {"path": "/tmp/notes.txt"}, o))
''', home)
    assert all(verdict["action"] is None for verdict in result.values()), result      # a tripwire blocks nothing
    reports = _reports(sidecar)
    assert [(r["tool"], r["named"], r["run"], r["session_id"]) for r in reports] == [
        ("read_file", ["api.key"], "owner", "owner-1"),
        ("terminal", ["api.key"], "owner", "owner-2"),
        ("terminal", ["answer_route"], "owner", "owner-3"),
        ("terminal", ["answer_route"], "owner", "owner-4"),
    ]
    assert all(r["kind"] == "answer_forgery_attempt" and r["refused"] is False and
               set(r) == {"kind", "tool", "named", "run", "session_id", "refused"}
               for r in reports)                                    # the call's arguments never leave the plugin


def test_a_guest_session_and_a_mind_run_are_reported_and_refused(home, sidecar):
    sidecar.mind_routes = True
    guest = probe(CHECKS + '''
import os
g = session("guest-1", "2002")
emit(read=check("read_file", {"path": "~/.colonymind/api.key"}, g),
     route=check("web_extract", {"urls": ["http://127.0.0.1:7850/v1/mind/answers"]}, g))
''', home)
    for name in ("read", "route"):
        verdict = guest[name]
        assert verdict["action"] == "block" and verdict["message"].endswith("(retry: false)"), (name, verdict)
        assert "not available in this conversation" in verdict["message"], (name, verdict)
    # A read that reaches the key file however it is spelled (a link to it, a content search over a directory
    # that holds it) counts in a mind run; the owner-authored files beside it stay readable, as ever.
    (home.workspace / "notes.txt").symlink_to(home.key_file)
    mind = probe(CHECKS + '''
import os
key_file, workspace = %r, %r
key_dir = os.path.dirname(key_file)
emit(read=check("read_file", {"path": "~/.colonymind/api.key"}),
     spelled=check("read_file", {"path": "~/.colonymind/API" + ".KEY"}),
     linked=check("read_file", {"path": os.path.join(workspace, "notes.txt")}),
     directory=check("search_files", {"pattern": ".", "path": key_dir, "target": "content"}),
     above=check("search_files", {"pattern": "sk-", "path": os.path.dirname(key_dir)}),
     names_only=check("search_files", {"pattern": "*.yaml", "path": key_dir, "target": "files"}),
     globbed=check("search_files", {"pattern": "autonomy", "path": key_dir, "file_glob": "colonymind.yaml"}),
     globbed_key=check("search_files", {"pattern": "x", "path": key_dir, "file_glob": "*.key"}),
     beside=check("read_file", {"path": os.path.join(key_dir, "colonymind.yaml")}),
     route=check("web_extract", {"urls": ["http://127.0.0.1:7850/v1/mind/asks/K7F/yes"]}),
     plain=check("read_file", {"path": "/tmp/notes.txt"}),
     write=check("write_file", {"path": "~/.colonymind/api.key", "content": "x"}))
''' % (str(home.key_file), str(home.workspace)), home, env=worker_env(home))
    for name in ("read", "spelled", "linked", "directory", "above", "globbed_key", "route", "write"):
        assert mind[name]["action"] == "block", (name, mind[name])
    assert "api.key" in mind["write"]["message"]
    for name in ("names_only", "globbed", "beside", "plain"):
        assert mind[name]["action"] is None, (name, mind[name])
    runs = [(r["tool"], r["run"]) for r in _reports(sidecar)]
    assert ("read_file", "guest") in runs and ("read_file", "mind") in runs    # once per signal and session
    assert all(r["refused"] is True for r in _reports(sidecar))                 # and the sidecar's note says so
    assert ("web_extract", "guest") in runs and ("web_extract", "mind") in runs


def test_the_tripwire_never_stands_in_the_way_when_the_sidecar_is_down(home, sidecar):
    home.write_config(**{**home.config, "plugins": {**home.config["plugins"], "colonymind": {
        "sidecar_url": "http://127.0.0.1:1", "key_file": str(home.key_file)}}})
    result = probe(CHECKS + '''
o = session("owner-1", "1001")
emit(read=check("read_file", {"path": "~/.colonymind/api.key"}, o))
''', home)
    assert result["read"]["action"] is None
    assert _reports(sidecar) == []


def test_a_mind_run_search_the_host_would_run_over_the_key_is_refused_however_it_is_spelled(home, sidecar):
    """Stock ``search_files`` runs ripgrep (else grep) with the call's ``file_glob`` as one ``--glob`` over the root its
    ``path`` names, and searches each part of a path that names nothing (split at commas, else at whitespace). Ripgrep
    reads a glob's ``{a,b}`` alternatives, ``**/``, escapes and ``[^...]`` classes, and searches a file named as the
    root whatever the glob says. Each spelling below runs the stock search over a synthetic key as well: every one that
    finds it there (on this host's engine) is refused in a mind run, and so is every one ripgrep would run over it. A
    names-only search, a glob that cannot match the key and a search elsewhere still run."""
    sidecar.mind_routes = True
    result = probe(CHECKS + '''
import os
from tools.file_tools import search_tool, _get_file_ops
key_file = %r
key_dir = os.path.dirname(key_file)
other = os.path.join(os.path.dirname(key_dir), "notes")
os.makedirs(other, exist_ok=True)
open(os.path.join(other, "n.txt"), "w").write("nothing here")
secret = open(key_file).read().strip()
# Hermes expands tilde in the terminal backend, whose profile home can differ
# from the Python process's home.
home_dir = "~/" + os.path.relpath(key_dir, _get_file_ops("t")._expand_path("~"))
spellings = {
    "brace": {"path": key_dir, "file_glob": "*.{key,txt}"},
    "nested_brace": {"path": key_dir, "file_glob": "{*.txt,{a,api}.k*}"},
    "any_depth": {"path": key_dir, "file_glob": "**/api.key"},
    "escaped": {"path": key_dir, "file_glob": "api\\\\.key"},
    "negated_class": {"path": key_dir, "file_glob": "api.[^x]ey"},
    "other_case": {"path": key_dir, "file_glob": "*.KEY"},
    "roots_comma": {"path": other + "," + key_dir},
    "roots_space": {"path": other + " " + key_dir},
    "roots_home": {"path": other + ", " + home_dir},
    "the_file": {"path": key_file, "file_glob": "*.txt"},
    "the_file_excluded": {"path": key_file, "file_glob": "!api.key"},
}
allowed = {
    "names_only": {"path": key_dir + "," + other, "target": "files", "pattern": "*"},
    "other_glob": {"path": key_dir, "file_glob": "*.{yaml,txt}"},
    "elsewhere": {"path": other},
}
found, verdicts = {}, {}
ops = _get_file_ops("t")
backend = "rg" if ops._has_command("rg") else "grep"
searches = []
original_search = ops._search_with_rg
def observed_search(*args, **kwargs):
    searches.append("rg")
    return original_search(*args, **kwargs)
ops._search_with_rg = observed_search
for name, args in {**spellings, **allowed}.items():
    pattern = args.pop("pattern", secret[:8])
    found[name] = secret in search_tool(pattern=pattern, task_id="t", **args)      # the stock search itself
    verdicts[name] = check("search_files", {"pattern": pattern, **args})
emit(backend=backend, searches=searches, found=found, verdicts=verdicts,
     spellings=sorted(spellings), allowed=sorted(allowed))
''' % str(home.key_file), home, env=worker_env(home))
    assert result["backend"] == "rg" and result["searches"], (
        "Tripwire qualification requires ripgrep (rg) in the Hermes terminal backend; "
        f"selected {result['backend']!r}, observed searches={result['searches']}. Install ripgrep."
    )
    found, verdicts = result["found"], result["verdicts"]
    assert found["roots_comma"] and found["roots_space"] and found["roots_home"], found   # the host searches them
    for name, reached in found.items():
        if reached:
            assert verdicts[name]["action"] == "block", (name, verdicts[name])
    for name in result["spellings"]:
        assert verdicts[name]["action"] == "block", (name, verdicts[name])
        assert "api.key" in verdicts[name]["message"], (name, verdicts[name])
    for name in result["allowed"]:
        assert not found[name] and verdicts[name]["action"] is None, (name, found[name], verdicts[name])
