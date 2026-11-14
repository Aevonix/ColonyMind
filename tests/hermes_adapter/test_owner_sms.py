"""SMS owner recognition follows deployment settings even inside an already cached session."""

import yaml

from conftest import OWNER, probe

# Deliberately fictional placeholder numbers.
SMS = "+15550000011"
OTHER = "+15550000012"


def test_local_sms_configuration_env_precedence_and_invalid_values(home):
    result = probe('''
from colonymind_hermes.client import load_settings
import yaml
settings = load_settings()
path = settings.home / "colonymind.yaml"
raw = yaml.safe_load(path.read_text())
raw["owner"] = {"sms_handle": "+15550000011"}
path.write_text(yaml.safe_dump(raw))
initial = settings.owner_sms_handle()
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = "+15550000012"
override = settings.owner_sms_handle()
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = ""
disabled = settings.owner_sms_handle()
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = "not-a-phone"
invalid = settings.owner_sms_handle()
emit(initial=initial, override=override, disabled=disabled, invalid=invalid)
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"initial": SMS, "override": OTHER, "disabled": "", "invalid": ""}


def test_owner_sms_session_and_contact_caches_revoke_when_config_changes(home, sidecar):
    config = yaml.safe_load((home.instance / "colonymind.yaml").read_text())
    config["owner"] = {"contact_id": OWNER, "sms_handle": SMS}
    (home.instance / "colonymind.yaml").write_text(yaml.safe_dump(config))
    result = probe('''
from colonymind_hermes.capture import SessionMap, branch_owner
from colonymind_hermes.client import load_settings, ColonyMindClient
settings = load_settings()
sessions = SessionMap(settings, ColonyMindClient(settings))
info = sessions.observe(session_id="sms-session", platform="sms", sender_id="+15550000011")
initial = (sessions.is_owner(info.session_id), sessions.contact_id(info.session_id))
wrong = sessions.sender_is_owner("sms", "+15550000012")
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = "+15550000012"
changed = (sessions.is_owner(info.session_id), sessions.contact_id(info.session_id),
           branch_owner(sessions, info.session_id) is not None)
new_owner = sessions.sender_is_owner("sms", "+15550000012")
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = ""
removed = sessions.sender_is_owner("sms", "+15550000012")
emit(initial=initial, wrong=wrong, changed=changed, new_owner=new_owner, removed=removed)
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result["changed"][1] != OWNER
    assert result["changed"][0] is False and result["changed"][2] is False
    assert {key: value for key, value in result.items() if key != "changed"} == {
        "initial": [True, OWNER], "wrong": False, "new_owner": True, "removed": False}


def test_preexisting_explicit_sms_contact_binding_still_resolves(home, sidecar):
    sidecar.contacts[("sms", SMS)] = {"contact_id": OWNER, "display_name": "Owner"}
    result = probe('''
from colonymind_hermes.capture import SessionMap
from colonymind_hermes.client import load_settings, ColonyMindClient
settings = load_settings()
sessions = SessionMap(settings, ColonyMindClient(settings))
emit(owner=sessions.sender_is_owner("sms", "+15550000011"))
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result["owner"] is True


def test_sms_owner_narrative_requires_a_direct_chat_and_live_binding(home):
    result = probe('''
from colonymind_hermes.capture import SessionMap
from colonymind_hermes.client import load_settings, ColonyMindClient
import colonymind_hermes.capture as capture
settings = load_settings()
sessions = SessionMap(settings, ColonyMindClient(settings))
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = "+15550000011"
capture.session_env = lambda: ("sms", "+15550000011", "+15550000011", "dm")
direct = sessions.owner_only({"session_id": "sms-session"})
standing = sessions.standing("sms-session")
capture.session_env = lambda: ("sms", "+15550000011", "group-placeholder", "group")
group = sessions.owner_only({"session_id": "sms-session"})
os.environ["COLONYMIND_OWNER_SMS_HANDLE"] = ""
capture.session_env = lambda: ("sms", "+15550000011", "+15550000011", "dm")
revoked = sessions.standing("sms-session")
emit(direct=direct, standing=standing, group=group, revoked=revoked)
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result == {"direct": True, "standing": ["owner", "sms"], "group": False,
                      "revoked": ["guest", "sms"]}


def test_sms_resolution_preserves_the_callers_timeout(home):
    result = probe('''
from colonymind_hermes.capture import SessionMap
from colonymind_hermes.client import load_settings, ColonyMindClient
settings = load_settings()
client = ColonyMindClient(settings)
calls = []
def resolve(*args, **kwargs):
    calls.append(kwargs)
    return None
client.resolve_contact = resolve
sessions = SessionMap(settings, client)
sessions.observe(session_id="sms-session", platform="sms", sender_id="+15550000012")
sessions.is_owner("sms-session", timeout=0.025)
sessions.contact_id("sms-session", timeout=0.030)
emit(calls=calls)
''', home, env={"COLONYMIND_BODY_THREAD": "0"})
    assert result["calls"] == [{"timeout": 0.025}, {"create": True, "timeout": 0.030}]
