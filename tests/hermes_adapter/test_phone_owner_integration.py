"""Owner phone turns through pinned Hermes context and hands' native delivery seam.

These fresh-process probes use no sockets, model calls or live profiles.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("COLONYMIND_HERMES_INTEGRATION") != "1",
    reason="integrator: set COLONYMIND_HERMES_INTEGRATION=1 and HERMES_SOURCE_ROOT",
)


@pytest.mark.parametrize("platform", ["sms", "rcs"])
def test_signed_owner_context_revokes_and_preserves_native_delivery(tmp_path, platform):
    source = os.environ.get("HERMES_SOURCE_ROOT")
    assert source and (Path(source) / "gateway/session_context.py").is_file()
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "COLONYMIND_"))}
    environment.update(HERMES_SOURCE_ROOT=source, HERMES_HOME=str(tmp_path / "hermes"),
                       COLONYMIND_HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), platform],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=35)
    assert result.returncode == 0, result.stdout + result.stderr


def run(platform):
    import importlib.util
    from types import SimpleNamespace
    import time

    # Select this checkout even when the shared interpreter has another editable install.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, os.environ["HERMES_SOURCE_ROOT"])
    spec = importlib.util.spec_from_file_location("colonymind_hermes", root / "plugins/hermes-plugin/__init__.py",
                                                  submodule_search_locations=[str(root / "plugins/hermes-plugin")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.session import SessionSource, build_session_key
    from gateway.session_context import set_session_vars
    from colonymind_hermes.capture import SessionMap, branch_owner
    from colonymind_hermes.client import Settings
    from colonymind_hermes.sms_ingress import owner_phone_sender
    from hands_fake_hermes import load_hands

    owner, other = "+15550000011", "+15550000012"
    home = Path(os.environ["COLONYMIND_HOME"])
    (home / "colonymind.yaml").write_text(f"owner:\n  contact_id: owner-contact\n  {platform}_handle: '{owner}'\n")
    settings = Settings("http://unused.invalid", home / "api.key", "fixture", home,
                        Path(os.environ["HERMES_HOME"]), home / "outbox.db")
    resolutions = []
    def resolve(*args, **kwargs):
        resolutions.append(kwargs)
        return None
    sessions = SessionMap(settings, SimpleNamespace(resolve_contact=resolve))
    now = time.time()
    message = {"senderId": owner, "chatId": owner, "isGroup": False, "body": "approve",
               "transportAttestation": {"verified": True, "method": "ecdsa-p256-sha256", "keyId": "fixture",
                   "receivedAt": now, "eventTime": now, "nonce": "fixture-nonce-000001", "bodySha256": "a" * 64}}
    sender = owner_phone_sender(message, platform=platform, bridge_url="http://127.0.0.1:9876", settings=settings)
    assert sender == owner
    if platform == "rcs":
        platform_registry.register(PlatformEntry("rcs", "Pixel RCS", lambda config: None, lambda: True))
    source = SessionSource(platform=Platform(platform), chat_id=owner, chat_type="dm", user_id=sender)
    event = MessageEvent(text=message["body"], source=source, message_id="signed-1")
    key = build_session_key(source)
    set_session_vars(platform=platform, user_id=sender, chat_id=owner, chat_type="dm",
                     session_key=key, session_id="owner-session", message_id=event.message_id, async_delivery=True)
    info = sessions.observe(session_id="owner-session", platform=platform, sender_id=sender)
    assert sessions.is_owner(info.session_id) is True
    assert sessions.contact_id(info.session_id) == "owner-contact"
    assert sessions.standing(info.session_id) == ("owner", platform)
    assert sessions.owner_only({"session_id": info.session_id})
    assert branch_owner(sessions, info.session_id) is info

    hands = load_hands()
    native = importlib.import_module(hands.__name__ + ".dispatch").NativeHermes()
    context = native.context(SimpleNamespace(session_id=info.session_id, _current_turn_id="turn-1"))
    assert context["platform"] == platform
    assert native.push_supported(context), "RCS needs the same detached completion consumer as SMS"

    variable = f"COLONYMIND_OWNER_{platform.upper()}_HANDLE"
    os.environ[variable] = other
    assert sessions.is_owner(info.session_id, timeout=0.025) is False
    assert resolutions[-1] == {"timeout": 0.025}
    assert sessions.contact_id(info.session_id) is None
    assert branch_owner(sessions, info.session_id) is None
    assert sessions.standing(info.session_id) == ("guest", platform)
    assert sessions.owner_only({"session_id": info.session_id}) is False
    assert sessions.sender_is_owner(platform, other)
    os.environ[variable] = ""
    assert not sessions.sender_is_owner(platform, other)
    assert owner_phone_sender(message, platform=platform, bridge_url="http://localhost", settings=settings) is None
    os.environ[variable] = owner
    message["transportAttestation"]["verified"] = False
    assert owner_phone_sender(message, platform=platform, bridge_url="http://localhost", settings=settings) is None
    guest = "rcs-unverified:fixture" if platform == "rcs" else "sms-unverified:fixture"
    assert not sessions.sender_is_owner(platform, guest)
    set_session_vars(platform=platform, user_id=owner, chat_id="group", chat_type="group")
    assert sessions.owner_only({"session_id": info.session_id}) is False


if __name__ == "__main__":
    run(sys.argv[1])
