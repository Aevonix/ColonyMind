"""Only the general plugin captures turns; memory lifecycle still confirms context."""
import json

import pytest

from test_colonymind_memory_provider import (
    provider_mod, _instance, _FakeHttpx, _install_session_context,
)


@pytest.mark.parametrize('general_active', [False, True])
@pytest.mark.parametrize('writer_setting', ['auto', 'enabled'])
def test_provider_never_writes_turns(tmp_path, monkeypatch, provider_mod, general_active, writer_setting):
    monkeypatch.setattr(provider_mod, '_active_hermes_home', lambda: tmp_path)
    (tmp_path / 'config.yaml').write_text(json.dumps({
        'plugins': {'enabled': ['colonymind'] if general_active else []},
        'memory': {'provider': 'colonymind-memory', 'config': {'turn_writer': writer_setting}},
    }))
    set_turn = _install_session_context(monkeypatch)
    set_turn(platform='cli', sender='', chat='')
    fake = _FakeHttpx(routes={('POST', '/v1/host/turns/sync'): {'accepted': True}})
    monkeypatch.setattr(provider_mod, 'httpx', fake)
    provider = provider_mod.ColonyMindMemoryProvider(config={
        'url': 'http://sidecar.test', 'contact_id': 'cid-base', 'turn_writer': writer_setting})
    confirmed = []
    monkeypatch.setattr(provider, '_confirm_shown', lambda session, text: confirmed.append((session, text)))
    provider.sync_turn('fixture question', 'fixture answer', session_id='s', turn_id='t')
    provider.on_session_end([{'role': 'user', 'content': 'fixture question'},
                             {'role': 'assistant', 'content': 'fixture answer'}])
    provider.shutdown()
    assert fake.requests == []
    assert confirmed == [('s', 'fixture question')]
    assert provider.get_diagnostics()['turn_writer'] == 'read-only'
