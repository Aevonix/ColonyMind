from types import SimpleNamespace
import sys
import pytest
from test_hands_definition_dispatch import setup, create
from test_hands_definition_pickup import enqueue, pickup
from hands_fake_hermes import Parent
from colonymind_hermes.capture import SessionMap, SessionInfo

# The borrowed fixture `setup` builds the sidecar's own controller.
pytestmark = pytest.mark.requires_sidecar


def sessions_for(runtime, tmp_path, monkeypatch, *, chat_type='dm'):
    import colonymind_hermes.capture as capture
    monkeypatch.setattr(capture, 'session_env', lambda: ('', '', '', ''))
    handles = {'telegram': ['owner-handle']}
    settings = SimpleNamespace(home=tmp_path/'standing-home', outbox_path=tmp_path/'outbox',
        owner_handles=lambda: handles, owner_contact_id=lambda: 'owner-contact')
    sessions = SessionMap(settings, SimpleNamespace(resolve_contact=lambda *a, **k: None))
    sessions._sessions['owner-session'] = SessionInfo(session_id='owner-session', platform='telegram',
        sender_id='owner-handle', chat_id='group-chat' if chat_type=='group' else 'owner-chat', chat_type=chat_type)
    runtime.sessions = sessions
    runtime.owner_check = lambda sid: sessions.standing(sid)[0] in {'owner','mind','system'}
    runtime.config['platforms'] = ['telegram']
    runtime.native.platform = 'telegram'
    original_context = runtime.native.context
    def context(parent):
        captured = original_context(parent)
        captured['origin_ref']['chat'] = sessions.get(parent.session_id).chat_id
        captured['chat_type'] = chat_type
        return captured
    monkeypatch.setattr(runtime.native, 'context', context)
    return sessions, handles


def test_group_parent_must_not_dispatch(setup, tmp_path, monkeypatch):
    runtime, native, client, c, http, _ = setup
    sessions, handles = sessions_for(runtime, tmp_path, monkeypatch, chat_type='group')
    parent = Parent()
    runtime.remember_parent(parent)
    queued = enqueue(setup)
    assert sessions.chat(parent.session_id) == ('group-chat', 'group')
    pickup(runtime)
    print('group standing:', sessions.standing(parent.session_id), 'native calls:', len(native.calls))
    assert not native.calls, 'Owner-authored group turn was selected as the automatic app-task parent'
    assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'


def test_rotated_parent_must_not_dispatch(setup):
    runtime, native, *_ = setup
    parent = Parent()
    runtime.remember_parent(parent)
    parent.session_id = 'guest-session'
    queued = enqueue(setup)
    pickup(runtime)
    print('actual parent:', parent.session_id, 'native calls:', len(native.calls))
    assert not native.calls, 'Cached old owner SID authorized a parent whose actual SID is now a guest'
    assert 'owner-session' not in runtime.parents


def test_revoked_owner_handles_must_not_dispatch(setup, tmp_path, monkeypatch):
    runtime, native, *_ = setup
    sessions, handles = sessions_for(runtime, tmp_path, monkeypatch)
    parent = Parent()
    runtime.remember_parent(parent)
    handles.clear()
    queued = enqueue(setup)
    pickup(runtime)
    print('revoked handle standing:', sessions.standing(parent.session_id), 'native calls:', len(native.calls))
    assert not native.calls, 'Removed owner handle retained cached owner authority for automatic pickup'
    assert runtime.client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'


def test_narrow_hands_key_must_not_indirectly_dispatch(setup):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.credentials import CredentialTable, secret_digest
    from colonymind.api.routers import hands
    runtime, native, _, c, *_ = setup
    app = FastAPI()
    app.include_router(hands.runtime_router)
    app.include_router(hands.router)
    app.state.hands_store = c.store
    app.add_middleware(ApiKeyMiddleware, api_key='test-instance-key',
        credentials=CredentialTable.from_config({'narrow-hands': {
            'owner': [], 'contacts':'own', 'access':['hands'], 'sha256':secret_digest('narrow-secret')}}))
    parent = Parent()
    runtime.remember_parent(parent)
    with TestClient(app, headers={'Authorization':'Bearer narrow-secret'}) as narrow:
        definition = narrow.post('/v1/hands/definitions', json={'name':'Narrow credential task', 'tools':['read_file']},
            headers={'Idempotency-Key':'narrow-definition'})
        assert definition.status_code == 200
        queued = narrow.post('/v1/hands/definitions/'+definition.json()['id']+'/tasks',
            json={'brief':'Read local sources without an owner turn'}, headers={'Idempotency-Key':'narrow-enqueue'})
        print('narrow definition:', definition.status_code, 'narrow enqueue:',queued.status_code)
        if queued.status_code == 200:
            pickup(runtime)
        print('narrow-caused native calls:',len(native.calls))
        assert not native.calls, 'Credential with owner=[] and access=[hands] indirectly caused native dispatch'
        assert queued.json()['dispatch_state'] == 'awaiting_owner_start'
        pending = narrow.get('/v1/hands/requests/pending').json()['tasks']
        assert pending[0]['owner_started'] == 0
        assert pending[0]['initiating_credential'] == 'narrow-hands'


def test_standing_revoked_after_plan_must_not_dispatch(setup, monkeypatch):
    runtime, native, *_ = setup
    standing = ['owner']
    runtime.sessions = SimpleNamespace(standing=lambda sid: (standing[0], 'sms'))
    runtime.owner_check = lambda sid: runtime.sessions.standing(sid)[0] in {'owner', 'mind', 'system'}
    parent=Parent()
    runtime.remember_parent(parent)
    queued=enqueue(setup)
    route=runtime._route
    def revoke(*args):
        credentials=route(*args)
        standing[0]='mind'
        return credentials
    monkeypatch.setattr(runtime,'_route',revoke)
    pickup(runtime)
    print('post-plan standing:', runtime.parent_standing(parent.session_id), 'native calls:',len(native.calls))
    assert not native.calls, 'Final bind only checks internal authorization, allowing lost owner standing'
    assert runtime.client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'


@pytest.mark.parametrize('chat_type', ['group', 'shared', 'guest', 'dm'])
def test_real_front_and_native_context_allow_group_pickup(setup, tmp_path, monkeypatch, chat_type):
    import importlib
    import gateway.session_context as session_context
    import tools.delegate_tool_dispatch as native_dispatch
    runtime, native, _, c, http, package = setup
    sessions, handles = sessions_for(runtime, tmp_path, monkeypatch, chat_type=chat_type)
    chat_id = sessions.get('owner-session').chat_id
    parent=Parent()
    env={'HERMES_SESSION_PLATFORM':'telegram','HERMES_SESSION_USER_ID':'owner-handle',
        'HERMES_SESSION_CHAT_ID':chat_id,'HERMES_SESSION_CHAT_TYPE':chat_type,
        'HERMES_SESSION_ID':'owner-session','HERMES_SESSION_MESSAGE_ID':'real-message'}
    monkeypatch.setattr(session_context,'get_session_env',lambda key, default='': env.get(key,default))
    monkeypatch.setattr(native_dispatch,'_capture_origin', lambda: ('','',None,None,False))
    front_module=importlib.import_module(package.__name__+'.front')
    front=front_module.FrontGuard(runtime, SimpleNamespace(), start_threads=False)
    event=SimpleNamespace(source=SimpleNamespace(platform='telegram',user_id='owner-handle',chat_id=chat_id,chat_type=chat_type),
        message_id='real-message',text='Owner says hello in group',session_id=parent.session_id,session_key='group-key')
    front.pre_gateway_dispatch(event=event)
    front._turn(parent.session_id,parent._current_turn_id,event.text,platform='telegram',sender_id='owner-handle',
        chat_id=chat_id,message_id='real-message',session_key='group-key')
    monkeypatch.setitem(sys.modules,'colonymind_hands_front',SimpleNamespace(origin_ref=front.origin_ref))
    dispatch_module=importlib.import_module(package.__name__+'.dispatch')
    monkeypatch.setattr(native,'context',lambda agent: dispatch_module.NativeHermes.context(native,agent))
    runtime.remember_parent(parent)
    queued=enqueue(setup)
    pickup(runtime)
    print('real captured group origin:',runtime.contexts[parent.session_id], 'native calls:',len(native.calls))
    assert len(native.calls) == (1 if chat_type == 'dm' else 0)


@pytest.mark.parametrize('change', ['handles', 'destination', 'chat_type', 'identity', 'mind', 'system'])
def test_live_authority_and_destination_checked_after_plan(setup, tmp_path, monkeypatch, change):
    runtime, native, client, c, http, _ = setup
    sessions, handles = sessions_for(runtime, tmp_path, monkeypatch)
    parent = Parent()
    runtime.remember_parent(parent)
    queued = enqueue(setup)
    assert sessions.standing(parent.session_id)[0] == 'owner'  # populate the old cache
    route = runtime._route
    def revoke(*args):
        result = route(*args)
        info = sessions.get(parent.session_id)
        if change == 'handles':
            handles.clear()
        elif change == 'destination':
            info.chat_id = 'another-chat'
        elif change == 'chat_type':
            info.chat_type = 'group'
        elif change == 'identity':
            parent.session_id = 'guest-session'
        else:
            info.inherited_standing = change
        return result
    monkeypatch.setattr(runtime, '_route', revoke)
    pickup(runtime)
    assert not native.calls
    assert client.get('/v1/hands/tasks/' + queued['id'])['dispatch_state'] == 'waiting_for_main_session'
    assert (http.portal.call(c.one, 'SELECT dispatched FROM definition_task_queue WHERE task_id=?',
                            (queued['id'],)))['dispatched'] == 0
    # Restoring a verified destination makes the same request retryable.
    monkeypatch.setattr(runtime, '_route', route)
    handles['telegram'] = ['owner-handle']
    parent.session_id = 'owner-session'
    info = sessions.get(parent.session_id)
    info.chat_id, info.chat_type, info.inherited_standing = 'owner-chat', 'dm', ''
    pickup(runtime)
    assert len(native.calls) == 1


def test_compressed_owner_is_recaptured_under_current_id(setup, tmp_path, monkeypatch):
    runtime, native, client, c, http, _ = setup
    sessions, _ = sessions_for(runtime, tmp_path, monkeypatch)
    parent = Parent()
    runtime.remember_parent(parent)
    parent.session_id = 'owner-compressed'
    sessions._sessions[parent.session_id] = SessionInfo(session_id=parent.session_id, platform='telegram',
        sender_id='owner-handle', chat_id='owner-chat', chat_type='dm', parent_session_id='owner-session')
    queued = enqueue(setup)
    pickup(runtime)
    assert len(native.calls) == 1 and 'owner-session' not in runtime.parents
    hand, = runtime.registry.all()
    assert hand.parent_session_id == parent.session_id and hand.subagent_id
    assert runtime.parents[parent.session_id] is parent


def test_companion_hands_owner_platform_dispatches_without_broader_access(setup):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.credentials import CredentialTable, secret_digest
    from colonymind.api.routers import hands
    runtime, native, client, c, *_ = setup
    app = FastAPI()
    app.include_router(hands.runtime_router)
    app.include_router(hands.router)
    app.state.hands_store = c.store
    app.add_middleware(ApiKeyMiddleware, api_key='test-instance-key',
        credentials=CredentialTable.from_config({'companion-hands': {
            'owner': ['companion'], 'contacts':'own', 'access':['hands'], 'sha256':secret_digest('owner-hands')}}))
    parent = Parent()
    runtime.remember_parent(parent)
    with TestClient(app, headers={'Authorization':'Bearer owner-hands'}) as companion:
        definition = companion.post('/v1/hands/definitions', json={'name':'Owner start', 'tools':['read_file']},
                                    headers={'Idempotency-Key':'definition'}).json()
        queued = companion.post('/v1/hands/definitions/' + definition['id'] + '/tasks', json={'brief':'Inspect'},
                                headers={'Idempotency-Key':'owner-start'}).json()
        assert queued['dispatch_state'] == 'queued'
        pickup(runtime)
        assert len(native.calls) == 1
        pending = client.get('/v1/hands/requests/pending')['tasks']
        assert not pending
        for action in ('claim', 'bind', 'release'):
            assert companion.post('/v1/hands/definitions/queue/' + action, json={},
                                  headers={'Idempotency-Key':action}).status_code == 403
        for route in ('/v1/mind/state', '/v1/host/health', '/docs', '/openapi.json', '/mcp'):
            assert companion.get(route).status_code == 403
        assert companion.post('/v1/hands/questions/x/answer', json={},
                              headers={'Idempotency-Key':'answer'}).status_code == 422
