"""A message parked behind a busy session runs pre_gateway_dispatch once, as an idle one does (busy_ingress)."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import time
from types import SimpleNamespace as NS

import pytest

from colonymind_hermes import busy_ingress


class Runner:
    """The gateway's dispatch runner contract: None drops, a different event carries rewritten text."""

    def __init__(self, result='allow', adapters=None, multiplex=False):
        self.result, self.calls = result, []
        self.adapters = adapters if adapters is not None else {}
        self.multiplex = multiplex

    def _multiplex_on(self):
        return self.multiplex

    def _hm_pre_gateway_dispatch_hook(self, event, source):
        self.calls.append((event, source, getattr(event, busy_ingress.BUSY, False)))
        if self.result == 'skip':
            return None
        if self.result == 'rewrite':
            return dataclasses.replace(event, text='sealed words')
        if self.result == 'raise':
            raise RuntimeError('hook runner down')
        return event


class AsyncRunner(Runner):
    async def _hm_pre_gateway_dispatch_hook(self, event, source):
        await asyncio.sleep(0)
        return super()._hm_pre_gateway_dispatch_hook(event, source)


@dataclasses.dataclass
class Event:
    text: str
    internal: bool = False
    command: bool = False
    source: object = dataclasses.field(default_factory=lambda: NS(platform='sms', user_id='owner', chat_id='c'))

    def is_command(self):
        return self.command


class Adapter:
    def __init__(self, handled=False):
        self.seen, self.handled = [], handled
        async def stock(event, key):
            self.seen.append((event, key, event.text))
            return self.handled
        self._busy_session_handler = stock

    def set_busy_session_handler(self, handler):
        self._busy_session_handler = handler


def run(adapter, event):
    return asyncio.run(adapter._busy_session_handler(event, 'route'))


@pytest.fixture(autouse=True)
def fresh_load_state(monkeypatch):
    """Each test starts as a fresh load of the module: nothing wrapped by it yet, not unloaded."""
    monkeypatch.setattr(busy_ingress, '_closed', False)
    monkeypatch.setattr(busy_ingress, '_installed', [])


def test_install_wraps_every_adapter_once():
    adapters = {'sms': Adapter(), 'whatsapp': Adapter()}
    runner = Runner(adapters=adapters)
    assert busy_ingress.install(runner) == 2
    assert busy_ingress.install(runner) == 0, 'the wrap is idempotent'
    adapters['rcs'] = Adapter()
    assert busy_ingress.install(runner) == 1, 'an adapter made later is wrapped on the next pass'


@pytest.mark.parametrize('runner_type', [Runner, AsyncRunner])
def test_parked_message_runs_the_chain_once_then_the_stock_handler(runner_type):
    adapter = Adapter()
    runner = runner_type(adapters={'sms': adapter})
    busy_ingress.install(runner)
    event = Event('Owner asks now')
    assert run(adapter, event) is False
    assert [(e, busy) for e, _, busy in runner.calls] == [(event, True)], 'the hooks can tell it is parked'
    assert adapter.seen == [(event, 'route', 'Owner asks now')]
    assert busy_ingress.seen(event)
    run(adapter, event)
    assert len(runner.calls) == 1, 'a second pass never runs the chain again'


@pytest.mark.parametrize('runner_type', [Runner, AsyncRunner])
def test_skip_drops_and_rewrite_changes_the_queued_event_in_place(runner_type):
    adapter = Adapter()
    busy_ingress.install(runner_type('skip', adapters={'sms': adapter}))
    assert run(adapter, Event('drop me')) is True
    assert adapter.seen == [], 'a dropped message never reaches the stock busy handling'

    adapter = Adapter()
    busy_ingress.install(runner_type('rewrite', adapters={'sms': adapter}))
    event = Event('my password is hunter2')
    run(adapter, event)
    assert event.text == 'sealed words'
    assert adapter.seen == [(event, 'route', 'sealed words')], 'the adapter queues the object it holds'


def test_internal_events_keep_the_stock_path_and_slash_text_is_words():
    adapter = Adapter(handled=True)
    runner = Runner(adapters={'sms': adapter})
    busy_ingress.install(runner)
    assert run(adapter, Event('completion', internal=True)) is True
    assert runner.calls == []
    # Hermes dispatches every resolvable command before the busy handler; what reaches it is words for the model,
    # a slash-prefixed path among them, and those words are sealed and heard like any other.
    path = Event('/tmp/config password: hunter2', command=True)
    run(adapter, path)
    assert [event for event, _, _ in runner.calls] == [path]
    assert len(adapter.seen) == 2


@pytest.mark.parametrize('runner_type', [Runner, AsyncRunner])
def test_a_failing_chain_still_queues_the_message(runner_type):
    adapter = Adapter()
    busy_ingress.install(runner_type('raise', adapters={'sms': adapter}))
    event = Event('Owner asks')
    assert run(adapter, event) is False
    assert adapter.seen == [(event, 'route', 'Owner asks')]
    assert busy_ingress.seen(event)


@pytest.mark.parametrize('cancel', [False, True])
def test_async_busy_hook_finishes_before_done_marker(cancel):
    async def check():
        entered, release = asyncio.Event(), asyncio.Event()
        adapter = Adapter()
        runner = Runner(adapters={'sms': adapter})

        async def hook(event, source):
            entered.set()
            await release.wait()
            event.text = 'sealed words'
            return event

        runner._hm_pre_gateway_dispatch_hook = hook
        busy_ingress.install(runner)
        event = Event('private words')
        task = asyncio.create_task(adapter._busy_session_handler(event, 'route'))
        await entered.wait()
        assert not busy_ingress.seen(event) and adapter.seen == []
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not busy_ingress.seen(event) and adapter.seen == []
        else:
            release.set()
            assert await task is False
            assert busy_ingress.seen(event)
            assert adapter.seen == [(event, 'route', 'sealed words')]
    asyncio.run(check())


def test_no_stock_handler_still_runs_the_chain():
    adapter = Adapter()
    adapter._busy_session_handler = None
    runner = Runner(adapters={'sms': adapter})
    busy_ingress.install(runner)
    assert run(adapter, Event('Owner asks')) is False
    assert len(runner.calls) == 1


def test_multiplexing_or_a_missing_runner_leaves_the_handler_alone(caplog, monkeypatch):
    monkeypatch.setattr(busy_ingress, '_warned', set())
    adapter = Adapter()
    stock = adapter._busy_session_handler
    with caplog.at_level(logging.WARNING):
        assert busy_ingress.install(Runner(adapters={'sms': adapter}, multiplex=True)) == 0
        assert busy_ingress.install(Runner(adapters={'sms': adapter}, multiplex=True)) == 0
        assert busy_ingress.install(NS(adapters={'sms': adapter})) == 0
    assert adapter._busy_session_handler is stock
    assert sum('multiplexing' in record.message for record in caplog.records) == 1, 'said once'
    assert any('no pre_gateway_dispatch runner' in record.message for record in caplog.records)


def test_unload_restores_each_handler_and_a_later_load_replaces_an_old_wrapper(monkeypatch):
    adapter = Adapter()
    stock = adapter._busy_session_handler
    runner = Runner(adapters={'sms': adapter})
    assert busy_ingress.install(runner) == 1
    assert busy_ingress.uninstall() == 1
    assert adapter._busy_session_handler is stock, 'unload gives the adapter its own handler back'
    assert busy_ingress.install(runner) == 0, 'nothing wraps again after unload'
    # A crashed unload leaves an earlier load's wrapper: the next load wraps the stock handler it holds.
    monkeypatch.setattr(busy_ingress, '_closed', False)
    busy_ingress.install(runner)
    left = adapter._busy_session_handler
    monkeypatch.setattr(busy_ingress, '_TOKEN', object())
    assert busy_ingress.install(runner) == 1
    assert adapter._busy_session_handler is not left
    assert adapter._busy_session_handler.__wrapped__ is stock
    event = Event('Owner asks')
    run(adapter, event)
    assert len(runner.calls) == 1, 'one chain run, not one per load'


def test_turn_platform_id_reads_the_running_agent_and_the_bound_parent(monkeypatch):
    gateway_run = pytest.importorskip('gateway.run')
    from agent.subagent_lifecycle import bind_subagent_parent
    opener = NS(session_id='owner-session', _current_turn_id='turn-1', _persist_user_message_platform_id='message-1')
    runner = NS(_running_agents={'route': opener})
    monkeypatch.setattr(gateway_run, '_gateway_runner_ref', lambda: runner)
    assert busy_ingress.turn_platform_id('owner-session', 'turn-1') == 'message-1'
    assert busy_ingress.turn_platform_id('owner-session', 'turn-2') is None
    rebuilt = NS(session_id='owner-session', _current_turn_id='turn-2', _persist_user_message_platform_id=None)
    with bind_subagent_parent(rebuilt):
        assert busy_ingress.turn_platform_id('owner-session', 'turn-2') == '', 'known, and empty: its own'


def test_register_hooks_and_keeper(monkeypatch):
    hooks, unloads = {}, []
    ctx = NS(register_hook=lambda name, fn: hooks.setdefault(name, []).append(fn), on_unload=unloads.append)
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't-1')
    busy_ingress.register(ctx)
    assert sorted(hooks) == ['pre_gateway_dispatch', 'pre_llm_call']
    assert len(unloads) == 1, 'a kanban worker runs no keeper, and its unload still restores handlers'
    adapter = Adapter()
    assert hooks['pre_gateway_dispatch'][0](event=Event('x'), gateway=Runner(adapters={'sms': adapter})) is None
    assert getattr(adapter._busy_session_handler, '_colonymind_busy_wrapped', False)
    monkeypatch.delenv('HERMES_KANBAN_TASK')
    started = []
    monkeypatch.setattr(busy_ingress.Keeper, 'start', lambda self: started.append(self))
    busy_ingress.register(ctx)
    assert len(started) == 1 and len(unloads) == 2
    unloads[-1]()
    assert started[0].stopped.is_set()
    assert adapter._busy_session_handler is not None and not getattr(
        adapter._busy_session_handler, '_colonymind_busy_wrapped', None), 'unload restored the stock handler'


def test_keeper_applies_the_wrap_until_stopped(monkeypatch):
    passes = []
    monkeypatch.setattr(busy_ingress, 'install', lambda gateway=None: passes.append(gateway))
    keeper = busy_ingress.Keeper(fast=0.01, settle=0.05, slow=0.02)
    keeper.start()
    time.sleep(0.2)
    keeper.stop()
    keeper.thread.join(1)
    assert len(passes) >= 3 and not keeper.thread.is_alive()


def test_real_adapter_busy_path_and_dispatch_runner(monkeypatch):
    """Hermes' own adapter parks the message and its own runner applies the hook results."""
    gateway_run = pytest.importorskip('gateway.run')
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from hermes_cli import plugins as hermes_plugins

    class Sms(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(), Platform.SMS)
            self._busy_text_debounce_seconds = 0

        async def connect(self):
            return True

        async def disconnect(self):
            pass

        async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
            raise AssertionError('nothing is sent')

        async def get_chat_info(self, chat_id):
            return {'chat_id': chat_id, 'chat_type': 'dm'}

    heard = []

    def hooks(name, **kwargs):
        if name != 'pre_gateway_dispatch':
            return []
        event = kwargs['event']
        heard.append((event.text, getattr(event, busy_ingress.BUSY, False)))
        return [{'action': 'rewrite', 'text': event.text.replace('hunter2', '[secret S-AAAAAAAA: password]')}]

    if inspect.iscoroutinefunction(gateway_run.GatewayRunner._hm_pre_gateway_dispatch_hook):
        async def async_hooks(name, **kwargs):
            return hooks(name, **kwargs)
        monkeypatch.setattr(hermes_plugins, 'ainvoke_hook', async_hooks)
    else:
        monkeypatch.setattr(hermes_plugins, 'invoke_hook', hooks)
    adapter = Sms()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.SMS: adapter}
    runner.config = NS(multiplex_profiles=False)

    async def stock_busy(event, key):
        return False   # busy text in queue mode: the adapter parks it

    adapter.set_busy_session_handler(stock_busy)
    assert busy_ingress.install(runner) == 1
    source = SessionSource(platform=Platform.SMS, chat_id='owner-chat', chat_type='dm', user_id='owner')
    event = MessageEvent(text='the password is hunter2', source=source, message_id='owner-message-2')
    key = adapter._event_session_key(event)
    adapter._message_handler = lambda event: None
    adapter._active_sessions[key] = asyncio.Event()   # the session is busy: no owner task, so never healed
    asyncio.run(adapter.handle_message(event))
    assert heard == [('the password is hunter2', True)]
    parked = adapter._pending_messages[key]
    assert parked.text == 'the password is [secret S-AAAAAAAA: password]', 'the parked message is sealed'
    assert busy_ingress.seen(parked)
    assert parked.message_id == 'owner-message-2'
