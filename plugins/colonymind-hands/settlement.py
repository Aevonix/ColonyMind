"""Observe successful native lease release without depending on fork hooks."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from contextvars import copy_context
from functools import wraps
import logging
from pathlib import Path
import threading

logger = logging.getLogger(__name__)
_MARKER = '_colonymind_hands_settlement'
_INSTALL_LOCK = threading.RLock()


def _home(value):
    return str(Path(value).resolve()) if value else None


class _Release:
    def __init__(self, runner, original):
        self.runner, self.original = runner, original
        self.owners = {}
        self.lock = threading.RLock()
        self.original_clear = getattr(runner, '_release_running_agent_state', None)
        self.clear_wrapper = None
        if callable(self.original_clear):
            @wraps(self.original_clear)
            def clearing(session_key, *, run_generation=None):
                if run_generation is not None:
                    try:
                        session_id, home, scoped = self._identity(session_key, run_generation)
                        owner = self._owner(home, scoped)
                        if session_id and owner is not None and callable(owner.prepare):
                            # Observe identity synchronously while native still
                            # exposes the running agent. No receipt is published.
                            with self._scope(home if scoped else None):
                                owner.prepare(session_id=session_id, session_key=session_key,
                                              run_generation=run_generation, gateway=self.runner)
                    except Exception:
                        logger.exception('Hands could not observe native turn cleanup')
                return self.original_clear(session_key, run_generation=run_generation)
            self.clear_wrapper = clearing

    def _identity(self, session_key, run_generation):
        session_id, home, scoped = '', None, False
        try:
            held = self.runner._held_turn_lease(session_key, run_generation)
            token = held[1].get(run_generation) if held is not None else None
            session_id = getattr(token, 'session_id', '')
            store = getattr(self.runner, 'session_store', None)
            scoped = bool(getattr(getattr(store, 'config', None), 'multiplex_profiles', False))
            if scoped:
                profile = store._named_profile_for_key(session_key)
                home = _home(store._profile_home_for_key(session_key) if profile else store._routing_home)
                if home is None:
                    raise ValueError('Settled turn profile is unavailable')
        except Exception:
            # Observation must never alter the native call or its exception behavior.
            session_id = ''
            logger.exception('Hands could not capture the turn lease')
        return session_id, home, scoped

    def _owner(self, home, scoped):
        with self.lock:
            owner = self.owners.get(home) if scoped else next(reversed(self.owners.values()), None)
            return owner if owner is not None and not owner.closed else None

    @staticmethod
    def _scope(home):
        if home is None:
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        return _profile_runtime_scope(Path(home), hydrate_secrets=False)

    def __call__(self, session_key, run_generation):
        session_id, home, scoped = self._identity(session_key, run_generation)
        result = self.original(session_key, run_generation)
        if result is True and session_id:
            try:
                owner = self._owner(home, scoped)
                if owner is not None:
                    self._notify(owner, session_id, session_key, run_generation, home if scoped else None)
            except Exception:
                logger.exception('Hands lease settlement notification failed')
        return result

    def _notify(self, owner, session_id, session_key, generation, home):
        def notify():
            try:
                with self.lock:
                    active = self.owners.get(owner.home) is owner
                if (active and not owner.closed
                        and self.runner._is_session_run_current(session_key, generation)):
                    owner.callback(session_id=session_id, session_key=session_key,
                                   run_generation=generation, gateway=self.runner)
            except Exception:
                logger.exception('Hands lease settlement notification failed')

        with self._scope(home):
            loop = getattr(self.runner, '_gateway_loop', None)
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if loop is not None and loop is not current_loop:
                # The callback inherits the routed profile, even after its worker scope exits.
                loop.call_soon_threadsafe(notify, context=copy_context())
            else:
                notify()


class Settlement:
    """One profile's observer, sharing a single wrapper across reloads and profiles."""
    def __init__(self, callback, home=None, *, prepare=None):
        self.callback, self.home = callback, _home(home)
        self.prepare = prepare
        self.runner = self.wrapper = None
        self.closed = self.warned = False

    def install(self, runner):
        if self.closed or runner is None:
            return False
        original = getattr(runner, '_release_turn_lease', None)
        if not callable(original) or not callable(getattr(runner, '_held_turn_lease', None)):
            if not self.warned:     # once: every inbound message asks for the install again
                self.warned = True
                missing = [name for name in ('_release_turn_lease', '_held_turn_lease')
                           if not callable(getattr(runner, name, None))]
                logger.warning('Hands cannot see when a turn ends: the gateway of this Hermes has no %s',
                               ' and no '.join(missing))
            return False
        if self.runner is not None and self.runner is not runner:
            self._remove()
        with _INSTALL_LOCK:
            original = runner._release_turn_lease
            wrapper = getattr(original, _MARKER, None)
            # Another component may copy a disposed marker with functools.wraps.
            # A fresh owner must also reinstall its pre-clear identity observer.
            if wrapper is not None and not wrapper.owners:
                wrapper = None
            if wrapper is None:
                wrapper = _Release(runner, original)
                setattr(wrapper, _MARKER, wrapper)
                runner._release_turn_lease = wrapper
                if wrapper.clear_wrapper is not None:
                    runner._release_running_agent_state = wrapper.clear_wrapper
            with wrapper.lock:
                wrapper.owners[self.home] = self
            self.runner, self.wrapper = runner, wrapper
        return True

    def _remove(self):
        with _INSTALL_LOCK:
            wrapper, runner = self.wrapper, self.runner
            if wrapper is not None:
                with wrapper.lock:
                    if wrapper.owners.get(self.home) is self:
                        wrapper.owners.pop(self.home)
                    if not wrapper.owners:
                        if runner._release_turn_lease is wrapper:
                            runner._release_turn_lease = wrapper.original
                        if (wrapper.clear_wrapper is not None
                                and runner._release_running_agent_state is wrapper.clear_wrapper):
                            runner._release_running_agent_state = wrapper.original_clear
            self.runner = self.wrapper = None

    def close(self):
        self.closed = True
        self._remove()
