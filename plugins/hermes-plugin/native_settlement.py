"""Keep capture cleanup after native turn release on stock Hermes."""
from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import logging
import threading

logger = logging.getLogger(__name__)
_MARKER = '_colonymind_capture_settlement'
_LOCK = threading.RLock()


def _outcome(value):
    """Match the host's narrow machine-readable outcome, without turn content."""
    if not isinstance(value, Mapping):
        return {}
    result = {key: value[key] for key in ('completed', 'failed', 'interrupted', 'failure_retryable')
              if isinstance(value.get(key), bool)}
    result.update({key: value[key][:256] for key in ('turn_exit_reason', 'failure_reason')
                   if isinstance(value.get(key), str)})
    return result


class _Settlement:
    def __init__(self, facade, loop, home_key):
        self.facade, self.loop = facade, loop
        self.owners = {}
        self.original = facade.run_conversation
        self.original_loop = loop.run_conversation
        self.frame = ContextVar('colonymind_capture_native_turn', default=None)

        @wraps(self.original_loop)
        def capture_identity(agent, *args, **kwargs):
            frame = self.frame.get()
            previous = getattr(agent, '_current_turn_id', None)
            try:
                return self.original_loop(agent, *args, **kwargs)
            finally:
                # The facade still holds its native lease here. A successor may
                # reuse this agent as soon as release returns, so do not read
                # these mutable fields in the after-release observer.
                try:
                    turn = getattr(agent, '_current_turn_id', None)
                    if (frame is not None and frame['agent'] is agent and turn and turn != previous
                            and not getattr(agent, '_persist_disabled', False)):
                        frame['native'] = dict(session_id=str(getattr(agent, 'session_id', '') or ''),
                            task_id=str(getattr(agent, '_current_task_id', '') or ''), turn_id=str(turn),
                            platform=str(getattr(agent, 'platform', '') or ''))
                except Exception:
                    logger.warning('Native capture identity unavailable', exc_info=True)

        @wraps(self.original)
        def settled(agent, *args, **kwargs):
            try:
                scope = home_key()
            except Exception:
                logger.warning('Native capture profile unavailable', exc_info=True)
                return self.original(agent, *args, **kwargs)
            frame = {'agent': agent, 'native': None}
            token = self.frame.set(frame)
            outcome = None
            try:
                outcome = self.original(agent, *args, **kwargs)
                return outcome
            except BaseException as error:
                outcome = {'failed': True, 'completed': False,
                           'interrupted': isinstance(error, (KeyboardInterrupt, InterruptedError))
                           or type(error).__name__ == 'CancelledError'}
                raise
            finally:
                self.frame.reset(token)
                try:
                    # Admission failure never entered a bound loop. Detached
                    # forks do not publish parent-session capture records.
                    if frame['native'] is not None:
                        with _LOCK:
                            observer = self.owners.get(scope)
                        if observer is not None:
                            native = dict(frame['native'], outcome=_outcome(outcome))
                            for callback in observer[1]:
                                try:
                                    callback(**native)
                                except Exception:
                                    logger.warning('Native capture settlement notification failed', exc_info=True)
                except Exception:
                    logger.warning('Native capture settlement unavailable', exc_info=True)

        self.wrapper, self.loop_wrapper = settled, capture_identity
        setattr(settled, _MARKER, self)

    def install(self):
        self.loop.run_conversation = self.loop_wrapper
        self.facade.run_conversation = self.wrapper

    def remove(self):
        if self.facade.run_conversation is self.wrapper:
            self.facade.run_conversation = self.original
        if self.loop.run_conversation is self.loop_wrapper:
            self.loop.run_conversation = self.original_loop


def register(ctx, capture):
    from hermes_cli.plugins import VALID_HOOKS
    if 'on_native_turn_settled' in VALID_HOOKS:
        # Preserve the host's existing after-release callback order and early exits.
        ctx.register_hook('on_native_turn_settled', capture.on_native_turn_settled)
        ctx.register_hook('on_native_turn_settled', capture.native_turn_settled)
        return
    from agent.turn_facade import TurnFacadeMixin
    from agent import conversation_loop
    from hermes_constants import hermes_home_key

    home, token = hermes_home_key(), object()
    with _LOCK:
        if getattr(ctx, '_load_abandoned', False):
            return
        current = TurnFacadeMixin.run_conversation
        state = getattr(current, _MARKER, None)
        # A later wrapper can copy a disposed owner's marker with functools.wraps.
        if state is not None and not state.owners:
            state = None
        fresh = state is None
        if state is None:
            state = _Settlement(TurnFacadeMixin, conversation_loop, hermes_home_key)
        state.owners[home] = (token, (capture.on_native_turn_settled, capture.native_turn_settled))

        def unload():
            with _LOCK:
                if state.owners.get(home, (None,))[0] is token:
                    state.owners.pop(home)
                if not state.owners:
                    state.remove()

        try:
            ctx.on_unload(unload)
            if getattr(ctx, '_load_abandoned', False):
                unload()
            if fresh and state.owners.get(home, (None,))[0] is token:
                state.install()
        except Exception:
            unload()
            raise
