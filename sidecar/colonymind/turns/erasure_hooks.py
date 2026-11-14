"""Process-local listeners for committed turn-source erasure."""
from threading import RLock
from typing import Any, Callable, Sequence

_HOOKS: dict[object, Callable[[Sequence[str]], Any]] = {}
_LOCK = RLock()


def bind_source_erasure(callback: Callable[[Sequence[str]], Any]) -> Callable[[], None]:
    """Bind a source-erasure listener and return its idempotent unbind operation."""
    token = object()
    with _LOCK:
        _HOOKS[token] = callback

    def unbind() -> None:
        with _LOCK:
            _HOOKS.pop(token, None)

    return unbind


def notify_source_erasure(ids: Sequence[str]) -> None:
    """Notify listeners after commit; failures propagate so callers can retry."""
    ids = tuple(dict.fromkeys(ids))
    with _LOCK:
        callbacks = tuple(_HOOKS.values())
    for callback in callbacks:
        callback(ids)
