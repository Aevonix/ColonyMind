"""Hands registration installs guards before activating delegation wrappers."""

from __future__ import annotations

from importlib import import_module
import logging
from typing import Any

from .wrap import RouteResolver, get_route_resolver, register_route_resolver, resolve

logger = logging.getLogger("colonymind-hands")
# The host resolver uses this to distinguish an intentional skip from a broken hands load.
ADAPTER_ABSENT = False

MODULES = (
    "wrap",
    "dispatch",
    "spool",
    "budgets",
    "allow",
    "effects",
    "reconcile",
    "degraded",
    "tools",
    "commands",
    "notices",
    "front",
    "checks",
    "authority",
    "middleware",
)


def register(ctx: Any) -> None:
    """Register every fixed module; a failure leaves delegation refused, never unguarded.

    Delegation wrappers become visible only after every guard and unload callback
    registers. A timed-out load cannot expose an unguarded dispatcher.
    """
    global ADAPTER_ABSENT
    ADAPTER_ABSENT = False
    try:
        try:
            import_module("colonymind_hermes")
        except ModuleNotFoundError as exc:
            if exc.name != "colonymind_hermes":
                raise
            ADAPTER_ABSENT = True
            # A stable logger survives Hermes loading this plugin under multiple profile names.
            if not getattr(logger, "_adapter_absence_logged", False):
                logger._adapter_absence_logged = True
                logger.info("hands need the ColonyMind adapter; not loaded in this process")
            return
        for name in MODULES:
            import_module(f".{name}", package=__name__).register(ctx)
        import_module('.dispatch', package=__name__).activate(ctx)
    except Exception:
        _refuse_delegation(ctx)
        raise


def _refuse_delegation(ctx=None) -> None:
    try:
        from .dispatch import refuse_delegation
        refuse_delegation(owner=getattr(ctx, 'hands_runtime', None), context=ctx)
    except Exception:
        pass


__all__ = ["MODULES", "RouteResolver", "get_route_resolver", "register", "register_route_resolver", "resolve"]
