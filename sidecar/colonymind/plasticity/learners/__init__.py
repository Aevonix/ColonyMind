"""Discover code-registered learner implementations once per process."""
from functools import cache
from importlib import import_module
from pkgutil import iter_modules


@cache
def load() -> None:
    """Import public learner modules in name order so they register their factories."""
    for name in sorted(module.name for module in iter_modules(__path__) if not module.name.startswith("_")):
        import_module(f"{__name__}.{name}")
