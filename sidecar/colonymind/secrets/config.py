"""``secrets:`` in ``colonymind.yaml``: capture, the vault key and the one-time migration.

::

    secrets:
      capture:
        enabled: true        # seal detected secrets wherever a turn is stored (the point of the fix)
        ingress: true        # seal gateway messages before Hermes sees them
        rules: [prefix, jwt, private_key, webhook, url_userinfo, url_token, assign, json_field, bearer,
                labeled, telegram, card]   # a rule can be turned off; none is added without a test
      vault:
        key_store: file      # file (secrets-vault.key, 0600) or keyring
        key_in_backups: false
      migrate:
        on_start: true       # seal what was stored before capture sealed (once, at the first start)

Parsed here, not in ``colonymind.config`` (whose validation keeps unknown keys as they are). A malformed section
never weakens capture: the defaults stand and the problem is logged.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from .detect import RULES
from .vault import KEY_STORES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecretsSettings:
    capture_enabled: bool = True
    ingress: bool = True
    rules: Tuple[str, ...] = RULES
    key_store: str = "file"
    key_in_backups: bool = False
    migrate_on_start: bool = True

    def as_dict(self) -> dict:
        return {"capture": {"enabled": self.capture_enabled, "ingress": self.ingress, "rules": list(self.rules)},
                "vault": {"key_store": self.key_store, "key_in_backups": self.key_in_backups},
                "migrate": {"on_start": self.migrate_on_start}}


def _flag(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"secrets.{name} must be true or false")


def _rules(value: Any) -> Tuple[str, ...]:
    if value is None:
        return RULES
    if isinstance(value, Mapping):
        unknown = sorted(set(map(str, value)) - set(RULES))
        if unknown:
            raise ValueError(f"secrets.capture.rules names unknown rules: {', '.join(unknown)}")
        off = {str(name) for name, on in value.items() if not _flag(on, True, f"capture.rules.{name}")}
        return tuple(rule for rule in RULES if rule not in off)
    if isinstance(value, (list, tuple)):
        named = [str(item).strip() for item in value]
        unknown = sorted(set(named) - set(RULES))
        if unknown:
            raise ValueError(f"secrets.capture.rules names unknown rules: {', '.join(unknown)}")
        return tuple(rule for rule in RULES if rule in named)
    raise ValueError("secrets.capture.rules must be a list of rule names or a mapping of rule: on/off")


def parse_settings(section: Any) -> SecretsSettings:
    """The ``secrets:`` section as settings; raises ``ValueError`` on a malformed one."""
    if section is None:
        return SecretsSettings()
    if not isinstance(section, Mapping):
        raise ValueError("secrets must be a mapping")
    capture = section.get("capture") or {}
    vault = section.get("vault") or {}
    migrate = section.get("migrate") or {}
    for name, part in (("capture", capture), ("vault", vault), ("migrate", migrate)):
        if not isinstance(part, Mapping):
            raise ValueError(f"secrets.{name} must be a mapping")
    key_store = str(vault.get("key_store") or "file").strip().lower()
    if key_store not in KEY_STORES:
        raise ValueError(f"secrets.vault.key_store must be one of {', '.join(KEY_STORES)}")
    return SecretsSettings(
        capture_enabled=_flag(capture.get("enabled"), True, "capture.enabled"),
        ingress=_flag(capture.get("ingress"), True, "capture.ingress"),
        rules=_rules(capture.get("rules")),
        key_store=key_store,
        key_in_backups=_flag(vault.get("key_in_backups"), False, "vault.key_in_backups"),
        migrate_on_start=_flag(migrate.get("on_start"), True, "migrate.on_start"),
    )


_CURRENT: Optional[SecretsSettings] = None
_LOCK = threading.Lock()


def current_settings() -> SecretsSettings:
    """The running instance's settings, read once from ``colonymind.yaml``; the defaults when it has none or the
    section is malformed (logged: capture never turns itself off)."""
    global _CURRENT
    if _CURRENT is not None:
        return _CURRENT
    with _LOCK:
        if _CURRENT is None:
            try:
                from colonymind.config import load_config
                _CURRENT = parse_settings(load_config().data.get("secrets"))
            except Exception as error:
                logger.warning("secrets settings unreadable (%s); capture keeps its defaults", error)
                _CURRENT = SecretsSettings()
        return _CURRENT


def set_settings(settings: Optional[SecretsSettings]) -> None:
    """Replace the cached settings (the server at start, tests); None reads the file again on next use."""
    global _CURRENT
    with _LOCK:
        _CURRENT = settings


__all__ = ["SecretsSettings", "current_settings", "parse_settings", "set_settings"]
