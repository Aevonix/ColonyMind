"""The sealed store: secrets that arrive in conversation, kept by reference (docs/SECRETS.md).

A secret in a turn (a key, a password, a code, a card number) moves into an encrypted store and a reference
stands in its place wherever the turn is kept, so memory search, the transcripts the agent reads back and every
queue derived from a turn hold the reference, never the value. The owner's own values stay usable through
``colonymind_secret`` and ``/v1/secrets``; another person's are never resolved.

Configured credentials (provider and tool keys) are the host's: they live in Hermes' own configuration, not here.

Modules: ``detect`` (the calibrated detector), ``refs`` (the reference format), ``vault`` (the store),
``seal`` (sealing messages and turns), ``migrate`` (sealing what was stored before, and releasing), ``config``
(the ``secrets:`` section of ``colonymind.yaml``), ``cli`` (``colonymind secrets``).
"""

from colonymind.secrets.detect import HIDDEN, REFERENCE, RULES, WITHHELD, find_secrets, hide_secrets

__all__ = ["HIDDEN", "REFERENCE", "RULES", "WITHHELD", "find_secrets", "hide_secrets"]
