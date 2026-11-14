"""Capture detector API; the implementation is shared with host adapters."""

# The repository scanner loads this file directly, without installing either distribution.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from colonymind_shared.secret_detection import (
        HIDDEN, Hit, REFERENCE, RULES, VENDOR_PATTERNS, WITHHELD, clean_label, find_secrets,
        hide_secrets, replace_secrets, _NOT_SECRET_NAME, _SECRET_NAME,
    )
finally:
    if not __package__:
        sys.path.pop(0)


__all__ = ["HIDDEN", "Hit", "REFERENCE", "RULES", "VENDOR_PATTERNS", "WITHHELD", "clean_label", "find_secrets",
           "hide_secrets", "replace_secrets"]
