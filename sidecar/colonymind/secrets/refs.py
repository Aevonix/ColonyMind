"""References: how a sealed value reads in stored words.

``[secret S-XXXXXXXX: <label>]``: eight base32 characters from an HMAC of the scope (the person whose message
carried the value) and the value under the vault key, and a label from the detector's fixed vocabulary. The same
value in the same scope always gets the same reference, so a replay or a re-import stores identical bytes; two
scopes get two references. Memory search, recall and the context packet find the label, so "the wifi password I
gave you" finds the turn and its reference.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import List

from .detect import REFERENCE, clean_label

ID_PREFIX = "S-"
FORGOTTEN_LABEL = "a value the owner deleted"


@dataclass(frozen=True)
class Reference:
    ref_id: str
    label: str
    start: int
    end: int
    text: str


def reference_id(key: bytes, scope: str, value: str, *, salt: int = 0) -> str:
    """``S-`` and eight base32 characters of HMAC-SHA256(key, scope NUL value [NUL salt])."""
    message = f"{scope}\x00{value}" + (f"\x00{salt}" if salt else "")
    digest = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    return ID_PREFIX + base64.b32encode(digest).decode("ascii")[:8]


def reference_text(ref_id: str, label: str) -> str:
    return f"[secret {ref_id}: {clean_label(label)}]"


def references_in(text: str) -> List[Reference]:
    """Every reference in ``text``, in order."""
    found = []
    for match in REFERENCE.finditer(str(text or "")):
        body = match.group(0)[len("[secret "):-1]
        ref_id, _, label = body.partition(": ")
        found.append(Reference(ref_id, label, match.start(), match.end(), match.group(0)))
    return found


def reference_ids(text: str) -> List[str]:
    return list(dict.fromkeys(item.ref_id for item in references_in(text)))


__all__ = ["FORGOTTEN_LABEL", "ID_PREFIX", "Reference", "reference_id", "reference_ids", "reference_text",
           "references_in"]
