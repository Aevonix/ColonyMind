"""Owned, static artwork and the Ideas presentation vocabulary."""
from functools import lru_cache
from importlib.resources import files

KINDS = {
    "packet": ("Comparison", "Money", "money"),
    "renewal": ("Renewal review", "Subscriptions", "subscriptions"),
    "document": ("Form checklist", "Paperwork", "paperwork"),
    "calendar": ("Meeting brief", "Planning", "planning"),
}
BUILD_STATUS = {"accepted": "in_progress", "working": "in_progress", "done": "complete", "failed": "failed"}
MEDIA_PREFIX = "/v1/host/initiatives/media/"


@lru_cache(maxsize=4)
def media(handle):
    allowed = {f"{kind}-{category}-v1.svg" for kind, (_, _, category) in KINDS.items()}
    if handle not in allowed:
        raise ValueError("unknown_idea_media")
    return files("colonymind.mind").joinpath("idea_assets", handle).read_bytes()


def presentation(kind, state=None):
    primary, category, category_id = KINDS[kind]
    handle = f"{kind}-{category_id}-v1.svg"
    icon = {"altText": f"{primary} icon", "variants": [{"url": MEDIA_PREFIX + handle, "width": 96, "height": 96,
            "mimeType": "image/svg+xml", "byteSize": len(media(handle)), "version": 1}]}
    # Clean rows: no kind/category labels, and a badge only while a build state needs attention.
    badge = {"accepted": "Queued", "working": "Building", "done": "Ready", "failed": "Build failed"}.get(state)
    return {"icon": icon, "labels": [], "badges": [{"text": badge}] if badge else [],
            **({"display_badge": {"text": badge}} if badge else {}), "actionLabel": "Build"}
