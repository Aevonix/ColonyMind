"""Upgrade notices and write guards for retired, inert controls."""

from collections.abc import Mapping
from functools import wraps
import logging


REMOVED_SETTINGS = (
    "hands.direct_line.enabled",
    "hands.triage.fast_timeout_ms",
    "hands.triage.deliberate_timeout_ms",
    "hands.triage.join_timeout_ms",
    "hands.triage.output_tokens",
    "hands.triage.max_per_hour",
    "hands.front.thinking_mode",
    "hands.front.soft_deadline_seconds",
    "hands.front.hard_deadline_seconds",
    "hands.front.tool_timeout_seconds",
    "hands.front.max_non_read_calls",
    "hands.front.max_read_calls",
    "hands.front.output_tokens",
    "hands.front.continuation_output_tokens",
    "hands.review.soft_deadline_seconds",
    "hands.review.output_tokens",

    "hands.brief.recall_tokens", "hands.brief.identity_chars",
    "hands.health.shedding_window_seconds", "hands.health.front_ttft_p95_seconds",
    "hands.health.voice_decode_p50_tokens_per_second", "hands.health.recovery_seconds",
    "hands.stalls.approval_lapse_seconds",
)
REMOVED_CONTACT_ARGUMENTS = (
    "contacts.enrichment", "contacts.import", "contacts.vcard", "contacts.retention",
    "contacts.enabled", "contacts.default_inbound_tier", "contacts.unknown_tier_grace_hours",
    "contacts.auto_merge_confidence_threshold", "contacts.merge_proposal_threshold",
    "contacts.soft_delete_retention_days", "contacts.audit_retention_days",
    "contacts.brief_on_tier_change", "contacts.brief_on_merge_proposals",
    "contacts.normalize_phones", "contacts.default_phone_region", "contacts.max_batch_size",
    "contacts.vcard_include_tier", "contacts.vcard_include_tags",
)
REMOVED_ENVIRONMENT = (
    "COLONYMIND_EMBED_CACHE_SIZE", "COLONYMIND_EMBED_BATCH_WINDOW_MS",
    "COLONYMIND_EMBED_QUANTIZATION",
)


def upgrade_message(names):
    message = (f"Removed settings: {', '.join(sorted(names))}. To upgrade, remove these entries; "
               "they never controlled runtime behavior.")
    if "COLONYMIND_EMBED_QUANTIZATION" in names:
        message += " Configure quantization at the provider; rebuild indexes after changing served weights."
    return message


def removed_settings(data):
    found = []
    for path in REMOVED_SETTINGS:
        value = data
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            found.append(path)
    return found


def reject_removed_settings(data):
    found = removed_settings(data)
    if found:
        raise ValueError(upgrade_message(found))


def remove_settings(data, paths):
    """Remove named leaves from a mutable configuration mapping."""
    for path in paths:
        parts = path.split(".")
        node = data
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]


def ignore_removed_settings(data):
    """Reads tolerate old init files; writes still reject retired settings."""
    found = removed_settings(data)
    for path in found:
        logging.getLogger(__name__).warning(upgrade_message([path]))
    remove_settings(data, found)
    return data


def reject_removed_environment(environ):
    found = set(REMOVED_ENVIRONMENT).intersection(environ)
    if found:
        raise ValueError(upgrade_message(found))


def reject_removed_arguments(retired):
    """Give Python configuration callers the same upgrade remedy as file/env callers."""
    def decorate(cls):
        original = cls.__init__

        @wraps(original)
        def initialize(self, *args, **kwargs):
            found = [retired[name] for name in kwargs if name in retired]
            if found:
                raise ValueError(upgrade_message(found))
            original(self, *args, **kwargs)

        cls.__init__ = initialize
        return cls
    return decorate
