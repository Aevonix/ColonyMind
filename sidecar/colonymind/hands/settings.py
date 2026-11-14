"""Validation shared by file configuration and isolated hands controllers."""

from colonymind.removed_settings import reject_removed_settings


def validate_limits(config):
    reject_removed_settings({"hands": config})
    if not isinstance(config, dict):
        raise ValueError("hands must be a mapping")
    for group, fields in (
        ("cascade", ("max_per_task", "max_per_hour")),
        ("retention", ("events_days", "rows_days")),
        ("obligations", ("overdue_seconds",)),
        ("brief", ("max_tokens",)),
    ):
        values = config.get(group)
        if not isinstance(values, dict):
            raise ValueError(f"hands.{group} must be a mapping")
        for field in fields:
            value = values.get(field)
            if type(value) is not int or not 0 <= value <= 3_153_600_000:
                raise ValueError(f"hands.{group}.{field} must be a non-negative whole number "
                                 "no greater than 3153600000")
            if group == "brief" and value == 0:
                raise ValueError("hands.brief.max_tokens must be positive")
            if group == "retention" and value == 0:
                raise ValueError(f"hands.retention.{field} must be at least 1 day")
    return config
