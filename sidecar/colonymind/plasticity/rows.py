"""Bind Instinct's re-rendering reader to the plasticity RowSource contract."""
from __future__ import annotations

from .manifests import check_row


class InstinctRows:
    """States exist only while the caller iterates. No replay text is cached or written."""

    def __init__(self, store, sources, *, source_of=None):
        self.store, self.sources = store, sources
        self._source_of = source_of

    def rows(self, objective, *, since=None):
        from ..instinct.outcomes import replay_rows
        seen = set()
        for point in objective.points or (None,):
            for row in replay_rows(self.store, self.sources, point=point, since=since):
                check_row(row)
                key = (row["ref"], row["teacher"])
                if key not in seen:
                    seen.add(key)
                    yield row

    def source_of(self, row):
        if self._source_of is not None:
            return self._source_of(row)
        # Owner labels and receipted resolver outcomes have no external supplier.
        # A future source must declare its attribution instead of escaping the cap.
        if row["source_kind"] in {"explicit", "resolver", "synthetic"}:
            return None
        return row["source_kind"]
