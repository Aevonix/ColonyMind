"""The atlas: a static, owner-only, offline HTML site over the ledger's cited claims.

One page per person (only their own words), per month, per topic and per cited message; every line
quotes its source and links to it, history stays visible, erased sources are gone after the next
build. ``colonymind atlas build`` writes it, ``colonymind atlas check`` audits it (docs/ATLAS.md).
"""
from .site import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, DEFAULT_PAGE_LINES, build_atlas, check_atlas
from .snapshot import AtlasError

__all__ = ["AtlasError", "DEFAULT_MAX_BYTES", "DEFAULT_MAX_LINES", "DEFAULT_PAGE_LINES", "build_atlas", "check_atlas"]
