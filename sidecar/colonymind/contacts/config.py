"""ColonyMind Contacts — configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from colonymind.removed_settings import REMOVED_CONTACT_ARGUMENTS, reject_removed_arguments


@reject_removed_arguments({path.split(".", 1)[1]: path for path in REMOVED_CONTACT_ARGUMENTS})
@dataclass
class ContactsConfig:
    sqlite_path: str = ":memory:"

    @classmethod
    def from_env(cls) -> "ContactsConfig":
        """Build config from the environment, persisting to the state dir.

        The bare default of ":memory:" exists for tests; a production
        sidecar must survive restarts (the contact store is the source of
        truth for who people are), so the
        server path resolves COLONYMIND_CONTACTS_DB or falls back to
        ``$COLONYMIND_STATE_DIR/colonymind-contacts.db``.
        """
        path = os.environ.get("COLONYMIND_CONTACTS_DB")
        if not path:
            # Anchor to the real state dir, never the process CWD: a
            # cwd-relative default silently forks the contact store per
            # working directory.
            state_dir = os.environ.get("COLONYMIND_STATE_DIR", "")
            if not state_dir:
                state_dir = os.path.join(os.path.expanduser("~"),
                                         ".colonymind", "data")
            path = os.path.join(state_dir, "colonymind-contacts.db")
        try:
            min_int = int(os.environ.get("COLONYMIND_GROUP_PROMOTE_MIN_INTERACTIONS", "") or 5)
        except ValueError:
            min_int = 5
        return cls(sqlite_path=path, group_promote_min_interactions=min_int)
    # A scope (group) member with this many interactions is listed as a promotion candidate for
    # the owner to approve; membership never grants 1:1 rights by itself.
    group_promote_min_interactions: int = 5
