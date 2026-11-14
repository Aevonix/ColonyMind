"""Read-only owner artifact authority, with no visibility or profile-based ownership fallback.

Library supplies canonical items and ACL versions through LibraryReader. The cron adapter
supplies verified destination handles and original paginated outputs. Missing adapters or
incomplete historical evidence are availability failures, never empty history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .authority import Audience
from .validation import GoalError, TIERS, artifact_ref


class ArtifactHidden(GoalError):
    """Known owner content excluded by this audience; filter it before pagination."""
    def __init__(self):
        super().__init__(404, "artifact_not_found")


@dataclass(frozen=True)
class LibraryItem:
    namespace_id: str
    id: str
    version: str
    acl_version: str
    sensitivity: str
    owner_id: str | None
    owner_acl: frozenset[str]
    # None means no additional content ACL; an empty set grants nobody.
    readers: frozenset[str] | None = None


class LibraryReader(Protocol):
    def get_item(self, ref: str) -> LibraryItem | None:
        """Read canonical namespace, object, authoritative owner/ACL and tier; None only if absent.

        Raise on unavailable authority. Imported metadata, labels and public visibility
        must never be substituted for an authoritative owner association or owner ACL.
        """
        ...


@dataclass(frozen=True)
class ArtifactHandle:
    owner_id: str
    type: str
    ref: str
    namespace_id: str
    object_version: str
    evidence_version: str
    sensitivity: str
    name: str = ""
    enabled: bool | None = None
    recurring: bool | None = None

    def binding(self):
        from dataclasses import asdict
        return asdict(self)


class CronReader(Protocol):
    def resolve_owner_artifact(self, owner_id: str, ref: str, audience: Audience) -> ArtifactHandle:
        ...

    def list_owner_job_runs(self, owner_id: str, handle: ArtifactHandle, audience: Audience,
                            limit: int, offset: int) -> dict:
        """Authorized original outputs, filtered before offsets, no history cap or silent gaps."""
        ...


class ArtifactResolver:
    def __init__(self, library: LibraryReader | None = None, cron: CronReader | None = None):
        self.library, self.cron = library, cron

    def resolve_owner_artifact(self, owner_id: str, type: str, ref: str, audience: Audience) -> ArtifactHandle:
        artifact_ref(type, ref)
        try:
            if type == "cron_job":
                if self.cron is None:
                    raise GoalError(503, "owner_goals_unavailable")
                handle = self.cron.resolve_owner_artifact(owner_id, ref, audience)
                if (not isinstance(handle, ArtifactHandle) or handle.owner_id != owner_id
                        or handle.type != type or handle.ref != ref or not handle.namespace_id
                        or not handle.object_version or not handle.evidence_version
                        or handle.sensitivity not in TIERS or type_bool(handle.enabled) is False
                        or type_bool(handle.recurring) is False):
                    raise GoalError(503, "owner_goals_unavailable")
                if not audience.allows(handle.sensitivity):
                    raise ArtifactHidden()
                return handle
            if self.library is None:
                raise GoalError(503, "owner_goals_unavailable")
            item = self.library.get_item(ref)
            if item is None:
                raise GoalError(404, "artifact_not_found")
            if (not isinstance(item, LibraryItem) or item.id != ref or not item.namespace_id
                    or not item.version or not item.acl_version or item.sensitivity not in TIERS):
                raise GoalError(503, "owner_goals_unavailable")
            if item.owner_id is None and not item.owner_acl:
                # No implemented ownership evidence is different from a known foreign owner.
                raise GoalError(503, "owner_goals_unavailable")
            if item.owner_id != owner_id and owner_id not in item.owner_acl:
                raise GoalError(404, "artifact_not_found")
            if (not audience.allows(item.sensitivity)
                    or (item.readers is not None and not audience.person_ids <= item.readers)):
                raise ArtifactHidden()
            return ArtifactHandle(owner_id, type, ref, item.namespace_id, item.version,
                                  item.acl_version, item.sensitivity)
        except GoalError:
            raise
        except Exception as exc:
            raise GoalError(503, "owner_goals_unavailable") from exc

    def list_owner_job_runs(self, owner_id, handle, audience, limit, offset):
        if self.cron is None:
            raise GoalError(503, "owner_goals_unavailable")
        try:
            return self.cron.list_owner_job_runs(owner_id, handle, audience, limit, offset)
        except GoalError:
            raise
        except Exception as exc:
            raise GoalError(503, "owner_goals_unavailable") from exc


def type_bool(value):
    return type(value) is bool


def resolve_owner_artifact(owner_id, type, ref, audience, *, resolver: ArtifactResolver | None = None):
    """Shared internal resolver boundary for attach, projection and migration."""
    return (resolver or ArtifactResolver()).resolve_owner_artifact(owner_id, type, ref, audience)
