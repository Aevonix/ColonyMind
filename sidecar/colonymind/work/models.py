"""Bounded transport facts for verified work, never raw tool transcripts."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Digest = Annotated[str, Field(pattern='^[0-9a-f]{64}$')]
ResourceRef = Annotated[str, Field(min_length=10, max_length=256, pattern='^resource:[A-Za-z0-9_.:-]+$')]
JOB_FIELDS = frozenset({'name', 'prompt', 'schedule', 'repeat', 'deliver', 'failure_deliver',
    'script', 'skill', 'workdir', 'no_agent', 'paused', 'monitor', 'attach_to_session', 'enabled'})
JOB_SCHEMA = {'properties': {key: {'type': 'boolean'} for key in sorted(JOB_FIELDS)},
    'additionalProperties': False, 'maxProperties': 64}
ACTION_TOOLS = frozenset({'terminal', 'send_message', 'cronjob', 'cronjob_manage', 'write_file', 'patch',
    'delegate_task', 'delegate', 'subagent', 'hands', 'kanban_create', 'read_file', 'search_files',
    'list_directory', 'web_search', 'web_extract', 'session_search', 'skills_list', 'skill_view',
    'kanban_get', 'kanban_list', 'tool_search', 'tool_describe'})


def job_schema(schema):
    for branch in schema.get('anyOf', [schema]):
        if branch.get('type') == 'object':
            branch.update(JOB_SCHEMA)


class Wire(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class SourceRef(Wire):
    source_id: Identifier
    source_version: Digest


Title = Annotated[str, Field(max_length=128, pattern=r'^\S+(?: [^\s]+){2,7}$')]


class WorkTitle(Wire):
    id: Annotated[str, Field(pattern=r'^w[1-9][0-9]?$')]
    title: Title


class WorkTitles(Wire):
    titles: list[WorkTitle] = Field(max_length=16)


class Evidence(Wire):
    kind: Literal['none', 'file_hash', 'message_receipt', 'saved_job', 'hand_outcome', 'read'] = 'none'
    method: Literal['native_post_write_sha256', 'sealed_sha256', 'native_saved_job_fields'] | None = None
    checked_operation_id: Identifier | None = None
    intended_sha256: Digest | None = None
    actual_sha256: Digest | None = None
    receipt_id: Identifier | None = None
    provider_message_id: Identifier | None = None
    requested_fields: dict[str, bool] | None = Field(default=None, json_schema_extra=job_schema)
    saved_fields: dict[str, bool] | None = Field(default=None, json_schema_extra=job_schema)
    verified: bool | None = None
    outcome: Literal['done', 'completed', 'failed', 'cancelled', 'interrupted', 'unknown', 'pending'] | None = None
    task_id: Identifier | None = None

    @model_validator(mode='after')
    def bounded_fields(self):
        for fields in (self.requested_fields, self.saved_fields):
            if fields is not None and (set(fields) - JOB_FIELDS or len(fields) > 64):
                raise ValueError('work evidence fields exceed bounds')
        if (self.intended_sha256 or self.actual_sha256) and self.method != 'sealed_sha256':
            raise ValueError('work hashes require sealed content')
        return self


class Operation(Wire):
    operation_id: Identifier
    tool_call_id: Identifier
    tool: Annotated[str, Field(min_length=1, max_length=128)]
    action: Annotated[str, Field(pattern='^[a-z_]{0,32}$', max_length=32)]
    effect: Literal['read', 'execute', 'background', 'write', 'send', 'create', 'edit',
                    'remove', 'dispatch', 'effect_unknown']
    resource_ref: ResourceRef | None = None
    success: bool | None = None
    # Mechanical hook facts; semantic/effect verification remains independent.
    request_id: Identifier | None = None
    model: Identifier | None = None
    error_type: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    failure_kind: Literal['malformed_arguments', 'scope_block', 'guardrail_block', 'plugin_block',
                          'timeout', 'cancellation', 'parse_failure', 'refusal', 'tool_error'] | None = None
    duration_ms: Annotated[float, Field(ge=0)] | None = None
    evidence: Evidence = Field(default_factory=Evidence)
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=64)

    @model_validator(mode='after')
    def known_action(self):
        if self.tool not in ACTION_TOOLS:
            self.action = ''
        return self


class Child(Wire):
    turn_id: Identifier
    parent_turn_id: Identifier


class WorkEnvelope(Wire):
    version: Literal[1]
    root_turn_id: Identifier
    parent_turn_id: Identifier | None = None
    origin_ref: Annotated[str, Field(min_length=1, max_length=1024)] | None = None
    operations: list[Operation] = Field(max_length=256)
    children: list[Child] = Field(max_length=256)
    truncated: bool

    @model_validator(mode='after')
    def unique_ids(self):
        if len({item.operation_id for item in self.operations}) != len(self.operations):
            raise ValueError('duplicate work operation identity')
        if len({item.turn_id for item in self.children}) != len(self.children):
            raise ValueError('duplicate work child identity')
        if len(self.model_dump_json().encode()) > 256 * 1024:
            raise ValueError('work envelope exceeds 256 KiB')
        return self


class WorkEvent(Wire):
    event_id: Identifier
    root_turn_id: Identifier
    session_id: Identifier
    work: WorkEnvelope
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=64)

    @model_validator(mode='after')
    def same_root(self):
        if self.work.root_turn_id != self.root_turn_id:
            raise ValueError('work event root mismatch')
        return self
