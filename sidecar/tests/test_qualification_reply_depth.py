"""Malformed model answers and arguments remain failed qualification evidence."""
import pytest

from colonymind.qualification import (
    cases, native_contact_identity, native_interactive, native_memory, native_perspective,
    native_recovery, native_unified, native_worker, public_export, schema_cases, screen_contracts,
)

DEEP = '{"k":' * 8000 + '}'


def test_role_completion_rejects_depth():
    assert cases.json_fields({"output": DEEP}, {"fields": []}) == {"output_is_json": False}

def test_schema_completion_rejects_depth():
    assert schema_cases.assess({"output": DEEP}, {}) == {
        "schema_in_serialized_request": False, "valid_json": False,
        "schema_valid": False, "grounded_fields": False}

def test_screen_completion_rejects_depth(monkeypatch):
    # Exercise this evaluator's second decode independently of json_fields.
    monkeypatch.setattr(screen_contracts, 'json_fields', lambda *args: {"output_is_json": False})
    assert screen_contracts.assess({"output": DEEP}, {"fields": []}) == {
        "output_is_json": False, "schema_in_serialized_request": False, "output_satisfies_schema": False}

def test_native_interactive_rejects_depth():
    assert native_interactive.decoded(DEEP) is None

def test_native_memory_rejects_depth():
    oracle = {"answer": {}, "required_source_ids": [], "forbidden_request_terms": [],
              "no_claim_sources": [], "claim_values": {}, "erased_source_ids": []}
    assert native_memory.assess({"output": DEEP}, oracle)["correct_final_answer"] is False

def test_native_unified_rejects_depth():
    oracle = {"arm": "base_hermes", "answer": {}}
    assert native_unified.assess({"output": DEEP}, oracle)["grounded_unknown_answer"] is False

def test_native_recovery_rejects_depth():
    oracle = {"answer": {}, "reserve_calls": [0, 1], "required_statuses": [], "available": 1, "applied": 0}
    assert native_recovery.assess({"output": DEEP}, oracle)["grounded_final_answer"] is False

def test_native_tool_arguments_reject_depth(tmp_path):
    response = {"messages": [{"tool_calls": [{"id": "one", "function": {
        "name": "read_file", "arguments": DEEP}}]}]}
    assert native_worker.tool_evidence(response, {}, tmp_path)["complete_fixture_reads"] == []

def test_native_perspective_arguments_reject_depth():
    effects = {"tool_results": [{"call": {"name": "colonymind_judgments", "arguments": DEEP}}]}
    assert native_perspective._calls(effects, "read") == []

def test_native_contact_arguments_reject_depth():
    effects = {"tool_results": [{"call": {"name": "colonymind_people", "arguments": DEEP}}]}
    assert native_contact_identity.controls(effects) == []


@pytest.mark.parametrize("fenced", [False, True])
def test_public_output_diagnostics_reject_depth(fenced):
    raw = '```json\n' + DEEP + '\n```' if fenced else DEEP
    result = public_export.output_diagnostics({"output": raw}, {"evaluator": "json_fields"})
    assert result["format_valid"] is False
    assert result["semantic_pass"] is None
