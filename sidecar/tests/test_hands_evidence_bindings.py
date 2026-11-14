"""Receipt flags are insufficient without execution and final-artifact bindings."""
from types import SimpleNamespace
from copy import deepcopy

import pytest

from colonymind.hands.evidence import acceptance_outcome, assess, evaluate_result, plan_hash, scope_hash
from colonymind.hands.types import Evidence, EvidenceLevel, EvidenceScope, HandState


def receipt(plan):
    binding = {'reference': 'head', 'dirty_tree_sha256': 'b' * 64, 'tree_sha256': 'c' * 64}
    return {'passed': True, 'exit_code': 0, 'output_hash': 'a' * 64,
            'scope': {'kind': 'coding', 'reference': 'head', 'dirty_tree_sha256': 'b' * 64,
                'command': plan.get('test_command'), 'details': {'runner': 'hands-checks-v1',
                    'plan_hash': plan_hash(plan), 'artifact_validated': True, 'requirement_passed': True,
                    'current': True, 'tree_sha256': 'c' * 64, 'final_binding': binding}}}


@pytest.mark.parametrize('mutation', ['command', 'binding', 'exit', 'hash', 'current'])
def test_claimed_pass_cannot_replace_execution_and_final_binding(mutation):
    plan = {'test_command': ['pytest']}
    value = receipt(plan)
    if mutation == 'command':
        value['scope']['command'] = ['true']
    elif mutation == 'binding':
        value['scope']['details']['final_binding']['tree_sha256'] = 'd' * 64
    elif mutation == 'exit':
        value['exit_code'] = 1
    elif mutation == 'hash':
        value['output_hash'] = None
    else:
        value['scope']['details'].pop('current')
    assert assess(kind='coding', acceptance=plan, receipts=[value]).level in {EvidenceLevel.E0, EvidenceLevel.E1}


def test_observed_earlier_test_cannot_establish_final_artifact():
    assert assess(kind='coding', acceptance={'test_command': ['pytest']}, tool_digest=[
        {'tool': 'terminal', 'exit_code': 0, 'result_hash': 'a' * 64}]).level == EvidenceLevel.E1


def test_e4_checks_all_producer_families_and_exact_scope():
    plan = {'test_command': ['pytest']}
    families = ['sam-route', 'robin-route']
    checked = assess(kind='coding', acceptance=plan, receipts=[receipt(plan)], producer_families=families)
    review = {'verdict': 'pass', 'family': 'independent-route', 'scope_hash': scope_hash(checked.scope)}
    assert assess(kind='coding', acceptance=plan, receipts=[receipt(plan)], producer_families=families,
                  review=review).level == EvidenceLevel.E4
    for family in families:
        review['family'] = family
        assert assess(kind='coding', acceptance=plan, receipts=[receipt(plan)], producer_families=families,
                      review=review).level == EvidenceLevel.E3
    review.update(family='independent-route', scope_hash='wrong')
    assert assess(kind='coding', acceptance=plan, receipts=[receipt(plan)], producer_families=families,
                  review=review).level == EvidenceLevel.E3


def test_critical_review_does_not_replace_applicable_failed_requirement():
    evidence = Evidence(level=EvidenceLevel.E4, scope=EvidenceScope(kind='coding', details={'requirement_passed': False}))
    assert acceptance_outcome(evidence, kind='coding', risk='critical', intent_version=1,
                              current_version=1, test_applies=True).state == HandState.BELOW_MINIMUM


def test_controller_adapter_uses_hand_intent_and_ignores_producer_level_claims():
    plan = {'test_command': ['pytest']}
    body = SimpleNamespace(receipts=[receipt(plan)], tool_digest=[], families_used=['sam-route'],
                           contracts_used=[], summary_ref='branch', summary_hash='a' * 64,
                           evidence={'level': 'E5'})
    task = {'intents': {'1': {'acceptance': plan}, '2': {'acceptance': {'test_command': ['other']}}}, 'intent_version': 2}
    hand = {'kind': 'coding', 'intent_version': 1}
    assert evaluate_result(hand, task, body)['level'] == 'E3'
    body.receipts = []
    assert evaluate_result(hand, task, body)['level'] == 'E0'


def test_research_flags_without_bound_quotes_cannot_promote():
    value = receipt({})
    value['scope'].update(kind='research', reference=None, quotes=[])
    assert assess(kind='research', acceptance={}, receipts=[value]).level == EvidenceLevel.E1


@pytest.mark.parametrize('order', ['old_first', 'old_last'])
def test_conflicting_final_receipt_cannot_reuse_an_earlier_pass(order):
    plan = {'test_command': ['pytest']}
    old = receipt(plan)
    final = deepcopy(old)
    final['passed'] = False
    final['scope']['details'].update(current=False, artifact_validated=False, requirement_passed=False)
    final['scope']['details']['final_binding']['tree_sha256'] = 'd' * 64
    receipts = [old, final] if order == 'old_first' else [final, old]
    assert assess(kind='coding', acceptance=plan, receipts=receipts).level == EvidenceLevel.E1


def test_family_iterators_cannot_hide_producer_family_from_review():
    plan = {'test_command': ['pytest']}
    checked = assess(kind='coding', acceptance=plan, receipts=[receipt(plan)], producer_families=['sam-route'])
    review = {'verdict': 'pass', 'family': 'sam-route', 'scope_hash': scope_hash(checked.scope)}
    assert assess(kind='coding', acceptance=plan, receipts=[receipt(plan)],
                  producer_families=iter(['sam-route']), review=review).level == EvidenceLevel.E3


def test_independent_review_can_check_exact_analysis_without_structural_artifact():
    summary = {'summary_ref': 'branch:analysis', 'summary_hash': 'a' * 64}
    checked = assess(kind='deep', acceptance={}, producer_families=['sam-route'], **summary)
    review = {'verdict': 'pass', 'family': 'robin-route', 'scope_hash': scope_hash(checked.scope)}
    assert checked.level == EvidenceLevel.E0
    assert assess(kind='deep', acceptance={}, producer_families=['sam-route'],
                  review=review, **summary).level == EvidenceLevel.E4
    assert assess(kind='deep', acceptance={}, producer_families=['sam-route'], review=review,
                  summary_ref='branch:changed', summary_hash='b' * 64).level == EvidenceLevel.E0
