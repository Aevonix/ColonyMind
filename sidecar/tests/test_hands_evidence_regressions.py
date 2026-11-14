"""Receipt consistency and independent review must gate every promotion."""
from copy import deepcopy

import pytest

from colonymind.hands.evidence import assess, scope_hash
from colonymind.hands.types import EvidenceLevel
from test_hands_evidence import receipt


PLAN = {'test_command': ['pytest']}


def bound_receipt():
    row = receipt(PLAN)
    scope = row['scope']
    scope['details']['final_binding'] = {
        'reference': scope['reference'], 'dirty_tree_sha256': scope['dirty_tree_sha256'],
        'tree_sha256': scope['details']['tree_sha256'],
    }
    return row


@pytest.mark.parametrize('change', ['artifact', 'exit', 'timeout', 'binding'])
def test_inconsistent_success_flags_do_not_grant_e3(change):
    row = bound_receipt()
    details = row['scope']['details']
    if change == 'artifact':
        details['artifact_validated'] = False
    elif change == 'exit':
        row['exit_code'] = 1
    elif change == 'timeout':
        details['timed_out'] = True
    else:
        details['final_binding']['reference'] = 'other-head'
    assert assess(kind='coding', acceptance=PLAN, receipts=[row]).level != EvidenceLevel.E3


def test_reviewer_family_must_differ_from_every_producer_fallback():
    row = bound_receipt()
    families = ['family-a', 'family-b']
    base = assess(kind='coding', acceptance=PLAN, receipts=[row], producer_families=families)
    review = {'verdict': 'pass', 'family': 'family-b', 'scope_hash': scope_hash(base.scope)}
    assert assess(kind='coding', acceptance=PLAN, receipts=[row],
                  producer_families=families, review=review).level != EvidenceLevel.E4
    review['family'] = 'family-c'
    assert assess(kind='coding', acceptance=PLAN, receipts=[row],
                  producer_families=families, review=review).level == EvidenceLevel.E4


def test_review_cannot_attest_a_different_artifact():
    row = bound_receipt()
    families = ['family-a']
    base = assess(kind='coding', acceptance=PLAN, receipts=[row], producer_families=families)
    review = {'verdict': 'pass', 'family': 'family-b', 'scope_hash': scope_hash(base.scope)}
    different = deepcopy(row)
    different['scope']['reference'] = 'other-head'
    assert assess(kind='coding', acceptance=PLAN, receipts=[different],
                  producer_families=families, review=review).level != EvidenceLevel.E4
