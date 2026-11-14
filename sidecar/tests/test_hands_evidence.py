"""OC4: levels come from checks, independently of effort and self-report."""
from colonymind.hands.evidence import acceptance_outcome, assess, plan_hash
from colonymind.hands.types import Evidence, EvidenceLevel, EvidenceScope, HandState


def receipt(acceptance, **changes):
    value = {'passed': True, 'exit_code': 0, 'output_hash': 'a' * 64,
             'scope': {'kind': 'coding', 'reference': 'head', 'dirty_tree_sha256': 'b' * 64,
                       'command': acceptance.get('test_command'), 'details': {
                           'runner': 'hands-checks-v1', 'plan_hash': plan_hash(acceptance),
                           'artifact_validated': True, 'requirement_passed': True,
                           'current': True, 'tree_sha256': 'c' * 64,
                           'final_binding': {'reference': 'head', 'dirty_tree_sha256': 'b' * 64,
                                             'tree_sha256': 'c' * 64}}}}
    value.update(changes)
    return value


def test_no_receipt_no_promotion_even_when_summary_claims_verified():
    evidence = assess(kind='coding', acceptance={}, receipts=[], tool_digest=[])
    assert evidence.level == EvidenceLevel.E0


def test_plan_receipt_and_final_binding_are_required_for_e3():
    plan = {'test_command': ['pytest']}
    good = receipt(plan)
    assert assess(kind='coding', acceptance=plan, receipts=[good]).level == EvidenceLevel.E3
    good['scope']['details']['current'] = False
    assert assess(kind='coding', acceptance=plan, receipts=[good]).level == EvidenceLevel.E1
    assert assess(kind='coding', acceptance={'test_command': ['other']}, receipts=[receipt(plan)]).level == EvidenceLevel.E0


def test_plain_successful_probe_is_only_execution():
    plan = {'probe': ['true'], 'requirements': ['service=ready']}
    value = receipt(plan, passed=False)
    value['scope'].update(kind='ops', command=None, probe=['true'])
    value['scope']['details'].update(requirement_passed=False, artifact_validated=False)
    assert assess(kind='ops', acceptance=plan, receipts=[value]).level == EvidenceLevel.E1


def test_risk_and_version_acceptance_override_preserves_evidence():
    own = Evidence(level=EvidenceLevel.E0, scope=EvidenceScope(kind='deep'))
    assert acceptance_outcome(own, kind='deep', risk='normal', intent_version=1, current_version=1).state == HandState.DONE
    assert acceptance_outcome(own, kind='deep', risk='high', intent_version=1, current_version=1).state == HandState.BELOW_MINIMUM
    old = acceptance_outcome(own, kind='deep', risk='low', intent_version=1, current_version=2, reason='useful')
    assert old.state == HandState.SUPERSEDED
    override = acceptance_outcome(own, kind='coding', risk='high', intent_version=1, current_version=1, reason='review later')
    assert override.state == HandState.DONE and override.unverified and override.reason == 'review later'
    assert own.level == EvidenceLevel.E0
