"""Evidence and acceptance from trusted executor receipts, never result prose.

The controller must pass only receipts from its authenticated executor and the
saved criteria for this intent. A field saying ``passed`` is not a credential.
These checks do not make shell-capable hands an authority boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from copy import deepcopy
import hashlib
import json
from typing import Mapping
from urllib.parse import urlsplit

from .types import Evidence, EvidenceLevel, EvidenceScope, HandState


def plan_hash(acceptance: Mapping) -> str:
    return hashlib.sha256(json.dumps(dict(acceptance), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def rank(level) -> int:
    return int(EvidenceLevel(level).value[1:])


def _dict(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    return deepcopy(dict(value))


def _hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _bound(kind, candidate, details, receipt, acceptance) -> bool:
    """Check complete runner bindings, rather than trusting a success flag alone.

    Receipts still require authenticated plugin provenance. A producer that can
    submit arbitrary receipts directly is outside this evaluator's trust seam.
    """
    if (details.get('current') is not True or details.get('timed_out')
            or not _hash(receipt.get('output_hash'))):
        return False
    final = details.get('final_binding')
    if kind in {'coding', 'integrate'}:
        return bool(acceptance.get('test_command')
                    and candidate.get('command') == acceptance['test_command']
                    and receipt.get('exit_code') == 0 and candidate.get('reference')
                    and _hash(candidate.get('dirty_tree_sha256')) and _hash(details.get('tree_sha256'))
                    and final == {'reference': candidate['reference'],
                                  'dirty_tree_sha256': candidate['dirty_tree_sha256'],
                                  'tree_sha256': details['tree_sha256']})
    if kind == 'ops':
        requirements = acceptance.get('requirements') or []
        return bool(acceptance.get('probe') and candidate.get('probe') == acceptance['probe']
                    and receipt.get('exit_code') == 0 and requirements
                    and details.get('matched_requirements') == requirements
                    and candidate.get('sha256') == receipt['output_hash']
                    and candidate.get('reference') == 'probe:' + receipt['output_hash']
                    and final == {'sha256': receipt['output_hash']})
    if kind == 'research':
        quotes = candidate.get('quotes') or []
        return bool(quotes and isinstance(final, dict) and all(
            isinstance(q.get('quote'), str) and q['quote']
            and _hash(q.get('content_hash')) and final.get(q.get('url')) == q['content_hash']
            and isinstance(q.get('start'), int) and q['start'] >= 0
            and q.get('end') == q['start'] + len(q['quote']) for q in quotes))
    artifacts = candidate.get('artifacts') or []
    validation = details.get('validation') or []
    return bool(artifacts and final == artifacts and len(validation) == len(artifacts)
                and all(a.get('reference') and _hash(a.get('sha256')) for a in artifacts))


def _requirement(kind, acceptance, candidate, details) -> bool:
    if details.get('requirement_passed') is not True:
        return False
    if kind == 'research':
        quotes = candidate.get('quotes') or []
        urls = {q['url'] for q in quotes}
        hosts = {(urlsplit(url).hostname or '').removeprefix('www.') for url in urls} - {''}
        required = acceptance.get('required_sources') or 2
        return len(hosts if acceptance.get('independent_sources', True) else urls) >= required
    if kind in {'coding', 'integrate', 'ops'}:
        return True  # Their bound check includes the exact saved command/probe.
    return bool(kind == 'media' and acceptance.get('output_type') and all(
        a.get('kind') == acceptance['output_type'] for a in candidate.get('artifacts', [])))


def assess(*, kind: str, acceptance: Mapping, receipts=(), tool_digest=(),
           producer_families=(), review: Mapping | None = None,
           owner_answer_ref: str | None = None, summary_ref: str | None = None,
           summary_hash: str | None = None, effort: int | None = None) -> Evidence:
    """Evaluate observations from the plugin and bound checks on the final state.

    ``review`` and ``owner_answer_ref`` are controller-owned review/verdict-reader
    outputs, never fields copied from the producer's result. A review must name
    the exact checked artifact. E5 retains all lower-check details.
    """
    level = EvidenceLevel.E0
    scope = EvidenceScope(kind=kind)
    producer_families = sorted(set(producer_families))
    if summary_ref and _hash(summary_hash):
        scope.reference, scope.sha256 = summary_ref, summary_hash
    for raw in tool_digest:
        tool = _dict(raw)
        if (tool.get('exit_code') is not None or tool.get('content_hash') or tool.get('result_hash')):
            level = EvidenceLevel.E1
    checked = [_dict(raw) for raw in receipts]
    checked = [r for r in checked if r.get('scope', {}).get('kind') == kind
               and r['scope'].get('details', {}).get('runner') == 'hands-checks-v1'
               and r['scope']['details'].get('plan_hash') == plan_hash(acceptance)]
    # A result is one final artifact/state. An earlier success cannot survive a
    # contradictory final check, regardless of the receipt list's ordering.
    consistent = all(r['scope']['details'].get('final_binding') == checked[0]['scope']['details'].get('final_binding')
                     and r['scope']['details'].get('current') is True and r.get('passed') is True
                     for r in checked) if checked else True
    for receipt in checked:
        candidate = receipt.get('scope', {})
        details = candidate.get('details', {})
        reached = EvidenceLevel.E1 if _hash(receipt.get('output_hash')) else EvidenceLevel.E0
        bound = _bound(kind, candidate, details, receipt, acceptance)
        if consistent and bound and receipt.get('passed') is True:
            if details.get('artifact_validated') is True:
                reached = EvidenceLevel.E2
            if reached == EvidenceLevel.E2 and _requirement(kind, acceptance, candidate, details):
                reached = EvidenceLevel.E3
        if rank(reached) >= rank(level):
            level = reached
            # Parse nested scope records through the fixed wire model.
            from pydantic import TypeAdapter
            scope = TypeAdapter(EvidenceScope).validate_python(candidate)
    scope.producer_families = list(producer_families)
    reviewable = rank(level) >= 2 or (kind in {'deep', 'plan', 'synthesize'}
                                     and scope.reference and _hash(scope.sha256))
    # Opening an attachment does not independently check the analysis. Retain
    # its exact binding so an eligible reviewer can still attest that scope.
    if kind == 'deep' and effort in {0, 1} and rank(level) > 1:
        level = EvidenceLevel.E1
    if (review and reviewable and review.get('verdict') == 'pass'
            and review.get('family') and producer_families
            and review['family'] not in producer_families
            and review.get('scope_hash') == scope_hash(scope)
            and (kind != 'research' or (scope.quotes and len(review.get('quote_judgments', [])) == len(scope.quotes)
                                        and all(v == 'supports' for v in review['quote_judgments'])))):
        scope.reviewer_family = review['family']
        scope.verdict = 'pass'
        level = EvidenceLevel.E4
    if owner_answer_ref:
        scope.details['owner_answer_ref'] = owner_answer_ref
        level = EvidenceLevel.E5
    return Evidence(level=level, scope=scope)


def evaluate_result(hand, task, body) -> dict:
    """Controller.evidence_evaluator callback using the producer's saved intent.

    Install explicitly on the controller. Its caller still owns acceptance and
    version checks; reviews and owner verdicts must enter through their trusted
    readers, not through producer-supplied result fields.
    """
    acceptance = task['intents'][str(hand['intent_version'])]['acceptance']
    families = set(body.families_used)
    for contract in getattr(body, 'contracts_used', ()):
        family = _dict(contract).get('family')
        if family:
            families.add(family)
    evidence = assess(kind=hand['kind'], acceptance=acceptance, receipts=body.receipts,
                      tool_digest=body.tool_digest, producer_families=sorted(families),
                      summary_ref=body.summary_ref, summary_hash=body.summary_hash,
                      effort=hand.get('effort'))
    if rank(evidence.level) < 2 and not evidence.scope.reference:
        evidence.scope.reference = body.summary_ref
        evidence.scope.sha256 = body.summary_hash
    return asdict(evidence)


def scope_hash(scope: EvidenceScope) -> str:
    return plan_hash(asdict(scope))


@dataclass(frozen=True)
class Acceptance:
    state: HandState
    unverified: bool = False
    reason: str | None = None


def acceptance_outcome(evidence: Evidence, *, kind: str, risk: str,
                       intent_version: int, current_version: int,
                       test_applies: bool = False, reason: str | None = None) -> Acceptance:
    """Fixed risk minimums; an explicit override records uncertainty, not a level."""
    if intent_version != current_version:
        return Acceptance(HandState.SUPERSEDED)
    minimum = {'low': 0, 'normal': 0 if kind in {'deep', 'plan', 'synthesize'} else 1,
               'high': 3 if test_applies else 4, 'critical': 4}[risk]
    enough = rank(evidence.level) >= minimum
    if risk == 'critical' and test_applies:
        enough = enough and evidence.scope.details.get('requirement_passed') is True
    if enough:
        return Acceptance(HandState.DONE, rank(evidence.level) < 3)
    if reason and reason.strip():
        return Acceptance(HandState.DONE, True, reason.strip())
    return Acceptance(HandState.BELOW_MINIMUM, True)
