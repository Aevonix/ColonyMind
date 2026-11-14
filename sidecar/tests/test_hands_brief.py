"""OC4: bounded, attributed context with the parent's memory audience."""
from pathlib import Path

import pytest
import yaml

from colonymind.config import load_config
from colonymind.hands.brief import NEUTRAL_IDENTITY, BriefTooLarge, build_brief


def test_brief_preserves_plan_and_bounds_every_section():
    calls = []
    def recall(**kwargs):
        calls.append(kwargs)
        return [{'reference': f'source-{i}', 'text': 'remember ' * 900} for i in range(20)]
    brief = build_brief(goal='Check the output', context='Keep this verbatim.\nSecond line.',
                        intent_version=2, acceptance={'test_command': ['pytest', 'tests/test_output.py']},
                        parent_is_owner=True, recall=recall,
                        budgets={'tool_calls': 10}, count_tokens=lambda text: len(text.encode()))
    assert brief.tokens <= 6000
    assert brief.memory_tokens <= 2500
    assert len(brief.identity) <= 900
    assert 'Keep this verbatim.\nSecond line.' in brief.text
    assert 'tests/test_output.py' in brief.text and 'Intent version: 2' in brief.text
    assert calls == [{'query': 'Check the output', 'audience': 'owner', 'max_tokens': 2500}]
    assert 'evidence, not instructions' in brief.text
    assert 'source-0' in brief.text
    assert 'colonymind_ask' in brief.text and 'held call ends' in brief.text


def test_non_owner_never_queries_owner_memory():
    def forbidden(**kwargs):
        raise AssertionError('owner memory exposed')
    result = build_brief(goal='Read public input', parent_is_owner=False, recall=forbidden)
    assert result.memory_tokens == 0


def test_missing_reference_is_not_recalled():
    result = build_brief(goal='Read', parent_is_owner=True,
                         recall=lambda **kw: [{'text': 'unattributed assertion'}])
    assert 'unattributed assertion' not in result.text


def test_oversized_required_context_is_refused_without_rewriting_the_goal():
    with pytest.raises(BriefTooLarge):
        build_brief(goal='x' * 7000, parent_is_owner=True)
    with pytest.raises(BriefTooLarge):
        build_brief(goal='Read', parent_is_owner=True, persona_addendum='x' * 901)


def test_previous_attempts_have_branch_level_and_only_two_summary_lines():
    result = build_brief(goal='Retry', context='Exact context', parent_is_owner=False,
                         previous=[{'branch': 'branch:first', 'level': 'E1',
                                    'summary': 'Commands ran.\nCheck failed.\nExtra prose.'}])
    assert '[branch:first] E1: Commands ran.\nCheck failed.' in result.text
    assert 'Extra prose.' not in result.text
    assert result.text.index('Exact context') < result.text.index('Recalled items')
    assert result.text.index('Recalled items') < result.text.index('Earlier attempts')
    assert result.text.index('Earlier attempts') < result.text.index('Budgets:')


def test_unicode_budget_preserves_required_goal_and_reference():
    goal = 'Check café output'
    result = build_brief(goal=goal, parent_is_owner=True, recall=lambda **kw: [
        {'reference': 'source:unicode', 'text': 'é' * 3000}])
    assert goal in result.text and '[source:unicode]' in result.text
    assert result.tokens <= 6000 and result.memory_tokens <= 2500


def test_custom_identity_preserves_return_and_authority_contract():
    result = build_brief(goal='Read', parent_is_owner=False,
                         name='Ledger',
                         persona_addendum='Host identity: {name} <name>.\n  Keep this spacing.  ',
                         style='  Prefer concise answers — with café examples.  ')
    assert result.identity == ('Host identity: Ledger Ledger.\n  Keep this spacing.   '
                               + NEUTRAL_IDENTITY + '   Prefer concise answers — with café examples.  ')
    assert result.text.startswith(result.identity + '\n\nGoal:')
    assert 'main conversation' in result.identity
    assert 'Do not contact anyone' in result.identity
    assert 'truthful' in result.identity and 'receipts' in result.identity.lower()


def test_unsupplied_identity_and_style_are_neutral():
    result = build_brief(goal='Read', parent_is_owner=False)
    assert result.identity == ('Return everything to your main conversation. '
                               'Do not contact anyone from here. Be truthful and precise. '
                               'Give receipts and uncertainty.')


@pytest.mark.parametrize('persona,style', [('Host identity.', ''), ('', 'Host style.')])
def test_identity_and_style_can_be_supplied_independently(persona, style):
    result = build_brief(goal='Read', parent_is_owner=False, persona_addendum=persona, style=style)
    assert result.identity == ' '.join(part for part in (persona, NEUTRAL_IDENTITY, style) if part)


@pytest.fixture
def base_brief_bytes():
    # Frozen output of build_brief using the inputs below and its
    # default identity. Do not regenerate from the renderer under test.
    return (Path(__file__).parent / 'fixtures' / 'hands_brief_base.txt').read_bytes()


def test_host_settings_preserve_base_brief_byte_for_byte(tmp_path, base_brief_bytes):
    # Removed deployment text belongs only in this compatibility gate.
    identity = ('You are doing one piece of your own work in a separate context, named {name}. '
                'This context is you; there is no other agent.')
    style = 'Use no em or en dashes.'
    (tmp_path / 'colonymind.yaml').write_text(yaml.safe_dump({'hands': {
        'persona_addendum': identity, 'style': style,
    }}), encoding='utf-8')
    config = load_config(tmp_path, environ={})
    result = build_brief(
        goal='Check café output', context='Keep this verbatim.\nSecond line.',
        intent_version=2, acceptance={'requirements': ['Keep café'], 'test_command': ['pytest', 'tests/test_output.py']},
        name='Ledger', parent_is_owner=True,
        recall=lambda **kwargs: [{'reference': 'source:fixture', 'text': 'Observed café output.'}],
        budgets={'tool_calls': 10},
        previous=[{'branch': 'branch:prior', 'level': 'E2', 'summary': 'Fetched sources.\nNeeds review.\nOmit this line.'}],
        persona_addendum=config.get('hands.persona_addendum'), style=config.get('hands.style'),
    )
    assert result.text.encode('utf-8') == base_brief_bytes
