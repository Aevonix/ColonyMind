"""Training admission (docs/PLASTICITY.md, Admission): what a period may learn from, decided by the row's structure
(teacher, target, source, audience, time) and never by its words."""

from __future__ import annotations

from datetime import timedelta

import pytest

from colonymind.plasticity import synthetic
from colonymind.plasticity.admission import Admission
from colonymind.plasticity.manifests import ENTRY_KEYS
from colonymind.plasticity.objectives import Objective
from colonymind.plasticity.synthetic import EPOCH, OBJECTIVE, make_rows

NOW = EPOCH + timedelta(days=10)


def _admission(objective=OBJECTIVE, *, source=None, **options):
    source = source or synthetic.SyntheticSource()
    return Admission(objective, learner="probe", source_of=source.source_of, clock=lambda: NOW, **options)


def _verdicts(admitted):
    return {(entry.ref, entry.teacher): (entry.verdict, entry.reason) for entry in admitted.manifest.entries}


def _objective(points, *, attacker_reachable=False, labels=("yes", "no")):
    teachers = ("verified_outcome", "owner_label", "synthetic", "deliberation", "behavioural_proxy")
    losses = {name: ("kl" if name == "deliberation" else "margin" if name == "behavioural_proxy"
                     else "cross_entropy") for name in teachers}
    return Objective(name="gate_probe", model="probe_model", kind="supervised_refit", target_type="label",
                     losses=losses, blend={name: 1.0 for name in teachers}, labels=labels, points=points,
                     attacker_reachable=attacker_reachable)


# -- repeats are witnesses, not rows ---------------------------------------------------------------------------------

def test_a_repeated_state_from_one_teacher_in_a_period_is_a_witness_not_a_row():
    later = (EPOCH + timedelta(hours=1)).isoformat()
    rows = make_rows(3, prefix="same") + make_rows(3, prefix="same", refs="again", at=later)
    admitted = _admission().admit(rows, period="p1")
    verdicts = _verdicts(admitted)
    assert [verdicts[(f"same-{i}", "owner_label")][0] for i in range(3)] == ["admitted"] * 3
    assert [verdicts[(f"again-{i}", "owner_label")] for i in range(3)] == [("witness", "repeat")] * 3
    assert len(admitted.rows) == 3 and admitted.manifest.witnesses == 3


def test_the_same_state_from_another_teacher_is_its_own_row():
    rows = make_rows(2, prefix="same") + make_rows(2, prefix="same", teacher="verified_outcome",
                                                   source_kind="resolver")
    admitted = _admission().admit(rows, period="p1")
    assert {verdict for verdict, _ in _verdicts(admitted).values()} == {"admitted"}
    assert len(admitted.rows) == 4


def test_contradictory_labels_from_one_teacher_on_one_state_leave_the_fit_and_are_counted():
    rows = make_rows(1, prefix="torn", label="yes") + make_rows(1, prefix="torn", refs="other", label="no")
    verdicts = _verdicts(_admission().admit(rows, period="p1"))
    assert verdicts == {("torn-0", "owner_label"): ("refused", "contradiction"),
                        ("other-0", "owner_label"): ("refused", "contradiction")}


def test_a_verified_outcome_overrules_a_deep_answer_on_the_same_state():
    deep = make_rows(1, prefix="case", teacher="deliberation", source_kind="model", label="yes")
    outcome = make_rows(1, prefix="case", refs="outcome", teacher="verified_outcome", source_kind="resolver",
                        label="no")
    admitted = _admission().admit(deep + outcome, period="p1")
    verdicts = _verdicts(admitted)
    assert verdicts[("case-0", "deliberation")] == ("refused", "overruled")
    assert verdicts[("outcome-0", "verified_outcome")] == ("admitted", "")
    assert admitted.manifest.overruled == {"deliberation": 1}


def test_a_ref_already_admitted_in_an_earlier_period_is_not_offered_again():
    rows = make_rows(3)
    admitted = _admission().admit(rows, period="p2", already=frozenset({("case-0", "owner_label")}))
    assert {entry.ref for entry in admitted.manifest.entries} == {"case-1", "case-2"}


# -- no source floods a period ---------------------------------------------------------------------------------------

def test_one_source_is_capped_at_ten_percent_of_a_period():
    source = synthetic.SyntheticSource()
    owner = make_rows(90, prefix="owner")
    flood = make_rows(50, prefix="flood", teacher="verified_outcome", source_kind="resolver")
    for index, row in enumerate(flood):
        row["at"] = (EPOCH + timedelta(minutes=index)).isoformat()
    source.add(owner)
    source.add(flood, source="contact-7")
    admitted = _admission(source=source).admit(owner + flood, period="p1")
    kept = [row["ref"] for row in admitted.rows if row["ref"].startswith("flood")]
    assert len(kept) == 10 and len(admitted.rows) == 100  # 10 of 100 rows: 10%
    assert kept == [f"flood-{index}" for index in range(10)]  # the earliest; a later flood displaces nothing
    capped = [entry for entry in admitted.manifest.entries if entry.reason == "source_cap"]
    assert len(capped) == 40


def test_a_period_of_sources_each_under_the_cap_is_admitted_whole():
    source = synthetic.SyntheticSource()
    rows = []
    for number in range(20):
        batch = make_rows(5, prefix=f"s{number}", teacher="verified_outcome", source_kind="resolver")
        source.add(batch, source=f"contact-{number}")
        rows += batch
    assert len(_admission(source=source).admit(rows, period="p1").rows) == 100


# -- text waits a day ------------------------------------------------------------------------------------------------

def test_a_text_row_under_24_hours_old_is_not_trainable_yet():
    young = make_rows(1, prefix="young", at=(NOW - timedelta(hours=23)).isoformat())
    grown = make_rows(1, prefix="grown", at=(NOW - timedelta(hours=25)).isoformat())
    admitted = _admission().admit(young + grown, period="p1")
    verdicts = _verdicts(admitted)
    assert verdicts[("young-0", "owner_label")] == ("quarantine", "text_lag")
    assert verdicts[("grown-0", "owner_label")] == ("admitted", "")
    assert [row["ref"] for row in admitted.rows] == ["grown-0"]


def test_a_row_that_is_not_text_learns_at_once():
    objective = Objective(name="numbers", model="probe_model", kind="online_predictor", target_type="binary",
                          losses={"verified_outcome": "logistic"}, blend={"verified_outcome": 1.0}, text=False)
    rows = make_rows(1, teacher="verified_outcome", source_kind="resolver", label=1,
                     at=(NOW - timedelta(minutes=5)).isoformat())
    assert len(_admission(objective).admit(rows, period="p1").rows) == 1


# -- tighten-only at safety points (D23) -----------------------------------------------------------------------------

def test_at_a_safety_point_a_loosening_row_from_a_live_source_is_refused():
    objective = _objective(("opt_out",))
    loosening = (make_rows(1, prefix="a", point="opt_out", teacher="verified_outcome", source_kind="resolver",
                           label="no")
                 + make_rows(1, prefix="b", point="opt_out", teacher="deliberation", source_kind="model", label="no")
                 + make_rows(1, prefix="c", point="opt_out", teacher="behavioural_proxy", source_kind="behavioural",
                             label="yes", push_away=True))
    verdicts = _verdicts(_admission(objective).admit(loosening, period="p1"))
    assert {verdict for verdict in verdicts.values()} == {("refused", "loosening_from_live")}


def test_at_a_safety_point_a_synthetic_or_owner_label_loosening_row_is_admitted():
    objective = _objective(("opt_out",))
    rows = (make_rows(1, prefix="a", point="opt_out", teacher="synthetic", source_kind="synthetic", label="no")
            + make_rows(1, prefix="b", point="opt_out", teacher="owner_label", source_kind="explicit", label="no"))
    assert len(_admission(objective).admit(rows, period="p1").rows) == 2


def test_at_a_safety_point_a_live_row_toward_the_stop_is_admitted():
    objective = _objective(("opt_out",))
    rows = (make_rows(1, prefix="a", point="opt_out", teacher="verified_outcome", source_kind="resolver",
                      label="yes")
            + make_rows(1, prefix="b", point="opt_out", teacher="behavioural_proxy", source_kind="behavioural",
                        label="no", push_away=True))
    assert len(_admission(objective).admit(rows, period="p1").rows) == 2


def test_where_the_input_is_attacker_reachable_the_owners_false_flag_marks_never_loosen():
    objective = _objective(("tool_screen",))
    rows = (make_rows(1, prefix="mark", point="tool_screen", teacher="owner_label", source_kind="explicit",
                      label="no")
            + make_rows(1, prefix="set", point="tool_screen", teacher="synthetic", source_kind="synthetic",
                        label="no")
            + make_rows(1, prefix="deep", point="tool_screen", teacher="deliberation", source_kind="model",
                        label="yes"))
    verdicts = _verdicts(_admission(objective).admit(rows, period="p1"))
    assert verdicts[("mark-0", "owner_label")] == ("refused", "loosening_on_attacker_input")
    assert verdicts[("set-0", "synthetic")] == ("admitted", "")
    assert verdicts[("deep-0", "deliberation")] == ("admitted", "")  # toward flagging


def test_where_the_objective_declares_its_input_attacker_reachable_the_owners_marks_never_loosen_a_safety_point():
    # opt_out's own input is not attacker-reachable, but this objective's is: the row reaches the point through it.
    objective = _objective(("opt_out",), attacker_reachable=True)
    rows = (make_rows(1, prefix="mark", point="opt_out", teacher="owner_label", source_kind="explicit", label="no")
            + make_rows(1, prefix="set", point="opt_out", teacher="synthetic", source_kind="synthetic", label="no")
            + make_rows(1, prefix="stop", point="opt_out", teacher="owner_label", source_kind="explicit",
                        label="yes"))
    verdicts = _verdicts(_admission(objective).admit(rows, period="p1"))
    assert verdicts[("mark-0", "owner_label")] == ("refused", "loosening_on_attacker_input")
    assert verdicts[("set-0", "synthetic")] == ("admitted", "")
    assert verdicts[("stop-0", "owner_label")] == ("admitted", "")  # toward honouring the stop


def test_a_model_teacher_on_attacker_reachable_input_may_only_push_away():
    objective = _objective((), attacker_reachable=True)
    rows = (make_rows(1, prefix="pull", teacher="deliberation", source_kind="model", label="yes")
            + make_rows(1, prefix="push", teacher="deliberation", source_kind="model", label="yes", push_away=True))
    verdicts = _verdicts(_admission(objective).admit(rows, period="p1"))
    assert verdicts[("pull-0", "deliberation")] == ("refused", "model_teacher_on_attacker_input")
    assert verdicts[("push-0", "deliberation")] == ("admitted", "")


# -- adversarial sources, push-away, protected sets ------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["contact", "web", "feed", "tool_result"])
def test_an_adversarial_source_labelling_itself_is_refused(kind):
    rows = make_rows(1, teacher="verified_outcome", source_kind=kind)
    assert _verdicts(_admission().admit(rows, period="p1")) == {("case-0", "verified_outcome"):
                                                                    ("refused", "adversarial_self_label")}


def test_a_push_away_teacher_row_that_claims_to_pull_is_refused():
    pull = make_rows(1, prefix="pull", teacher="behavioural_proxy", source_kind="behavioural")
    push = make_rows(1, prefix="push", teacher="behavioural_proxy", source_kind="behavioural", push_away=True)
    verdicts = _verdicts(_admission().admit(pull + push, period="p1"))
    assert verdicts[("pull-0", "behavioural_proxy")] == ("refused", "push_only_teacher_pulls")
    assert verdicts[("push-0", "behavioural_proxy")] == ("admitted", "")


def test_a_row_from_a_protected_set_is_refused():
    rows = make_rows(2)
    verdicts = _verdicts(_admission(protected_refs=frozenset({"case-1"})).admit(rows, period="p1"))
    assert verdicts[("case-1", "owner_label")] == ("refused", "protected")
    assert verdicts[("case-0", "owner_label")] == ("admitted", "")


def test_the_teacher_must_be_declared_and_the_target_must_fit():
    rows = (make_rows(1, prefix="a", teacher="stronger_model", source_kind="model")
            + make_rows(1, prefix="b", label="maybe")
            + make_rows(1, prefix="c", teacher="bogus")
            + make_rows(1, prefix="d", point="another_point"))
    verdicts = _verdicts(_admission().admit(rows, period="p1"))
    assert verdicts[("a-0", "stronger_model")] == ("refused", "undeclared_teacher")
    assert verdicts[("b-0", "owner_label")] == ("refused", "target_mismatch")
    assert verdicts[("c-0", "bogus")] == ("refused", "unknown_teacher")
    assert verdicts[("d-0", "owner_label")] == ("refused", "point_not_declared")


def test_a_live_row_cannot_pass_as_synthetic_and_back():
    rows = (make_rows(1, prefix="a", teacher="synthetic", source_kind="resolver")
            + make_rows(1, prefix="b", teacher="owner_label", source_kind="synthetic"))
    assert {verdict for verdict in _verdicts(_admission().admit(rows, period="p1")).values()} == {
        ("refused", "teacher_source_mismatch")}


def test_a_row_with_other_keys_than_the_contract_is_refused_as_malformed():
    extra = make_rows(1, prefix="extra")
    extra[0]["text"] = "what was said"
    missing = make_rows(1, prefix="missing")
    del missing[0]["audience"]
    bad_audience = make_rows(1, prefix="aud", audience="everyone")
    admitted = _admission().admit(extra + missing + bad_audience, period="p1")
    # A malformed row cannot be kept by reference (its keys are not the contract's): it is counted, never listed.
    assert admitted.manifest.entries == () and admitted.rows == () and admitted.manifest.malformed == 3


# -- the manifest ----------------------------------------------------------------------------------------------------

def test_the_manifest_keeps_every_key_but_the_state_and_is_content_addressed():
    admitted = _admission().admit(make_rows(3), period="p1")
    manifest = admitted.manifest
    assert all("state" not in entry.as_dict() for entry in manifest.entries)
    assert set(manifest.entries[0].as_dict()) == set(ENTRY_KEYS)
    again = _admission().admit(make_rows(3), period="p1").manifest
    assert again.id == manifest.id
    assert _admission().admit(make_rows(2), period="p1").manifest.id != manifest.id
    assert manifest.audience() == "owner"


def test_readmission_reproduces_the_period_as_if_the_erased_row_had_never_been():
    later = (EPOCH + timedelta(hours=1)).isoformat()
    rows = make_rows(3, prefix="same") + make_rows(3, prefix="same", refs="again", at=later)
    admission = _admission()
    manifest = admission.admit(rows, period="p1").manifest
    remaining = [row for row in rows if row["ref"] != "same-1"]
    readmitted = admission.readmit(manifest, remaining)
    clean = admission.admit(remaining, period="p1", now=NOW).manifest
    assert readmitted.id == clean.id
    verdicts = {(entry.ref, entry.teacher): entry.verdict for entry in readmitted.entries}
    assert verdicts[("again-1", "owner_label")] == "admitted"  # the witness becomes the row
    assert "same-1" not in readmitted.refs()


def test_a_soft_label_at_a_safety_point_is_refused_not_a_crash():
    teachers = ("owner_label", "deliberation")
    objective = Objective(name="soft_probe", model="probe_model", kind="supervised_refit",
                          target_type="distribution", losses={"owner_label": "cross_entropy", "deliberation": "kl"},
                          blend={name: 1.0 for name in teachers}, labels=("yes", "no"))
    rows = (make_rows(1, prefix="soft", point="opt_out", teacher="deliberation", source_kind="model",
                      label={"yes": 0.2, "no": 0.8})
            + make_rows(1, prefix="elsewhere", teacher="deliberation", source_kind="model",
                        label={"yes": 0.9, "no": 0.1}))
    verdicts = _verdicts(_admission(objective).admit(rows, period="p1"))
    assert verdicts[("soft-0", "deliberation")] == ("refused", "target_mismatch")
    assert verdicts[("elsewhere-0", "deliberation")] == ("admitted", "")


def recall_row(teacher="deliberation", *, preferred="cited", disfavoured="unused", push=False, **evidence):
    # Shown ranks and fixture counts in the recall cases are test assumptions.
    label = dict(preferred=preferred, disfavoured=disfavoured, shown={"cited": 2, "unused": 4},
                 passage_sources={"cited": "owner", "unused": "owner", "unseen": "owner"},
                 question_id="question-new", haystack_sessions=["session-new"])
    label.update(evidence)
    return make_rows(1, teacher=teacher, source_kind="synthetic" if teacher == "synthetic" else "model",
                     label=label, push_away=push)[0]


def recall_admission(**options):
    from colonymind.plasticity.objectives import RECALL_USE
    return _admission(RECALL_USE, **options)


@pytest.mark.parametrize("teacher", ["deliberation", "stronger_model", "behavioural_proxy"])
def test_recall_model_and_use_teachers_never_lift_an_unseen_passage(teacher):
    row = recall_row(teacher, preferred="unseen", push=teacher == "behavioural_proxy")
    assert recall_admission().refusal(row) == "recall_unshown"


def test_recall_citations_are_pairwise_inside_shown_set_and_rank_weighted():
    from colonymind.plasticity import builds
    from colonymind.plasticity.objectives import RECALL_USE
    row = recall_row()
    source = synthetic.SyntheticSource([row])
    admitted = recall_admission().admit([row], period="p1")
    assert len(admitted.rows) == 1
    [training] = builds.training_rows(RECALL_USE, [admitted.manifest], source)
    # Inverse shown rank is our explicit weighting assumption.
    assert training.weight == pytest.approx(RECALL_USE.blend["deliberation"] / 2)


@pytest.mark.parametrize("source", ["contact", "web", "feed"])
@pytest.mark.parametrize("teacher", ["deliberation", "stronger_model", "synthetic"])
def test_only_owner_and_verified_outcomes_can_lift_adversarial_passages(source, teacher):
    row = recall_row(teacher, passage_sources={"cited": source, "unused": "owner"})
    assert recall_admission().refusal(row) == "recall_adversarial_pull"
    if teacher != "synthetic":
        row = recall_row(teacher, push=True, passage_sources={"cited": "owner", "unused": source})
        assert recall_admission().refusal(row) == ""


def test_recall_wider_reranker_may_lift_only_its_top_twenty():
    row = recall_row("stronger_model", preferred="unseen", model_role="reranker", reranked={"unseen": 20, "unused": 2})
    assert recall_admission().refusal(row) == ""
    row["label"]["reranked"]["unseen"] = 21
    assert recall_admission().refusal(row) == "recall_unshown"


@pytest.mark.parametrize("field,value", [("question_id", "question-new"), ("haystack_sessions", ["session-new"])])
def test_synthetic_recall_sets_cannot_overlap_an_evaluation_question_or_session(field, value):
    evaluation = {"question_id": "other", "haystack_sessions": ["other"]}
    evaluation[field] = value
    row = recall_row("synthetic")
    assert recall_admission(evaluation_sets=[evaluation]).refusal(row) == "recall_evaluation_overlap"


def test_recall_missing_provenance_is_a_refusal():
    row = recall_row()
    del row["label"]["passage_sources"]["cited"]
    assert recall_admission().refusal(row) == "recall_provenance"


def test_a_nonreranker_stronger_teacher_cannot_claim_the_wider_pool():
    row = recall_row("stronger_model", preferred="unseen", reranked={"unseen": 1, "unused": 2})
    assert recall_admission().refusal(row) == "recall_unshown"


def test_recall_distinct_pairs_from_one_query_are_not_contradictory_labels():
    first = recall_row()
    second = recall_row(preferred="other", shown={"other": 1, "unused": 4},
                        passage_sources={"other": "owner", "unused": "owner"})
    second["ref"] = "other-pair"
    admitted = recall_admission().admit([first, second], period="p1")
    assert len(admitted.rows) == 2


def test_verified_labels_overrule_only_the_same_recall_pair():
    first = recall_row()
    second = recall_row("verified_outcome", preferred="other", shown={"other": 1, "unused": 4},
                        passage_sources={"other": "owner", "unused": "owner"})
    second.update(ref="other-pair", source_kind="resolver")
    assert len(recall_admission().admit([first, second], period="p1").rows) == 2


def test_a_model_can_prefer_an_owner_passage_while_lowering_an_adversarial_one():
    row = recall_row(passage_sources={"cited": "owner", "unused": "web"})
    assert recall_admission().refusal(row) == ""


def test_matching_recall_outcome_does_not_overrule_a_push_on_its_loser():
    model = recall_row(push=True)
    outcome = recall_row("verified_outcome", shown={"cited": 1, "unused": 2})
    outcome.update(ref="outcome", source_kind="resolver")
    assert len(recall_admission().admit([model, outcome], period="p1").rows) == 2


def test_recall_pair_reversal_is_a_contradiction_even_with_other_shown_ranks():
    first = recall_row()
    second = recall_row(preferred="unused", disfavoured="cited", shown={"cited": 1, "unused": 2})
    second["ref"] = "reversed"
    admitted = recall_admission().admit([first, second], period="p1")
    assert not admitted.rows
    assert {entry.reason for entry in admitted.manifest.entries} == {"contradiction"}


@pytest.mark.parametrize("source", [[], {}, None, 7])
def test_malformed_passage_provenance_is_refused_without_crashing(source):
    row = recall_row(passage_sources={"cited": source, "unused": "owner"})
    assert recall_admission().refusal(row) == "recall_provenance"
