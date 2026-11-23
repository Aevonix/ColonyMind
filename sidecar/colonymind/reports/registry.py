"""Every faculty a report covers: its plain name, what it does, the one setting that switches it off and the
paired benchmark arm that measures it (``all-on`` against ``all-on-<faculty>``; docs/PAIRED-AGENT-BENCHMARK.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Faculty:
    key: str
    title: str             # a heading: "Skills"
    noun: str              # inside a sentence: "with skills", "without the status block"
    what: str              # one sentence: what it does when it is on
    setting: str           # the colonymind.yaml setting that switches it off
    arm: Optional[str]     # the benchmark arm that differs from all-on in it alone (without it, or for the feeling
                           # rules, with them in place of the feelings; for initiative, the mind off)

    @property
    def plural(self) -> bool:
        """Whether ``noun`` takes a plural verb ("skills stay on", "the owner gate stays on")."""
        return self.noun in PLURAL_NOUNS

    def verb(self, singular: str, plural_form: str) -> str:
        return plural_form if self.plural else singular

    @property
    def off_instruction(self) -> str:
        if not self.setting:
            return "Context fitting and its caps are mandatory."
        if self.key == "initiative":
            return ("To switch it off, turn the mind off: set mind.enabled to false in colonymind.yaml, or run "
                    "colonymind mind off.")
        return f"To switch it off, set {self.setting} to false in colonymind.yaml."


PLURAL_NOUNS = frozenset({"the drives", "goals", "opinions", "lessons", "the learning rules", "skills", "wishes",
                          "wishes in its own words", "feelings", "the feeling rules", "the memory blocks",
                          "contradictions by meaning", "earlier statements by meaning", "work records"})


def _mind(key: str, title: str, noun: str, what: str) -> Faculty:
    return Faculty(key, title, noun, what, f"mind.faculties.{key}", f"all-on-{key}")


FACULTIES: Tuple[Faculty, ...] = (
    Faculty("initiative", "Initiative", "initiative",
            "The mind acting on its own: the tasks, messages and goals it decides on, asks about or lets go.",
            "mind.enabled", "all-on-initiative"),
    _mind("drives", "Drives", "the drives",
          "Duty, curiosity, mastery, upkeep and social drives set which work comes first, with satiation and goal "
          "adoption; off, every drive weighs the same."),
    _mind("deliberation", "Deliberation", "deliberation",
          "One model call per tick turns an open-ended concern into a task or a goal; off, only templates are used."),
    _mind("goals", "Goals", "goals", "Goals the mind adopts for itself and works through step by step."),
    _mind("broadcast", "Broadcast", "the broadcast",
          "The top three concerns are carried into the owner's turns and into recall."),
    _mind("people", "People", "the people faculty",
          "Check-ins with contacts, messages composed for them, per-contact digests and questions about who is who."),
    _mind("outreach", "Owner outreach", "owner outreach",
          "Messages the mind starts itself to the owner: findings, open loops, care and answers the owner asked for."),
    _mind("owner_gate", "Owner gate", "the owner gate",
          "Every unprompted message to the owner is reviewed first: let through now, made to wait, held for the "
          "digest or dropped."),
    _mind("semantic_recall", "Semantic recall", "semantic recall",
          "Recall by meaning through embeddings; off, recall matches words only."),
    _mind("priors_by_meaning", "Earlier statements by meaning", "earlier statements by meaning",
          "When a message's statements are recorded, earlier records found by meaning are offered beside those that "
          "share its words, so a change said in new words replaces the record it changes; off, only records that "
          "share words are offered."),
    Faculty("blocks", "Memory blocks", "the memory blocks",
            "Every turn carries budgeted text blocks, fresh state first, with a marker for anything held back.",
            "", None),
    Faculty("status", "Status block", "the status block",
            "The owner's turn carries a status block: autonomy level, quiet hours, budgets used, the breaker, "
            "faculties switched off and health problems.", "context.status", "all-on-status"),
    _mind("consolidation", "Nightly consolidation", "the nightly consolidation",
          "Each night the mind settles contradictions (it keeps the newer statement itself), writes per-contact "
          "digests and summarises episodes."),
    _mind("contradictions_by_meaning", "Contradictions by meaning", "contradictions by meaning",
          "Each night two of a person's statements filed under different names that mean the same thing are "
          "compared too, and a pair that disagrees goes to the same replace-or-add decision; off, only statements "
          "under the very same name are compared."),
    _mind("self_narrative", "Self-narrative", "the self-narrative",
          "The mind keeps an account of its recent work, strengths and interests for the owner's turns."),
    _mind("opinions", "Opinions", "opinions", "Views the mind forms from evidence, revises and withdraws."),
    _mind("lessons", "Lessons", "lessons", "What verified results taught the mind, carried into later work."),
    _mind("learning", "Learning rules", "the learning rules",
          "Rules over lessons and remembered statements: a restatement is a witness, not a new item; a changed "
          "preference needs the owner's word; unused items cool."),
    _mind("plasticity", "Plasticity", "plasticity",
          "Small learners compile habits from the record, with live comparisons and automatic rollback; "
          "text keeps facts and authority."),
    _mind("working_memory", "Working memory", "working memory",
          "Finished workers and compressed sessions stay available as branches the owner can consult together."),
    _mind("work_records", "Work records", "work records",
          "Owner requests retain their effects and the evidence that proves them, without a model call."),
    _mind("skills", "Skills", "skills", "Proven lessons are written out as skills the body can load."),
    _mind("wishes", "Wishes", "wishes",
          "Each night the mastery drive may wish for a capability it lacks, asks the owner, and measures the result."),
    _mind("wishes_named", "Wishes in its own words", "wishes in its own words",
          "A wish may carry the model's own words for the capability, shown beside the typed capability only after "
          "a check, and never what a yes gives."),
    _mind("affect", "Feelings", "feelings",
          "The mind's own worry, curiosity, contentment, frustration and load shape priority and strategy."),
    Faculty("affect_rules", "Feeling rules", "the feeling rules",
            "Fixed rules stand in for the running feelings wherever feelings are read.",
            "mind.faculties.affect_rules", "all-on-plus-affect_rules"),
    _mind("drift_check", "Identity check", "the identity check",
          "A battery of identity questions and drift signals, compared from run to run."),
    _mind("senses", "Senses", "the senses faculty",
          "Outside observations raise bounded curiosity and upkeep notes; images remain unverified and only "
          "the owner sees them. With no source posting, this faculty does nothing."),
    _mind("hands", "Hands", "the hands faculty",
          "Durable tasks, delegated attempts, evidence and owner obligations. Until a controller runs, this "
          "faculty does nothing."),
)
BY_KEY: Dict[str, Faculty] = {faculty.key: faculty for faculty in FACULTIES}
MIND_KEYS = tuple(faculty.key for faculty in FACULTIES if faculty.setting.startswith("mind.faculties."))
CONTEXT_KEYS = ("status",)

__all__ = ["BY_KEY", "CONTEXT_KEYS", "FACULTIES", "Faculty", "MIND_KEYS"]
