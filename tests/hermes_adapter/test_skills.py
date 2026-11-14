"""ColonyMind's skills on stock Hermes (build plan M9 acceptance 5; architecture 4.8 item 4).

With ``skills`` on, the sidecar writes a promoted lesson as ``SKILL.md`` into ColonyMind's own
``skills.external_dirs`` entry and bumps a generation in ``/v1/mind/state``. Hermes caches its skills
prompt index per process with no file times in the key, so the body clears that cache when the
generation changes: a new session sees the skill without a restart. Loads of ColonyMind's skills
(``on_skill_lifecycle``, action ``loaded``) reach ``POST /v1/mind/skills/used``; other skills' do not.
"""

import json

from conftest import MIND_PRELUDE, probe

SKILL = "colonymind-order-codes"
TEXT = """---
name: colonymind-order-codes
description: "When an order code is asked for"
---

# Order codes

Channel letter first.
"""


def configure(home, sidecar):
    sidecar.mind_routes = True
    skills = home.instance / "skills"
    skills.mkdir()
    home.write_config(skills={"external_dirs": [str(skills)]})
    state = home.root / "skills-state.json"
    state.write_text(json.dumps({"generation": 0}))
    sidecar.mind.skills_state = state
    return skills, state


def test_a_promoted_skill_is_visible_to_a_new_session_without_a_restart(home, sidecar):
    skills, state = configure(home, sidecar)
    result = probe(f'''
from pathlib import Path
import agent.prompt_builder as prompt_builder
from agent.prompt_builder import build_skills_system_prompt
clears = []
_clear = prompt_builder.clear_skills_system_prompt_cache
def counting_clear(*args, **kwargs):
    clears.append(1)
    return _clear(*args, **kwargs)
prompt_builder.clear_skills_system_prompt_cache = counting_clear
TOOLS = {{"skills_list", "skill_view"}}
def index():
    return build_skills_system_prompt(available_tools=TOOLS)
tick()                                   # the first generation a process sees is only recorded
before = index()
folder = Path({str(skills)!r}) / {SKILL!r}
folder.mkdir()
(folder / "SKILL.md").write_text({TEXT!r})
tick()                                   # a skill on disk, but no generation change: the cache is kept
cached = index()
clears_before_change = len(clears)
Path({str(state)!r}).write_text(json.dumps({{"generation": 1}}))
tick()                                   # the sidecar reported a change: the body clears the cache
after = index()
emit(before={SKILL!r} in before, after={SKILL!r} in after, clears=[clears_before_change, len(clears)])
''', home, prelude=MIND_PRELUDE)
    # What ColonyMind owes: the sidecar's generation change, and only it, clears Hermes' skills cache, so a promoted
    # skill is visible to a new session without a restart. (Whether Hermes already shows the file before the change
    # is its own cache: 0.21.7 keys external skill dirs by file signature, 0.21.3 does not.)
    assert (result["before"], result["after"], result["clears"]) == (False, True, [0, 1])


def test_a_process_without_the_dispatcher_also_sees_a_promoted_skill(home, sidecar):
    """Hermes caches the skills index per process, and only one process owns the kanban dispatcher: a CLI
    session or a gateway without the dispatcher lock reads the generation on its own body tick too."""
    skills, state = configure(home, sidecar)
    result = probe(f'''
from pathlib import Path
from agent.prompt_builder import build_skills_system_prompt
TOOLS = {{"skills_list", "skill_view"}}
index = lambda: build_skills_system_prompt(available_tools=TOOLS)
first = body.run_once()                  # no dispatcher tick seen here: capture only, never the board
before = index()
folder = Path({str(skills)!r}) / {SKILL!r}
folder.mkdir()
(folder / "SKILL.md").write_text({TEXT!r})
Path({str(state)!r}).write_text(json.dumps({{"generation": 1}}))
second = body.run_once()
emit(before={SKILL!r} in before, after={SKILL!r} in index(), mind=[first["mind"], second["mind"]])
''', home, prelude=MIND_PRELUDE)
    assert result == {"before": False, "after": True, "mind": [False, False]}
    assert sidecar.calls("/v1/mind/dispatch") == []


def test_loads_of_colonymind_skills_reach_the_sidecar_and_others_do_not(home, sidecar):
    skills, _ = configure(home, sidecar)
    folder = skills / SKILL
    folder.mkdir()
    (folder / "SKILL.md").write_text(TEXT)
    result = probe(f'''
from tools.skills_tool import _skill_view_with_bump
viewed = json.loads(_skill_view_with_bump({{"name": {SKILL!r}}}, task_id="t-7", session_id="s-1"))
invoke_hook("on_skill_lifecycle", action="loaded", skill_name="someone-elses-skill", provenance="local",
            task_id="", session_id="s-1", use_count=1, reused=False, reuse_after_patch=False)
invoke_hook("on_skill_lifecycle", action="patched", skill_name="colonymind-other", provenance="external",
            task_id="", session_id="s-1", use_count=None, reused=None, reuse_after_patch=None)
queued = [row["payload"] for row in body.outbox.rows()]
tick()
emit(viewed=viewed.get("success"), queued=queued)
''', home, prelude=MIND_PRELUDE)
    assert result["viewed"] is True
    assert [row.get("kind") for row in result["queued"]] == ["skill_use"]
    assert sidecar.mind.skill_loads == [{"skill": SKILL, "session_id": "s-1", "task_id": "t-7"}]
