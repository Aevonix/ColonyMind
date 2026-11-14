"""Tests for colonymind.intelligence.components: the owner preference learner, its one component."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from colonymind.intelligence.components.preference_learner import (
    Preference,
    PreferenceLearner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph():
    """Mock graph client without advertising a live Neo4j driver."""

    graph = MagicMock()
    # MagicMock fabricates every attribute, which makes InitiativeEngine enter
    # its live-driver loaders and leaves AsyncMock coroutines un-awaited. Unit
    # component tests use in-memory context, so make that boundary explicit.
    del graph.driver
    return graph


# ===========================================================================
# PreferenceLearner
# ===========================================================================


class TestPreferenceModel:
    """Test Preference data model."""

    def test_basic_construction(self):
        p = Preference(
            category="communication_style",
            key="length",
            value="short",
            confidence=0.9,
            learned_from="explicit",
        )
        assert p.category == "communication_style"
        assert p.key == "length"
        assert p.value == "short"
        assert isinstance(p.last_updated, datetime)


class TestPreferenceLearner:
    """Test PreferenceLearner functionality."""

    @pytest.mark.asyncio
    async def test_learn_from_feedback(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        await pl.learn_from_feedback("communication_style", "keep it short")

        # Should store with high confidence
        prefs = await pl.get_all_preferences("communication_style")
        assert len(prefs) == 1
        assert prefs[0].confidence == 0.9
        assert prefs[0].learned_from == "explicit"

    @pytest.mark.asyncio
    async def test_learn_from_behavior_accumulates(self, mock_graph):
        pl = PreferenceLearner(mock_graph)

        # A recognised action ("short") infers communication_style.length=short
        # at implicit confidence (0.5), then accumulates with repetition.
        await pl.learn_from_behavior("clicked_short_response")
        prefs = await pl.get_all_preferences()
        assert len(prefs) == 1
        assert prefs[0].category == "communication_style"
        assert prefs[0].value == "short"
        assert prefs[0].confidence == pytest.approx(0.5)

        await pl.learn_from_behavior("clicked_short_response")
        prefs = await pl.get_all_preferences()
        assert len(prefs) == 1
        assert prefs[0].confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_unrecognised_behavior_stores_nothing(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        await pl.learn_from_behavior("opened_settings_panel")
        assert len(await pl.get_all_preferences()) == 0

    @pytest.mark.asyncio
    async def test_get_all_preferences_for_unknown_category(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        assert await pl.get_all_preferences("nonexistent") == []

    @pytest.mark.asyncio
    async def test_get_all_preferences_after_learn(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        # "concise" maps to the length dimension (short), not a raw 'general' value.
        await pl.learn_from_feedback("communication_style", "concise answers please")

        prefs = await pl.get_all_preferences("communication_style")
        assert {pref.key: pref.value for pref in prefs}["length"] == "short"

    # --- Owner-directive lane + persistence (wired into turn_sync) ----------

    def test_detect_directive_matches_and_ignores(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        assert pl.detect_directive("be more concise") == ("communication_style", "length", "short")
        assert pl.detect_directive("stop using emoji") == ("communication_style", "emoji", "off")
        # A style word with no directive cue is not a directive.
        assert pl.detect_directive("I took a short walk this morning") is None
        # Ordinary conversation never trips it.
        assert pl.detect_directive("what's the weather today") is None

    @pytest.mark.asyncio
    async def test_learn_directive_captures_multiple(self, mock_graph):
        pl = PreferenceLearner(mock_graph)
        primary = await pl.learn_directive("from now on be concise and use bullet points")
        assert primary == ("communication_style", "length", "short")
        brief = pl.build_brief()
        assert "short" in brief.lower()
        assert "bullet" in brief.lower()

    @pytest.mark.asyncio
    async def test_directives_persist_across_restart(self, tmp_path):
        db = str(tmp_path / "prefs.db")
        pl = PreferenceLearner(db_path=db)
        await pl.learn_directive("keep replies short")
        await pl.learn_directive("no emoji please")
        # Reopen from disk — explicit directives survive.
        pl2 = PreferenceLearner(db_path=db)
        brief = pl2.build_brief()
        assert "short" in brief.lower()
        assert "emoji" in brief.lower()


class TestPackageImports:
    """The package exports the preference learner and nothing else."""

    def test_all_imports(self):
        from colonymind.intelligence.components import Preference, PreferenceLearner

        assert callable(PreferenceLearner) and callable(Preference)
