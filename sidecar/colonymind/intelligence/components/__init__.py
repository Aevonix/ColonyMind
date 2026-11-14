"""The owner preference learner: explicit communication directives the owner gave, kept source-backed.

Public API:
    - ``PreferenceLearner`` / ``Preference``: user preference extraction
"""

from .preference_learner import Preference, PreferenceLearner

__all__ = ["Preference", "PreferenceLearner"]
