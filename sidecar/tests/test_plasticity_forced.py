"""The sampling count and shares are design assumptions, measured without a GPU."""
import math

import pytest

from colonymind.plasticity.forced import ForcedRuns


def test_forced_runs_are_keyed_repeatable_and_within_the_99_percent_interval():
    sampler = ForcedRuns(b"fixture sampling key", learner="probe", share=0.05)
    runs = [sampler.choose(str(i), eligible=True) for i in range(10000)]
    count = sum(run.forced for run in runs)
    # Central binomial 99% interval, using its PMF recurrence rather than a normal approximation.
    probability, cumulative, lower, upper = math.exp(10000 * math.log1p(-0.05)), 0.0, None, None
    for successes in range(10001):
        cumulative += probability
        if lower is None and cumulative >= 0.005:
            lower = successes
        if cumulative >= 0.995:
            upper = successes
            break
        probability *= (10000 - successes) / (successes + 1) * 0.05 / 0.95
    assert lower <= count <= upper
    assert runs == [sampler.choose(str(i), eligible=True) for i in range(10000)]
    other = ForcedRuns(b"different fixture key", learner="probe", share=0.05)
    assert runs != [other.choose(str(i), eligible=True) for i in range(10000)]
    assert all(run.manifest_row()["probability"] == 0.05 for run in runs)
    assert all(run.outcome_weight == (20.0 if run.forced else 0.0) for run in runs)
    ordinary = sampler.choose("ordinary", eligible=False)
    assert not ordinary.forced and ordinary.probability == 1.0 and ordinary.outcome_weight == 1.0


@pytest.mark.parametrize("share", [0, -1, 1.1, float("nan"), True])
def test_invalid_sampling_shares_are_refused(share):
    with pytest.raises(ValueError, match="share"):
        ForcedRuns(b"fixture key", learner="probe", share=share)


def test_a_sampling_key_and_learner_are_required():
    with pytest.raises(ValueError):
        ForcedRuns(b"", learner="probe")
    with pytest.raises(ValueError):
        ForcedRuns(b"fixture key", learner="")
