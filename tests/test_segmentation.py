"""Hard segmentation (deadband) and breakpoint precision/recall (CLAUDE.md §9).

The deadband suppresses low-confidence (~P=0.5) argmax flips that fragment long tracts —
the fix for using tslai's posterior in tract-length / admixture-pulse dating.
"""
import numpy as np
import pytest

from tslai.output import Segment, INFORMATIVE, MISSING_INFO, hard_segments
from tslai.validate import breakpoint_precision_recall, switch_density


def _track(rows):
    return [Segment(l, r, np.array(p, float), st) for (l, r, p, st) in rows]


def test_deadband_suppresses_low_confidence_flip():
    # A long state-0 tract with a single low-confidence (margin 0.1) blip to state 1.
    track = _track([
        (0, 100, [0.9, 0.1], INFORMATIVE),
        (100, 200, [0.45, 0.55], INFORMATIVE),     # argmax=1 but near 0.5
        (200, 300, [0.95, 0.05], INFORMATIVE),
    ])
    assert hard_segments(track, deadband=0.0) == [(0, 100, 0), (100, 200, 1), (200, 300, 0)]
    # deadband 0.3 carries state 0 across the blip -> one tract, no spurious switch
    assert hard_segments(track, deadband=0.3) == [(0, 300, 0)]


def test_deadband_keeps_confident_switch():
    track = _track([(0, 100, [0.9, 0.1], INFORMATIVE), (100, 200, [0.1, 0.9], INFORMATIVE)])
    assert hard_segments(track, deadband=0.5) == [(0, 100, 0), (100, 200, 1)]


def test_missing_info_carries_previous_state():
    track = _track([(0, 100, [0.8, 0.2], INFORMATIVE),
                    (100, 200, [0.5, 0.5], MISSING_INFO)])
    assert hard_segments(track, deadband=0.0) == [(0, 200, 0)]


def test_breakpoint_precision_recall_and_density():
    # inferred has a spurious switch at 50; the real switch at 100 matches truth (at 90).
    inferred = [(0, 50, 0), (50, 100, 1), (100, 200, 0)]
    true = [(0, 90, 1), (90, 200, 0)]
    pr = breakpoint_precision_recall(inferred, true, tol=20)
    assert pr["n_inferred"] == 2 and pr["n_true"] == 1
    assert np.isclose(pr["precision"], 0.5)        # switch@50 spurious, switch@100 real
    assert np.isclose(pr["recall"], 1.0)           # the one true switch (90) is recovered
    assert np.isclose(switch_density(inferred, 200), 2 / 200)
    assert np.isclose(switch_density(true, 200), 1 / 200)


@pytest.mark.slow
def test_fragmentation_experiment_deadband_not_worse_than_argmax():
    from tslai.experiments import fragmentation_experiment
    r = fragmentation_experiment(n_admix=4, n_ref=4, sequence_length=3e5, T_admix=100,
                                 seed=1, include_rfmix=False)
    m = r["methods"]
    assert {"tslai_argmax", "nearest_ref"} <= set(m)
    db = next(k for k in m if k.startswith("tslai_deadband"))
    # the deadband suppresses spurious flips, so it never fragments MORE than raw argmax
    assert m[db]["switches_per_mb"] <= m["tslai_argmax"]["switches_per_mb"] + 1e-9
    for v in m.values():
        assert 0.0 <= v["precision"] <= 1.0 and 0.0 <= v["recall"] <= 1.0
