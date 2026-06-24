"""Rung 8a gate (CLAUDE.md §7.3, §9): validation metrics on synthetic inputs."""
import numpy as np

from tslai.output import Segment, INFORMATIVE, MISSING_INFO
from tslai.validate import (map_truth, per_base_accuracy, balanced_accuracy,
                            mean_confidence, reliability_curve, breakpoint_flicker,
                            tract_boundary_error)


def seg(left, right, post, status=INFORMATIVE):
    return Segment(left, right, np.array(post, float), status)


def test_map_truth():
    truth = {0: [(0.0, 5.0, 7), (5.0, 10.0, 9)]}
    mapped = map_truth(truth, {7: 0, 9: 1})
    assert mapped[0] == [(0.0, 5.0, 0), (5.0, 10.0, 1)]


def test_per_base_accuracy():
    tracks = {0: [seg(0, 1, [0.9, 0.1]), seg(1, 2, [0.2, 0.8])]}   # argmax: 0 then 1
    assert per_base_accuracy(tracks, {0: [(0, 1, 0), (1, 2, 1)]}) == 1.0   # both right
    assert per_base_accuracy(tracks, {0: [(0, 2, 0)]}) == 0.5             # 2nd half wrong


def test_accuracy_excludes_missing_info():
    tracks = {0: [seg(0, 1, [0.9, 0.1]), seg(1, 2, [0.9, 0.1], MISSING_INFO)]}
    # only the informative [0,1) counts, and it is correct
    assert per_base_accuracy(tracks, {0: [(0, 1, 0), (1, 2, 1)]}, exclude_missing=True) == 1.0


def test_balanced_accuracy_robust_to_imbalance():
    tracks = {0: [seg(0, 9, [0.9, 0.1]), seg(9, 10, [0.9, 0.1])]}   # paints all state 0
    truth = {0: [(0, 9, 0), (9, 10, 1)]}                            # truth 90% state 0
    assert per_base_accuracy(tracks, truth) == 0.9                  # plain acc is majority-fooled
    assert np.isclose(balanced_accuracy(tracks, truth), 0.5)        # balanced: chance


def test_mean_confidence():
    tracks = {0: [seg(0, 1, [0.5, 0.5]), seg(1, 2, [1.0, 0.0])]}
    assert np.isclose(mean_confidence(tracks), 0.5)                 # mean(|0|, |1|)
    assert mean_confidence({0: [seg(0, 1, [0.5, 0.5])]}) == 0.0     # uninformative


def test_reliability_curve_diagonal_when_calibrated():
    # all positions predict P(0)=0.7; exactly 7/10 of the length is truly state 0
    tracks = {0: [seg(0, 7, [0.7, 0.3]), seg(7, 10, [0.7, 0.3])]}
    truth = {0: [(0, 7, 0), (7, 10, 1)]}
    rc = reliability_curve(tracks, truth, state=0, n_bins=10)
    assert np.allclose(rc["pred"], 0.7)
    assert np.allclose(rc["emp"], 0.7)


def test_breakpoint_flicker():
    tracks = {0: [seg(0, 1, [0.9, 0.1]), seg(1, 2, [0.6, 0.4]), seg(2, 3, [0.2, 0.8])]}
    f = breakpoint_flicker(tracks, 0, state=0)
    assert np.isclose(f["mean_abs_diff"], 0.35)   # mean(|0.9-0.6|, |0.6-0.2|)
    assert f["n_boundaries"] == 2
    assert np.isclose(f["flip_rate"], 0.5)         # argmax 0,0,1 -> one flip of two


def test_tract_boundary_error():
    tracks = {0: [seg(0, 1, [0.9, 0.1]), seg(1, 2, [0.1, 0.9])]}   # inferred switch at 1.0
    truth = {0: [(0, 1.2, 0), (1.2, 2, 1)]}                          # true switch at 1.2
    e = tract_boundary_error(tracks, truth, 0)
    assert e["n_true_switches"] == 1
    assert np.isclose(e["median_error"], 0.2)
