"""Rung 8a gate (CLAUDE.md §7.3, §9): validation metrics on synthetic inputs."""
import numpy as np
import pytest

from tspaint.output import Segment, INFORMATIVE, MISSING_INFO
from tspaint.validate import (map_truth, per_base_accuracy, balanced_accuracy,
                            mean_confidence, reliability_curve, breakpoint_flicker,
                            tract_boundary_error, painting_summary, PaintingSummary)


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


# --- painting_summary (the plot-title read-out) ---------------------------------------------

# Two samples, L=1000: sample 0 is painted perfectly; sample 1 has one spurious extra switch (700).
_SUM_SOFT = {0: [seg(0, 500, [1, 0]), seg(500, 1000, [0, 1])],
             1: [seg(0, 1000, [1, 0])]}
_SUM_HARD = {0: [(0, 500, 0), (500, 1000, 1)],                 # 1 switch @500 (matches truth)
             1: [(0, 300, 0), (300, 700, 1), (700, 1000, 0)]}  # 2 switches @300 (real) + 700 (spurious)
_SUM_TRUTH = {0: [(0, 500, 0), (500, 1000, 1)],                # switch @500
              1: [(0, 300, 0), (300, 1000, 1)]}                # switch @300


def test_painting_summary_no_truth():
    s = painting_summary(_SUM_SOFT, _SUM_HARD, 1000.0)
    assert s["n_samples"] == 2
    assert np.allclose(s["proportion"], [0.75, 0.25])           # soft global fraction per state
    assert np.isclose(s["switch_per_mb"], 3 / (2 * 1000 / 1e6))  # 3 switches over 2 Mb-spans
    # reference-free metrics are always present ...
    assert np.isclose(s["confidence"], 1.0)                     # every segment is one-hot
    assert s["flicker"]["n_boundaries"] == 1                   # only sample 0 has a boundary
    assert np.isclose(s["flicker"]["flip_rate"], 1.0) and np.isclose(s["flicker"]["mean_abs_diff"], 1.0)
    # ... and every truth-dependent metric is absent without truth
    for k in ("precision", "accuracy", "balanced_accuracy", "boundary_error",
              "reliability", "accuracy_by_size"):
        assert k not in s


def test_painting_summary_with_truth():
    s = painting_summary(_SUM_SOFT, _SUM_HARD, 1000.0, truth=_SUM_TRUTH)
    assert np.isclose(s["precision"], 0.75)     # mean over samples: (1.0 + 0.5) / 2
    assert np.isclose(s["recall"], 1.0)         # every true switch recovered
    assert np.isclose(s["switch_ratio"], 1.5)   # 3 inferred / 2 true
    assert np.allclose(s["proportion_true"], [0.4, 0.6])
    # the folded-in tspaint.metrics
    assert np.isclose(s["accuracy"], 0.65)                     # 1300/2000 bp correct (sample 1 half wrong)
    assert np.isclose(s["balanced_accuracy"], (1.0 + 500 / 1200) / 2)  # class 0 perfect, class 1 = 5/12
    assert s["boundary_error"]["n_true_switches"] == 2         # one true switch per sample
    assert set(s["reliability"]) == {"pred", "emp", "weight"}
    assert set(s["accuracy_by_size"]) == {"edges", "accuracy", "weight", "n_segments"}


def test_painting_summary_length_fallback():
    # non-positive length falls back to the painted extent (max hard-segment right)
    s = painting_summary(_SUM_SOFT, _SUM_HARD, 0.0)
    assert np.isclose(s["switch_per_mb"], 3 / (2 * 1000 / 1e6))


# --- PaintingSummary object (repr / attribute access / size-stratified plot) -----------------

def test_painting_summary_is_paintingsummary_dict_compatible():
    s = painting_summary(_SUM_SOFT, _SUM_HARD, 1000.0, truth=_SUM_TRUTH)
    assert isinstance(s, PaintingSummary) and isinstance(s, dict)   # dict subclass -> back-compatible
    assert s["accuracy"] == s.accuracy                             # attribute access mirrors items
    assert s.to_dict() == dict(s) and type(s.to_dict()) is dict
    with pytest.raises(AttributeError):
        _ = s.does_not_exist


def test_painting_summary_repr_formatted_and_conditional():
    r = repr(painting_summary(_SUM_SOFT, _SUM_HARD, 1000.0, truth=_SUM_TRUTH))
    assert r.startswith("PaintingSummary(n_samples=2)")
    for tok in ("ancestry proportion", "accuracy", "precision", "fragmentation", "size-stratified"):
        assert tok in r
    r0 = repr(painting_summary(_SUM_SOFT, _SUM_HARD, 1000.0))       # no truth -> truth-only rows gone
    assert "confidence" in r0 and "accuracy" not in r0 and "precision" not in r0


def test_plot_size_stratified():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    soft = {0: [seg(0, 5000, [1, 0]), seg(5000, 20000, [0, 1])]}
    hard = {0: [(0, 5000, 0), (5000, 20000, 1)]}                    # tracts 5 kb + 15 kb populate bins
    s = painting_summary(soft, hard, 20000.0, truth={0: [(0, 5000, 0), (5000, 20000, 1)]})
    fig, ax = s.plot_size_stratified(return_plot=True)
    assert ax.get_xscale() == "log" and ax.get_ylabel() == "per-base accuracy"
    plt.close("all")
    with pytest.raises(ValueError):                                 # no truth -> nothing to stratify
        painting_summary(soft, hard, 20000.0).plot_size_stratified()
