"""SegmentTrack / compare_tracks: plot any per-sample segments (hard tuples or soft Segments) the
same way as Painting.plot(), so tspaint / rfmix / gnomix outputs compare in one style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pytest                        # noqa: E402

from tspaint import SegmentTrack, compare_tracks     # noqa: E402
from tspaint.output import Segment, INFORMATIVE       # noqa: E402


HARD = {0: [(0., 100., 0), (100., 300., 1), (300., 500., 0)],
        1: [(0., 250., 1), (250., 500., 0)]}


def _soft():
    return {0: [Segment(0., 250., np.array([0.9, 0.1]), INFORMATIVE),
                Segment(250., 500., np.array([0.2, 0.8]), INFORMATIVE)],
            1: [Segment(0., 500., np.array([0.5, 0.5]), INFORMATIVE)]}


def test_hard_tuples_become_one_hot_segments():
    st = SegmentTrack(HARD, length=500)
    assert st.length == 500.0
    seg = st.posteriors[0][0]
    assert isinstance(seg, Segment) and list(seg.posterior) == [1.0, 0.0]
    assert st.segments()[0] == HARD[0]               # round-trips back to the hard tuples


def test_soft_segments_pass_through_and_infer_length():
    soft = _soft()
    st = SegmentTrack(soft)                            # no length -> inferred from max right
    assert st.length == 500.0
    assert st.posteriors[0][0] is soft[0][0]           # soft Segments passed through unchanged
    assert np.allclose(st.posterior_at(0, 300), [0.2, 0.8])


def test_wraps_a_track_reusing_posteriors_and_length():
    st = SegmentTrack(HARD, length=500)
    st2 = SegmentTrack(st)                            # wrapping a SoftTrack reuses its data + length
    assert st2.length == st.length
    assert st2.segments()[0] == st.segments()[0]


def test_infer_K_from_hard_states():
    st = SegmentTrack({0: [(0., 10., 0), (10., 20., 2)]}, length=20)   # 3 states present
    assert len(st.posteriors[0][0].posterior) == 3


def test_hi_label_and_state_override():
    st = SegmentTrack(HARD, length=500, hi_state=1, hi_label="P(B)")
    assert st._hi_state == 1 and st._hi_label == "P(B)"


def test_plot_runs_for_hard_and_soft():
    SegmentTrack(HARD, length=500).plot(truth={0: [(0., 500., 0)], 1: [(0., 500., 1)]})
    plt.close("all")
    fig, axes = SegmentTrack(_soft()).plot(return_plot=True)
    assert len(axes) == 2                             # one row per sample
    plt.close("all")


def test_compare_tracks_stacks_tools_plus_truth():
    fig, axes = compare_tracks({"tspaint": _soft(), "rfmix": HARD}, sample=0,
                               truth={0: [(0., 250., 0), (250., 500., 1)]},
                               length=500, title="q0", return_plot=True)
    assert len(axes) == 3                             # 2 tools + truth row
    plt.close("all")


def test_compare_tracks_without_truth():
    fig, axes = compare_tracks({"a": HARD, "b": _soft()}, sample=1, length=500, return_plot=True)
    assert len(axes) == 2
    plt.close("all")
