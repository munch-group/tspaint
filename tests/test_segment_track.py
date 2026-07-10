"""SegmentTrack / compare_tracks: plot any per-sample segments (hard tuples or soft Segments) the
same way as Painting.plot(), so tspaint / rfmix / gnomix outputs compare in one style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pytest                        # noqa: E402

from tspaint import SegmentTrack, compare_tracks     # noqa: E402
from tspaint.output import Segment, INFORMATIVE       # noqa: E402
from tspaint.track import _Colorizer                  # noqa: E402


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


def test_default_title_is_summary_stats():
    # SegmentTrack.plot() shows the same performance-stats default title as Painting.plot()
    st = SegmentTrack(_soft(), length=500)
    truth = {0: [(0., 250., 0), (250., 500., 1)], 1: [(0., 500., 0)]}

    _fig, axes = st.plot(return_plot=True)                       # no truth -> proportions + fragmentation
    title = axes[0].get_title(); plt.close("all")
    assert "fragmentation" in title and "sw/Mb" in title and "ancestry" in title
    assert "precision" not in title

    _fig, axes = st.plot(truth=truth, return_plot=True)          # truth -> adds precision / recall
    title_t = axes[0].get_title(); plt.close("all")
    for token in ("precision", "recall", "fragmentation", "ancestry"):
        assert token in title_t

    _fig, axes = st.plot(title="mine", return_plot=True)         # explicit title wins
    assert axes[0].get_title() == "mine"; plt.close("all")

    s = st.summary(truth=truth)                                  # same numbers as a dict
    assert set(s) >= {"proportion", "switch_per_mb", "precision", "recall", "switch_ratio"}


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


def test_plot_samples_filter_and_missing():
    st = SegmentTrack(_soft(), length=500)                       # samples 0, 1
    fig, axes = st.plot(samples=[1], return_plot=True)           # restrict to one row
    assert len(axes) == 1
    plt.close("all")
    with pytest.raises(KeyError):                                # an unpainted sample is rejected
        st.plot(samples=[99])


def test_legend_column_is_uniform_across_K():
    # item 2: a K=2 (colour bar) and a K>2 (categorical legend) plot reserve the SAME legend column,
    # so a painting and a ghost plot default to the same figure layout instead of differing by it.
    from tspaint.track import _LEGEND_W
    st2 = SegmentTrack({0: [(0., 50., 0), (50., 100., 1)]}, length=100)            # K=2
    st4 = SegmentTrack({0: [(0., 50., 0), (50., 100., 3)]}, length=100)            # states 0,3 -> K=4
    for st in (st2, st4):
        _f, axes = st.plot(return_plot=True)
        assert list(axes[0].get_gridspec().get_width_ratios()) == [1, _LEGEND_W]
        plt.close("all")


# --- colour consistency + generalisation to > 2 sources -------------------------------------

def test_colorizer_soft_one_hot_matches_hard_all_K():
    # the consistency guarantee: a confident (one-hot) soft posterior renders as its hard colour,
    # so soft band, hard segments and truth bars agree for every state, in both regimes.
    for K in (2, 3, 5):
        cz = _Colorizer(K)
        for s in range(K):
            assert np.allclose(cz.soft(np.eye(K)[s]), cz.hard(s)), (K, s)


def test_colorizer_regimes_and_backward_compat():
    assert not _Colorizer(2).categorical                 # <=2 states -> diverging colour bar
    assert _Colorizer(3).categorical                     # >2 states  -> categorical palette
    cz = _Colorizer(3)
    assert len({tuple(np.round(cz.hard(s), 3)) for s in range(3)}) == 3   # a distinct hue per state
    # K=2 custom `colors` still builds a diverging colormap (the notebook's usage), not a palette
    assert not _Colorizer(2, colors=["#2D85EE", "#DBDCDE", "#E82C45"]).categorical


def test_plot_generalises_to_three_sources_with_legend():
    soft = {0: [Segment(0, 300, np.array([0.9, 0.05, 0.05]), INFORMATIVE),
                Segment(300, 700, np.array([0.05, 0.9, 0.05]), INFORMATIVE),
                Segment(700, 1000, np.array([0.05, 0.05, 0.9]), INFORMATIVE)]}
    truth = {0: [(0, 300, 0), (300, 700, 1), (700, 1000, 2)]}
    fig, _axes = SegmentTrack(soft, length=1000).plot(truth=truth, return_plot=True)
    legends = [a.get_legend() for a in fig.axes if a.get_legend() is not None]
    assert legends, "a K>2 plot should carry a per-state legend"
    assert [t.get_text() for t in legends[0].get_texts()] == ["A", "B", "C"]
    plt.close("all")


def test_plot_two_sources_keeps_colorbar():
    fig, _axes = SegmentTrack(_soft(), length=500).plot(return_plot=True)
    assert all(a.get_legend() is None for a in fig.axes)                 # no categorical legend
    assert any(a.get_ylabel() == "P(ancestry A)" for a in fig.axes)      # a diverging colour bar
    plt.close("all")


def test_compare_tracks_three_sources_uses_categorical():
    soft = {0: [Segment(0, 500, np.array([0.8, 0.1, 0.1]), INFORMATIVE),
                Segment(500, 1000, np.array([0.1, 0.1, 0.8]), INFORMATIVE)]}
    hard = {0: [(0., 400., 0), (400., 1000., 2)]}                        # states 0 and 2 present
    fig, _axes = compare_tracks({"soft": soft, "hard": hard}, sample=0, length=1000, return_plot=True)
    assert any(a.get_legend() is not None for a in fig.axes)             # categorical legend for K=3
    plt.close("all")


def test_ghost_truth_state_labelled_ghost_not_a_source_letter():
    # a truth state embedded ABOVE the painting's states (a ghost) is labelled "ghost" in the legend,
    # not the next source letter (which would collide with a source/proxy named e.g. "D").
    soft = {0: [Segment(0, 500, np.array([0.9, 0.05, 0.05]), INFORMATIVE),
                Segment(500, 1000, np.array([0.05, 0.05, 0.9]), INFORMATIVE)]}   # 3 painted sources
    truth = {0: [(0, 400, 0), (400, 700, 2), (700, 1000, 3)]}                    # state 3 = the ghost
    fig, _ = SegmentTrack(soft, length=1000).plot(truth=truth, return_plot=True)
    labels = [t.get_text() for a in fig.axes if a.get_legend() for t in a.get_legend().get_texts()]
    assert labels == ["A", "B", "C", "ghost"]
    plt.close("all")


def test_soft_opacity_encodes_confidence():
    # confidence-as-opacity: opacity is 0 at a uniform (no-information) posterior, 1 at a one-hot,
    # monotone in the winning probability, and the shown hue is the argmax state's hue.
    def oneish(K, win):
        v = [(1.0 - win) / (K - 1)] * K
        v[0] = win
        return v
    for K in (2, 3, 4):
        cz = _Colorizer(K)
        assert cz.soft([1.0 / K] * K)[3] == 0.0                          # uniform -> transparent
        assert np.isclose(cz.soft(np.eye(K)[0])[3], 1.0)                 # one-hot  -> opaque
        assert 0.0 < cz.soft(oneish(K, 0.7))[3] < cz.soft(oneish(K, 0.95))[3] <= 1.0   # monotone
    # `alpha` is the max opacity: soft opacity spans [0, alpha], so a fully-confident locus is drawn at
    # exactly `alpha` and alpha<1 fades (a given confidence becomes LESS opaque)
    assert np.isclose(_Colorizer(3, alpha=0.6).soft(np.eye(3)[0])[3], 0.6)
    assert _Colorizer(2, alpha=0.5).soft([0.75, 0.25])[3] < _Colorizer(2, alpha=1.0).soft([0.75, 0.25])[3]
    # `alpha` fades the HARD segments / truth bars too (opacity == alpha), and the soft(one-hot)==hard
    # invariant then holds at ANY alpha
    cz = _Colorizer(3, alpha=0.6)
    assert np.isclose(cz.hard(1)[3], 0.6)
    assert np.allclose(cz.soft(np.eye(3)[1]), cz.hard(1))
    # hue = argmax state
    cz = _Colorizer(3)
    assert np.allclose(cz.soft([0.2, 0.7, 0.1])[:3], cz.hard(1)[:3])
