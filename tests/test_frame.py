"""Painting/SoftTrack DataFrame exports: posteriors_as_frame (wide) and segments_as_frame (tracts)."""
import numpy as np

from tspaint import SegmentTrack
from tspaint.api import Painting
from tspaint.output import Segment, INFORMATIVE, MISSING_INFO


def _soft():
    return {0: [Segment(0., 250., np.array([0.9, 0.1]), INFORMATIVE),
                Segment(250., 500., np.array([0.2, 0.8]), INFORMATIVE)],
            1: [Segment(0., 500., np.array([0.5, 0.5]), MISSING_INFO)]}   # isolated -> missing-info


def _painting(posteriors=None):
    posteriors = _soft() if posteriors is None else posteriors   # {} stays empty (not falsy-swapped)
    return Painting(posteriors=posteriors,
                    Q=np.array([[-1e-3, 1e-3], [1e-3, -1e-3]]), pi=np.array([0.5, 0.5]),
                    w={}, loglik_history=[], queries=list(posteriors),
                    ts=None, labels=None, default_deadband=0.4, _seqlen=500.0)


def test_posteriors_as_frame_wide_layout():
    df = _painting().posteriors_as_frame()
    assert list(df.columns) == ["haplotype", "left", "right", "A", "B", "status"]
    assert len(df) == 3                                        # 2 segments (hap 0) + 1 (hap 1)
    row = df[(df.haplotype == 0) & (df.left == 250.0)].iloc[0]
    assert np.isclose(row.A, 0.2) and np.isclose(row.B, 0.8) and row.right == 500.0
    assert np.allclose((df["A"] + df["B"]).to_numpy(), 1.0)   # per-state columns are a distribution
    # a missing-info span is tagged, not silently a genuine 50-50 informative call (CLAUDE.md §4.2)
    assert df[df.haplotype == 1].iloc[0].status == MISSING_INFO


def test_segments_as_frame_hard_tracts():
    df = _painting().segments_as_frame()
    assert list(df.columns) == ["haplotype", "start", "end", "ancestry"]
    hap0 = df[df.haplotype == 0]                               # confident A|B switch at 250 (deadband 0.4)
    assert list(hap0.ancestry) == ["A", "B"]
    assert list(hap0.start) == [0.0, 250.0] and list(hap0.end) == [250.0, 500.0]


def test_segments_as_frame_deadband_override():
    # a high dead-band suppresses the switch -> hap 0 collapses to one tract
    assert len(_painting().segments_as_frame(deadband=0.99)[lambda d: d.haplotype == 0]) == 1


def test_frame_methods_inherited_by_segmenttrack_with_samples_filter():
    st = SegmentTrack(_soft(), length=500)                     # methods live on SoftTrack -> inherited
    df = st.posteriors_as_frame(samples=[0])
    assert set(df.haplotype.unique()) == {0}                   # only the requested haplotype
    assert list(df.columns) == ["haplotype", "left", "right", "A", "B", "status"]
    sdf = st.segments_as_frame(samples=[1])
    assert set(sdf.haplotype.unique()) == {1}


def test_posteriors_as_frame_ensemble_adds_std_columns():
    from tspaint.ensemble import MergedSegment
    post = {0: [MergedSegment(0., 500., np.array([0.7, 0.3]), INFORMATIVE,
                              np.array([0.05, 0.06]), 5)]}
    df = _painting(post).posteriors_as_frame()
    assert list(df.columns) == ["haplotype", "left", "right", "A", "B", "A_std", "B_std", "status"]
    assert np.isclose(df.iloc[0].A_std, 0.05) and np.isclose(df.iloc[0].B_std, 0.06)


def test_frames_empty_painting():
    df = _painting({}).posteriors_as_frame()
    assert list(df.columns) == ["haplotype", "left", "right", "A", "B", "status"] and len(df) == 0
    assert list(_painting({}).segments_as_frame().columns) == ["haplotype", "start", "end", "ancestry"]
