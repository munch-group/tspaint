"""Serialization round-trips (tspaint.serialize): .npz save/load is exact and reloadable."""
import numpy as np
import pytest

import tspaint
from tspaint import serialize
from tspaint.output import Segment, INFORMATIVE, MISSING_INFO
from tspaint.ensemble import MergedSegment
from tspaint.dating.em import RateThroughTime
from tspaint.introgression import ReferenceQC, GhostResult
from tspaint.archaic import ArchaicResult


def _seg_table():
    return {
        6: [Segment(0.0, 100.0, np.array([0.98, 0.02]), INFORMATIVE),
            Segment(100.0, 250.0, np.array([0.11, 0.89]), INFORMATIVE),
            Segment(250.0, 300.0, np.array([0.5, 0.5]), MISSING_INFO)],
        7: [Segment(0.0, 300.0, np.array([0.7, 0.3]), INFORMATIVE)],
    }


def _merged_table():
    return {
        6: [MergedSegment(0.0, 150.0, np.array([0.9, 0.1]), INFORMATIVE,
                          np.array([0.05, 0.05]), 3),
            MergedSegment(150.0, 300.0, np.array([0.2, 0.8]), INFORMATIVE,
                          np.array([0.1, 0.1]), 2)],
    }


def _assert_tracks_equal(a, b):
    assert a.keys() == b.keys()
    for s in a:
        assert len(a[s]) == len(b[s])
        for x, y in zip(a[s], b[s]):
            assert x.left == y.left and x.right == y.right and x.status == y.status
            np.testing.assert_array_equal(x.posterior, y.posterior)   # bit-exact
            if hasattr(x, "posterior_std"):
                np.testing.assert_array_equal(x.posterior_std, y.posterior_std)
                assert x.n_informative == y.n_informative


def test_params_round_trip(tmp_path):
    p = tmp_path / "params.npz"
    Q = np.array([[-3e-4, 3e-4], [5e-4, -5e-4]])
    pi = np.array([0.5, 0.5])
    w = {3: 0.87, 9: 1.0, 1: 0.123456789}
    labels = {0: 0, 1: 0, 2: 1, 3: 1}
    serialize.save_params(p, Q=Q, pi=pi, w=w, K=2, labels=labels, deadband=0.4,
                          estimate_pi=False, loglik_history=[-10.0, -9.5, -9.4])
    d = serialize.load_params(p)
    np.testing.assert_array_equal(d["Q"], Q)
    np.testing.assert_array_equal(d["pi"], pi)
    assert d["w"] == w and d["labels"] == labels
    assert d["K"] == 2 and d["deadband"] == 0.4 and d["estimate_pi"] is False
    assert d["loglik_history"] == [-10.0, -9.5, -9.4]


def test_painting_round_trip(tmp_path):
    p = tmp_path / "x.painting.npz"
    tracks = _seg_table()
    serialize.save_painting(p, tracks)
    _assert_tracks_equal(serialize.load_painting(p), tracks)
    assert serialize.load_painting_meta(p) == {}     # no model meta when not supplied


def test_painting_merged_round_trip(tmp_path):
    p = tmp_path / "m.painting.npz"
    tracks = _merged_table()
    serialize.save_painting(p, tracks)
    out = serialize.load_painting(p)
    assert all(isinstance(s, MergedSegment) for s in out[6])
    _assert_tracks_equal(out, tracks)


def test_painting_object_save_load(tmp_path):
    p = tmp_path / "full.painting.npz"
    paint = tspaint.Painting(posteriors=_seg_table(), Q=np.array([[-1e-3, 1e-3], [2e-3, -2e-3]]),
                             pi=np.array([0.5, 0.5]), w={3: 0.8}, loglik_history=[-1.0],
                             queries=[6, 7], ts=None, labels={0: 0, 2: 1},
                             default_deadband=0.3, _seqlen=300.0)
    paint.save(p)
    back = tspaint.Painting.load(p)
    assert back.ts is None and back.length == 300.0
    assert back.queries == [6, 7] and back.labels == {0: 0, 2: 1}
    assert back.w == {3: 0.8} and back.default_deadband == 0.3
    np.testing.assert_array_equal(back.Q, paint.Q)
    _assert_tracks_equal(back.posteriors, paint.posteriors)


def test_adversarial_floats_exact(tmp_path):
    p = tmp_path / "adv.painting.npz"
    weird = np.array([0.1, 1e-300, np.pi, 1.0 - 1e-16])
    tracks = {0: [Segment(0.0, 1.0, weird.copy(), INFORMATIVE)]}
    serialize.save_painting(p, tracks)
    out = serialize.load_painting(p)
    np.testing.assert_array_equal(out[0][0].posterior, weird)    # bit-for-bit


def test_rate_through_time_round_trip(tmp_path):
    p = tmp_path / "rtt.npz"
    rtt = RateThroughTime(centers=np.array([10.0, 100.0, 1000.0]),
                          q_AB=np.array([1e-4, 2e-4, 3e-4]), q_BA=np.array([5e-4, 4e-4, 3e-4]),
                          D=np.ones((3, 2)), J=np.zeros((3, 2, 2)), loglik_history=[-3.0, -2.0])
    serialize.save_rate_through_time(p, rtt)
    d = serialize.load_rate_through_time(p)
    np.testing.assert_array_equal(d["centers"], rtt.centers)
    np.testing.assert_array_equal(d["q_AB"], rtt.q_AB)
    np.testing.assert_array_equal(d["q_BA"], rtt.q_BA)
    np.testing.assert_array_equal(d["J"], rtt.J)
    assert d["loglik_history"] == [-3.0, -2.0]


def test_reference_qc_round_trip(tmp_path):
    p = tmp_path / "qc.npz"
    maps = {0: [Segment(0.0, 50.0, np.array([0.9, 0.1]), INFORMATIVE),
                Segment(50.0, 100.0, np.array([0.3, 0.7]), INFORMATIVE)],
            1: [Segment(0.0, 100.0, np.array([0.95, 0.05]), INFORMATIVE)]}
    qc = ReferenceQC(labels={0: 0, 1: 0}, credibility={0: 0.6, 1: 0.95},
                     loo_agreement={0: 0.6, 1: 0.95}, learned_w={0: 0.6},
                     anchors={1}, maps=maps, Q=np.array([[-1e-3, 1e-3], [1e-3, -1e-3]]),
                     pi=np.array([0.5, 0.5]), _length=100.0)
    serialize.save_reference_qc(p, qc, deadband=0.3)
    d = serialize.load_reference_qc(p)
    expect = qc.summary(0.3)
    assert d["summary"] == expect                       # least-credible first, all fields
    _assert_tracks_equal(d["maps"], maps)
    np.testing.assert_array_equal(d["Q"], qc.Q)


def test_ghost_round_trip(tmp_path):
    p = tmp_path / "ghost.npz"
    g = GhostResult(burden={5: 0.1, 6: 0.0}, tracts_by_sample={5: [(10.0, 20.0), (40.0, 55.0)], 6: []})
    serialize.save_ghost(p, g)
    d = serialize.load_ghost(p)
    assert d["burden"] == g.burden
    assert d["tracts_by_sample"][5] == [(10.0, 20.0), (40.0, 55.0)]
    assert d["tracts_by_sample"].get(6, []) == []


def test_archaic_round_trip(tmp_path):
    p = tmp_path / "arch.npz"
    a = ArchaicResult(posteriors={5: [(0.0, 100.0, 0.02), (100.0, 200.0, 0.97)]},
                      burden={5: 0.495}, mu=np.array([1.0, 3.0]), sd=np.array([0.5, 0.4]),
                      A=np.array([[0.98, 0.02], [0.05, 0.95]]), pi0=np.array([0.9, 0.1]),
                      loglik_history=[-5.0, -4.0])
    serialize.save_archaic(p, a)
    d = serialize.load_archaic(p)
    assert d["posteriors"][5] == [(0.0, 100.0, 0.02), (100.0, 200.0, 0.97)]
    assert d["burden"] == a.burden
    np.testing.assert_array_equal(d["mu"], a.mu)
    np.testing.assert_array_equal(d["A"], a.A)


def test_foreign_tracts_round_trip(tmp_path):
    p = tmp_path / "ft.npz"
    tracts = {5: [(10.0, 20.0, 0.8), (30.0, 45.0, 0.62)], 7: [(0.0, 5.0, 0.51)]}
    serialize.save_foreign_tracts(p, tracts)
    assert serialize.load_foreign_tracts(p) == tracts


def test_format_guard(tmp_path):
    p = tmp_path / "params.npz"
    serialize.save_params(p, Q=np.eye(2), pi=np.array([0.5, 0.5]), w={}, K=2, labels={0: 0})
    with pytest.raises(ValueError, match="tspaint-painting"):
        serialize.load_painting(p)          # params file is not a painting
