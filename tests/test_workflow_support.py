"""Tests for the benchmark-workflow support code: migration sim, new metrics, score_full, aggregate.

Offline and fast (no external tools). The VCF-native tspaint painter (which needs tsinfer) is
covered separately in :mod:`test_benchmark_tspaint`.
"""
import csv
import json

import numpy as np
import pytest

import tspaint
from tspaint import metrics
from tspaint.output import Segment, INFORMATIVE
from tspaint import serialize
import tspaint.benchmark as bscore   # score_full / write_metrics / aggregate are package-level


# --- migration demography -------------------------------------------------------------------

def test_migration_demography_adds_symmetric_migration():
    from tspaint.sim import admixture_demography
    d = admixture_demography(Ne=10000, T_admix=100, T_split=2000, migration_rate=1e-4)
    mig = [e for e in d.events if type(e).__name__ == "MigrationRateChange"]
    assert len(mig) == 2 and all(abs(e.rate - 1e-4) < 1e-15 for e in mig)
    # isolated model (default) has no migration events
    d0 = admixture_demography(Ne=10000, T_admix=100, T_split=2000)
    assert not [e for e in d0.events if type(e).__name__ == "MigrationRateChange"]


def test_migration_sim_runs_and_truth_defined():
    ts = tspaint.simulate_admixture(n_admix=4, n_ref=5, sequence_length=2e5, random_seed=3,
                                    Ne=10000, T_admix=100, T_split=2000, f_A=0.5, migration_rate=1e-4)
    truth, _ = tspaint.local_ancestry_truth(ts)
    # every sample's truth tiles [0, L) with no gaps
    for segs in truth.values():
        assert segs[0][0] == 0.0 and segs[-1][1] == ts.sequence_length


# --- proportions ----------------------------------------------------------------------------

def test_proportions():
    tracks = {0: [Segment(0, 100, np.array([0.9, 0.1]), INFORMATIVE),
                  Segment(100, 200, np.array([0.2, 0.8]), INFORMATIVE)]}
    truth = {0: [(0, 100, 0), (100, 200, 1)]}
    assert abs(metrics.global_proportion(tracks, 0) - 0.55) < 1e-9    # (0.9+0.2)/2
    assert metrics.true_proportion(truth, 0) == 0.5


# --- accuracy by true segment size ----------------------------------------------------------

def test_accuracy_by_segment_size_bins():
    # one short true tract (len 50, correctly painted) and one long (len 500, wrongly painted)
    tracks = {0: [Segment(0, 50, np.array([1.0, 0.0]), INFORMATIVE),
                  Segment(50, 550, np.array([1.0, 0.0]), INFORMATIVE)]}
    truth = {0: [(0, 50, 0), (50, 550, 1)]}
    r = metrics.accuracy_by_segment_size(tracks, truth, bins=[1, 100, 1000])
    # bin 0 = [1,100): the short tract -> correct (acc 1); bin 1 = [100,1000): long tract -> wrong (acc 0)
    assert r["n_segments"].tolist() == [1, 1]
    np.testing.assert_allclose(r["accuracy"], [1.0, 0.0])


# --- score_full + aggregate -----------------------------------------------------------------

def _truth_npz(path, truth):
    samp, left, right, state = [], [], [], []
    for s, segs in truth.items():
        for (l, r, st) in segs:
            samp.append(s); left.append(l); right.append(r); state.append(st)
    with open(path, "wb") as f:
        np.savez_compressed(f, _format="tspaint-truth", _version=1,
                            sample=np.array(samp, np.int64), left=np.array(left, float),
                            right=np.array(right, float), state=np.array(state, np.int8))


def test_score_full_perfect_painting(tmp_path):
    L = 1_000_000.0
    truth = {0: [(0, 4e5, 0), (4e5, L, 1)], 1: [(0, 6e5, 1), (6e5, L, 0)]}
    tp = tmp_path / "truth.npz"
    _truth_npz(str(tp), truth)
    perfect = {k: [Segment(l, r, (np.array([1.0, 0.0]) if s == 0 else np.array([0.0, 1.0])),
                           INFORMATIVE) for (l, r, s) in segs] for k, segs in truth.items()}
    pp = tmp_path / "p.npz"
    serialize.save_painting(str(pp), perfect, seqlen=L)

    res = bscore.score_full(str(tp), str(pp), name="perfect", meta={"model": "isolated"},
                            bins=[1e3, 1e5, 1e7])
    assert res["balanced_accuracy"] == 1.0
    assert abs(res["proportion_error"]) < 1e-9
    assert res["switch_ratio"] == 1.0
    # both true tracts are in the [1e5, 1e7) bin and perfectly painted (other bins are empty -> nan)
    import math
    acc = [a for a in res["size_accuracy"] if a is not None and not math.isnan(a)]
    assert acc and all(abs(a - 1.0) < 1e-9 for a in acc)
    assert res["meta"]["model"] == "isolated"


def test_write_and_aggregate(tmp_path):
    L = 1e6
    truth = {0: [(0, 5e5, 0), (5e5, L, 1)]}
    tp = tmp_path / "truth.npz"
    _truth_npz(str(tp), truth)
    # two painters: one perfect, one all-state-0
    perfect = {0: [Segment(0, 5e5, np.array([1.0, 0.0]), INFORMATIVE),
                   Segment(5e5, L, np.array([0.0, 1.0]), INFORMATIVE)]}
    wrong = {0: [Segment(0, L, np.array([1.0, 0.0]), INFORMATIVE)]}
    jsons = []
    for nm, painting in [("perfect", perfect), ("wrong", wrong)]:
        npz = tmp_path / f"{nm}.npz"
        serialize.save_painting(str(npz), painting, seqlen=L)
        res = bscore.score_full(str(tp), str(npz), name=nm, meta={"model": "isolated", "seed": "1"})
        j = tmp_path / f"{nm}.json"
        bscore.write_metrics(str(j), res)
        jsons.append(str(j))

    scalar, size = bscore.aggregate(jsons, str(tmp_path / "summary"))
    rows = list(csv.DictReader(open(scalar)))
    assert {r["painter"] for r in rows} == {"perfect", "wrong"}
    assert {"model", "seed", "painter", "balanced_accuracy", "switch_ratio"} <= set(rows[0])
    perf = next(r for r in rows if r["painter"] == "perfect")
    assert float(perf["balanced_accuracy"]) == 1.0
    # by-size CSV is long-form with bin edges
    srows = list(csv.DictReader(open(size)))
    assert srows and {"size_lo", "size_hi", "accuracy", "painter"} <= set(srows[0])


def test_write_metrics_nan_to_null(tmp_path):
    # empty intersection -> nan fields -> JSON null
    L = 1e6
    truth = {0: [(0, L, 0)]}
    tp = tmp_path / "t.npz"
    _truth_npz(str(tp), truth)
    other = {99: [Segment(0, L, np.array([1.0, 0.0]), INFORMATIVE)]}   # disjoint key
    npz = tmp_path / "o.npz"
    serialize.save_painting(str(npz), other, seqlen=L)
    res = bscore.score_full(str(tp), str(npz), name="x")
    j = tmp_path / "x.json"
    bscore.write_metrics(str(j), res)
    loaded = json.load(open(j))
    assert loaded["balanced_accuracy"] is None and loaded["n_samples"] == 0
