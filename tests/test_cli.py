"""tspaint CLI (tspaint.cli): the GWF spine round-trips and matches the library."""
import json
import subprocess
import sys

import numpy as np
import pytest
from click.testing import CliRunner

import tspaint
from tspaint.cli import cli, read_labels, read_id_list


def _run(*args):
    res = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res


# --- fast ----------------------------------------------------------------------------------

def test_core_imports_without_click():
    """The core package never imports click (only tspaint.cli does)."""
    code = ("import sys; sys.modules['click'] = None; "
            "import tspaint, tspaint.parallel, tspaint.serialize, tspaint.em, tspaint.output, "
            "tspaint.api; print('OK')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_cli_help():
    res = CliRunner().invoke(cli, ["--help"])
    assert res.exit_code == 0
    for sub in ("fit", "paint", "merge", "date", "qc", "introgress", "ghost", "archaic",
                "simulate", "trees"):
        assert sub in res.output


def test_read_helpers(tmp_path):
    lab = tmp_path / "labels.json"
    lab.write_text(json.dumps({"0": 0, "3": 1}))
    assert read_labels(lab) == {0: 0, 3: 1}
    assert read_id_list("3,4, 5") == [3, 4, 5]
    assert read_id_list(None) is None
    f = tmp_path / "ids.txt"
    f.write_text("7 8\n9\n")
    assert read_id_list(f"@{f}") == [7, 8, 9]


def test_paint_needs_params_or_labels(tmp_path):
    t = tmp_path / "x.trees"
    tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4, random_seed=1).dump(str(t))
    res = CliRunner().invoke(cli, ["paint", str(t), "-o", str(tmp_path / "o.npz")])
    assert res.exit_code != 0 and "params or --labels" in res.output


# --- slow (end-to-end) ----------------------------------------------------------------------

def _truth_dict(path):
    d = np.load(path)
    truth = {}
    for s, l, r, st in zip(d["sample"], d["left"], d["right"], d["state"]):
        truth.setdefault(int(s), []).append((float(l), float(r), int(st)))
    return truth


@pytest.mark.slow
def test_simulate_fit_paint_merge_spine(tmp_path):
    t = tmp_path / "sim.trees"
    labels = tmp_path / "labels.json"
    truth = tmp_path / "truth.npz"
    params = tmp_path / "params.npz"
    m0 = tmp_path / "m0.painting.npz"
    merged = tmp_path / "merged.npz"

    _run("simulate", "--n-admix", 6, "--n-ref", 6, "--length", 1e5, "--seed", 3,
         "--t-admix", 30, "-o", t, "--labels-out", labels, "--truth", truth)
    assert t.exists() and labels.exists() and truth.exists()

    _run("fit", t, "--labels", labels, "-o", params)
    _run("paint", t, "--params", params, "-o", m0)
    _run("merge", m0, "-o", merged)

    # painting loads, covers [0, L), and is accurate vs the simulated truth
    p = tspaint.Painting.load(m0)
    ts_len = p.length
    for q, segs in p.posteriors.items():
        assert segs[0].left == 0.0 and segs[-1].right == ts_len
    bal = tspaint.metrics.balanced_accuracy(p.posteriors, _truth_dict(truth), samples=p.queries)
    assert bal > 0.85                                     # recent admixture, true ARG

    mg = tspaint.Painting.load(merged)
    assert set(mg.posteriors) == set(p.posteriors)
    assert hasattr(next(iter(mg.posteriors.values()))[0], "posterior_std")   # merged band


@pytest.mark.slow
def test_paint_labels_matches_api(tmp_path):
    t = tmp_path / "sim.trees"
    labels = tmp_path / "labels.json"
    out = tmp_path / "p.npz"
    _run("simulate", "--n-admix", 4, "--n-ref", 4, "--length", 8e4, "--seed", 5,
         "-o", t, "--labels-out", labels)
    _run("paint", t, "--labels", labels, "-o", out)

    import tskit
    p_cli = tspaint.Painting.load(out)
    p_api = tspaint.paint(tskit.load(str(t)), read_labels(labels))
    assert p_cli.queries == p_api.queries
    for q in p_api.queries:
        a, b = p_cli.posteriors[q], p_api.posteriors[q]
        assert len(a) == len(b)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x.posterior, y.posterior, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_paint_cores_match(tmp_path):
    t = tmp_path / "sim.trees"
    labels = tmp_path / "labels.json"
    params = tmp_path / "params.npz"
    _run("simulate", "--n-admix", 4, "--n-ref", 4, "--length", 8e4, "--seed", 6,
         "-o", t, "--labels-out", labels)
    _run("fit", t, "--labels", labels, "-o", params)
    o1 = tmp_path / "o1.npz"
    o2 = tmp_path / "o2.npz"
    _run("paint", t, "--params", params, "-j", 1, "-o", o1)
    _run("paint", t, "--params", params, "-j", 2, "-o", o2)
    a, b = tspaint.Painting.load(o1), tspaint.Painting.load(o2)
    for q in a.posteriors:                               # painting is exact across n_jobs
        for x, y in zip(a.posteriors[q], b.posteriors[q]):
            assert x.left == y.left and x.right == y.right
            np.testing.assert_array_equal(x.posterior, y.posterior)


@pytest.mark.slow
def test_date_and_qc_run(tmp_path):
    t = tmp_path / "sim.trees"
    labels = tmp_path / "labels.json"
    _run("simulate", "--n-admix", 6, "--n-ref", 6, "--length", 1e5, "--seed", 7,
         "-o", t, "--labels-out", labels)
    rtt = tmp_path / "rtt.npz"
    qc_out = tmp_path / "qc.npz"
    suspects = tmp_path / "suspects.txt"
    ghost_out = tmp_path / "ghost.npz"
    foreign_out = tmp_path / "foreign.npz"
    _run("date", t, "--labels", labels, "--n-cells", 12, "--n-iter", 5, "-o", rtt)
    _run("qc", t, "--labels", labels, "--soft-refs-out", suspects, "-o", qc_out)
    _run("ghost", t, "--labels", labels, "--depth", "rank", "--max-iter", 15, "-o", ghost_out)
    _run("introgress", t, "--labels", labels, "--samples", "@" + str(suspects),
         "--min-depth", 0.9, "--mode", "fit", "-o", foreign_out)

    from tspaint import serialize
    d = serialize.load_rate_through_time(rtt)
    assert d["centers"].shape == d["q_AB"].shape
    q = serialize.load_reference_qc(qc_out)
    assert len(q["summary"]) == len(read_labels(labels))
    assert suspects.exists()                                # qc emitted the soft-refs id-list
    g = serialize.load_ghost(ghost_out)                     # the HMM result round-trips
    assert "posteriors" in g and "mu" in g
    serialize.load_foreign_tracts(foreign_out)              # the deep-flag output round-trips
