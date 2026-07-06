"""Bit-exact process parallelism (tspaint.parallel): the contract in the module docstring."""
import os
from functools import reduce

import numpy as np
import pytest

import tspaint
from tspaint import parallel
from tspaint.accumulate import accumulate_sufficient_statistics
from tspaint.em import build_emissions, fit
from tspaint.model import make_generator_2state
from tspaint.output import posterior_table
from tspaint.sim import SOURCE_A, SOURCE_B


def _setup(L=1e5, seed=1):
    ts = tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=L, recombination_rate=1e-8,
                                    random_seed=seed, Ne=1000, T_admix=30, T_split=5000, f_A=0.5)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    Q = make_generator_2state(1e-3, 1e-3)
    pi = np.array([0.5, 0.5])
    w = {}
    emissions = build_emissions(ts, labels, w, pi)
    return ts, labels, Q, pi, w, emissions


# --- non-process tests (fast) ---------------------------------------------------------------

def test_resolve_cores(monkeypatch):
    for v in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "TSPAINT_CORES"):
        monkeypatch.delenv(v, raising=False)
    assert parallel.resolve_cores(4) == 4            # explicit wins
    assert parallel.resolve_cores(None) == (os.cpu_count() or 1)   # default -> all CPUs
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "8")
    assert parallel.resolve_cores() == 8
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "4(x2),3")   # compact SLURM form
    assert parallel.resolve_cores() == 11
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    assert parallel.resolve_cores() == 6             # per-task preferred over per-node
    assert parallel.resolve_cores(2) == 2            # explicit still wins


def test_genome_chunks_partition():
    ts, *_ = _setup()
    T = ts.num_trees
    chunks = parallel.genome_chunks(ts, 4)
    assert chunks[0][0] == 0 and chunks[-1][1] == T
    assert all(hi > lo for lo, hi in chunks)
    for i in range(1, len(chunks)):
        assert chunks[i][0] == chunks[i - 1][1]       # contiguous, no gaps/overlap
    assert sum(hi - lo for lo, hi in chunks) == T


def test_chunks_bank_every_edge_once():
    """Each edges_in event (hence each edge's span-weighted contribution) lands in one chunk."""
    ts, *_ = _setup()
    counts = [len(ein) for (_iv, _eo, ein) in ts.edge_diffs()]
    chunks = parallel.genome_chunks(ts, 4)
    banked = sum(sum(counts[lo:hi]) for lo, hi in chunks)
    assert banked == sum(counts) == ts.num_edges      # exactly-once coverage


def test_accumulate_njobs1_byte_identical():
    ts, labels, Q, pi, w, emissions = _setup()
    legacy = accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels)
    got = parallel.accumulate_parallel(ts, Q, pi, w=w, labels=labels, n_jobs=1)
    np.testing.assert_array_equal(got.S_dwell, legacy.S_dwell)
    np.testing.assert_array_equal(got.S_jumps, legacy.S_jumps)
    np.testing.assert_array_equal(got.S_root, legacy.S_root)
    assert got.loglik == legacy.loglik


def test_tree_range_partition_reduces_to_full():
    """Serial proof of the mechanic (no processes): the chunk partition folds to the full loop."""
    ts, labels, Q, pi, w, emissions = _setup()
    legacy = accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels)
    chunks = parallel.genome_chunks(ts, 5)
    folded = reduce(parallel.add_suffstats,
                    [accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                                      tree_range=c) for c in chunks])
    # vs the single loop: only the reduction order differs -> a few ULP (relative).
    np.testing.assert_allclose(folded.S_dwell, legacy.S_dwell, rtol=1e-12, atol=0)
    np.testing.assert_allclose(folded.S_jumps, legacy.S_jumps, rtol=1e-12, atol=0)
    np.testing.assert_allclose(folded.S_root, legacy.S_root, rtol=1e-12, atol=0)


def test_as_path_reuse_and_temp(tmp_path):
    ts, *_ = _setup(L=5e4)
    p = tmp_path / "x.trees"
    ts.dump(str(p))
    with parallel.as_path(str(p)) as path:
        assert path == str(p)                         # string path reused, not re-dumped
    with parallel.as_path(ts) as path:                # in-memory ts -> temp .trees
        assert path.endswith(".trees") and os.path.exists(path)
    assert not os.path.exists(path)                   # cleaned up on exit


# --- process-pool tests (slow) --------------------------------------------------------------

@pytest.mark.slow
def test_accumulate_parallel_bitexact_and_close():
    ts, labels, Q, pi, w, emissions = _setup()
    chunks = parallel.genome_chunks(ts, 4)
    ref = reduce(parallel.add_suffstats,
                 [accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels,
                                                   tree_range=c) for c in chunks])
    got = parallel.accumulate_parallel(ts, Q, pi, w=w, labels=labels, n_jobs=4)
    np.testing.assert_array_equal(got.S_dwell, ref.S_dwell)     # == same-chunking serial, bit-for-bit
    np.testing.assert_array_equal(got.S_jumps, ref.S_jumps)
    np.testing.assert_array_equal(got.S_root, ref.S_root)
    legacy = accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels)
    np.testing.assert_allclose(got.S_dwell, legacy.S_dwell, rtol=1e-12, atol=0)   # ~ULP vs single loop
    np.testing.assert_allclose(got.S_jumps, legacy.S_jumps, rtol=1e-12, atol=0)


@pytest.mark.slow
def test_accumulate_exact_is_serial():
    ts, labels, Q, pi, w, emissions = _setup()
    legacy = accumulate_sufficient_statistics(ts, Q, pi, emissions, labels=labels)
    got = parallel.accumulate_parallel(ts, Q, pi, w=w, labels=labels, n_jobs=4, exact=True)
    np.testing.assert_array_equal(got.S_dwell, legacy.S_dwell)  # exact -> byte-identical to legacy
    np.testing.assert_array_equal(got.S_jumps, legacy.S_jumps)


@pytest.mark.slow
def test_posterior_table_parallel_exact():
    ts, labels, Q, pi, w, emissions = _setup()
    queries = [int(s) for s in ts.samples() if int(s) not in labels]
    serial = posterior_table(ts, Q, pi, emissions, focal=queries)
    par = parallel.posterior_table_parallel(ts, Q, pi, w=w, labels=labels, focal=queries, n_jobs=4)
    assert serial.keys() == par.keys()
    for q in queries:
        assert len(serial[q]) == len(par[q])
        for a, b in zip(serial[q], par[q]):
            assert a.left == b.left and a.right == b.right and a.status == b.status
            np.testing.assert_array_equal(a.posterior, b.posterior)   # painting is exact, any P


@pytest.mark.slow
def test_loo_posterior_table_parallel_exact():
    from tspaint.output import loo_posterior_table
    ts, labels, Q, pi, w, emissions = _setup()
    refs = list(labels)                                            # LOO map is per reference
    serial = loo_posterior_table(ts, Q, pi, emissions, focal=refs)
    par = parallel.loo_posterior_table_parallel(ts, Q, pi, w=w, labels=labels, focal=refs, n_jobs=4)
    assert serial.keys() == par.keys()
    for q in refs:
        assert len(serial[q]) == len(par[q])
        for a, b in zip(serial[q], par[q]):
            assert a.left == b.left and a.right == b.right and a.status == b.status
            np.testing.assert_array_equal(a.posterior, b.posterior)   # leave-one-out is exact, any P


@pytest.mark.slow
def test_fit_parallel_close_to_serial():
    ts, labels, _Q, _pi, _w, _e = _setup()
    Q0 = make_generator_2state(1e-3, 1e-3)
    r1 = fit(ts, labels, Q0=Q0, max_iter=4, tol=0.0, estimate_pi=False, n_jobs=1)
    r4 = fit(ts, labels, Q0=Q0, max_iter=4, tol=0.0, estimate_pi=False, n_jobs=4)
    np.testing.assert_allclose(r4.Q, r1.Q, rtol=1e-9, atol=0)              # ULP accumulated over EM
    np.testing.assert_array_equal(r4.pi, r1.pi)                            # estimate_pi=False -> fixed
    np.testing.assert_allclose(np.array(r4.loglik_history), np.array(r1.loglik_history),
                               rtol=1e-12, atol=0)


# --- n_jobs default resolution: all CPUs, or the SLURM allocation ------------------------------

def test_resolve_cores_default_is_cpus_or_slurm(monkeypatch):
    import os
    from tspaint.parallel import resolve_cores
    for v in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "TSPAINT_CORES"):
        monkeypatch.delenv(v, raising=False)
    ncpu = os.cpu_count() or 1
    assert resolve_cores(None) == ncpu and resolve_cores() == ncpu   # default -> all CPUs
    assert resolve_cores(3) == 3 and resolve_cores(1) == 1           # explicit wins
    # SLURM_JOB_CPUS_PER_NODE respected, including the compact N(xM) form
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "8")
    assert resolve_cores(None) == 8
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "4(x2)")
    assert resolve_cores(None) == 8
    assert resolve_cores(2) == 2                                     # explicit still wins over SLURM
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")                   # cleaner per-task value wins
    assert resolve_cores(None) == 5


def test_public_n_jobs_defaults_are_none():
    """Every public entry point defaults n_jobs to None (-> resolve_cores -> all CPUs / SLURM)."""
    import inspect
    import tspaint
    from tspaint import em
    for fn in (tspaint.paint, tspaint.fit, tspaint.reference_qc, tspaint.foreign_tracts,
               tspaint.detect_ghost):
        assert inspect.signature(fn).parameters["n_jobs"].default is None, fn.__name__
    assert inspect.signature(tspaint.Painting.rate_through_time).parameters["n_jobs"].default is None
