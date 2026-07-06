"""SINGER front-end tests (CLAUDE.md §7.4).

Most run the external SINGER binary, so they are marked ``slow`` and **skip** when the
binary is unavailable (env ``TSPAINT_SINGER`` / the io_singer default path), keeping the
suite portable. The ``_read_singer_arg`` reader test is binary-free (it feeds synthetic
node/branch/mut text tables) and always runs.
"""
import os

import numpy as np
import pytest

import tspaint
from tspaint.io_singer import DEFAULT_SINGER, _read_singer_arg

singer_missing = not os.path.exists(DEFAULT_SINGER)
needs_singer = pytest.mark.skipif(singer_missing, reason="SINGER binary not available")


def test_read_singer_arg_keeps_sites_across_whole_window(tmp_path):
    """Regression: the SINGER->tskit reader must keep sites across the **whole** window, not just
    the first 1 Mb.

    ``_read_singer_arg`` once hard-coded a ``muts[i, 0] < 1e6`` position cutoff — a mis-port of
    SINGER's ``convert_long_ARG.py`` (which uses ``< length``, the window length). On any region
    wider than 1 Mb it silently dropped every mutation past 1 Mb (and piled them onto the last
    sub-1 Mb site), so e.g. a 2 Mb SINGER run lost ~half its sites. The cutoff must track the
    actual window length (``max(edge.right)``) instead.
    """
    seqlen = 2_000_000
    # two samples (time 0) under one ancestor; edges span the whole [0, seqlen) window.
    np.savetxt(tmp_path / "nodes.txt", np.array([0.0, 0.0, 1234.0]))
    np.savetxt(tmp_path / "branches.txt", np.array([[0.0, seqlen, 2, 0], [0.0, seqlen, 2, 1]]))
    # mutations straddling the old 1e6 cutoff (cols: pos, node, <unused>, derived_state).
    mut_pos = [500.0, 999_999.0, 1_000_500.0, 1_800_000.0]
    muts = np.array([[p, i % 2, 0, 1] for i, p in enumerate(mut_pos)])
    np.savetxt(tmp_path / "muts.txt", muts)

    ts = _read_singer_arg(str(tmp_path / "nodes.txt"), str(tmp_path / "branches.txt"),
                          str(tmp_path / "muts.txt"))

    assert ts.sequence_length == seqlen
    pos = ts.tables.sites.position
    assert ts.num_sites == len(mut_pos)                 # every distinct position kept as a site
    assert np.allclose(np.sort(pos), mut_pos)
    assert int((pos > 1e6).sum()) == 2                  # the regression: upper-half sites survive
    assert ts.num_mutations == len(mut_pos)             # no mutation lost or piled onto one site


@pytest.mark.slow
@needs_singer
def test_singer_tree_sequences_sample_aligned():
    import msprime
    ts = tspaint.simulate_admixture(n_admix=6, n_ref=6, sequence_length=4e4,
                                  recombination_rate=1e-8, random_seed=1, Ne=1000, T_split=5000)
    tsm = msprime.sim_mutations(ts, rate=2.5e-7, random_seed=1,
                                model=msprime.BinaryMutationModel())
    samples = tspaint.io.singer(tsm, _Ne=1000, _m=2.5e-7,
                                _r=1e-8, ts=4, mcmc_step=2,
                                mcmc_burnin=4, _seed=7)
    assert len(samples) >= 1
    for g in samples:
        assert g.num_samples == tsm.num_samples        # sample count + order preserved
        assert g.num_trees >= 1


@pytest.mark.slow
@needs_singer
def test_singer_ensemble_experiment_runs():
    from tspaint.experiments import singer_ensemble_experiment
    r = singer_ensemble_experiment(n_admix=8, n_ref=8, sequence_length=4e4,
                                   mutation_rate=2.5e-7, n_singer=6, thin=2, burn_in=2,
                                   max_iter=4, seed=1)
    assert r["M"] >= 1
    assert 0.0 <= r["merged_balanced"] <= 1.0
    assert set(r["merged_tracks"].keys()) == set(r["queries"])
