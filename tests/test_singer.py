"""SINGER front-end tests (CLAUDE.md §7.4).

These run the external SINGER binary, so they are marked ``slow`` and **skip** when the
binary is unavailable (env ``TSLAI_SINGER`` / the io_singer default path), keeping the
suite portable.
"""
import os

import pytest

import tslai
from tslai.io_singer import DEFAULT_SINGER

singer_missing = not os.path.exists(DEFAULT_SINGER)
needs_singer = pytest.mark.skipif(singer_missing, reason="SINGER binary not available")


@pytest.mark.slow
@needs_singer
def test_singer_tree_sequences_sample_aligned():
    import msprime
    ts = tslai.simulate_admixture(n_admix=6, n_ref=6, sequence_length=4e4,
                                  recombination_rate=1e-8, random_seed=1, Ne=1000, T_split=5000)
    tsm = msprime.sim_mutations(ts, rate=2.5e-7, random_seed=1,
                                model=msprime.BinaryMutationModel())
    samples = tslai.io.singer_tree_sequences(tsm, Ne=1000, mutation_rate=2.5e-7,
                                          recombination_rate=1e-8, n_samples=6, thin=2,
                                          burn_in=2, seed=7)
    assert len(samples) >= 1
    for g in samples:
        assert g.num_samples == tsm.num_samples        # sample count + order preserved
        assert g.num_trees >= 1


@pytest.mark.slow
@needs_singer
def test_singer_ensemble_experiment_runs():
    from tslai.experiments import singer_ensemble_experiment
    r = singer_ensemble_experiment(n_admix=8, n_ref=8, sequence_length=4e4,
                                   mutation_rate=2.5e-7, n_singer=6, thin=2, burn_in=2,
                                   max_iter=4, seed=1)
    assert r["M"] >= 1
    assert 0.0 <= r["merged_balanced"] <= 1.0
    assert set(r["merged_tracks"].keys()) == set(r["queries"])
