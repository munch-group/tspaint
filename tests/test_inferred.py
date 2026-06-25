"""§9 inferred-ARG validation (CLAUDE.md §9): does tree-native LAI survive
tree-inference error?

Painting on a tsinfer-inferred tree sequence is **bounded by ARG accuracy**. With
enough variants for tsinfer to recover a good ARG, accuracy on the inferred ts is
well above chance; with sparse variants the inferred ARG is poor and accuracy falls
toward chance. Measured at this scenario: ~0.88 (mu=4e-7, ~5400 sites) down to ~0.53
(mu=5e-8, ~650 sites), vs ~1.0 on the true ARG. Tree accuracy is the binding
constraint, not tract length.
"""
import pytest

from tspaint.experiments import admixture_experiment


@pytest.mark.slow
def test_inferred_arg_painting_above_chance_with_dense_data():
    r = admixture_experiment(infer=True, mutation_rate=4e-7, T_admix=300, n_admix=6,
                             n_ref=10, sequence_length=4e5, f_A=0.5, Ne=1000,
                             T_split=5000, max_iter=6, seed=1)
    assert r["inferred"] is True
    assert r["n_sites"] > 2000                 # dense enough for tsinfer to recover the ARG
    assert r["accuracy"] > 0.7                 # measured ~0.88; chance = 0.5
