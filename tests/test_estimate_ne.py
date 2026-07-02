"""Estimating SINGER's Ne from between-individual nucleotide diversity (io_genotypes.estimate_ne).

Ne = pi / (4*mu), where pi is the mean per-bp difference between haplotypes of *different*
individuals, over sites called in both.
"""
import os

import numpy as np
import pytest

import tspaint
from tspaint import io
from tspaint.io_genotypes import Variants, estimate_ne, variants_from_vcf
from tspaint.io_singer import DEFAULT_SINGER


def _panmictic(Ne, mu, L=2e6, n=25, seed=3):
    import msprime
    ts = msprime.sim_ancestry(samples=n, population_size=Ne, sequence_length=L,
                              recombination_rate=1e-8, random_seed=seed, ploidy=2)
    return msprime.sim_mutations(ts, rate=mu, random_seed=seed,
                                 model=msprime.BinaryMutationModel())


def _diploid_variants(ts, L):
    G = ts.genotype_matrix().astype(np.int8)
    H = G.shape[1]
    return Variants(np.asarray(ts.tables.sites.position), G, [("0", "1")] * G.shape[0], float(L),
                    sample_names=[f"s{i // 2}_{i % 2}" for i in range(H)], ploidy=2)


def _structured_variants():
    """Diploid a,b (reference 0) and c,d (reference 1): heavy *fixed* divergence between the two
    references, light polymorphism within each. So all-pairs pi is dominated by the between-reference
    differences, but same-reference (groups) pi sees only the within-reference polymorphism."""
    def blk(a, b, c, d, n):                        # n identical sites; each arg is a 2-homolog tuple
        return np.tile(np.array([*a, *b, *c, *d], np.int8), (n, 1))
    G = np.vstack([blk((0, 0), (1, 1), (0, 0), (0, 0), 10),    # within-ref-0 polymorphism (a vs b)
                   blk((0, 0), (0, 0), (0, 0), (1, 1), 10),    # within-ref-1 polymorphism (c vs d)
                   blk((0, 0), (0, 0), (1, 1), (1, 1), 80)])   # fixed ref0=0 / ref1=1 divergence
    S = G.shape[0]
    names = ["a_0", "a_1", "b_0", "b_1", "c_0", "c_1", "d_0", "d_1"]
    return Variants(np.arange(S, dtype=float), G, [("0", "1")] * S, 1000.0,
                    sample_names=names, ploidy=2)


def test_recovers_ne_on_panmictic_sim():
    Ne, mu, L = 1000, 5e-8, 2e6
    est = estimate_ne(_diploid_variants(_panmictic(Ne, mu, L), L), mu)
    assert Ne / 3 < est < Ne * 3, est          # pi/(4mu) recovers Ne up to coalescent variance


def test_missing_data_does_not_bias_estimate():
    """Co-called normalisation: masking calls at random must not inflate/deflate Ne."""
    Ne, mu, L = 1000, 5e-8, 2e6
    v = _diploid_variants(_panmictic(Ne, mu, L), L)
    base = estimate_ne(v, mu)
    rng = np.random.default_rng(0)
    miss = rng.random(v.genotypes.shape) < 0.4
    v_miss = Variants(v.positions, np.where(miss, 0, v.genotypes).astype(np.int8), v.alleles,
                      L, sample_names=v.sample_names, ploidy=2, missing=miss)
    est = estimate_ne(v_miss, mu)
    assert abs(est - base) / base < 0.2, (base, est)   # ~unchanged despite 40% missing


def test_excludes_within_individual_pairs():
    """With diploid grouping (ploidy=2) within-individual pairs are dropped; ploidy=1 keeps them."""
    S, L, mu = 10, 1000.0, 1e-3
    # individual 0 = (0, 0), individual 1 = (1, 1) at every site: within-pairs identical, cross differ
    G = np.tile(np.array([0, 0, 1, 1], np.int8), (S, 1))
    v2 = Variants(np.arange(S, dtype=float), G, [("0", "1")] * S, L,
                  sample_names=["a_0", "a_1", "b_0", "b_1"], ploidy=2)
    v1 = Variants(v2.positions, G, v2.alleles, L, sample_names=list("abcd"), ploidy=1)
    assert estimate_ne(v2, mu) > estimate_ne(v1, mu)   # excluding identical within-pairs raises pi


def test_accepts_ts_vcf_and_variants_sources(tmp_path):
    Ne, mu, L = 1000, 5e-8, 5e5
    ts = _panmictic(Ne, mu, L)
    vcf = str(tmp_path / "p.vcf")
    with open(vcf, "w") as f:
        ts.write_vcf(f, individual_names=[f"s{i}" for i in range(ts.num_individuals)],
                     position_transform=lambda x: 1 + np.floor(x).astype(int))
    from_ts = estimate_ne(ts, mu)
    from_vcf = estimate_ne(vcf, mu)
    from_var = estimate_ne(variants_from_vcf(vcf), mu)
    assert from_vcf == pytest.approx(from_var)          # same object, same answer
    assert 0.5 < from_ts / from_vcf < 2.0               # ts (all pairs) ~ vcf (cross-individual)


def test_raises_without_variation():
    v = Variants(np.array([1.0]), np.zeros((1, 4), np.int8), [("0", "1")], 100.0, ploidy=2)
    with pytest.raises(ValueError):
        estimate_ne(v, 1e-8)


def test_groups_none_equals_all_pairs():
    v = _structured_variants()
    assert estimate_ne(v, 1e-3, groups=None) == estimate_ne(v, 1e-3)


def test_groups_restricts_to_same_reference_pairs():
    """Comparing only same-reference pairs drops the between-reference divergence -> lower Ne."""
    v = _structured_variants()
    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert estimate_ne(v, 1e-3, groups=labels) < estimate_ne(v, 1e-3)


def test_groups_array_matches_mapping():
    """An array of one group value per individual == the equivalent {sample id: group} mapping."""
    v = _structured_variants()
    by_map = estimate_ne(v, 1e-3, groups={"a": 0, "b": 0, "c": 1, "d": 1})
    by_arr = estimate_ne(v, 1e-3, groups=[0, 0, 1, 1])       # _group_columns order: a, b, c, d
    assert by_arr == by_map


def test_groups_excludes_unlabeled_individuals():
    """Individuals absent from the mapping are dropped: perturbing them cannot move the estimate."""
    base = estimate_ne(_structured_variants(), 1e-3, groups={"a": 0, "b": 0})   # only a,b; c,d dropped
    v = _structured_variants()
    v.genotypes[:, 4:] ^= 1                                   # flip every c,d call
    assert estimate_ne(v, 1e-3, groups={"a": 0, "b": 0}) == base


def test_groups_without_two_individuals_per_group_raises():
    v = _structured_variants()
    with pytest.raises(ValueError):                          # each in its own group -> no pairs
        estimate_ne(v, 1e-3, groups={"a": 0, "b": 1, "c": 2, "d": 3})


def test_groups_matching_nothing_raises():
    v = _structured_variants()
    with pytest.raises(ValueError):
        estimate_ne(v, 1e-3, groups={"x": 0, "y": 0})        # keys match no individual


def test_exclude_drops_soft_ref_individuals():
    """exclude= leaves soft/suspect refs out of the estimate — by sample id or individual index."""
    S1, S2, S = 20, 60, 80
    G = np.zeros((S, 8), np.int8)
    rng = np.random.default_rng(0)
    for i in range(S1):
        G[i, rng.integers(0, 6)] = 1                          # singletons among a, b, c
    G[S1:, 6] = 1; G[S1:, 7] = 1                              # d is divergent -> inflates diversity
    v = Variants(np.arange(S, dtype=float), G, [("0", "1")] * S, 1000.0,
                 sample_names=["a_0", "a_1", "b_0", "b_1", "c_0", "c_1", "d_0", "d_1"], ploidy=2)
    base = estimate_ne(v, 1e-3)
    assert estimate_ne(v, 1e-3, exclude=None) == base
    assert estimate_ne(v, 1e-3, exclude={"d"}) < base         # excluding the divergent ref lowers Ne
    assert estimate_ne(v, 1e-3, exclude={3}) == estimate_ne(v, 1e-3, exclude={"d"})   # index == name
    # composes with groups: exclude a member of a within-reference group
    g = {"a": 0, "b": 0, "c": 0}
    assert estimate_ne(v, 1e-3, groups=g, exclude={"c"}) != estimate_ne(v, 1e-3, groups=g)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_estimates_ne_when_omitted(tmp_path):
    """io.singer runs with Ne omitted (estimated internally) and preserves sample count/stamping."""
    ts = io.add_mutations(tspaint.simulate_admixture(n_admix=3, n_ref=3, sequence_length=8e4,
                          recombination_rate=1e-8, random_seed=11, Ne=1000, T_admix=30,
                          T_split=5000, f_A=0.5), rate=6e-7, random_seed=11)
    vcf = str(tmp_path / "cohort.vcf")
    with open(vcf, "w") as f:
        ts.write_vcf(f, individual_names=[f"samp{i}" for i in range(ts.num_individuals)],
                     position_transform=lambda x: 1 + np.floor(x).astype(int))
    ens = io.singer(vcf, mutation_rate=6e-7, recombination_rate=1e-8,   # no Ne
                    n_samples=6, thin=2, burn_in=2, seed=42)
    g = ens[0] if isinstance(ens, list) else ens
    assert g.num_samples == ts.num_samples
    assert g.individual(0).metadata == {"id": "samp0"}
