"""subset_data — normalise a source and slice by region / samples (CLAUDE.md §5)."""
import numpy as np
import pytest

import tspaint
from tspaint.io import subset_data as subset_data_io
from tspaint.io_genotypes import Variants, resolve_variants, subset_data


def _toy():
    """4 sites x 6 haplotypes (3 diploid samples s0/s1/s2), over [0, 5000)."""
    return Variants(
        positions=np.array([10.0, 200.0, 3000.0, 4000.0]),
        genotypes=np.array([[0, 1, 0, 1, 0, 1],
                            [1, 1, 0, 0, 1, 0],
                            [0, 0, 1, 1, 0, 0],
                            [1, 0, 1, 0, 1, 1]], dtype=np.int8),
        alleles=[("A", "C"), ("G", "T"), ("A", "G"), ("C", "A")],
        sequence_length=5000.0,
        sample_names=["s0_0", "s0_1", "s1_0", "s1_1", "s2_0", "s2_1"])


def test_exported_everywhere():
    assert subset_data is tspaint.subset_data is subset_data_io


def test_region_filters_sites_and_sets_length():
    v = subset_data(_toy(), start=0, end=1000)
    assert v.num_sites == 2
    assert list(v.positions) == [10.0, 200.0]
    assert v.sequence_length == 1000.0
    assert v.num_haplotypes == 6                      # samples untouched
    assert v.alleles == [("A", "C"), ("G", "T")]


def test_region_defaults_keep_everything():
    v = subset_data(_toy())
    assert v.num_sites == 4 and v.sequence_length == 5000.0


def test_start_only_keeps_tail_and_preserves_coordinates():
    v = subset_data(_toy(), start=250)
    assert list(v.positions) == [3000.0, 4000.0]      # absolute positions, not shifted
    assert v.sequence_length == 5000.0


def test_samples_by_integer_columns():
    v = subset_data(_toy(), samples=[0, 2, 4])
    assert v.num_haplotypes == 3
    assert v.sample_names == ["s0_0", "s1_0", "s2_0"]
    np.testing.assert_array_equal(v.genotypes[0], [0, 0, 0])   # cols 0,2,4 of row 0


def test_samples_by_name_expands_a_diploid_sample():
    v = subset_data(_toy(), samples=["s0", "s2"])
    assert v.sample_names == ["s0_0", "s0_1", "s2_0", "s2_1"]  # base name -> both haplotypes


def test_samples_by_exact_haplotype_name():
    v = subset_data(_toy(), samples=["s1_1"])
    assert v.sample_names == ["s1_1"] and v.num_haplotypes == 1


def test_samples_scalar_name():
    v = subset_data(_toy(), samples="s0")
    assert v.sample_names == ["s0_0", "s0_1"]


def test_samples_bool_mask_and_slice_agree():
    mask = np.array([True, False, True, False, True, False])
    a = subset_data(_toy(), samples=mask)
    b = subset_data(_toy(), samples=slice(None, None, 2))
    assert a.sample_names == b.sample_names == ["s0_0", "s1_0", "s2_0"]


def test_region_and_samples_together():
    v = subset_data(_toy(), start=0, end=1000, samples=["s1"])
    assert v.num_sites == 2 and v.num_haplotypes == 2
    assert v.sample_names == ["s1_0", "s1_1"]


def test_result_is_a_variants_a_front_end_accepts():
    v = subset_data(_toy(), samples=[0, 1])
    assert isinstance(v, Variants)
    assert resolve_variants(v) is v                   # singer()/paint() would take it directly


def test_bad_name_and_missing_names_raise():
    with pytest.raises(ValueError):
        subset_data(_toy(), samples=["nope"])
    nameless = _toy()
    nameless.sample_names = None
    with pytest.raises(ValueError):
        subset_data(nameless, samples=["s0"])


def test_bool_mask_wrong_length_raises():
    with pytest.raises(ValueError):
        subset_data(_toy(), samples=np.array([True, False]))


def test_tree_sequence_source():
    import msprime
    ts = msprime.sim_ancestry(samples=4, sequence_length=1e4, recombination_rate=1e-8,
                              population_size=1e4, random_seed=3)
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=3)
    pos = ts.tables.sites.position
    exp = int(((pos >= 0) & (pos < 5000)).sum())
    assert exp > 0
    v = subset_data(ts, start=0, end=5000, samples=[0, 1, 2, 3])
    assert isinstance(v, Variants)
    assert v.num_sites == exp
    assert v.num_haplotypes == 4                        # 4 of the 8 sample nodes
    assert v.sequence_length == 5000.0
    assert set(np.unique(v.genotypes)) <= {0, 1}        # collapsed to biallelic
