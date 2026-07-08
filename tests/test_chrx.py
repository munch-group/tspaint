"""chrX / hemizygosity handling: sex + PAR inference, pseudohaploid, mixed-ploidy VCF reading.

tskit / tsinfer / SINGER are haplotype-level; the correct chrX input is one haploid sample per real
haplotype (females 2, males 1 outside the pseudo-autosomal regions). :func:`tspaint.io.pseudohaploid`
produces that; :func:`tspaint.io_genotypes.variants_from_vcf` reads mixed ploidy without dropping
names and, given a sex map, collapses hemizygous males.
"""
import os
import tempfile

import numpy as np
import pytest

import tspaint
from tspaint import io
from tspaint.io_genotypes import (Variants, variants_from_vcf, estimate_ne, pseudohaploid,
                                  _resolve_sex_and_par, _positions_in_par, _group_columns)
from tspaint.sim import admixture_demography


# --- synthetic chrX: diploid females + males homozygous outside PAR, heterozygous inside ------

def _chrx_variants(n_female=2, n_male=2, L=1000, seed=1, par=((0, 50), (950, 1000))):
    """A synthetic chrX Variants (ploidy-2 encoded): males are homozygous outside PAR."""
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(1, L), size=240, replace=False)).astype(float)
    inpar = _positions_in_par(pos, [(float(a), float(b)) for a, b in par])
    S = len(pos)
    cols, names = [], []
    for i in range(n_female):
        cols += [rng.integers(0, 2, S), rng.integers(0, 2, S)]      # diploid, het throughout
        names += [f"F{i}_0", f"F{i}_1"]
    for i in range(n_male):
        x = rng.integers(0, 2, S)
        cols += [x, np.where(inpar, rng.integers(0, 2, S), x)]      # 2nd copy differs only in PAR
        names += [f"M{i}_0", f"M{i}_1"]
    G = np.stack(cols, axis=1).astype(np.int8)
    return Variants(pos, G, [("0", "1")] * S, float(L), sample_names=names, ploidy=2), inpar


# --- sex / PAR inference ---------------------------------------------------------------------

def test_infer_sex_from_interior_heterozygosity():
    v, _ = _chrx_variants(n_female=2, n_male=2)
    _, base, sex, par = _resolve_sex_and_par(v, None)
    assert sex == {"F0": "F", "F1": "F", "M0": "M", "M1": "M"}
    assert par and par[0][0] == 0.0 and par[-1][1] == v.sequence_length   # PAR at both ends


def test_infer_sex_robust_to_male_majority():
    v, _ = _chrx_variants(n_female=1, n_male=3)          # females are the minority
    sex = _resolve_sex_and_par(v, None)[2]
    assert sex == {"F0": "F", "M0": "M", "M1": "M", "M2": "M"}


def test_autosome_all_female_no_false_males():
    """A normal diploid autosome (het throughout) has no inferred males."""
    rng = np.random.default_rng(0)
    S = 240
    G = rng.integers(0, 2, (S, 6)).astype(np.int8)
    v = Variants(np.sort(rng.choice(np.arange(1, 1000), S, replace=False)).astype(float),
                 G, [("0", "1")] * S, 1000.0, sample_names=[f"s{i//2}_{i%2}" for i in range(6)], ploidy=2)
    assert set(_resolve_sex_and_par(v, None)[2].values()) == {"F"}


# --- sex_map (dict + DataFrame) --------------------------------------------------------------

def test_sex_map_dict_overrides_inference():
    v, _ = _chrx_variants()
    sex = _resolve_sex_and_par(v, {"M0": "F"})[2]        # partial map: rest inferred
    assert sex["M0"] == "F" and sex["M1"] == "M" and sex["F0"] == "F"


def test_sex_map_dataframe_and_value_forms():
    pd = pytest.importorskip("pandas")
    v, _ = _chrx_variants()
    df = pd.DataFrame({"id": ["F0", "F1", "M0", "M1"], "sex": ["female", "F", "male", "M"]})
    assert _resolve_sex_and_par(v, df)[2] == {"F0": "F", "F1": "F", "M0": "M", "M1": "M"}


def test_sex_map_bad_value_raises():
    v, _ = _chrx_variants()
    with pytest.raises(ValueError, match="F.*M|female|male"):
        _resolve_sex_and_par(v, {"M0": "yes"})


# --- pseudohaploid ---------------------------------------------------------------------------

def test_pseudohaploid_drop_default():
    v, _ = _chrx_variants(n_female=2, n_male=2)
    ph = pseudohaploid(v)                                # SINGER-safe default: males -> 1 hap
    assert ph.sample_names == ["F0_1", "F0_2", "F1_1", "F1_2", "M0", "M1"]
    assert ph.ploidy == 1


def test_pseudohaploid_keep_par_masks_outside_par():
    v, par_truth = _chrx_variants(n_female=1, n_male=1)
    ph = pseudohaploid(v, keep_par=True)
    assert ph.sample_names == ["F0_1", "F0_2", "M0_1", "M0_2"]
    par = _resolve_sex_and_par(v, None)[3]
    par_mask = _positions_in_par(ph.positions, par)
    j = ph.sample_names.index("M0_2")                    # real inside PAR, missing outside
    assert bool(ph.missing[~par_mask, j].all()) and not bool(ph.missing[par_mask, j].any())


def test_pseudohaploid_autosome_splits_all():
    rng = np.random.default_rng(2)
    S = 240
    G = rng.integers(0, 2, (S, 4)).astype(np.int8)
    v = Variants(np.sort(rng.choice(np.arange(1, 1000), S, replace=False)).astype(float),
                 G, [("0", "1")] * S, 1000.0, sample_names=["a_0", "a_1", "b_0", "b_1"], ploidy=2)
    assert pseudohaploid(v).sample_names == ["a_1", "a_2", "b_1", "b_2"]   # every diploid split


# --- variants_from_vcf: mixed ploidy + chrX collapse -----------------------------------------

_HDR = ("##fileformat=VCFv4.2\n##contig=<ID=X,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")


def _write_vcf(path, sample_line, rows):
    with open(path, "w") as f:
        f.write(_HDR + sample_line + "\n")
        for pos, gts in rows:
            f.write(f"X\t{pos}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")


def test_variants_from_vcf_mixed_ploidy_retains_names(tmp_path):
    """A mixed-ploidy VCF (haploid-GT males) keeps per-hap names + a sample grouping (no silent drop)."""
    vcf = str(tmp_path / "m.vcf")
    _write_vcf(vcf, "F1\tF2\tM1\tM2",
               [(10, ["0|1", "1|0", "0", "1"]), (20, ["1|1", "0|0", "1", "0"])])
    v = variants_from_vcf(vcf)
    assert v.num_haplotypes == 6                                  # 2 females*2 + 2 males*1
    assert v.sample_names == ["F1_0", "F1_1", "F2_0", "F2_1", "M1", "M2"]
    assert v.sample_index.tolist() == [0, 0, 1, 1, 2, 3]
    assert _group_columns(v)[1] == ["F1", "F2", "M1", "M2"]


def test_variants_from_vcf_sex_map_collapses_pseudodiploid_males(tmp_path):
    """Males encoded homozygous-diploid are collapsed to one haplotype when a sex map is given."""
    vcf = str(tmp_path / "x.vcf")
    _write_vcf(vcf, "F1\tF2\tM1",
               [(10, ["0|1", "1|0", "0|0"]), (500, ["1|1", "0|1", "1|1"]), (990, ["0|0", "0|1", "0|1"])])
    faithful = variants_from_vcf(vcf)                             # no sex_map -> read as-encoded
    assert faithful.num_haplotypes == 6 and faithful.sample_index is None
    # full sex map -> deterministic (no inference on this tiny 3-site example)
    aware = variants_from_vcf(vcf, sex_map={"F1": "F", "F2": "F", "M1": "M"})   # chrX-aware collapse
    assert aware.num_haplotypes == 5                              # F1(2) + F2(2) + M1(1)
    assert aware.sample_names == ["F1_1", "F1_2", "F2_1", "F2_2", "M1"]
    assert aware.sample_index.tolist() == [0, 0, 1, 1, 2]         # females diploid, male haploid


def test_variants_from_vcf_autosome_unchanged(tmp_path):
    vcf = str(tmp_path / "a.vcf")
    _write_vcf(vcf, "s0\ts1", [(10, ["0|1", "1|0"]), (20, ["1|1", "0|0"])])
    v = variants_from_vcf(vcf)                                    # no sex_map -> unchanged
    assert v.ploidy == 2 and v.sample_index is None
    assert v.sample_names == ["s0_0", "s0_1", "s1_0", "s1_1"]


# --- estimate_ne honours mixed-ploidy grouping -----------------------------------------------

def test_estimate_ne_uses_sample_index_grouping():
    """estimate_ne excludes within-female pairs via sample_index (males have none)."""
    v, _ = _chrx_variants(n_female=3, n_male=2)
    aware = pseudohaploid(v, keep_par=True)              # mixed via missing; grouping ploidy 1
    ne = estimate_ne(aware, 5e-8)
    assert np.isfinite(ne) and ne > 0


# --- end-to-end: pseudohaploid -> front end stamps + paints ----------------------------------

def test_pseudohaploid_feeds_tsinfer_and_paints():
    """A pseudo-haploid Variants flows through io.tsinfer (stamped) and paints by name."""
    ts = io.add_mutations(tspaint.simulate_admixture(admixture_demography(Ne=1000, T_admix=30,
                          T_split=5000, f_A=0.5), n_query=3, n_reference=3, sequence_length=1e5,
                          recombination_rate=1e-8, random_seed=4).ts, rate=6e-7, random_seed=4)
    # emulate a diploid VCF read: haplotype columns grouped as diploids, all female (autosome-like)
    from tspaint.io_genotypes import _variants_from_ts
    v = _variants_from_ts(ts)
    v = Variants(v.positions, v.genotypes, v.alleles, v.sequence_length,
                 sample_names=[f"ind{i//2}_{i%2}" for i in range(v.num_haplotypes)], ploidy=2)
    # explicit all-female map -> deterministic split (this is an autosome, not a sex chromosome)
    hap = pseudohaploid(v, sex_map={f"ind{i}": "F" for i in range(v.num_haplotypes // 2)})
    assert hap.ploidy == 1 and hap.num_haplotypes == ts.num_samples
    ti = io.tsinfer(hap)                                 # front end stamps the per-chromosome ids
    assert ti.num_samples == ts.num_samples
    assert ti.individual(0).metadata["id"] == "ind0_1"   # each haplotype is its own individual
