"""Unified I/O front ends: io.tsinfer / io.singer / io.relate accept ts | VCF | VCF-Zarr,
with the pre-unification names as deprecated aliases (CLAUDE.md §5)."""
import warnings

import numpy as np
import pytest

import tspaint
from tspaint import io, io_genotypes
from tspaint.sim import simulate_admixture


def _mutated_ts(seed=1):
    return io.add_mutations(
        simulate_admixture(n_admix=4, n_ref=4, sequence_length=5e4, recombination_rate=1e-8,
                           random_seed=seed),
        rate=3e-7, random_seed=seed)


def _write_haploid_vcf(ts, path):
    io.write_haploid_vcf(ts, path)        # drops individuals -> one haploid column per sample node


def _write_vcz(ts, path):
    # minimal VCF-Zarr (vcz) the way bio2zarr would lay it out: core arrays + dimension attrs,
    # which tsinfer.VariantData (the chunked reader) requires.
    import zarr
    G = ts.genotype_matrix()                                   # (sites, samples)
    V, N = G.shape
    root = zarr.open(path, mode="w")
    def arr(name, data, dims):
        root[name] = data
        root[name].attrs["_ARRAY_DIMENSIONS"] = dims
    arr("variant_position", np.asarray(ts.tables.sites.position).astype("i8"), ["variants"])
    arr("call_genotype", G[:, :, None].astype("i1"), ["variants", "samples", "ploidy"])
    arr("variant_allele", np.array([["0", "1"]] * V), ["variants", "alleles"])
    arr("variant_contig", np.zeros(V, "i4"), ["variants"])
    arr("contig_id", np.array(["1"]), ["contigs"])
    arr("sample_id", np.array([f"n{i}" for i in range(N)]), ["samples"])
    arr("variant_ancestral_allele", np.array(["0"] * V), ["variants"])
    return path


def test_source_kind_dispatch(tmp_path):
    ts = _mutated_ts()
    vcf = str(tmp_path / "d.vcf"); _write_haploid_vcf(ts, vcf)
    zarr_path = _write_vcz(ts, str(tmp_path / "d.zarr"))
    assert io_genotypes.source_kind(ts) == "ts"
    assert io_genotypes.source_kind(vcf) == "vcf"
    assert io_genotypes.source_kind(zarr_path) == "zarr"


def test_variants_readers_agree_with_ts(tmp_path):
    ts = _mutated_ts()
    n = ts.num_samples
    vcf = str(tmp_path / "d.vcf"); _write_haploid_vcf(ts, vcf)
    zarr_path = _write_vcz(ts, str(tmp_path / "d.zarr"))
    v_vcf = io_genotypes.variants_from_vcf(vcf)
    v_zarr = io_genotypes.variants_from_zarr(zarr_path)
    assert v_vcf.num_haplotypes == n and v_zarr.num_haplotypes == n
    assert v_vcf.num_sites > 0 and v_zarr.num_sites == ts.num_sites
    # genotypes are biallelic 0/1
    assert set(np.unique(v_vcf.genotypes)) <= {0, 1}
    assert set(np.unique(v_zarr.genotypes)) <= {0, 1}


def test_tsinfer_accepts_ts_vcf_and_zarr(tmp_path):
    ts = _mutated_ts()
    n = ts.num_samples
    vcf = str(tmp_path / "d.vcf"); _write_haploid_vcf(ts, vcf)
    zarr_path = _write_vcz(ts, str(tmp_path / "d.zarr"))
    for source, label in ((ts, "ts"), (vcf, "vcf"), (zarr_path, "zarr")):
        inferred = io.tsinfer(source)
        assert inferred.num_samples == n, label
        assert inferred.num_trees >= 1


def test_deprecated_aliases_warn_but_work(tmp_path):
    ts = _mutated_ts()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = io.infer_tree_sequence(ts)
    assert out.num_samples == ts.num_samples
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)


def test_singer_and_relate_require_their_binaries(tmp_path):
    ts = _mutated_ts()
    vcf = str(tmp_path / "d.vcf"); _write_haploid_vcf(ts, vcf)
    # singer dispatches on the source then needs its binary
    with pytest.raises(FileNotFoundError):
        io.singer(vcf, Ne=1000, mutation_rate=1e-8, recombination_rate=1e-8,
                  singer_bin=str(tmp_path / "no_singer"))
    # relate needs the relate_lib Convert binary
    with pytest.raises(FileNotFoundError):
        io.relate(str(tmp_path / "x.anc"), str(tmp_path / "x.mut"),
                  convert_bin=str(tmp_path / "no_convert"))
