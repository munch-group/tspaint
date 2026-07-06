"""Unified I/O front ends: io.tsinfer / io.singer / io.relate accept ts | VCF | VCF-Zarr,
with the pre-unification names as deprecated aliases (CLAUDE.md §5)."""
import warnings

import numpy as np
import pytest

import os

import tspaint
from tspaint import io, io_genotypes, io_relate
from tspaint.io_singer import DEFAULT_SINGER
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
        io.singer(vcf, _Ne=1000, _m=1e-8, _r=1e-8,
                  singer_bin=str(tmp_path / "no_singer"))
    # relate_convert (the .anc/.mut -> tskit step) needs the relate_lib Convert binary
    with pytest.raises(FileNotFoundError):
        io.relate_convert(str(tmp_path / "x.anc"), str(tmp_path / "x.mut"),
                          convert_bin=str(tmp_path / "no_convert"))
    # the relate FRONT END needs a mutation rate and the Relate binaries
    with pytest.raises(ValueError, match="mutation rate"):
        io.relate(ts)
    with pytest.raises(FileNotFoundError):
        io.relate(ts, mutation_rate=1e-8, file_formats_bin=str(tmp_path / "no_rff"))


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(io_relate.relate_binary_path()),
                    reason="Relate binaries not available (tspaint install relate)")
def test_relate_frontend_paints(tmp_path):
    """io.relate(source) runs the whole Relate pipeline (RelateFileFormats -> Relate ->
    EstimatePopulationSize -> Convert) and returns a paintable, order-preserving tree sequence."""
    import tspaint
    ts = io.add_mutations(
        tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=2e5, recombination_rate=1e-8,
                                   random_seed=1, Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
        rate=1.25e-8, random_seed=1)
    gts = io.relate(ts, mutation_rate=1.25e-8, recombination_rate=1e-8, Ne=1000, seed=1,
                    workdir=str(tmp_path))
    assert gts.num_samples == ts.num_samples and gts.num_trees >= 1
    labels = {i: (0 if i < 4 else 1) for i in range(8, 16)}      # references (node index)
    p = tspaint.paint(gts, labels, queries=list(range(8)))
    assert set(p.posteriors) == set(range(8))                    # order preserved -> queries paint
    # skipping EstimatePopulationSize is a valid, faster path
    gts2 = io.relate(ts, mutation_rate=1.25e-8, Ne=1000, estimate_population_size=False, seed=1,
                     workdir=str(tmp_path / "noeps"))
    assert gts2.num_samples == ts.num_samples


# --- individual + haplotype indexing must survive the VCF -> tool -> tskit round-trip ----------

def _diploid_two_pop_vcf(path, seed=7):
    """Write a diploid VCF of two deeply-split populations, individuals named ``A_k`` / ``B_k`` so
    the haplotype index (and its group) is knowable. Returns ``(A_ids, B_ids, n_individuals)``."""
    import msprime
    d = msprime.Demography()
    for name in ("A", "B", "ANC"):
        d.add_population(name=name, initial_size=1000)
    d.add_population_split(time=8000, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": 4, "B": 4}, demography=d, sequence_length=3e5,
                              recombination_rate=1e-8, random_seed=seed, ploidy=2)
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=seed, model=msprime.BinaryMutationModel())
    grp = {i: ("A" if ts.node(ts.individual(i).nodes[0]).population == 0 else "B")
           for i in range(ts.num_individuals)}
    names = [f"{grp[i]}_{i}" for i in range(ts.num_individuals)]
    with open(path, "w") as f:
        ts.write_vcf(f, individual_names=names,
                     position_transform=lambda x: 1 + np.floor(x).astype(int))
    A = [names[i] for i in range(ts.num_individuals) if grp[i] == "A"]
    B = [names[i] for i in range(ts.num_individuals) if grp[i] == "B"]
    return A, B, ts.num_individuals


def _assert_indexing_persists(g, A_ids, B_ids, n_individuals):
    """The converted tree sequence must keep both indexings: (1) each source **individual** regrouped
    from its 2 named haplotype nodes, and (2) each **haplotype** genealogically in its own group — so
    a reference stamped by id really is that source haplotype (a scrambled index would flip this)."""
    from tspaint.ids import resolve_ids
    assert g.num_individuals == n_individuals                          # (1) individuals
    for k in range(g.num_individuals):
        ind = g.individual(k)
        base = ind.metadata["id"]
        assert sorted(g.node(int(n)).metadata["id"] for n in ind.nodes) == [f"{base}_1", f"{base}_2"]
    A, B = resolve_ids(g, A_ids), resolve_ids(g, B_ids)               # (2) haplotype identity
    assert len(A) == 2 * len(A_ids) and len(B) == 2 * len(B_ids)
    for grp, other in ((A, B), (B, A)):
        for n in grp:
            own = [x for x in grp if x != n]
            assert float(g.divergence([[n], own], mode="site")) < \
                float(g.divergence([[n], other], mode="site")), n     # own group is the closer one


def test_tsinfer_preserves_individual_and_haplotype_indexing(tmp_path):
    A, B, n = _diploid_two_pop_vcf(str(tmp_path / "d.vcf"))
    _assert_indexing_persists(io.tsinfer(str(tmp_path / "d.vcf")), A, B, n)


def test_tsinfer_date_requires_mutation_rate():
    ts = _mutated_ts()
    with pytest.raises(ValueError, match="mutation_rate"):
        io.tsinfer(ts, date=True)                              # fails fast, before running inference


@pytest.mark.slow
def test_tsinfer_date_calibrates_times_and_enables_dating(tmp_path):
    """date=True runs tsdate: node ages become GENERATIONS (>> the uncalibrated ~1), sample ids
    survive, dating works on the dated ts, and the guard refuses the undated one."""
    from tspaint.dating import fit_rate_through_time
    A, B, n = _diploid_two_pop_vcf(str(tmp_path / "d.vcf"))
    vcf = str(tmp_path / "d.vcf")

    undated = io.tsinfer(vcf)
    assert float(undated.tables.nodes.time.max()) < 2.0        # tsinfer: uncalibrated ~[0,1]
    dated = io.tsinfer(vcf, date=True, mutation_rate=1.25e-8)
    assert float(dated.tables.nodes.time.max()) > 100.0        # tsdate: generations
    assert dated.num_samples == undated.num_samples
    _assert_indexing_persists(dated, A, B, n)                  # stamped ids survive preprocess+date

    labels = {**{a: 0 for a in A}, **{b: 1 for b in B}}        # date the A/B split
    with pytest.raises(ValueError, match="uncalibrated|GENERATIONS"):
        fit_rate_through_time(undated, labels)                 # undated -> guarded
    rtt = fit_rate_through_time(dated, labels, n_iter=3, n_cells=15)
    assert float(rtt.centers.max()) > 10.0                     # a real (generations) time grid


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_preserves_individual_and_haplotype_indexing(tmp_path):
    A, B, n = _diploid_two_pop_vcf(str(tmp_path / "d.vcf"))
    g = io.singer(str(tmp_path / "d.vcf"), _Ne=1000, _m=1.25e-8, _r=1e-8,
                  ts=3, mcmc_step=2, mcmc_burnin=2, _seed=7)
    _assert_indexing_persists(g[-1] if isinstance(g, list) else g, A, B, n)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(io_relate.relate_binary_path()),
                    reason="Relate binaries not available (tspaint install relate)")
def test_relate_preserves_individual_and_haplotype_indexing(tmp_path):
    A, B, n = _diploid_two_pop_vcf(str(tmp_path / "d.vcf"))
    g = io.relate(str(tmp_path / "d.vcf"), mutation_rate=1.25e-8, recombination_rate=1e-8,
                  Ne=1000, seed=7, workdir=str(tmp_path / "rel"))
    _assert_indexing_persists(g, A, B, n)


def test_relate_windows_splits_for_painting():
    """io.relate_windows tiles a (whole-chromosome) ts into per-window tree sequences: coordinates
    preserved so per-window paintings reassemble by position, samples preserved so the same labels
    paint every window, and each window holds only its interval's genealogy."""
    import tskit
    ts = simulate_admixture(n_admix=4, n_ref=4, sequence_length=4e5, recombination_rate=1e-8,
                            random_seed=2)
    L = float(ts.sequence_length)
    ws = io.relate_windows(ts, 1e5)
    assert len(ws) == 4                                        # 4e5 / 1e5, even division
    covered = 0.0
    for k, w in enumerate(ws):
        lo, hi = k * 1e5, min((k + 1) * 1e5, L)
        assert isinstance(w, tskit.TreeSequence)
        assert w.sequence_length == L                         # coords preserved -> reassemble by position
        assert w.num_samples == ts.num_samples                # same labels paint every window
        e = w.tables.edges
        if e.num_rows:                                         # window k carries only [lo, hi) genealogy
            assert e.left.min() >= lo and e.right.max() <= hi
        covered += hi - lo
    assert covered == L                                       # windows tile the whole sequence

    assert len(io.relate_windows(ts, 3e5)) == 2               # uneven -> ceil windows (last shorter)
    assert len(io.relate_windows(ts, 10 * L)) == 1            # window >= L -> single whole-genome window
    wt = io.relate_windows(ts, 1e5, trim=True)                # trim -> compact ts starting at 0
    assert abs(float(wt[0].sequence_length) - 1e5) < 1
    with pytest.raises(ValueError):
        io.relate_windows(ts, 0)
