"""SINGER-ensemble plumbing for the VCF-native tspaint painter — offline (no SINGER binary).

Covers the sample<->label layout shared by the tsinfer and SINGER front ends, the ``--arg``
dispatch, and that the SINGER path fails fast with an actionable error when the binary is absent.
The end-to-end SINGER run needs the binary (``TSPAINT_SINGER``) and is exercised separately.
"""
import os

import numpy as np
import pytest

import tspaint.benchmark as bm
from tspaint.benchmark._common import resolve_panel
from tspaint.benchmark._tspaint import _combined_variants
from tspaint.io_genotypes import Variants, resolve_variants
from tspaint.io_singer import DEFAULT_SINGER

_HAVE_SINGER = os.path.exists(DEFAULT_SINGER)


def _vcf(path, samples, rows, *, contig="1"):
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=%s>\n" % contig)
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for pos, gts in rows:
            f.write(f"{contig}\t{pos}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")


def _map(path, pairs):
    with open(path, "w") as f:
        for name, lab in pairs:
            f.write(f"{name}\t{lab}\n")


def _panel(tmp_path):
    q, r, m = tmp_path / "q.vcf", tmp_path / "r.vcf", tmp_path / "m.tsv"
    _vcf(q, ["q0", "q1"], [(100, ["0|1", "1|0"]), (200, ["1|1", "0|0"])])
    _vcf(r, ["r0", "r1", "r2", "r3"],
         [(100, ["0|0", "0|1", "1|1", "1|0"]), (200, ["0|1", "1|1", "0|0", "1|0"])])
    _map(m, [("r0", 0), ("r1", 0), ("r2", 1), ("r3", 1)])
    return resolve_panel(str(q), str(r), sample_map=str(m)), q, r, m


def test_resolve_variants_accepts_variants():
    # The in-memory Variants is now a first-class source (so SINGER can take it without a VCF
    # round-trip). resolve_variants must return it unchanged.
    v = Variants(positions=np.array([1.0, 5.0]), genotypes=np.zeros((2, 3), dtype=np.int8),
                 alleles=[("A", "T"), ("A", "T")], sequence_length=6.0)
    assert resolve_variants(v) is v


def test_combined_variants_query_first_layout(tmp_path):
    panel, *_ = _panel(tmp_path)
    variants, n_query, ref_states = _combined_variants(panel)
    # 2 diploid query samples -> 4 query haps first; 4 diploid refs -> 8 ref haps after.
    assert n_query == 4
    assert ref_states == [0, 0, 0, 0, 1, 1, 1, 1]               # r0,r1 -> 0 ; r2,r3 -> 1 (2 haps each)
    assert variants.genotypes.shape == (2, 4 + 8)              # sites x (query + ref) haplotypes
    assert list(variants.positions) == [100.0, 200.0]
    assert variants.sequence_length == panel.sequence_length


def test_tspaint_arg_dispatch_rejects_unknown(tmp_path):
    panel, q, r, m = _panel(tmp_path)
    with pytest.raises(ValueError, match="unknown arg front end"):
        bm.tspaint(str(q), str(r), sample_map=str(m), arg="bogus", out=None)


def test_tspaint_singer_missing_binary_is_actionable(tmp_path):
    panel, q, r, m = _panel(tmp_path)
    with pytest.raises(FileNotFoundError, match="SINGER"):
        bm.tspaint(str(q), str(r), sample_map=str(m), arg="singer", n_singer=2, thin=1,
                   burn_in=0, singer_bin="/nonexistent/singer", out=None)


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_SINGER,
                    reason="SINGER binary not built (run `tspaint install singer`)")
def test_tspaint_singer_end_to_end(tmp_path):
    # Full path on a small sim where SINGER is installed: SINGER MCMC -> ensemble -> paint -> merge.
    import tspaint
    from tspaint.io_tsinfer import add_mutations
    from tspaint.sim import ADMIXED, SOURCE_A, SOURCE_B, admixture_demography
    from tspaint.benchmark.score import load_truth
    from tspaint.validate import balanced_accuracy

    ts = tspaint.simulate_admixture(admixture_demography(Ne=10000, T_admix=100, T_split=2000, f_A=0.5),
                                    n_query=4, n_reference=5, sequence_length=5e5,
                                    recombination_rate=1e-8, random_seed=2).ts
    ts = add_mutations(ts, rate=1e-7, random_seed=2)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    labels = {int(s): (0 if npop[s] == A else 1) for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]

    paths = bm.export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=2)
    out = tmp_path / "singer.npz"
    tracks = bm.tspaint(paths["query_vcf"], paths["ref_vcf"], sample_map=paths["sample_map"],
                        arg="singer", n_singer=4, thin=4, burn_in=2, Ne=10000,
                        mutation_rate=1e-7, recombination_rate=1e-8, out=str(out))
    truth = load_truth(paths["truth"])
    assert set(tracks) == set(truth)                  # same hap-index key space as the truth
    assert out.exists()
    assert balanced_accuracy(tracks, truth, samples=list(truth)) > 0.6   # ensemble paint beats chance
