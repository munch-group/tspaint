"""VCF-native tspaint painter (tspaint.benchmark.tspaint): tsinfer ARG + paint, hap-index keyed.

Uses tsinfer on a small simulation, so it is marked ``slow`` (still seconds). It verifies the
painter produces the same hap-index key space as the tool runners + export-vcf truth, so all
painters score identically.
"""
import numpy as np
import pytest

import tspaint
import tspaint.benchmark as bm
from tspaint.benchmark.score import load_truth
from tspaint.sim import admixture_demography

pytestmark = pytest.mark.slow


def test_tspaint_vcf_painter_keys_match_truth(tmp_path):
    ts = tspaint.simulate_admixture(admixture_demography(Ne=10000, T_admix=100, T_split=2000, f_A=0.5),
                                    n_query=4, n_reference=5, sequence_length=5e5,
                                    recombination_rate=1e-8, random_seed=2).ts
    from tspaint.io_tsinfer import add_mutations
    ts = add_mutations(ts, rate=1e-7, random_seed=2)            # ensure plenty of sites for tsinfer

    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    labels = {int(s): (0 if npop[s] == A else 1) for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]

    paths = bm.export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=2)
    out = tmp_path / "tspaint.npz"
    tracks = bm.tspaint(paths["query_vcf"], paths["ref_vcf"], sample_map=paths["sample_map"],
                        out=str(out), smooth=True)

    truth = load_truth(paths["truth"])
    assert set(tracks) == set(truth)                           # same hap-index key space
    L = ts.sequence_length
    for k, segs in tracks.items():
        assert segs[0].left == 0.0                             # starts at 0, tiles contiguously
        assert all(segs[i].right == segs[i + 1].left for i in range(len(segs) - 1))
        assert segs[-1].right >= 0.99 * L                      # covers ~all of [0, L) (tsinfer spans the sites)

    # it actually paints (clean true-source structure here): beats chance
    from tspaint.validate import balanced_accuracy
    assert balanced_accuracy(tracks, truth, samples=list(truth)) > 0.6
