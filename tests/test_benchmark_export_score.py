"""Benchmark export + score loop (tspaint.benchmark.export / score).

Offline: simulate known-truth admixture, export diploid VCFs + truth, then score a *perfect*
painting (built from the truth) and a constant-wrong one. Checks the keys line up (truth hap
index == resolved query hap index) so any tool's .npz scores against the truth directly.
"""
import numpy as np

import tspaint
from tspaint.benchmark import export_vcf, score, resolve_panel
from tspaint.benchmark.score import load_truth
from tspaint.output import Segment, INFORMATIVE
from tspaint import serialize
from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED


def _admixture(seed=1):
    ts = tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=5e5,
                                    recombination_rate=1e-8, random_seed=seed, Ne=1000,
                                    T_admix=30, T_split=5000, f_A=0.5)
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    return ts, labels, queries, sop


def test_export_writes_inputs_and_truth(tmp_path):
    ts, labels, queries, _sop = _admixture()
    paths = export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=1)
    for k in ("query_vcf", "ref_vcf", "sample_map", "truth"):
        assert paths[k] and __import__("os").path.exists(paths[k])

    # truth is keyed by query haplotype index 0..len(queries)-1
    truth = load_truth(paths["truth"])
    assert set(truth) == set(range(len(queries)))

    # resolving the exported VCFs reproduces exactly those query hap keys
    panel = resolve_panel(paths["query_vcf"], paths["ref_vcf"], sample_map=paths["sample_map"])
    assert panel.query_keys == list(range(len(queries)))
    assert panel.K == 2


def test_score_perfect_and_wrong(tmp_path):
    ts, labels, queries, sop = _admixture()
    paths = export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=1)
    truth = load_truth(paths["truth"])
    L = ts.sequence_length

    # perfect painting: one-hot of the true state on each true tract
    perfect = {k: [Segment(l, r, _one_hot(s), INFORMATIVE) for (l, r, s) in segs]
               for k, segs in truth.items()}
    pp = tmp_path / "perfect.npz"
    serialize.save_painting(str(pp), perfect, seqlen=L)

    # constant-wrong painting: always state 0
    wrong = {k: [Segment(0.0, L, np.array([1.0, 0.0]), INFORMATIVE)] for k in truth}
    wp = tmp_path / "wrong.npz"
    serialize.save_painting(str(wp), wrong, seqlen=L)

    rows = {r["name"]: r for r in score(paths["truth"], {"perfect": str(pp), "wrong": str(wp)})}
    assert rows["perfect"]["balanced_accuracy"] == 1.0
    assert rows["perfect"]["switch_density_ratio"] == 1.0          # identical hard segmentation
    assert rows["perfect"]["n_samples"] == len(queries)
    assert rows["wrong"]["balanced_accuracy"] < 0.75              # only the majority class right


def test_score_matches_validate_directly(tmp_path):
    # The score() balanced accuracy equals validate.balanced_accuracy on the same objects.
    ts, labels, queries, sop = _admixture(seed=3)
    paths = export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=3)
    truth = load_truth(paths["truth"])
    L = ts.sequence_length
    painting = {k: [Segment(0.0, L, np.array([0.6, 0.4]), INFORMATIVE)] for k in truth}
    pth = tmp_path / "p.npz"
    serialize.save_painting(str(pth), painting, seqlen=L)

    from tspaint.validate import balanced_accuracy
    direct = balanced_accuracy(painting, truth, samples=list(truth))
    row = score(paths["truth"], {"x": str(pth)})[0]
    assert abs(row["balanced_accuracy"] - direct) < 1e-12


def _one_hot(s):
    p = np.zeros(2)
    p[int(s)] = 1.0
    return p
