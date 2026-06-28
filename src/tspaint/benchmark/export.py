"""Export a tree sequence to the benchmark's VCF inputs + a matching truth table.

Closes the loop ``simulate → benchmark → score`` (CLAUDE.md §9): given a tree sequence with
known local ancestry (e.g. from :func:`tspaint.simulate_admixture`), write the **diploid phased**
query and reference VCFs, the reference sample map, and a ``truth.npz`` — all keyed by the same
**query haplotype index** the runners use, so any tool's output ``.npz`` scores directly against
the truth (:func:`tspaint.benchmark.score.score`).

Haplotypes are paired into diploid VCF samples (the tools require diploidy) regardless of the
sim's ploidy: query haplotype ``k`` (its position in the query-node list) is column ``k`` of the
query VCF and key ``k`` of the truth; reference haplotypes are paired **within a state** so each
VCF sample carries a single ancestry label.
"""
from __future__ import annotations

import os

import numpy as np
import tskit

from . import _common as C
from ._common import _strictly_increasing

__all__ = ["export_vcf"]


def _ensure_sites(ts, mutation_rate, seed):
    if ts.num_sites > 0:
        return ts
    import msprime
    return msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)


def _pair(nodes):
    """Pair a node list into ``(a, b)`` diploid columns; duplicate the last if odd."""
    nodes = list(nodes)
    if len(nodes) % 2:
        nodes.append(nodes[-1])
    return [(nodes[i], nodes[i + 1]) for i in range(0, len(nodes), 2)]


def export_vcf(ts, labels, queries=None, *, outdir, mutation_rate=4e-7, seed=1,
               chromosome="1", prefix=""):
    """Write query/reference VCFs, a sample map, and a truth table from a tree sequence.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence with known local ancestry (census nodes; e.g. from
        :func:`tspaint.simulate_admixture`). Mutations are overlaid if it has no sites.
    labels : dict[int, int]
        Reference sample-node id → ancestry state.
    queries : iterable[int], optional
        Query sample-node ids; defaults to every sample not in ``labels``. Must be an even
        count (haplotypes are paired into diploid VCF samples).
    outdir : str
        Output directory (created if absent).
    mutation_rate : float, optional
        Rate for overlaid mutations if ``ts`` is bare (default ``4e-7``).
    seed : int, optional
        Seed for the mutation overlay (default 1).
    chromosome : str, optional
        Contig label written into the VCFs (default ``"1"``).
    prefix : str, optional
        Filename prefix for the outputs.

    Returns
    -------
    dict
        Paths ``{"query_vcf", "ref_vcf", "sample_map", "truth"}``.
    """
    ts = _ensure_sites(ts, mutation_rate, seed)
    os.makedirs(outdir, exist_ok=True)
    labels = {int(k): int(v) for k, v in labels.items()}
    if queries is None:
        queries = [int(s) for s in ts.samples() if int(s) not in labels]
    queries = [int(q) for q in queries]
    if len(queries) % 2:
        raise ValueError(f"need an even number of query haplotypes to pair into diploid VCF "
                         f"samples; got {len(queries)}")

    node_pop = ts.tables.nodes.population
    state_of_pop = {int(node_pop[n]): s for n, s in labels.items()}
    K = max(labels.values()) + 1

    sidx = {int(n): i for i, n in enumerate(ts.samples())}
    geno = ts.genotype_matrix()                                       # (S, H), col = sample node
    positions = _strictly_increasing(np.floor(ts.tables.sites.position).astype(np.int64))
    seqlen = float(ts.sequence_length)
    alleles = [("A", "T")] * geno.shape[0]
    panel = C.Panel(positions=positions, geno=geno, alleles=alleles, query=[], ref=[], K=K,
                    contig=str(chromosome), sequence_length=seqlen)

    # query: pair in node order -> hap key = node position in the query list
    q_pairs = _pair(queries)
    query_inds = [(f"{prefix}q{j}", (sidx[a], sidx[b])) for j, (a, b) in enumerate(q_pairs)]
    key_of_node = {q: k for k, q in enumerate(queries)}

    # reference: pair within state -> one label per VCF sample
    ref_inds, sample_map_rows = [], []
    by_state = {}
    for n, s in labels.items():
        by_state.setdefault(s, []).append(n)
    for s in sorted(by_state):
        for j, (a, b) in enumerate(_pair(sorted(by_state[s]))):
            name = f"{prefix}r{s}_{j}"
            ref_inds.append((name, (sidx[a], sidx[b])))
            sample_map_rows.append((name, s))

    qv = os.path.join(outdir, f"{prefix}query.vcf")
    rv = os.path.join(outdir, f"{prefix}reference.vcf")
    sm = os.path.join(outdir, f"{prefix}sample_map.tsv")
    truth = os.path.join(outdir, f"{prefix}truth.npz")
    C.write_phased_vcf(qv, panel, query_inds)
    C.write_phased_vcf(rv, panel, ref_inds)
    with open(sm, "w") as f:
        for (name, s) in sample_map_rows:
            f.write(f"{name}\t{s}\n")

    _write_truth(truth, ts, queries, key_of_node, state_of_pop)
    return {"query_vcf": qv, "ref_vcf": rv, "sample_map": sm, "truth": truth}


def _write_truth(path, ts, queries, key_of_node, state_of_pop):
    """Write the tspaint-truth ``.npz`` (sample=hap key) for the query haplotypes."""
    from ..sim import local_ancestry_truth
    tracts, _ = local_ancestry_truth(ts)
    samp, left, right, state = [], [], [], []
    qset = set(queries)
    for node, segs in tracts.items():
        if node not in qset:
            continue
        key = key_of_node[node]
        for (lo, hi, pid) in segs:
            if pid in state_of_pop:
                samp.append(key); left.append(lo); right.append(hi); state.append(state_of_pop[pid])
    with open(path, "wb") as f:
        np.savez_compressed(f, _format="tspaint-truth", _version=1,
                            sample=np.array(samp, np.int64), left=np.array(left, float),
                            right=np.array(right, float), state=np.array(state, np.int8))
