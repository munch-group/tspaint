"""RFMix v2 front end / painter — the segment-based head-to-head comparator (CLAUDE.md §9, §10).

RFMix (Maples et al., 2013, *Am. J. Hum. Genet.* 93, 278-288) is the canonical
discriminative LAI method: a random-forest classifier over haplotype windows with a
conditional-random-field smoother. It is **genotype-native**, not ARG-native, so unlike
``tslai_paint``/``nearest_reference_paint`` it does not read the tree sequence — it reads
phased VCFs. This module bridges the two so RFMix scores through the same
:func:`tslai.compare.score_painter` harness:

1. ensure the tree sequence carries mutations (the "true" ARG substrate has none) — RFMix
   needs genotypes, so we drop ``sim_mutations`` on if the ARG is bare;
2. write a phased query VCF (admixed individuals), a phased reference VCF (source
   individuals), a reference sample-map, and a linear genetic map from the sim's
   recombination rate;
3. shell out to the ``rfmix`` binary (isolated ``compare`` pixi env; path via the
   ``TSLAI_RFMIX`` env var or ``.pixi/envs/compare/bin/rfmix``);
4. parse the ``.fb.tsv`` per-marker **posteriors** (RFMix's calibrated soft output, the
   fair comparison against tslai's soft calls) back to per-haplotype
   :class:`tslai.output.Segment` tracks keyed by the original sample-node ids.

Reference populations are named by the integer ancestry **state** (``"0"``, ``"1"``) in the
sample map, so RFMix's population indices line up with tslai's; the ``.fb.tsv`` header is
parsed to recover the exact column order regardless.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .output import Segment, INFORMATIVE

__all__ = ["rfmix_paint", "run_rfmix", "DEFAULT_RFMIX"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RFMIX = os.environ.get(
    "TSLAI_RFMIX", os.path.join(_REPO_ROOT, ".pixi", "envs", "compare", "bin", "rfmix"))

CONTIG = "1"


def _ensure_sites(ts, mutation_rate, seed):
    """RFMix needs genotypes; if the ARG is bare, drop neutral mutations on it."""
    if ts.num_sites > 0:
        return ts
    import msprime
    return msprime.sim_mutations(ts, rate=mutation_rate, random_seed=seed)


def _sample_index(ts):
    return {int(n): i for i, n in enumerate(ts.samples())}


def _classify_individuals(ts, labels, queries):
    """Split diploid individuals into query vs reference by their sample nodes.

    Returns ``(query_inds, ref_inds)`` where each entry is ``(name, (hap0_node, hap1_node))``
    and ref entries also carry the population state. ``name`` is a stable VCF column id.
    """
    queries = set(int(q) for q in queries)
    labels = {int(k): int(v) for k, v in labels.items()}
    query_inds, ref_inds = [], []
    for ind in ts.individuals():
        nodes = [int(n) for n in ind.nodes]
        if len(nodes) != 2:
            continue                                  # only diploid individuals
        name = f"ind{ind.id}"
        if any(n in queries for n in nodes):
            query_inds.append((name, (nodes[0], nodes[1])))
        elif all(n in labels for n in nodes):
            state = labels[nodes[0]]                  # both haps share a source here
            ref_inds.append((name, (nodes[0], nodes[1]), state))
    return query_inds, ref_inds


def _transformed_positions(ts):
    """Distinct, strictly increasing integer VCF positions (≈ original bp; ±1 bp rounding
    is far below tract length, so scoring on [0, L) is unaffected)."""
    pos = ts.tables.sites.position
    out = np.empty(pos.shape[0], dtype=np.int64)
    last = 0
    for i, p in enumerate(pos):
        v = max(int(np.floor(p)) + 1, last + 1)
        out[i] = v
        last = v
    return out


def _write_vcf(path, ts, columns, names, positions, geno):
    """Phased biallelic VCF: one diploid column per (hap0, hap1) node pair in ``columns``."""
    sidx = _sample_index(ts)
    cols = [(sidx[h0], sidx[h1]) for (h0, h1) in columns]
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={CONTIG}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(names) + "\n")
        for s in range(geno.shape[0]):
            row = geno[s]
            gts = "\t".join(f"{row[i0]}|{row[i1]}" for (i0, i1) in cols)
            f.write(f"{CONTIG}\t{positions[s]}\t.\tA\tT\t.\tPASS\t.\tGT\t{gts}\n")


def _write_sample_map(path, ref_inds):
    with open(path, "w") as f:
        for name, _nodes, state in ref_inds:
            f.write(f"{name}\t{state}\n")


def _write_genetic_map(path, positions, recombination_rate):
    """3-column ``chrom  pos(bp)  cM`` map; linear cM = pos · r · 100 (uniform recomb)."""
    with open(path, "w") as f:
        for p in positions:
            f.write(f"{CONTIG}\t{int(p)}\t{p * recombination_rate * 100.0:.10f}\n")


def run_rfmix(ts, labels, queries, *, recombination_rate, generations,
              rfmix_bin=None, workdir=None, extra_args=None):
    """Write inputs, run RFMix, and return the output basename (``.fb.tsv``/``.msp.tsv``).

    ``ts`` must carry sites. Raises ``FileNotFoundError`` if the binary is absent and
    ``RuntimeError`` on a nonzero RFMix exit (stderr tail included)."""
    rfmix_bin = rfmix_bin or DEFAULT_RFMIX
    if not os.path.exists(rfmix_bin):
        raise FileNotFoundError(
            f"rfmix binary not found at {rfmix_bin}; set TSLAI_RFMIX or pass rfmix_bin "
            "(install via the `compare` pixi env: pixi install -e compare)")

    query_inds, ref_inds = _classify_individuals(ts, labels, queries)
    if not query_inds or not ref_inds:
        raise ValueError("need both query and reference diploid individuals")

    tmp = workdir or tempfile.mkdtemp(prefix="tslai_rfmix_")
    os.makedirs(tmp, exist_ok=True)
    positions = _transformed_positions(ts)
    geno = ts.genotype_matrix()

    qvcf = os.path.join(tmp, "query.vcf")
    rvcf = os.path.join(tmp, "reference.vcf")
    smap = os.path.join(tmp, "sample_map.tsv")
    gmap = os.path.join(tmp, "genetic_map.tsv")
    out = os.path.join(tmp, "rfmix_out")

    _write_vcf(qvcf, ts, [c for _n, c in query_inds], [n for n, _c in query_inds],
               positions, geno)
    _write_vcf(rvcf, ts, [c for _n, c, _s in ref_inds], [n for n, _c, _s in ref_inds],
               positions, geno)
    _write_sample_map(smap, ref_inds)
    _write_genetic_map(gmap, positions, recombination_rate)

    cmd = [rfmix_bin, "-f", qvcf, "-r", rvcf, "-m", smap, "-g", gmap,
           "-o", out, "--chromosome=" + CONTIG, "-G", str(int(round(generations)))]
    if extra_args:
        cmd += list(extra_args)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    if res.returncode != 0:
        raise RuntimeError(f"rfmix failed (exit {res.returncode}):\n{res.stderr[-1500:]}")
    return out


def _parse_fb(fb_path, query_inds, K, sequence_length):
    """Parse ``.fb.tsv`` per-marker posteriors into per-node Segment tracks.

    The header columns after the first four are ``<name>:::hap<1|2>:::<pop>``; we recover
    the (name, hap, pop) → column map, then for each query haplotype build a piecewise-
    constant painting over [0, L) from the marker posteriors (pop order = state order).
    """
    with open(fb_path) as f:
        lines = f.read().splitlines()
    # Skip leading comment line(s) (e.g. "#reference_panel_population:..."); the real
    # header is the first line whose 1st field is the chromosome column label.
    hdr_i = next(i for i, ln in enumerate(lines)
                 if ln and not ln.startswith("#reference_panel"))
    header = lines[hdr_i].split("\t")
    data = np.array([ln.split("\t") for ln in lines[hdr_i + 1:] if ln], dtype=object)
    phys = data[:, 1].astype(float)

    # column index for each (name, hap, pop)
    colmap = {}
    pops = set()
    for j, h in enumerate(header[4:], start=4):
        if ":::" not in h:
            continue
        name, hap, pop = h.split(":::")
        colmap[(name, hap, pop)] = j
        pops.add(pop)
    pops = sorted(pops, key=lambda p: int(p) if p.isdigit() else p)   # state order

    # marker interval boundaries on [0, L): marker i paints [b_i, b_{i+1})
    bnd = np.empty(phys.shape[0] + 1)
    bnd[0] = 0.0
    bnd[1:-1] = phys[1:]
    bnd[-1] = sequence_length

    tracks = {}
    for name, (h0, h1) in query_inds:
        for hap_label, node in (("hap1", h0), ("hap2", h1)):
            cols = [colmap.get((name, hap_label, p)) for p in pops]
            if any(c is None for c in cols):
                continue
            post = data[:, cols].astype(float)        # (n_markers, K) in state order
            if post.shape[1] < K:                     # pad if a pop never appears
                post = np.hstack([post, np.zeros((post.shape[0], K - post.shape[1]))])
            segs = []
            for i in range(post.shape[0]):
                left, right = float(bnd[i]), float(bnd[i + 1])
                if right <= left:
                    continue
                p = post[i, :K]
                s = p.sum()
                p = p / s if s > 0 else np.full(K, 1.0 / K)
                if segs and np.allclose(segs[-1].posterior, p) and segs[-1].right == left:
                    segs[-1] = Segment(segs[-1].left, right, segs[-1].posterior, INFORMATIVE)
                else:
                    segs.append(Segment(left, right, p, INFORMATIVE))
            tracks[node] = segs
    return tracks


def rfmix_paint(ts, labels, queries, K=2, *, recombination_rate=1e-8, generations=30.0,
                mutation_rate=4e-7, seed=1, rfmix_bin=None, extra_args=None):
    """RFMix painter with the standard ``painter(ts, labels, queries)`` signature.

    Adds mutations if ``ts`` has none (the true-ARG substrate), runs RFMix, and returns
    ``{query_node: [Segment]}`` from the ``.fb.tsv`` posteriors. ``generations`` is RFMix's
    ``-G`` (generations since admixture — known from the sim's ``T_admix``)."""
    ts = _ensure_sites(ts, mutation_rate, seed)
    query_inds, _ref = _classify_individuals(ts, labels, queries)
    out = run_rfmix(ts, labels, queries, recombination_rate=recombination_rate,
                    generations=generations, rfmix_bin=rfmix_bin, extra_args=extra_args)
    return _parse_fb(out + ".fb.tsv", query_inds, K, ts.sequence_length)
