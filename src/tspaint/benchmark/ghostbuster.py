"""GhostBuster benchmark runner — **tree sequence** in, tspaint ``.npz`` painting out.

GhostBuster (Loya, Hinch, Palamara, Speidel & Myers 2026) is the one comparator that consumes the
*same object tspaint paints*: a tskit tree sequence. Every other runner in this package goes
VCF → tool → painting, which means any head-to-head also compares the tools' front ends. Here there
is no VCF bridge and no front-end confound — **same ARG in, different model, compare the posteriors**
(plans/PLAN_PUBLICATION.md §2.2). That is why this runner's signature is ``(ts, labels, queries)``
rather than ``(query_vcf, ref_vcf, sample_map=...)``, and why it is *not* in
:data:`tspaint.benchmark.PAINTERS` (which is the VCF-native registry).

Method-wise it is the direct descendant of MOSAIC with the copying model swapped for the genealogy:
an EM mixture in which each latent ancestry component has its own piecewise-constant **coalescence
rate profile** to each reference group, and the E-step posterior over components *is* the local
ancestry call. As with MOSAIC, **the latent components are not our reference states** — a component
is free to match no reference at all (that is how it detects ghosts) — so the bridge maps each
component onto the reference group it coalesces with fastest, using GhostBuster's own fitted rates.

Override the install with ``TSPAINT_GHOSTBUSTER_DIR`` / ``TSPAINT_GHOSTBUSTER_CMD``.
"""
from __future__ import annotations

import glob
import os
import tempfile

import numpy as np

from . import _common as C
from ._msp import tracks_from_marker_posteriors
from ..output import Segment, INFORMATIVE, MISSING_INFO

__all__ = ["ghostbuster"]


def ghostbuster(ts, labels, queries=None, *, chromosome=1, recomb_rate=1e-8, clusters=None,
                hmm=True, n_iters=200, start_time=None, end_time=None, n_epochs=None,
                out=None, workdir=None, extra_args=None, log=None):
    """Run GhostBuster on a tree sequence and return per-query-haplotype **soft** Segment tracks.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The genealogy to paint — the *same* object handed to :func:`tspaint.paint` for a clean
        head-to-head (tsinfer / Relate / SINGER / a true ARG; GhostBuster's own runs use Relate).
    labels : dict[int, int]
        Reference sample-node id → ancestry state (as for :func:`tspaint.paint`).
    queries : iterable[int], optional
        Query sample-node ids. Defaults to every sample not in ``labels``.
    chromosome : int, optional
        Chromosome number GhostBuster keys its files on (default 1).
    recomb_rate : float, optional
        Per-base rate for the generated HapMap-format recombination map (default ``1e-8``).
    clusters : int, optional
        Number of latent ancestry components (GhostBuster's ``-k``). Defaults to the number of
        reference states. **Not** the reference panels — see the module docstring.
    hmm : bool, optional
        Use GhostBuster's genome-axis HMM (its ``--hmm``; default ``True``). Its own guidance is to
        turn this **off** for events older than ~1000 generations, where ancestry linkage is
        negligible and enforcing continuity biases the coalescence-rate estimates.
    n_iters : int, optional
        EM iterations (its ``-i``; default 200).
    start_time, end_time : float, optional
        Time window to fit over, **in log scale** (its ``-start_time`` / ``-end_time``). Left to
        GhostBuster's defaults when omitted.
    n_epochs : int, optional
        Number of time epochs the window is split into (its ``-num_epochs``).
    extra_args : iterable[str], optional
        Extra arguments appended to the GhostBuster command.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype key (its sample-node id), a **soft** painting over ``[0, L)``.

    Notes
    -----
    Unlike the VCF runners, the keys here are the tree sequence's **sample node ids**, not
    ``2*j+h`` VCF haplotype indices — there is no VCF, so there is nothing to re-key against.
    """
    import tskit

    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_ghostbuster_")
    os.makedirs(workdir, exist_ok=True)
    if not C.tool_available("ghostbuster"):
        raise FileNotFoundError(
            f"GhostBuster not installed at {C.GHOSTBUSTER_DIR} — run "
            f"`tspaint benchmark install ghostbuster`")

    samples = list(ts.samples())
    queries = list(queries) if queries is not None else [s for s in samples if s not in labels]
    if not queries:
        raise ValueError("no query samples (every sample is in `labels`)")
    states = sorted(set(labels.values()))
    K = len(states)
    k = int(clusters or K)

    # --- inputs -------------------------------------------------------------------------------
    # GhostBuster assumes strictly **binary** trees (Relate's): it does `tree.children(u)[1]`
    # unguarded and dies with an IndexError on any unary node. tspaint's own simulations carry
    # census nodes, which *are* unary, so strip them. `filter_nodes=False` keeps the node table (and
    # therefore every sample id, which the caller's `labels` / `queries` are keyed by) intact.
    ts_gb = ts.simplify(samples=samples, keep_unary=False, filter_nodes=False)

    # trees: GhostBuster does `tskit.load(args.trees + str(chr) + ".trees")`, so --trees is a prefix.
    tprefix = os.path.join(workdir, "gb_chr")
    _with_relate_edge_metadata(ts_gb).dump(f"{tprefix}{chromosome}.trees")

    # poplabels: Relate's format, ONE ROW PER DIPLOID INDIVIDUAL (ID GROUP SAMPLING_TIME INCLUDE).
    # Sample node 2i / 2i+1 are individual i's two haplotypes, so a sample's group is read off the
    # individual. A reference individual's two haplotypes must therefore share a state; mixed pairs
    # are rejected rather than silently mislabelled.
    poplabels = os.path.join(workdir, "poplabels.txt")
    with open(poplabels, "w") as f:
        f.write("ID\tGROUP\tSAMPLING_TIME\tINCLUDE\n")
        for i in range(0, len(samples), 2):
            a, b = samples[i], samples[i + 1] if i + 1 < len(samples) else samples[i]
            sa, sb = labels.get(a), labels.get(b)
            if sa is not None and sb is not None and sa != sb:
                raise ValueError(
                    f"reference individual {i} has haplotypes of different states ({sa}, {sb}); "
                    f"GhostBuster's poplabels are per-individual, so a reference individual's two "
                    f"haplotypes must share a state")
            grp = f"P{sa}" if sa is not None else ("P{}".format(sb) if sb is not None else "TARGET")
            f.write(f"tsk_{i // 2}\t{grp}\t0\t1\n")

    # recombination map: HapMap 4-column, one file per chromosome
    rprefix = os.path.join(workdir, "recmap_chr")
    pos = np.array([s.position for s in ts_gb.sites()], float)
    if pos.size == 0:                                       # no mutations: a 2-point flat map
        pos = np.array([0.0, float(ts.sequence_length) - 1])
    with open(f"{rprefix}{chromosome}.txt", "w") as f:
        f.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
        for p in pos:
            f.write(f"chr{chromosome}\t{int(p)}\t{recomb_rate * 1e8:.6f}"
                    f"\t{p * recomb_rate * 100.0:.10f}\n")

    prefix = os.path.join(workdir, "gb")
    args = ["--trees", tprefix, "--poplabels", poplabels, "--rec", rprefix,
            "--chrs", str(int(chromosome)), "-o", prefix,
            "--sample_id", *[str(int(q)) for q in queries],
            "--groups", *[f"P{s}" for s in states],
            "-k", str(k), "-i", str(int(n_iters)),
            "--hmm", "True" if hmm else "False"]
    if start_time is not None:
        args += ["-start_time", str(float(start_time))]
    if end_time is not None:
        args += ["-end_time", str(float(end_time))]
    if n_epochs is not None:
        args += ["-num_epochs", str(int(n_epochs))]
    if extra_args:
        args += list(extra_args)
    C.run_tool("ghostbuster", args, cwd=workdir, log=log)

    # --- output -------------------------------------------------------------------------------
    comp_state = _component_states(prefix, states)
    # <prefix>_overall_membership_<name>_sample_id_<id>.csv: chr, pos, prob_0..prob_{k-1}, genpos
    L = float(ts.sequence_length)
    tracks = {}
    for q in queries:
        hits = glob.glob(f"{prefix}_overall_membership_*_sample_id_{int(q)}.csv")
        if not hits:
            tracks[q] = [Segment(0.0, L, np.full(K, 1.0 / K), MISSING_INFO)]
            continue
        rows = [ln.split("\t") for ln in open(hits[0]).read().splitlines() if ln]
        hdr, data = rows[0], rows[1:]
        pcols = [j for j, h in enumerate(hdr) if h.startswith("prob_")]
        ipos = hdr.index("pos")
        p = np.array([float(r[ipos]) for r in data])
        post = np.array([[float(r[j]) for j in pcols] for r in data])       # (S, k)
        # Collapse the latent components onto our K states via `comp_state`. A ghost component (one
        # that matches no reference) has no state and is dropped: it is not one of the reference
        # ancestries, and the remaining mass is renormalised.
        cols = np.zeros((len(p), K))
        for a in range(post.shape[1]):
            st = comp_state[a]
            if st is not None:
                cols[:, st] += post[:, a]
        s = cols.sum(axis=1, keepdims=True)
        cols = np.divide(cols, s, out=np.full_like(cols, 1.0 / K), where=s > 0)
        tracks.update(tracks_from_marker_posteriors(p, {q: cols}, L, atol=1e-9))

    for q in queries:
        if not tracks.get(q):
            tracks[q] = [Segment(0.0, L, np.full(K, 1.0 / K), MISSING_INFO)]
    if out:
        from ..serialize import save_painting
        save_painting(out, tracks, seqlen=L, deadband=0.0,
                      sample_names={q: f"node{q}" for q in queries})
        if log:
            log(f"ghostbuster: {len(queries)} query haplotypes -> {out}")
    return tracks


def _with_relate_edge_metadata(ts):
    """Return ``ts`` with Relate-style edge metadata, which GhostBuster requires.

    GhostBuster is documented as working on "a genealogy in tree sequence format", and its paper
    says it is "in principle compatible with any genealogy inference method" — but the code is not:
    it reads Relate-specific **edge metadata** and dies with an ``IndexError`` on any other tree
    sequence (msprime, tsinfer, SINGER, a true ARG). Three fields are consumed::

        metadata = "<edge_left> <edge_right> <n_mutations_on_edge>"

    (``[0]``/``[1]`` in ``infer_node_persistence``, which uses them to measure how far an edge
    persists *across* local trees; ``[2]`` in ``calc_tree_stats``, the mutation count.)

    Every one of those is already in the tree sequence — the edge table's own ``left``/``right``,
    and the mutations sitting on the edge's child within that span. So this is a lossless
    re-derivation of what ``relate_lib``'s ``Convert`` writes, not a fabrication, and it is what
    lets the head-to-head run on the *same ARG* tspaint paints regardless of which front end
    produced it. (It is also a nice independent confirmation of CLAUDE.md §5: GhostBuster's node
    persistence rests on exactly the cross-tree edge persistence our accumulator does.)
    """
    import tskit

    tables = ts.dump_tables()
    mut_node = tables.mutations.node
    mut_pos = tables.sites.position[tables.mutations.site] if ts.num_mutations else np.empty(0)

    # group mutation positions by the node they sit on, so each edge is one slice + a range count
    order = np.lexsort((mut_pos, mut_node)) if ts.num_mutations else np.empty(0, int)
    mn = mut_node[order] if ts.num_mutations else np.empty(0, int)
    mp = mut_pos[order] if ts.num_mutations else np.empty(0)
    lo = np.searchsorted(mn, np.arange(ts.num_nodes), side="left")
    hi = np.searchsorted(mn, np.arange(ts.num_nodes), side="right")

    meta = []
    for e in ts.edges():
        p = mp[lo[e.child]:hi[e.child]]
        n = int(np.count_nonzero((p >= e.left) & (p < e.right)))
        meta.append(f"{e.left} {e.right} {n}".encode())

    tables.edges.metadata_schema = tskit.MetadataSchema.null()
    tables.edges.packset_metadata(meta)
    return tables.tree_sequence()


def _component_states(prefix, states):
    """Map each latent component onto a reference state, using GhostBuster's own fitted rates.

    **This is load-bearing, not cosmetic.** GhostBuster's components are *free*: like MOSAIC's, they
    are decoupled from the reference panels (that is how a ghost can be represented at all), so they
    come out in whatever order EM lands on — **not** in ``--groups`` order. Assuming
    "component *a* == state *a*" silently produces a *perfectly swapped* painting: measured on a
    strong-structure sim, balanced accuracy **0.000 at confidence 0.998** — confidently wrong
    everywhere, which is far worse than being wrong noisily.

    The honest mapping is GhostBuster's own output: ``<prefix>_gamma_<ids>.npy`` is the fitted
    coalescence-rate array ``(component, group, epoch)``, and the ``.coal`` file's first line names
    the groups (``np.unique`` of the poplabels ``GROUP`` column, so alphabetical). A component's
    ancestry is the **reference group it coalesces with fastest** — higher rate, sooner common
    ancestor. The truth table is never consulted.

    Returns
    -------
    dict[int, int | None]
        Component index → reference state, or ``None`` for a component that matches no reference
        group (a ghost), which the caller drops from the painting.
    """
    gam = sorted(glob.glob(f"{prefix}_gamma_*.npy"))
    coal = sorted(glob.glob(f"{prefix}_*.coal"))
    gam = [g for g in gam if "nohmm" not in g] or gam
    if not gam or not coal:
        raise FileNotFoundError(
            f"GhostBuster wrote no gamma/.coal files under {prefix}* — cannot map its latent "
            f"components onto reference states without them")

    with open(coal[0]) as f:
        groups = f.readline().split()
    gamma = np.load(gam[0])                                  # (k, n_groups, n_epochs)
    ref_idx = [i for i, g in enumerate(groups) if g.startswith("P") and g[1:].isdigit()]
    if not ref_idx:
        raise ValueError(f"{coal[0]}: no reference groups among {groups}")
    ref_state = [int(groups[i][1:]) for i in ref_idx]

    total = gamma[:, ref_idx, :].sum(axis=2)                 # (k, n_ref) total rate per group
    out = {}
    for a in range(gamma.shape[0]):
        if not np.any(total[a] > 0):
            out[a] = None                                    # coalesces with no reference: a ghost
            continue
        st = ref_state[int(total[a].argmax())]
        out[a] = st if st in states else None
    return out
