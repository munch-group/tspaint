"""Per-locus foreignness diagnostics for references and queries (CLAUDE.md §2.3, §9).

The shared engine (Plan A) behind reference QC, anonymous foreign-tract inference, and
ghost-source detection. For each focal sample, per genomic segment, three components:

* ``loo``   — the leave-one-out posterior (the outside message,
  :func:`tspaint.output.loo_posterior_table`): what the rest of the genealogy says about the
  tip *ignoring its own label*.
* ``fit``   — ``max_s loo[s]``: the genealogy's confidence in *any* panel state. Low ``fit``
  (≈ ``1/K``) means the tract fits no reference well ("fits nothing").
* ``depth`` — coalescence depth to the **nearest labelled reference**, rank-normalised
  genome-wide by default (calibration-robust; ``depth="time"`` keeps raw coalescent time).
  High ``depth`` = a deep outlier.

The separation that matters: a merely *uninformative* tract has low ``fit`` but **shallow**
``depth``; a *ghost / archaic* tract has low ``fit`` AND **high** ``depth``. That is why
``depth`` is carried alongside ``fit`` — it distinguishes "fits nothing because the local
tree can't tell" from "fits nothing because it descends from a population not in the panel".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tskit

from .pruning import prune_tree
from .output import INFORMATIVE, MISSING_INFO, loo_posterior_table, DEFAULT_QC_DEADBAND
from .em import fit, build_emissions
from .model import make_generator_2state

__all__ = ["ForeignnessSegment", "foreignness_track", "ReferenceQC", "reference_qc",
           "foreign_tracts"]


@dataclass
class ForeignnessSegment:
    """A contiguous span carrying the per-locus foreignness components for one sample.

    Attributes
    ----------
    left, right : float
        Half-open genomic interval ``[left, right)``.
    loo : numpy.ndarray
        ``(K,)`` leave-one-out posterior (the outside message excluding the sample's own
        emission).
    fit : float
        ``max_s loo[s]`` in ``[1/K, 1]`` — the genealogy's confidence in any panel state.
    depth : float
        Coalescence depth to the nearest labelled reference: a genome-wide rank in
        ``[0, 1]`` (``depth="rank"``, default) or raw coalescent time (``depth="time"``);
        ``nan`` where no reference is reachable.
    status : str
        :data:`tspaint.output.INFORMATIVE` or :data:`tspaint.output.MISSING_INFO`.
    """
    left: float
    right: float
    loo: np.ndarray
    fit: float
    depth: float
    status: str


def _nearest_ref_depth(tree, s, ref_ids, node_time):
    """Coalescent time to the nearest labelled reference (excluding ``s`` itself)."""
    best = np.inf
    for r in ref_ids:
        if r == s:
            continue
        m = tree.mrca(s, r)
        if m == tskit.NULL:
            continue
        t = node_time[m]
        if t < best:
            best = t
    return float(best) if np.isfinite(best) else float("nan")


def _rank_normalise_depth(tracks):
    """Replace raw nearest-ref depths with their span-weighted genome-wide quantile in ``[0, 1]``."""
    depths, spans = [], []
    for segs in tracks.values():
        for seg in segs:
            if seg.status == INFORMATIVE and np.isfinite(seg.depth):
                depths.append(seg.depth)
                spans.append(seg.right - seg.left)
    if not depths:
        for segs in tracks.values():
            for seg in segs:
                seg.depth = float("nan")
        return
    depths = np.asarray(depths, float)
    spans = np.asarray(spans, float)
    order = np.argsort(depths, kind="mergesort")
    sd = depths[order]
    csum = np.cumsum(spans[order])
    total = csum[-1]
    for segs in tracks.values():
        for seg in segs:
            if seg.status == INFORMATIVE and np.isfinite(seg.depth):
                idx = int(np.searchsorted(sd, seg.depth, side="right"))
                seg.depth = float(csum[idx - 1] / total) if idx > 0 else 0.0
            else:
                seg.depth = float("nan")


def foreignness_track(ts, Q, pi, emissions, labels, focal=None, depth="rank", merge_tol=1e-9,
                      tree_range=None):
    """Per-sample foreignness components as contiguous segments covering ``[0, L)``.

    Single pass over the marginal trees (one pruning per tree); for each focal sample records
    the leave-one-out posterior ``loo``, the ``fit = max_s loo[s]`` and the nearest-reference
    coalescence ``depth``. Adjacent segments with equal components are merged.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence whose marginal trees are pruned.
    Q : (K, K) numpy.ndarray
        Ancestry CTMC generator.
    pi : (K,) array_like
        Root frequencies ``π`` (prior fallback on uninformative spans).
    emissions : dict[int, numpy.ndarray]
        Per-sample emission vectors (e.g. from :func:`tspaint.em.build_emissions`).
    labels : dict[int, int] or iterable[int]
        The labelled reference sample ids (only the keys / ids are used — they define which
        tips the nearest-reference depth is measured against).
    focal : iterable[int], optional
        Samples to record; defaults to every sample in ``ts``.
    depth : {"rank", "time"}, optional
        ``"rank"`` (default) rank-normalises the nearest-reference depth genome-wide into
        ``[0, 1]`` (robust to branch-length miscalibration, CLAUDE.md §6); ``"time"`` keeps the
        raw coalescent time.
    merge_tol : float, optional
        Absolute tolerance for merging adjacent segments with equal ``loo``.

    Returns
    -------
    dict[int, list[ForeignnessSegment]]
        Per focal sample, the foreignness components as contiguous
        :class:`ForeignnessSegment`\\ s covering ``[0, L)``.
    """
    if depth not in ("rank", "time"):
        raise ValueError("depth must be 'rank' or 'time'")
    pi = np.asarray(pi, float)
    node_time = ts.tables.nodes.time
    ref_ids = [int(r) for r in labels]
    samples = [int(s) for s in (ts.samples() if focal is None else focal)]
    tracks = {s: [] for s in samples}

    lo, hi = (0, ts.num_trees) if tree_range is None else tree_range
    for ti, tree in enumerate(ts.trees()):
        if ti < lo:
            continue
        if ti >= hi:
            break
        left, right = tree.interval.left, tree.interval.right
        res = prune_tree(tree, emissions, Q, node_time, pi)
        for s in samples:
            loo = np.asarray(res.loo.get(s, pi), float)
            fit = float(loo.max())
            status = MISSING_INFO if s in res.missing_info else INFORMATIVE
            d = _nearest_ref_depth(tree, s, ref_ids, node_time)
            segs = tracks[s]
            same_depth = (segs and ((segs[-1].depth == d)
                                    or (np.isnan(segs[-1].depth) and np.isnan(d))))
            if (segs and segs[-1].right == left and segs[-1].status == status and same_depth
                    and np.allclose(segs[-1].loo, loo, atol=merge_tol, rtol=0)):
                segs[-1].right = right
            else:
                segs.append(ForeignnessSegment(left, right, loo, fit, d, status))

    if depth == "rank" and tree_range is None:   # a chunk returns raw depth; parent normalises genome-wide
        _rank_normalise_depth(tracks)
    return tracks


# --- Workflow 1: Reference QC ---------------------------------------------------------------

def _loo_agreement(segs, label):
    """Span-weighted mean leave-one-out posterior on a reference's own label (its credibility)."""
    num = den = 0.0
    for seg in segs:
        if seg.status == MISSING_INFO:
            continue
        w = seg.right - seg.left
        den += w
        num += w * float(seg.posterior[label])
    return num / den if den > 0 else float("nan")


def _merge_segments(tracts):
    """Merge adjacent same-state ``(left, right, state)`` tracts."""
    out = []
    for (l, r, st) in tracts:
        if out and out[-1][2] == st and out[-1][1] == l:
            out[-1] = (out[-1][0], r, st)
        else:
            out.append((l, r, st))
    return out


@dataclass
class ReferenceQC:
    """Result of :func:`reference_qc` — a per-reference admixture/mislabel audit.

    Attributes
    ----------
    labels : dict[int, int]
        Reference **sample-node (haplotype) index** -> its nominal ancestry-state label. Keys are
        haplotype nodes even when :func:`reference_qc` was called with base-id / individual labels,
        because credibility is learned per tip (per haplotype). Use :attr:`individual_ids` /
        :attr:`sample_ids` (or :meth:`summary`) to read them back as the ids you passed in.
    credibility : dict[int, float]
        Per-reference credibility in ``[0, 1]`` — the learned ``w_i`` for softened (suspect)
        references, the leave-one-out self-agreement for the hard-clamped anchor core. Low =
        the genealogy disagrees with the label (admixed / mislabelled).
    loo_agreement : dict[int, float]
        Pass-1 leave-one-out self-agreement for every reference (the hard-clamp baseline).
    learned_w : dict[int, float]
        Learned credibility ``w_i`` for the softened references only.
    anchors : set[int]
        References kept hard-clamped as the trusted core (CLAUDE.md §6).
    maps : dict[int, list]
        Per reference, its leave-one-out introgression map (:func:`tspaint.output.loo_posterior_table`
        segments) — where its own genealogy dissents from its label.
    Q, pi : numpy.ndarray
        The fitted generator and root frequencies (from the refined pass when refining).
    sample_ids : dict[int, str] or None
        Per-reference haplotype-node -> its stamped **haplotype** id (e.g. ``"PD_0213_1"``), when the
        tree sequence carried sample names (from :func:`tspaint.io.tsinfer` / ``singer`` / ``relate``);
        ``None`` for an unstamped (bare-sim) tree sequence.
    individual_ids : dict[int, str] or None
        Per-reference haplotype-node -> its **individual** id (e.g. ``"PD_0213"``) — a diploid
        reference's two haplotype nodes share one individual id. Lets :meth:`summary` /
        :meth:`soft_refs` speak in the ids you passed in rather than raw node indices.
    """
    labels: dict
    credibility: dict
    loo_agreement: dict
    learned_w: dict
    anchors: set
    maps: dict
    Q: np.ndarray
    pi: np.ndarray
    _length: float = 0.0
    sample_ids: dict = None
    individual_ids: dict = None

    def introgression_map(self, ref):
        """The leave-one-out introgression map (list of segments) for one reference."""
        return self.maps[int(ref)]

    def flagged_tracts(self, ref, deadband=DEFAULT_QC_DEADBAND):
        """Foreign tracts of ``ref``: where the genealogy confidently prefers a non-label state.

        A segment is flagged where ``loo[foreign] - loo[label] >= deadband`` (``foreign`` =
        the best non-label state). Returns merged ``[(left, right, foreign_state)]``.
        """
        ref = int(ref)
        lab = self.labels[ref]
        out = []
        for seg in self.maps[ref]:
            if seg.status == MISSING_INFO:
                continue
            p = np.asarray(seg.posterior, float)
            foreign = int(np.argmax([p[s] if s != lab else -np.inf for s in range(p.shape[0])]))
            if p[foreign] - p[lab] >= deadband:
                out.append((seg.left, seg.right, foreign))
        return _merge_segments(out)

    def foreign_fraction(self, ref, deadband=DEFAULT_QC_DEADBAND):
        """Fraction of ``ref``'s genome flagged foreign at the given ``deadband``."""
        tracts = self.flagged_tracts(ref, deadband)
        return sum(r - l for (l, r, _) in tracts) / self._length if self._length else float("nan")

    def summary(self, deadband=DEFAULT_QC_DEADBAND):
        """Per-reference-**haplotype** QC rows, least-credible first (the audit table).

        One row per reference haplotype (two per diploid individual — credibility is learned per
        tip). ``ref`` is the sample-node index; when the tree sequence carried sample names the row
        also names the ``individual`` (base id) and ``haplotype`` (per-haplotype id) so the table is
        legible in your own ids, not just node integers.
        """
        rows = []
        for r in sorted(self.labels, key=lambda r: self.credibility[r]):
            row = {}
            if self.individual_ids and self.individual_ids.get(r) is not None:
                row["individual"] = self.individual_ids[r]
            if self.sample_ids and self.sample_ids.get(r) is not None:
                row["haplotype"] = self.sample_ids[r]
            row.update({"ref": r, "label": self.labels[r], "credibility": self.credibility[r],
                        "is_anchor": r in self.anchors,
                        "foreign_fraction": self.foreign_fraction(r, deadband)})
            rows.append(row)
        return rows

    def soft_refs(self, max_credibility=None, by="node"):
        """References to soften when re-painting — feed straight to ``paint(..., soft_refs=...)``.

        The Task-1 action: down-weight the contaminated references so they stop anchoring the
        painting where they carry foreign ancestry. With ``max_credibility=None`` (default) returns
        the suspect set the QC already softened (the non-anchor references); pass a cutoff to take
        every reference with ``credibility < max_credibility`` instead. Keep the trusted anchors
        hard-clamped — never let the whole panel float (CLAUDE.md §6).

        Parameters
        ----------
        max_credibility : float, optional
            Cutoff (see above); default returns the QC's softened (non-anchor) set.
        by : {"node", "individual"}, optional
            ``"node"`` (default) returns sample-**node** indices; ``"individual"`` returns the
            distinct **individual** ids (e.g. ``"PD_0213"``) — more legible, and still accepted by
            ``paint(..., soft_refs=...)`` (which resolves ids). ``"individual"`` requires a stamped
            tree sequence; a diploid individual is included if *either* of its haplotypes is suspect.

        Returns
        -------
        set[int] or set[str]
            Node indices (``by="node"``) or individual ids (``by="individual"``) to pass as
            ``soft_refs``.
        """
        if max_credibility is None:
            nodes = {int(r) for r in self.labels if r not in self.anchors}
        else:
            nodes = {int(r) for r, c in self.credibility.items() if c < max_credibility}
        if by == "node":
            return nodes
        if by != "individual":
            raise ValueError(f"by must be 'node' or 'individual', got {by!r}")
        if not self.individual_ids:
            raise ValueError("soft_refs(by='individual') needs a stamped tree sequence (from "
                             "io.tsinfer / io.singer / io.relate with sample names); this QC has none")
        return {self.individual_ids[n] for n in nodes if n in self.individual_ids}

    def mask(self, deadband=DEFAULT_QC_DEADBAND):
        """Per-reference foreign spans to **mask** out before re-painting.

        The alternative Task-1 action to :meth:`soft_refs`: rather than down-weight a whole
        reference, drop only its contaminated spans. Returns ``{ref: [(left, right), ...]}`` from
        :meth:`flagged_tracts` (foreign state dropped) for the references that have at least one
        flagged tract.
        """
        out = {}
        for r in self.labels:
            spans = [(l, rr) for (l, rr, _s) in self.flagged_tracts(r, deadband)]
            if spans:
                out[int(r)] = spans
        return out


def _qc_loo_maps(members, res, labels, focal, *, n_jobs, progress):
    """Per-reference leave-one-out introgression maps, for a single tree sequence or an ensemble.

    Single tree sequence: the plain :func:`~tspaint.output.loo_posterior_table`. **Ensemble** (e.g.
    SINGER posterior samples): the LOO table is painted **per member** on the shared pooled fit
    ``res`` and then **merged per reference** — mean map + ARG-uncertainty band — reusing
    :func:`tspaint.ensemble.merge_posterior_tables`, exactly as :func:`tspaint.paint` marginalises an
    ensemble (CLAUDE.md §7.4). The merged ``MergedSegment``\\ s are duck-compatible with ``Segment``,
    so :meth:`ReferenceQC.flagged_tracts` / :meth:`~ReferenceQC.mask` and ``_loo_agreement`` consume
    them unchanged (and gain ``posterior_std``)."""
    from .parallel import loo_posterior_table_parallel
    if len(members) == 1:
        em = build_emissions(members[0], labels, res.w, res.pi)
        return loo_posterior_table_parallel(members[0], res.Q, res.pi, w=res.w, labels=labels,
                                             focal=focal, emissions=em, n_jobs=n_jobs, progress=progress)
    tables = [loo_posterior_table_parallel(g, res.Q, res.pi, w=res.w, labels=labels, focal=focal,
                                           n_jobs=n_jobs, progress=progress) for g in members]
    from .ensemble import merge_posterior_tables
    return merge_posterior_tables(tables, samples=list(focal))


def reference_qc(ts, labels, *, anchors=None, refine=True, anchor_frac=0.5, K=2, Q0=None,
                 max_iter=8, alpha=20.0, beta=1.0, n_jobs=None, progress=False):
    """Audit a reference panel for admixture / mislabelling (Plan A Workflow 1; CLAUDE.md §9).

    Two passes (the second optional). Pass 1 hard-clamps every reference and reads each one's
    **leave-one-out** self-agreement — what the rest of the genealogy says about it, ignoring
    its own label — which flags foreignness even under hard clamps. Pass 2 keeps a trusted
    hard-clamped anchor core and **softens the rest**, learning their credibility ``w_i`` and
    sharpening their introgression maps.

    .. note::
       The auto-anchor bootstrap assumes the **clean references are the majority** (it picks the
       most self-consistent ``anchor_frac`` as the core). If impurity is widespread the core is
       contaminated and discrimination degrades — pass a known-trusted ``anchors`` set instead,
       and never let the whole panel float (CLAUDE.md §6).

    Parameters
    ----------
    ts : tskit.TreeSequence or list[tskit.TreeSequence]
        Tree sequence whose samples include the references (no queries required), or **an ensemble**
        (e.g. SINGER posterior samples). For an ensemble one ``theta`` is fit **pooled** across all
        members, each reference's leave-one-out introgression map is painted **per member** and
        **merged** (mean map + ARG-uncertainty ``posterior_std`` band), exactly as :func:`tspaint.paint`
        marginalises an ensemble (CLAUDE.md §7.4). Members must share samples and coordinates.
    labels : dict[int, int]
        Reference sample id -> nominal ancestry-state label (keyed on the first member for an ensemble).
    anchors : iterable[int], optional
        Known-trusted references to hold as the hard-clamped core (the rest are softened). If
        given, overrides the ``anchor_frac`` auto-selection — the robust choice when you know
        which references are clean.
    refine : bool, optional
        Run the second (soften-suspects) pass when ``anchors`` is not given. Default ``True``.
    anchor_frac : float, optional
        Fraction of references (most self-consistent) auto-selected as the anchor core when
        ``anchors`` is not given. Default ``0.5``.
    K : int, optional
        Number of ancestry states.
    Q0 : numpy.ndarray, optional
        Initial generator (default a symmetric ``1e-3`` 2-state generator).
    max_iter : int, optional
        Maximum EM iterations per pass.
    alpha, beta : float, optional
        Beta prior for the softened references (refine pass).
    n_jobs : int, optional
        Worker processes for the EM fits and the leave-one-out paints (the slow, genome-scale part;
        the per-pass work parallelises exactly like :func:`tspaint.paint`). Default: all CPUs / the SLURM allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for serial.
        (The result's :meth:`ReferenceQC.soft_refs` / :meth:`~ReferenceQC.summary` are cheap
        table lookups — nothing to parallelise there; this is where the compute lives.)
    progress : bool, optional
        Show :mod:`tqdm` bars for the EM fits and LOO paints. Default ``False``.

    Returns
    -------
    ReferenceQC
        Per-reference credibility, introgression maps and the flagged-tract / summary helpers.
    """
    from .ids import resolve_labels, resolve_ids
    members = list(ts) if isinstance(ts, (list, tuple)) else [ts]
    if not members:
        raise ValueError("reference_qc() got an empty ensemble; pass at least one tree sequence")
    ref_ts = members[0]
    labels = resolve_labels(ref_ts, labels)      # keys may be sample-ID strings or node indices
    refs = list(labels)
    Q0 = Q0 if Q0 is not None else make_generator_2state(1e-3, 1e-3)
    # One theta fit pooled across the ensemble (the M-step is scale-invariant, so summing sufficient
    # statistics over members == averaging; CLAUDE.md §7.4). A single tree sequence is the 1-member case.
    fit_ts = members if len(members) > 1 else members[0]
    fit_labels = [labels] * len(members) if len(members) > 1 else labels

    res1 = fit(fit_ts, fit_labels, K=K, Q0=Q0, max_iter=max_iter, estimate_pi=False,
               n_jobs=n_jobs, progress=progress)
    maps1 = _qc_loo_maps(members, res1, labels, refs, n_jobs=n_jobs, progress=progress)
    agreement = {r: _loo_agreement(maps1[r], labels[r]) for r in refs}

    learned = {}
    Q, pi, maps = res1.Q, res1.pi, dict(maps1)
    credibility = dict(agreement)

    if anchors is not None:
        anchor_set = set(resolve_ids(ref_ts, anchors))
    elif refine and len(refs) >= 2:
        n_anchor = min(len(refs) - 1, max(1, int(round(anchor_frac * len(refs)))))
        ranked = sorted(refs, key=lambda r: (agreement[r] if np.isfinite(agreement[r]) else -1.0),
                        reverse=True)
        anchor_set = set(ranked[:n_anchor])
    else:
        anchor_set = set(refs)
    soft = set(refs) - anchor_set

    if soft:
        res2 = fit(fit_ts, fit_labels, K=K, Q0=Q0, max_iter=max_iter, estimate_pi=False,
                   soft_refs=soft, alpha=alpha, beta=beta, n_jobs=n_jobs, progress=progress)
        maps2 = _qc_loo_maps(members, res2, labels, sorted(soft), n_jobs=n_jobs, progress=progress)
        Q, pi = res2.Q, res2.pi
        for r in soft:
            learned[r] = float(res2.w.get(r, 1.0))
            credibility[r] = learned[r]
            maps[r] = maps2[r]

    sample_ids, individual_ids = _ref_id_maps(ref_ts, refs)
    return ReferenceQC(labels=labels, credibility=credibility, loo_agreement=agreement,
                       learned_w=learned, anchors=anchor_set, maps=maps, Q=Q, pi=pi,
                       _length=float(ref_ts.sequence_length),
                       sample_ids=sample_ids, individual_ids=individual_ids)


def _ref_id_maps(ts, refs):
    """``(node -> haplotype id, node -> individual id)`` for the reference nodes, from the stamped
    tree-sequence metadata (:func:`tspaint.ids.attach_sample_ids`); ``(None, None)`` if unstamped."""
    hap, ind = {}, {}
    for r in refs:
        node = ts.node(int(r))
        md = node.metadata
        hid = md.get("id") if isinstance(md, dict) else None
        if hid is not None:
            hap[int(r)] = str(hid)
        if node.individual != -1:
            imd = ts.individual(node.individual).metadata
            iid = imd.get("id") if isinstance(imd, dict) else None
            if iid is not None:
                ind[int(r)] = str(iid)
    return (hap or None), (ind or None)


# --- Workflows 2 & 3: anonymous foreign tracts & ghost-source detection ----------------------

def _fit_and_foreignness(ts, labels, samples, K, Q0, max_iter, soft_refs, depth="rank",
                         n_jobs=None, progress=False):
    """Fit ``θ`` on the panel and return the per-sample foreignness track (parallel over genome
    chunks when ``n_jobs > 1``). ``Q0=None`` lets :func:`tspaint.fit` scale the initial generator to
    the time axis (robust on a large / calibrated node-age scale)."""
    from .parallel import foreignness_track_parallel
    res = fit(ts, labels, K=K, Q0=Q0, max_iter=max_iter, estimate_pi=False, soft_refs=soft_refs,
              n_jobs=n_jobs, progress=progress)
    em = build_emissions(ts, labels, res.w, res.pi)
    return foreignness_track_parallel(ts, res.Q, res.pi, w=res.w, labels=labels, emissions=em,
                                      focal=samples, depth=depth, n_jobs=n_jobs, progress=progress)


def _merge_scored(segs):
    """Merge contiguous ``(left, right, score)`` runs, span-averaging the score."""
    out = []
    for (l, r, sc) in segs:
        if out and abs(out[-1][1] - l) < 1e-9:
            pl, pr, psc = out[-1]
            w1, w2 = pr - pl, r - l
            out[-1] = (pl, r, (psc * w1 + sc * w2) / (w1 + w2))
        else:
            out.append((l, r, sc))
    return out


def _merge_intervals(intervals):
    """Merge contiguous ``(left, right)`` intervals."""
    out = []
    for (l, r) in intervals:
        if out and abs(out[-1][1] - l) < 1e-9:
            out[-1] = (out[-1][0], r)
        else:
            out.append((l, r))
    return out


def foreign_tracts(ts, labels, samples, *, min_score=0.5, min_depth=None, mode="auto", K=2,
                   Q0=None, max_iter=8, soft_refs=None, n_jobs=None, progress=False):
    """Anonymous foreign-tract inference: where a haplotype is foreign to its expectation
    (CLAUDE.md §2.3, §9).

    Flags tracts **without attributing a donor**. The foreignness score is per segment:

    * a **labelled** sample (in ``labels``) — ``1 - loo[label]``: how much the leave-one-out
      genealogy withholds from its own label (its introgression, source-agnostic);
    * an **unlabelled** query — ``(1 - fit) / (1 - 1/K)`` where ``fit = max_s loo[s]``: how
      poorly it fits any panel source ("fits nothing"), normalised so a maximally-ambiguous
      tract scores 1.

    With ``min_depth`` set, a segment must **also** sit on an anomalously deep branch (high
    rank-normalised nearest-reference coalescence depth) to be flagged — the fast, deterministic,
    calibration-robust **ghost flag** (this subsumes the former ``detect_ghost`` *flag*; the
    accurate generative ghost detector is now :func:`tspaint.detect_ghost`, the depth-emission HMM).

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to analyse.
    labels : dict[int, int]
        Reference sample id -> ancestry-state label.
    samples : iterable[int]
        Samples to scan (references and/or queries).
    min_score : float, optional
        Flag segments whose foreignness score ``>= min_score`` (default ``0.5`` — the expected
        ancestry gets less than half the support).
    min_depth : float, optional
        If given, additionally require the rank-normalised nearest-reference depth ``>= min_depth``
        (in ``[0, 1]``) — flag only *deep* foreign tracts (ghost-like; e.g. ``0.9`` for the
        deepest 10%). Default ``None`` (no depth filter). Combined with ``mode="fit"`` /
        ``min_score`` this reproduces the former ghost flag (``fit < f`` ⇔ ``score > 2(1-f)``).
    mode : {"auto", "label", "fit"}, optional
        ``"auto"`` (default) uses the label rule for labelled samples and the fit rule for
        queries; ``"label"`` / ``"fit"`` force one rule for all samples.
    K, Q0, max_iter, soft_refs
        Model / EM controls (see :func:`tspaint.fit`).
    n_jobs : int, optional
        Worker processes for the fit **and** the per-tree foreignness pass (split by genome chunk,
        exactly equal to serial). Default: all CPUs / the SLURM allocation (:func:`tspaint.parallel.resolve_cores`); pass ``1`` for serial.
    progress : bool, optional
        Show a progress bar for the fit / foreignness pass. Default ``False``.

    Returns
    -------
    dict[int, list[tuple[float, float, float]]]
        Per sample, merged foreign tracts ``(left, right, score)``.
    """
    if mode not in ("auto", "label", "fit"):
        raise ValueError("mode must be 'auto', 'label' or 'fit'")
    from .ids import resolve_labels, resolve_ids
    labels = resolve_labels(ts, labels)          # keys may be sample-ID strings or node indices
    samples = resolve_ids(ts, samples)
    ft = _fit_and_foreignness(ts, labels, samples, K, Q0, max_iter, soft_refs, depth="rank",
                              n_jobs=n_jobs, progress=progress)
    out = {}
    for s in samples:
        use_label = (mode == "label") or (mode == "auto" and s in labels)
        flagged = []
        for seg in ft[s]:
            if seg.status == MISSING_INFO:
                continue
            if min_depth is not None and not (np.isfinite(seg.depth) and seg.depth >= min_depth):
                continue                       # deep-only (ghost) filter
            score = ((1.0 - float(seg.loo[labels[s]])) if use_label
                     else (1.0 - seg.fit) / (1.0 - 1.0 / K))
            if score >= min_score:
                flagged.append((seg.left, seg.right, score))
        out[s] = _merge_scored(flagged)
    return out


# NOTE: the former Plan-A ``detect_ghost`` *flag* (fit < t AND depth >= t) and its ``GhostResult``
# have been folded into ``foreign_tracts(mode="fit", min_score=..., min_depth=...)`` above. The name
# ``detect_ghost`` now refers to the accurate depth-emission HMM in :mod:`tspaint.archaic`.
