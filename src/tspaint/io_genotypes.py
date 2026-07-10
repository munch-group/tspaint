"""Unified genotype ingestion for the inference front ends (CLAUDE.md §5).

The inference front ends (:func:`tspaint.io.tsinfer`, :func:`tspaint.io.singer`) accept the
same ``source``: a :class:`tskit.TreeSequence`, a **VCF Zarr** store (the sgkit / ``vcf2zarr``
layout), or a **VCF** file. The native ``ts`` path is handled by each tool directly; this module
normalises the other two into a tool-agnostic :class:`Variants` and builds whatever each tool
needs from it (tsinfer ``SampleData`` or a haploid VCF).

For **inference**, the zarr path is read **chunked** via :func:`variant_data_from_zarr`
(``tsinfer.VariantData`` — tsinfer's native VCF-Zarr reader, scalable to whole-genome data, no
new dependency); the in-memory :class:`Variants` readers below back the VCF path and SINGER's
VCF export. Scope (v1, dependency-light — only ``zarr``, which tsinfer already needs):
**biallelic** sites, a single contig, the ``GT`` field; ``.`` (missing) is read as the reference
allele. Convert with ``bio2zarr`` / ``vcf2zarr`` upstream for richer VCF/Zarr data.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass

import numpy as np
import tskit

__all__ = ["Variants", "source_kind", "resolve_variants", "subset_data", "variants_from_vcf",
           "variants_from_zarr", "variant_data_from_zarr", "to_sample_data", "write_haploid_vcf",
           "write_vcz", "sample_names_from_zarr", "estimate_ne", "pseudohaploid"]


@dataclass
class Variants:
    """A tool-agnostic biallelic variant matrix (one column per **haplotype**).

    Attributes
    ----------
    positions : numpy.ndarray
        ``(S,)`` integer site positions (sorted, strictly increasing).
    genotypes : numpy.ndarray
        ``(S, H)`` allele indices (``int8``; 0 = ancestral/REF, 1 = derived/ALT) over ``H``
        haplotype columns (``samples × ploidy``).
    alleles : list[tuple[str, str]]
        Per-site ``(ref, alt)`` allele strings.
    sequence_length : float
        Sequence length for the inferred tree sequence.
    sample_names : list[str] or None
        Optional haplotype/sample names (length ``H``). For ``ploidy > 1`` these are the flattened
        per-haplotype names ``"<sample>_<k>"`` (0-based ``k``), as produced by the readers below.
    ploidy : int
        Haplotypes per sample (``H = num_samples * ploidy``); ``1`` for haploid data. Lets the front
        ends recover the base sample id from :attr:`sample_names` when stamping the tree sequence
        (:func:`tspaint.ids.attach_sample_ids`).
    missing : numpy.ndarray or None
        Optional ``(S, H)`` boolean mask, ``True`` where the genotype was **missing** (``.`` in a
        VCF, negative in a Zarr) — in :attr:`genotypes` a missing call is stored as ``0`` (REF), so
        this mask is the only record of it. ``None`` when there is no missing data. Used by
        :func:`estimate_ne` to compare only the sites called in both haplotypes.
    sample_index : numpy.ndarray or None
        Optional ``(H,)`` integer array mapping each haplotype column to the **base sample /
        individual** it belongs to. ``None`` means uniform ploidy — column ``j`` belongs to
        individual ``j // ploidy`` — which is the common case. It is set only when the samples have
        **mixed ploidy** (e.g. a chrX VCF with diploid females and haploid males), where a scalar
        ``ploidy`` cannot express the grouping; :func:`estimate_ne` and the id helpers use it to
        keep individuals correct.
    """
    positions: np.ndarray
    genotypes: np.ndarray
    alleles: list
    sequence_length: float
    sample_names: list = None
    ploidy: int = 1
    missing: np.ndarray = None
    sample_index: np.ndarray = None

    @property
    def num_sites(self):
        """Number of variant sites ``S`` (length of :attr:`positions`)."""
        return int(self.positions.shape[0])

    @property
    def num_haplotypes(self):
        """Number of haplotype columns ``H`` (``num_samples * ploidy``)."""
        return int(self.genotypes.shape[1])


def source_kind(source):
    """Classify a front-end ``source`` as ``"ts"``, ``"zarr"`` or ``"vcf"``.

    A :class:`tskit.TreeSequence` is ``"ts"``; a path ending ``.vcf`` / ``.vcf.gz`` / ``.vcf.bgz``
    (case-insensitively) is ``"vcf"``; anything else path-like (a ``.zarr`` store / directory /
    mapping) is ``"zarr"``.

    Parameters
    ----------
    source : tskit.TreeSequence, str, or zarr store / mapping
        The front-end genotype source to classify.

    Returns
    -------
    str
        Exactly one of ``"ts"`` (a :class:`tskit.TreeSequence`), ``"vcf"`` (a VCF path), or
        ``"zarr"`` (any other path-like source, read by :func:`variants_from_zarr`).
    """
    if isinstance(source, tskit.TreeSequence):
        return "ts"
    s = str(source).lower()
    if s.endswith(".vcf") or s.endswith(".vcf.gz") or s.endswith(".vcf.bgz"):
        return "vcf"
    return "zarr"


def resolve_variants(source):
    """Normalise a ``source`` into :class:`Variants` (a :class:`Variants` is returned as-is).

    Parameters
    ----------
    source : Variants, str, or zarr store / mapping
        A genotype source to load: an existing :class:`Variants` (returned unchanged), a **VCF**
        path (read by :func:`variants_from_vcf`), or a **VCF Zarr** store (read by
        :func:`variants_from_zarr`). A :class:`tskit.TreeSequence` is **not** accepted here — the
        ``ts`` path is handled by each front end directly (see Raises).

    Returns
    -------
    Variants
        The normalised variant matrix.

    Raises
    ------
    ValueError
        If ``source`` is a :class:`tskit.TreeSequence` (kind ``"ts"``); resolve a tree sequence
        through the front end's own ``ts`` path instead.
    """
    if isinstance(source, Variants):
        return source
    kind = source_kind(source)
    if kind == "vcf":
        return variants_from_vcf(source)
    if kind == "zarr":
        return variants_from_zarr(source)
    raise ValueError(f"resolve_variants expects a VCF or zarr source, not a {kind}")


# --- chrX / hemizygosity: sex, pseudo-autosomal regions, pseudo-haploid expansion ------------
#
# tskit / tsinfer / SINGER are all haplotype-level: every sample node is one haplotype and there is
# no hemizygosity concept. chrX is handled correctly by giving them the right haploid samples —
# females contribute 2, males 1 outside the pseudo-autosomal regions (PAR, where males are diploid).
# These helpers determine sex + PAR (from a sex map, or inferred from X-heterozygosity) and expand a
# source to one haploid sample per real haplotype (:func:`pseudohaploid`).


def _normalize_sex_map(sex_map):
    """Coerce a ``sex_map`` to ``{str id: 'F'|'M'}``.

    Accepts a ``{id: 'F'/'M'}`` mapping or a 2-column ``(id, sex)`` pandas DataFrame; values are
    read case-insensitively by first letter (``'female'`` / ``'f'`` -> ``'F'``). Returns ``None``
    for ``None``.
    """
    if sex_map is None:
        return None
    if hasattr(sex_map, "iloc") and hasattr(sex_map, "columns"):        # pandas DataFrame
        if sex_map.shape[1] < 2:
            raise ValueError("sex_map DataFrame must have 2 columns (id, sex)")
        items = zip(sex_map.iloc[:, 0].astype(str).tolist(), sex_map.iloc[:, 1].astype(str).tolist())
    elif hasattr(sex_map, "items"):
        items = ((str(k), str(v)) for k, v in sex_map.items())
    else:
        raise TypeError("sex_map must be a {id: 'F'/'M'} dict or a 2-column (id, sex) DataFrame")
    out = {}
    for k, v in items:
        s = v.strip().upper()[:1] if v else ""
        if s not in ("F", "M"):
            raise ValueError(f"sex_map value {v!r} for {k!r} must be 'F'/'M' (female/male)")
        out[k] = s
    return out


def _group_columns(v):
    """Group haplotype columns into base samples: ``(list_of_column_lists, base_ids)``.

    Uses :attr:`Variants.sample_index` when present (mixed ploidy), else the scalar
    :attr:`Variants.ploidy` (uniform, column ``j`` -> individual ``j // ploidy``). Base ids come from
    :attr:`Variants.sample_names` (the shared prefix with the trailing ``_<k>`` stripped).
    """
    H = v.num_haplotypes
    if v.sample_index is not None:
        idx = np.asarray(v.sample_index)
        groups = [list(np.nonzero(idx == k)[0]) for k in range(int(idx.max()) + 1)]
        groups = [g for g in groups if g]
    else:
        p = max(1, int(v.ploidy))
        groups = [list(range(k, min(k + p, H))) for k in range(0, H, p)]
    names = v.sample_names
    base_ids = []
    for g in groups:
        if names is None:
            base_ids.append(f"s{len(base_ids)}")
        elif len(g) > 1:
            base_ids.append(re.sub(r"_\d+$", "", str(names[g[0]])))
        else:
            base_ids.append(str(names[g[0]]))
    return groups, base_ids


def _het_mask(v, a, b):
    """Boolean over sites: haplotype columns ``a`` and ``b`` differ and are both called."""
    both = np.ones(v.genotypes.shape[0], bool)
    if v.missing is not None:
        both = ~v.missing[:, a] & ~v.missing[:, b]
    return (v.genotypes[:, a] != v.genotypes[:, b]) & both


def _infer_sex(v, groups, base_ids):
    """Infer ``{base_id: 'F'|'M'}`` from **interior** X-heterozygosity.

    A male carries one X, so he is heterozygous only across the small PAR at the chromosome ends and
    is **homozygous throughout the interior**; a female is heterozygous everywhere. So a sample is
    male if it is a single (haploid-encoded) column, or if its heterozygosity in the central half of
    the chromosome is a tiny fraction (<=10%) of the diploid level (the 90th-percentile central het,
    a robust "definitely female" reference that survives a male-majority panel). When no sample is
    heterozygous in the interior everyone is female — so :func:`pseudohaploid` on an autosome simply
    splits every diploid, and an all-male diploid-encoded panel needs an explicit ``sex_map``.
    """
    pos = np.asarray(v.positions)
    L = float(v.sequence_length)
    central = (pos >= 0.25 * L) & (pos < 0.75 * L)          # deep non-PAR: males are homozygous here
    het_c = np.zeros(len(groups))
    for k, g in enumerate(groups):
        if len(g) < 2:
            continue
        both = central.copy()
        if v.missing is not None:
            both &= ~v.missing[:, g[0]] & ~v.missing[:, g[1]]
        n = int(both.sum())
        het_c[k] = int((_het_mask(v, g[0], g[1]) & central).sum()) / n if n else 0.0
    fem_ref = float(np.quantile(het_c, 0.9)) if np.any(het_c > 0) else 0.0
    thresh = 0.1 * fem_ref
    return {bid: ("M" if (len(groups[k]) < 2 or (fem_ref > 0 and het_c[k] <= thresh)) else "F")
            for k, bid in enumerate(base_ids)}


def _infer_par(v, groups, sex, base_ids):
    """Infer PAR as the chromosome-end interval(s) spanning male heterozygous sites.

    Males are heterozygous only in the pseudo-autosomal regions (they are diploid there); the large
    central gap in the male-het positions is the (haploid) non-PAR, and the ends around it are PAR.
    Returns a list of ``(start, end)`` intervals (empty if no diploid-encoded males show any het).
    """
    L = float(v.sequence_length)
    het = np.zeros(v.genotypes.shape[0], bool)
    for k, g in enumerate(groups):
        if sex[base_ids[k]] == "M" and len(g) >= 2:
            het |= _het_mask(v, g[0], g[1])
    het_pos = np.sort(np.asarray(v.positions)[het])
    if het_pos.size == 0:
        return []
    bounds = np.concatenate([[0.0], het_pos, [L]])
    i = int(np.argmax(np.diff(bounds)))              # largest gap = the non-PAR interior
    lo, hi = float(bounds[i]), float(bounds[i + 1])
    return [seg for seg in ((0.0, lo), (hi, L)) if seg[1] > seg[0]]


def _positions_in_par(positions, par):
    """Boolean mask over ``positions`` for membership in any PAR ``(start, end)`` interval."""
    pos = np.asarray(positions)
    mask = np.zeros(pos.shape[0], bool)
    for lo, hi in par:
        mask |= (pos >= lo) & (pos < hi)
    return mask


def _resolve_sex_and_par(v, sex_map):
    """Return ``(groups, base_ids, sex, par)`` — sex from ``sex_map`` (ids absent are inferred) or
    all-inferred, and PAR always inferred from the resulting males' heterozygosity."""
    groups, base_ids = _group_columns(v)
    smap = _normalize_sex_map(sex_map)
    if smap is None:
        sex = _infer_sex(v, groups, base_ids)
    else:
        inferred = _infer_sex(v, groups, base_ids)
        sex = {bid: smap.get(bid, inferred[bid]) for bid in base_ids}
    return groups, base_ids, sex, _infer_par(v, groups, sex, base_ids)


def _combine_missing(src_missing, col, override):
    """Missing mask for an output column: source column ``col``'s mask OR-ed with an ``override``."""
    base = None if src_missing is None else src_missing[:, col]
    if override is None:
        return base
    return override if base is None else (override | base)


def _expand_haploid(v, *, sex_map, keep_par, split_females):
    """Emit one haploid column per real haplotype, collapsing hemizygous (non-PAR) males.

    Shared by :func:`pseudohaploid` (``split_females=True`` — every haplotype its own individual) and
    the chrX-aware read in :func:`variants_from_vcf` (``split_females=False`` — females stay diploid,
    only males are collapsed). Males keep one X (plus, if ``keep_par``, a PAR-only second copy that is
    missing outside PAR); females keep both copies.
    """
    groups, base_ids, sex, par = _resolve_sex_and_par(v, sex_map)
    par_mask = _positions_in_par(v.positions, par)
    cols, miss_cols, names, sidx = [], [], [], []
    for k, g in enumerate(groups):
        bid = base_ids[k]
        if sex[bid] == "F" or len(g) < 2:                  # female / already-haploid: keep every copy
            emitted = [(c, None, f"{bid}_{h + 1}" if len(g) > 1 else bid) for h, c in enumerate(g)]
        else:                                              # male: one X (+ optional PAR second copy)
            emitted = [(g[0], None, f"{bid}_1" if keep_par else bid)]
            if keep_par:
                emitted.append((g[1], ~par_mask, f"{bid}_2"))
        indiv = len(names) if split_females else (max(sidx) + 1 if sidx else 0)
        for h, (c, override, nm) in enumerate(emitted):
            cols.append(v.genotypes[:, c])
            miss_cols.append(_combine_missing(v.missing, c, override))
            names.append(nm)
            sidx.append(len(names) - 1 if split_females else indiv)

    genotypes = np.stack(cols, axis=1).astype(np.int8)
    S = genotypes.shape[0]
    missing = None
    if any(m is not None for m in miss_cols):
        missing = np.stack([np.zeros(S, bool) if m is None else m for m in miss_cols], axis=1)
    return Variants(positions=v.positions, genotypes=genotypes, alleles=v.alleles,
                    sequence_length=v.sequence_length, sample_names=names, ploidy=1, missing=missing,
                    sample_index=None if split_females else np.asarray(sidx))


def pseudohaploid(source, *, sex_map=None, keep_par=False):
    r"""Expand a source to **one haploid sample per real haplotype** — the chrX-safe input.

    tskit / tsinfer / SINGER model haplotypes, not diploid individuals, so the correct way to handle
    the sex chromosome (or simply to treat every chromosome as haploid) is to give them one haploid
    sample per real haplotype. Females contribute both X copies; a **male** carries a single X across
    the hemizygous region, so he must **not** be encoded as a homozygous diploid (that would double
    him into two identical lineages and bias the ARG).

    Females -> two haploid samples (``id_1``, ``id_2``). Males -> **one** haploid sample (``id``),
    dropping the redundant second copy — the default, which is safe for SINGER (whose VCF reader has
    no missing-data concept: it reads only each genotype's first character, so a masked call would
    become reference). Pseudo-autosomal regions (PAR), where males *are* diploid, are still detected
    and reported. With ``keep_par=True`` a male keeps a second haplotype (``id_2``) that is real
    inside PAR and **missing** outside it — correct for :func:`tspaint.io.tsinfer` (which supports
    missing data) but **not** for SINGER.

    Parameters
    ----------
    source : Variants, tskit.TreeSequence, or str
        Genotypes (as in :func:`resolve_variants`). A diploid VCF/Zarr is the usual input.
    sex_map : dict or pandas.DataFrame, optional
        Per-sample sex — a ``{id: 'F'/'M'}`` mapping or a 2-column ``(id, sex)`` DataFrame. Ids not
        listed (and the whole panel when ``sex_map`` is ``None``) have their sex **inferred** from
        X-heterozygosity (males are ~homozygous). The PAR borders are always inferred from where the
        males are heterozygous (the chromosome ends).
    keep_par : bool, optional
        Keep each male's second haplotype inside PAR (missing elsewhere). Default ``False`` (drop it
        everywhere — one haploid sample per male, a single clean SINGER run).

    Returns
    -------
    Variants
        A haploid (``ploidy=1``) matrix, one named column per haplotype, with :attr:`Variants.missing`
        set for any masked (male non-PAR second) calls.
    """
    v = (source if isinstance(source, Variants)
         else _variants_from_ts(source) if source_kind(source) == "ts" else resolve_variants(source))
    return _expand_haploid(v, sex_map=sex_map, keep_par=keep_par, split_females=True)


_UNSET = object()   # sentinel: an individual not assigned to any comparison group


def _pair_diversity(G, called, indiv_cols):
    """Pairwise-diversity sufficient stats over between-individual pairs *within one group*.

    ``indiv_cols`` is a list of per-individual haplotype-column lists, all in the same comparison
    group. Returns ``(cross_diff, cross_cocalled, n_pairs)``: differences at co-called sites and
    co-called variant-site pairs, both summed over haplotype pairs from *different* individuals in
    the group, plus the number of such haplotype pairs. (With one group of every individual this is
    exactly the all-pairs computation.)
    """
    cols = [c for g in indiv_cols for c in g]
    if len(cols) < 2:
        return 0, 0, 0
    Gg, cg = G[:, cols], called[:, cols]
    n1 = ((Gg == 1) & cg).sum(1).astype(np.int64)
    n0 = ((Gg == 0) & cg).sum(1).astype(np.int64)
    nc = n0 + n1
    all_diff = n0 * n1                                # differing called pairs (all pairs in group)
    all_pairs = nc * (nc - 1) // 2                    # co-called pairs (all pairs in group)

    within_diff = np.zeros(G.shape[0], np.int64)
    within_pairs = np.zeros(G.shape[0], np.int64)
    n_within = 0
    for g in indiv_cols:                             # drop within-individual (homolog) pairs
        if len(g) < 2:
            continue
        cgi = called[:, g]
        c1 = ((G[:, g] == 1) & cgi).sum(1).astype(np.int64)
        c0 = ((G[:, g] == 0) & cgi).sum(1).astype(np.int64)
        cc = c0 + c1
        within_diff += c0 * c1
        within_pairs += cc * (cc - 1) // 2
        n_within += len(g) * (len(g) - 1) // 2
    cross_diff = int((all_diff - within_diff).sum())
    cross_cocalled = int((all_pairs - within_pairs).sum())
    H = len(cols)
    return cross_diff, cross_cocalled, H * (H - 1) // 2 - n_within


def _group_assignment(groups, v, indiv_cols, base_ids):
    """Group value per individual (``_UNSET`` when uncovered), for :func:`estimate_ne`'s ``groups``.

    ``groups`` is a mapping keyed by base sample id / integer individual index (e.g. the painting
    ``labels``), or an array-like of one value per individual (or per haplotype). Individuals with
    no entry stay ``_UNSET`` so the caller drops them from every pair.
    """
    n = len(indiv_cols)
    if isinstance(groups, dict):
        out = []
        for k in range(n):
            if base_ids[k] in groups:
                out.append(groups[base_ids[k]])
            elif k in groups:                        # integer individual-index key
                out.append(groups[k])
            else:
                out.append(_UNSET)
        return out
    arr = list(groups)
    if len(arr) == v.num_haplotypes and v.num_haplotypes != n:   # per-haplotype -> per-individual
        arr = [arr[g[0]] for g in indiv_cols]
    if len(arr) != n:
        raise ValueError(f"groups has {len(arr)} entries != {n} individuals in the panel")
    return arr


def estimate_ne(source, mutation_rate, groups=None, exclude=None):
    r"""Estimate the (diploid) effective population size from between-individual diversity.

    Uses Tajima's pairwise estimator of :math:`\theta = 4 N_e \mu`: nucleotide diversity
    :math:`\pi` — the mean number of sequence differences per base pair between two haplotypes —
    computed over haplotype pairs drawn from **different individuals** (so it measures population
    diversity, not within-individual heterozygosity), then :math:`N_e = \pi / (4\mu)`.

    Only sites **called in both** haplotypes of a pair contribute (missing calls are excluded), and
    the per-pair denominator is the callable length in base pairs: the analysed region
    (:attr:`Variants.sequence_length`) minus the sites known to be uncalled in that pair. Sites not
    present in the variant table are treated as invariant and callable, so the estimate is correct
    whether the input is a variant-only or an all-sites VCF. Individuals are the consecutive
    ``ploidy``-column blocks of :attr:`Variants.sample_names` (as the readers lay them out).

    For a structured / admixed panel, all-pairs :math:`\pi` includes cross-population comparisons, so
    it reflects the whole sample's deep between-population coalescent depth. **This all-pairs default
    (``groups=None``) is what a SINGER prior needs** (:func:`tspaint.io.singer`): SINGER calibrates
    ``4·Ne·μ ≈ π`` over the whole sample and must keep those deep coalescences on-scale, matching its
    own ``singer_master`` auto-Ne. Passing ``groups`` (e.g. the painting ``labels``) instead compares
    **only pairs of individuals in the same group** — a *smaller, within-population* :math:`N_e` that
    will **under-calibrate** the SINGER prior on a structured sample; reach for it only when you
    specifically want the within-population value. ``exclude`` (drop admixed / mislabelled individuals)
    is the appropriate refinement when the estimate feeds SINGER.

    Parameters
    ----------
    source : Variants, tskit.TreeSequence, or str
        Genotypes — a :class:`Variants`, a tree sequence, or a VCF / VCF-Zarr path (resolved as in
        :func:`resolve_variants`). Missing calls are honoured when the source records them.
    mutation_rate : float
        Per-base, per-generation mutation rate :math:`\mu`.
    groups : mapping or array-like, optional
        Restrict the comparison to pairs of individuals **in the same group**. A mapping is keyed by
        base sample id (as in :attr:`Variants.sample_names`) or integer individual index — the
        painting ``labels`` dict works directly; an array-like gives one value per individual (or per
        haplotype). Individuals with no entry (absent from the mapping, or ``None`` / NaN) are dropped
        from every pair. ``None`` (default) puts everyone in one group — the all-pairs behaviour.
    exclude : iterable, optional
        Individuals to leave out of the estimate entirely — by base sample id (as in
        :attr:`Variants.sample_names`) or integer individual index. Use it to drop **soft / suspect
        references** (admixed or mislabelled) whose inflated diversity would bias :math:`N_e`, so the
        SINGER prior is set from the clean panel only.

    Returns
    -------
    float
        The estimated diploid :math:`N_e = \pi / (4\mu)`.

    Raises
    ------
    ValueError
        If there are fewer than two haplotypes from distinct individuals, or no callable variation
        from which to estimate :math:`\pi`.
    """
    if isinstance(source, Variants):
        v = source
    elif source_kind(source) == "ts":
        v = _variants_from_ts(source)
    else:
        v = resolve_variants(source)

    G = np.asarray(v.genotypes)                       # (S, H) 0/1
    S, H = G.shape
    if H < 2:
        raise ValueError("need >= 2 haplotypes to estimate Ne")
    called = np.ones((S, H), bool) if v.missing is None else ~np.asarray(v.missing, bool)

    # Individuals (mixed-ploidy safe via Variants.sample_index), partitioned into comparison groups:
    # only haplotype pairs from *different* individuals *within the same group* are compared, so a
    # structured panel can restrict Ne to within-reference pairs. groups=None -> one all-individual
    # group == the historical all-pairs estimate.
    indiv_cols, base_ids = _group_columns(v)
    excluded = set()
    if exclude:                                              # drop soft/suspect refs (by id or index)
        ex = set(exclude)
        excluded = {k for k in range(len(indiv_cols)) if base_ids[k] in ex or k in ex}
    if groups is None:
        buckets = [[indiv_cols[k] for k in range(len(indiv_cols)) if k not in excluded]]
    else:
        by_val = {}
        for k, val in enumerate(_group_assignment(groups, v, indiv_cols, base_ids)):
            if k in excluded:
                continue                                     # soft/suspect ref -> excluded
            if val is _UNSET or val is None or (isinstance(val, float) and val != val):
                continue                                     # uncovered / NaN -> excluded
            by_val.setdefault(val, []).append(indiv_cols[k])
        buckets = list(by_val.values())
        if not buckets:
            raise ValueError("groups matched no individuals in the panel (check the keys)")

    cross_diff = cross_cocalled = n_pairs = 0
    for indiv_group in buckets:
        cd, cc, npr = _pair_diversity(G, called, indiv_group)
        cross_diff += cd
        cross_cocalled += cc
        n_pairs += npr
    if n_pairs <= 0:
        raise ValueError("need haplotypes from >= 2 individuals to estimate Ne")
    if cross_cocalled <= 0 or cross_diff <= 0:
        raise ValueError("no callable variation to estimate Ne from (pass Ne explicitly)")

    L = float(v.sequence_length)
    # pi per base pair = (differences per co-called variant-site pair) x (variant density S / L).
    # Normalising by co-called pairs makes missing data cancel between numerator and denominator
    # (only sites known in both haplotypes count); x S/L converts per-variant-site to per-bp, so
    # the estimate is correct for a variant-only *or* an all-sites table.
    pi = (cross_diff / cross_cocalled) * (S / L)
    return pi / (4.0 * float(mutation_rate))


def subset_data(source, *, start=None, end=None, samples=None):
    """Normalise a genotype ``source`` and restrict it to a region and/or a sub-panel.

    A convenience wrapper over :func:`resolve_variants` that turns any
    :func:`~tspaint.io.singer`-style ``source`` — a :class:`tskit.TreeSequence`, a **VCF**, a
    **VCF Zarr** store, or a :class:`Variants` — into a :class:`Variants` keeping only the sites
    in ``[start, end)`` and the selected haplotype columns. This is the boilerplate you would
    otherwise write before handing a slice of a chromosome (or a handful of samples) to
    :func:`~tspaint.io.singer`, :func:`~tspaint.paint`, etc.

    Parameters
    ----------
    source : tskit.TreeSequence, str, or Variants
        Genotypes, resolved exactly like :func:`~tspaint.io.singer`'s ``source``. A
        ``tsinfer.VariantData`` is **not** a valid source — pass the underlying ``.vcz`` store
        path (the same one you gave ``VariantData``) instead.
    start, end : float, optional
        Keep sites with ``start <= position < end``. ``start`` defaults to 0 and ``end`` to the
        source's sequence length. Positions are **not** shifted — genomic coordinates are
        preserved and the returned :attr:`Variants.sequence_length` is ``end``.
    samples : sequence or slice, optional
        Haplotype columns to keep (default ``None`` = all). May be integer column indices, a
        :class:`slice`, a boolean mask of length ``num_haplotypes``, or names matched against
        :attr:`Variants.sample_names` — a name selects an exact match **or** a whole
        diploid/polyploid sample by its base name (e.g. ``"NA12878"`` selects the
        ``"NA12878_0"`` / ``"NA12878_1"`` columns).

    Returns
    -------
    Variants
        The region- and sample-restricted matrix, ready for :func:`~tspaint.io.singer` / painting.

    Examples
    --------
    >>> from tspaint.io import singer, subset_data
    >>> region = subset_data("chr20.vcz", start=0, end=2_000_000, samples=range(20))
    >>> tss = singer(region, _Ne=1e4, _m=1.25e-8, _r=1e-8)
    """
    v = _variants_from_ts(source) if source_kind(source) == "ts" else resolve_variants(source)

    lo = 0.0 if start is None else float(start)
    hi = float(v.sequence_length) if end is None else float(end)
    site_idx = np.nonzero((v.positions >= lo) & (v.positions < hi))[0]
    cols = _resolve_sample_columns(v, samples)

    genotypes = v.genotypes[site_idx]
    missing = v.missing[site_idx] if v.missing is not None else None
    names = v.sample_names
    sample_index = v.sample_index
    if cols is not None:
        genotypes = genotypes[:, cols]
        missing = missing[:, cols] if missing is not None else None
        names = [names[c] for c in cols] if names is not None else None
        if sample_index is not None:                       # keep grouping; renumber to 0..k-1
            sample_index = np.unique(np.asarray(sample_index)[cols], return_inverse=True)[1]
    return Variants(positions=v.positions[site_idx], genotypes=genotypes,
                    alleles=[v.alleles[i] for i in site_idx], sequence_length=hi,
                    sample_names=names, ploidy=v.ploidy, missing=missing, sample_index=sample_index)


def _variants_from_ts(ts):
    """A biallelic :class:`Variants` view of a tree sequence (one column per sample node)."""
    G = ts.genotype_matrix()
    alleles = [(s.ancestral_state, s.mutations[0].derived_state if s.mutations else ".")
               for s in ts.sites()]
    missing = G < 0                                    # tskit marks missing calls as -1
    return Variants(positions=np.asarray(ts.tables.sites.position),
                    genotypes=np.where(G > 0, 1, 0).astype(np.int8), alleles=alleles,
                    sequence_length=float(ts.sequence_length),
                    missing=missing if missing.any() else None)


def _resolve_sample_columns(v, samples):
    """Resolve the ``samples`` selector to an array of haplotype-column indices (or ``None``)."""
    if samples is None:
        return None
    H = v.num_haplotypes
    if isinstance(samples, slice):
        return np.arange(H)[samples]
    arr = np.atleast_1d(np.asarray(samples))
    if arr.dtype == bool:
        if arr.shape != (H,):
            raise ValueError(f"boolean sample mask must have length {H}, got {arr.shape[0]}")
        return np.nonzero(arr)[0]
    if arr.dtype.kind in ("U", "S", "O"):                     # select by name
        names = v.sample_names
        if names is None:
            raise ValueError("cannot select samples by name: source has no sample_names "
                             "(select by integer haplotype-column index instead)")
        cols = []
        for s in (str(x) for x in arr.tolist()):
            hit = [i for i, nm in enumerate(names) if nm == s or nm.startswith(s + "_")]
            if not hit:
                raise ValueError(f"sample {s!r} not found in sample_names")
            cols.extend(hit)
        return np.asarray(cols, dtype=int)
    return arr.astype(int)                                    # integer column indices


def variants_from_vcf(path, sex_map=None):
    """Read a biallelic VCF into :class:`Variants` (minimal pure-Python parser; CLAUDE.md §5).

    Each sample's ``GT`` contributes as many haplotype columns as it has alleles (phased or unphased —
    alleles are taken positionally); ploidy may differ **between** samples (e.g. a chrX VCF with
    diploid females and haploid males), and per-haplotype sample names are retained regardless via
    :attr:`Variants.sample_index`. Multiallelic records are skipped; ``.`` is read as REF but recorded
    in :attr:`Variants.missing`.

    Parameters
    ----------
    path : str
        Path to a ``.vcf`` or ``.vcf.gz`` file.
    sex_map : dict or pandas.DataFrame, optional
        If given, the VCF is treated as a sex chromosome and made hemizygosity-correct: males (from
        the map, or inferred for unlisted ids) are collapsed to one haplotype outside the PAR so they
        are not double-counted, while females stay diploid (via :func:`pseudohaploid`'s shared logic).
        **Without** ``sex_map`` the VCF is read faithfully as-encoded — no sex inference — so an
        autosome is never mis-collapsed; use :func:`pseudohaploid` to also split females to haploid.

    Returns
    -------
    Variants
    """
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    positions, genos, alleles, miss = [], [], [], []
    samples, sample_ploidy = None, None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            ref, alt = cols[3], cols[4]
            if "," in alt:                       # multiallelic — skipped in v1
                continue
            fmt = cols[8].split(":") if len(cols) > 8 else ["GT"]
            gi = fmt.index("GT") if "GT" in fmt else 0
            gts = [(cell.split(":")[gi] if cell else ".").replace("|", "/").split("/")
                   for cell in cols[9:]]
            if sample_ploidy is None:                          # per-sample allele count (from row 1)
                sample_ploidy = [len(g) for g in gts]
            hap, hap_miss = [], []
            for g in gts:
                for a in g:
                    hap.append(1 if a not in (".", "0") else 0)
                    hap_miss.append(a in (".", ""))            # missing call (stored as REF above)
            positions.append(int(cols[1]))
            genos.append(hap)
            miss.append(hap_miss)
            alleles.append((ref, alt))
    if not positions:
        raise ValueError(f"no biallelic GT records parsed from {path}")
    positions = np.asarray(positions, float)
    genotypes = np.asarray(genos, np.int8)
    missing = np.asarray(miss, bool)

    names, sample_index, ploidy = None, None, 1
    if samples is not None and sample_ploidy and sum(sample_ploidy) == genotypes.shape[1]:
        if len(set(sample_ploidy)) == 1:                       # uniform ploidy (the common case)
            ploidy = sample_ploidy[0]
            names = ([f"{s}_{k}" for s in samples for k in range(ploidy)] if ploidy > 1
                     else list(samples))
        else:                                                  # mixed ploidy -> keep names + grouping
            names, sample_index = [], []
            for si, (s, pl) in enumerate(zip(samples, sample_ploidy)):
                for k in range(pl):
                    names.append(f"{s}_{k}" if pl > 1 else s)
                    sample_index.append(si)
            sample_index = np.asarray(sample_index)
    v = Variants(positions=positions, genotypes=genotypes, alleles=alleles,
                 sequence_length=float(positions.max() + 1), sample_names=names, ploidy=ploidy,
                 missing=missing if missing.any() else None, sample_index=sample_index)
    if sex_map is not None:                                    # chrX-aware: collapse hemizygous males
        v = _expand_haploid(v, sex_map=sex_map, keep_par=False, split_females=False)
    return v


def variants_from_zarr(store):
    """Read a VCF Zarr (sgkit / ``vcf2zarr`` layout) into :class:`Variants` (CLAUDE.md §5).

    Reads the standard core arrays ``call_genotype`` ``(variants, samples, ploidy)``,
    ``variant_position`` and ``variant_allele``; the ``(samples, ploidy)`` calls are flattened to
    haplotype columns. Biallelic sites only.

    Parameters
    ----------
    store : str or zarr store / mapping
        A VCF Zarr store (e.g. produced by ``bio2zarr``'s ``vcf2zarr``).

    Returns
    -------
    Variants
    """
    import zarr
    root = zarr.open(store, mode="r")
    gt = np.asarray(root["call_genotype"])                       # (V, N, P)
    pos = np.asarray(root["variant_position"], float)            # (V,)
    allele = np.asarray(root["variant_allele"])                  # (V, A)
    V, N, P = gt.shape
    haps = gt.reshape(V, N * P)
    missing = haps < 0                                           # (V, H) missing calls (sgkit: -1)
    haps = np.where(haps > 0, 1, 0).astype(np.int8)              # collapse to biallelic, missing(-1)->0
    alleles = [(_as_str(allele[i, 0]), _as_str(allele[i, 1]) if allele.shape[1] > 1 else ".")
               for i in range(V)]
    try:
        names = [_as_str(s) for s in np.asarray(root["sample_id"])]
        names = [f"{s}_{k}" for s in names for k in range(P)] if P > 1 else names
    except KeyError:
        names = None
    return Variants(positions=pos, genotypes=haps, alleles=alleles,
                    sequence_length=float(pos.max() + 1), sample_names=names, ploidy=P,
                    missing=missing if missing.any() else None)


def _as_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def sample_names_from_zarr(store):
    """Read only the sample ids and ploidy from a VCF-Zarr store — **without** loading genotypes.

    The cheap identity read for the scalable :func:`variant_data_from_zarr` path: reads the small
    ``sample_id`` array and infers ploidy from ``call_genotype``'s shape metadata (no chunk I/O), so
    :func:`tspaint.io.tsinfer` can stamp whole-genome tree sequences with sample ids affordably.

    Parameters
    ----------
    store : str or zarr store / mapping
        A VCF Zarr store (e.g. produced by ``bio2zarr``'s ``vcf2zarr``).

    Returns
    -------
    (list[str] or None, int)
        Flattened per-haplotype names ``"<sample>_<k>"`` (or the base names for haploid), matching
        :attr:`Variants.sample_names`, and the ploidy. ``(None, 1)`` if no ``sample_id`` is present.
    """
    import zarr
    root = zarr.open(store, mode="r")
    ploidy = int(root["call_genotype"].shape[2]) if "call_genotype" in root else 1
    try:
        base = [_as_str(s) for s in np.asarray(root["sample_id"])]
    except KeyError:
        return None, ploidy
    names = [f"{s}_{k}" for s in base for k in range(ploidy)] if ploidy > 1 else base
    return names, ploidy


def variant_data_from_zarr(store):
    """Build a tsinfer ``VariantData`` from a VCF-Zarr store — **chunked / scalable** (CLAUDE.md §5).

    The preferred zarr path for :func:`tspaint.io.tsinfer`: tsinfer's native VCF-Zarr reader
    accesses the genotypes lazily (no in-memory genotype matrix), so it scales to whole-genome
    data with no new dependency. Uses the ``variant_ancestral_allele`` array if present, else the
    REF allele as the ancestral state.

    Parameters
    ----------
    store : str or zarr store / mapping
        A VCF Zarr store (e.g. from ``bio2zarr``'s ``vcf2zarr``).

    Returns
    -------
    tsinfer.VariantData
    """
    import tsinfer
    import zarr
    root = zarr.open(store, mode="r")
    if "variant_ancestral_allele" in root:
        ancestral = np.asarray(root["variant_ancestral_allele"]).astype(str)
    else:
        allele = np.asarray(root["variant_allele"])
        ancestral = np.array([_as_str(allele[i, 0]) for i in range(allele.shape[0])])
    return tsinfer.VariantData(store, ancestral_state=ancestral)


def write_vcz(ts, path):
    """Write a minimal **VCF-Zarr** store from a *mutated* tree sequence — a lightweight stand-in for
    ``bio2zarr``'s ``vcf2zarr`` output, so a simulation can be fed through the zarr front ends.

    Writes the core arrays that :func:`variants_from_zarr` / :func:`variant_data_from_zarr` /
    :func:`sample_names_from_zarr` read — ``call_genotype`` ``(variants, samples, ploidy=1)``,
    ``variant_position``, ``variant_allele``, ``variant_contig`` / ``contig_id``, ``sample_id`` and
    ``variant_ancestral_allele`` — so it round-trips through :func:`tspaint.io.tsinfer` /
    :func:`tspaint.io.singer`. Each sample **node** becomes one haploid sample ``"n<i>"``; biallelic
    ``0`` / ``1`` sites (e.g. from :func:`add_mutations`).

    Parameters
    ----------
    ts : tskit.TreeSequence
        A mutated tree sequence — must carry sites (overlay them with :func:`add_mutations` first).
    path : str
        Output ``.vcz`` store path.

    Returns
    -------
    str
        ``path`` (for chaining).
    """
    import zarr
    if ts.num_sites == 0:
        raise ValueError("write_vcz needs a mutated tree sequence; overlay sites with "
                         "add_mutations() first")
    G = ts.genotype_matrix()                       # (sites, samples), biallelic 0/1
    V, N = G.shape
    root = zarr.open(path, mode="w")

    def arr(nm, data, dims):
        root[nm] = data
        root[nm].attrs["_ARRAY_DIMENSIONS"] = dims

    arr("variant_position", np.asarray(ts.tables.sites.position).astype("i8"), ["variants"])
    arr("call_genotype", G[:, :, None].astype("i1"), ["variants", "samples", "ploidy"])
    arr("variant_allele", np.array([["0", "1"]] * V), ["variants", "alleles"])
    arr("variant_contig", np.zeros(V, "i4"), ["variants"])
    arr("contig_id", np.array(["1"]), ["contigs"])
    arr("sample_id", np.array([f"n{i}" for i in range(N)]), ["samples"])
    arr("variant_ancestral_allele", np.array(["0"] * V), ["variants"])
    return path


def to_sample_data(variants):
    """Build a tsinfer ``SampleData`` from :class:`Variants` (for :func:`tspaint.io.tsinfer`).

    Missing calls (:attr:`Variants.missing`) are passed to tsinfer as ``tskit.MISSING_DATA`` (``-1``)
    rather than reference, so a :func:`pseudohaploid` ``keep_par=True`` male's absent non-PAR second
    copy is treated as missing (tsinfer supports it) instead of a spurious reference lineage.

    Parameters
    ----------
    variants : Variants
        The variant matrix to convert (one site per :attr:`Variants.positions` entry).

    Returns
    -------
    tsinfer.SampleData
        A finalised ``SampleData`` with one site per variant, missing calls encoded as
        ``tskit.MISSING_DATA``.
    """
    import tsinfer
    miss = variants.missing
    with tsinfer.SampleData(sequence_length=variants.sequence_length) as sd:
        for i in range(variants.num_sites):
            g = variants.genotypes[i]
            if miss is not None and miss[i].any():
                g = np.where(miss[i], tskit.MISSING_DATA, g)
            sd.add_site(variants.positions[i], g, alleles=list(variants.alleles[i]))
    return sd


def write_haploid_vcf(variants, path):
    """Write :class:`Variants` as a haploid VCF (one column per haplotype) for SINGER.

    Parameters
    ----------
    variants : Variants
        The variant matrix to write. Each haplotype column becomes a VCF sample column, named from
        :attr:`Variants.sample_names` (falling back to ``"h0"``, ``"h1"``, ... when absent).
    path : str
        Output VCF path to write (an existing file is overwritten).

    Notes
    -----
    Site positions are floored to distinct, strictly increasing 1-based integers; genotypes are
    written haploid (a single ``0`` / ``1`` per column). Returns nothing — contrast
    :func:`write_vcz`, which returns its ``path``.
    """
    H = variants.num_haplotypes
    names = variants.sample_names or [f"h{j}" for j in range(H)]
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##contig=<ID=1,length=%d>\n' % int(variants.sequence_length))
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n")
        last = 0
        for i in range(variants.num_sites):
            p = max(last + 1, int(np.floor(variants.positions[i])))   # distinct 1-based positions
            last = p
            ref, alt = variants.alleles[i]
            gts = "\t".join(str(int(g)) for g in variants.genotypes[i])
            f.write(f"1\t{p}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{gts}\n")
