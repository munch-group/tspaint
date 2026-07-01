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
from dataclasses import dataclass

import numpy as np
import tskit

__all__ = ["Variants", "source_kind", "resolve_variants", "subset_data", "variants_from_vcf",
           "variants_from_zarr", "variant_data_from_zarr", "to_sample_data", "write_haploid_vcf",
           "sample_names_from_zarr"]


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
    """
    positions: np.ndarray
    genotypes: np.ndarray
    alleles: list
    sequence_length: float
    sample_names: list = None
    ploidy: int = 1

    @property
    def num_sites(self):
        return int(self.positions.shape[0])

    @property
    def num_haplotypes(self):
        return int(self.genotypes.shape[1])


def source_kind(source):
    """Classify a front-end ``source`` as ``"ts"``, ``"zarr"`` or ``"vcf"``.

    A :class:`tskit.TreeSequence` is ``"ts"``; a path ending ``.vcf`` / ``.vcf.gz`` is ``"vcf"``;
    anything else path-like (a ``.zarr`` store / directory / mapping) is ``"zarr"``.
    """
    if isinstance(source, tskit.TreeSequence):
        return "ts"
    s = str(source).lower()
    if s.endswith(".vcf") or s.endswith(".vcf.gz") or s.endswith(".vcf.bgz"):
        return "vcf"
    return "zarr"


def resolve_variants(source):
    """Normalise a ``source`` into :class:`Variants` (a :class:`Variants` is returned as-is)."""
    if isinstance(source, Variants):
        return source
    kind = source_kind(source)
    if kind == "vcf":
        return variants_from_vcf(source)
    if kind == "zarr":
        return variants_from_zarr(source)
    raise ValueError(f"resolve_variants expects a VCF or zarr source, not a {kind}")


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
    >>> tss = singer(region, Ne=1e4, mutation_rate=1.25e-8, recombination_rate=1e-8)
    """
    v = _variants_from_ts(source) if source_kind(source) == "ts" else resolve_variants(source)

    lo = 0.0 if start is None else float(start)
    hi = float(v.sequence_length) if end is None else float(end)
    site_idx = np.nonzero((v.positions >= lo) & (v.positions < hi))[0]
    cols = _resolve_sample_columns(v, samples)

    genotypes = v.genotypes[site_idx]
    names = v.sample_names
    if cols is not None:
        genotypes = genotypes[:, cols]
        names = [names[c] for c in cols] if names is not None else None
    return Variants(positions=v.positions[site_idx], genotypes=genotypes,
                    alleles=[v.alleles[i] for i in site_idx], sequence_length=hi,
                    sample_names=names, ploidy=v.ploidy)


def _variants_from_ts(ts):
    """A biallelic :class:`Variants` view of a tree sequence (one column per sample node)."""
    G = ts.genotype_matrix()
    alleles = [(s.ancestral_state, s.mutations[0].derived_state if s.mutations else ".")
               for s in ts.sites()]
    return Variants(positions=np.asarray(ts.tables.sites.position),
                    genotypes=np.where(G > 0, 1, 0).astype(np.int8), alleles=alleles,
                    sequence_length=float(ts.sequence_length))


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


def variants_from_vcf(path):
    """Read a biallelic VCF into :class:`Variants` (minimal pure-Python parser; CLAUDE.md §5).

    Each sample's ``GT`` contributes ``ploidy`` haplotype columns (phased or unphased — alleles
    are taken positionally). Multiallelic records are skipped; ``.`` is read as REF.

    Parameters
    ----------
    path : str
        Path to a ``.vcf`` or ``.vcf.gz`` file.

    Returns
    -------
    Variants
    """
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    positions, genos, alleles = [], [], []
    samples = None
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
            hap = []
            for cell in cols[9:]:
                gt = cell.split(":")[gi] if cell else "."
                for a in gt.replace("|", "/").split("/"):
                    hap.append(1 if a not in (".", "0") else 0)
            positions.append(int(cols[1]))
            genos.append(hap)
            alleles.append((ref, alt))
    if not positions:
        raise ValueError(f"no biallelic GT records parsed from {path}")
    positions = np.asarray(positions, float)
    genotypes = np.asarray(genos, np.int8)
    names, ploidy = None, 1
    if samples is not None and genotypes.shape[1] % len(samples) == 0:
        ploidy = genotypes.shape[1] // len(samples)
        names = [f"{s}_{k}" for s in samples for k in range(ploidy)] if ploidy > 1 else list(samples)
    return Variants(positions=positions, genotypes=genotypes, alleles=alleles,
                    sequence_length=float(positions.max() + 1), sample_names=names, ploidy=ploidy)


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
    haps = np.where(haps > 0, 1, 0).astype(np.int8)              # collapse to biallelic, missing(-1)->0
    alleles = [(_as_str(allele[i, 0]), _as_str(allele[i, 1]) if allele.shape[1] > 1 else ".")
               for i in range(V)]
    try:
        names = [_as_str(s) for s in np.asarray(root["sample_id"])]
        names = [f"{s}_{k}" for s in names for k in range(P)] if P > 1 else names
    except KeyError:
        names = None
    return Variants(positions=pos, genotypes=haps, alleles=alleles,
                    sequence_length=float(pos.max() + 1), sample_names=names, ploidy=P)


def _as_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def sample_names_from_zarr(store):
    """Read only the sample ids and ploidy from a VCF-Zarr store — **without** loading genotypes.

    The cheap identity read for the scalable :func:`variant_data_from_zarr` path: reads the small
    ``sample_id`` array and infers ploidy from ``call_genotype``'s shape metadata (no chunk I/O), so
    :func:`tspaint.io.tsinfer` can stamp whole-genome tree sequences with sample ids affordably.

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


def to_sample_data(variants):
    """Build a tsinfer ``SampleData`` from :class:`Variants` (for :func:`tspaint.io.tsinfer`)."""
    import tsinfer
    with tsinfer.SampleData(sequence_length=variants.sequence_length) as sd:
        for i in range(variants.num_sites):
            sd.add_site(variants.positions[i], variants.genotypes[i],
                        alleles=list(variants.alleles[i]))
    return sd


def write_haploid_vcf(variants, path):
    """Write :class:`Variants` as a haploid VCF (one column per haplotype) for SINGER."""
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
