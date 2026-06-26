"""Unified genotype ingestion for the inference front ends (CLAUDE.md §5).

The inference front ends (:func:`tspaint.io.tsinfer`, :func:`tspaint.io.singer`) accept the
same ``source``: a :class:`tskit.TreeSequence`, a **VCF Zarr** store (the sgkit / ``vcf2zarr``
layout), or a **VCF** file. The native ``ts`` path is handled by each tool directly; this module
normalises the other two into a tool-agnostic :class:`Variants` and builds whatever each tool
needs from it (tsinfer ``SampleData`` or a haploid VCF).

Scope (v1, dependency-light — only ``zarr``, which tsinfer already needs): **biallelic** sites,
a single contig, the ``GT`` field; ``.`` (missing) is read as the reference allele. Richer VCF /
Zarr handling is a follow-up — convert with ``bio2zarr`` / ``sgkit`` upstream if you need it.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

import numpy as np
import tskit

__all__ = ["Variants", "source_kind", "resolve_variants", "variants_from_vcf",
           "variants_from_zarr", "to_sample_data", "write_haploid_vcf"]


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
        Optional haplotype/sample names (length ``H``).
    """
    positions: np.ndarray
    genotypes: np.ndarray
    alleles: list
    sequence_length: float
    sample_names: list = None

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
    """Normalise a non-``ts`` ``source`` (VCF or VCF Zarr) into :class:`Variants`."""
    kind = source_kind(source)
    if kind == "vcf":
        return variants_from_vcf(source)
    if kind == "zarr":
        return variants_from_zarr(source)
    raise ValueError(f"resolve_variants expects a VCF or zarr source, not a {kind}")


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
    names = None
    if samples is not None and genotypes.shape[1] % len(samples) == 0:
        ploidy = genotypes.shape[1] // len(samples)
        names = [f"{s}_{k}" for s in samples for k in range(ploidy)] if ploidy > 1 else list(samples)
    return Variants(positions=positions, genotypes=genotypes, alleles=alleles,
                    sequence_length=float(positions.max() + 1), sample_names=names)


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
                    sequence_length=float(pos.max() + 1), sample_names=names)


def _as_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


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
