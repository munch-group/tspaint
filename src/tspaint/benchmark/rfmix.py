"""RFMix v2 benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9, §10).

RFMix (Maples et al., 2013, *Am. J. Hum. Genet.* 93, 278-288): random-forest windows + a
conditional-random-field smoother, the field-standard genotype-native LAI incumbent. This runner
is the VCF-native sibling of :func:`tspaint.io_rfmix.rfmix_paint` (which paints a tree sequence):
it takes query/reference VCFs directly, writes RFMix's inputs, runs the ``rfmix`` binary (isolated
``compare`` pixi env; path via ``TSPAINT_RFMIX``), and parses the ``.fb.tsv`` **posteriors** — the
calibrated soft output, a fair soft-vs-soft comparison against tspaint.
"""
from __future__ import annotations

import os
import tempfile

from . import _common as C
from ._msp import parse_fb

__all__ = ["rfmix"]


def rfmix(query_vcf, ref_vcf=None, *, sample_map, genetic_map=None, chromosome=None,
          recomb_rate=1e-8, generations=8, out=None, workdir=None, rfmix_bin=None,
          extra_args=None, log=None):
    """Run RFMix on a query VCF and return per-query-haplotype posterior Segment tracks.

    Parameters
    ----------
    query_vcf : str
        Phased diploid query VCF (or a combined VCF if ``ref_vcf`` is omitted).
    ref_vcf : str, optional
        Separate phased diploid reference VCF; omit for a combined query+reference VCF.
    sample_map : str
        ``<ref-sample>\\t<ancestry>`` map (:func:`tspaint.benchmark.read_sample_map`).
    genetic_map : str, optional
        RFMix genetic map (3 columns ``chrom pos cM``); a uniform map at ``recomb_rate`` is
        generated when omitted.
    chromosome : str, optional
        Contig passed to ``--chromosome`` (default: the VCF's contig).
    recomb_rate : float, optional
        Per-base recombination rate for the generated genetic map (default ``1e-8``).
    generations : float, optional
        Generations since admixture (RFMix's ``-G``; default 8).
    out : str, optional
        If given, also write the painting to this ``.npz`` (tspaint-painting format).
    workdir : str, optional
        Working directory for the inputs/outputs (default: a fresh tempdir).
    rfmix_bin : str, optional
        Path to the ``rfmix`` binary (default ``TSPAINT_RFMIX`` / the ``compare`` env).
    extra_args : iterable[str], optional
        Extra arguments appended to the RFMix command.
    log : callable, optional
        Called with progress strings (e.g. ``print``).

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, the ``.fb.tsv`` soft painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_rfmix_")
    panel, qv, rv, sm = C.setup_inputs(query_vcf, ref_vcf, sample_map, workdir)
    contig = str(chromosome) if chromosome is not None else panel.contig

    gmap = genetic_map
    if gmap is None:
        gmap = os.path.join(workdir, "genetic_map.tsv")
        C.write_genetic_map(gmap, panel, recomb_rate, fmt="plink")

    if not C.tool_available("rfmix", bin_override=rfmix_bin):
        raise FileNotFoundError(
            f"rfmix binary not found at {rfmix_bin or C.RFMIX_BIN}; set TSPAINT_RFMIX or install "
            "the `compare` pixi env (pixi install -e compare)")
    out_base = os.path.join(workdir, "rfmix_out")
    args = ["-f", qv, "-r", rv, "-m", sm, "-g", gmap, "-o", out_base,
            "--chromosome=" + contig, "-G", str(int(round(generations)))]
    if extra_args:
        args += list(extra_args)
    C.run_tool("rfmix", args, cwd=workdir, log=log, bin_override=rfmix_bin)

    tracks = parse_fb(out_base + ".fb.tsv", C.parse_inds(panel), panel.K, panel.sequence_length)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"rfmix: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
