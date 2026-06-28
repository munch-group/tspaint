"""Gnomix benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9, §10).

Gnomix (Hilmarsson et al., 2021): a gradient-boosted base + smoother LAI method. By default this
runner uses gnomix's **train-from-reference** mode (it simulates admixture from the supplied
reference panel and trains a model in-run), so it works on arbitrary marker sets; pass ``model=``
a pretrained ``.pkl`` for inference-only mode. We parse gnomix's ``query_results.fb`` **posteriors**
(columns ``<sample>:::hap<1|2>:::<pop>``, the same shape as RFMix's ``.fb``).

Gnomix runs in its own pixi env: ``pixi run --manifest-path ~/gnomix python ~/gnomix/gnomix.py …``;
relocate with ``TSPAINT_GNOMIX_DIR`` or replace the launcher with ``TSPAINT_GNOMIX_CMD``.
"""
from __future__ import annotations

import os
import tempfile

from . import _common as C
from ._msp import parse_fb

__all__ = ["gnomix"]


def gnomix(query_vcf, ref_vcf=None, *, sample_map, genetic_map=None, chromosome=None,
           recomb_rate=1e-8, model=None, phase=False, out=None, workdir=None,
           extra_args=None, log=None):
    """Run Gnomix on a query VCF and return per-query-haplotype posterior Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, genetic_map, chromosome, recomb_rate, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix` (``genetic_map`` here is 3-column
        ``chrom pos cM``; a uniform one is generated when omitted — unused in pretrained mode).
    model : str, optional
        Path to a pretrained gnomix ``.pkl`` model. If given, gnomix runs inference-only
        (``reference``/``sample_map``/``genetic_map`` are not used); otherwise it trains from
        the reference panel.
    phase : bool, optional
        gnomix's ``<phase>`` flag (Gnofix phasing-error correction). Default ``False`` — inputs
        are taken as already phased.
    extra_args : iterable[str], optional
        Extra positional arguments appended after the standard gnomix arguments (e.g. a config).

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, the ``query_results.fb`` soft painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_gnomix_")
    panel, qv, rv, sm = C.setup_inputs(query_vcf, ref_vcf, sample_map, workdir,
                                       sample_map_header="#Sample\tPanel")
    contig = str(chromosome) if chromosome is not None else panel.contig
    out_dir = os.path.join(workdir, "gnomix_out")

    if model is not None:
        C.require(model, "pretrained gnomix model .pkl not found")
        args = [qv, out_dir, contig, str(bool(phase)), model]
    else:
        gmap = genetic_map
        if gmap is None:
            gmap = os.path.join(workdir, "genetic_map.tsv")
            C.write_genetic_map(gmap, panel, recomb_rate, fmt="plink", header=True)
        args = [qv, out_dir, contig, str(bool(phase)), gmap, rv, sm]
    if extra_args:
        args += list(extra_args)

    if not C.tool_available("gnomix"):
        raise FileNotFoundError(
            f"gnomix not found at {C.GNOMIX_DIR}; set TSPAINT_GNOMIX_DIR or TSPAINT_GNOMIX_CMD")
    C.run_tool("gnomix", args, cwd=workdir, log=log)

    fb = os.path.join(out_dir, "query_results.fb")
    tracks = parse_fb(fb, C.parse_inds(panel), panel.K, panel.sequence_length)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"gnomix: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
