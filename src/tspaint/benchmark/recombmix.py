"""Recomb-Mix benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9, §10).

Recomb-Mix (Wang et al.): a recombination-aware minimum-cost-path LAI over a reference panel.
This runner shells out to the ``RecombMix_v0.7`` binary and parses its per-haplotype **segment**
output (``<name>_<hap>  start end state  …``), reported as one-hot (0/1) posteriors (Recomb-Mix
writes hard calls only; CLAUDE.md "report 0 or 1").

Binary path via ``TSPAINT_RECOMBMIX`` (default ``~/Recomb-Mix/RecombMix_v0.7``) or replace the
launcher with ``TSPAINT_RECOMBMIX_CMD``.
"""
from __future__ import annotations

import os
import tempfile

from . import _common as C
from ._msp import parse_recombmix_segments

__all__ = ["recombmix"]


def recombmix(query_vcf, ref_vcf=None, *, sample_map, genetic_map=None, chromosome=None,
              recomb_rate=1e-8, weight=1.5, threads=1, out=None, workdir=None,
              recombmix_bin=None, extra_args=None, log=None):
    """Run Recomb-Mix on a query VCF and return per-query-haplotype one-hot Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, genetic_map, chromosome, recomb_rate, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix` (``genetic_map`` here is a 4-column HapMap
        map; a uniform one is generated when omitted).
    weight : float, optional
        Recombination-rate weight in the cost function (Recomb-Mix's ``-e``; default 1.5).
    threads : int, optional
        Worker threads (Recomb-Mix's ``-t``; default 1).
    recombmix_bin : str, optional
        Path to the ``RecombMix_v0.7`` binary (default ``TSPAINT_RECOMBMIX``).
    extra_args : iterable[str], optional
        Extra arguments appended to the Recomb-Mix command.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, a one-hot (0/1) painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_recombmix_")
    panel, qv, rv, sm = C.setup_inputs(query_vcf, ref_vcf, sample_map, workdir,
                                       sample_map_header="#Sample_ID\tPopulation_Label")

    gmap = genetic_map
    if gmap is None:
        gmap = os.path.join(workdir, "genetic_map.txt")
        C.write_genetic_map(gmap, panel, recomb_rate, fmt="hapmap")

    if not C.tool_available("recombmix", bin_override=recombmix_bin):
        raise FileNotFoundError(
            f"RecombMix binary not found at {recombmix_bin or C.RECOMBMIX_BIN}; set TSPAINT_RECOMBMIX")
    inferred = "recombmix_out.txt"
    args = ["-p", rv, "-q", qv, "-a", sm, "-g", gmap,
            "-o", os.path.join(workdir, ""), "-i", inferred,
            "-e", str(weight), "-t", str(int(threads))]
    if extra_args:
        args += list(extra_args)
    C.run_tool("recombmix", args, cwd=workdir, log=log, bin_override=recombmix_bin)

    tracks = parse_recombmix_segments(os.path.join(workdir, inferred), C.parse_inds(panel),
                                      panel.K, panel.sequence_length)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"recombmix: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
