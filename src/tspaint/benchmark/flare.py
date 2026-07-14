"""FLARE benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9; papers/hapmix.md).

FLARE (Browning, Waples & Browning 2023, *AJHG* 110:326–335) is the maintained, modern descendant of
the HAPMIX / Li–Stephens copying-model lineage — and one of the three methods that beat Recomb-Mix
past ~200 generations, i.e. an opponent in *tspaint's own target regime*. HAPMIX itself is 2009 C and
is not worth reviving; FLARE is what we run in its place.

It is a **soft** comparator: run with ``probs=true`` it reports posterior ancestry probability
vectors per haplotype per marker (``ANP1`` / ``ANP2`` in its ``.anc.vcf.gz``), so it is scored
soft-vs-soft against tspaint rather than being flattened to hard calls — the same fair comparison we
get from RFMix's ``.fb`` and gnomix, and which Recomb-Mix / SALAI-Net cannot support.

The jar is built from source at the pinned commit (upstream ships no release); ``java`` comes from
FLARE's own pixi env. Override with ``TSPAINT_FLARE`` (jar path) / ``TSPAINT_FLARE_DIR`` /
``TSPAINT_FLARE_CMD``.
"""
from __future__ import annotations

import os
import tempfile

from . import _common as C
from ._msp import parse_flare_anc_vcf

__all__ = ["flare"]


def flare(query_vcf, ref_vcf=None, *, sample_map, genetic_map=None, chromosome=None,
          recomb_rate=1e-8, generations=10.0, probs=True, min_maf=0.0, min_mac=0, em=True,
          threads=None, seed=1, out=None, workdir=None, flare_jar=None, xmx="4g",
          extra_args=None, log=None):
    """Run FLARE on a query VCF and return per-query-haplotype **soft** Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, chromosome, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix`.
    genetic_map : str, optional
        A **PLINK ``.map``**: 4 columns ``chrom  marker-id  cM  bp`` (note cM *before* bp). FLARE
        rejects the 3-column map RFMix calls a genetic map. Generated uniformly from
        ``recomb_rate`` when omitted.
    recomb_rate : float, optional
        Per-base rate for the auto-generated map (default ``1e-8``).
    generations : float, optional
        Generations since admixture (FLARE's ``gen``; default 10).
    probs : bool, optional
        Report posterior ancestry probabilities (FLARE's ``probs``; default ``True``). Leave this
        on — it is what makes FLARE a soft comparator.
    min_maf, min_mac : float, int, optional
        FLARE's reference-panel marker filters. **Defaulted to 0 here, not to FLARE's own
        0.005 / 50**: those defaults silently discard every marker on the small panels used in
        simulation benchmarks (a 6-individual reference panel has a maximum MAC of 12), and FLARE
        then exits with no markers. Set them explicitly to match a published filter.
    em : bool, optional
        Estimate model parameters by EM (FLARE's ``em``; default ``True``).
    threads : int, optional
        Computational threads (FLARE's ``nthreads``; default: all cores).
    seed : int, optional
        RNG seed (default 1).
    flare_jar : str, optional
        Path to ``flare.jar`` (default :data:`tspaint.benchmark._common.FLARE_JAR`).
    xmx : str, optional
        JVM max heap (``-Xmx``; default ``"4g"``).
    extra_args : iterable[str], optional
        Extra ``key=value`` arguments appended to the FLARE command.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, a **soft** painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_flare_")
    panel, qv, rv, sm = C.setup_inputs(query_vcf, ref_vcf, sample_map, workdir)

    gmap = genetic_map
    if gmap is None:
        gmap = os.path.join(workdir, "plink.map")
        C.write_genetic_map(gmap, panel, recomb_rate, fmt="plink-map")

    if not C.tool_available("flare", bin_override=flare_jar):
        raise FileNotFoundError(
            f"flare.jar not found at {flare_jar or C.FLARE_JAR}; run `tspaint benchmark install "
            f"flare`, or set TSPAINT_FLARE")

    prefix = os.path.join(workdir, "flare_out")
    args = [f"ref={rv}", f"ref-panel={sm}", f"gt={qv}", f"map={gmap}", f"out={prefix}",
            f"probs={'true' if probs else 'false'}", f"gen={float(generations)}",
            f"min-maf={min_maf}", f"min-mac={int(min_mac)}",
            f"em={'true' if em else 'false'}", f"seed={int(seed)}"]
    if threads:
        args.append(f"nthreads={int(threads)}")
    if extra_args:
        args += list(extra_args)
    # -Xmx must precede -jar, so it rides in front of the tool args via the jar override hook.
    cmd = C.tool_command("flare", args, bin_override=flare_jar)
    cmd.insert(cmd.index("-jar"), f"-Xmx{xmx}")
    C.run_tool_argv(cmd, "flare", cwd=workdir, log=log)

    tracks = parse_flare_anc_vcf(prefix + ".anc.vcf.gz", C.parse_inds(panel), panel.K,
                                 panel.sequence_length, probs=probs)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"flare: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
