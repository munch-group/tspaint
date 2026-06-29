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
import shutil
import tempfile

from . import _common as C
from ._msp import parse_fb

__all__ = ["gnomix"]


def _genetic_length_cM(gmap_path):
    """Total map length in cM = max of the last column over numeric rows.

    Works for both genetic-map formats :func:`tspaint.benchmark._common.write_genetic_map` emits
    (plink ``chrom pos cM`` and hapmap ``... Map(cM)``) — cM is the last column in each — with
    header / comment rows skipped (non-numeric last field).
    """
    best = 0.0
    with open(gmap_path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                best = max(best, float(parts[-1]))
            except ValueError:
                continue
    return best


def _fit_config(cfg, total_cM):
    """Scale gnomix's window / smoother so ``W >= 2*smooth_size`` holds on a short region.

    gnomix's shipped configs target whole chromosomes; the smoother asserts ``W >= 2*smooth_size``
    (``src/Smooth/models.py``) where the window count ``W ≈ total_cM / window_size_cM``
    (``gnomix.py``: ``M = round(window_size_cM * C/total_cM)``, ``W = C // M``). We shrink
    ``window_size_cM`` toward ``total_cM / (3*smooth_size)`` (≈1.5x margin over the floor) without
    coarsening past the config's own window, and cap ``smooth_size`` as a fallback for very short
    regions. Mutates and returns ``cfg``.
    """
    model = cfg.setdefault("model", {})
    smooth = int(model.get("smooth_size") or 75)
    window = float(model.get("window_size_cM") or 0.2)
    if total_cM > 0:
        window = min(window, max(total_cM / (3 * smooth), 0.01))
        W = int(total_cM / window)
        if W < 2 * smooth:                       # very short region: shrink the smoother too
            smooth = max(2, W // 2)
    model["window_size_cM"] = window
    model["smooth_size"] = smooth
    return cfg


def gnomix(query_vcf, ref_vcf=None, *, sample_map, genetic_map=None, chromosome=None,
           recomb_rate=1e-8, model=None, phase=False, config=None, out=None, workdir=None,
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
    config : str, optional
        Path to a gnomix ``config.yaml``. Defaults to the one shipped in the gnomix repo
        (``<GNOMIX_DIR>/config.yaml``). gnomix otherwise reads ``./config.yaml`` from its working
        directory, which our temp workdir does not have — so it is staged there (see below).
    extra_args : iterable[str], optional
        Extra positional arguments appended after the standard gnomix arguments.

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

    # Build the gnomix positional args (train-from-reference unless a pretrained model is given).
    if model is not None:
        C.require(model, "pretrained gnomix model .pkl not found")
        args = [qv, out_dir, contig, str(bool(phase)), model]
        gmap = None
    else:
        gmap = genetic_map
        if gmap is None:
            gmap = os.path.join(workdir, "genetic_map.tsv")
            C.write_genetic_map(gmap, panel, recomb_rate, fmt="plink", header=True)
        args = [qv, out_dir, contig, str(bool(phase)), gmap, rv, sm]
    if extra_args:
        args += list(extra_args)

    # gnomix hardcodes its config to ``./config.yaml`` relative to its cwd (only train mode can
    # override it via a trailing arg; pre-trained mode cannot at all). We run with ``cwd=workdir``
    # (a fresh temp dir), so stage a config there to satisfy both modes. gnomix's shipped configs
    # target whole chromosomes: their smoother asserts ``W >= 2*smooth_size`` windows, where
    # ``W ≈ total_cM / window_size_cM``, which fails on the benchmark's short regions. So when the
    # default config is used, scale ``window_size_cM`` / ``smooth_size`` to the region length.
    src_cfg = config or os.path.join(C.GNOMIX_DIR, "config.yaml")
    C.require(src_cfg, "gnomix config.yaml not found — pass config= or set TSPAINT_GNOMIX_DIR")
    staged_cfg = os.path.join(workdir, "config.yaml")
    if config is None and gmap is not None:
        import yaml
        with open(src_cfg) as fh:
            cfg_dict = yaml.safe_load(fh)
        total_cM = _genetic_length_cM(gmap)
        _fit_config(cfg_dict, total_cM)
        with open(staged_cfg, "w") as fh:
            yaml.safe_dump(cfg_dict, fh, sort_keys=False)
        if log:
            m = cfg_dict["model"]
            log(f"gnomix: fitted config to a {total_cM:.2f} cM region "
                f"(window_size_cM={m['window_size_cM']:.4g}, smooth_size={m['smooth_size']})")
    elif os.path.abspath(src_cfg) != os.path.abspath(staged_cfg):
        shutil.copyfile(src_cfg, staged_cfg)

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
