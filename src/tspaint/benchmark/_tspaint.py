"""tspaint as a VCF-native benchmark painter — the fair head-to-head entry (CLAUDE.md §9).

The external comparators are genotype-native, so to put tspaint on equal footing this painter
takes the **same** query + reference VCFs, infers an ARG from them with **tsinfer** (the realistic
substrate where tree-inference accuracy is the binding constraint, not the true genealogy), fits
the ancestry model and paints the queries — emitting the **same hap-index-keyed** ``.npz`` as the
tool runners (and the :func:`tspaint.benchmark.export_vcf` truth), so all painters score
identically.

For the upper-bound (true-ARG) tspaint number, paint the simulation's ``.trees`` directly with
``tspaint paint`` and score against the node-id truth — the aggregate metrics are comparable.
"""
from __future__ import annotations

import tempfile

import numpy as np

from . import _common as C

__all__ = ["tspaint"]


def _infer_combined_arg(panel):
    """tsinfer an ARG from the panel's query+reference haplotypes (query haps first).

    Returns ``(ts, n_query_haps, ref_states)``; inferred sample node ``i`` is haplotype column
    ``i`` — query hap index ``i`` for ``i < n_query_haps``, then one reference hap per state in
    ``ref_states``.
    """
    import tsinfer
    from ..io_genotypes import Variants, to_sample_data

    q_cols = [c for (_n, cols, _k) in panel.query for c in cols]      # query haps, hap-key order
    r_cols, ref_states = [], []
    for (_n, cols, state) in panel.ref:
        for c in cols:
            r_cols.append(c)
            ref_states.append(int(state))
    geno = panel.geno[:, q_cols + r_cols]
    variants = Variants(positions=panel.positions.astype(float), genotypes=geno,
                        alleles=panel.alleles, sequence_length=panel.sequence_length)
    ts = tsinfer.infer(to_sample_data(variants))
    return ts, len(q_cols), ref_states


def tspaint(query_vcf, ref_vcf=None, *, sample_map, arg="tsinfer", smooth=True, estimate_pi=False,
            max_iter=12, out=None, workdir=None, log=None):
    """Paint queries with tspaint over an ARG inferred from the VCFs (hap-index keyed).

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix` — query (or combined) VCF, optional separate
        reference VCF, the sample map, output ``.npz``, work dir, progress sink.
    arg : str, optional
        ARG front end. ``"tsinfer"`` (default) infers the ARG from the VCFs. (The true-ARG upper
        bound is obtained outside this runner by painting the simulation ``.trees`` directly.)
    smooth : bool, optional
        Apply the horizontal BP smoother (recommended on inferred ARGs; CLAUDE.md §7). Default True.
    estimate_pi : bool, optional
        Re-estimate ``π`` (default False — hold uniform, robust to the π-degeneracy; CLAUDE.md §6).
    max_iter : int, optional
        EM iterations (default 12).

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, the painted posterior over ``[0, L)``.
    """
    if arg != "tsinfer":
        raise ValueError(f"unknown arg front end {arg!r} (only 'tsinfer' is supported here)")
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_bench_")
    panel = C.resolve_panel(query_vcf, ref_vcf, sample_map=sample_map)
    if log:
        log(f"tspaint: inferring ARG (tsinfer) from {panel.positions.size} sites, "
            f"{panel.n_query_haps} query + {2 * len(panel.ref)} reference haplotypes")

    ts, n_query, ref_states = _infer_combined_arg(panel)
    queries = list(range(n_query))                                   # = query hap indices
    labels = {n_query + i: s for i, s in enumerate(ref_states)}

    from .. import paint as _paint                                   # api.paint (lazy)
    painting = _paint(ts, labels, queries=queries, smooth=smooth, estimate_pi=estimate_pi,
                      max_iter=max_iter)
    tracks = {int(k): list(v) for k, v in painting.posteriors.items()}
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"tspaint: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
