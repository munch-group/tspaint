"""External-LAI benchmark bridges: run RFMix / gnomix / SALAI-Net / Recomb-Mix from VCF input
and write their calls in tspaint's ``.npz`` painting format (CLAUDE.md §9, §10).

Each of the four field comparators is a supervised, diploid, genotype-native local-ancestry
caller. This package gives each one the **same** VCF-in / ``.npz``-out contract so they sit beside
the native painter on equal footing:

>>> import tspaint.benchmark as bm
>>> tracks = bm.rfmix("query.vcf", "reference.vcf", sample_map="refs.tsv", out="rfmix.npz")
>>> tracks = bm.run("salai", "query.vcf", "reference.vcf", sample_map="refs.tsv", out="salai.npz")

Input is either two VCFs (query + reference) or one combined VCF split by the sample map; output
is ``{query-haplotype-index: [Segment]}`` (posteriors for RFMix/gnomix, one-hot 0/1 for
SALAI-Net/Recomb-Mix, which emit hard calls only) saved as a ``tspaint-painting`` ``.npz``.

Closing the loop on simulated truth:

>>> paths = bm.export_vcf(ts, labels, outdir="sim/")          # diploid VCFs + sample map + truth
>>> bm.score(paths["truth"], {"rfmix": "rfmix.npz"})          # leaderboard vs the truth

The four tools run in **their own environments** (each ships a ``pixi.toml``); see
:func:`tspaint.benchmark._common.tool_command` for the launcher and its ``TSPAINT_*`` overrides.
The terminal interface is ``tspaint benchmark <tool> --vcf …`` (:mod:`tspaint.benchmark.cli`).
"""
from __future__ import annotations

from .rfmix import rfmix
from .gnomix import gnomix
from .salai import salai
from .recombmix import recombmix
from ._tspaint import tspaint
from .export import export_vcf
from .score import score, load_truth, format_table, score_full, write_metrics, aggregate
from ._provision import setup, tool_status, load_manifest
from ._common import (resolve_panel, read_sample_map, save_tracks, tool_available, tool_command,
                      Panel)

__all__ = [
    "tspaint", "rfmix", "gnomix", "salai", "recombmix", "run", "BENCHMARK_TOOLS", "PAINTERS",
    "export_vcf", "score", "load_truth", "format_table", "score_full", "write_metrics", "aggregate",
    "setup", "tool_status", "load_manifest",
    "resolve_panel", "read_sample_map", "save_tracks", "tool_available", "tool_command", "Panel",
]

#: Name → external comparator runner (what ``setup`` / ``status`` manage). Every runner has the
#: signature ``f(query_vcf, ref_vcf=None, *, sample_map, ..., out=None) -> {hap_index: [Segment]}``.
BENCHMARK_TOOLS = {
    "rfmix": rfmix,
    "gnomix": gnomix,
    "salai": salai,
    "recombmix": recombmix,
}

#: Name → painter, including tspaint itself (VCF-native, tsinfer ARG) for the head-to-head.
PAINTERS = {"tspaint": tspaint, **BENCHMARK_TOOLS}


def run(tool, query_vcf, ref_vcf=None, **kwargs):
    """Dispatch to a painter by name (``"tspaint"`` or an external comparator).

    Forwards ``query_vcf``, ``ref_vcf`` and any painter keyword arguments (e.g. ``sample_map=``,
    ``out=``, ``generations=``, ``model=``, ``smooth=``) to the selected painter in :data:`PAINTERS`.
    """
    try:
        runner = PAINTERS[tool]
    except KeyError:
        raise ValueError(f"unknown painter {tool!r}; choose from {sorted(PAINTERS)}")
    return runner(query_vcf, ref_vcf, **kwargs)
