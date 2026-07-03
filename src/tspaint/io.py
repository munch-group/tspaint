"""Input front ends: obtain a tspaint-ready tree sequence (CLAUDE.md §5, §7.4, §9).

tspaint paints any tree sequence with cross-tree node-ID stability. This namespace gathers the
supported ways to get one, under a **unified template** — name = the tool, return a tree
sequence (or, for SINGER, an ensemble of posterior samples):

* :func:`tsinfer` — tsinfer point estimate;
* :func:`relate` — Relate ``--compress`` conversion (run Relate upstream);
* :func:`singer` — SINGER Bayesian posterior ARG samples (a ``list`` — the input to the
  ensemble merge, CLAUDE.md §7.4). Needs an explicit ``Ne`` (SINGER's binary requires ``-Ne``);
  get one from :func:`estimate_ne` (π/4μ).
* :func:`argweaver` — ARGweaver posterior ARG samples (a ``list``; an alternative to
  :func:`singer`). Also needs an explicit ``Ne`` (``arg-sample`` requires ``-N``, via
  :func:`estimate_ne`); its ``.smc`` samples carry only trees (no mutations), which is all
  :func:`tspaint.paint` needs.

The two **inference** front ends (:func:`tsinfer`, :func:`singer`) take the same ``source``: a
:class:`tskit.TreeSequence` (with mutations), a **VCF Zarr** store, or a **VCF** file (normalised
by :mod:`tspaint.io_genotypes`). :func:`relate` takes Relate's ``.anc`` / ``.mut`` output instead.

Helpers: :func:`add_mutations` (overlay variants on a bare ARG for the sim pipeline),
:func:`check_persistence` (the §5.1 go/no-go). The pre-unification names (``infer_tree_sequence``,
``singer_tree_sequences``, ``convert_relate``) remain as **deprecated aliases**.

RFMix is a *comparator* (genotype-native), not an ARG front end — see
:func:`tspaint.compare.rfmix_paint`.
"""
from __future__ import annotations

from .io_tsinfer import tsinfer, add_mutations, infer_tree_sequence
from .io_singer import (singer, singer_windowed, singer_tree_sequences, write_haploid_vcf,
                        singer_window, build_merge_table, run_merge_arg)
from .io_argweaver import argweaver, write_sites
from .io_relate import relate, check_persistence, convert_relate
from .io_genotypes import subset_data, resolve_variants, Variants, estimate_ne, pseudohaploid
from .ids import attach_sample_ids, resolve_labels, resolve_ids, sample_id_index

__all__ = [
    # unified front ends (name = tool)
    "tsinfer", "relate", "singer", "argweaver",
    # SINGER long-region path: one call (many cores, no cluster) ...
    "singer_windowed",
    # ... or the per-window primitives it is built from (the cluster/GWF unit)
    "singer_window", "build_merge_table", "run_merge_arg",
    # data prep (normalise / slice a source before a front end)
    "subset_data", "resolve_variants", "Variants", "estimate_ne", "pseudohaploid",
    # sample identity: front ends stamp source ids; labels/queries resolve str-or-int keys
    "attach_sample_ids", "resolve_labels", "resolve_ids", "sample_id_index",
    # helpers
    "add_mutations", "write_haploid_vcf", "write_sites", "check_persistence",
    # deprecated aliases (pre-unification names)
    "infer_tree_sequence", "singer_tree_sequences", "convert_relate",
]
