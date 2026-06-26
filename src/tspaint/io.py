"""Input front ends: obtain a tspaint-ready tree sequence (CLAUDE.md §5, §7.4, §9).

tspaint paints any tree sequence with cross-tree node-ID stability. This namespace gathers the
supported ways to get one, under a **unified template** — name = the tool, return a tree
sequence (or, for SINGER, an ensemble of posterior samples):

* :func:`tsinfer` — tsinfer point estimate;
* :func:`relate` — Relate ``--compress`` conversion (run Relate upstream);
* :func:`singer` — SINGER Bayesian posterior ARG samples (a ``list`` — the input to the
  ensemble merge, CLAUDE.md §7.4).

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
from .io_singer import singer, singer_tree_sequences, write_haploid_vcf
from .io_relate import relate, check_persistence, convert_relate

__all__ = [
    # unified front ends (name = tool)
    "tsinfer", "relate", "singer",
    # helpers
    "add_mutations", "write_haploid_vcf", "check_persistence",
    # deprecated aliases (pre-unification names)
    "infer_tree_sequence", "singer_tree_sequences", "convert_relate",
]
