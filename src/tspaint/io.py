"""Input front ends: obtain a tspaint-ready tree sequence from genotypes (CLAUDE.md §5, §7.4, §9).

tspaint paints any tree sequence with cross-tree node-ID stability. This namespace gathers the
supported ways to get one:

* :func:`infer_tree_sequence` (+ :func:`add_mutations`) — tsinfer point estimate;
* :func:`singer_tree_sequences` — SINGER Bayesian posterior ARG samples (for the ensemble merge);
* Relate ``--compress`` — convert externally (CLAUDE.md §5), then ``tskit.load``.

RFMix is a *comparator* (genotype-native), not an ARG front end — see :func:`tspaint.compare.rfmix_paint`.
"""
from __future__ import annotations

from .io_tsinfer import add_mutations, infer_tree_sequence
from .io_singer import singer_tree_sequences, write_haploid_vcf

__all__ = [
    "add_mutations",
    "infer_tree_sequence",
    "singer_tree_sequences",
    "write_haploid_vcf",
]
