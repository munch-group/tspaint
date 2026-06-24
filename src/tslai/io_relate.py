"""Relate -> tskit front end (CLAUDE.md §5) — parallel integration track.

The core method is developed and validated on msprime/tsinfer tree sequences
(:mod:`tslai.sim`), which carry node persistence natively. Relate is the real-data
front end and must be converted with ``--compress`` (load-bearing, CLAUDE.md §5):
it assigns the same node age/id to nodes with identical descendant sets across
adjacent trees, which is the persistence invariant the edge-blocking depends on.

The persistence go/no-go (:func:`tslai.diagnostics.persistence_summary`) is
front-end-agnostic and should be run on any converted file before inference.
"""
from __future__ import annotations

from .diagnostics import persistence_summary

__all__ = ["convert_relate", "check_persistence"]


def convert_relate(anc, mut, out_prefix, compress=True, convert_bin="Convert"):  # pragma: no cover
    """Wrap ``relate_lib`` ``Convert --mode ConvertToTreeSequence [--compress]``.

    Lands when a Relate toolchain / example data is available (CLAUDE.md §5, §8.1).
    """
    raise NotImplementedError("Relate conversion lands with the io_relate track (CLAUDE.md §5)")


def check_persistence(ts):
    """Run the §5.1 persistence go/no-go on an already-loaded tree sequence."""
    return persistence_summary(ts)
