"""Relate -> tskit front end (CLAUDE.md §5).

The core method is developed and validated on msprime / tsinfer tree sequences
(:mod:`tspaint.sim`), which carry node persistence natively. Relate is the real-data front end
and must be converted with ``--compress`` (load-bearing, CLAUDE.md §5): it assigns the same node
age / id to nodes with identical descendant sets across adjacent trees — the persistence
invariant the edge-blocking depends on.

:func:`relate` wraps ``relate_lib``'s ``Convert`` binary (run Relate itself externally first);
the binary path defaults to env ``TSPAINT_RELATE_CONVERT`` or ``Convert`` on ``PATH``. Run the
:func:`check_persistence` go/no-go (§5.1) on any converted file before inference.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

from .diagnostics import persistence_summary

__all__ = ["relate", "check_persistence", "convert_relate"]

DEFAULT_CONVERT = os.environ.get("TSPAINT_RELATE_CONVERT", "Convert")


def relate(anc, mut, *, out_prefix=None, compress=True, convert_bin=None):
    """Convert Relate output (``.anc`` / ``.mut``) to a tskit tree sequence (CLAUDE.md §5).

    Wraps ``relate_lib`` ``Convert --mode ConvertToTreeSequence [--compress]`` (run Relate itself
    upstream). ``--compress`` is **load-bearing** — it unifies persistent clades into one node id
    across adjacent trees, the persistence invariant the edge-blocking depends on (§5) — so keep
    it on. Run :func:`check_persistence` on the result before inference (§5.1).

    Parameters
    ----------
    anc : str
        Path to the Relate ``.anc`` (``.gz``) file.
    mut : str
        Path to the Relate ``.mut`` (``.gz``) file.
    out_prefix : str, optional
        Output prefix for the converted ``.trees`` (default: a tempfile prefix).
    compress : bool, optional
        Pass ``--compress`` (default ``True`` — load-bearing; do not turn off, §5).
    convert_bin : str, optional
        Path to the ``relate_lib`` ``Convert`` binary (default: env ``TSPAINT_RELATE_CONVERT``
        or ``Convert`` on ``PATH``).

    Returns
    -------
    tskit.TreeSequence
        The converted tree sequence (node ids stable across trees thanks to ``--compress``).

    Raises
    ------
    FileNotFoundError
        If the ``Convert`` binary or an input file is absent.
    RuntimeError
        If ``Convert`` exits nonzero.
    """
    import tskit
    convert_bin = convert_bin or DEFAULT_CONVERT
    if shutil.which(convert_bin) is None and not os.path.exists(convert_bin):
        raise FileNotFoundError(
            f"relate_lib Convert binary not found ({convert_bin!r}); set TSPAINT_RELATE_CONVERT or "
            "pass convert_bin (https://github.com/leospeidel/relate_lib)")
    for path, what in ((anc, "anc"), (mut, "mut")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Relate {what} file not found: {path}")
    out_prefix = out_prefix or os.path.join(tempfile.mkdtemp(prefix="tspaint_relate_"), "relate")
    cmd = [convert_bin, "--mode", "ConvertToTreeSequence", "--anc", anc, "--mut", mut, "-o", out_prefix]
    if compress:
        cmd.append("--compress")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Relate Convert failed:\n{res.stderr[-1000:]}")
    return tskit.load(out_prefix + ".trees")


def check_persistence(ts):
    """Run the §5.1 persistence go/no-go on an already-loaded tree sequence."""
    return persistence_summary(ts)


def convert_relate(anc, mut, out_prefix, compress=True, convert_bin="Convert"):  # pragma: no cover
    """Deprecated alias for :func:`relate`."""
    warnings.warn("tspaint.io.convert_relate is deprecated; use tspaint.io.relate",
                  DeprecationWarning, stacklevel=2)
    return relate(anc, mut, out_prefix=out_prefix, compress=compress, convert_bin=convert_bin)
