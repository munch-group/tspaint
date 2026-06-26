"""tsinfer front end (CLAUDE.md §5).

tsinfer tree sequences carry native cross-tree node-ID stability, so they are a first-class
front end for tspaint — no Relate C++ toolchain required. :func:`tsinfer` turns genotypes (a
tree sequence with mutations, a VCF Zarr, or a VCF) into an *inferred* tree sequence: the
realistic substrate where tree accuracy becomes the binding constraint (§9).

``tsinfer`` is an optional dependency, imported lazily so the core package does not require it.
"""
from __future__ import annotations

import warnings

__all__ = ["tsinfer", "add_mutations", "infer_tree_sequence"]


def add_mutations(ts, rate=1e-8, random_seed=None):
    """Overlay biallelic mutations on a tree sequence so the inference front ends have variant
    data.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence to mutate (typically a bare, mutationless ARG).
    rate : float, optional
        Per-base, per-generation mutation rate (default ``1e-8``).
    random_seed : int, optional
        Seed for the mutation simulation.

    Returns
    -------
    tskit.TreeSequence
        ``ts`` with biallelic mutations overlaid under ``msprime.BinaryMutationModel``.
    """
    import msprime
    return msprime.sim_mutations(ts, rate=rate, random_seed=random_seed,
                                 model=msprime.BinaryMutationModel())


def tsinfer(source):
    """Infer a tree sequence from genotypes via tsinfer (CLAUDE.md §5, §9).

    Produces the realistic substrate where tree accuracy, rather than the true ARG, becomes the
    binding constraint (§9).

    Parameters
    ----------
    source : tskit.TreeSequence or str
        The genotypes to infer from — a tree sequence carrying variant sites (e.g. from
        :func:`add_mutations`), a **VCF Zarr** store, or a **VCF** file (see
        :mod:`tspaint.io_genotypes` for the unified handling and its v1 limits).

    Returns
    -------
    tskit.TreeSequence
        The tsinfer-inferred tree sequence. Sample nodes are preserved in input order, so
        per-sample labels and truth transfer by sample id.

    Notes
    -----
    A ``ts`` source uses ``SampleData.from_tree_sequence`` (deprecated in tsinfer 0.5 but valid
    while tsinfer is pinned ``<0.6``); VCF / Zarr sources are normalised by
    :mod:`tspaint.io_genotypes` and added site-by-site.
    """
    import tsinfer as _tsinfer
    from .io_genotypes import source_kind, resolve_variants, to_sample_data
    if source_kind(source) == "ts":
        sample_data = _tsinfer.SampleData.from_tree_sequence(source)
    else:
        sample_data = to_sample_data(resolve_variants(source))
    return _tsinfer.infer(sample_data)


def infer_tree_sequence(ts_with_mutations):
    """Deprecated alias for :func:`tsinfer`."""
    warnings.warn("tspaint.io.infer_tree_sequence is deprecated; use tspaint.io.tsinfer",
                  DeprecationWarning, stacklevel=2)
    return tsinfer(ts_with_mutations)
