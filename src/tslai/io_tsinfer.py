"""tsinfer front end (CLAUDE.md §5).

tsinfer tree sequences carry native cross-tree node-ID stability, so they are a
first-class front end for tslai — no Relate C++ toolchain required, and the spec
notes tsinfer is an alternative or even preferable front end. This module turns a
ts-with-mutations into an *inferred* tree sequence: the realistic substrate where
tree accuracy becomes the binding constraint (§9), as opposed to the true ARG.

``tsinfer`` is an optional dependency, imported lazily so the core package does not
require it.
"""
from __future__ import annotations

__all__ = ["add_mutations", "infer_tree_sequence"]


def add_mutations(ts, rate=1e-8, random_seed=None):
    """Overlay biallelic mutations on a tree sequence so tsinfer has variant data.

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
        ``ts`` with biallelic mutations overlaid under
        ``msprime.BinaryMutationModel``.
    """
    import msprime
    return msprime.sim_mutations(ts, rate=rate, random_seed=random_seed,
                                 model=msprime.BinaryMutationModel())


def infer_tree_sequence(ts_with_mutations):
    """Infer a tree sequence from the genotypes of ``ts_with_mutations`` via tsinfer.

    Produces the realistic substrate where tree accuracy becomes the binding
    constraint (§9), as opposed to the true ARG.

    Parameters
    ----------
    ts_with_mutations : tskit.TreeSequence
        Tree sequence carrying variant sites (e.g. from :func:`add_mutations`).

    Returns
    -------
    tskit.TreeSequence
        The tsinfer-inferred tree sequence. Sample nodes are preserved in input
        order, so per-sample labels and truth from the source ts transfer by
        sample id.

    Notes
    -----
    ``SampleData.from_tree_sequence`` is deprecated in tsinfer 0.5 (the forward
    path is a zarr-backed ``VariantData``); tsinfer is pinned ``<0.6`` so this
    stays valid. Migrate to ``VariantData`` when bumping tsinfer.
    """
    import tsinfer
    # SampleData.from_tree_sequence is deprecated in tsinfer 0.5 (the forward path is a
    # zarr-backed VariantData); tsinfer is pinned <0.6 so this stays valid. Migrate to
    # VariantData when bumping tsinfer.
    sample_data = tsinfer.SampleData.from_tree_sequence(ts_with_mutations)
    return tsinfer.infer(sample_data)
