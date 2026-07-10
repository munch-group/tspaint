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


def _tsdate_calibrate(ts, mutation_rate, tsdate_kwargs=None):
    """Calibrate a tree sequence's node ages to **generations** with tsdate.

    tsinfer node times are *uncalibrated* — the inferred ancestors are ordered by frequency, so the
    times are ~``[0, 1]`` rather than generations. tsdate re-times the nodes in generations from the
    mutation clock, which is what :meth:`tspaint.Painting.rate_through_time` (and any node-age
    read-out) needs. Runs :func:`tsdate.preprocess_ts` first, since tsdate requires unary nodes
    removed / disjoint regions split (a raw tsinfer ARG has unary nodes). Uses tsdate's default
    ``variational_gamma`` method, which is mutation-clock based and needs **no** ``Ne`` (for a method
    that uses a coalescent ``population_size`` prior, pass ``tsdate_kwargs={"method": ..., ...}``).
    Sample nodes, individuals and any stamped sample ids are preserved.
    """
    if mutation_rate is None:
        raise ValueError("io.tsinfer(date=True) needs mutation_rate=... to calibrate node ages to "
                         "generations (tsdate's mutation clock).")
    try:
        import tsdate
    except ImportError as e:                                            # optional dep
        raise ImportError("io.tsinfer(date=True) requires tsdate — `pip install tsdate` or install "
                          "the 'dating' extra (pip install tspaint[dating]).") from e
    pre = tsdate.preprocess_ts(ts)                                      # tsdate needs no unary nodes
    kw = dict(tsdate_kwargs or {})
    try:
        return tsdate.date(pre, mutation_rate=float(mutation_rate), **kw)
    except AssertionError as e:
        # tsdate's default time-rescaling ("Use fewer rescaling intervals") can fail on small tree
        # sequences (few distinct node ages); retry with fewer intervals so it doesn't crash. Real
        # whole-genome data uses the default. Only intervene if the caller didn't set it themselves.
        if "rescaling" not in str(e).lower() or "rescaling_intervals" in kw:
            raise
        import warnings
        ri = max(10, min(500, pre.num_nodes))
        warnings.warn(f"tsdate's default time-rescaling failed on this (small) tree sequence; "
                      f"retrying with rescaling_intervals={ri}. Node ages are less reliable on "
                      f"small data — tsdate calibrates best on larger samples/regions.",
                      RuntimeWarning, stacklevel=3)
        return tsdate.date(pre, mutation_rate=float(mutation_rate), rescaling_intervals=ri, **kw)


def tsinfer(source, *, date=False, mutation_rate=None, tsdate_kwargs=None):
    """Infer a tree sequence from genotypes via tsinfer (CLAUDE.md §5, §9).

    Produces the realistic substrate where tree accuracy, rather than the true ARG, becomes the
    binding constraint (§9).

    Parameters
    ----------
    source : tskit.TreeSequence, str, or Variants
        The genotypes to infer from — a tree sequence carrying variant sites (e.g. from
        :func:`add_mutations`), a **VCF Zarr** store, a **VCF** file, or a
        :class:`~tspaint.io_genotypes.Variants` (e.g. from :func:`~tspaint.io.subset_data` /
        :func:`~tspaint.io.pseudohaploid`) — see :mod:`tspaint.io_genotypes` for the unified
        handling and its v1 limits.
    date : bool, optional
        Calibrate the inferred node ages to **generations** with tsdate (default ``False``). A raw
        tsinfer ARG has *uncalibrated* times (~``[0, 1]``), which is fine for :func:`tspaint.paint`
        (it needs only relative branch lengths) but makes :meth:`tspaint.Painting.rate_through_time`
        meaningless. Set ``date=True`` (with ``mutation_rate``) to get times in generations.
    mutation_rate : float, optional
        Per-base mutation rate for tsdate's clock — **required** when ``date=True``.
    tsdate_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`tsdate.date` (e.g. ``method=``); the default
        method is ``variational_gamma`` (mutation-clock, no ``Ne`` needed).

    Returns
    -------
    tskit.TreeSequence
        The tsinfer-inferred tree sequence (node ages in **generations** when ``date=True``, else
        uncalibrated). Sample nodes are preserved in input order, so per-sample labels and truth
        transfer by sample id. For a **VCF** / **VCF-Zarr** source the source's sample ids are
        stamped onto the sample nodes (:func:`tspaint.ids.attach_sample_ids`), so
        :func:`tspaint.paint` accepts ``labels`` keyed by sample-ID string as well as by node index
        (the stamping survives dating). A bare **ts** source is returned unstamped.

    Notes
    -----
    A **zarr** source is read chunked via ``tsinfer.VariantData``
    (:func:`tspaint.io_genotypes.variant_data_from_zarr`) — scalable to whole-genome data; a **ts**
    source uses ``SampleData.from_tree_sequence`` (deprecated in tsinfer 0.5 but valid while tsinfer
    is pinned ``<0.6``); a **VCF** is parsed in-memory by :mod:`tspaint.io_genotypes`.
    """
    if date and mutation_rate is None:                                  # fail fast, before inference
        raise ValueError("io.tsinfer(date=True) needs mutation_rate=... to calibrate node ages to "
                         "generations (tsdate's mutation clock).")
    import tsinfer as _tsinfer
    from .ids import attach_sample_ids
    from .io_genotypes import (Variants, source_kind, variants_from_vcf, to_sample_data,
                               variant_data_from_zarr, sample_names_from_zarr)
    if isinstance(source, Variants):                                    # e.g. subset_data / pseudohaploid
        inferred = attach_sample_ids(_tsinfer.infer(to_sample_data(source)),
                                     source.sample_names, source.ploidy)
    else:
        kind = source_kind(source)
        if kind == "ts":
            inferred = _tsinfer.infer(_tsinfer.SampleData.from_tree_sequence(source))
        elif kind == "zarr":
            ts = _tsinfer.infer(variant_data_from_zarr(source))         # chunked / scalable
            names, ploidy = sample_names_from_zarr(source)              # cheap: no genotype I/O
            inferred = attach_sample_ids(ts, names, ploidy)
        else:
            v = variants_from_vcf(source)                               # VCF -> in-memory SampleData
            inferred = attach_sample_ids(_tsinfer.infer(to_sample_data(v)), v.sample_names, v.ploidy)
    if date:
        inferred = _tsdate_calibrate(inferred, mutation_rate, tsdate_kwargs)
    return inferred


def infer_tree_sequence(ts_with_mutations):
    """Deprecated alias for :func:`tsinfer`.

    Kept for backward compatibility; call :func:`tsinfer` instead. Forwards its argument as the
    ``source`` of :func:`tsinfer` (every other option left at its default, notably ``date=False``),
    emitting a ``DeprecationWarning``, so it returns *uncalibrated* inferred node times.

    Parameters
    ----------
    ts_with_mutations : tskit.TreeSequence
        A tree sequence carrying variant sites (e.g. from :func:`add_mutations`), passed straight
        through as :func:`tsinfer`'s ``source``.

    Returns
    -------
    tskit.TreeSequence
        The tsinfer-inferred tree sequence (uncalibrated node times); see :func:`tsinfer`.
    """
    warnings.warn("tspaint.io.infer_tree_sequence is deprecated; use tspaint.io.tsinfer",
                  DeprecationWarning, stacklevel=2)
    return tsinfer(ts_with_mutations)
