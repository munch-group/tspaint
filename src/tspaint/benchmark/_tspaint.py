"""tspaint as a VCF-native benchmark painter — the fair head-to-head entry (CLAUDE.md §9).

The external comparators are genotype-native, so to put tspaint on equal footing this painter
takes the **same** query + reference VCFs and infers the genealogy from them before painting,
emitting the **same hap-index-keyed** ``.npz`` as the tool runners (and the
:func:`tspaint.benchmark.export_vcf` truth), so all painters score identically. Two ARG front
ends (``arg=``):

* ``"tsinfer"`` (default) — a single inferred tree sequence (the cheap, point-estimate substrate);
* ``"singer"`` — a SINGER posterior ARG **ensemble** (``n_singer`` thinned post-burn-in samples);
  tspaint fits one ``(Q, π, w)`` pooled across the ensemble, paints each member and averages the
  posteriors, so the ``.npz`` carries the ensemble-mean painting (CLAUDE.md §7.4).

For the upper-bound (true-ARG) tspaint number, paint the simulation's ``.trees`` directly with
``tspaint paint`` and score against the node-id truth — the aggregate metrics are comparable.
"""
from __future__ import annotations

import tempfile

from . import _common as C

__all__ = ["tspaint"]


def _combined_variants(panel):
    """The panel's query+reference haplotypes as one :class:`~tspaint.io_genotypes.Variants`.

    Column order is **query haps first** (hap-key order), then one column per reference haplotype;
    returns ``(variants, n_query_haps, ref_states)`` so sample/column ``i`` is query hap index
    ``i`` for ``i < n_query_haps`` and reference state ``ref_states[i - n_query_haps]`` after.
    """
    from ..io_genotypes import Variants

    q_cols = [c for (_n, cols, _k) in panel.query for c in cols]      # query haps, hap-key order
    r_cols, ref_states = [], []
    for (_n, cols, state) in panel.ref:
        for c in cols:
            r_cols.append(c)
            ref_states.append(int(state))
    geno = panel.geno[:, q_cols + r_cols]
    variants = Variants(positions=panel.positions.astype(float), genotypes=geno,
                        alleles=panel.alleles, sequence_length=panel.sequence_length)
    return variants, len(q_cols), ref_states


def _infer_combined_arg(panel):
    """tsinfer a single ARG from the panel's combined haplotypes (query haps first).

    Returns ``(ts, n_query_haps, ref_states)`` with the column→sample mapping of
    :func:`_combined_variants`.
    """
    import tsinfer
    from ..io_genotypes import to_sample_data

    variants, n_query, ref_states = _combined_variants(panel)
    ts = tsinfer.infer(to_sample_data(variants))
    return ts, n_query, ref_states


def _singer_ensemble(panel, *, Ne, mutation_rate, recombination_rate, n_singer, thin, burn_in,
                     seed, singer_bin=None, log=None):
    """A SINGER posterior ARG ensemble for the panel's combined haplotypes (query haps first).

    Draws ``n_singer + burn_in`` MCMC samples (``thin`` apart) and discards the ``burn_in`` leading
    ones, so ``~n_singer`` post-burn-in tree sequences remain. Returns ``(ensemble, n_query_haps,
    ref_states)`` (``ensemble`` a list of tree sequences sharing the column→sample mapping of
    :func:`_combined_variants`).
    """
    from ..io_singer import singer

    variants, n_query, ref_states = _combined_variants(panel)
    if log:
        log(f"tspaint: running SINGER (n={n_singer}, burn_in={burn_in}, thin={thin}) on "
            f"{variants.positions.size} sites, {variants.genotypes.shape[1]} haplotypes")
    ensemble = singer(variants, _Ne=Ne, _m=mutation_rate,
                      _r=recombination_rate, ts=n_singer,
                      mcmc_step=thin, mcmc_burnin=burn_in * thin, _seed=seed, singer_bin=singer_bin)
    if not isinstance(ensemble, (list, tuple)):     # singer() collapses a 1-member ensemble
        ensemble = [ensemble]
    return list(ensemble), n_query, ref_states


def tspaint(query_vcf, ref_vcf=None, *, sample_map, arg="tsinfer", smooth=True, estimate_pi=False,
            max_iter=12, n_jobs=None, Ne=1e4, mutation_rate=1.25e-8, recombination_rate=1e-8,
            n_singer=100, thin=20, burn_in=20, singer_seed=42, singer_bin=None,
            out=None, workdir=None, log=None):
    """Paint queries with tspaint over an ARG inferred from the VCFs (hap-index keyed).

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix` — query (or combined) VCF, optional separate
        reference VCF, the sample map, output ``.npz``, work dir, progress sink.
    arg : str, optional
        ARG front end: ``"tsinfer"`` (default, a single inferred tree sequence) or ``"singer"``
        (a SINGER posterior ARG ensemble; needs the SINGER binary — ``TSPAINT_SINGER`` /
        ``singer_bin``). (The true-ARG upper bound is obtained outside this runner by painting the
        simulation ``.trees`` directly.)
    smooth : bool, optional
        Apply the horizontal BP smoother (recommended on inferred ARGs; CLAUDE.md §7). Default True.
    estimate_pi : bool, optional
        Re-estimate ``π`` (default False — hold uniform, robust to the π-degeneracy; CLAUDE.md §6).
    max_iter : int, optional
        EM iterations (default 12).
    n_jobs : int, optional
        Worker processes for the genome E-step / painting (default: SLURM allocation else all CPUs).
    Ne, mutation_rate, recombination_rate : float, optional
        Demographic / rate parameters passed to SINGER (``arg="singer"`` only).
    n_singer, thin, burn_in : int, optional
        SINGER ensemble size (post-burn-in samples to paint), MCMC thinning interval, and burn-in
        (``arg="singer"`` only; defaults 100 / 20 / 20).
    singer_seed : int, optional
        SINGER base random seed (``arg="singer"`` only).
    singer_bin : str, optional
        Path to the SINGER binary (default: ``TSPAINT_SINGER``).

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, the painted posterior over ``[0, L)`` (the
        ensemble-mean posterior for ``arg="singer"``).
    """
    from ..parallel import resolve_cores
    n_jobs = resolve_cores(n_jobs)
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_bench_")
    panel = C.resolve_panel(query_vcf, ref_vcf, sample_map=sample_map)
    if log:
        log(f"tspaint[{arg}]: {panel.positions.size} sites, {panel.n_query_haps} query + "
            f"{2 * len(panel.ref)} reference haplotypes, n_jobs={n_jobs}")

    if arg == "tsinfer":
        paint_input, n_query, ref_states = _infer_combined_arg(panel)
    elif arg == "singer":
        paint_input, n_query, ref_states = _singer_ensemble(
            panel, Ne=Ne, mutation_rate=mutation_rate, recombination_rate=recombination_rate,
            n_singer=n_singer, thin=thin, burn_in=burn_in, seed=singer_seed,
            singer_bin=singer_bin, log=log)
        if log:
            log(f"tspaint[singer]: painting an ensemble of {len(paint_input)} posterior ARGs")
    else:
        raise ValueError(f"unknown arg front end {arg!r} (use 'tsinfer' or 'singer')")

    queries = list(range(n_query))                                   # = query hap indices
    labels = {n_query + i: s for i, s in enumerate(ref_states)}

    from .. import paint as _paint                                   # api.paint (lazy)
    painting = _paint(paint_input, labels, queries=queries, smooth=smooth,
                      estimate_pi=estimate_pi, max_iter=max_iter, n_jobs=n_jobs)
    tracks = {int(k): list(v) for k, v in painting.posteriors.items()}
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"tspaint: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
