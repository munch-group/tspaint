"""Merge local-ancestry evidence across a posterior ensemble of tree sequences.

When the ARG is uncertain (the §9 binding constraint), a single point-estimate tree
sequence (tsinfer/Relate) inherits that ARG's error with no way to know it is wrong.
Given instead an ensemble of tree sequences sampling the ARG posterior — e.g. thinned
MCMC samples from SINGER (Deng et al., 2024), or any set of inferred ARGs sharing the
same samples and coordinates — the principled deliverable marginalises the ARG:

    P(s_i(x)=A | data) = E_{G ~ P(G|data)}[ gamma_i^G(x, A; theta) ]
                       ~= (1/M) sum_m gamma_i^{G_m}(x, A; theta)

i.e. paint each member with :func:`tslai.output.posterior_table` and **average the
per-position posteriors** here. The spread across members is an ARG-uncertainty band.

The ancestry-CTMC parameters ``theta`` are fit once, pooled across the ensemble, by
:func:`tslai.em.fit` with a list of tree sequences (its M-step is scale-invariant, so
summing sufficient statistics over the ensemble == averaging). This is a modular
("cut") model: the ARG posterior comes from the genotypes alone, not refined by the
ancestry labels — justified because the labels are sparse tip annotations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .output import INFORMATIVE, MISSING_INFO

__all__ = ["MergedSegment", "merge_posterior_tables"]


@dataclass
class MergedSegment:
    left: float
    right: float
    posterior: np.ndarray       # (K,) mean posterior over the ensemble (the painting)
    status: str                 # INFORMATIVE if any member is informative here, else MISSING_INFO
    posterior_std: np.ndarray   # (K,) std over the ensemble — the ARG-uncertainty band
    n_informative: int          # how many ensemble members were informative on this span


def _refine_tracks(track_list):
    """Yield ``(lo, hi, [Segment_per_member])`` over the common breakpoint refinement
    of M piecewise-constant tracks that each cover the same ``[0, L)`` contiguously.

    Generalises the 2-way :func:`tslai.validate._walk_overlap` to M tracks: advance
    every member whose current segment ends at the current right breakpoint.
    """
    m = len(track_list)
    ptr = [0] * m
    while all(ptr[i] < len(track_list[i]) for i in range(m)):
        cur = [track_list[i][ptr[i]] for i in range(m)]
        lo = max(seg.left for seg in cur)
        hi = min(seg.right for seg in cur)
        if hi > lo:
            yield lo, hi, cur
        for i in range(m):
            if track_list[i][ptr[i]].right == hi:
                ptr[i] += 1


def merge_posterior_tables(tables, samples=None):
    """Average per-haplotype posteriors across an ensemble of paintings.

    Parameters
    ----------
    tables : list[dict[int, list[Segment]]]
        One :func:`tslai.output.posterior_table` per ensemble member; all over the same
        samples and genome coordinates (sample ids preserved across members).
    samples : iterable[int], optional
        Restrict to these sample ids (default: those in the first table).

    Returns
    -------
    dict[int, list[MergedSegment]]
        Per sample, the ensemble-mean posterior with an uncertainty band, on the common
        breakpoint refinement. Duck-compatible with ``Segment`` (``left``/``right``/
        ``posterior``/``status``), so :mod:`tslai.validate` metrics score it directly.
    """
    if not tables:
        raise ValueError("need at least one posterior table to merge")
    sample_ids = list(tables[0].keys()) if samples is None else [int(s) for s in samples]

    merged = {}
    for s in sample_ids:
        track_list = [tab[s] for tab in tables]
        out = []
        for lo, hi, segs in _refine_tracks(track_list):
            posts = np.stack([seg.posterior for seg in segs])   # (M, K)
            n_inf = sum(1 for seg in segs if seg.status == INFORMATIVE)
            status = INFORMATIVE if n_inf > 0 else MISSING_INFO
            out.append(MergedSegment(lo, hi, posts.mean(axis=0), status,
                                     posts.std(axis=0), n_inf))
        merged[s] = out
    return merged
