"""Validation metrics — accuracy, calibration, and fragmentation / tract-length fidelity.

The public ``tslai.metrics`` namespace; the implementation lives in :mod:`tslai.validate`.
Operates on the per-haplotype :class:`~tslai.output.Segment` tracks from a
:class:`~tslai.Painting` (or :func:`tslai.posterior_table`) and the true local-ancestry tracts
from :func:`tslai.local_ancestry_truth` (mapped to ancestry-state indices via :func:`map_truth`).
"""
from __future__ import annotations

from .validate import (
    map_truth,
    per_base_accuracy,
    balanced_accuracy,
    mean_confidence,
    reliability_curve,
    breakpoint_flicker,
    tract_boundary_error,
    breakpoint_precision_recall,
    switch_density,
)

__all__ = [
    "map_truth",
    "per_base_accuracy",
    "balanced_accuracy",
    "mean_confidence",
    "reliability_curve",
    "breakpoint_flicker",
    "tract_boundary_error",
    "breakpoint_precision_recall",
    "switch_density",
]
