"""Validation metrics — accuracy, calibration, and fragmentation / tract-length fidelity.

The public ``tspaint.metrics`` namespace; the implementation lives in :mod:`tspaint.validate`.
Operates on the per-haplotype :class:`~tspaint.output.Segment` tracks from a
:class:`~tspaint.Painting` (or :func:`tspaint.posterior_table`) and the true local-ancestry tracts
from :func:`tspaint.local_ancestry_truth` (mapped to ancestry-state indices via :func:`map_truth`).
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
