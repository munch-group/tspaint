"""Loopy belief propagation / expectation propagation over the ARG coupling graph (CLAUDE.md §7).

Triggered by the §9 fragmentation finding: at older admixture the blocked-EM per-tree posterior
wobbles near 0.5 and the hard `argmax` segmentation over-fragments tracts (biasing admixture-pulse
dating). §3.5's dropped horizontal coupling is the cause; this package restores it.

Implemented: :func:`~tslai.bp.horizontal.bp_paint` — a **single-pass** horizontal smoother
(genome-axis forward-backward over each tip's per-tree beliefs, switch penalty ``epsilon``), the
EP first half of §7.2's schedule. The full-loopy iteration (re-feeding smoothed beliefs into the
vertical pruning of shared internal nodes) is the remaining extension.
"""
from .horizontal import bp_smooth, bp_smooth_track, bp_paint
from .experiments import bp_vs_deadband_experiment

__all__ = ["bp_smooth", "bp_smooth_track", "bp_paint", "bp_vs_deadband_experiment"]
