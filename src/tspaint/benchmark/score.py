"""Score benchmark paintings against a truth table — the head-to-head leaderboard (CLAUDE.md §9).

Given a ``truth.npz`` (from :func:`tspaint.benchmark.export.export_vcf` or ``tspaint simulate
--truth``) and one or more tool painting ``.npz`` files (any source that wrote the
``tspaint-painting`` format, including the native painter), reports per tool:

* **balanced accuracy** — class-balanced per-base correctness (``tspaint.validate``);
* **accuracy** — plain span-weighted per-base correctness;
* **confidence** — mean ``|2·P − 1|`` (a calibrated soft caller relaxes toward 0 where it cannot
  tell, so high accuracy with *moderate* confidence is the honest regime; CLAUDE.md §9);
* **switch-density ratio** — inferred ÷ true ancestry switches/length at a confidence ``deadband``
  (1.0 = faithful tract-length distribution for admixture dating; >1 over-fragments).
"""
from __future__ import annotations

import numpy as np

from ..output import hard_segments
from ..serialize import load_painting
from ..validate import balanced_accuracy, per_base_accuracy, mean_confidence

__all__ = ["load_truth", "score", "format_table"]


def load_truth(path):
    """Load a ``tspaint-truth`` ``.npz`` into ``{sample: [(left, right, state)]}``."""
    with np.load(path, allow_pickle=False) as d:
        if str(d.get("_format", "")) != "tspaint-truth":
            raise ValueError(f"{path} is not a tspaint-truth file")
        truth = {}
        for s, l, r, st in zip(d["sample"], d["left"], d["right"], d["state"]):
            truth.setdefault(int(s), []).append((float(l), float(r), int(st)))
    for segs in truth.values():
        segs.sort()
    return truth


def _n_switches(segs):
    """Number of ancestry-state changes in a ``[(left, right, state)]`` list."""
    return sum(1 for a, b in zip(segs, segs[1:]) if a[2] != b[2])


def _switch_ratio(tracks, truth, samples, deadband):
    """Total inferred ÷ total true ancestry switches over ``samples``."""
    inf = sum(_n_switches(hard_segments(tracks[s], deadband=deadband)) for s in samples)
    tru = sum(_n_switches(truth[s]) for s in samples)
    return inf / tru if tru > 0 else float("nan")


def score(truth, paintings, *, deadband=0.4, K=2):
    """Score one or more tool paintings against ``truth``.

    Parameters
    ----------
    truth : str or dict
        A ``tspaint-truth`` ``.npz`` path or an already-loaded ``{sample: [(l, r, state)]}``.
    paintings : dict[str, str] or iterable[str]
        Tool paintings: ``{name: npz_path}``, or paths (named by basename).
    deadband : float, optional
        Confidence dead-band for the hard segmentation used in the switch-density ratio
        (default 0.4; CLAUDE.md §9 — argmax-tspaint over-fragments, a deadband recovers the
        true tract-length distribution).
    K : int, optional
        Number of ancestry states (default 2).

    Returns
    -------
    list[dict]
        One row per painting with ``name``, ``balanced_accuracy``, ``accuracy``,
        ``confidence``, ``switch_density_ratio`` and ``n_samples``.
    """
    if isinstance(truth, str):
        truth = load_truth(truth)
    if not isinstance(paintings, dict):
        import os
        paintings = {os.path.basename(p): p for p in paintings}

    rows = []
    for name, path in paintings.items():
        tracks = load_painting(path)
        samples = sorted(set(truth) & set(k for k in tracks if tracks[k]))
        if not samples:
            rows.append(dict(name=name, balanced_accuracy=float("nan"), accuracy=float("nan"),
                             confidence=float("nan"), switch_density_ratio=float("nan"),
                             n_samples=0))
            continue
        rows.append(dict(
            name=name,
            balanced_accuracy=balanced_accuracy(tracks, truth, samples=samples, K=K),
            accuracy=per_base_accuracy(tracks, truth, samples=samples),
            confidence=mean_confidence(tracks, samples=samples),
            switch_density_ratio=_switch_ratio(tracks, truth, samples, deadband),
            n_samples=len(samples),
        ))
    return rows


def format_table(rows):
    """Render :func:`score` rows as a fixed-width text table."""
    # (key, header, width, decimals|None) — None decimals ⇒ left-aligned string / integer.
    cols = [("name", "tool", 16, None), ("balanced_accuracy", "bal-acc", 8, 3),
            ("accuracy", "acc", 7, 3), ("confidence", "conf", 7, 3),
            ("switch_density_ratio", "sw-ratio", 9, 3), ("n_samples", "n", 4, 0)]

    def cell(val, width, dec):
        if dec is None:
            return f"{str(val):<{width}}"
        if isinstance(val, float) and np.isnan(val):
            return f"{'nan':>{width}}"
        return f"{val:>{width}.{dec}f}" if dec else f"{int(val):>{width}d}"

    lines = ["  ".join(f"{h:<{w}}" if d is None else f"{h:>{w}}" for _k, h, w, d in cols)]
    for r in rows:
        lines.append("  ".join(cell(r[k], w, d) for k, _h, w, d in cols))
    return "\n".join(lines)
