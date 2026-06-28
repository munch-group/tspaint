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

import csv
import json
import os

import numpy as np

from ..output import hard_segments
from ..serialize import load_painting
from ..validate import (balanced_accuracy, per_base_accuracy, mean_confidence, global_proportion,
                        true_proportion, accuracy_by_segment_size, breakpoint_precision_recall,
                        DEFAULT_SIZE_BINS)

__all__ = ["load_truth", "score", "format_table", "score_full", "write_metrics", "aggregate"]


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


# --- full per-painting metrics (proportions / fragmentation / accuracy-by-size) -------------

def _sequence_length(truth, painting_meta):
    """Genome length L for switch-density normalisation: stored seqlen, else max truth right."""
    if painting_meta and painting_meta.get("seqlen"):
        return float(painting_meta["seqlen"])
    return max((r for segs in truth.values() for (_l, r, _s) in segs), default=0.0)


def score_full(truth, painting, *, name="", meta=None, bins=None, deadband=0.4, state=0, K=2):
    """Full benchmark metrics for one painting vs ``truth`` — the three families (CLAUDE.md §9).

    Parameters
    ----------
    truth : str or dict
        A ``tspaint-truth`` ``.npz`` path or a loaded ``{sample: [(l, r, state)]}``.
    painting : str
        Path to a ``tspaint-painting`` ``.npz``.
    name : str, optional
        Painter label stored in the result.
    meta : dict, optional
        Scenario metadata (model, T_split, …) stored verbatim for aggregation.
    bins : array_like, optional
        True-segment-length bin edges (default :data:`tspaint.validate.DEFAULT_SIZE_BINS`).
    deadband : float, optional
        Confidence dead-band for the hard segmentation in the fragmentation metric (default 0.4).
    state : int, optional
        Ancestry state whose global proportion is reported (default 0).
    K : int, optional
        Number of ancestry states (default 2).

    Returns
    -------
    dict
        ``name``, ``meta``, ``n_samples``; **overall proportions** (true/est/error);
        **accuracy** (balanced/per-base/confidence); **fragmentation** (true & inferred switches
        per Mb, ratio, breakpoint precision/recall); and **accuracy by true segment size**
        (``size_edges`` / ``size_accuracy`` / ``size_weight`` / ``size_n_segments``).
    """
    if isinstance(truth, str):
        truth = load_truth(truth)
    from ..serialize import load_painting_meta
    tracks = load_painting(painting)
    pmeta = load_painting_meta(painting)
    samples = sorted(set(truth) & {k for k in tracks if tracks[k]})
    L = _sequence_length(truth, pmeta)
    bins = DEFAULT_SIZE_BINS if bins is None else np.asarray(bins, float)

    out = {"name": name, "meta": dict(meta or {}), "n_samples": len(samples)}
    if not samples:
        return {**out, "proportion_true": float("nan"), "proportion_est": float("nan"),
                "proportion_error": float("nan"), "balanced_accuracy": float("nan"),
                "accuracy": float("nan"), "confidence": float("nan"),
                "switch_true_per_mb": float("nan"), "switch_inferred_per_mb": float("nan"),
                "switch_ratio": float("nan"), "bp_precision": float("nan"),
                "bp_recall": float("nan"), "size_edges": bins.tolist(),
                "size_accuracy": [], "size_weight": [], "size_n_segments": []}

    p_true = true_proportion(truth, state=state, samples=samples)
    p_est = global_proportion(tracks, state=state, samples=samples)

    tol = 0.005 * L if L > 0 else 0.0
    inf_sw = tru_sw = 0
    precs, recs = [], []
    for s in samples:
        hard = hard_segments(tracks[s], deadband=deadband)
        inf_sw += _n_switches(hard)
        tru_sw += _n_switches(truth[s])
        pr = breakpoint_precision_recall(hard, truth[s], tol)
        if not np.isnan(pr["precision"]):
            precs.append(pr["precision"])
        if not np.isnan(pr["recall"]):
            recs.append(pr["recall"])
    span_mb = len(samples) * L / 1e6 if L > 0 else float("nan")
    size = accuracy_by_segment_size(tracks, truth, bins=bins, samples=samples)

    return {
        **out,
        "proportion_true": p_true, "proportion_est": p_est, "proportion_error": p_est - p_true,
        "balanced_accuracy": balanced_accuracy(tracks, truth, samples=samples, K=K),
        "accuracy": per_base_accuracy(tracks, truth, samples=samples),
        "confidence": mean_confidence(tracks, samples=samples),
        "switch_true_per_mb": tru_sw / span_mb if span_mb else float("nan"),
        "switch_inferred_per_mb": inf_sw / span_mb if span_mb else float("nan"),
        "switch_ratio": inf_sw / tru_sw if tru_sw > 0 else float("nan"),
        "bp_precision": float(np.mean(precs)) if precs else float("nan"),
        "bp_recall": float(np.mean(recs)) if recs else float("nan"),
        "size_edges": size["edges"].tolist(),
        "size_accuracy": size["accuracy"].tolist(),
        "size_weight": size["weight"].tolist(),
        "size_n_segments": size["n_segments"].tolist(),
    }


def write_metrics(path, result):
    """Write a :func:`score_full` result to ``path`` as JSON (NaN → null)."""
    def clean(x):
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    with open(path, "w") as f:
        json.dump(clean(result), f, indent=2)


_SCALAR_FIELDS = ["proportion_true", "proportion_est", "proportion_error", "balanced_accuracy",
                  "accuracy", "confidence", "switch_true_per_mb", "switch_inferred_per_mb",
                  "switch_ratio", "bp_precision", "bp_recall", "n_samples"]


def aggregate(json_paths, out_dir):
    """Collect :func:`score_full` JSONs into two tidy CSVs in ``out_dir``.

    Writes ``summary_scalar.csv`` (one row per painting: scenario ``meta`` columns + painter
    ``name`` + the scalar metrics) and ``summary_by_size.csv`` (long form: one row per painting ×
    true-segment-length bin, with the bin's accuracy / weight / segment count) — ready for plotting
    the three metric families across the grid.

    Returns the two CSV paths.
    """
    results = []
    for p in json_paths:
        with open(p) as f:
            results.append(json.load(f))
    meta_keys = sorted({k for r in results for k in (r.get("meta") or {})})
    os.makedirs(out_dir, exist_ok=True)
    scalar_csv = os.path.join(out_dir, "summary_scalar.csv")
    size_csv = os.path.join(out_dir, "summary_by_size.csv")

    with open(scalar_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(meta_keys + ["painter"] + _SCALAR_FIELDS)
        for r in results:
            m = r.get("meta") or {}
            w.writerow([m.get(k, "") for k in meta_keys] + [r.get("name", "")]
                       + [r.get(k) for k in _SCALAR_FIELDS])

    with open(size_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(meta_keys + ["painter", "size_lo", "size_hi", "accuracy", "weight", "n_segments"])
        for r in results:
            m = r.get("meta") or {}
            edges = r.get("size_edges") or []
            acc, wt, ns = r.get("size_accuracy") or [], r.get("size_weight") or [], \
                r.get("size_n_segments") or []
            for i in range(len(acc)):
                w.writerow([m.get(k, "") for k in meta_keys] + [r.get("name", ""),
                           edges[i], edges[i + 1], acc[i], wt[i], ns[i]])
    return scalar_csv, size_csv
