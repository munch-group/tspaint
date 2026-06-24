"""tslai — Tree-Sequence Local Ancestry Inference.

Soft, calibrated local ancestry along haplotypes from an inferred tree sequence, via an
ancestry CTMC fit by edge-blocked, span-weighted EM. See CLAUDE.md for the authoritative spec.

Quick start
-----------
>>> import tslai
>>> ts = tslai.simulate_admixture(n_admix=10, n_ref=10)   # or tslai.io.infer_tree_sequence(...)
>>> labels = {0: 0, 1: 0, 2: 1, 3: 1}                     # reference sample-node -> ancestry state
>>> painting = tslai.paint(ts, labels)                    # EM-fit on references, paint the queries
>>> painting.posteriors[q]                                # soft per-position posterior (Segments)
>>> painting.segments(deadband=0.4)                       # hard ancestry tracts (for dating)

Public API
----------
Core
    paint, Painting, fit, FitResult, posterior_table, hard_segments, Segment,
    INFORMATIVE, MISSING_INFO, make_generator_2state
Simulation (examples / benchmarks)
    simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
Namespaces
    tslai.metrics      accuracy, calibration, fragmentation / tract-length metrics
    tslai.compare      painters (tslai_paint, nearest_reference_paint, rfmix_paint) + head_to_head
    tslai.io           input front ends (tsinfer, SINGER)
    tslai.experiments  end-to-end benchmark drivers
Lower-level machinery lives in the named submodules (``tslai.model``, ``tslai.pruning``,
``tslai.accumulate``, ``tslai.em``, ``tslai.output``, ``tslai.ensemble``, ``tslai.ranked``,
``tslai.diagnostics``).
"""
from __future__ import annotations

# Core public API ---------------------------------------------------------------------------
from .api import paint, Painting
from .em import fit, FitResult
from .output import posterior_table, hard_segments, Segment, INFORMATIVE, MISSING_INFO
from .model import make_generator_2state
from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED

# Namespaces (grouped functionality; submodules also importable directly) -------------------
from . import validate as metrics
from . import (  # noqa: F401  (exposed as tslai.<name>)
    compare, io, experiments, sim, model, ensemble, ranked, validate,
    em, output, pruning, accumulate, branch_stats, diagnostics,
    io_tsinfer, io_singer, io_rfmix,
)

try:  # version is best-effort; not required for use
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("tslai")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    # core
    "paint", "Painting", "fit", "FitResult",
    "posterior_table", "hard_segments", "Segment", "INFORMATIVE", "MISSING_INFO",
    "make_generator_2state",
    # simulation
    "simulate_admixture", "local_ancestry_truth", "SOURCE_A", "SOURCE_B", "ADMIXED",
    # namespaces
    "metrics", "compare", "io", "experiments",
    "__version__",
]
