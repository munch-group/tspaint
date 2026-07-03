"""tspaint — Tree-Sequence Local Ancestry Inference.

Soft, calibrated local ancestry along haplotypes from an inferred tree sequence, via an
ancestry CTMC fit by edge-blocked, span-weighted EM. See CLAUDE.md for the authoritative spec.

Quick start
-----------
>>> import tspaint
>>> ts = tspaint.simulate_admixture(n_admix=10, n_ref=10)   # or tspaint.io.tsinfer(...)
>>> labels = {0: 0, 1: 0, 2: 1, 3: 1}                     # reference sample-node -> ancestry state
>>> painting = tspaint.paint(ts, labels)                    # EM-fit on references, paint the queries
>>> painting.posteriors[q]                                # soft per-position posterior (Segments)
>>> painting.segments(deadband=0.4)                       # hard ancestry tracts (for dating)
>>> painting.plot()                                       # per-haplotype figure (soft posterior + hard tracts)

On an inferred (tsinfer / Relate) ARG, add ``smooth=True`` to suppress tree-inference-induced
spurious switches with the horizontal BP smoother (CLAUDE.md §7).

Public API
----------
Core
    paint, Painting, fit, FitResult, posterior_table, hard_segments, Segment,
    INFORMATIVE, MISSING_INFO, make_generator_2state; SegmentTrack / compare_tracks
    (plot any per-sample segments — tspaint or an external tool like RFMix/gnomix — like a Painting)
Simulation (examples / benchmarks)
    simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
Dating (admixture rate through time — an optional, separate deliverable)
    fit_rate_through_time, RateThroughTime; also ``Painting.rate_through_time()``
Reference QC — control panel contamination via soft_refs / masking (Task 1)
    reference_qc (``.soft_refs`` / ``.mask``), foreign_tracts, loo_posterior_table; also
    ``Painting.introgression_map()``
Ghost / archaic introgression search — accurate calibrated segments (Task 2)
    detect_ghost — a depth-emission HMM that learns the ghost (deep/archaic) state with no ghost
    reference; accepts a SINGER ensemble for accuracy. (``detect_archaic`` is a deprecated alias.)
Namespaces
    tspaint.metrics      accuracy, calibration, fragmentation / tract-length metrics
    tspaint.compare      painters (tspaint_paint, nearest_reference_paint, rfmix_paint) + head_to_head
    tspaint.io           input front ends (tsinfer, SINGER)
    tspaint.bp           horizontal BP/EP smoother (helps on inferred ARGs; CLAUDE.md §7)
    tspaint.dating       time-inhomogeneous directional mugration EM (admixture dating)
    tspaint.experiments  end-to-end benchmark drivers
    tspaint.benchmark    run RFMix / gnomix / SALAI-Net / Recomb-Mix from VCF → tspaint .npz; export + score
    tspaint.introgression  reference QC, anonymous foreign tracts (+ the deep ghost flag)
    tspaint.archaic      detect_ghost — reference-free ghost / archaic HMM (depth emission)
Lower-level machinery lives in the named submodules (``tspaint.model``, ``tspaint.pruning``,
``tspaint.accumulate``, ``tspaint.em``, ``tspaint.output``, ``tspaint.ensemble``, ``tspaint.ranked``,
``tspaint.diagnostics``).
"""
from __future__ import annotations

# Core public API ---------------------------------------------------------------------------
from .api import paint, Painting
from .track import SoftTrack, SegmentTrack, compare_tracks
from .em import fit, FitResult
from .output import (posterior_table, loo_posterior_table, hard_segments, Segment,
                     INFORMATIVE, MISSING_INFO)
from .model import make_generator_2state
from .io_genotypes import subset_data
from .sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
from .dating import fit_rate_through_time, RateThroughTime, EnsembleRateThroughTime
from .introgression import reference_qc, foreign_tracts
from .archaic import detect_ghost, GhostResult, detect_archaic

# Namespaces (grouped functionality; submodules also importable directly) -------------------
from . import (  # noqa: F401  (exposed as tspaint.<name>)
    metrics, compare, io, experiments, bp, dating, sim, model, ensemble, ranked, validate,
    em, output, pruning, accumulate, branch_stats, diagnostics, introgression, archaic,
    io_tsinfer, io_singer, io_argweaver, io_relate, io_genotypes, io_rfmix, benchmark,
)

try:  # version is best-effort; not required for use
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("tspaint")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    # core
    "paint", "Painting", "SoftTrack", "SegmentTrack", "compare_tracks", "fit", "FitResult",
    "posterior_table", "hard_segments", "Segment", "INFORMATIVE", "MISSING_INFO",
    "make_generator_2state",
    # data prep (slice / normalise a genotype source before a front end)
    "subset_data",
    # simulation
    "simulate_admixture", "local_ancestry_truth", "SOURCE_A", "SOURCE_B", "ADMIXED",
    # dating (admixture rate through time)
    "fit_rate_through_time", "RateThroughTime", "EnsembleRateThroughTime",
    # Task 1 — reference QC (control panel contamination via soft_refs / masking)
    "reference_qc", "foreign_tracts", "loo_posterior_table",
    # Task 2 — dedicated ghost / archaic introgression search (depth-emission HMM)
    "detect_ghost", "GhostResult", "detect_archaic",
    # namespaces
    "metrics", "compare", "io", "bp", "dating", "experiments", "benchmark", "introgression",
    "archaic", "__version__",
]
