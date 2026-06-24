"""tslai — Tree-Sequence Local Ancestry Inference.

Soft, calibrated local ancestry along haplotypes from an inferred tree sequence,
via an ancestry CTMC fit by edge-blocked, span-weighted EM. See CLAUDE.md for the
authoritative spec.
"""
from __future__ import annotations

from .branch_stats import branch_expected_stats, vanloan_integral
from .model import (
    make_generator_2state,
    validate_generator,
    transition_matrix,
    stationary_distribution,
    tip_emission,
    query_emission,
)
from .sim import (
    simulate_admixture,
    local_ancestry_truth,
    admixture_demography,
    SOURCE_A,
    SOURCE_B,
    ADMIXED,
    ANCESTRAL,
)
from .diagnostics import persistence_summary, node_persistence, edge_span_summary
from .pruning import prune_tree, prune_root, PruneResult
from .accumulate import accumulate_sufficient_statistics, SuffStats
from .em import m_step_Q, m_step_pi, m_step_w, fit, FitResult
from .output import (posterior_table, missing_info_mask, posterior_at, hard_segments,
                     Segment, INFORMATIVE, MISSING_INFO)
from .validate import (map_truth, per_base_accuracy, balanced_accuracy,
                       mean_confidence, reliability_curve, breakpoint_flicker,
                       tract_boundary_error, breakpoint_precision_recall, switch_density)
from .experiments import (admixture_experiment, flicker_vs_true_boundaries, age_sweep,
                          scaling_sweep, arg_ensemble_experiment, singer_ensemble_experiment,
                          fragmentation_experiment)
from .ensemble import merge_posterior_tables, MergedSegment
from .compare import tslai_paint, nearest_reference_paint, head_to_head, score_painter
from .ranked import ranked_tree_sequence
from .io_tsinfer import add_mutations, infer_tree_sequence
from .io_singer import singer_tree_sequences, write_haploid_vcf
from .io_rfmix import rfmix_paint, run_rfmix
from .bp import bp_paint, bp_smooth, bp_smooth_track

try:  # version is best-effort; not required for use
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("tslai")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "branch_expected_stats",
    "vanloan_integral",
    "make_generator_2state",
    "validate_generator",
    "transition_matrix",
    "stationary_distribution",
    "tip_emission",
    "query_emission",
    "simulate_admixture",
    "local_ancestry_truth",
    "admixture_demography",
    "persistence_summary",
    "node_persistence",
    "edge_span_summary",
    "prune_tree",
    "prune_root",
    "PruneResult",
    "accumulate_sufficient_statistics",
    "SuffStats",
    "m_step_Q",
    "m_step_pi",
    "m_step_w",
    "fit",
    "FitResult",
    "posterior_table",
    "missing_info_mask",
    "posterior_at",
    "hard_segments",
    "Segment",
    "INFORMATIVE",
    "MISSING_INFO",
    "map_truth",
    "per_base_accuracy",
    "balanced_accuracy",
    "mean_confidence",
    "reliability_curve",
    "breakpoint_flicker",
    "tract_boundary_error",
    "breakpoint_precision_recall",
    "switch_density",
    "admixture_experiment",
    "flicker_vs_true_boundaries",
    "age_sweep",
    "scaling_sweep",
    "arg_ensemble_experiment",
    "singer_ensemble_experiment",
    "fragmentation_experiment",
    "merge_posterior_tables",
    "MergedSegment",
    "tslai_paint",
    "nearest_reference_paint",
    "head_to_head",
    "score_painter",
    "ranked_tree_sequence",
    "add_mutations",
    "infer_tree_sequence",
    "singer_tree_sequences",
    "write_haploid_vcf",
    "rfmix_paint",
    "run_rfmix",
    "bp_paint",
    "bp_smooth",
    "bp_smooth_track",
    "SOURCE_A",
    "SOURCE_B",
    "ADMIXED",
    "ANCESTRAL",
    "__version__",
]
