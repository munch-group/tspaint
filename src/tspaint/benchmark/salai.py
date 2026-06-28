"""SALAI-Net benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9, §10).

SALAI-Net (Oriol Sabat et al., 2022): a reference-based template-matching network + learned
smoother. It is **inference-only** and ships pretrained models (``main_model`` for whole-genome,
``hapmap_model`` for shorter sequences). This runner shells out to ``src/SALAI.py`` and parses its
``predictions.msp.tsv`` **hard** windowed calls, reported as one-hot (0/1) posteriors (SALAI-Net
writes no per-class probabilities; CLAUDE.md "report 0 or 1").

SALAI-Net runs in its own pixi env: ``pixi run --manifest-path ~/SALAI-Net python …``; relocate
with ``TSPAINT_SALAI_DIR`` or replace the launcher with ``TSPAINT_SALAI_CMD``.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import _common as C
from ._msp import parse_msp

__all__ = ["salai"]


def _resolve_model(model):
    """Resolve ``model`` (``None`` / ``"main"`` / ``"hapmap"`` / a ``.pth`` path) to a checkpoint."""
    if model in (None, "main", "hapmap"):
        kind = "hapmap_model" if model == "hapmap" else "main_model"
        return os.path.join(C.SALAI_DIR, "models", kind, "models", "best_model.pth")
    return model


def _code_to_state(out_folder):
    """SALAI codes are indices into ``population_ids.npy`` (sorted labels) → tspaint states."""
    pids = os.path.join(out_folder, "population_ids.npy")
    if not os.path.exists(pids):
        return None
    try:
        labels = [str(x) for x in np.load(pids, allow_pickle=True)]
        return {i: int(l) for i, l in enumerate(labels)}
    except (ValueError, OSError):
        return None


def salai(query_vcf, ref_vcf=None, *, sample_map, model=None, chromosome=None, out=None,
          workdir=None, extra_args=None, log=None):
    """Run SALAI-Net on a query VCF and return per-query-haplotype one-hot Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, chromosome, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix`. SALAI-Net needs **no** genetic map.
    model : str, optional
        Pretrained checkpoint: a ``.pth`` path, or ``"main"`` / ``"hapmap"`` to select a shipped
        model under ``TSPAINT_SALAI_DIR`` (default ``"main"``).
    extra_args : iterable[str], optional
        Extra arguments appended to the SALAI command (e.g. ``["-b", "8"]``).

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, a one-hot (0/1) painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_salai_")
    panel, qv, rv, sm = C.setup_inputs(query_vcf, ref_vcf, sample_map, workdir)
    out_folder = os.path.join(workdir, "salai_out")           # must NOT pre-exist (SALAI requirement)

    mcp = _resolve_model(model)
    if not C.tool_available("salai"):
        raise FileNotFoundError(
            f"SALAI-Net not found at {C.SALAI_DIR}; set TSPAINT_SALAI_DIR or TSPAINT_SALAI_CMD")
    C.require(mcp, "SALAI-Net model checkpoint not found (extract models.tar.gz, or pass model=)")

    args = ["--model-cp", mcp, "--query", qv, "--reference", rv, "--map", sm, "-o", out_folder]
    if extra_args:
        args += list(extra_args)
    C.run_tool("salai", args, cwd=workdir, log=log)

    msp = os.path.join(out_folder, "predictions.msp.tsv")
    tracks = parse_msp(msp, C.parse_inds(panel), panel.K, panel.sequence_length,
                       code_to_state=_code_to_state(out_folder))
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"salai: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
