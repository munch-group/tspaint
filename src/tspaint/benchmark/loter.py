"""Loter benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9; papers/recomb-mix.md).

Loter (Dias-Alves, Mairal & Blum 2018, *MBE* 35:2318–2326) frames LAI as a graph/dynamic-programming
problem and takes **no biological parameters at all** — no genetic map, no admixture time, no
training, no reference-panel tuning. That makes it (a) one of the three methods that beat Recomb-Mix
past ~200 generations, i.e. an opponent in tspaint's own target regime, and (b) the closest
competitor to tspaint's *own* "works on non-model organisms without a genetic map" pitch — which is
exactly why it is worth measuring against rather than conceding to.

It reports **hard calls** (one ancestry per SNP per haplotype), so like Recomb-Mix and SALAI-Net it
is stored as one-hot 0/1 posteriors.

Loter is a Python **library**, not a CLI, and its env is mutually hostile to tspaint's (it needs
numpy < 2 — see external/envs/Loter). So this bridge writes the panel to ``.npy``, runs a small
driver *inside Loter's own pixi env*, and reads the ancestry matrix back. Override the env with
``TSPAINT_LOTER_DIR`` / ``TSPAINT_LOTER_CMD``.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import _common as C
from ._msp import tracks_from_marker_posteriors

__all__ = ["loter"]


#: Driver run inside Loter's env: loads the panel, calls Loter, writes the ancestry matrix.
#: ``loter_smooth`` is Loter's recommended entry point (bagging + a smoothing/vote pass); it takes
#: ``l_H`` (one reference haplotype matrix per ancestry) and ``h_adm`` (the admixed haplotypes),
#: all ``(n_hap, n_snp)`` uint8, and returns ``(n_adm_hap, n_snp)`` ancestry indices.
_DRIVER = '''\
import sys, numpy as np
import loter.locanc.local_ancestry as lc

npz, out, n_boot, threads = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = np.load(npz)
K = int(d["K"])
l_H = [d[f"ref{k}"].astype(np.uint8) for k in range(K)]
h_adm = d["adm"].astype(np.uint8)
res = lc.loter_smooth(l_H=l_H, h_adm=h_adm, nb_bagging=n_boot, num_threads=threads)
np.save(out, np.asarray(res, dtype=np.int16))
'''


def loter(query_vcf, ref_vcf=None, *, sample_map, chromosome=None, n_bagging=20, threads=1,
          out=None, workdir=None, extra_args=None, log=None):
    """Run Loter on a query VCF and return per-query-haplotype one-hot Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, chromosome, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix`. **No genetic map**: Loter needs none, which is
        the whole point of it.
    n_bagging : int, optional
        Bootstrap replicates Loter averages over (its ``nb_bagging``; default 20, upstream's).
    threads : int, optional
        Worker threads (default 1).
    extra_args : iterable[str], optional
        Ignored (Loter is driven through its Python API, not a CLI); accepted so every runner
        shares one signature.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, a one-hot (0/1) painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_loter_")
    os.makedirs(workdir, exist_ok=True)
    panel = C.resolve_panel(query_vcf, ref_vcf, sample_map=sample_map)

    if not C.tool_available("loter"):
        raise FileNotFoundError(
            f"Loter not installed at {C.LOTER_DIR} — run `tspaint benchmark install loter`")

    # geno is (S, H) with query columns first; Loter wants (n_hap, n_snp) per ancestry.
    geno = panel.geno
    arrays = {"K": np.array(panel.K)}
    for k in range(panel.K):
        cols = [c for (_n, (c0, c1), st) in panel.ref for c in (c0, c1) if st == k]
        if not cols:
            raise ValueError(f"reference panel has no haplotypes for ancestry {k}")
        arrays[f"ref{k}"] = geno[:, cols].T.astype(np.uint8)
    qcols = [c for (_n, (c0, c1), _keys) in panel.query for c in (c0, c1)]
    arrays["adm"] = geno[:, qcols].T.astype(np.uint8)

    npz = os.path.join(workdir, "panel.npz")
    res_npy = os.path.join(workdir, "ancestry.npy")
    drv = os.path.join(workdir, "run_loter.py")
    np.savez(npz, **arrays)
    with open(drv, "w") as f:
        f.write(_DRIVER)

    C.run_tool("loter", [drv, npz, res_npy, str(int(n_bagging)), str(int(threads))],
               cwd=workdir, log=log)

    anc = np.load(res_npy)                                  # (n_query_haps, S), ancestry indices
    if anc.shape != (len(qcols), geno.shape[0]):
        raise ValueError(f"loter returned {anc.shape}, expected {(len(qcols), geno.shape[0])}")

    per_hap = {}
    for j, key in enumerate(panel.query_keys):
        per_hap[key] = np.eye(panel.K)[np.clip(anc[j], 0, panel.K - 1)]
    tracks = tracks_from_marker_posteriors(panel.positions, per_hap, panel.sequence_length)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"loter: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks
