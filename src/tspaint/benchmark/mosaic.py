"""MOSAIC benchmark runner — VCF in, tspaint ``.npz`` painting out (CLAUDE.md §9; papers/mosaic.md).

MOSAIC (Salter-Townshend & Myers 2019, *Genetics* 212:869–889) is the strongest **genotype-based**
comparator in the stack: a two-layer (nested) HMM — an outer local-ancestry chain over an inner
Li–Stephens copying model — that beats RFMix, ELAI and LAMP-LD on three-way admixture *even when
those methods are handed the panel↔ancestry correspondence and MOSAIC is not*, and is best-in-class
on small panels. It is also the direct ancestor of GhostBuster, so it belongs in any head-to-head we
publish.

**Its latent ancestries are not our reference states.** That is the whole point of the method: the
mixing sources are *decoupled* from the observed donor panels, and the relationship between them is
inferred as a copying-probability matrix ``Mu`` (panels × ancestries). So the bridge cannot assume
"ancestry k == state k"; it maps each latent ancestry onto the panel it copies from most
(``argmax_p Mu[p, a]``) — i.e. it uses **MOSAIC's own inferred panel↔ancestry relationship**, which
is the honest reading of its output and never touches the truth table.

MOSAIC is an R package with a bespoke input format (per-population fixed-width genofiles, a 6-column
snpfile, a 3-row rates file), so the bridge writes that layout, drives it through an R script inside
MOSAIC's own pixi env, and reads the gridded posteriors back — mapping MOSAIC's cM grid onto our SNP
positions with its own ``grid_to_pos``. Override with ``TSPAINT_MOSAIC_DIR`` / ``TSPAINT_MOSAIC_CMD``.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import _common as C
from ._msp import tracks_from_marker_posteriors

__all__ = ["mosaic"]


#: R driver: run MOSAIC, then export the gridded local ancestry mapped onto the SNP positions.
#: ``localanc`` is ``(A, 2*NUMI, G)`` on MOSAIC's cM grid; ``grid_to_pos`` maps it to ``(A, 2*NUMI, S)``.
#: R arrays are column-major, so the binary dump reads back with ``order="F"``.
_DRIVER = r'''
args <- commandArgs(trailingOnly = TRUE)
datasource <- args[1]; target <- args[2]; A <- as.integer(args[3]); NUMI <- as.integer(args[4])
chr <- as.integer(args[5]); Ne <- as.numeric(args[6]); GpcM <- as.integer(args[7])
rounds <- as.integer(args[8]); maxcores <- as.integer(args[9])
PHASE <- as.logical(args[10]); EM <- as.logical(args[11])
resultsdir <- args[12]; outbin <- args[13]; outdims <- args[14]
pops <- strsplit(args[15], ",")[[1]]

suppressMessages(library(MOSAIC, lib.loc = file.path(Sys.getenv("MOSAIC_HOME"), "Rlib")))

run_mosaic(target = target, datasource = datasource, chrnos = chr, A = A, NUMI = NUMI,
           pops = pops, PHASE = PHASE, EM = EM, GpcM = GpcM, Ne = Ne, MC = maxcores,
           REPS = rounds, resultsdir = resultsdir, PLOT = FALSE, doFst = FALSE,
           verbose = FALSE, ffpath = tempdir())

# Anchor on .RData: MOSAIC also drops "<target>_<A>way_..._EMlog.out" logs beside the results, and
# those sort *first* (1way < 2way), so a bare "^<target>_" pattern picks up a text file and load()
# dies on it.
la <- list.files(resultsdir, pattern = paste0("^localanc_", target, "_.*\\.RData$"),
                 full.names = TRUE)[1]
mn <- list.files(resultsdir, pattern = paste0("^", target, "_.*\\.RData$"),
                 full.names = TRUE)[1]
if (is.na(la) || is.na(mn)) stop("MOSAIC produced no .RData results in ", resultsdir)
load(la)   # localanc, final.flips, g.loc
load(mn)   # Mu (kLL x A), alpha, ..., kLL

lap <- MOSAIC:::grid_to_pos(localanc, datasource, g.loc, chr)
X <- lap[[1]]                                   # (A, 2*NUMI, S)

# Each latent ancestry -> the donor panel it copies from most. Mu is (panels x ancestries) and its
# panel order is the `pops` order we passed in, so this indexes straight back into our states.
anc2panel <- apply(Mu, 2, which.max)

writeBin(as.numeric(X), outbin, size = 8)
write(c(dim(X), anc2panel), file = outdims, ncolumns = 1)
'''


def mosaic(query_vcf, ref_vcf=None, *, sample_map, chromosome=None, recomb_rate=1e-8,
           ancestries=None, Ne=9e4, gridpoints_per_cm=60, rounds=5, phase=True, em=True,
           maxcores=0, out=None, workdir=None, extra_args=None, log=None):
    """Run MOSAIC on a query VCF and return per-query-haplotype **soft** Segment tracks.

    Parameters
    ----------
    query_vcf, ref_vcf, sample_map, chromosome, out, workdir, log
        As for :func:`tspaint.benchmark.rfmix.rfmix`. **No genetic map argument**: MOSAIC takes its
        own 3-row ``rates`` file, generated uniformly from ``recomb_rate``.
    recomb_rate : float, optional
        Per-base rate for the generated ``rates`` file (default ``1e-8``).
    ancestries : int, optional
        Number of *latent* mixing ancestries to fit (MOSAIC's ``A``). Defaults to the number of
        reference states ``K``. These are **not** the reference panels — see the module docstring.
    Ne : float, optional
        Effective population size (MOSAIC's ``Ne``; default its own ``9e4``, which is a human
        genome-wide value — lower it to match a simulation).
    gridpoints_per_cm : int, optional
        MOSAIC's grid density (``GpcM``; default 60). Its posteriors live on this grid and are
        mapped back onto the SNPs by MOSAIC's own ``grid_to_pos``.
    rounds : int, optional
        Thin → phase → EM rounds (default 5).
    phase : bool, optional
        Let MOSAIC re-phase the target haplotypes (its ``PHASE``; default ``True``, upstream's).

        .. warning::
           Re-phasing can **swap haplotype 0 and 1 within an individual**, which per-haplotype
           scoring will read as error even when the diploid call is right. That is MOSAIC's genuine
           behaviour (and a real strength — it corrects an unbounded number of phase errors, unlike
           RFMix's ≤1-per-window). Pass ``phase=False`` for a strict per-haplotype comparison.
    em : bool, optional
        Estimate model parameters by EM (default ``True``).
    maxcores : int, optional
        MOSAIC's ``MC`` (``0`` → it grabs half the available cores).
    extra_args : iterable[str], optional
        Ignored (MOSAIC is driven through an R script); accepted for signature parity.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype index ``2*j+h``, a **soft** painting over ``[0, L)``.
    """
    workdir = workdir or tempfile.mkdtemp(prefix="tspaint_mosaic_")
    os.makedirs(workdir, exist_ok=True)
    panel = C.resolve_panel(query_vcf, ref_vcf, sample_map=sample_map)

    if not C.tool_available("mosaic"):
        raise FileNotFoundError(
            f"MOSAIC not installed at {C.MOSAIC_DIR} (no Rlib/MOSAIC) — run "
            f"`tspaint benchmark install mosaic`")

    A = int(ancestries or panel.K)
    chrom = 1                                               # MOSAIC keys its files on an integer chr
    data = os.path.join(workdir, "data")
    os.makedirs(data, exist_ok=True)
    _write_mosaic_inputs(data, panel, chrom, recomb_rate)

    pops = [f"P{k}" for k in range(panel.K)]                # donor panel per reference state
    resdir = os.path.join(workdir, "MOSAIC_RESULTS")
    outbin, outdims = os.path.join(workdir, "la.bin"), os.path.join(workdir, "la.dims")
    drv = os.path.join(workdir, "run_mosaic.R")
    with open(drv, "w") as f:
        f.write(_DRIVER)

    args = [drv, data, "TARGET", str(A), str(len(panel.query)), str(chrom), str(float(Ne)),
            str(int(gridpoints_per_cm)), str(int(rounds)), str(int(maxcores)),
            "TRUE" if phase else "FALSE", "TRUE" if em else "FALSE",
            resdir, outbin, outdims, ",".join(pops)]
    C.run_tool("mosaic", args, cwd=workdir, log=log,
               env={"MOSAIC_HOME": C.MOSAIC_DIR, "R_LIBS": os.path.join(C.MOSAIC_DIR, "Rlib")})

    dims = [int(float(x)) for x in open(outdims).read().split()]
    nA, nH, nS = dims[0], dims[1], dims[2]
    anc2panel = dims[3:3 + nA]                              # 1-based index into `pops`
    X = np.fromfile(outbin, dtype=np.float64).reshape((nA, nH, nS), order="F")
    if nS != len(panel.positions):
        raise ValueError(f"MOSAIC returned {nS} sites, panel has {len(panel.positions)}")

    # latent ancestry -> our state, via MOSAIC's own copying matrix (never the truth table)
    anc_state = [int(pops[p - 1][1:]) for p in anc2panel]
    per_hap = {}
    for h, key in enumerate(panel.query_keys):
        if h >= nH:
            break
        post = np.zeros((nS, panel.K))
        for a in range(nA):
            post[:, anc_state[a]] += X[a, h, :]
        s = post.sum(axis=1, keepdims=True)
        per_hap[key] = np.divide(post, s, out=np.full_like(post, 1.0 / panel.K), where=s > 0)

    tracks = tracks_from_marker_posteriors(panel.positions, per_hap, panel.sequence_length,
                                           atol=1e-9)
    C.fill_missing(tracks, panel)
    if out:
        C.save_tracks(out, tracks, panel)
        if log:
            log(f"mosaic: {panel.n_query_haps} query haplotypes -> {out}")
    return tracks


def _write_mosaic_inputs(data, panel, chrom, recomb_rate):
    """Write MOSAIC's bespoke input layout into ``data/`` (README.txt "INPUTS").

    * ``<pop>genofile.<chr>`` — ``#snps`` rows × ``#haps`` columns of ``0``/``1`` **characters with
      no separator** (MOSAIC reads them fixed-width, one column per character).
    * ``TARGETgenofile.<chr>`` — the same, for the admixed targets.
    * ``snpfile.<chr>`` — 6 columns: rsID, chr, cM, bp, allele1, allele2.
    * ``rates.<chr>`` — **3 rows**: number of sites; the positions; the cumulative cM.
    * ``sample.names`` — first column carries the donor population names.
    """
    geno, pos = panel.geno, panel.positions
    cm = pos.astype(float) * recomb_rate * 100.0

    def _geno(path, cols):
        with open(path, "w") as f:
            for s in range(geno.shape[0]):
                f.write("".join(str(int(geno[s, c])) for c in cols) + "\n")

    for k in range(panel.K):
        cols = [c for (_n, (c0, c1), st) in panel.ref for c in (c0, c1) if st == k]
        if not cols:
            raise ValueError(f"reference panel has no haplotypes for ancestry {k}")
        _geno(os.path.join(data, f"P{k}genofile.{chrom}"), cols)
    _geno(os.path.join(data, f"TARGETgenofile.{chrom}"),
          [c for (_n, (c0, c1), _k) in panel.query for c in (c0, c1)])

    with open(os.path.join(data, f"snpfile.{chrom}"), "w") as f:
        for s, p in enumerate(pos):
            ref, alt = panel.alleles[s]
            f.write(f"rs{int(p)} {chrom} {cm[s]:.10f} {int(p)} {ref} {alt}\n")

    with open(os.path.join(data, f"rates.{chrom}"), "w") as f:
        f.write(f"{len(pos)}\n")
        f.write(" ".join(str(int(p)) for p in pos) + "\n")
        f.write(" ".join(f"{c:.10f}" for c in cm) + "\n")

    with open(os.path.join(data, "sample.names"), "w") as f:
        for k in range(panel.K):
            f.write(f"P{k} P{k}\n")
