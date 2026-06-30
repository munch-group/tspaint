"""SINGER posterior-ARG front end (CLAUDE.md §7.4).

SINGER (Deng et al., 2024; bioRxiv 10.1101/2024.03.16.585351) is a Bayesian MCMC method
that **samples ARGs from the posterior** under the SMC, as tskit tree sequences. Those
thinned posterior samples are the ideal input to the ensemble merge layer
(:func:`tspaint.ensemble.merge_posterior_tables`): unlike a point estimate (tsinfer/Relate)
they represent ARG *uncertainty*, so averaging the per-member paintings genuinely
marginalises it (the §9 binding constraint).

SINGER is an external binary (built per platform). The path defaults to the env var
``TSPAINT_SINGER`` or a known location; override via ``singer_bin``. This module imports
nothing heavy at load time, so the core package never requires SINGER.

Pipeline (validated): haploid VCF (one column per sample node) -> ``singer -ploidy 1``
(with an auto-retry loop à la SINGER's ``singer_master``) -> raw node/branch/mut text
tables -> tskit tree sequence. **Sample order is preserved** (VCF column i -> sample i),
so per-sample labels/truth transfer by id. Node times are in generations.

Note: SINGER's own ``convert_to_tskit`` omits ``compute_mutation_parents()`` and crashes
on recurrent mutations; the conversion here includes it.
"""
from __future__ import annotations

import glob
import os
import random
import subprocess
import sys
import tempfile
import warnings

import numpy as np

__all__ = ["singer", "write_haploid_vcf", "singer_tree_sequences",
           "singer_window", "build_merge_table", "run_merge_arg",
           "singer_install_dir", "singer_binary_path", "repo_root"]


def repo_root():
    """Locate the tspaint repo root (which holds ``external/`` and the build recipes).

    Robust to how the package is installed: an **editable** install resolves it from this file
    (cwd-independent); a **non-editable** copy in ``site-packages`` (which a ``pip install .`` can
    create, clobbering the editable ``.pth``) instead resolves it from the current working
    directory — so ``tspaint install singer`` / ``benchmark setup``, run from the repo, still find
    ``external/`` rather than a bogus path inside the env. Searches up for the ``external/tools.ini``
    marker, falling back to the legacy ``<this>/../../..`` guess.
    """
    marker = os.path.join("external", "tools.ini")
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        d = os.path.abspath(start)
        while True:
            if os.path.exists(os.path.join(d, marker)):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tools_dir():
    """Clone root for external tools (``$TSPAINT_TOOLS_DIR`` or ``<repo>/external``)."""
    return os.path.expanduser(
        os.environ.get("TSPAINT_TOOLS_DIR", os.path.join(repo_root(), "external")))


def singer_install_dir():
    """Clone root that ``tspaint install singer`` builds into (``<tools-dir>/SINGER``)."""
    return os.path.join(_tools_dir(), "SINGER")


def singer_binary_path():
    """Path to the ``singer`` binary built by ``tspaint install singer``."""
    return os.path.join(singer_install_dir(), "SINGER", "SINGER", "singer")


#: SINGER binary: ``$TSPAINT_SINGER`` if set, else the ``tspaint install singer`` build location.
DEFAULT_SINGER = os.environ.get("TSPAINT_SINGER") or singer_binary_path()
#: SINGER's ``merge_ARG.py`` helper, at the same install location unless ``$TSPAINT_MERGE_ARG``.
DEFAULT_MERGE_ARG = os.environ.get("TSPAINT_MERGE_ARG") or os.path.join(
    singer_install_dir(), "SINGER", "SINGER", "merge_ARG.py")


def write_haploid_vcf(ts, path):
    """Write one haploid VCF column per sample node.

    Drops diploid individuals so SINGER, run with ``-ploidy 1``, sees each
    haplotype as its own sample in node order. Positions are mapped to distinct
    1-based integers.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence with variant sites.
    path : str
        Destination path for the VCF.
    """
    tables = ts.dump_tables()
    tables.nodes.set_columns(flags=tables.nodes.flags, time=tables.nodes.time,
                             population=tables.nodes.population)   # individual -> -1
    tables.individuals.clear()
    ts_hap = tables.tree_sequence()
    with open(path, "w") as f:
        ts_hap.write_vcf(f, ploidy=1,
                         position_transform=lambda x: 1 + np.floor(x).astype(int))


def _read_singer_arg(node_file, branch_file, mut_file=None):
    """Build a tskit tree sequence from SINGER's raw node/branch/mut text tables.

    Port of SINGER's ``convert_to_tskit`` ``read_ARG``, adding the missing
    ``compute_mutation_parents()`` so recurrent mutations don't crash the build.

    Parameters
    ----------
    node_file : str
        Path to the node-times text table.
    branch_file : str
        Path to the branch (edge) text table.
    mut_file : str, optional
        Path to the mutations text table; if omitted, no sites are added.

    Returns
    -------
    tskit.TreeSequence
        The reconstructed tree sequence (node times in generations).
    """
    import tskit
    node_time = np.loadtxt(node_file)
    edge = np.loadtxt(branch_file)
    edge = edge[edge[:, 2] >= 0, :]
    tables = tskit.TableCollection(sequence_length=float(max(edge[:, 1])))
    prev = -1.0
    for tm in node_time:
        if tm == 0:
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
        else:
            tm = max(prev + 1e-4, float(tm))
            tables.nodes.add_row(time=tm)
            prev = tm
    tables.edges.set_columns(left=edge[:, 0], right=edge[:, 1],
                             parent=np.array(edge[:, 2], dtype=np.int32),
                             child=np.array(edge[:, 3], dtype=np.int32))
    if mut_file:
        muts = np.loadtxt(mut_file)
        if muts.ndim == 1:
            muts = muts.reshape(1, -1)
        mp = -1.0
        for i in range(muts.shape[0]):
            if muts[i, 0] != mp and muts[i, 0] < 1e6:
                tables.sites.add_row(position=muts[i, 0], ancestral_state='0')
                mp = muts[i, 0]
            tables.mutations.add_row(site=tables.sites.num_rows - 1, node=int(muts[i, 1]),
                                     derived_state=str(int(muts[i, 3])))
    tables.sort()
    if mut_file and tables.mutations.num_rows:
        tables.build_index()
        tables.compute_mutation_parents()
    return tables.tree_sequence()


def _write_singer_vcf(source, prefix, sequence_length=None):
    """Write the haploid VCF SINGER reads as ``prefix.vcf``; return the sequence length."""
    from .io_genotypes import source_kind, resolve_variants
    from .io_genotypes import write_haploid_vcf as _write_variants_vcf
    if source_kind(source) == "ts":
        write_haploid_vcf(source, prefix + ".vcf")
        return int(source.sequence_length)
    v = resolve_variants(source)
    _write_variants_vcf(v, prefix + ".vcf")
    return int(sequence_length or v.sequence_length)


def _singer_indices(out_prefix):
    """The MCMC sample indices SINGER wrote at ``{out_prefix}_nodes_<i>.txt`` (sorted)."""
    return sorted(int(f.split("_nodes_")[1].split(".txt")[0])
                  for f in glob.glob(out_prefix + "_nodes_*.txt"))


def _run_singer(prefix, out, *, start, end, Ne, mutation_rate, recombination_rate, n_samples,
                thin, ploidy=1, seed=42, polar=0.5, max_retries=50, singer_bin=None):
    """Invoke the bare SINGER binary on ``[start, end)`` with the ``singer_master`` retry loop.

    Reads ``prefix.vcf``, writes ``{out}_nodes_<i>.txt`` / ``_branches_<i>.txt`` / ``_muts_<i>.txt``;
    returns the indices written. On a nonzero exit it re-invokes ``-debug`` with fresh seeds.
    """
    singer_bin = singer_bin or DEFAULT_SINGER
    if not os.path.exists(singer_bin):
        raise FileNotFoundError(
            f"SINGER binary not found at {singer_bin}; set TSPAINT_SINGER or pass singer_bin")
    base = [singer_bin, "-Ne", str(Ne), "-m", str(mutation_rate), "-r", str(recombination_rate),
            "-ploidy", str(ploidy), "-input", prefix, "-output", out,
            "-start", str(int(start)), "-end", str(int(end)), "-polar", str(polar),
            "-n", str(n_samples), "-thin", str(thin)]
    rng = random.Random(seed)
    seeds = [seed] + [rng.randint(0, 2 ** 30 - 1) for _ in range(max_retries)]
    last = None
    for k, sd in enumerate(seeds):
        cmd = base + (["-debug"] if k > 0 else []) + ["-seed", str(sd)]
        last = subprocess.run(cmd, capture_output=True, text=True)
        if last.returncode == 0:
            break
    else:
        raise RuntimeError(f"SINGER failed after {max_retries} retries:\n{(last.stderr or '')[-1000:]}")
    return _singer_indices(out)


def singer(source, *, Ne, mutation_rate, recombination_rate, n_samples=21,
           thin=10, burn_in=20, seed=42, ploidy=1, workdir=None,
           singer_bin=None, with_mutations=True, max_retries=50, sequence_length=None):
    """Sample posterior ARGs from genotypes via SINGER (CLAUDE.md §7.4).

    SINGER's MCMC samples ARGs from ``P(ARG | genotypes)``; the thinned post-burn-in
    samples are the ideal input to :func:`tspaint.ensemble.merge_posterior_tables`,
    since they represent genuine ARG uncertainty (§7.4).

    Parameters
    ----------
    source : tskit.TreeSequence or str
        The genotypes to sample ARGs from — a tree sequence carrying mutations, a **VCF Zarr**
        store, or a **VCF** file (normalised by :mod:`tspaint.io_genotypes`). All are written out
        as a haploid VCF for SINGER.
    Ne : float
        Effective population size passed to SINGER (``-Ne``).
    mutation_rate : float
        Per-base mutation rate (``-m``).
    recombination_rate : float
        Per-base recombination rate (``-r``).
    n_samples : int, optional
        Number of MCMC samples SINGER draws (``-n``, default 20).
    thin : int, optional
        MCMC thinning interval (``-thin``, default 10).
    burn_in : int, optional
        Number of leading samples to discard as burn-in (default 5).
    seed : int, optional
        Base random seed; retries derive fresh seeds from it (default 42).
    ploidy : int, optional
        Ploidy passed to SINGER (``-ploidy``, default 1).
    workdir : str, optional
        Working directory for VCF and ARG text tables (default: a fresh tempdir).
    singer_bin : str, optional
        Path to the SINGER binary (default: ``DEFAULT_SINGER`` / ``TSPAINT_SINGER``).
    with_mutations : bool, optional
        Whether to read mutations into the returned tree sequences (default True).
    max_retries : int, optional
        Maximum ``-debug`` re-invocations with fresh seeds on failure (default 50).
    sequence_length : float, optional
        Override the sequence length (SINGER ``-end``) for VCF / Zarr sources; defaults to the
        max variant position. Ignored for a ``ts`` source (its ``sequence_length`` is used).

    Returns
    -------
    list of tskit.TreeSequence
        Post-burn-in posterior samples (one per thinned MCMC sample), with sample
        order preserved (VCF column ``i`` -> sample ``i``).

    Raises
    ------
    FileNotFoundError
        If the SINGER binary is absent at ``singer_bin``.
    RuntimeError
        If SINGER still exits nonzero after ``max_retries`` retries.

    Notes
    -----
    Mirrors SINGER's ``singer_master`` retry loop: on a nonzero exit it re-invokes
    ``-debug`` with fresh seeds.
    """
    tmp = workdir or tempfile.mkdtemp(prefix="tspaint_singer_")
    os.makedirs(tmp, exist_ok=True)
    prefix = os.path.join(tmp, "data")
    L = _write_singer_vcf(source, prefix, sequence_length)
    out = os.path.join(tmp, "arg")
    _run_singer(prefix, out, start=0, end=L, Ne=Ne, mutation_rate=mutation_rate,
                recombination_rate=recombination_rate, n_samples=n_samples, thin=thin,
                ploidy=ploidy, seed=seed, max_retries=max_retries, singer_bin=singer_bin)

    samples = []
    for i in _singer_indices(out):
        if i < burn_in:
            continue
        mf = f"{out}_muts_{i}.txt" if with_mutations else None
        samples.append(_read_singer_arg(f"{out}_nodes_{i}.txt", f"{out}_branches_{i}.txt", mf))
    if len(samples) == 1:
        return samples[0]
    return samples


def singer_window(source, *, start, end, out_prefix, Ne, mutation_rate, recombination_rate,
                  n_samples=20, thin=10, ploidy=1, seed=42, polar=0.5, max_retries=50,
                  singer_bin=None):
    """Run SINGER on ONE genomic window ``[start, end)``, leaving the raw node/branch/mut tables.

    The per-window unit of a GWF window × member parallelisation: run this for each window, then
    stitch the per-window tables for each MCMC member into a chromosome-length ARG with
    :func:`build_merge_table` + :func:`run_merge_arg`. Uses the **bare SINGER binary** (not
    ``singer_master``, whose ``-m`` rate path is broken in the bundled copy — an ``arggs.ploidy``
    typo and a dead ``-mut_map`` branch).

    Parameters
    ----------
    source : tskit.TreeSequence or str
        Genotypes: a tree sequence (a haploid VCF is written next to ``out_prefix``), or a VCF
        path (used directly — SINGER reads ``<prefix>.vcf``; no rewrite, so a GWF grid shares it).
    start, end : float
        Window bounds passed to SINGER ``-start`` / ``-end``.
    out_prefix : str
        Output prefix; SINGER writes ``{out_prefix}_nodes_<i>.txt`` etc.
    Ne, mutation_rate, recombination_rate, n_samples, thin, ploidy, seed, polar, max_retries, singer_bin
        As for :func:`singer` / SINGER's flags.

    Returns
    -------
    list[int]
        The MCMC sample indices written for this window.
    """
    from .io_genotypes import source_kind
    if source_kind(source) == "vcf":
        s = str(source)
        vcf_prefix = s[:-4] if s.endswith(".vcf") else s
    else:
        vcf_prefix = out_prefix + "_input"
        _write_singer_vcf(source, vcf_prefix)
    return _run_singer(vcf_prefix, out_prefix, start=start, end=end, Ne=Ne,
                       mutation_rate=mutation_rate, recombination_rate=recombination_rate,
                       n_samples=n_samples, thin=thin, ploidy=ploidy, seed=seed, polar=polar,
                       max_retries=max_retries, singer_bin=singer_bin)


def build_merge_table(windows, member, *, skip_gaps=None, coords="local"):
    """Build the ``merge_ARG.py`` file-table rows for one MCMC ``member`` across ``windows``.

    Parameters
    ----------
    windows : iterable of (window_index, start, end, out_prefix)
        One entry per genomic window (as produced by the :func:`singer_window` jobs).
    member : int
        The MCMC sample index to stitch (the ``<i>`` in ``{out_prefix}_nodes_<i>.txt``).
    skip_gaps : list[tuple[float, float]], optional
        ``(lo, hi)`` regions to skip (e.g. a centromere) — windows overlapping any are dropped.
    coords : {"local", "absolute"}
        ``merge_ARG.py`` adds the 4th column to every coordinate, so for SINGER's window-local
        outputs the block coordinate is the window **start** (``"local"``, the default). Use
        ``"absolute"`` (block coordinate 0) only if your SINGER emits absolute coordinates.

    Returns
    -------
    list[tuple[str, str, str, float]]
        ``(nodes_file, branches_file, muts_file, block_coordinate)`` rows in genome order.
    """
    gaps = skip_gaps or []
    rows = []
    for (_widx, s, e, prefix) in sorted(windows, key=lambda r: float(r[1])):
        s, e = float(s), float(e)
        if any(not (e <= lo or s >= hi) for (lo, hi) in gaps):    # overlaps a skip gap
            continue
        block = s if coords == "local" else 0.0
        rows.append((f"{prefix}_nodes_{member}.txt", f"{prefix}_branches_{member}.txt",
                     f"{prefix}_muts_{member}.txt", block))
    return rows


def run_merge_arg(rows, out, *, script=None, python=None):
    """Stitch per-window ARG tables into ``out`` by shelling out to SINGER's ``merge_ARG.py``.

    ``rows`` come from :func:`build_merge_table`. ``script`` defaults to ``DEFAULT_MERGE_ARG``
    (env ``TSPAINT_MERGE_ARG``); ``python`` to the current interpreter — note ``merge_ARG.py``
    imports ``tszip``, so that interpreter needs ``tskit + numpy + tszip``.
    """
    script = script or DEFAULT_MERGE_ARG
    python = python or sys.executable
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"merge_ARG.py not found at {script}; set TSPAINT_MERGE_ARG or pass script")
    fd, table = tempfile.mkstemp(suffix="_file_table.txt")
    os.close(fd)
    try:
        with open(table, "w") as f:
            for (n, b, m, blk) in rows:
                f.write(f"{n} {b} {m} {int(blk)}\n")
        r = subprocess.run([python, script, "--file_table", table, "--output", out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"merge_ARG.py failed:\n{(r.stderr or '')[-2000:]}")
    finally:
        os.remove(table)
    return out


def singer_tree_sequences(ts, **kwargs):
    """Deprecated alias for :func:`singer`."""
    warnings.warn("tspaint.io.singer_tree_sequences is deprecated; use tspaint.io.singer",
                  DeprecationWarning, stacklevel=2)
    return singer(ts, **kwargs)
