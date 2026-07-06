"""Relate -> tskit front end (CLAUDE.md §5).

The core method is developed and validated on msprime / tsinfer tree sequences
(:mod:`tspaint.sim`), which carry node persistence natively. Relate is the real-data front end
and must be converted with ``--compress`` (load-bearing, CLAUDE.md §5): it assigns the same node
age / id to nodes with identical descendant sets across adjacent trees — the persistence
invariant the edge-blocking depends on.

:func:`relate` wraps ``relate_lib``'s ``Convert`` binary (run Relate itself externally first);
the binary path defaults to env ``TSPAINT_RELATE_CONVERT`` or ``Convert`` on ``PATH``. Run the
:func:`check_persistence` go/no-go (§5.1) on any converted file before inference.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

from .diagnostics import persistence_summary
from .io_singer import _tools_dir            # shared external-tools clone-root resolution

__all__ = ["relate", "relate_convert", "windows", "check_persistence", "convert_relate",
           "relate_install_dir", "relate_binary_path", "relate_file_formats_path",
           "estimate_population_size_path", "relate_lib_install_dir", "relate_convert_path"]


def relate_lib_install_dir():
    """Clone root that ``tspaint install relate`` builds ``relate_lib`` into (``<tools-dir>/relate_lib``)."""
    return os.path.join(_tools_dir(), "relate_lib")


def relate_convert_path():
    """Path to the ``relate_lib`` ``Convert`` binary built by ``tspaint install relate``."""
    return os.path.join(relate_lib_install_dir(), "bin", "Convert")


def relate_install_dir():
    """Clone root that ``tspaint install relate`` builds Relate into (``<tools-dir>/relate``)."""
    return os.path.join(_tools_dir(), "relate")


def relate_binary_path():
    """Path to the ``Relate`` inference binary built by ``tspaint install relate``."""
    return os.path.join(relate_install_dir(), "bin", "Relate")


def relate_file_formats_path():
    """Path to ``RelateFileFormats`` (VCF/haps <-> Relate conversion) built by ``tspaint install relate``."""
    return os.path.join(relate_install_dir(), "bin", "RelateFileFormats")


def estimate_population_size_path():
    """Path to Relate's ``EstimatePopulationSize.sh`` — the genome-wide coalescence-rate / Ne(t) step.

    Run this on a **whole chromosome** (not a small region): it re-estimates the coalescence-rate
    history and re-dates the branch lengths from the full genealogy, so the calibrated times feeding
    :func:`tspaint.paint` reflect genome-scale signal. Split the resulting chromosome-length tree
    sequence into paint-sized pieces with :func:`windows`.
    """
    return os.path.join(relate_install_dir(), "scripts", "EstimatePopulationSize",
                        "EstimatePopulationSize.sh")


def _default_convert():
    """Resolve the ``Convert`` binary: ``$TSPAINT_RELATE_CONVERT``, else the ``tspaint install
    relate`` build location if present, else ``Convert`` on ``PATH``."""
    env = os.environ.get("TSPAINT_RELATE_CONVERT")
    if env:
        return env
    built = relate_convert_path()
    return built if os.path.exists(built) else "Convert"


#: ``relate_lib`` ``Convert`` binary: ``$TSPAINT_RELATE_CONVERT`` if set, else the ``tspaint install
#: relate`` build location (falling back to ``Convert`` on ``PATH``). :func:`relate` re-resolves this
#: per call, so an install done after import is still picked up.
DEFAULT_CONVERT = _default_convert()


def relate_convert(anc, mut, *, out_prefix=None, compress=True, convert_bin=None):
    """Convert **existing** Relate output (``.anc`` / ``.mut``) to a tskit tree sequence (CLAUDE.md §5).

    The low-level Convert step used by the :func:`relate` front end, exposed for when you have already
    run Relate yourself. Wraps ``relate_lib`` ``Convert --mode ConvertToTreeSequence [--compress]``.
    ``--compress`` is **load-bearing** — it unifies persistent clades into one node id across adjacent
    trees, the persistence invariant the edge-blocking depends on (§5) — so keep it on. Run
    :func:`check_persistence` on the result before inference (§5.1). To go from **genotypes** all the
    way to a tree sequence (running Relate + ``EstimatePopulationSize`` for you), use :func:`relate`.

    Parameters
    ----------
    anc : str
        Path to the Relate ``.anc`` (``.gz``) file.
    mut : str
        Path to the Relate ``.mut`` (``.gz``) file.
    out_prefix : str, optional
        Output prefix for the converted ``.trees`` (default: a tempfile prefix).
    compress : bool, optional
        Pass ``--compress`` (default ``True`` — load-bearing; do not turn off, §5).
    convert_bin : str, optional
        Path to the ``relate_lib`` ``Convert`` binary (default: env ``TSPAINT_RELATE_CONVERT``
        or ``Convert`` on ``PATH``).

    Returns
    -------
    tskit.TreeSequence
        The converted tree sequence (node ids stable across trees thanks to ``--compress``).

    Raises
    ------
    FileNotFoundError
        If the ``Convert`` binary or an input file is absent.
    RuntimeError
        If ``Convert`` exits nonzero.
    """
    import tskit
    convert_bin = convert_bin or _default_convert()
    if shutil.which(convert_bin) is None and not os.path.exists(convert_bin):
        raise FileNotFoundError(
            f"relate_lib Convert binary not found ({convert_bin!r}); set TSPAINT_RELATE_CONVERT or "
            "pass convert_bin (https://github.com/leospeidel/relate_lib)")
    for path, what in ((anc, "anc"), (mut, "mut")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Relate {what} file not found: {path}")
    out_prefix = out_prefix or os.path.join(tempfile.mkdtemp(prefix="tspaint_relate_"), "relate")
    cmd = [convert_bin, "--mode", "ConvertToTreeSequence", "--anc", anc, "--mut", mut, "-o", out_prefix]
    if compress:
        cmd.append("--compress")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Relate Convert failed:\n{res.stderr[-1000:]}")
    return tskit.load(out_prefix + ".trees")


def _run_relate_step(cmd, *, cwd, what, tolerate_if=None):
    """Run one Relate CLI step; raise with captured output on failure.

    If ``tolerate_if`` is given and returns True after a nonzero exit, the failure is treated as
    success — used for ``EstimatePopulationSize``, whose final R/ggplot2 *plot* step exits nonzero
    when ggplot2 is absent even though the re-dated genealogy was already written.
    """
    res = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if res.returncode == 0 or (tolerate_if is not None and tolerate_if()):
        return res
    tail = ((res.stderr or "") + "\n" + (res.stdout or "")).strip()[-1500:]
    raise RuntimeError(f"Relate step '{what}' failed (exit {res.returncode}):\n{tail}")


def _write_flat_genetic_map(workdir, vcf_path, recombination_rate):
    """Write a constant-rate Relate genetic map (``pos COMBINED_rate[cM/Mb] Genetic_Map[cM]``) that
    spans every SNP in ``vcf_path`` (``cM/Mb = r·1e8``; cumulative cM at ``p`` is ``p·r·100``)."""
    maxpos = 0
    with open(vcf_path) as f:
        for line in f:
            if line and not line.startswith("#"):
                maxpos = max(maxpos, int(line.split("\t", 2)[1]))
    end = maxpos + 1
    cm_per_mb = recombination_rate * 1e8
    path = os.path.join(workdir, "genetic_map.txt")
    with open(path, "w") as f:
        f.write("pos COMBINED_rate Genetic_Map\n")
        f.write(f"0 {cm_per_mb:g} 0\n")
        f.write(f"{end} {cm_per_mb:g} {end * recombination_rate * 100:g}\n")
    return path


def _write_single_pop_poplabels(workdir, sample_path):
    """Write a single-population ``.poplabels`` (one row per ``.sample`` individual, in order) for
    ``EstimatePopulationSize``; haploid samples take SEX 1. Pass your own ``poplabels`` to
    :func:`relate` for a structure-aware (per-population) coalescence-rate estimate."""
    with open(sample_path) as f:
        rows = [ln.split()[0] for ln in f.read().splitlines()[2:] if ln.strip()]   # skip 2 headers
    path = os.path.join(workdir, "pop.poplabels")
    with open(path, "w") as f:
        f.write("sample population group sex\n")
        for s in rows:
            f.write(f"{s} POP1 POP1 1\n")
    return path


def relate(source, *, mutation_rate=None, recombination_rate=1e-8, Ne=None, genetic_map=None,
           poplabels=None, estimate_population_size=True, seed=1, memory=None, num_iter=None,
           compress=True, workdir=None, relate_bin=None, file_formats_bin=None, eps_script=None,
           convert_bin=None, relate_args=None, m=None, r=None):
    r"""Infer a tree sequence from genotypes with **Relate**, end to end (CLAUDE.md §5).

    The real-data analogue of :func:`tspaint.io.tsinfer` / :func:`tspaint.io.singer`: hand it
    genotypes and get back a tskit tree sequence — it runs the whole Relate pipeline for you, so **no
    Relate command line is required**:

    1. write a haploid VCF and convert it to Relate haps/sample (``RelateFileFormats``);
    2. infer genome-wide genealogies (``Relate --mode All``);
    3. (default) re-estimate the coalescence-rate history and re-date branch lengths **genome-wide**
       (``EstimatePopulationSize`` — best on a whole chromosome, so give it one);
    4. convert to tskit with ``--compress`` (:func:`relate_convert`; §5 persistence invariant).

    Build the binaries once with ``tspaint install relate``. Paint the result with
    ``tspaint.paint(ts, labels, n_jobs=…)`` (or stream a huge chromosome with ``window_size=…,
    out_dir=…``).

    Parameters
    ----------
    source : tskit.TreeSequence, str, or Variants
        Genotypes — a tree sequence with mutations, a **VCF**, a **VCF Zarr**, or a
        :class:`~tspaint.io_genotypes.Variants` (e.g. from :func:`~tspaint.io.subset_data`), resolved
        as for the other front ends. Each sample haplotype becomes one Relate haplotype, in order.
    mutation_rate : float
        Per-base, per-generation mutation rate (Relate ``-m``; alias ``m``). **Required.**
    recombination_rate : float, optional
        Per-base recombination rate for a synthesised constant-rate genetic map (default ``1e-8``;
        alias ``r``). Ignored when ``genetic_map`` is given.
    Ne : float, optional
        Diploid effective size for Relate's initial prior (``-N`` receives the haploid ``2·Ne``).
        Relate re-estimates it via ``EstimatePopulationSize``, so it is only a starting point; the
        default estimates it from the data (:func:`tspaint.io.estimate_ne`, all-pairs ``π / 4μ``).
    genetic_map : str, optional
        Path to a Relate genetic map (``pos COMBINED_rate Genetic_Map``); default synthesises a flat
        one from ``recombination_rate``.
    poplabels : str, optional
        Path to a ``.poplabels`` file for ``EstimatePopulationSize`` (population/group/sex per sample,
        in ``.sample`` order); default treats all samples as one population. Pass per-population labels
        for a structure-aware coalescence-rate estimate on an admixed / multi-species panel.
    estimate_population_size : bool, optional
        Run ``EstimatePopulationSize`` (step 3), the genome-wide branch-length recalibration (default
        ``True``). ``False`` converts Relate's initial ``--mode All`` trees directly (faster, less
        calibrated).
    seed : int, optional
        Random seed for Relate and ``EstimatePopulationSize`` (default 1).
    memory : float, optional
        Relate ``--memory`` (GB) hint for large data.
    num_iter : int, optional
        ``EstimatePopulationSize`` iterations (default: the script's own).
    compress : bool, optional
        Convert with ``--compress`` (default ``True`` — load-bearing, §5).
    workdir : str, optional
        Directory for the intermediate files (default: a fresh tempdir).
    relate_bin, file_formats_bin, eps_script, convert_bin : str, optional
        Override the ``Relate`` / ``RelateFileFormats`` / ``EstimatePopulationSize.sh`` / ``Convert``
        paths (defaults: the ``tspaint install relate`` build locations).
    relate_args : list, optional
        Extra flags appended verbatim to ``Relate --mode All``.

    Returns
    -------
    tskit.TreeSequence
        The Relate tree sequence (node ids stable across trees via ``--compress``; branch lengths in
        generations). Sample nodes are in input order; for a **VCF** / **VCF-Zarr** / ``Variants``
        source the source's sample ids are stamped on them (:func:`tspaint.ids.attach_sample_ids`), so
        :func:`tspaint.paint` accepts ``labels`` keyed by sample-ID string as well as by node index.

    Raises
    ------
    FileNotFoundError
        If a required Relate binary is absent (build them with ``tspaint install relate``).
    RuntimeError
        If a Relate step fails.
    """
    from .ids import attach_sample_ids
    from .io_genotypes import estimate_ne
    from .io_singer import _source_sample_ids, _write_singer_vcf

    mutation_rate = m if m is not None else mutation_rate
    recombination_rate = r if r is not None else recombination_rate
    if mutation_rate is None:
        raise ValueError("relate needs a mutation rate: pass mutation_rate= (or m=)")
    rff = file_formats_bin or relate_file_formats_path()
    rel = relate_bin or relate_binary_path()
    eps = eps_script or estimate_population_size_path()
    for path, hint in ((rff, "RelateFileFormats"), (rel, "Relate")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{hint} not found at {path}; build it with `tspaint install relate`")

    src, names, in_ploidy = _source_sample_ids(source)
    if Ne is None:
        Ne = estimate_ne(src, mutation_rate)

    tmp = workdir or tempfile.mkdtemp(prefix="tspaint_relate_")
    os.makedirs(tmp, exist_ok=True)
    vcf_prefix = os.path.join(tmp, "data")
    _write_singer_vcf(src, vcf_prefix)                                  # haploid VCF -> prefix.vcf

    _run_relate_step([rff, "--mode", "ConvertFromVcf", "--haps", "data.haps", "--sample",
                      "data.sample", "-i", "data"], cwd=tmp, what="ConvertFromVcf")

    if genetic_map is None:
        gmap = os.path.basename(_write_flat_genetic_map(tmp, vcf_prefix + ".vcf", recombination_rate))
    else:
        gmap = os.path.abspath(genetic_map)

    rcmd = [rel, "--mode", "All", "-m", mutation_rate, "-N", 2 * Ne, "--haps", "data.haps",
            "--sample", "data.sample", "--map", gmap, "--seed", seed, "-o", "relate_out"]
    if memory is not None:
        rcmd += ["--memory", memory]
    if relate_args:
        rcmd += list(relate_args)
    _run_relate_step(rcmd, cwd=tmp, what="Relate --mode All")

    stem = "relate_out"
    if estimate_population_size:
        pl = os.path.abspath(poplabels) if poplabels else \
            os.path.basename(_write_single_pop_poplabels(tmp, os.path.join(tmp, "data.sample")))
        ecmd = [eps, "-i", "relate_out", "-m", mutation_rate, "--poplabels", pl,
                "--seed", seed, "-o", "relate_popsize"]
        if num_iter is not None:
            ecmd += ["--num_iter", num_iter]
        _run_relate_step(
            ecmd, cwd=tmp, what="EstimatePopulationSize",
            tolerate_if=lambda: any(os.path.exists(os.path.join(tmp, f"relate_popsize.anc{e}"))
                                    for e in ("", ".gz")))
        stem = "relate_popsize"

    def _pick(*cands):
        for c in cands:
            if os.path.exists(c):
                return c
        raise RuntimeError(f"expected Relate output not found (looked for {cands})")

    anc = _pick(os.path.join(tmp, f"{stem}.anc.gz"), os.path.join(tmp, f"{stem}.anc"))
    mut = _pick(os.path.join(tmp, f"{stem}.mut.gz"), os.path.join(tmp, f"{stem}.mut"))
    ts = relate_convert(anc, mut, compress=compress, convert_bin=convert_bin,
                        out_prefix=os.path.join(tmp, "converted"))
    return attach_sample_ids(ts, names, in_ploidy)


def windows(ts, window_size, *, trim=False):
    """Split a (whole-chromosome) tree sequence into per-window tree sequences for painting.

    Relate's coalescence-rate / effective-size estimation (``EstimatePopulationSize``) is best run
    **genome-wide** — on a whole chromosome at once — so the calibrated branch lengths reflect the
    full genealogy (see :func:`estimate_population_size_path`). The resulting chromosome-length tree
    sequence is then usually too large to paint in one call; this tiles ``[0, sequence_length)`` into
    contiguous, non-overlapping ``window_size``-bp windows, each a standalone tree sequence you can
    hand to :func:`tspaint.paint` (and, e.g., process in parallel).

    Painting is topology-driven, so windows are painted **independently and reassembled by genomic
    position**. Do **not** pass the returned list to :func:`tspaint.paint` as a single argument — a
    list of tree sequences is treated as a posterior *ensemble* and averaged, which is wrong for
    disjoint genomic regions. Paint each window on its own and concatenate the per-window segments.
    (To share one fitted ``Q`` across windows, seed each :func:`tspaint.paint` call with ``Q0`` from a
    genome-wide fit; otherwise each window fits its own ``Q`` from its trees.)

    Parameters
    ----------
    ts : tskit.TreeSequence
        The (whole-chromosome) tree sequence, e.g. from :func:`relate`. Any tree sequence works — the
        function is not Relate-specific — but this is the intended genome-wide-Relate → paint path.
    window_size : float
        Window width in bp; ``[0, sequence_length)`` is tiled into ``ceil(L / window_size)`` windows
        (the last is shorter if the length does not divide evenly).
    trim : bool, optional
        If ``False`` (default) each window keeps the **full genomic coordinates** — the flanking
        regions carry no edges and paint as missing-info — so per-window paintings already sit at
        their true positions and reassemble by concatenation. If ``True`` each window is
        :meth:`~tskit.TreeSequence.trim`\\ med to start at 0 (compact; offset window ``k`` by
        ``k * window_size`` to place it).

    Returns
    -------
    list of tskit.TreeSequence
        One tree sequence per window, in genome order. Sample nodes (and their ids / metadata) are
        preserved, so ``labels`` keyed by sample id or by index work on every window.

    Raises
    ------
    ValueError
        If ``window_size <= 0``.
    """
    return [w.trim() if trim else w for _k, _lo, _hi, w in _iter_windows(ts, window_size)]


def _iter_windows(ts, window_size):
    """Lazily yield ``(k, lo, hi, window_ts)`` for each window — the memory-bounded primitive.

    Materialises **one** window tree sequence at a time (the previous one is released before the
    next is cut), so a caller that paints-and-releases per window — e.g.
    ``paint(ts, window_size=…, out_dir=…)`` — never holds more than a single window. :func:`windows`
    is the eager ``list`` of this.
    """
    import math
    L = float(ts.sequence_length)
    window_size = float(window_size)
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    n = max(1, math.ceil(L / window_size))
    for k in range(n):
        lo = k * window_size
        hi = min((k + 1) * window_size, L)
        yield k, lo, hi, ts.keep_intervals([[lo, hi]], simplify=True)


def check_persistence(ts):
    """Run the §5.1 persistence go/no-go on an already-loaded tree sequence."""
    return persistence_summary(ts)


def convert_relate(anc, mut, out_prefix, compress=True, convert_bin="Convert"):  # pragma: no cover
    """Deprecated alias for :func:`relate_convert` (the ``.anc``/``.mut`` → tskit Convert step).

    Note ``tspaint.io.relate`` is now the **genotypes → tree sequence** front end (it runs Relate for
    you); the Convert-only step you probably want is :func:`relate_convert`.
    """
    warnings.warn("tspaint.io.convert_relate is deprecated; use tspaint.io.relate_convert",
                  DeprecationWarning, stacklevel=2)
    return relate_convert(anc, mut, out_prefix=out_prefix, compress=compress, convert_bin=convert_bin)
