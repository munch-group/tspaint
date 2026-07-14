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

from .ids import attach_sample_ids

__all__ = ["singer", "singer_windowed", "write_haploid_vcf", "singer_tree_sequences",
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

    Returns
    -------
    str
        Absolute path to the repo root — the nearest ancestor (of this file, then of the cwd)
        containing the ``external/tools.ini`` marker, else the legacy ``<this>/../../..`` fallback.
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
    """Clone root that ``tspaint install singer`` builds into (``<tools-dir>/SINGER``).

    Returns
    -------
    str
        The path ``<tools-dir>/SINGER``, where ``tools-dir`` is ``$TSPAINT_TOOLS_DIR`` if set,
        else ``<repo>/external`` (:func:`repo_root`).
    """
    return os.path.join(_tools_dir(), "SINGER")


def singer_binary_path():
    """Path to the ``singer`` binary built by ``tspaint install singer``.

    Returns
    -------
    str
        The binary path under :func:`singer_install_dir` (so it honours ``$TSPAINT_TOOLS_DIR``).
        This is the build location only; the runtime default ``DEFAULT_SINGER`` additionally
        honours ``$TSPAINT_SINGER``.
    """
    return os.path.join(singer_install_dir(), "SINGER", "SINGER", "singer")


#: SINGER binary: ``$TSPAINT_SINGER`` if set, else the ``tspaint install singer`` build location.
DEFAULT_SINGER = os.environ.get("TSPAINT_SINGER") or singer_binary_path()
#: SINGER's ``merge_ARG.py`` helper, at the same install location unless ``$TSPAINT_MERGE_ARG``.
DEFAULT_MERGE_ARG = os.environ.get("TSPAINT_MERGE_ARG") or os.path.join(
    singer_install_dir(), "SINGER", "SINGER", "merge_ARG.py")

#: Raised when ``Ne`` is omitted: SINGER's binary requires ``-Ne`` (it has no auto-Ne — that lives
#: only in the ``singer_master`` wrapper), so tspaint does not estimate one silently. The user picks
#: the value, optionally via :func:`tspaint.io.estimate_ne`.
_NE_REQUIRED = (
    "singer requires Ne: the SINGER binary needs -Ne and errors without it (auto-Ne lives only in "
    "SINGER's singer_master wrapper, not the binary). SINGER calibrates its coalescent prior so that "
    "4*Ne*mu ~= pi (observed nucleotide diversity), so estimate Ne over the WHOLE analysed panel "
    "(all-pairs pi/4mu -- it must place the deep between-population coalescences on SINGER's "
    "representable time range) and pass it, e.g.\n"
    "    Ne = tspaint.io.estimate_ne(source, mutation_rate)   # all-pairs pi/4mu; optional exclude=soft_refs\n"
    "    tspaint.io.singer(source, _Ne=Ne, _m=mutation_rate)\n"
    "Do NOT restrict with groups=labels here: a within-population Ne under-calibrates SINGER's prior "
    "on a structured / multi-population sample and can push coalescence times off-scale.")

#: Defaults for the unified sampling knobs: tree sequences returned (``ts``), MCMC iterations between
#: saved samples (``mcmc_step``), and burn-in iterations (``mcmc_burnin``). The chain runs
#: ``ts * mcmc_step + mcmc_burnin`` iterations, saving every ``mcmc_step`` and keeping ``ts`` of them.
_TS_DEFAULT, _STEP_DEFAULT, _BURNIN_DEFAULT = 20, 50, 200


def _reject_both(plain_name, plain, flag_name, flag):
    """A plain sampling knob and its terminal-flag ``_``-counterpart cannot both be set: the plain one
    takes precedence, so passing both is ambiguous."""
    if plain is not None and flag is not None:
        raise ValueError(
            f"pass either {plain_name}= or {flag_name}=, not both: the plain '{plain_name}' takes "
            f"precedence over the terminal-flag '{flag_name}'.")


def _select(written, discard, keep):
    """Keep ``keep`` raw samples after discarding the first ``discard`` (burn-in)."""
    return list(written)[int(discard):][:int(keep)]


def _singer_sampling(ts, mcmc_step, mcmc_burnin, _n_samples, _thin):
    """Resolve the unified knobs (+ optional native ``_n_samples`` / ``_thin``) into SINGER's run.

    Returns ``(n, thin, discard, keep)``: SINGER writes ``n`` samples ``thin`` iterations apart
    (``-n`` / ``-thin``); the caller discards the first ``discard`` and keeps ``keep``. With the
    defaults ``n = ts + mcmc_burnin // mcmc_step``, so the chain runs ``ts*mcmc_step + mcmc_burnin``
    iterations and ``keep == ts``.
    """
    _reject_both("ts", ts, "_n_samples", _n_samples)
    _reject_both("mcmc_step", mcmc_step, "_thin", _thin)
    thin = int(_thin) if _thin is not None else (_STEP_DEFAULT if mcmc_step is None else int(mcmc_step))
    burnin = _BURNIN_DEFAULT if mcmc_burnin is None else int(mcmc_burnin)
    if thin < 1:
        raise ValueError(f"mcmc_step / _thin must be >= 1, got {thin}")
    if burnin < 0:
        raise ValueError(f"mcmc_burnin must be >= 0, got {burnin}")
    discard = burnin // thin
    if _n_samples is not None:
        n = int(_n_samples); keep = n - discard
    else:
        keep = _TS_DEFAULT if ts is None else int(ts); n = keep + discard
    if keep < 1:
        raise ValueError("resolved to < 1 returned sample; raise ts / _n_samples or lower mcmc_burnin")
    return n, thin, discard, keep


def _argweaver_sampling(ts, mcmc_step, mcmc_burnin, _iters, _sample_step):
    """Resolve the unified knobs (+ optional native ``_iters`` / ``_sample_step``) into ARGweaver's
    run. Returns ``(iters, sample_step, discard, keep)``: ``arg-sample`` runs ``iters`` iterations
    saving one every ``sample_step`` (``-n`` / ``--sample-step``), so it writes
    ``iters // sample_step + 1`` ARGs; the caller discards the first ``discard`` and keeps ``keep``.
    With the defaults ``iters = ts*mcmc_step + mcmc_burnin`` and ``keep == ts``.
    """
    _reject_both("mcmc_step", mcmc_step, "_sample_step", _sample_step)
    _reject_both("ts", ts, "_iters", _iters)
    _reject_both("mcmc_burnin", mcmc_burnin, "_iters", _iters)
    step = int(_sample_step) if _sample_step is not None else (_STEP_DEFAULT if mcmc_step is None else int(mcmc_step))
    burnin = _BURNIN_DEFAULT if mcmc_burnin is None else int(mcmc_burnin)
    if step < 1:
        raise ValueError(f"mcmc_step / _sample_step must be >= 1, got {step}")
    if burnin < 0:
        raise ValueError(f"mcmc_burnin must be >= 0, got {burnin}")
    discard = burnin // step
    ts_eff = _TS_DEFAULT if ts is None else int(ts)
    if _iters is not None:
        iters = int(_iters); keep = (iters // step + 1) - discard
    else:
        iters = ts_eff * step + burnin; keep = ts_eff
    if keep < 1:
        raise ValueError("resolved to < 1 returned sample; raise ts / _iters or lower mcmc_burnin")
    return iters, step, discard, keep


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

    def _distinct(x):
        # strictly-increasing distinct 1-based integer positions (SINGER errors /
        # merges on duplicate POS; matches io_genotypes.write_haploid_vcf's guard).
        pos = np.floor(np.asarray(x)).astype(int) + 1
        out = np.empty(len(pos), dtype=int)
        last = 0
        for i, p in enumerate(pos):
            p = max(int(p), last + 1)
            out[i] = p
            last = p
        return out

    with open(path, "w") as f:
        ts_hap.write_vcf(f, ploidy=1, position_transform=_distinct)


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
    seqlen = float(max(edge[:, 1]))
    tables = tskit.TableCollection(sequence_length=seqlen)
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
            # Add each distinct within-window mutation position as a site. SINGER writes an
            # end-of-window sentinel at position == sequence_length; keep the guard SINGER's own
            # convert_long_ARG.py uses (``< length``) but with the real window length, NOT a
            # hard-coded 1e6 (which silently dropped every site past 1 Mb in any region > 1 Mb).
            if muts[i, 0] != mp and muts[i, 0] < seqlen:
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


def _source_sample_ids(source):
    """Resolve ``(source, sample_names, ploidy, sample_index)`` for id-stamping the output ts.

    A VCF / VCF-Zarr / :class:`~tspaint.io_genotypes.Variants` source is parsed once into a
    ``Variants`` and returned (so the write step reuses it — no second parse); a ``ts`` source
    carries no VCF sample names, so ``(source, None, 1, None)``. ``sample_index`` groups haplotype
    columns into individuals for mixed ploidy (e.g. chrX); ``None`` for uniform ploidy.
    """
    from .io_genotypes import source_kind, resolve_variants
    if source_kind(source) == "ts":
        return source, None, 1, None
    v = resolve_variants(source)
    return v, v.sample_names, v.ploidy, v.sample_index


def _singer_indices(out_prefix):
    """The MCMC sample indices SINGER wrote at ``{out_prefix}_nodes_<i>.txt`` (sorted)."""
    return sorted(int(f.split("_nodes_")[1].split(".txt")[0])
                  for f in glob.glob(out_prefix + "_nodes_*.txt"))


def _run_singer(prefix, out, *, start, end, Ne, mutation_rate, recombination_rate, n_samples,
                thin, ploidy=1, seed=42, polar=0.5, penalty=None, hmm_epsilon=None, psmc_bins=None,
                recomb_map=None, mut_map=None, fast=False, singer_args=None,
                max_retries=50, singer_bin=None):
    """Invoke the bare SINGER binary on ``[start, end)`` with the ``singer_master`` retry loop.

    Builds the command from SINGER's own flags, each passed **as-is**: ``-Ne -m/-mut_map
    -r/-recomb_map -polar -n -thin -ploidy -seed`` plus the optional ``-penalty -hmm_epsilon
    -psmc_bins -fast`` and any ``singer_args`` passthrough (appended last, so it overrides). The
    genome-window (``-start -end``) and I/O (``-input -output``) flags are managed by this wrapper;
    SINGER's stateful ``-resume`` / ``-debug`` resume-from-``.log`` operations are not standalone
    flags (``-debug`` is how this loop re-runs a failed case). Reads ``prefix.vcf``, writes
    ``{out}_nodes_<i>.txt`` / ``_branches_<i>.txt`` / ``_muts_<i>.txt``; returns the indices written.
    On a nonzero exit it re-invokes ``-debug`` (resuming that run's ``.log``) with fresh seeds.
    """
    singer_bin = singer_bin or DEFAULT_SINGER
    if not os.path.exists(singer_bin):
        raise FileNotFoundError(
            f"SINGER binary not found at {singer_bin}; set TSPAINT_SINGER or pass singer_bin")
    base = [singer_bin, "-Ne", str(Ne), "-ploidy", str(ploidy), "-input", prefix, "-output", out,
            "-start", str(int(start)), "-end", str(int(end)), "-polar", str(polar),
            "-n", str(n_samples), "-thin", str(thin)]
    base += ["-mut_map", str(mut_map)] if mut_map is not None else ["-m", str(mutation_rate)]
    base += ["-recomb_map", str(recomb_map)] if recomb_map is not None else ["-r", str(recombination_rate)]
    if penalty is not None:
        base += ["-penalty", str(penalty)]
    if hmm_epsilon is not None:
        base += ["-hmm_epsilon", str(hmm_epsilon)]
    if psmc_bins is not None:
        base += ["-psmc_bins", str(psmc_bins)]
    if fast:
        base += ["-fast"]
    if singer_args:
        base += [str(a) for a in singer_args]
    rng = random.Random(seed)
    seeds = [seed] + [rng.randint(0, 2 ** 30 - 1) for _ in range(max_retries)]
    last = None
    for k, sd in enumerate(seeds):
        cmd = base + (["-debug"] if k > 0 else []) + ["-seed", str(sd)]
        last = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if last.returncode == 0:
            return _singer_indices(out)
        if last.returncode == -11:        # SIGSEGV: a hard crash inside SINGER, not a numerical
            break                         # edge case fresh seeds can dodge — don't burn 50 retries

    # Failure. SINGER prints its log/errors to *stdout* (stderr is usually empty, esp. on a crash),
    # so surface stdout, and name the signal when it was killed (negative returncode).
    import signal
    rc = last.returncode
    if rc < 0:
        try:
            why = f"crashed ({signal.Signals(-rc).name})"
        except ValueError:
            why = f"was killed by signal {-rc}"
        why += " — a SINGER-side bug on this input; reseeding cannot fix a deterministic crash"
    else:
        why = f"failed after {max_retries} retries (exit {rc})"
    tail = (last.stdout or last.stderr or "").strip()[-1500:]
    raise RuntimeError(
        f"SINGER {why} on [{int(start)},{int(end)}), {n_samples} samples. SINGER's last output:\n"
        f"{tail}\n(workarounds: a smaller -start/-end window, a newer SINGER build, or arg='tsinfer'.)")


def _resolve_singer_rates(*, _m, _r, _ratio, _mut_map, _recomb_map):
    """Resolve SINGER's rate flags: derive ``-r`` from ``-m * -ratio`` when neither ``_r`` nor a
    ``_recomb_map`` is given (SINGER's ``-ratio``, default 1)."""
    if _m is None and _mut_map is None:
        raise ValueError("singer needs a mutation rate: pass _m= (or _mut_map=)")
    if _r is None and _recomb_map is None:
        _r = _m * (1.0 if _ratio is None else _ratio)     # SINGER's -ratio (default 1): r = m * ratio
    return _m, _r


def singer(source, *, ts=None, mcmc_step=None, mcmc_burnin=None,
           _Ne=None, _m=None, _r=None, _ratio=None, _n_samples=None, _thin=None,
           _polar=0.5, _ploidy=1, _seed=42, _penalty=None, _hmm_epsilon=None, _psmc_bins=None,
           _fast=False, _recomb_map=None, _mut_map=None,
           singer_args=None, workdir=None, singer_bin=None,
           with_mutations=True, max_retries=50, sequence_length=None):
    """Sample posterior ARGs from genotypes via SINGER (CLAUDE.md §7.4).

    SINGER's MCMC samples ARGs from ``P(ARG | genotypes)``; the thinned post-burn-in
    samples are the ideal input to :func:`tspaint.ensemble.merge_posterior_tables`,
    since they represent genuine ARG uncertainty (§7.4).

    **Posterior sampling is controlled by three unified knobs, shared with**
    :func:`tspaint.io.argweaver` — ``ts`` (how many tree sequences you get back), ``mcmc_step`` (MCMC
    iterations between saved samples) and ``mcmc_burnin`` (burn-in iterations). The chain runs
    ``ts * mcmc_step + mcmc_burnin`` iterations; tspaint translates the knobs into SINGER's native
    ``-n`` / ``-thin`` (see Returns). Every **native terminal flag** is exposed underscore-prefixed to
    mark its 1:1 correspondence to the SINGER CLI: ``_Ne`` (``-Ne``, **required**), ``_m`` (``-m``),
    ``_r`` (``-r``) or ``_ratio`` (``-ratio``, default 1 — so ``r = m*ratio``), ``_polar``, ``_ploidy``,
    ``_seed``, ``_recomb_map`` / ``_mut_map``, ``_penalty`` / ``_hmm_epsilon`` / ``_psmc_bins``,
    ``_fast``, and the raw sampling flags ``_n_samples`` (``-n``) / ``_thin`` (``-thin``) — normally
    left to inference from the three unified knobs. Passing a plain knob **and** its ``_``-counterpart
    (e.g. ``ts`` and ``_n_samples``) raises: the plain one takes precedence.

    Parameters
    ----------
    source : tskit.TreeSequence or str
        The genotypes to sample ARGs from — a tree sequence carrying mutations, a **VCF Zarr**
        store, or a **VCF** file (normalised by :mod:`tspaint.io_genotypes`). All are written out
        as a haploid VCF for SINGER.
    ts : int, optional
        Number of posterior tree sequences returned (default 20). Exactly this many come back (see
        Returns) — a single :class:`tskit.TreeSequence` when 1, else a list.
    mcmc_step : int, optional
        MCMC iterations between saved samples (SINGER ``-thin``; default 50). Larger ⇒ more
        decorrelated returned samples.
    mcmc_burnin : int, optional
        Burn-in MCMC iterations discarded before the kept samples (default 200); ``mcmc_burnin //
        mcmc_step`` leading saved samples are dropped.
    _Ne : float
        Diploid effective population size (``-Ne``). **Required** — SINGER needs it and tspaint does
        not estimate one silently. SINGER calibrates its prior so ``4·_Ne·_m ≈ π`` (observed
        diversity), so estimate _Ne over the **whole analysed panel** with
        :func:`tspaint.io.estimate_ne` (all-pairs ``π/4μ``). On a structured / multi-species sample
        that value is legitimately large — a too-small _Ne pushes deep coalescences off SINGER's
        representable range into noisy / failing runs. Omitting it raises ``ValueError``.
    _m : float
        Per-base mutation rate (``-m``). Required (or ``_mut_map``).
    _r : float, optional
        Per-base recombination rate (``-r``). Defaults to ``_m * _ratio``.
    _ratio, _polar, _ploidy, _seed, _penalty, _hmm_epsilon, _psmc_bins, _fast, _recomb_map, _mut_map
        The remaining SINGER terminal flags. All are forwarded 1:1 (``-polar`` 0.5, ``-ploidy`` 1,
        ``-seed`` 42, ``-fast`` off; the rest optional) **except** ``_ratio`` (default 1), which is
        not forwarded but sets ``_r = _m * _ratio`` when neither ``_r`` nor ``_recomb_map`` is set
        (mirrors SINGER's own ``-ratio``).
    _n_samples, _thin : int, optional
        SINGER's raw ``-n`` / ``-thin`` — normally inferred from ``ts`` / ``mcmc_step`` / ``mcmc_burnin``;
        set directly for full control (but not alongside the plain knob they correspond to).
    singer_args : list, optional
        Extra raw SINGER command-line tokens appended after tspaint's own flags (so they take
        precedence). Default ``None``.
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
    tskit.TreeSequence or list of tskit.TreeSequence
        Exactly ``ts`` posterior samples — a **single** :class:`tskit.TreeSequence` when ``ts == 1``,
        else a **list** of ``ts``. SINGER runs ``-n = ts + mcmc_burnin // mcmc_step`` samples ``-thin =
        mcmc_step`` iterations apart (``ts*mcmc_step + mcmc_burnin`` iterations total); tspaint discards
        the ``mcmc_burnin // mcmc_step`` burn-in samples and keeps ``ts``. E.g. ``ts=20, mcmc_step=2,
        mcmc_burnin=200`` → 20 samples over 240 iterations. Sample order is preserved (VCF column ``i``
        -> sample ``i``); for a VCF / VCF-Zarr / ``Variants`` source the sample ids are stamped onto the
        sample nodes (:func:`tspaint.ids.attach_sample_ids`), so :func:`tspaint.paint` accepts
        ``labels`` keyed by sample-ID string as well as by node index.

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
    _m, _r = _resolve_singer_rates(_m=_m, _r=_r, _ratio=_ratio, _mut_map=_mut_map, _recomb_map=_recomb_map)
    n_raw, thin, discard, keep = _singer_sampling(ts, mcmc_step, mcmc_burnin, _n_samples, _thin)
    # NB: ``_ploidy`` is SINGER's ``-ploidy`` for the *haploid* VCF (always the input's per-haplotype
    # columns), separate from the source's own diploidy used for id stamping.
    if _Ne is None:
        raise ValueError(_NE_REQUIRED)
    source, names, in_ploidy, sidx = _source_sample_ids(source)
    tmp = workdir or tempfile.mkdtemp(prefix="tspaint_singer_")
    os.makedirs(tmp, exist_ok=True)
    prefix = os.path.join(tmp, "data")
    L = _write_singer_vcf(source, prefix, sequence_length)
    out = os.path.join(tmp, "arg")
    _run_singer(prefix, out, start=0, end=L, Ne=_Ne, mutation_rate=_m,
                recombination_rate=_r, n_samples=n_raw, thin=thin,
                ploidy=_ploidy, seed=_seed, polar=_polar, penalty=_penalty, hmm_epsilon=_hmm_epsilon,
                psmc_bins=_psmc_bins, recomb_map=_recomb_map, mut_map=_mut_map, fast=_fast,
                singer_args=singer_args, max_retries=max_retries, singer_bin=singer_bin)

    samples = []
    for i in _select(_singer_indices(out), discard, keep):
        mf = f"{out}_muts_{i}.txt" if with_mutations else None
        arg = _read_singer_arg(f"{out}_nodes_{i}.txt", f"{out}_branches_{i}.txt", mf)
        samples.append(attach_sample_ids(arg, names, in_ploidy, sample_index=sidx))   # name-keyed labels
    if len(samples) == 1:
        return samples[0]
    return samples


def singer_window(source, *, start, end, out_prefix, _Ne, _m=None, _r=None,
                  _ratio=None, _n_samples=20, _thin=10, _ploidy=1, _seed=42, _polar=0.5,
                  _penalty=None, _hmm_epsilon=None, _psmc_bins=None, _recomb_map=None, _mut_map=None,
                  _fast=False, singer_args=None, max_retries=50, singer_bin=None):
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
    _Ne, _m, _r, _ploidy, _seed, _polar, max_retries, singer_bin
        The underscore-prefixed SINGER flags (plus the retry / binary controls), as for :func:`singer`.
    _n_samples, _thin : int, optional
        SINGER's raw ``-n`` / ``-thin``, passed directly for this one window (this per-window
        primitive has no unified ``ts`` / ``mcmc_step`` knobs). Defaults ``20`` / ``10``.
    _ratio, _penalty, _hmm_epsilon, _psmc_bins, _recomb_map, _mut_map, _fast, singer_args
        The remaining SINGER pass-through flags, as for :func:`singer`.

    Returns
    -------
    list[int]
        The MCMC sample indices written for this window.
    """
    from .io_genotypes import source_kind
    _m, _r = _resolve_singer_rates(_m=_m, _r=_r, _ratio=_ratio, _mut_map=_mut_map, _recomb_map=_recomb_map)
    if source_kind(source) == "vcf":
        s = str(source)
        vcf_prefix = s[:-4] if s.endswith(".vcf") else s
    else:
        vcf_prefix = out_prefix + "_input"
        _write_singer_vcf(source, vcf_prefix)
    return _run_singer(vcf_prefix, out_prefix, start=start, end=end, Ne=_Ne,
                       mutation_rate=_m, recombination_rate=_r,
                       n_samples=_n_samples, thin=_thin, ploidy=_ploidy, seed=_seed, polar=_polar,
                       penalty=_penalty, hmm_epsilon=_hmm_epsilon, psmc_bins=_psmc_bins,
                       recomb_map=_recomb_map, mut_map=_mut_map, fast=_fast,
                       singer_args=singer_args, max_retries=max_retries, singer_bin=singer_bin)


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


def _merge_script_with_mutation_parents(script):
    """Return a ``merge_ARG.py`` path whose multi-window builder calls ``compute_mutation_parents()``.

    SINGER's ``merge_ARG.py`` has two builders: ``read_short_ARG`` calls ``compute_mutation_parents()``
    but the multi-window ``read_long_ARG`` builds the tree sequence right after ``tables.sort()``
    **without** it (the same omission its ``convert_to_tskit`` has), so the file-table merge crashes
    on recurrent mutations with ``TSK_ERR_BAD_MUTATION_PARENT``. Insert the missing call at exactly
    that spot: if the ``sort() -> tree_sequence()`` pattern is present, write a patched **temp copy**
    (``merge_ARG.py`` imports only stdlib/numpy/tskit/tszip, so a relocated copy runs fine) and
    return ``(patched_path, patched_path)``; otherwise return ``(script, None)``.
    """
    import re
    with open(script) as f:
        src = f.read()
    patched, n = re.subn(
        r"tables\.sort\(\)[ \t]*\n(\s*)ts = tables\.tree_sequence\(\)",
        r"tables.sort()\n\g<1>tables.build_index()\n\g<1>tables.compute_mutation_parents()"
        r"\n\g<1>ts = tables.tree_sequence()",
        src, count=1)
    if n == 0:
        return script, None                       # already patched there, or an unfamiliar layout
    fd, path = tempfile.mkstemp(prefix="merge_ARG_patched_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(patched)
    return path, path


def run_merge_arg(rows, out, *, script=None, python=None):
    """Stitch per-window ARG tables into ``out`` by shelling out to SINGER's ``merge_ARG.py``.

    ``rows`` come from :func:`build_merge_table`. ``script`` defaults to ``DEFAULT_MERGE_ARG``
    (env ``TSPAINT_MERGE_ARG``); ``python`` to the current interpreter — note ``merge_ARG.py``
    imports ``tszip``, so that interpreter needs ``tskit + numpy + tszip``. The script is run
    through :func:`_merge_script_with_mutation_parents`, which repairs SINGER's missing
    ``compute_mutation_parents()`` call (else recurrent mutations abort the merge).

    Parameters
    ----------
    rows : list of tuple
        ``(nodes_file, branches_file, muts_file, block_coordinate)`` rows from
        :func:`build_merge_table` (one per genomic window, in genome order).
    out : str
        Destination path for the stitched, region-length tree sequence.
    script : str, optional
        Path to SINGER's ``merge_ARG.py``. Default ``None`` — uses ``DEFAULT_MERGE_ARG``
        (env ``TSPAINT_MERGE_ARG``, else the SINGER install location).
    python : str, optional
        Interpreter used to run the script (needs ``tskit + numpy + tszip``). Default ``None`` —
        uses the current interpreter (``sys.executable``).

    Returns
    -------
    str
        ``out`` — the path of the stitched tree sequence written by ``merge_ARG.py``.

    Raises
    ------
    FileNotFoundError
        If ``merge_ARG.py`` is not found at ``script``.
    RuntimeError
        If ``merge_ARG.py`` exits nonzero.
    """
    script = script or DEFAULT_MERGE_ARG
    python = python or sys.executable
    if not os.path.exists(script):
        raise FileNotFoundError(
            f"merge_ARG.py not found at {script}; set TSPAINT_MERGE_ARG or pass script")
    run_script, tmp_script = _merge_script_with_mutation_parents(script)
    fd, table = tempfile.mkstemp(suffix="_file_table.txt")
    os.close(fd)
    try:
        with open(table, "w") as f:
            for (n, b, m, blk) in rows:
                f.write(f"{n} {b} {m} {int(blk)}\n")
        r = subprocess.run([python, run_script, "--file_table", table, "--output", out],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise RuntimeError(f"merge_ARG.py failed:\n{(r.stderr or '')[-2000:]}")
    finally:
        os.remove(table)
        if tmp_script:
            os.remove(tmp_script)
    return out


def singer_windowed(source, *, window_size, ts=None, mcmc_step=None, mcmc_burnin=None,
                    _Ne=None, _m=None, _r=None, _ratio=None, _n_samples=None, _thin=None,
                    _polar=0.5, _ploidy=1, _seed=42, _penalty=None, _hmm_epsilon=None, _psmc_bins=None,
                    _fast=False, _recomb_map=None, _mut_map=None,
                    singer_args=None, n_jobs=None,
                    sequence_length=None, skip_gaps=None, workdir=None, singer_bin=None,
                    merge_arg_script=None, merge_python=None, log=None):
    """Sample posterior ARGs for a long region on ONE machine: window SINGER, then stitch.

    The single-machine analogue of the per-window × per-member cluster workflow (``workflow.py``)
    — for a laptop, a big multi-core node, or a Jupyter notebook, with no Slurm. It

    1. writes the haploid VCF **once** (any :func:`singer`-style ``source``);
    2. runs :func:`singer_window` on each contiguous ``window_size`` window **in parallel** across
       ``n_jobs`` worker threads (each shells out to the SINGER binary, so threads give real
       parallelism); then
    3. stitches every post-burn-in MCMC member across the windows into a region-length ARG with
       SINGER's ``merge_ARG.py`` (:func:`build_merge_table` + :func:`run_merge_arg`), in parallel.

    Use it instead of :func:`singer` when the region is longer than SINGER handles in one run. The
    return value has the **same shape** as :func:`singer` (a posterior-ARG tree sequence, or a list
    of them), so downstream code — :func:`tspaint.paint`, the ensemble merge — is unchanged.

    Parameters
    ----------
    source : tskit.TreeSequence, str, or Variants
        Genotypes, resolved like :func:`singer`'s ``source`` (ts / VCF / VCF-Zarr store /
        :class:`~tspaint.io_genotypes.Variants`). Slice it first with
        :func:`~tspaint.io_genotypes.subset_data` to restrict the region / panel.
    window_size : float
        Contiguous window width (bp); ``[0, L)`` is tiled into non-overlapping windows. Pick a
        width SINGER handles comfortably (≈0.5–2 Mb) — smaller windows parallelise better but the
        per-window ARG ignores linkage across its boundaries.
    ts, mcmc_step, mcmc_burnin, _Ne, _m, _r, _ratio, _ploidy, _polar, singer_bin
        As for :func:`singer` — ``ts`` / ``mcmc_step`` / ``mcmc_burnin`` control the posterior sampling
        identically (the SAME raw samples are run in every window, then burn-in + thinning select the
        ``ts`` members present in all windows to stitch). ``_Ne`` is **required** (see :func:`singer`
        for the ``4·Ne·μ ≈ π`` calibration): estimate it once over the whole region with the all-pairs
        :func:`tspaint.io.estimate_ne` and pass it — the same value is reused for every window.
        ``_seed`` is offset per window (``_seed + window_index``) so windows are independent but
        reproducible.
    mcmc_burnin : int, optional
        Burn-in MCMC iterations discarded before the kept samples (default 200), as for
        :func:`singer`.
    _n_samples, _thin, _seed, _penalty, _hmm_epsilon, _psmc_bins, _fast, _recomb_map, _mut_map
        The remaining SINGER pass-through flags, as for :func:`singer` (``_seed`` is offset per
        window, as above; ``_n_samples`` / ``_thin`` inferred from ``ts`` / ``mcmc_step`` knobs).
    singer_args : list, optional
        Extra raw SINGER tokens appended after tspaint's flags (as for :func:`singer`).
    n_jobs : int, optional
        Worker threads for the window and merge stages (default: the SLURM / CPU core count).
    sequence_length : float, optional
        Override the region length ``L`` (default: the source's, or the max variant position).
    skip_gaps : list[tuple[float, float]], optional
        ``(lo, hi)`` regions to skip entirely (e.g. a centromere) — windows overlapping any are
        neither run nor stitched.
    workdir : str, optional
        Directory for the VCF, per-window tables, and merged ``member_<i>.trees`` (default: a
        fresh tempdir).
    merge_arg_script, merge_python : str, optional
        Path to ``merge_ARG.py`` and the interpreter to run it with (defaults: the SINGER install /
        the current interpreter, which needs ``tskit + numpy + tszip``).
    log : callable, optional
        Progress sink (e.g. ``print``).

    Returns
    -------
    tskit.TreeSequence or list of tskit.TreeSequence
        The stitched posterior ARG(s) — a single tree sequence if only one member survives
        burn-in, else the ensemble (as :func:`singer`).
    """
    import tskit
    from concurrent.futures import ThreadPoolExecutor
    from .parallel import resolve_cores

    n_jobs = max(1, resolve_cores(n_jobs))
    _m, _r = _resolve_singer_rates(_m=_m, _r=_r, _ratio=_ratio, _mut_map=_mut_map, _recomb_map=_recomb_map)
    n_raw, thin, discard, keep = _singer_sampling(ts, mcmc_step, mcmc_burnin, _n_samples, _thin)
    if _Ne is None:
        raise ValueError(_NE_REQUIRED)
    # Resolve sample ids for stamping; keep separate from ``_ploidy`` (SINGER's -ploidy, see singer()).
    source, names, in_ploidy, sidx = _source_sample_ids(source)
    tmp = workdir or tempfile.mkdtemp(prefix="tspaint_singer_win_")
    os.makedirs(tmp, exist_ok=True)

    # 1) one haploid VCF for every window (write once; SINGER reuses <prefix>.vcf in place).
    prefix = os.path.join(tmp, "data")
    L = _write_singer_vcf(source, prefix, sequence_length)
    vcf = prefix + ".vcf"

    # 2) tile [0, L) into contiguous windows; drop any that fall in a skip gap.
    gaps = [(float(lo), float(hi)) for (lo, hi) in (skip_gaps or [])]
    windows, w = [], 0
    while w * window_size < L:
        lo, hi = w * window_size, min((w + 1) * window_size, L)
        if not any(not (hi <= glo or lo >= ghi) for (glo, ghi) in gaps):
            windows.append((w, float(lo), float(hi), os.path.join(tmp, f"w{w}")))
        w += 1
    if not windows:
        raise ValueError("no windows to run (empty region or fully covered by skip_gaps)")
    if log:
        log(f"SINGER: {len(windows)} window(s) × {n_raw} raw sample(s) over [0,{L:g}) on "
            f"{n_jobs} worker(s)")

    # 3) run the windows in parallel — each shells out to SINGER, so it releases the GIL.
    def _run(win):
        _w, s, e, pfx = win
        return set(singer_window(vcf, start=s, end=e, out_prefix=pfx, _Ne=_Ne,
                                 _m=_m, _r=_r, _n_samples=n_raw, _thin=thin, _ploidy=_ploidy,
                                 _seed=_seed + _w, _polar=_polar, _penalty=_penalty,
                                 _hmm_epsilon=_hmm_epsilon, _psmc_bins=_psmc_bins, _recomb_map=_recomb_map,
                                 _mut_map=_mut_map, _fast=_fast, singer_args=singer_args,
                                 singer_bin=singer_bin))
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        idx_sets = list(ex.map(_run, windows))

    # 4) the raw members written by EVERY window, then burn-in + thinning to `ts` (as singer()).
    common = sorted(set.intersection(*idx_sets))
    members = _select(common, discard, keep)
    if not members:
        raise RuntimeError(
            f"no MCMC members common to all {len(windows)} windows after burn-in/thinning "
            f"(ts={ts}, mcmc_step={mcmc_step}, mcmc_burnin={mcmc_burnin}); raise ts or lower mcmc_burnin")
    if log:
        log(f"stitching {len(members)} member(s) across windows")

    # 5) stitch each member across the windows (merge_ARG.py), in parallel; load the results back.
    def _merge(member):
        rows = build_merge_table(windows, member, coords="local")
        out = os.path.join(tmp, f"member_{member}.trees")
        run_merge_arg(rows, out, script=merge_arg_script, python=merge_python)
        return out
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        paths = list(ex.map(_merge, members))
    ensemble = [attach_sample_ids(tskit.load(p), names, in_ploidy, sample_index=sidx) for p in paths]
    return ensemble[0] if len(ensemble) == 1 else ensemble


def singer_tree_sequences(ts, **kwargs):
    """Deprecated alias for :func:`singer` (emits a :class:`DeprecationWarning`).

    Parameters
    ----------
    ts : tskit.TreeSequence or str
        The genotype ``source`` forwarded to :func:`singer` as its first argument. Named ``ts``
        for historical reasons — it is the *source*, **not** :func:`singer`'s ``ts`` sample-count
        knob.
    **kwargs
        Forwarded verbatim to :func:`singer`.

    Returns
    -------
    tskit.TreeSequence or list of tskit.TreeSequence
        Whatever :func:`singer` returns (a single posterior tree sequence, or a list of them).
    """
    warnings.warn("tspaint.io.singer_tree_sequences is deprecated; use tspaint.io.singer",
                  DeprecationWarning, stacklevel=2)
    return singer(ts, **kwargs)
