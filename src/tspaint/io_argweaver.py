"""ARGweaver posterior-ARG front end — an alternative to SINGER (CLAUDE.md §7.4).

ARGweaver (Rasmussen, Hubisz, Gronau & Siepel, 2014, *PLoS Genet* 10(5):e1004342) samples ARGs
from the posterior under the sequentially-Markov coalescent by MCMC — like :func:`tspaint.io.singer`,
a source of posterior ARG **samples** (the ideal input to the ensemble merge, §7.4). Its sampler
``arg-sample`` reads a ``.sites`` file and writes ``<out>.<i>.smc.gz`` local-tree samples; this
module writes the ``.sites``, runs the sampler, and converts each ``.smc(.gz)`` sample into a tskit
tree sequence — so ``argweaver`` and ``singer`` are drop-in interchangeable front ends.

Pipeline: source (ts | VCF | VCF-Zarr | Variants) → ``.sites`` (one column per sample **haplotype**,
sample order preserved by name) → ``arg-sample`` → ``.smc.gz`` → tskit. Node times are in
generations. The SMC format numbers every node and **keeps a node's number+age across successive
local trees until an SPR changes it**, so the conversion preserves the cross-tree node-ID stability
the edge-blocked accumulator relies on (CLAUDE.md §5), keyed here on ``(argweaver node, age)``.

**No mutations.** ARGweaver's ``.smc`` records only the local *trees*, not where mutations fall, so
the returned tree sequences carry no sites/mutations. This is fine for :func:`tspaint.paint`, which
paints from genealogies (tip labels + branch lengths), not from mutations.

External binary ``arg-sample`` — build from source (https://mdrasmus.github.io/argweaver/:
``make``; only the C++ binary is needed, not the Python-2 package) or ``tspaint install argweaver``.
Path via ``$TSPAINT_ARGWEAVER`` or the install location.
"""
from __future__ import annotations

import glob
import gzip
import os
import re
import subprocess
import tempfile

import numpy as np

from .ids import attach_sample_ids
from .io_singer import repo_root, _tools_dir, _source_sample_ids

__all__ = ["argweaver", "write_sites", "argweaver_install_dir", "argweaver_binary_path"]


def argweaver_install_dir():
    """Clone root that ``tspaint install argweaver`` builds into (``<tools-dir>/argweaver``)."""
    return os.path.join(_tools_dir(), "argweaver")


def argweaver_binary_path():
    """Path to the ``arg-sample`` binary built by ``tspaint install argweaver``."""
    return os.path.join(argweaver_install_dir(), "bin", "arg-sample")


#: ARGweaver sampler binary: ``$TSPAINT_ARGWEAVER`` if set, else the install location.
DEFAULT_ARGWEAVER = os.environ.get("TSPAINT_ARGWEAVER") or argweaver_binary_path()

#: Raised when ``Ne`` is omitted: ARGweaver's ``arg-sample`` requires ``-N`` (as SINGER requires
#: ``-Ne``), so tspaint does not estimate one silently. The user picks the value, optionally via
#: :func:`tspaint.io.estimate_ne`.
_NE_REQUIRED = (
    "argweaver requires Ne: arg-sample needs -N and errors without it. Estimate one from the data "
    "and pass it, e.g.\n"
    "    Ne = tspaint.io.estimate_ne(source, mutation_rate, groups=labels)  # optional exclude=soft_refs\n"
    "    tspaint.io.argweaver(source, Ne=Ne, mutation_rate=mutation_rate, recombination_rate=r)")

# ARGweaver treats the sequences as DNA; for a biallelic 0/1 site we emit two distinct bases (real
# REF/ALT when they are single A/C/G/T characters, else a canonical pair), and 'N' for missing.
_BASES = "ACGT"


def _allele_char(allele, index):
    """One DNA character for allele ``index`` at a site (real base if ACGT, else canonical)."""
    if isinstance(allele, str) and len(allele) == 1 and allele in _BASES:
        return allele
    return _BASES[index % 4]


def write_sites(source, path, *, sequence_length=None, names=None):
    """Write an ARGweaver ``.sites`` file — one haplotype column per sample, variant rows only.

    Parameters
    ----------
    source : tskit.TreeSequence or tspaint Variants
        Genotypes. A tree sequence is read via its variants; a :class:`~tspaint.io_genotypes.Variants`
        supplies positions / genotypes / alleles directly.
    path : str
        Destination ``.sites`` path.
    sequence_length : float, optional
        Region length for the ``REGION`` line (default: the source's).
    names : list[str], optional
        Per-haplotype names for the ``NAMES`` line (default: ``n0..n{H-1}``).

    Returns
    -------
    int
        The sequence length written (ARGweaver ``END``).
    """
    import tskit
    if isinstance(source, tskit.TreeSequence):
        H = source.num_samples
        L = int(sequence_length or source.sequence_length)
        rows = []
        for var in source.variants():
            g = var.genotypes
            if g.min() == g.max():                          # invariant on the sample set: skip
                continue
            alleles = var.alleles
            chars = [("N" if gi < 0 else _allele_char(alleles[gi], gi)) for gi in g]
            rows.append((int(var.site.position) + 1, "".join(chars)))
    else:
        from .io_genotypes import resolve_variants
        v = resolve_variants(source)
        G = np.asarray(v.genotypes)
        H = v.num_haplotypes
        L = int(sequence_length or v.sequence_length)
        miss = None if v.missing is None else np.asarray(v.missing, bool)
        rows = []
        for s in range(v.num_sites):
            col = G[s]
            if col.min() == col.max():
                continue
            ref, alt = (v.alleles[s] + ("A", "C"))[:2] if v.alleles else ("A", "C")
            chars = [_allele_char(alt if gi else ref, int(gi)) for gi in col]
            if miss is not None:
                for h in np.nonzero(miss[s])[0]:
                    chars[h] = "N"
            rows.append((int(v.positions[s]) + 1, "".join(chars)))
    names = list(names) if names is not None else [f"n{i}" for i in range(H)]
    if len(names) != H:
        raise ValueError(f"names has {len(names)} entries != {H} haplotypes")
    with open(path, "w") as f:
        f.write("NAMES\t" + "\t".join(names) + "\n")
        f.write(f"REGION\tchr\t1\t{L}\n")
        for pos, bases in rows:
            f.write(f"{pos}\t{bases}\n")
    return L


def _parse_smc_newick(s):
    """Parse an ARGweaver SMC Newick into ``(edges, ages, leaves)``.

    ``edges`` are ``(parent_name, child_name)`` (names are ARGweaver's integer node labels as
    strings); ``ages`` maps every node name to its ``[&&NHX:age=…]`` age (generations); ``leaves``
    is the set of leaf node names. The trees are binary with numeric internal-node labels.
    """
    s = s.strip().rstrip(";")
    pos = 0
    edges, ages, leaves = [], {}, set()

    def node():
        nonlocal pos
        children = []
        if s[pos] == "(":
            pos += 1
            children.append(node())
            while s[pos] == ",":
                pos += 1
                children.append(node())
            pos += 1                                        # consume ')'
        start = pos
        while pos < len(s) and s[pos] not in ":[,()":
            pos += 1
        name = s[start:pos]
        if pos < len(s) and s[pos] == ":":                 # :branch_length (ignored; ages drive time)
            pos += 1
            while pos < len(s) and s[pos] not in "[,()":
                pos += 1
        age = 0.0
        if pos < len(s) and s[pos] == "[":                 # [&&NHX:age=…]
            end = s.index("]", pos)
            m = re.search(r"age=([0-9eE.+\-]+)", s[pos + 1:end])
            if m:
                age = float(m.group(1))
            pos = end + 1
        ages[name] = age
        if children:
            for c in children:
                edges.append((name, c))
        else:
            leaves.add(name)
        return name

    node()
    return edges, ages, leaves


def read_argweaver_smc(path, *, orig_names=None):
    """Convert one ARGweaver ``.smc`` / ``.smc.gz`` sample into a tskit tree sequence.

    Each ``TREE start end <newick>`` block is one non-recombining interval ``[start-1, end)``
    (ARGweaver coordinates are 1-based, end-inclusive). A tskit node is created per distinct
    ``(ARGweaver node label, age)`` — so a node that keeps its label **and** age across successive
    intervals is one long-span tskit node (persistence; CLAUDE.md §5), and one that is re-timed by an
    SPR becomes a fresh node from that interval on. Sample nodes are placed in ``orig_names`` order
    (the ``.sites`` order), so per-sample labels/truth transfer by id even though SMC may reorder the
    ``NAMES`` line.

    Parameters
    ----------
    path : str
        The ``.smc`` or ``.smc.gz`` file.
    orig_names : list[str], optional
        Original per-haplotype names (as written to ``.sites``); used to restore sample order.
        Defaults to the SMC ``NAMES`` order.

    Returns
    -------
    tskit.TreeSequence
        The sample's marginal trees (node times in generations; no sites/mutations).
    """
    import tskit
    opener = gzip.open if str(path).endswith(".gz") else open
    smc_names, region, intervals = None, None, []
    with opener(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            if parts[0] == "NAMES":
                smc_names = parts[1:]
            elif parts[0] == "REGION":
                region = (parts[1], int(parts[2]), int(parts[3]))
            elif parts[0] == "TREE":
                edges, ages, leaves = _parse_smc_newick(parts[3])
                intervals.append((int(parts[1]), int(parts[2]), edges, ages))
    if smc_names is None or region is None or not intervals:
        raise ValueError(f"{path}: not a valid SMC file (missing NAMES / REGION / TREE)")

    n = len(smc_names)
    L = float(region[2])                                    # REGION end (1-based inclusive) = length
    orig = list(orig_names) if orig_names is not None else list(smc_names)
    # SMC leaf label str(i) -> the original-order sample index of the sequence named smc_names[i].
    orig_index = {name: k for k, name in enumerate(orig)}
    leaf_to_sample = {str(i): orig_index.get(smc_names[i], i) for i in range(n)}

    tables = tskit.TableCollection(sequence_length=L)
    for _ in range(n):                                      # n sample nodes, original order, time 0
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    internal = {}                                           # (label, rounded age) -> tskit node id

    def node_id(name, age):
        if name in leaf_to_sample:                         # a leaf (label 0..n-1)
            return leaf_to_sample[name]
        key = (name, round(float(age), 6))
        nid = internal.get(key)
        if nid is None:
            nid = tables.nodes.add_row(time=float(age))
            internal[key] = nid
        return nid

    open_edges = {}                                         # (p, c) -> [left, right] currently open
    for (start, end, edges, ages) in intervals:
        left, right = float(start - 1), float(end)
        for (pname, cname) in edges:
            key = (node_id(pname, ages[pname]), node_id(cname, ages[cname]))
            span = open_edges.get(key)
            if span is not None and span[1] == left:       # contiguous with the open span -> extend
                span[1] = right
            else:
                if span is not None:
                    tables.edges.add_row(left=span[0], right=span[1], parent=key[0], child=key[1])
                open_edges[key] = [left, right]
    for (p, c), (l, r) in open_edges.items():
        tables.edges.add_row(left=l, right=r, parent=p, child=c)

    # ARGweaver discretises node ages (the --ntimes grid), so a parent and child can land on the
    # same grid time; tskit requires time[parent] > time[child] strictly. Nudge parents just above
    # any equal-age child by a tiny epsilon (a fixed-point sweep up the tree). Raw ages are otherwise
    # preserved, so branch lengths / the CTMC timescale are intact.
    pairs = list(open_edges.keys())
    flags = tables.nodes.flags
    times = tables.nodes.time.astype(float).copy()
    eps = 1e-3
    changed = True
    while changed:
        changed = False
        for (p, c) in pairs:
            if times[p] <= times[c]:
                times[p] = times[c] + eps
                changed = True
    tables.nodes.set_columns(flags=flags, time=times)
    tables.sort()
    return tables.tree_sequence()


def _run_argweaver(sites, out_prefix, *, N, recombination_rate, mutation_rate, ntimes=20,
                   maxtime=200e3, compress=1, iters=100, sample_step=None, seed=None,
                   argweaver_args=None, argweaver_bin=None, log=None):
    """Invoke ``arg-sample`` on a ``.sites`` file; return the sample-iteration indices written.

    Builds the command from ARGweaver's own flags, each passed **as-is**: ``-N -r -m --ntimes
    --maxtime -c -n`` (and ``--sample-step`` / ``--randseed`` when given), plus any
    ``argweaver_args`` passthrough. Writes ``{out_prefix}.<i>.smc.gz``.
    """
    argweaver_bin = argweaver_bin or DEFAULT_ARGWEAVER
    if not os.path.exists(argweaver_bin):
        raise FileNotFoundError(
            f"arg-sample binary not found at {argweaver_bin}; set TSPAINT_ARGWEAVER, pass "
            f"argweaver_bin=, or build it (`tspaint install argweaver`)")
    cmd = [argweaver_bin, "-s", sites, "-o", out_prefix, "-N", str(N),
           "-r", str(recombination_rate), "-m", str(mutation_rate),
           "--ntimes", str(int(ntimes)), "--maxtime", str(maxtime), "-c", str(int(compress)),
           "-n", str(int(iters)), "--overwrite"]
    if sample_step is not None:
        cmd += ["--sample-step", str(int(sample_step))]
    if seed is not None:
        cmd += ["--randseed", str(int(seed))]
    if argweaver_args:
        cmd += [str(a) for a in argweaver_args]
    if log:
        log("arg-sample " + " ".join(cmd[1:]))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        tail = (res.stdout or res.stderr or "").strip()[-1500:]
        raise RuntimeError(
            f"arg-sample failed (exit {res.returncode}) on {sites}. Last output:\n{tail}")
    return _argweaver_indices(out_prefix)


def _argweaver_indices(out_prefix):
    """Sorted sample-iteration indices ``i`` for which ``{out_prefix}.<i>.smc(.gz)`` exists."""
    idxs = set()
    for p in glob.glob(out_prefix + ".*.smc*"):
        m = re.search(r"\.(\d+)\.smc(?:\.gz)?$", p)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def _smc_path(out_prefix, i):
    """The written ``.smc.gz`` (preferred) or ``.smc`` path for sample iteration ``i``."""
    gzp = f"{out_prefix}.{i}.smc.gz"
    return gzp if os.path.exists(gzp) else f"{out_prefix}.{i}.smc"


def argweaver(source, *, Ne=None, mutation_rate=None, recombination_rate=None, ntimes=20,
              maxtime=200e3, compress=1, iters=100, sample_step=None, burn_in=0, thin=1, seed=None,
              argweaver_args=None, workdir=None, argweaver_bin=None, sequence_length=None,
              names=None, log=None,
              N=None, r=None, m=None, n=None):
    """Sample posterior ARGs from genotypes via ARGweaver (an alternative to :func:`singer`).

    Runs ARGweaver's ``arg-sample`` MCMC and returns its post-burn-in samples as tskit tree
    sequences — the same shape as :func:`singer` (a single tree sequence, or a list), so the
    ensemble merge / :func:`tspaint.paint` are unchanged.

    The options mirror ARGweaver's own ``arg-sample`` flags, passed **as-is**: ``Ne`` (``-N``,
    **required** — see below), ``mutation_rate`` (``-m``), ``recombination_rate`` (``-r``),
    ``ntimes`` (``--ntimes``), ``maxtime`` (``--maxtime``), ``compress`` (``-c``), ``iters``
    (``-n`` MCMC iterations), ``sample_step`` (``--sample-step``), ``seed`` (``--randseed``).
    ARGweaver's short names ``N`` / ``m`` / ``r`` / ``n`` are accepted as aliases (the ARGweaver-flag
    name wins if both are given); ``argweaver_args`` is a passthrough for anything else.

    Parameters
    ----------
    source : tskit.TreeSequence or str
        Genotypes — a tree sequence carrying mutations, a **VCF Zarr** store, or a **VCF** file
        (normalised by :mod:`tspaint.io_genotypes`); written out as an ARGweaver ``.sites`` file.
    Ne : float
        Diploid effective population size (``arg-sample -N``). **Required** — ARGweaver needs ``-N``
        and errors without it, and tspaint does not estimate one silently. Get one from
        :func:`tspaint.io.estimate_ne` (``pi / 4mu``) and pass it. Omitting it raises ``ValueError``.
    mutation_rate, recombination_rate : float
        Per-base mutation (``-m``) and recombination (``-r``) rates.
    ntimes, maxtime : int, float, optional
        Discretised time grid: ``--ntimes`` steps up to ``--maxtime`` generations (defaults 20, 2e5).
    compress : int, optional
        Block compression in base pairs (``-c``, default 1 = none). **ARGweaver is much slower than
        SINGER, with cost roughly ``sequence_length / compress``**, so raise ``compress`` (e.g.
        10–100) and/or keep regions short (≈≤10–50 kb) — ``-c 1`` on a Mb-scale region will not
        finish in reasonable time. For large data prefer :func:`singer`.
    iters : int, optional
        Number of MCMC iterations (``-n``, default 100).
    sample_step : int, optional
        Save a sample every this many iterations (``--sample-step``); ARGweaver's default if ``None``.
    burn_in : int, optional
        Number of leading saved samples to discard (default 0).
    thin : int, optional
        Keep every ``thin``-th sample after burn-in (default 1 = all).
    seed : int, optional
        Random seed (``--randseed``).
    workdir : str, optional
        Working directory for the ``.sites`` and ``.smc.gz`` (default: a fresh tempdir).
    argweaver_bin : str, optional
        Path to ``arg-sample`` (default ``DEFAULT_ARGWEAVER`` / ``$TSPAINT_ARGWEAVER``).
    sequence_length : float, optional
        Override the region length (VCF / Zarr sources; defaults to the max variant position).
    names : list[str], optional
        Per-haplotype names for the ``.sites`` NAMES line (default: the source's, else ``n0..``).
    log : callable, optional
        Progress sink (e.g. ``print``).

    Returns
    -------
    tskit.TreeSequence or list of tskit.TreeSequence
        Post-burn-in posterior samples (no sites/mutations — ARGweaver's ``.smc`` carries only
        trees, which is all :func:`tspaint.paint` needs). Sample ids are stamped onto the sample
        nodes (:func:`tspaint.ids.attach_sample_ids`) for a VCF / Zarr / ``Variants`` source.

    Raises
    ------
    ValueError
        If ``Ne`` is omitted.
    FileNotFoundError
        If the ``arg-sample`` binary is absent.
    RuntimeError
        If ``arg-sample`` exits nonzero.
    """
    Ne = N if N is not None else Ne
    mutation_rate = m if m is not None else mutation_rate
    recombination_rate = r if r is not None else recombination_rate
    iters = n if n is not None else iters
    if Ne is None:
        raise ValueError(_NE_REQUIRED)
    if mutation_rate is None or recombination_rate is None:
        raise ValueError("argweaver needs mutation_rate (-m) and recombination_rate (-r)")

    source, src_names, in_ploidy = _source_sample_ids(source)
    tmp = workdir or tempfile.mkdtemp(prefix="tspaint_argweaver_")
    os.makedirs(tmp, exist_ok=True)
    sites = os.path.join(tmp, "data.sites")
    site_names = names if names is not None else src_names
    write_sites(source, sites, sequence_length=sequence_length, names=site_names)
    # The .sites NAMES order (what the converter maps SMC leaves back to):
    with open(sites) as f:
        orig_names = f.readline().rstrip("\n").split("\t")[1:]

    out = os.path.join(tmp, "out")
    _run_argweaver(sites, out, N=Ne, recombination_rate=recombination_rate,
                   mutation_rate=mutation_rate, ntimes=ntimes, maxtime=maxtime, compress=compress,
                   iters=iters, sample_step=sample_step, seed=seed, argweaver_args=argweaver_args,
                   argweaver_bin=argweaver_bin, log=log)

    kept = _argweaver_indices(out)[burn_in:][:: max(1, int(thin))]
    samples = []
    for i in kept:
        arg = read_argweaver_smc(_smc_path(out, i), orig_names=orig_names)
        samples.append(attach_sample_ids(arg, src_names, in_ploidy))
    if not samples:
        raise RuntimeError(
            f"no post-burn-in ARGweaver samples (iters={iters}, burn_in={burn_in}); "
            f"raise iters / lower burn_in")
    return samples[0] if len(samples) == 1 else samples
