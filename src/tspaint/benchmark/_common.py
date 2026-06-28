"""Shared plumbing for the external-LAI benchmark bridges (CLAUDE.md §9, §10).

The four comparators (RFMix, gnomix, SALAI-Net, Recomb-Mix) are all **supervised, diploid,
genotype-native** local-ancestry callers: each consumes a phased query VCF, a phased reference
VCF, and a reference sample→ancestry map (three of four also a genetic map), and emits
per-haplotype ancestry calls. This module normalises *our* side of that contract so every
bridge in :mod:`tspaint.benchmark` shares one input resolver, one set of input writers, one
subprocess launcher, and one output path back into the tspaint ``.npz`` painting format.

Input (two interchangeable shapes, the user picks):

* **two files** — a query VCF and a separate reference VCF, harmonised onto their shared
  biallelic sites (:func:`resolve_panel`);
* **one combined VCF** — queries and references together, split by the sample map (every mapped
  sample is a reference, the rest are queries).

Output is keyed by **query haplotype index** ``2*j + h`` (query sample ``j`` in VCF-column order,
haplotype ``h ∈ {0, 1}``) — a tree-free, recoverable key. The original ``sample.hap`` name of each
key is stored alongside the painting (:func:`save_tracks`) so the table joins back to the VCF.

External tools run in **their own environments** (each ships its own ``pixi.toml``); we shell out
via ``pixi run --manifest-path <tool-dir> …`` (or the tool's binary), every invocation overridable
by an environment variable so the bridge is testable and relocatable.
"""
from __future__ import annotations

import gzip
import os
import shlex
import subprocess
from dataclasses import dataclass, field

import numpy as np

from ..output import Segment, INFORMATIVE, MISSING_INFO

__all__ = [
    "Panel", "resolve_panel", "read_sample_map", "assign_states",
    "write_phased_vcf", "write_sample_map", "write_genetic_map",
    "query_hap_names", "fill_missing", "save_tracks", "one_hot", "parse_inds", "setup_inputs",
    "tool_command", "run_tool", "tool_available", "require",
    "TOOLS_DIR", "GNOMIX_DIR", "SALAI_DIR", "RECOMBMIX_BIN", "RFMIX_BIN",
]


# --- minimal VCF reader (biallelic GT; one column per haplotype) ----------------------------

@dataclass
class _VCF:
    positions: np.ndarray          # (S,) int64, strictly increasing
    geno: np.ndarray               # (S, n_samples * ploidy) int8 — column = sample*ploidy + hap
    alleles: list                  # per-site (ref, alt)
    samples: list                  # base sample names (len n_samples)
    ploidy: int
    contig: str

    @property
    def sequence_length(self):
        return float(self.positions.max() + 1) if self.positions.size else 1.0


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith((".gz", ".bgz")) else open(path, "rt")


def read_vcf(path):
    """Read a biallelic-``GT`` VCF into :class:`_VCF` (pure-Python; CLAUDE.md §5 scope).

    One column per haplotype, ``sample*ploidy + hap`` order; ``.`` and multiallelic ALT are
    handled as in :func:`tspaint.io_genotypes.variants_from_vcf` (``.`` → REF, multiallelic
    skipped). Ploidy and contig are taken from the first data record.
    """
    positions, genos, alleles = [], [], []
    samples, ploidy, contig = None, None, None
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            ref, alt = cols[3], cols[4]
            if "," in alt:                                  # multiallelic — skipped in v1
                continue
            fmt = cols[8].split(":") if len(cols) > 8 else ["GT"]
            gi = fmt.index("GT") if "GT" in fmt else 0
            row = []
            for cell in cols[9:]:
                gt = cell.split(":")[gi] if cell else "."
                alle = gt.replace("|", "/").split("/")
                if ploidy is None:
                    ploidy = len(alle)
                row.extend(1 if a not in (".", "0") else 0 for a in alle)
            if contig is None:
                contig = cols[0]
            positions.append(int(cols[1]))
            genos.append(row)
            alleles.append((ref, alt))
    if not positions:
        raise ValueError(f"no biallelic GT records parsed from {path}")
    if samples is None:
        raise ValueError(f"no #CHROM header in {path}")
    pos = np.asarray(positions, np.int64)
    order = np.argsort(pos, kind="stable")
    return _VCF(positions=pos[order], geno=np.asarray(genos, np.int8)[order],
                alleles=[alleles[i] for i in order], samples=list(samples),
                ploidy=int(ploidy), contig=str(contig))


# --- sample map (reference sample -> ancestry) ----------------------------------------------

def read_sample_map(path):
    """Read a sample map ``<sample>\\t<label>`` (``#``/blank lines skipped) → ``[(name, label)]``.

    Order is preserved (it determines integer-state assignment for non-integer labels;
    :func:`assign_states`).
    """
    pairs = []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"sample-map line needs 2 columns: {line!r}")
            pairs.append((parts[0], parts[1]))
    if not pairs:
        raise ValueError(f"empty sample map {path}")
    return pairs


def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def assign_states(pairs):
    """Map reference labels to ancestry-state integers ``0..K-1``.

    Integer-valued labels are taken **literally** (``"0"`` → state 0), so the states match a
    tspaint truth table regardless of row order; non-integer labels are assigned by order of
    first appearance. Returns ``(states: {name: int}, K, state_label: {int: str})``.
    """
    labels = [l for _, l in pairs]
    if all(_is_int(l) for l in labels):
        states = {n: int(l) for n, l in pairs}
        K = max(states.values()) + 1
        state_label = {int(l): str(int(l)) for _, l in pairs}
    else:
        order = []
        for _, l in pairs:
            if l not in order:
                order.append(l)
        idx = {l: i for i, l in enumerate(order)}
        states = {n: idx[l] for n, l in pairs}
        K, state_label = len(order), {i: l for l, i in idx.items()}
    return states, K, state_label


# --- resolved panel (query + reference on a shared site set) --------------------------------

@dataclass
class Panel:
    """Query + reference haplotypes on a shared biallelic site set, ready to write tool inputs.

    Attributes
    ----------
    positions : numpy.ndarray
        ``(S,)`` strictly-increasing integer site positions shared by query and reference.
    geno : numpy.ndarray
        ``(S, Hq + Hr)`` allele matrix; query haplotype columns first, then reference columns.
    alleles : list
        Per-site ``(ref, alt)``.
    query : list
        ``(name, geno_cols=(c0, c1), keys=(k0, k1))`` per query sample; ``keys`` are the
        ``2*j+h`` hap indices the painting is keyed by, ``geno_cols`` index :attr:`geno`.
    ref : list
        ``(name, geno_cols=(c0, c1), state)`` per reference sample.
    K : int
        Number of ancestry states.
    contig : str
        Contig label written into the tool VCFs / maps.
    sequence_length : float
        ``positions.max() + 1`` — paintings cover ``[0, sequence_length)``.
    state_label : dict
        ``{state: original-label}`` for output decoding / provenance.
    """
    positions: np.ndarray
    geno: np.ndarray
    alleles: list
    query: list
    ref: list
    K: int
    contig: str
    sequence_length: float
    state_label: dict = field(default_factory=dict)

    @property
    def n_query_haps(self):
        return 2 * len(self.query)

    @property
    def query_keys(self):
        return [k for q in self.query for k in q[2]]


def _diploid_inds(samples, ploidy, col_base=0, states=None):
    """Build ``(name, geno_cols, extra)`` per sample; requires diploid (ploidy 2)."""
    if ploidy != 2:
        raise ValueError(
            f"benchmark tools require diploid (phased, 2 haplotypes/sample) VCFs; got ploidy "
            f"{ploidy}. Pair haplotypes upstream, or use `tspaint benchmark export-vcf` which "
            "writes diploid VCFs from a (haploid or diploid) tree sequence.")
    inds = []
    for j, name in enumerate(samples):
        cols = (col_base + 2 * j, col_base + 2 * j + 1)
        if states is None:
            inds.append((name, cols))
        else:
            if name not in states:
                continue
            inds.append((name, cols, states[name]))
    return inds


def resolve_panel(query_vcf, ref_vcf=None, *, sample_map):
    """Resolve query + reference haplotypes onto a shared site set (CLAUDE.md §9).

    Two input shapes:

    * ``ref_vcf is None`` — *combined* VCF: ``query_vcf`` holds everyone; every sample named in
      ``sample_map`` is a labelled reference, the rest are queries (same sites by construction).
    * ``ref_vcf`` given — *two files*: query and reference VCFs are intersected on their shared
      integer positions (queries needing no labels), references labelled by ``sample_map``.

    Parameters
    ----------
    query_vcf : str
        Path to the query (or combined) phased diploid VCF.
    ref_vcf : str, optional
        Path to a separate reference VCF; omit for the combined-VCF shape.
    sample_map : str
        Path to ``<ref-sample>\\t<ancestry>`` (:func:`read_sample_map`).

    Returns
    -------
    Panel
    """
    pairs = read_sample_map(sample_map)
    states, K, state_label = assign_states(pairs)

    if ref_vcf is None:
        v = read_vcf(query_vcf)
        ref_names = set(states)
        q_samples = [s for s in v.samples if s not in ref_names]
        r_samples = [s for s in v.samples if s in ref_names]
        if not q_samples:
            raise ValueError("combined VCF: every sample is in the map; no queries left")
        if not r_samples:
            raise ValueError("combined VCF: no sample matches the sample map")
        sidx = {s: i for i, s in enumerate(v.samples)}
        query = [(s, (sidx[s] * 2, sidx[s] * 2 + 1)) for s in q_samples]
        ref = [(s, (sidx[s] * 2, sidx[s] * 2 + 1), states[s]) for s in r_samples]
        positions, geno, alleles, contig, seqlen = (
            v.positions, v.geno, v.alleles, v.contig, v.sequence_length)
        if v.ploidy != 2:
            _diploid_inds(v.samples, v.ploidy)                       # raise the clear error
    else:
        q, r = read_vcf(query_vcf), read_vcf(ref_vcf)
        if q.ploidy != 2:
            _diploid_inds(q.samples, q.ploidy)
        if r.ploidy != 2:
            _diploid_inds(r.samples, r.ploidy)
        positions, qi, ri = _intersect(q.positions, r.positions)
        if positions.size == 0:
            raise ValueError("query and reference VCFs share no positions")
        qgeno, rgeno = q.geno[qi], r.geno[ri]
        geno = np.hstack([qgeno, rgeno])
        alleles = [q.alleles[i] for i in qi]
        contig, seqlen = q.contig, float(positions.max() + 1)
        query = [(s, (2 * j, 2 * j + 1)) for j, s in enumerate(q.samples)]
        Hq = qgeno.shape[1]
        ref = [(s, (Hq + 2 * j, Hq + 2 * j + 1), states[s])
               for j, s in enumerate(r.samples) if s in states]
        if not ref:
            raise ValueError("no reference VCF sample matches the sample map")

    query = [(name, cols, (2 * j, 2 * j + 1)) for j, (name, cols) in enumerate(query)]
    return Panel(positions=_strictly_increasing(positions), geno=geno, alleles=alleles,
                 query=query, ref=ref, K=K, contig=str(contig),
                 sequence_length=float(seqlen), state_label=state_label)


def _intersect(pa, pb):
    """Shared positions of two sorted integer arrays → ``(positions, idx_a, idx_b)``."""
    common = np.intersect1d(pa, pb, assume_unique=False)
    ia = {int(p): i for i, p in enumerate(pa)}
    ib = {int(p): i for i, p in enumerate(pb)}
    idx_a = np.array([ia[int(p)] for p in common], np.int64)
    idx_b = np.array([ib[int(p)] for p in common], np.int64)
    return common.astype(np.int64), idx_a, idx_b


def _strictly_increasing(pos):
    """Force strictly-increasing integer positions (tools reject duplicates/zeros)."""
    out = np.empty(pos.shape[0], np.int64)
    last = 0
    for i, p in enumerate(pos):
        v = max(int(p), last + 1)
        out[i] = v
        last = v
    return out


# --- tool input writers ---------------------------------------------------------------------

def write_phased_vcf(path, panel, inds, *, names=None):
    """Write a phased biallelic VCF: one diploid column ``c0|c1`` per ``inds`` entry."""
    names = names or [e[0] for e in inds]
    cols = [e[1] for e in inds]
    geno, pos, contig = panel.geno, panel.positions, panel.contig
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={contig},length={int(panel.sequence_length)}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n")
        for s in range(geno.shape[0]):
            ref, alt = panel.alleles[s]
            gts = "\t".join(f"{geno[s, c0]}|{geno[s, c1]}" for (c0, c1) in cols)
            f.write(f"{contig}\t{int(pos[s])}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{gts}\n")


def write_sample_map(path, panel, *, header=None):
    """Write the reference sample map ``<name>\\t<state>`` (state = the integer ancestry)."""
    with open(path, "w") as f:
        if header:
            f.write(header + "\n")
        for (name, _cols, state) in panel.ref:
            f.write(f"{name}\t{state}\n")


def write_genetic_map(path, panel, recomb_rate, *, fmt="plink", header=False):
    """Write a uniform (linear) genetic map at the panel's sites: ``cM = pos · r · 100``.

    ``fmt="plink"`` writes 3 columns ``chrom  pos(bp)  cM`` (RFMix / gnomix); ``fmt="hapmap"``
    writes 4 columns ``Chromosome  Position(bp)  Rate(cM/Mb)  Map(cM)`` with a header (Recomb-Mix).
    Pass an explicit ``--genetic-map`` to a runner to use a real map instead.
    """
    contig = panel.contig
    with open(path, "w") as f:
        if fmt == "hapmap":
            f.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
            for p in panel.positions:
                cm = float(p) * recomb_rate * 100.0
                f.write(f"{contig}\t{int(p)}\t{recomb_rate * 1e8:.6f}\t{cm:.10f}\n")
        else:
            if header:
                f.write("#chm\tpos\tcM\n")
            for p in panel.positions:
                f.write(f"{contig}\t{int(p)}\t{float(p) * recomb_rate * 100.0:.10f}\n")


# --- output: keys, names, missing-fill, save ------------------------------------------------

def query_hap_names(panel):
    """``{hap-key: "<sample>.<hap>"}`` for every query haplotype (round-trips keys → VCF)."""
    out = {}
    for (name, _cols, (k0, k1)) in panel.query:
        out[k0] = f"{name}.0"
        out[k1] = f"{name}.1"
    return out


def fill_missing(tracks, panel):
    """Ensure every query hap key is present; absent ones get a single MISSING_INFO span.

    Guarantees the painting covers all query haplotypes over ``[0, L)`` even if the tool
    dropped one (so the ``.npz`` and downstream scoring see the full panel).
    """
    L, K = panel.sequence_length, panel.K
    for k in panel.query_keys:
        if not tracks.get(k):
            tracks[k] = [Segment(0.0, L, np.full(K, 1.0 / K), MISSING_INFO)]
    return tracks


def save_tracks(path, tracks, panel, *, deadband=0.0):
    """Write ``tracks`` (keyed by query hap index) as a tspaint ``.npz`` painting.

    Reuses :func:`tspaint.serialize.save_painting` (same ``tspaint-painting`` format as the
    native painter) and stores the per-key ``<sample>.<hap>`` names for traceability.
    """
    from ..serialize import save_painting
    save_painting(path, tracks, seqlen=panel.sequence_length, deadband=deadband,
                  sample_names=query_hap_names(panel))


def one_hot(state, K):
    p = np.zeros(K)
    p[int(state)] = 1.0
    return p


def parse_inds(panel):
    """The ``[(name, (key0, key1))]`` view of the query samples (for the output parsers)."""
    return [(name, keys) for (name, _cols, keys) in panel.query]


def setup_inputs(query_vcf, ref_vcf, sample_map, workdir, *, sample_map_header=None):
    """Resolve the panel and write the common tool inputs (query VCF, ref VCF, sample map).

    Returns ``(panel, query_vcf_path, ref_vcf_path, sample_map_path)``; the genetic map is
    written per-tool by the caller (its format differs). ``workdir`` is created if absent.
    """
    os.makedirs(workdir, exist_ok=True)
    panel = resolve_panel(query_vcf, ref_vcf, sample_map=sample_map)
    qv = os.path.join(workdir, "query.vcf")
    rv = os.path.join(workdir, "reference.vcf")
    sm = os.path.join(workdir, "sample_map.tsv")
    write_phased_vcf(qv, panel, panel.query)
    write_phased_vcf(rv, panel, panel.ref)
    write_sample_map(sm, panel, header=sample_map_header)
    return panel, qv, rv, sm


# --- subprocess launchers (each tool in its own env; all env-overridable) -------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PIXI = os.environ.get("TSPAINT_PIXI", "pixi")

# The git-only comparators are cloned into external/<dir> by `tspaint benchmark setup`
# (gitignored). Relocate the whole root with TSPAINT_TOOLS_DIR, or any one tool with its own
# var. RFMix is a bioconda package (the `compare` pixi feature), not a git clone.
TOOLS_DIR = os.path.expanduser(
    os.environ.get("TSPAINT_TOOLS_DIR", os.path.join(_REPO_ROOT, "external")))
GNOMIX_DIR = os.path.expanduser(os.environ.get("TSPAINT_GNOMIX_DIR", os.path.join(TOOLS_DIR, "gnomix")))
SALAI_DIR = os.path.expanduser(os.environ.get("TSPAINT_SALAI_DIR", os.path.join(TOOLS_DIR, "SALAI-Net")))
RECOMBMIX_BIN = os.path.expanduser(
    os.environ.get("TSPAINT_RECOMBMIX", os.path.join(TOOLS_DIR, "Recomb-Mix", "RecombMix_v0.7")))
RFMIX_BIN = os.environ.get(
    "TSPAINT_RFMIX", os.path.join(_REPO_ROOT, ".pixi", "envs", "compare", "bin", "rfmix"))


def _pixi_prefix(manifest_dir, script):
    return [_PIXI, "run", "--manifest-path", manifest_dir, "python", script]


def tool_command(tool, args, *, bin_override=None):
    """Build the argv to run ``tool`` (``"rfmix"``/``"gnomix"``/``"salai"``/``"recombmix"``).

    Each defaults to the tool's own pixi env / binary but is fully overridable: set
    ``TSPAINT_<TOOL>_CMD`` to a shell string to replace the launcher prefix entirely (e.g.
    ``TSPAINT_GNOMIX_CMD="python /path/gnomix.py"``), the ``TSPAINT_<TOOL>_DIR`` / binary env
    vars above to relocate the install, or pass ``bin_override`` (a binary path for the
    compiled tools) from a runner's ``*_bin`` argument.
    """
    override = os.environ.get(f"TSPAINT_{tool.upper()}_CMD")
    if override:
        return shlex.split(override) + list(args)
    if tool == "rfmix":
        return [bin_override or RFMIX_BIN] + list(args)
    if tool == "gnomix":
        return _pixi_prefix(GNOMIX_DIR, os.path.join(GNOMIX_DIR, "gnomix.py")) + list(args)
    if tool == "salai":
        return _pixi_prefix(SALAI_DIR, os.path.join(SALAI_DIR, "src", "SALAI.py")) + list(args)
    if tool == "recombmix":
        return [bin_override or RECOMBMIX_BIN] + list(args)
    raise ValueError(f"unknown benchmark tool {tool!r}")


def tool_available(tool, *, bin_override=None):
    """Whether ``tool``'s install location exists (binary file or pixi manifest dir)."""
    if os.environ.get(f"TSPAINT_{tool.upper()}_CMD"):
        return True
    return {
        "rfmix": os.path.exists(bin_override or RFMIX_BIN),
        "gnomix": os.path.isdir(GNOMIX_DIR),
        "salai": os.path.isdir(SALAI_DIR),
        "recombmix": os.path.exists(bin_override or RECOMBMIX_BIN),
    }.get(tool, False)


def run_tool(tool, args, *, cwd=None, log=None, bin_override=None):
    """Run ``tool`` with ``args``; raise :class:`RuntimeError` on a nonzero exit (stderr tail)."""
    cmd = tool_command(tool, args, bin_override=bin_override)
    if log is not None:
        log(" ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(
            f"{tool} failed (exit {res.returncode}):\n"
            f"--- stdout ---\n{res.stdout[-1500:]}\n--- stderr ---\n{res.stderr[-1500:]}")
    return res


def require(path, hint):
    """Raise :class:`FileNotFoundError` with ``hint`` unless ``path`` exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — {hint}")
    return path
