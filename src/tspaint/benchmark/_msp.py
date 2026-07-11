"""Parsers for the comparators' output formats → tspaint :class:`~tspaint.output.Segment` tracks.

Three on-disk shapes cover all four tools:

* :func:`parse_fb` — the RFMix / gnomix ``.fb`` forward-backward **posteriors** (per-marker /
  per-window, columns ``<name>:::hap<1|2>:::<pop>``), the calibrated soft output;
* :func:`parse_msp` — the RFMix / gnomix / SALAI-Net ``.msp.tsv`` **hard** windowed calls
  (columns ``<name>.<0|1>``), returned as one-hot (0/1) posteriors;
* :func:`parse_recombmix_segments` — Recomb-Mix's per-haplotype **segment** text
  (``<name>_<hap>  start end state  start end state …``), also one-hot.

Every parser is keyed by an opaque integer (the caller passes the query haplotype index) and
returns ``{key: [Segment]}`` tiling ``[0, sequence_length)`` with adjacent equal segments merged.
"""
from __future__ import annotations

import numpy as np

from ..output import Segment, INFORMATIVE

__all__ = ["parse_fb", "parse_msp", "parse_recombmix_segments"]


def _one_hot(state, K):
    p = np.zeros(K)
    if 0 <= int(state) < K:
        p[int(state)] = 1.0
    return p


def _append(segs, left, right, post, *, atol=0.0):
    """Append ``[left, right)`` with posterior ``post``, merging into the previous if equal."""
    if right <= left:
        return
    if segs and segs[-1].right == left and np.allclose(segs[-1].posterior, post, atol=atol, rtol=0):
        segs[-1].right = right
    else:
        segs.append(Segment(float(left), float(right), np.asarray(post, float), INFORMATIVE))


def _boundaries(starts, sequence_length):
    """Window/marker start positions → tiling boundaries on ``[0, L)`` (first→0, append L)."""
    b = np.empty(len(starts) + 1)
    b[0] = 0.0
    b[1:-1] = starts[1:]
    b[-1] = float(sequence_length)
    return b


# --- RFMix / gnomix .fb posteriors ----------------------------------------------------------

def parse_fb(fb_path, query_inds, K, sequence_length):
    """Parse an RFMix / gnomix ``.fb`` posterior file into per-haplotype Segment tracks.

    Parameters
    ----------
    fb_path : str
        Path to the ``.fb`` / ``.fb.tsv`` file.
    query_inds : list[tuple[str, tuple[int, int]]]
        ``(sample_name, (key_hap1, key_hap2))`` per query sample; the keys are used as the
        output dict keys (hap ``1`` → first key, hap ``2`` → second).
    K : int
        Number of ancestry states.
    sequence_length : float
        Genome length ``L``; tracks tile ``[0, L)``.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype key, a piecewise-constant **soft** painting.
    """
    with open(fb_path) as f:
        lines = [ln for ln in f.read().splitlines() if ln]
    hdr_i = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
    header = lines[hdr_i].split("\t")
    data = np.array([ln.split("\t") for ln in lines[hdr_i + 1:]], dtype=object)
    if data.size == 0:
        return {k: [] for _n, keys in query_inds for k in keys}
    phys = data[:, 1].astype(float)

    colmap, pops = {}, set()
    for j, h in enumerate(header[4:], start=4):
        if ":::" not in h:
            continue
        name, hap, pop = h.split(":::")
        colmap[(name, hap, pop)] = j
        pops.add(pop)
    numeric = all(p.lstrip("-").isdigit() for p in pops)
    pops = sorted(pops, key=lambda p: int(p) if p.lstrip("-").isdigit() else p)   # deterministic order
    # State index for each pop label: its integer label when numeric (matching
    # export/assign_states, which set state = int(label)); else its sorted rank.
    pop_state = {p: (int(p) if numeric else i) for i, p in enumerate(pops)}
    bnd = _boundaries(phys, sequence_length)

    tracks = {}
    for name, (k0, k1) in query_inds:
        for hap_label, key in (("hap1", k0), ("hap2", k1)):
            cols = {pop_state[p]: colmap.get((name, hap_label, p)) for p in pops}
            if any(c is None for c in cols.values()):
                tracks[key] = []
                continue
            post = np.zeros((data.shape[0], K))                      # column p -> state int(p)
            for st, c in cols.items():
                if 0 <= st < K:
                    post[:, st] = data[:, c].astype(float)
            segs = []
            for i in range(post.shape[0]):
                p = post[i, :K]
                s = p.sum()
                _append(segs, bnd[i], bnd[i + 1], p / s if s > 0 else np.full(K, 1.0 / K),
                        atol=1e-12)
            tracks[key] = segs
    return tracks


# --- RFMix / gnomix / SALAI .msp.tsv hard windowed calls ------------------------------------

def parse_msp(msp_path, query_inds, K, sequence_length, *, code_to_state=None):
    """Parse a ``.msp.tsv`` hard windowed call file into one-hot per-haplotype tracks.

    Handles RFMix, gnomix and SALAI-Net ``.msp.tsv`` (same layout): 6 metadata columns
    (``chm spos epos sgpos egpos n_snps``) then one integer-ancestry column per
    ``<sample>.<0|1>`` haplotype, one row per window.

    Parameters
    ----------
    msp_path : str
        Path to the ``.msp.tsv`` file.
    query_inds : list[tuple[str, tuple[int, int]]]
        ``(sample_name, (key_hap0, key_hap1))`` per query sample (``.0`` → first key).
    K : int
        Number of ancestry states.
    sequence_length : float
        Genome length ``L``; tracks tile ``[0, L)``.
    code_to_state : dict[int, int], optional
        Map a tool ancestry code to a tspaint state. Defaults to the ``#Subpopulation
        order/codes:`` header line if present, else identity.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype key, a one-hot (0/1) piecewise-constant painting.
    """
    with open(msp_path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if code_to_state is None:
        code_to_state = {}
        for ln in lines:
            if ln.startswith("#Subpopulation"):
                for tok in ln.split(":", 1)[1].split():
                    if "=" in tok:
                        label, code = tok.split("=")       # .msp writes <name/label>=<code>
                        code_to_state[int(code)] = int(label)
    hdr = next(ln for ln in lines if ln.lstrip("#").startswith("chm"))
    cols = hdr.lstrip("#").split("\t")

    name_to_key = {}
    for name, (k0, k1) in query_inds:
        name_to_key[f"{name}.0"] = k0
        name_to_key[f"{name}.1"] = k1
    col_key = {j: name_to_key[c] for j, c in enumerate(cols) if c in name_to_key}

    data = [ln.split("\t") for ln in lines if not ln.startswith("#")]
    if not data:
        return {k: [] for _n, keys in query_inds for k in keys}
    starts = [float(r[1]) for r in data]
    bnd = _boundaries(starts, sequence_length)

    tracks = {key: [] for key in col_key.values()}
    for j, key in col_key.items():
        segs = tracks[key]
        for k, r in enumerate(data):
            state = code_to_state.get(int(r[j]), int(r[j]))
            _append(segs, bnd[k], bnd[k + 1], _one_hot(state, K))
    for _n, keys in query_inds:                                       # keys with no column
        for key in keys:
            tracks.setdefault(key, [])
    return tracks


# --- Recomb-Mix per-haplotype segment text --------------------------------------------------

def parse_recombmix_segments(path, query_inds, K, sequence_length, *, code_to_state=None):
    """Parse Recomb-Mix's ``admix_inferred_ancestral_values_local.txt`` into one-hot tracks.

    The file is one line per query haplotype::

        #Population label and ID: 0=0\t1=1
        <name>_<hap>  <s1> <e1> <a1>  <s2> <e2> <a2>  …

    where each ``(start, end, ancestry)`` triple is a hard segment.

    Parameters
    ----------
    path : str
        Path to the inferred-ancestry text file.
    query_inds : list[tuple[str, tuple[int, int]]]
        ``(sample_name, (key_hap0, key_hap1))`` per query sample (``_0`` → first key).
    K : int
        Number of ancestry states.
    sequence_length : float
        Genome length ``L``; tracks tile ``[0, L)``.
    code_to_state : dict[int, int], optional
        Map a Recomb-Mix code to a tspaint state; defaults to the ``#Population label and ID:``
        header if present, else identity.

    Returns
    -------
    dict[int, list[Segment]]
        Per query haplotype key, a one-hot (0/1) painting tiling ``[0, L)``.
    """
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if code_to_state is None:
        code_to_state = {}
        for ln in lines:
            if ln.startswith("#") and ":" in ln:
                for tok in ln.split(":", 1)[1].split():
                    if "=" in tok:
                        label, code = tok.split("=")
                        code_to_state[int(code)] = int(label)

    name_to_key = {}
    for name, (k0, k1) in query_inds:
        name_to_key[f"{name}_0"] = k0
        name_to_key[f"{name}_1"] = k1

    tracks = {k: [] for _n, keys in query_inds for k in keys}
    for ln in lines:
        if ln.startswith("#"):
            continue
        parts = ln.split()
        key = name_to_key.get(parts[0])
        if key is None:
            continue
        triples = parts[1:]
        segs = tracks[key]
        n = len(triples) // 3
        for t in range(n):
            s, e, a = triples[3 * t], triples[3 * t + 1], triples[3 * t + 2]
            left = 0.0 if t == 0 else float(s)
            right = float(sequence_length) if t == n - 1 else float(e)
            state = code_to_state.get(int(a), int(a))
            _append(segs, left, right, _one_hot(state, K))
    return tracks
