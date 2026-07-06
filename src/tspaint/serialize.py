"""On-disk serialization of tspaint results as NumPy ``.npz`` (the CLI / GWF interchange).

Every *computed* artifact a :mod:`tspaint.cli` subcommand writes — fitted params, a painting,
a dating curve, a QC / ghost / archaic table — is stored here as a single ``.npz`` of named
arrays. ``.npz`` is exact (it preserves the IEEE-754 bits of every float), compact, and
reloadable, so a later job (notably ``merge``) recovers the posterior arrays bit-for-bit.

Two of these round-trips reconstruct real objects because the pipeline reloads them:
:func:`load_params` (consumed by ``paint --params``) and :func:`load_painting` (consumed by
``merge``). The rest are terminal outputs whose ``load_*`` returns a plain dict of the stored
arrays — enough to inspect or test, without rebuilding the result dataclass.

Hand-authored *inputs* (labels, sample-id lists) are deliberately **not** here — those stay as
text/JSON so they are easy to edit and template from a workflow (see :mod:`tspaint.cli`).
"""
from __future__ import annotations

import numpy as np

from .output import Segment, INFORMATIVE, MISSING_INFO, DEFAULT_QC_DEADBAND
from .ensemble import MergedSegment

__all__ = [
    "save_params", "load_params",
    "save_painting", "load_painting", "load_painting_meta",
    "save_rate_through_time", "load_rate_through_time",
    "save_reference_qc", "load_reference_qc",
    "save_ghost", "load_ghost",
    "save_archaic", "load_archaic",
    "save_foreign_tracts", "load_foreign_tracts",
]

_STATUS_TO_INT = {INFORMATIVE: 0, MISSING_INFO: 1}
_INT_TO_STATUS = {0: INFORMATIVE, 1: MISSING_INFO}


# --- low-level npz helpers --------------------------------------------------------------------

def _savez(path, **arrays):
    """Write ``arrays`` to exactly ``path`` (a real file handle ⇒ no ``.npz`` auto-append)."""
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def _loadz(path):
    """Load an ``.npz`` written by :func:`_savez` into a plain ``{name: ndarray}`` dict."""
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _check_format(d, expected):
    fmt = str(d.get("_format", ""))
    if fmt != expected:
        raise ValueError(f"not a {expected!r} file (found _format={fmt!r})")


# --- painting tracks <-> flat columns -------------------------------------------------------

def _flatten_tracks(tracks):
    """A ``dict[int -> list[Segment|MergedSegment]]`` to flat per-segment column arrays.

    Stores ``samples_all`` (every key, so samples with no segments survive the round-trip) and,
    when the segments are :class:`~tspaint.ensemble.MergedSegment`\\ s, the ``posterior_std`` /
    ``n_informative`` band columns.
    """
    samples_all = [int(s) for s in tracks.keys()]
    first = next((segs[0] for segs in tracks.values() if segs), None)
    merged = first is not None and hasattr(first, "posterior_std")
    K = int(first.posterior.shape[0]) if first is not None else 0

    samp, left, right, post, status = [], [], [], [], []
    std, ninf = [], []
    for s, segs in tracks.items():
        for seg in segs:
            samp.append(int(s))
            left.append(float(seg.left))
            right.append(float(seg.right))
            post.append(np.asarray(seg.posterior, float))
            status.append(_STATUS_TO_INT[seg.status])
            if merged:
                std.append(np.asarray(seg.posterior_std, float))
                ninf.append(int(seg.n_informative))

    cols = dict(
        samples_all=np.array(samples_all, np.int64),
        sample=np.array(samp, np.int64),
        left=np.array(left, float),
        right=np.array(right, float),
        posterior=np.array(post, float) if post else np.zeros((0, K)),
        status=np.array(status, np.int8),
        K=int(K),
    )
    if merged:
        cols["posterior_std"] = np.array(std, float) if std else np.zeros((0, K))
        cols["n_informative"] = np.array(ninf, np.int64)
    return cols


def _unflatten_tracks(d):
    """Inverse of :func:`_flatten_tracks` — rebuild ``Segment`` / ``MergedSegment`` tracks."""
    tracks = {int(s): [] for s in d["samples_all"]}
    merged = "posterior_std" in d
    samp, left, right = d["sample"], d["left"], d["right"]
    post, status = d["posterior"], d["status"]
    std = d["posterior_std"] if merged else None
    ninf = d["n_informative"] if merged else None
    for i in range(len(samp)):
        s = int(samp[i])
        st = _INT_TO_STATUS[int(status[i])]
        if merged:
            seg = MergedSegment(float(left[i]), float(right[i]), np.array(post[i], float),
                                st, np.array(std[i], float), int(ninf[i]))
        else:
            seg = Segment(float(left[i]), float(right[i]), np.array(post[i], float), st)
        tracks.setdefault(s, []).append(seg)
    return tracks


# --- params (fit -> paint) ------------------------------------------------------------------

def save_params(path, *, Q, pi, w, K, labels, deadband=0.0, estimate_pi=False,
                loglik_history=()):
    """Write the fitted ancestry model so ``paint --params`` can repaint without re-fitting.

    Carries the **labels** and per-tip credibility ``w`` as well as ``Q`` / ``π`` — painting a
    query rebuilds the reference-tip emissions from them (:func:`tspaint.em.build_emissions`),
    so the file is self-contained.
    """
    w = {int(k): float(v) for k, v in (w or {}).items()}
    labels = {int(k): int(v) for k, v in (labels or {}).items()}
    _savez(path, _format="tspaint-params", _version=1,
           Q=np.asarray(Q, float), pi=np.asarray(pi, float),
           K=int(K), deadband=float(deadband), estimate_pi=bool(estimate_pi),
           w_nodes=np.array(sorted(w), np.int64),
           w_vals=np.array([w[k] for k in sorted(w)], float),
           label_nodes=np.array(sorted(labels), np.int64),
           label_states=np.array([labels[k] for k in sorted(labels)], np.int64),
           loglik_history=np.asarray(list(loglik_history), float))


def load_params(path):
    """Reload :func:`save_params` as ``{Q, pi, K, deadband, estimate_pi, w, labels, loglik_history}``."""
    d = _loadz(path)
    _check_format(d, "tspaint-params")
    w = {int(n): float(v) for n, v in zip(d["w_nodes"], d["w_vals"])}
    labels = {int(n): int(s) for n, s in zip(d["label_nodes"], d["label_states"])}
    return dict(Q=d["Q"], pi=d["pi"], K=int(d["K"]), deadband=float(d["deadband"]),
                estimate_pi=bool(d["estimate_pi"]), w=w, labels=labels,
                loglik_history=d["loglik_history"].tolist())


# --- painting (paint -> merge; Painting.save/load) ------------------------------------------

def save_painting(path, tracks, *, Q=None, pi=None, w=None, queries=None, labels=None,
                  seqlen=None, deadband=None, sample_names=None):
    """Write a painting (``dict[int -> list[Segment|MergedSegment]]``) as a flat segment table.

    The optional ``Q/pi/w/queries/labels/seqlen/deadband`` are stored alongside so a full
    :class:`~tspaint.api.Painting` round-trips (:meth:`tspaint.api.Painting.save`); ``merge``
    needs only the table, which :func:`load_painting` returns. ``sample_names`` (``{key: name}``,
    e.g. the benchmark bridges' ``"<sample>.<hap>"`` labels) is stored for traceability and
    returned by :func:`load_painting_meta`.
    """
    cols = _flatten_tracks(tracks)
    meta = {}
    if sample_names:
        sn = {int(k): str(v) for k, v in sample_names.items()}
        meta["name_keys"] = np.array(sorted(sn), np.int64)
        meta["name_vals"] = np.array([sn[k] for k in sorted(sn)], dtype="U")
    if Q is not None:
        meta["Q"] = np.asarray(Q, float)
    if pi is not None:
        meta["pi"] = np.asarray(pi, float)
    if w is not None:
        w = {int(k): float(v) for k, v in w.items()}
        meta["w_nodes"] = np.array(sorted(w), np.int64)
        meta["w_vals"] = np.array([w[k] for k in sorted(w)], float)
    if queries is not None:
        meta["queries"] = np.array([int(q) for q in queries], np.int64)
    if labels is not None:
        labels = {int(k): int(v) for k, v in labels.items()}
        meta["label_nodes"] = np.array(sorted(labels), np.int64)
        meta["label_states"] = np.array([labels[k] for k in sorted(labels)], np.int64)
    if seqlen is not None:
        meta["seqlen"] = float(seqlen)
    if deadband is not None:
        meta["deadband"] = float(deadband)
    _savez(path, _format="tspaint-painting", _version=1, **cols, **meta)


def load_painting(path):
    """Reload :func:`save_painting` as the painting table ``dict[int -> list[Segment|MergedSegment]]``."""
    d = _loadz(path)
    _check_format(d, "tspaint-painting")
    return _unflatten_tracks(d)


def load_painting_meta(path):
    """The model metadata stored with a painting (``Q/pi/w/queries/labels/seqlen/deadband``), if any."""
    d = _loadz(path)
    _check_format(d, "tspaint-painting")
    m = {}
    if "Q" in d:
        m["Q"] = d["Q"]
    if "pi" in d:
        m["pi"] = d["pi"]
    if "w_nodes" in d:
        m["w"] = {int(n): float(v) for n, v in zip(d["w_nodes"], d["w_vals"])}
    if "queries" in d:
        m["queries"] = [int(q) for q in d["queries"]]
    if "label_nodes" in d:
        m["labels"] = {int(n): int(s) for n, s in zip(d["label_nodes"], d["label_states"])}
    if "seqlen" in d:
        m["seqlen"] = float(d["seqlen"])
    if "deadband" in d:
        m["deadband"] = float(d["deadband"])
    if "name_keys" in d:
        m["sample_names"] = {int(k): str(v) for k, v in zip(d["name_keys"], d["name_vals"])}
    return m


# --- dating (rate through time) -------------------------------------------------------------

def save_rate_through_time(path, rtt):
    """Write a :class:`~tspaint.dating.RateThroughTime` (``centers, q_AB, q_BA, D, J``)."""
    _savez(path, _format="tspaint-rtt", _version=1,
           centers=np.asarray(rtt.centers, float),
           q_AB=np.asarray(rtt.q_AB, float), q_BA=np.asarray(rtt.q_BA, float),
           D=np.asarray(rtt.D, float), J=np.asarray(rtt.J, float),
           loglik_history=np.asarray(list(rtt.loglik_history), float))


def load_rate_through_time(path):
    """Reload :func:`save_rate_through_time` as a dict of its arrays."""
    d = _loadz(path)
    _check_format(d, "tspaint-rtt")
    return dict(centers=d["centers"], q_AB=d["q_AB"], q_BA=d["q_BA"],
                D=d["D"], J=d["J"], loglik_history=d["loglik_history"].tolist())


# --- reference QC ---------------------------------------------------------------------------

def save_reference_qc(path, qc, deadband=DEFAULT_QC_DEADBAND):
    """Write a :class:`~tspaint.introgression.ReferenceQC` — the per-reference audit table
    (``ref, label, credibility, is_anchor, foreign_fraction``, least-credible first), the fitted
    ``Q/π``, and each reference's leave-one-out introgression map."""
    rows = qc.summary(deadband)
    maps_cols = _flatten_tracks(qc.maps)
    _savez(path, _format="tspaint-qc", _version=1,
           ref=np.array([r["ref"] for r in rows], np.int64),
           label=np.array([r["label"] for r in rows], np.int64),
           credibility=np.array([r["credibility"] for r in rows], float),
           is_anchor=np.array([r["is_anchor"] for r in rows], bool),
           foreign_fraction=np.array([r["foreign_fraction"] for r in rows], float),
           Q=np.asarray(qc.Q, float), pi=np.asarray(qc.pi, float),
           **{f"map_{k}": v for k, v in maps_cols.items()})


def load_reference_qc(path):
    """Reload :func:`save_reference_qc` as ``{summary: list[dict], Q, pi, maps: dict}``."""
    d = _loadz(path)
    _check_format(d, "tspaint-qc")
    summary = [dict(ref=int(d["ref"][i]), label=int(d["label"][i]),
                    credibility=float(d["credibility"][i]),
                    is_anchor=bool(d["is_anchor"][i]),
                    foreign_fraction=float(d["foreign_fraction"][i]))
               for i in range(len(d["ref"]))]
    maps = _unflatten_tracks({k[len("map_"):]: v for k, v in d.items() if k.startswith("map_")})
    return dict(summary=summary, Q=d["Q"], pi=d["pi"], maps=maps)


# --- ghost / archaic / foreign tracts -------------------------------------------------------

def save_ghost(path, ghost):
    """Write a :class:`~tspaint.GhostResult` — per-locus ``P(ghost)`` + the learned depth-HMM.

    Stores the scalar ``P(ghost)`` per segment (``posterior[1]``); :func:`load_ghost` rebuilds the
    two-state :class:`~tspaint.output.Segment` posterior ``[P(modern), P(ghost)]`` from it. The
    ensemble ``posterior_std`` band, if any, is not persisted.
    """
    samp, left, right, p = [], [], [], []
    for s in sorted(ghost.posteriors):
        for seg in ghost.posteriors[s]:
            samp.append(int(s)); left.append(float(seg.left)); right.append(float(seg.right))
            p.append(float(seg.posterior[1]))
    nodes = sorted(ghost.burden)
    _savez(path, _format="tspaint-ghost", _version=1,
           sample=np.array(samp, np.int64), left=np.array(left, float),
           right=np.array(right, float), p_ghost=np.array(p, float),
           burden_nodes=np.array(nodes, np.int64),
           burden_vals=np.array([ghost.burden[k] for k in nodes], float),
           mu=np.asarray(ghost.mu, float), sd=np.asarray(ghost.sd, float),
           A=np.asarray(ghost.A, float), pi0=np.asarray(ghost.pi0, float),
           loglik_history=np.asarray(list(ghost.loglik_history), float))


def load_ghost(path):
    """Reload :func:`save_ghost` as a dict (``posteriors, burden, mu, sd, A, pi0, loglik_history``).

    ``posteriors`` is rebuilt as per-sample :class:`~tspaint.output.Segment` lists whose
    ``posterior`` is ``[P(modern), P(ghost)]`` (all ``INFORMATIVE``).
    """
    d = _loadz(path)
    _check_format(d, "tspaint-ghost")
    post = {}
    for s, a, b, pg in zip(d["sample"], d["left"], d["right"], d["p_ghost"]):
        post.setdefault(int(s), []).append(
            Segment(float(a), float(b), np.array([1.0 - float(pg), float(pg)]), INFORMATIVE))
    burden = {int(n): float(v) for n, v in zip(d["burden_nodes"], d["burden_vals"])}
    return dict(posteriors=post, burden=burden, mu=d["mu"], sd=d["sd"], A=d["A"], pi0=d["pi0"],
                loglik_history=d["loglik_history"].tolist())


# detect_archaic was renamed detect_ghost; keep the old serializer names as aliases.
save_archaic = save_ghost
load_archaic = load_ghost


def save_foreign_tracts(path, tracts):
    """Write :func:`tspaint.foreign_tracts` output (``dict[int -> list[(left, right, score)]]``)."""
    samp, left, right, score = [], [], [], []
    for s in sorted(tracts):
        for (a, b, sc) in tracts[s]:
            samp.append(int(s)); left.append(float(a)); right.append(float(b)); score.append(float(sc))
    _savez(path, _format="tspaint-foreign", _version=1,
           sample=np.array(samp, np.int64), left=np.array(left, float),
           right=np.array(right, float), score=np.array(score, float))


def load_foreign_tracts(path):
    """Reload :func:`save_foreign_tracts` as ``dict[int -> list[(left, right, score)]]``."""
    d = _loadz(path)
    _check_format(d, "tspaint-foreign")
    out = {}
    for s, a, b, sc in zip(d["sample"], d["left"], d["right"], d["score"]):
        out.setdefault(int(s), []).append((float(a), float(b), float(sc)))
    return out
