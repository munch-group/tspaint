"""Sample identity on a tree sequence: stamp source sample IDs, resolve label keys (CLAUDE.md §5).

The inference front ends (:func:`tspaint.io.singer`, :func:`tspaint.io.tsinfer`) return tree
sequences whose sample nodes are anonymous integers in input (VCF-column) order. A user building
``labels`` / ``queries`` / ``soft_refs`` knows their **sample IDs** (from the VCF header /
metadata), not that integer order. So the front ends **stamp** the source's sample IDs onto the
returned tree sequence with :func:`attach_sample_ids`, and the painting API **resolves** ID
strings against them with :func:`resolve_labels` / :func:`resolve_ids`.

Stamped layout
--------------
* one **individual** per sample, ``metadata = {"id": "<base name>"}`` (permissive-JSON schema);
* its haplotype **nodes** linked to it, each ``metadata = {"id": "<base>_<h>"}`` with ``h`` 1-based
  (``"<base>_1"``, ``"<base>_2"``, ...). A **haploid** sample's node id is the base name itself
  (no suffix).

Resolution rules for a label/query key
--------------------------------------
* an ``int`` is a sample-node index, used as-is — so existing integer-keyed dicts pass through
  unchanged (backward compatible);
* a ``str`` matches an **individual** id (→ *all* its haplotype nodes) or a **per-haplotype** node
  id (→ that one node); a digit-only string with no id match falls back to an integer index (so a
  JSON labels file keyed ``{"3": 0}`` still means node 3).

This module imports only ``numpy`` and ``tskit`` (lazily), so the core package stays light.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["attach_sample_ids", "sample_id_index", "resolve_nodes", "resolve_labels",
           "resolve_ids"]

_SUFFIX = re.compile(r"_(\d+)$")


def _split_haplotype(name, ploidy):
    """``(base_id, per_haplotype_id)`` for one flattened sample name.

    For ``ploidy > 1`` the front ends flatten a sample ``S`` into columns ``S_0 .. S_{p-1}``; strip
    that trailing ``_<k>`` to recover the base and re-index 1-based (``S_0 -> S_1``). For haploid
    data the name is the base and the per-haplotype id equals it (no suffix).
    """
    name = str(name)
    if ploidy <= 1:
        return name, name
    base = _SUFFIX.sub("", name)
    m = _SUFFIX.search(name)
    k = int(m.group(1)) if m else 0
    return base, f"{base}_{k + 1}"


def attach_sample_ids(ts, names, ploidy=1, sample_index=None):
    """Stamp source sample IDs onto ``ts`` (individuals + per-haplotype nodes); return a new ts.

    Groups the ``ts`` sample nodes into individuals by base name (so a diploid sample's two
    haplotype columns share one individual) and writes ``{"id": ...}`` metadata under a
    permissive-JSON schema — the identity :func:`resolve_labels` reads back. Uniform across front
    ends: it rebuilds the individuals table and resets non-sample ``individual`` links, and merges
    the per-haplotype id into each **sample** node's metadata while preserving other nodes'
    metadata bytes (e.g. tsinfer's ``ancestor_data_id``).

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence to stamp (its sample nodes are the haplotype columns, in order).
    names : sequence[str] or None
        Per-haplotype sample names aligned to ``ts.samples()`` (i.e. the source's
        :attr:`~tspaint.io_genotypes.Variants.sample_names`). If ``None`` or its length does not
        match the number of sample nodes, ``ts`` is returned unchanged (nothing to stamp).
    ploidy : int, optional
        Haplotypes per sample in ``names`` (used to recover base names). Default ``1``.
    sample_index : sequence[int] or None, optional
        Per-haplotype individual index (:attr:`~tspaint.io_genotypes.Variants.sample_index`); when
        given it groups the haplotype columns into individuals (mixed-ploidy safe, e.g. chrX),
        overriding the scalar ``ploidy``.

    Returns
    -------
    tskit.TreeSequence
        ``ts`` with sample IDs stamped (or ``ts`` unchanged if ``names`` is absent/mismatched).
    """
    import tskit

    samples = [int(s) for s in ts.samples()]
    if not names or len(names) != len(samples):
        return ts
    ploidy = int(ploidy) if ploidy and int(ploidy) > 0 else 1

    # Group sample nodes into individuals, in first-appearance order.
    groups = []                 # [(base, [nodes...]), ...]
    node_id_of = {}             # sample node -> per-haplotype id
    if sample_index is not None and len(sample_index) == len(samples):
        # Mixed-ploidy: group columns by individual index (mirrors io_genotypes._group_columns),
        # so a diploid sample's haplotypes share one individual even when ploidy varies per sample.
        by_ind = {}
        for col, k in enumerate(int(i) for i in sample_index):
            by_ind.setdefault(k, []).append(col)
        for k in sorted(by_ind):
            cols = by_ind[k]
            multi = len(cols) > 1
            base = _SUFFIX.sub("", str(names[cols[0]])) if multi else str(names[cols[0]])
            groups.append((base, [samples[col] for col in cols]))
            for pos, col in enumerate(cols):
                node_id_of[samples[col]] = f"{base}_{pos + 1}" if multi else base
    else:
        where = {}                  # base -> group index
        for col, node in enumerate(samples):
            base, hap_id = _split_haplotype(names[col], ploidy)
            if base not in where:
                where[base] = len(groups)
                groups.append((base, []))
            groups[where[base]][1].append(node)
            node_id_of[node] = hap_id

    tables = ts.dump_tables()
    schema = tskit.MetadataSchema.permissive_json()

    # Individuals: one per group, {"id": base}; record the node -> individual mapping.
    tables.individuals.clear()
    tables.individuals.metadata_schema = schema
    ind_of_node = {}
    for gi, (base, member_nodes) in enumerate(groups):
        tables.individuals.add_row(metadata={"id": base})
        for node in member_nodes:
            ind_of_node[node] = gi

    # Nodes: link samples to their individual (reset every other link), and merge the per-haplotype
    # id into sample metadata while keeping other nodes' metadata bytes intact.
    nt = tables.nodes
    individual_col = np.full(nt.num_rows, tskit.NULL, dtype=np.int32)
    for node in samples:
        individual_col[node] = ind_of_node[node]

    old_meta, old_off = nt.metadata, nt.metadata_offset
    old_is_null = nt.metadata_schema.schema is None
    sample_set = set(samples)
    new_meta = []
    for n in range(nt.num_rows):
        if n in sample_set:
            new_meta.append(schema.validate_and_encode_row({"id": node_id_of[n]}))
        else:
            raw = bytes(old_meta[old_off[n]:old_off[n + 1]])
            new_meta.append(b"{}" if (old_is_null or not raw) else raw)
    packed, offset = tskit.pack_bytes(new_meta)
    nt.set_columns(flags=nt.flags, time=nt.time, population=nt.population,
                   individual=individual_col, metadata=packed, metadata_offset=offset)
    nt.metadata_schema = schema
    return tables.tree_sequence()


def _meta_id(metadata):
    """The ``"id"`` field of a decoded metadata value, or ``None`` (tolerating raw bytes)."""
    if isinstance(metadata, dict):
        v = metadata.get("id")
        return None if v is None else str(v)
    return None


def sample_id_index(ts):
    """Map every stamped id to the sample-node ids it names.

    Builds ``{id_string: [sample_node, ...]}`` from the individual metadata (a base id → *all* its
    haplotype nodes) and the per-haplotype node metadata (→ that one node) written by
    :func:`attach_sample_ids`. Empty when ``ts`` was never stamped (integer keys still resolve).

    Parameters
    ----------
    ts : tskit.TreeSequence
        A tree sequence whose sample nodes / individuals may carry ``{"id": ...}`` metadata (as
        stamped by :func:`attach_sample_ids`).

    Returns
    -------
    dict[str, list[int]]
        Maps each stamped id string — every base/individual id and every per-haplotype node id — to
        the sample-node ids it names, in sample order with duplicates removed. Empty if ``ts``
        carries no stamped ids.
    """
    index = {}
    for s in ts.samples():
        s = int(s)
        node = ts.node(s)
        nid = _meta_id(node.metadata)
        if nid is not None:
            index.setdefault(nid, []).append(s)
        if node.individual != -1:
            iid = _meta_id(ts.individual(node.individual).metadata)
            if iid is not None:
                index.setdefault(iid, []).append(s)
    return {k: list(dict.fromkeys(v)) for k, v in index.items()}


def resolve_nodes(ts, key, index=None):
    """Resolve one label/query ``key`` to a list of sample-node ids.

    An ``int`` (or integer-valued key) is a node index. A ``str`` is matched against the stamped
    ids (:func:`sample_id_index`); a digit-only string with no id match falls back to an integer
    index. Pass a prebuilt ``index`` to avoid rebuilding it per key.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence whose stamped ids ``key`` is resolved against.
    key : int or str
        A label/query key: an ``int`` (or ``numpy`` integer) sample-node index used as-is; or a
        sample-ID string — a base/individual id (expands to *all* its haplotype nodes) or a
        per-haplotype node id (that one node). A digit-only string with no id match falls back
        to an integer node index.
    index : dict[str, list[int]], optional
        A prebuilt :func:`sample_id_index` to reuse across keys. Default ``None`` — built on demand
        (and skipped entirely for an integer ``key``).

    Returns
    -------
    list[int]
        The sample-node ids named by ``key`` — one for an integer key or a per-haplotype id, or
        several for a base id naming a whole (diploid / polyploid) sample.

    Raises
    ------
    KeyError
        If a string ``key`` matches no stamped id and is not a plain integer index.
    """
    if isinstance(key, (int, np.integer)):
        return [int(key)]
    index = sample_id_index(ts) if index is None else index
    skey = str(key)
    if skey in index:
        return list(index[skey])
    if re.fullmatch(r"-?\d+", skey):                 # e.g. a JSON labels file keyed {"3": 0}
        return [int(skey)]
    raise KeyError(
        f"sample id {key!r} not found among the tree sequence's stamped ids "
        f"({len(index)} known); pass an integer node index, or a ts from io.singer/io.tsinfer "
        f"whose source carried sample names")


def resolve_labels(ts, labels):
    """Resolve a ``{id: state}`` label dict to ``{sample_node: state}`` (int keys pass through).

    Keys may be integer node indices or sample-ID strings (base or per-haplotype); a base id
    expands to *all* the sample's haplotype nodes. See the module docstring for the rules.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The stamped tree sequence to resolve against.
    labels : dict
        Maps each key — an integer sample-node index or a sample-ID string (base/individual id or
        per-haplotype node id) — to an ancestry-state index (cast to ``int``). A base id assigns
        its state to *all* the sample's haplotype nodes.

    Returns
    -------
    dict[int, int]
        Keyed by **sample-node id**, each mapped to its integer state. A base id that expands to
        several nodes gives them each that state; if two keys resolve to one node the last wins.

    Raises
    ------
    KeyError
        Propagated from :func:`resolve_nodes` when a string key matches no stamped id and is not an
        integer index.
    """
    index = sample_id_index(ts)
    out = {}
    for key, state in labels.items():
        for node in resolve_nodes(ts, key, index):
            out[node] = int(state)
    return out


def resolve_ids(ts, ids):
    """Resolve an iterable of ids (queries / soft_refs / anchors / samples) to sample-node ids.

    ``None`` passes through as ``None``. Order is preserved and duplicates removed (a base id that
    expands to several haplotype nodes contributes each once).

    Parameters
    ----------
    ts : tskit.TreeSequence
        The stamped tree sequence to resolve against.
    ids : iterable or None
        Keys to resolve — integer sample-node indices and/or sample-ID strings (base or
        per-haplotype), as accepted by :func:`resolve_nodes`. ``None`` passes straight through.

    Returns
    -------
    list[int] or None
        The resolved sample-node ids in first-appearance order with duplicates removed, or ``None``
        when ``ids`` is ``None``.

    Raises
    ------
    KeyError
        Propagated from :func:`resolve_nodes` when a string id matches no stamped id and is not an
        integer index.
    """
    if ids is None:
        return None
    index = sample_id_index(ts)
    out = []
    for key in ids:
        out.extend(resolve_nodes(ts, key, index))
    return list(dict.fromkeys(out))
