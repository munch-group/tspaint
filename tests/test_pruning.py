"""Rung 3 gate (CLAUDE.md §3.1, §4): Felsenstein pruning validated against
brute-force enumeration over all K^(#nodes) joint state assignments, across the
tskit structures the data model forces — polytomy, multi-root forest, internal
sample, isolated sample — and for K > 2 (generator-agnostic)."""
import itertools

import numpy as np
import pytest
import tskit
from scipy.linalg import expm

from tslai.pruning import prune_tree


# --- helpers -------------------------------------------------------------------

def build_ts(parents, times, samples, L=1.0):
    """Build a single-tree tree sequence from a parent map and node times."""
    tables = tskit.TableCollection(sequence_length=L)
    for u in range(len(times)):
        flags = tskit.NODE_IS_SAMPLE if u in samples else 0
        tables.nodes.add_row(flags=flags, time=times[u])
    for c, p in parents.items():
        if p != -1:
            tables.edges.add_row(left=0, right=L, parent=p, child=c)
    tables.sort()
    return tables.tree_sequence()


def random_Q(K, rng):
    Q = rng.random((K, K)) + 0.05
    np.fill_diagonal(Q, 0.0)
    for i in range(K):
        Q[i, i] = -Q[i].sum()
    return Q


def random_inputs(ts, K, rng):
    pi = rng.random(K) + 0.1
    pi /= pi.sum()
    Q = random_Q(K, rng)
    emissions = {int(s): rng.random(K) + 0.1 for s in ts.samples()}
    return Q, pi, emissions


def brute_force(tree, emissions, Q, pi, node_time):
    """Exact marginals by summing over every joint assignment of states to nodes."""
    K = len(pi)
    nodes = list(tree.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    roots = set(tree.roots)
    Pe = {}
    for u in nodes:
        p = tree.parent(u)
        if p != tskit.NULL:
            Pe[u] = expm(Q * (node_time[p] - node_time[u]))

    gamma = {u: np.zeros(K) for u in nodes}
    xi = {(tree.parent(u), u): np.zeros((K, K)) for u in nodes if tree.parent(u) != tskit.NULL}
    Z = 0.0
    for assign in itertools.product(range(K), repeat=len(nodes)):
        st = {u: assign[idx[u]] for u in nodes}
        w = 1.0
        for r in roots:
            w *= pi[st[r]]
        for u in nodes:
            p = tree.parent(u)
            if p != tskit.NULL:
                w *= Pe[u][st[p], st[u]]
        for u, e in emissions.items():
            w *= e[st[u]]
        if w == 0.0:
            continue
        Z += w
        for u in nodes:
            gamma[u][st[u]] += w
        for (p, c) in xi:
            xi[(p, c)][st[p], st[c]] += w
    for u in nodes:
        gamma[u] /= Z
    for key in xi:
        xi[key] /= Z
    return gamma, xi, Z


# --- structures ----------------------------------------------------------------

def ts_binary():
    parents = {0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 6: -1}
    times = [0, 0, 0, 0, 1.0, 1.3, 2.0]
    return build_ts(parents, times, samples={0, 1, 2, 3})


def ts_polytomy():
    # node 4 has three children (0,1,2); root 5 has children 4 and sample 3
    parents = {0: 4, 1: 4, 2: 4, 3: 5, 4: 5, 5: -1}
    times = [0, 0, 0, 0, 1.0, 2.0]
    return build_ts(parents, times, samples={0, 1, 2, 3})


def ts_forest():
    parents = {0: 4, 1: 4, 2: 5, 3: 5, 4: -1, 5: -1}
    times = [0, 0, 0, 0, 1.0, 1.0]
    return build_ts(parents, times, samples={0, 1, 2, 3})


def ts_internal_sample():
    # sample 2 is internal: it has children (0,1) AND its own emission; root 3 latent
    parents = {0: 2, 1: 2, 2: 3, 3: -1}
    times = [0.0, 0.0, 1.0, 2.0]
    return build_ts(parents, times, samples={0, 1, 2})


def ts_isolated_sample():
    # samples 0,1 under root 3; sample 2 isolated (no parent, no children)
    parents = {0: 3, 1: 3, 3: -1}
    times = [0, 0, 0, 1.0]
    return build_ts(parents, times, samples={0, 1, 2})


# --- tests ---------------------------------------------------------------------

@pytest.mark.parametrize("ts_fn", [ts_binary, ts_polytomy, ts_forest, ts_internal_sample])
@pytest.mark.parametrize("K", [2, 3])
def test_pruning_matches_brute_force(ts_fn, K):
    ts = ts_fn()
    tree = ts.first()
    node_time = ts.tables.nodes.time
    rng = np.random.default_rng(1000 * K + len(ts_fn.__name__))
    Q, pi, emissions = random_inputs(ts, K, rng)

    res = prune_tree(tree, emissions, Q, node_time, pi)
    g_bf, xi_bf, Z = brute_force(tree, emissions, Q, pi, node_time)

    for u in g_bf:
        np.testing.assert_allclose(res.gamma[u], g_bf[u], rtol=1e-9, atol=1e-11,
                                   err_msg=f"gamma mismatch at node {u}")
    for key in xi_bf:
        np.testing.assert_allclose(res.xi[key], xi_bf[key], rtol=1e-9, atol=1e-11,
                                   err_msg=f"xi mismatch at edge {key}")
    np.testing.assert_allclose(res.loglik, np.log(Z), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("ts_fn", [ts_binary, ts_polytomy, ts_forest, ts_internal_sample])
def test_posteriors_normalised_and_consistent(ts_fn):
    ts = ts_fn()
    tree = ts.first()
    node_time = ts.tables.nodes.time
    rng = np.random.default_rng(0)
    Q, pi, emissions = random_inputs(ts, 2, rng)
    res = prune_tree(tree, emissions, Q, node_time, pi)

    for u, g in res.gamma.items():
        assert np.isclose(g.sum(), 1.0)
    for (p, c), x in res.xi.items():
        assert np.isclose(x.sum(), 1.0)
        # xi marginals must equal the node posteriors
        np.testing.assert_allclose(x.sum(axis=1), res.gamma[p], rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(x.sum(axis=0), res.gamma[c], rtol=1e-9, atol=1e-11)


def test_forest_has_two_roots_each_from_prior():
    ts = ts_forest()
    tree = ts.first()
    assert len(tree.roots) == 2
    node_time = ts.tables.nodes.time
    rng = np.random.default_rng(2)
    Q, pi, emissions = random_inputs(ts, 2, rng)
    res = prune_tree(tree, emissions, Q, node_time, pi)
    assert set(res.root_marginal) == set(tree.roots)
    # loglik of a forest is the sum of independent per-root log-likelihoods
    _, _, Z = brute_force(tree, emissions, Q, pi, node_time)
    np.testing.assert_allclose(res.loglik, np.log(Z), rtol=1e-9, atol=1e-9)


def test_isolated_sample_is_missing_info_and_prior():
    ts = ts_isolated_sample()
    tree = ts.first()
    node_time = ts.tables.nodes.time
    rng = np.random.default_rng(3)
    Q, pi, emissions = random_inputs(ts, 2, rng)
    res = prune_tree(tree, emissions, Q, node_time, pi)

    assert 2 in res.missing_info                       # tagged distinctly
    np.testing.assert_allclose(res.gamma[2], pi)       # falls back to prior, not 50-50
    # the non-isolated subtree is unaffected: still matches brute force
    g_bf, _, _ = brute_force(tree, emissions, Q, pi, node_time)
    for u in (0, 1, 3):
        np.testing.assert_allclose(res.gamma[u], g_bf[u], rtol=1e-9, atol=1e-11)
