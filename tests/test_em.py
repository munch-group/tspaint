"""Rung 5 gate (CLAUDE.md §11.1.5): hard-clamp EM.

Recovery is tested on synthetic balanced trees with uniform, *informative* branch
lengths (Q·t ≈ 0.5), which isolates the EM machinery from the coalescent
tree-shape confound (most coalescent branches are tiny uninformative tip branches,
so rate estimates there are high-variance — a Rung 8 question, CLAUDE.md §6/§9).
We forward-simulate ancestry states under a known (Q_true, pi_true), hard-clamp
every tip, and check EM recovers the rates and root frequencies with a
monotonically non-decreasing log-likelihood.
"""
import numpy as np
import tskit
from scipy.linalg import expm

from tspaint.model import make_generator_2state, stationary_distribution
from tspaint.em import fit


def balanced_binary_ts(depth, bl=0.6):
    """Complete binary tree: 2**depth tips, every branch length ``bl``."""
    tables = tskit.TableCollection(sequence_length=1.0)
    cur = []
    for _ in range(2 ** depth):
        cur.append(tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0))
    edges = []
    for level in range(1, depth + 1):
        nxt = []
        for j in range(0, len(cur), 2):
            p = tables.nodes.add_row(flags=0, time=level * bl)
            edges += [(cur[j], p), (cur[j + 1], p)]
            nxt.append(p)
        cur = nxt
    for c, p in edges:
        tables.edges.add_row(0, 1.0, p, c)
    tables.sort()
    return tables.tree_sequence()


def simulate_states(tree, Q, pi, node_time, rng):
    """Draw a root state from pi, then propagate down each branch under expm(Q t)."""
    K = len(pi)
    cache = {}

    def P(t):
        key = round(float(t), 12)
        if key not in cache:
            cache[key] = expm(Q * t)
        return cache[key]

    state = {}
    for r in tree.roots:
        state[r] = rng.choice(K, p=pi)
    for u in tree.nodes(order="preorder"):
        for c in tree.children(u):
            state[c] = rng.choice(K, p=P(node_time[u] - node_time[c])[state[u]])
    return state


def make_dataset(Q_true, pi_true, depth, n_trees, bl, seed):
    rng = np.random.default_rng(seed)
    ts_list, labels_list = [], []
    for _ in range(n_trees):
        ts = balanced_binary_ts(depth, bl)
        states = simulate_states(ts.first(), Q_true, pi_true, ts.tables.nodes.time, rng)
        labels_list.append({int(u): int(states[u]) for u in ts.samples()})
        ts_list.append(ts)
    return ts_list, labels_list


def test_em_recovers_known_Q_and_pi():
    Q_true = make_generator_2state(0.8, 1.2)
    pi_true = stationary_distribution(Q_true)
    ts_list, labels_list = make_dataset(Q_true, pi_true, depth=7, n_trees=12, bl=0.6, seed=0)

    res = fit(ts_list, labels_list, K=2, Q0=make_generator_2state(0.3, 0.3),
              max_iter=25, tol=1e-6)
    Q_hat, pi_hat = res.Q, res.pi
    assert res.w == {}                                   # hard-clamp -> no learned credibility

    # Q is estimated from thousands of branches -> recovers tightly.
    assert abs(Q_hat[0, 1] - 0.8) < 0.15, Q_hat[0, 1]
    assert abs(Q_hat[1, 0] - 1.2) < 0.20, Q_hat[1, 0]
    assert Q_hat[1, 0] > Q_hat[0, 1]                     # q_BA > q_AB asymmetry recovered
    # pi is estimated from the tree-roots only (12 here) -> only validity + ordering are
    # asserted; tight pi recovery needs many marginal trees (the genome setting; Rung 8).
    assert np.isclose(pi_hat.sum(), 1.0) and np.all(pi_hat >= 0)
    assert pi_hat[0] > pi_hat[1]                         # pi_true = [0.6, 0.4]


def test_em_loglik_monotone_nondecreasing():
    Q_true = make_generator_2state(1.2, 0.6)
    pi_true = stationary_distribution(Q_true)
    ts_list, labels_list = make_dataset(Q_true, pi_true, depth=7, n_trees=3, bl=0.6, seed=7)

    history = fit(ts_list, labels_list, K=2,
                  Q0=make_generator_2state(0.2, 0.05), max_iter=25, tol=1e-10).loglik_history
    assert len(history) > 2
    diffs = np.diff(history)
    assert np.all(diffs >= -1e-6), diffs[diffs < 0]      # EM never decreases the likelihood


def test_em_single_treesequence_runs():
    # production call path: one genome (linked trees), refs labelled, queries free
    import tspaint
    ts = tspaint.simulate_admixture(n_admix=3, n_ref=4, sequence_length=5e4,
                                  recombination_rate=1e-8, random_seed=3)
    node_pop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    src = {p for p, nm in names.items() if nm in ("A", "B")}
    label_of = {p: i for i, p in enumerate(sorted(src))}
    labels = {int(s): label_of[node_pop[s]] for s in ts.samples() if node_pop[s] in src}

    res = fit(ts, labels, K=2, Q0=make_generator_2state(1e-4, 1e-4), max_iter=6)
    Q_hat, pi_hat = res.Q, res.pi
    assert np.all(np.isfinite(Q_hat)) and np.allclose(Q_hat.sum(axis=1), 0.0)
    assert np.isclose(pi_hat.sum(), 1.0) and np.all(pi_hat >= 0)
    assert len(res.loglik_history) >= 1
