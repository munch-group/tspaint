"""Fragment masking (CLAUDE.md §2.3): position-dependent tip emissions — mask flagged reference
spans as **unlabelled** (the ref emits the query emission there) instead of down-weighting the whole
individual (soft ``w``). So a contaminated reference anchors only on its clean spans."""
import numpy as np
import pytest

import tspaint
from tspaint.em import build_emissions, fit
from tspaint.model import MaskedEmissions, query_emission, emissions_for
from tspaint.output import posterior_table, Segment, INFORMATIVE
from tspaint.parallel import posterior_table_parallel


# --- the position-dependent emission provider (fast, no fit) -----------------------------------

def test_masked_emissions_provider():
    base = {0: np.array([0.9, 0.1]), 1: np.array([0.2, 0.8]), 2: np.array([0.3, 0.7])}
    pi = np.array([0.5, 0.5])
    em = MaskedEmissions(base, {1: [(0.0, 100.0)]}, pi)
    over = em.for_interval(10, 20)                        # midpoint 15 -> ref 1 masked
    assert np.allclose(over[1], query_emission(pi))       # masked ref -> query emission
    assert over[0] is base[0] and over[2] is base[2]      # others untouched
    assert em.for_interval(200, 210) is base              # nothing masked here -> the base dict itself
    # (left, right, score) spans (foreign_tracts format) are accepted too
    em3 = MaskedEmissions(base, {1: [(0.0, 100.0, 0.9)]}, pi)
    assert np.allclose(em3.for_interval(10, 20)[1], query_emission(pi))
    # emissions_for is a no-op on a plain dict (backward compatible)
    assert emissions_for(base, 10, 20) is base
    assert emissions_for(em, 10, 20) is over


# --- build_emissions(mask=...) returns the provider; None keeps the plain dict ------------------

def _ts():
    return tspaint.io.add_mutations(
        tspaint.simulate_admixture(tspaint.sim.admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
                                   n_query=4, n_reference=6, sequence_length=6e5, recombination_rate=1e-8,
                                   random_seed=2).ts,
        rate=2e-7, random_seed=2)


@pytest.mark.slow
def test_build_emissions_mask_and_byte_identical_parallel():
    ts = _ts()
    labels = {i: (0 if i < 3 else 1) for i in range(4, 10)}
    res = fit(ts, labels, max_iter=6)
    L = ts.sequence_length
    assert not isinstance(build_emissions(ts, labels, res.w, res.pi), MaskedEmissions)   # None -> dict
    assert isinstance(build_emissions(ts, labels, res.w, res.pi, {4: [(0, L)]}), MaskedEmissions)

    mask = {4: [(0.0, L * 0.4)], 7: [(L * 0.6, L)]}
    focal = list(range(4)) + [4, 7]
    em = build_emissions(ts, labels, res.w, res.pi, mask)
    serial = posterior_table(ts, res.Q, res.pi, em, focal=focal)                  # fixed fit
    par = posterior_table_parallel(ts, res.Q, res.pi, w=res.w, labels=labels, focal=focal,
                                   mask=mask, n_jobs=3)                            # byte-exact
    assert serial.keys() == par.keys()
    for s in focal:
        assert len(serial[s]) == len(par[s])
        for a, b in zip(serial[s], par[s]):
            assert a.left == b.left and a.right == b.right and np.array_equal(a.posterior, b.posterior)


def test_painting_plot_is_reference_aware():
    """Painting.plot labels reference rows by nominal ancestry and hatches their masked spans."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tspaint.api import Painting
    posteriors = {
        0: [Segment(0.0, 100.0, np.array([0.9, 0.1]), INFORMATIVE)],           # a query
        1: [Segment(0.0, 50.0, np.array([0.2, 0.8]), INFORMATIVE),             # a ref painting B (foreign)
            Segment(50.0, 100.0, np.array([1.0, 0.0]), INFORMATIVE)],
    }
    p = Painting(posteriors=posteriors, Q=np.eye(2), pi=np.array([0.5, 0.5]), w={},
                 loglik_history=[], queries=[0, 1], labels={1: 0}, _seqlen=100.0,
                 mask={1: [(0.0, 50.0)]})
    fig, axes = p.plot(return_plot=True)
    assert len(axes) == 2
    assert axes[0].get_ylabel() == "hapl. 0"          # query keeps the default row label
    assert axes[1].get_ylabel() == "ref 1 (A)"        # reference labelled by nominal ancestry
    # the masked span is drawn as a hatched patch on the ref row
    assert any(getattr(pt, "get_hatch", lambda: None)() for pt in axes[1].patches)
    plt.close(fig)
    # refs=False suppresses the reference annotation
    fig2, axes2 = p.plot(return_plot=True, refs=False, mark_masked=False)
    assert axes2[1].get_ylabel() == "hapl. 1"
    plt.close(fig2)


@pytest.mark.slow
def test_paint_mask_takes_effect():
    ts = _ts()
    labels = {i: (0 if i < 3 else 1) for i in range(4, 10)}
    queries = list(range(4))
    L = ts.sequence_length
    base = tspaint.paint(ts, labels, queries=queries, n_jobs=1)
    masked = tspaint.paint(ts, labels, queries=queries, n_jobs=1, mask={4: [(0.0, L)], 7: [(0.0, L)]})
    def flat(p):
        return np.concatenate([[s.posterior[0] for s in p.posteriors[q]] for q in queries])
    b, m = flat(base), flat(masked)
    assert (len(b) != len(m)) or not np.allclose(b, m)        # unlabelling two refs changed the paint
