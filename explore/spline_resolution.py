"""Can a penalised log-spline capture abrupt cross-coalescence-rate changes as sharply as the
coalescent data support? (admix-dating design exploration.)

A clean A/B divergence at T_split gives an ABRUPT, well-powered cross-coalescence onset: the
pairwise A-B coalescence hazard is exactly 0 below T_split and ~1/N above (haploid). We extract
that real hazard vs time from an msprime ARG (pairwise MRCA times, span-weighted), then compare:

  * a fine-bin histogram MLE  (the data's raw, model-free resolution), and
  * penalised log-splines q(t)=exp(spline(log t)) across a range of penalties,

to see whether the spline recovers the step as sharply as the events warrant, or blurs it.
Also runs a brief gene-flow PULSE (a localized spike) as the harder abrupt-feature test.
"""
import numpy as np
import msprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import minimize

N = 1000


def sim_split(seed=1, L=1.5e6, n=6, pulse=None):
    """Clean A/B split at T_split=2000 (haploid). `pulse=(time, frac)` adds a B->A mass
    migration (gene-flow pulse) for the spike test."""
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    if pulse is not None:
        t, frac = pulse
        d.add_mass_migration(time=t, source="A", dest="B", proportion=frac)
    d.add_population_split(time=2000.0, derived=["A", "B"], ancestral="ANC")
    return msprime.sim_ancestry(samples={"A": n, "B": n}, demography=d, sequence_length=L,
                                recombination_rate=1e-8, random_seed=seed, ploidy=1)


def cross_events_exposure(ts, edges):
    """Span-weighted A-B pairwise coalescence events + survival exposure per time bin →
    the cross-coalescence hazard data the M-step would see."""
    pop = ts.tables.nodes.population
    A = [int(s) for s in ts.samples() if pop[s] == 0]
    B = [int(s) for s in ts.samples() if pop[s] == 1]
    pairs = [(a, b) for a in A for b in B]
    nt = ts.tables.nodes.time
    times, wts = [], []
    for tree in ts.trees():
        s = tree.span
        for (a, b) in pairs:
            times.append(nt[tree.mrca(a, b)])
            wts.append(s)
    times = np.asarray(times)
    wts = np.asarray(wts)
    total = len(pairs) * ts.sequence_length            # at-risk weight at t=0
    ev, _ = np.histogram(times, bins=edges, weights=wts)
    cum = np.cumsum(ev)
    atrisk = total - np.concatenate([[0.0], cum[:-1]])  # survival at each bin start
    exposure = atrisk * np.diff(edges)
    return ev, exposure


def fit_logspline(centers, events, exposure, lam, n_knots=30, degree=3):
    """Penalised Poisson log-spline: q(t)=exp(B(log t)·beta), 2nd-difference roughness penalty."""
    x = np.log(centers)
    inner = np.linspace(x.min(), x.max(), n_knots)
    knots = np.r_[[inner[0]] * degree, inner, [inner[-1]] * degree]
    B = BSpline.design_matrix(x, knots, degree).toarray()
    nc = B.shape[1]
    Dpen = np.diff(np.eye(nc), 2, axis=0)
    P = Dpen.T @ Dpen
    m = exposure > 0

    def nll(b):
        eta = B @ b
        mu = exposure * np.exp(eta)
        return -np.sum((events * eta - mu)[m]) + lam * b @ P @ b

    def grad(b):
        eta = B @ b
        mu = exposure * np.exp(eta)
        return B.T @ ((mu - events) * m) + 2 * lam * P @ b

    res = minimize(nll, np.full(nc, np.log((events[m].sum() + 1) / (exposure[m].sum() + 1))),
                   jac=grad, method="L-BFGS-B")
    return np.exp(B @ res.x)


def rise_width(centers, rate, lo_frac=0.1, hi_frac=0.9):
    """10-90% rise width (in generations) of a rate profile around its main increase."""
    r = rate / np.nanmax(rate)
    above_lo = np.where(r >= lo_frac)[0]
    above_hi = np.where(r >= hi_frac)[0]
    if len(above_lo) == 0 or len(above_hi) == 0:
        return np.nan
    return centers[above_hi[0]] - centers[above_lo[0]]


def run(scenario, pulse, seeds=(1, 2, 3, 4)):
    edges = np.geomspace(20, 30000, 70)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ev = np.zeros(len(centers))
    ex = np.zeros(len(centers))
    for sd in seeds:
        e, x = cross_events_exposure(sim_split(seed=sd, pulse=pulse), edges)
        ev += e
        ex += x
    hist = np.where(ex > 0, ev / ex, np.nan)
    fits = {lam: fit_logspline(centers, ev, ex, lam) for lam in (0.03, 1.0, 30.0)}
    # data resolution proxy: width of the bin containing the first substantial events
    onset_bin = np.argmax(ev > 0.02 * ev.max())
    print(f"\n== {scenario} ==  total event-weight={ev.sum():.2e}")
    print(f"  histogram 10-90 rise width: {rise_width(centers, np.nan_to_num(hist)):.0f} gen")
    for lam, f in fits.items():
        print(f"  spline lam={lam:<5} 10-90 rise width: {rise_width(centers, f):.0f} gen")
    print(f"  bin width at onset (~T_split): {edges[onset_bin+1]-edges[onset_bin]:.0f} gen "
          f"(center {centers[onset_bin]:.0f})")
    return edges, centers, ev, ex, hist, fits


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, (scen, pulse) in zip(axes, [("clean split (step onset)", None),
                                     ("split + gene-flow pulse @600", (600.0, 0.3))]):
    edges, centers, ev, ex, hist, fits = run(scen, pulse)
    ax.step(centers, np.nan_to_num(hist) * N, where="mid", color="0.5", lw=1,
            label="fine histogram (raw)")
    for lam, f in fits.items():
        ax.plot(centers, f * N, lw=2, label=f"spline λ={lam}")
    ax.axvline(2000, color="k", ls="--", lw=1, label="T_split")
    if pulse:
        ax.axvline(pulse[0], color="C3", ls=":", lw=1, label="pulse")
    ax.set_xscale("log"); ax.set_xlabel("time (generations ago)")
    ax.set_ylabel("cross-coalescence rate × N"); ax.set_title(scen)
    ax.set_ylim(-0.05, 2.0); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("explore/spline_resolution.png", dpi=120)
print("\nwrote explore/spline_resolution.png")
