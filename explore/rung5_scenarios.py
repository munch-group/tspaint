"""Rung 5: does the rate-through-time profile resolve demographic structure?

Three scenarios (true ARG), each fit with the full time-inhomogeneous directional EM:
  1. A->B gene-flow PULSE at T_pulse plus a deep split  -> tests a localised bump AND asymmetry
     (only A->B moved, so the bump should sit in q_AB, not q_BA).
  2. ONGOING symmetric migration until the split          -> tests an elevated/plateau rate at
     recent times (vs the clean-split baseline of ~0 below the split).
The symmetric oracle cross-coalescence hazard is overlaid for reference.
"""
import numpy as np
import msprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tslai.dating import log_time_grid, cell_centers, fit_rate_through_time

N = 1000


def _demog():
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    return d


def sim_pulse(seed=1, T_split=4000.0, T_pulse=800.0, f=0.5, L=5e5, n=8):
    d = _demog()
    d.add_mass_migration(time=T_pulse, source="A", dest="B", proportion=f)   # A->B only (asym)
    d.add_population_split(time=T_split, derived=["A", "B"], ancestral="ANC")
    return msprime.sim_ancestry(samples={"A": n, "B": n}, demography=d, sequence_length=L,
                                recombination_rate=1e-8, random_seed=seed, ploidy=1)


def sim_ongoing(seed=1, T_split=6000.0, m=2e-4, L=5e5, n=8):
    d = _demog()
    d.set_migration_rate(source="A", dest="B", rate=m)
    d.set_migration_rate(source="B", dest="A", rate=m)
    d.add_population_split(time=T_split, derived=["A", "B"], ancestral="ANC")
    return msprime.sim_ancestry(samples={"A": n, "B": n}, demography=d, sequence_length=L,
                                recombination_rate=1e-8, random_seed=seed, ploidy=1)


def oracle_hazard(ts, edges):
    pop = ts.tables.nodes.population
    A = [int(s) for s in ts.samples() if pop[s] == 0]
    B = [int(s) for s in ts.samples() if pop[s] == 1]
    pairs = [(a, b) for a in A for b in B]
    nt = ts.tables.nodes.time
    times, wts = [], []
    for tree in ts.trees():
        s = tree.span
        for a, b in pairs:
            times.append(nt[tree.mrca(a, b)])
            wts.append(s)
    times, wts = np.asarray(times), np.asarray(wts)
    total = len(pairs) * ts.sequence_length
    ev, _ = np.histogram(times, bins=edges, weights=wts)
    atrisk = total - np.concatenate([[0.0], np.cumsum(ev)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(atrisk > 0, ev / (atrisk * np.diff(edges)), np.nan)


def fit(ts, edges):
    pop = ts.tables.nodes.population
    labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}
    return fit_rate_through_time(ts, labels, edges, n_iter=10, n_knots=18)


edges = log_time_grid(20.0, 30000.0, 50)
c = cell_centers(edges)

print("scenario 1: A->B pulse @800 + split @4000 ...")
ts1 = sim_pulse()
r1 = fit(ts1, edges)
h1 = oracle_hazard(ts1, edges)
print("scenario 2: ongoing migration until split @6000 ...")
ts2 = sim_ongoing()
r2 = fit(ts2, edges)
h2 = oracle_hazard(ts2, edges)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
ax[0].plot(c, r1.q_AB * N, "-", lw=2.3, color="C2", label="q_AB·N (A→B)")
ax[0].plot(c, r1.q_BA * N, "--", lw=1.8, color="C1", label="q_BA·N (B→A)")
ax[0].plot(c, np.nan_to_num(h1) * N, ":", lw=1.4, color="0.5", label="oracle hazard·N")
ax[0].axvline(800, color="C3", ls=":", lw=1, label="T_pulse")
ax[0].axvline(4000, color="k", ls=":", lw=1, label="T_split")
ax[0].set_title("A→B pulse @800 + split @4000  (pulse + asymmetry)")

ax[1].plot(c, r2.q_AB * N, "-", lw=2.3, color="C2", label="q_AB·N")
ax[1].plot(c, r2.q_BA * N, "--", lw=1.8, color="C1", label="q_BA·N")
ax[1].plot(c, np.nan_to_num(h2) * N, ":", lw=1.4, color="0.5", label="oracle hazard·N")
ax[1].axvline(6000, color="k", ls=":", lw=1, label="T_split")
ax[1].set_title("ongoing symmetric migration (m=2e-4) until split @6000")

for a in ax:
    a.set_xscale("log")
    a.set_xlabel("time (generations ago)")
    a.set_ylabel("rate × N")
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("explore/rung5_scenarios.png", dpi=120)


def near(t, c, r, lo=0.6, hi=1.6):
    msk = (c > lo * t) & (c < hi * t)
    return float(np.nanmean(np.nan_to_num(r)[msk]))


print("\n--- pulse + asymmetry ---")
print(f"q_AB near T_pulse(800)·N = {near(800, c, r1.q_AB)*N:.3f}   "
      f"q_BA near T_pulse·N = {near(800, c, r1.q_BA)*N:.3f}   "
      f"(A->B pulse => expect q_AB > q_BA here)")
print(f"q_AB peak at {c[np.nanargmax(np.nan_to_num(r1.q_AB))]:.0f} gen")
print("\n--- ongoing migration ---")
print(f"recent (t<1500) mean rate·N: pulse-scn={near(700,c,r1.q_AB)*N:.3f}  "
      f"ongoing-scn q_AB·N={float(np.nanmean(np.nan_to_num(r2.q_AB)[c<1500]))*N:.3f}  "
      f"(ongoing should be elevated at recent times)")
print("wrote explore/rung5_scenarios.png")
