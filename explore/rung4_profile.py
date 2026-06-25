"""Rung 4 payoff: does the full time-inhomogeneous EM sharpen the onset vs Stage-1?

Clean A/B split (true cross-coalescence onset = a step at T_split=2000). Compare:
  * Stage-1 binned profile (homogeneous E-step; known to smear the onset earlier), and
  * the full time-inhomogeneous EM (fit_rate_through_time),
against the model-free oracle cross-coalescence hazard. Hypothesis: the full EM concentrates the
rate at/above T_split (less leakage below).
"""
import numpy as np
import msprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tslai.dating import (log_time_grid, cell_centers, rate_through_time_binned,
                          fit_rate_through_time)

N, T_SPLIT = 1000, 2000.0


def sim(seed=1, L=5e5, n=8):
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_SPLIT, derived=["A", "B"], ancestral="ANC")
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


edges = log_time_grid(20.0, 20000.0, 45)
centers = cell_centers(edges)
ts = sim()
pop = ts.tables.nodes.population
labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}
print(f"trees={ts.num_trees}; stage-1 ...")
stage1 = rate_through_time_binned(ts, labels, edges, max_iter=8)
print("full inhomogeneous EM ...")
full = fit_rate_through_time(ts, labels, edges, n_iter=12, n_knots=18)
haz = oracle_hazard(ts, edges)

fig, ax = plt.subplots(figsize=(7.4, 4.7))
ax.plot(centers, np.nan_to_num(stage1["q_AB"]) * N, "o-", ms=3, color="C0", alpha=0.6,
        label="Stage-1 q_AB·N (homogeneous E-step)")
ax.plot(centers, full.q_AB * N, "-", lw=2.4, color="C2", label="full EM q_AB·N")
ax.plot(centers, full.q_BA * N, "--", lw=1.6, color="C1", label="full EM q_BA·N")
ax.plot(centers, np.nan_to_num(haz) * N, ":", lw=1.5, color="0.4", label="oracle hazard·N")
ax.axvline(T_SPLIT, color="k", ls=":", lw=1, label="T_split")
ax.set_xscale("log")
ax.set_xlabel("time (generations ago)")
ax.set_ylabel("rate × N")
ax.set_ylim(-0.05, max(0.6, np.nanmax(full.q_AB * N) * 1.2))
ax.set_title("Rung 4: full time-inhomogeneous EM vs Stage-1")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("explore/rung4_profile.png", dpi=120)


def onset(c, r):
    r = np.nan_to_num(r)
    pk = r.max()
    if pk <= 0:
        return np.nan
    idx = np.where(r >= 0.5 * pk)[0]
    return c[idx[0]] if len(idx) else np.nan


print("EM loglik history:", [round(x, 1) for x in full.loglik_history])
print(f"onset (time at 50% of peak):  stage-1={onset(centers, stage1['q_AB']):.0f}  "
      f"full-EM={onset(centers, full.q_AB):.0f}  oracle={onset(centers, haz):.0f}  "
      f"(T_split={T_SPLIT:.0f})")
print("wrote explore/rung4_profile.png")
