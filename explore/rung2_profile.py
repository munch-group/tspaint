"""Rung 2 go/no-go: does the Stage-1 binned rate-through-time profile show the T_split feature?

Clean A/B split at T_split=2000 (true cross-coalescence onset is a step there). Fit a homogeneous
Q with tslai, bin the per-branch jumps/dwell by time (tslai.dating.rate_through_time_binned), and
overlay the model-free oracle cross-coalescence hazard. If the directional rate rises around
T_split, the vertical mugration signal carries the divergence timing -> proceed to the full
time-inhomogeneous E-step (which should sharpen it).
"""
import numpy as np
import msprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tslai
from tslai.dating import log_time_grid, rate_through_time_binned

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
    times = np.asarray(times)
    wts = np.asarray(wts)
    total = len(pairs) * ts.sequence_length
    ev, _ = np.histogram(times, bins=edges, weights=wts)
    atrisk = total - np.concatenate([[0.0], np.cumsum(ev)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(atrisk > 0, ev / (atrisk * np.diff(edges)), np.nan)


edges = log_time_grid(20.0, 20000.0, 45)
ts = sim()
pop = ts.tables.nodes.population
labels = {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}
print(f"trees={ts.num_trees} samples={ts.num_samples}; fitting + binning ...")

prof = rate_through_time_binned(ts, labels, edges, max_iter=8)
haz = oracle_hazard(ts, edges)
c = prof["centers"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(c, prof["q_AB"] * N, "o-", ms=3, label="q_AB(t)·N  (tslai dating)")
ax.plot(c, prof["q_BA"] * N, "s-", ms=3, label="q_BA(t)·N  (tslai dating)")
ax.plot(c, haz * N, "--", color="0.5", lw=1.5, label="oracle cross-coal. hazard·N")
ax.axvline(T_SPLIT, color="k", ls=":", lw=1, label="T_split")
ax.set_xscale("log")
ax.set_xlabel("time (generations ago)")
ax.set_ylabel("rate × N")
ax.set_ylim(-0.1, 3.0)
ax.set_title("Stage-1 rate-through-time (homogeneous E-step, binned)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("explore/rung2_profile.png", dpi=120)

print("fitted Q       :", np.round(prof["Q"], 6).tolist())
print("centers (gen)  :", np.round(c[::5]).astype(int).tolist())
print("q_AB·N         :", np.round(np.nan_to_num(prof["q_AB"]) * N, 2)[::5].tolist())
print("oracle haz·N   :", np.round(np.nan_to_num(haz) * N, 2)[::5].tolist())
print("wrote explore/rung2_profile.png")
