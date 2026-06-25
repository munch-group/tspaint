"""Rung 6: does the rate-through-time survive inferred / mis-calibrated node times?

The profile lives entirely on the node-age axis, so time calibration is the make-or-break for
real data. Clean A/B split (true onset at T_split=2000). Compare the q_AB(t) profile on:
  * the TRUE ARG (exact times) — baseline,
  * the true ARG with systematically MIS-CALIBRATED times (a monotone warp t·(t/t_ref)^(a-1),
    the §6 panmictic-prior-bias concern) — how much does the onset move?
  * a SINGER posterior ARG (Bayesian, coalescent-CALIBRATED times in generations) — the realistic
    front end (tsinfer alone gives uncalibrated times; tsdate not installed).
"""
import numpy as np
import msprime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tspaint
from tspaint.dating import log_time_grid, cell_centers, fit_rate_through_time

N, T_SPLIT, MU = 1000, 2000.0, 1e-6


def sim(seed=1, L=3e5, n=6):
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_SPLIT, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": n, "B": n}, demography=d, sequence_length=L,
                              recombination_rate=1e-8, random_seed=seed, ploidy=1)
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed)


def labels_of(ts):
    pop = ts.tables.nodes.population
    return {int(s): (0 if pop[s] == 0 else 1) for s in ts.samples()}


def miscalibrate(ts, a=0.8, t_ref=500.0):
    """Monotone time warp t' = t·(t/t_ref)^(a-1) (a<1 compresses deep times; recent ~unchanged)."""
    tables = ts.dump_tables()
    t = tables.nodes.time.copy()
    nz = t > 0
    tw = t.copy()
    tw[nz] = t[nz] * (t[nz] / t_ref) ** (a - 1.0)
    tables.nodes.time = tw
    tables.mutations.clear()                  # dating uses only the genealogy + times,
    tables.sites.clear()                      # so drop mutations (whose times would break sort)
    tables.sort()
    return tables.tree_sequence()


def profile(ts, edges, labels):
    return fit_rate_through_time(ts, labels, edges, n_iter=8, n_knots=16)


edges = log_time_grid(20.0, 20000.0, 45)
c = cell_centers(edges)
ts = sim()
LABELS = labels_of(ts)          # SINGER drops populations but preserves sample order/ids -> reuse

print("true ARG ...")
r_true = profile(ts, edges, LABELS)
print("mis-calibrated times ...")
r_mis = profile(miscalibrate(ts), edges, LABELS)

singer_ok = False
try:
    print(f"SINGER ({ts.num_sites} sites) ...")
    samples = tspaint.io_singer.singer_tree_sequences(
        ts, Ne=N, mutation_rate=MU, recombination_rate=1e-8,
        n_samples=8, thin=5, burn_in=3, seed=7)
    r_singer = profile(samples[-1], edges, LABELS)
    singer_ok = True
    print(f"  SINGER returned {len(samples)} samples; using the last")
except Exception as e:
    print("SINGER failed:", type(e).__name__, str(e)[:200])

fig, ax = plt.subplots(figsize=(7.6, 4.7))
ax.plot(c, r_true.q_AB * N, "-", lw=2.4, color="C2", label="true ARG (exact times)")
ax.plot(c, r_mis.q_AB * N, "--", lw=1.9, color="C3", label="mis-calibrated times (warp a=0.8)")
if singer_ok:
    ax.plot(c, r_singer.q_AB * N, "-.", lw=2.0, color="C0", label="SINGER (calibrated)")
ax.axvline(T_SPLIT, color="k", ls=":", lw=1, label="T_split")
ax.set_xscale("log")
ax.set_xlabel("time (generations ago)")
ax.set_ylabel("q_AB rate × N")
ax.set_title("Rung 6: rate-through-time vs node-time calibration")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("explore/rung6_calibration.png", dpi=120)


def onset(cc, r):
    r = np.nan_to_num(r)
    pk = r.max()
    if pk <= 0:
        return float("nan")
    idx = np.where(r >= 0.5 * pk)[0]
    return cc[idx[0]] if len(idx) else float("nan")


print(f"\nonset (time at 50% of peak):  T_split={T_SPLIT:.0f}")
print(f"  true ARG       = {onset(c, r_true.q_AB):.0f}")
print(f"  mis-calibrated = {onset(c, r_mis.q_AB):.0f}")
if singer_ok:
    print(f"  SINGER         = {onset(c, r_singer.q_AB):.0f}")
print("wrote explore/rung6_calibration.png")
