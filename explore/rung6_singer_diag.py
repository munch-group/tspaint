"""Rung 6 diagnostic: is SINGER's failure the panmictic-prior bias on a structured sample?

Directly compares the median A-B pairwise coalescence time on the TRUE ARG vs SINGER ARGs
(denser data + more MCMC than the first pass). Truth for a clean split: A-B pairs coalesce only
> T_split (=2000), so the true median is > 2000. If SINGER's median is << 2000, its panmictic
prior is pulling the cross-population coalescences recent — mis-timing exactly the deep structure
the rate-through-time dating reads.
"""
import numpy as np
import msprime
import tslai

N, T_SPLIT, MU = 1000, 2000.0, 1e-6


def sim(seed=1, L=5e5, n=6):
    d = msprime.Demography()
    d.add_population(name="A", initial_size=N)
    d.add_population(name="B", initial_size=N)
    d.add_population(name="ANC", initial_size=N)
    d.add_population_split(time=T_SPLIT, derived=["A", "B"], ancestral="ANC")
    ts = msprime.sim_ancestry(samples={"A": n, "B": n}, demography=d, sequence_length=L,
                              recombination_rate=1e-8, random_seed=seed, ploidy=1)
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed)


def median_cross_coal(g, A, B):
    """Span-weighted median A-B pairwise coalescence time. A, B are sample-node ids (the same
    in the true and SINGER ARGs, which are order-aligned — SINGER drops the population labels)."""
    nt = g.tables.nodes.time
    times, wts = [], []
    for tree in g.trees():
        s = tree.span
        for a in A:
            for b in B:
                times.append(nt[tree.mrca(a, b)])
                wts.append(s)
    times, wts = np.asarray(times), np.asarray(wts)
    order = np.argsort(times)
    cw = np.cumsum(wts[order])
    return float(times[order][np.searchsorted(cw, cw[-1] / 2)])


ts = sim()
pop = ts.tables.nodes.population
A = [int(s) for s in ts.samples() if pop[s] == 0]
B = [int(s) for s in ts.samples() if pop[s] == 1]
true_med = median_cross_coal(ts, A, B)
print(f"sites={ts.num_sites};  A={A}  B={B}")
print(f"TRUE   median A-B coal time = {true_med:.0f}   (T_split={T_SPLIT:.0f}; expect > T_split)")
samples = tslai.io_singer.singer_tree_sequences(ts, Ne=N, mutation_rate=MU,
                                                recombination_rate=1e-8, n_samples=12,
                                                thin=5, burn_in=5, seed=3)
print(f"SINGER returned {len(samples)} post-burn-in samples")
meds = [median_cross_coal(g, A, B) for g in samples]
print(f"SINGER median A-B coal time = {np.mean(meds):.0f} ± {np.std(meds):.0f}  (per-sample: "
      f"{[round(m) for m in meds]})")
print(f"=> SINGER/true ratio = {np.mean(meds)/true_med:.2f} "
      f"(~1 => SINGER times are accurate; the panmictic prior does NOT mis-time the structure)")
