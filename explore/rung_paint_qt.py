"""Does painting with the time-inhomogeneous Q(t) help or hurt LAI accuracy? And how much slower?

Admixture sim with known local ancestry. Paint the admixed queries two ways:
  * homogeneous Q  (the current tslai painter), and
  * the time-inhomogeneous Q(t) from the admixture-rate-through-time fit,
and compare balanced accuracy / mean confidence + wall-clock. (The cross-coalescence path is a
separate subpackage; this only asks whether feeding its Q(t) back into the painter changes LAI.)
"""
import time
import numpy as np

import tslai
from tslai.em import fit, build_emissions
from tslai.model import make_generator_2state
from tslai.output import posterior_table
from tslai.sim import simulate_admixture, local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
from tslai.validate import map_truth, balanced_accuracy, mean_confidence
from tslai.dating import log_time_grid, fit_rate_through_time, make_Q_of_cell
from tslai.dating.estep import paint_qt


def setup(seed=1, L=1e6, T_admix=100, Ne=1000, T_split=5000):
    ts = simulate_admixture(n_admix=8, n_ref=8, sequence_length=L, recombination_rate=1e-8,
                            random_seed=seed, Ne=Ne, T_admix=T_admix, T_split=T_split, f_A=0.5)
    npop = ts.tables.nodes.population
    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in name.items() if n == SOURCE_A)
    B = next(p for p, n in name.items() if n == SOURCE_B)
    admix = next(p for p, n in name.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    truth = map_truth({q: local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    return ts, labels, queries, truth


for T_admix in (100, 300):
    ts, labels, queries, truth = setup(T_admix=T_admix)
    edges = log_time_grid(20.0, 20000.0, 40)

    t0 = time.time()
    res = fit(ts, labels, Q0=make_generator_2state(1e-3, 1e-3), max_iter=8)
    emissions = build_emissions(ts, labels, res.w, res.pi)
    homog = posterior_table(ts, res.Q, res.pi, emissions, focal=queries)
    t_homog = time.time() - t0

    t0 = time.time()
    rtt = fit_rate_through_time(ts, labels, edges, n_iter=8, n_knots=16)
    Qof = make_Q_of_cell(rtt.q_AB, rtt.q_BA)
    qt = paint_qt(ts, emissions, Qof, res.pi, edges, queries)
    t_qt = time.time() - t0

    ba_h = balanced_accuracy(homog, truth, samples=queries)
    ba_q = balanced_accuracy(qt, truth, samples=queries)
    cf_h = mean_confidence(homog, samples=queries)
    cf_q = mean_confidence(qt, samples=queries)
    print(f"\n== T_admix={T_admix} ==")
    print(f"  homogeneous Q : bal={ba_h:.3f} conf={cf_h:.3f}   [{t_homog:.1f}s]")
    print(f"  Q(t)          : bal={ba_q:.3f} conf={cf_q:.3f}   [{t_qt:.1f}s]")
    print(f"  delta bal = {ba_q-ba_h:+.3f}   slowdown = {t_qt/max(t_homog,1e-9):.0f}x")
print("\nDONE")
