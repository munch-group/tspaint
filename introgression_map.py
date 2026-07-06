"""Worked example: HOW ``painting.posteriors[ref]`` comes to show a reference's OWN introgression —
the Felsenstein down-pass mechanism, plus a reference-inclusive plot (see ``introgression_map.md``).

Run:  pixi run python introgression_map.py

Mechanism (per tip, K=2, states A/B):
  up-pass    : a tip's leaf likelihood IS its emission vector.
  down-pass  : gamma_tip  =  normalize( emission_tip * U_tip )
               where U_tip = the "outside message" — what the REST of the tree says about the tip
               (its genealogical neighbours), independent of the tip's own label.
  A hard-clamped reference has emission = one-hot [1,0] (label A). Over a B-introgressed tract the
  genealogy coalesces with B references, so U_tip says B (U_B large). But the down-pass multiplies
  emission * U = [1*U_A, 0*U_B] = [U_A, 0] -> the 0 KILLS the B evidence -> gamma pinned to [1,0].
  The introgression is real and present in U, but the tip's own certainty hides it in the painting.
  Fragment masking sets emission -> query_emission(pi) = pi (flat). Now emission * U = pi * U ∝ U,
  so gamma_tip = U_tip: the down-pass becomes the tree's verdict and the B tract appears.
"""
import os

import numpy as np

import tspaint
from tspaint.sim import (SOURCE_A, SOURCE_B, ADMIXED, REF_A_IMPURE, REF_B_IMPURE,
                         simulate_admixture_impure_refs)
from tspaint.metrics import map_truth
from tspaint.em import fit, build_emissions
from tspaint.output import posterior_table, loo_posterior_table


def main():
    # Impure references: the SOURCE_A-labelled panel carries a known 25% minority of real SOURCE_B
    # tracts (census truth). Recent admixture (T_admix=60) -> strong genealogical foreign-tract signal.
    ts = tspaint.io.add_mutations(
        simulate_admixture_impure_refs(n_admix=6, n_pure=6, n_impure=6, sequence_length=3e6,
                                       recombination_rate=1e-8, random_seed=3, ref_impurity=0.25,
                                       Ne=1000, T_admix=60, T_split=5000, f_A=0.5),
        rate=2e-7, random_seed=3)
    npop = ts.tables.nodes.population
    nm = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    pid = {n: p for p, n in nm.items()}
    of = lambda name: [int(s) for s in ts.samples() if npop[s] == pid[name]]
    impA = of(REF_A_IMPURE)                                # nominally "A" but ~25% real B
    queries = of(ADMIXED)
    labels = {s: 0 for s in of(SOURCE_A) + impA}
    labels.update({s: 1 for s in of(SOURCE_B) + of(REF_B_IMPURE)})

    truth, _ = tspaint.local_ancestry_truth(ts)
    sop = {pid[SOURCE_A]: 0, pid[SOURCE_B]: 1}
    ref_truth = map_truth({r: truth[r] for r in impA}, sop)
    r = max(impA, key=lambda x: sum(rr - l for (l, rr, s) in ref_truth[x] if s == 1))
    true_B = [(l, rr) for (l, rr, s) in ref_truth[r] if s == 1]

    # Detect this ref's foreign tracts (the shipped QC), then read the mechanism off ONE fit.
    qc = tspaint.reference_qc(ts, labels)
    mask = qc.mask()
    res = fit(ts, labels, estimate_pi=False, max_iter=10)                    # paint's default fit
    em_hard = build_emissions(ts, labels, res.w, res.pi)
    em_mask = build_emissions(ts, labels, res.w, res.pi, mask)
    g_hard = posterior_table(ts, res.Q, res.pi, em_hard, focal=[r])          # down-pass, hard clamp
    U = loo_posterior_table(ts, res.Q, res.pi, em_hard, focal=[r])           # the outside message U
    g_mask = posterior_table(ts, res.Q, res.pi, em_mask, focal=[r])          # down-pass, masked

    def mean_pA(tab):                                                        # span-weighted P(A) over true-B
        num = den = 0.0
        for seg in tab[r]:
            for (l, rr) in true_B:
                ov = max(0.0, min(seg.right, rr) - max(seg.left, l))
                if ov > 0:
                    num += ov * float(seg.posterior[0]); den += ov
        return num / den if den else float("nan")

    print("impure ref node %d, label A; its TRUE B (introgressed) tract = %.0f bp" %
          (r, sum(rr - l for l, rr in true_B)))
    print("emission over the tract:  hard clamp = [1, 0]   |   masked = pi = %s (flat)\n" % np.round(res.pi, 2))
    print("mean P(A) over that true-B tract   [gamma = normalize(emission * U)]:")
    print("  outside message   U   : P(A)=%.3f   <- the TREE's verdict (coalesces with B here)" % mean_pA(U))
    print("  hard down-pass  gamma : P(A)=%.3f   <- emission*U = [U_A, 0*U_B] -> pinned to A, B HIDDEN" % mean_pA(g_hard))
    print("  masked down-pass gamma: P(A)=%.3f   <- emission=pi -> gamma ∝ U -> matches U, B REVEALED" % mean_pA(g_mask))
    print("  (masked gamma == outside message U: %s)\n" % np.isclose(mean_pA(g_mask), mean_pA(U), atol=2e-2))

    # ---- reference-inclusive plot: 3 queries + the impure ref (its masked spans hatched) ----
    focal = queries[:3] + [r]
    painting = tspaint.paint(ts, labels, queries=focal, mask=mask)
    tmap = map_truth({s: truth[s] for s in focal}, sop)
    fig, _ = painting.plot(truth=tmap, return_plot=True,
                           title="queries + impure ref %d (row 'ref %d (A)'): its B tract shows; masked span hatched"
                           % (r, r))
    png = os.path.abspath("introgression_map.png")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print("saved plot ->", png)


if __name__ == "__main__":
    main()
