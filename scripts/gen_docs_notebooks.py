"""Generate the docs/notebooks/*.ipynb showcase notebooks (CLAUDE.md docs task).

Each notebook computes against the real tspaint API and labels its figure cells
(#| label: fig-...) so the docs pages can embed them with {{< embed >}}.
"""
import os
import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "notebooks")
os.makedirs(OUT, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src, label=None, cap=None, fig=True):
    hdr = ""
    if label:
        hdr += f"#| label: {label}\n"
        if cap:
            key = "fig-cap" if fig else "tbl-cap"
            hdr += f'#| {key}: "{cap}"\n'
    return nbf.v4.new_code_cell(hdr + src.strip("\n"))


def notebook(title, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_raw_cell(f'---\ntitle: "{title}"\n---')] + cells   # quote: titles may contain ':'
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"}}
    return nb


# Shared visualization helper, embedded in each notebook so each is self-contained.
VIZ = '''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import tspaint
from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

def admixture(n_admix=8, n_ref=8, L=2e6, T_admix=100, Ne=1000, T_split=5000, f_A=0.5,
              seed=1, infer=False, mutation_rate=4e-7):
    """Simulate admixture with known truth; return (ts, labels, queries, truth_states)."""
    ts = tspaint.simulate_admixture(n_admix=n_admix, n_ref=n_ref, sequence_length=L,
                                  recombination_rate=1e-8, random_seed=seed, Ne=Ne,
                                  T_admix=T_admix, T_split=T_split, f_A=f_A)
    pop = ts.tables.nodes.population
    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in name.items() if n == SOURCE_A)
    B = next(p for p, n in name.items() if n == SOURCE_B)
    admix = next(p for p, n in name.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[pop[s]] for s in ts.samples() if pop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if pop[s] == admix]
    truth = tspaint.metrics.map_truth({q: tspaint.local_ancestry_truth(ts)[0][q] for q in queries}, sop)
    work = ts
    if infer:
        work = tspaint.io.infer_tree_sequence(tspaint.io.add_mutations(ts, rate=mutation_rate,
                                                                   random_seed=seed))
    return work, labels, queries, truth

CMAP = "RdBu_r"   # red = ancestry A (state 0), blue = ancestry B (state 1)

def plot_painting(painting, truth, ts, title="", segments=None):
    """Soft posterior P(A) painted along the genome per query haplotype, with a thin truth
    strip beneath each. If `segments` (hard tracts) is given, draw those instead of soft."""
    qs = painting.queries
    L = ts.sequence_length
    sm = cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(qs) + 1.2))
    for i, q in enumerate(qs):
        if segments is None:
            for seg in painting.posteriors[q]:
                ax.barh(i, seg.right - seg.left, left=seg.left, height=0.74,
                        color=sm.to_rgba(seg.posterior[0]), edgecolor="none")
        else:
            for (l, r, s) in segments[q]:
                ax.barh(i, r - l, left=l, height=0.74,
                        color=sm.to_rgba(1.0 if s == 0 else 0.0), edgecolor="none")
        for (l, r, s) in truth[q]:
            ax.barh(i - 0.46, r - l, left=l, height=0.13,
                    color=sm.to_rgba(1.0 if s == 0 else 0.0), edgecolor="none")
    ax.set_xlim(0, L); ax.set_ylim(-0.8, len(qs) - 0.2)
    ax.set_yticks(range(len(qs))); ax.set_yticklabels([f"hap {q}" for q in qs], fontsize=8)
    ax.set_xlabel("genomic position (bp)"); ax.set_title(title)
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("P(ancestry A)")
    ax.text(0, len(qs) - 0.5, "thin strip below each = true ancestry", fontsize=7, color="0.4")
    fig.tight_layout()
    return fig
'''

# ---------------------------------------------------------------------------- painting
painting_nb = notebook("Haplotype painting", [
    md("""
# Haplotype painting

The signature deliverable: a **soft, calibrated** posterior over ancestry at every position of
every query haplotype. We simulate an admixed sample with known local ancestry, paint it with
[`tspaint.paint`](../api/paint.html), and visualise the result against the truth.
"""),
    code(VIZ, fig=False),
    md("## Simulate and paint\n\nRecent admixture (long tracts), strong structure so the "
       "genealogy discriminates sharply."),
    code('''
ts, labels, queries, truth = admixture(n_admix=8, n_ref=8, L=2e6, T_admix=60, seed=1)
painting = tspaint.paint(ts, labels)
painting
'''),
    md("## The painting\n\nEach row is a query haplotype; colour is the posterior probability of "
       "ancestry **A** (red) vs **B** (blue), white where the tree cannot tell. The thin strip "
       "beneath each row is the true ancestry."),
    code('fig = plot_painting(painting, truth, ts, "Soft posterior (tspaint.paint)")\nfig.show()',
         label="fig-painting", cap="Soft local-ancestry posterior along each query haplotype "
         "(red = ancestry A, blue = B), with true ancestry as the thin strip beneath each row."),
    md("## Hard tracts for downstream analysis\n\nCollapse the soft posterior to hard ancestry "
       "tracts with a confidence **deadband** (suppresses low-confidence flips that fragment "
       "long tracts — see the [fragmentation notebook](fragmentation.ipynb))."),
    code('fig = plot_painting(painting, truth, ts, "Hard tracts (deadband 0.4)", '
         'segments=painting.segments(deadband=0.4))\nfig.show()',
         label="fig-painting-hard", cap="Hard ancestry tracts from the same posterior "
         "(deadband 0.4), matching the true tract structure."),
    md("## Accuracy\n\nBalanced accuracy and mean confidence of the soft painting."),
    code('''
ba = tspaint.metrics.balanced_accuracy(painting.posteriors, truth, samples=queries)
conf = tspaint.metrics.mean_confidence(painting.posteriors, samples=queries)
print(f"balanced accuracy = {ba:.3f}   mean confidence = {conf:.3f}")
'''),
])

# ------------------------------------------------------------------------- calibration
calibration_nb = notebook("Calibration & accuracy vs admixture age", [
    md("""
# Calibration & accuracy vs admixture age

tspaint's edge is a **calibrated** soft posterior. Here we show the reliability of `P(A)` and how
discrimination decays as admixture ages (the reference signal is lost when admixed lineages
coalesce among themselves before an old pulse — a coalescent limit, not tree-inference error).
"""),
    code(VIZ, fig=False),
    md("## Reliability diagram\n\nPredicted vs empirical `P(A)`, span-weighted by probability bin "
       "(pooled over several seeds for a smooth curve). A calibrated painter lies on the diagonal."),
    code('''
import numpy as np
pred, emp, wt = [], [], []
for seed in range(1, 7):
    ts, labels, queries, truth = admixture(n_admix=8, n_ref=8, L=1e6, T_admix=200, seed=seed)
    p = tspaint.paint(ts, labels)
    rc = tspaint.metrics.reliability_curve(p.posteriors, truth, state=0, n_bins=10)
    pred.append(rc["pred"]); emp.append(rc["emp"]); wt.append(rc["weight"])
# weighted average across seeds, per bin
pred = np.concatenate(pred); emp = np.concatenate(emp); wt = np.concatenate(wt)
order = np.argsort(pred)
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
ax.scatter(pred[order], emp[order], s=12 + 200 * wt[order] / wt.max(), alpha=0.6, color="C3")
ax.set_xlabel("predicted P(A)"); ax.set_ylabel("empirical P(A)")
ax.set_title("Reliability"); ax.legend(fontsize=8); fig.tight_layout()
fig.show()
''', label="fig-reliability", cap="Reliability diagram: predicted vs empirical P(ancestry A); "
       "marker size ∝ span weight. Points near the diagonal indicate calibration."),
    md("## Accuracy vs admixture age\n\nBalanced accuracy and mean confidence as the admixture "
       "pulse ages (deep split, so only the query↔reference link — not tract length — varies)."),
    code('''
ages = [30, 100, 300, 1000, 3000]
rows = tspaint.experiments.age_sweep(ages, n_admix=8, n_ref=8, sequence_length=1e6,
                                   Ne=1000, T_split=8000, f_A=0.5, seed=1, max_iter=8)
ba = [r["balanced_accuracy"] for r in rows]; cf = [r["confidence"] for r in rows]
fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(ages, ba, "o-", label="balanced accuracy")
ax.plot(ages, cf, "s--", label="mean confidence", color="C1")
ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("admixture age (generations)"); ax.set_ylim(0, 1.02)
ax.set_title("Discrimination vs admixture age"); ax.legend(fontsize=8); fig.tight_layout()
fig.show()
''', label="fig-accuracy-age", cap="Balanced accuracy and confidence vs admixture age: tspaint "
       "discriminates well at recent–moderate admixture; the reference signal is lost at old "
       "admixture under present-day sampling."),
])

# ----------------------------------------------------------------------- fragmentation
fragmentation_nb = notebook("Fragmentation & tract-length fidelity", [
    md("""
# Fragmentation & tract-length fidelity

Downstream admixture-pulse dating reads the **segment-length distribution**, so spurious short
opposite-ancestry calls (fragmenting a long tract) bias the inferred pulse *older*. Naive
`argmax` of the posterior over-fragments; a confidence **deadband** on the calibrated posterior
([`hard_segments`](../api/output.hard_segments.html)) recovers the true distribution.
"""),
    code(VIZ, fig=False),
    md("## Switch density: argmax vs deadband vs truth\n\nThe switch-density ratio "
       "(inferred / true) is the dating-relevant quantity: >1 fragments (biases older)."),
    code('''
import numpy as np
from tspaint.output import hard_segments
from tspaint.metrics import switch_density

ts, labels, queries, truth = admixture(n_admix=10, n_ref=10, L=5e6, T_admix=200, seed=1)
p = tspaint.paint(ts, labels)
L = ts.sequence_length
true_d = np.mean([switch_density(truth[q], L) for q in queries]) * 1e6
def density(db):
    return np.mean([switch_density(hard_segments(p.posteriors[q], db), L) for q in queries]) * 1e6
deadbands = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
dens = [density(c) for c in deadbands]
fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.axhline(true_d, color="k", ls="--", lw=1, label=f"truth ({true_d:.2f}/Mb)")
ax.plot(deadbands, dens, "o-", color="C2")
ax.set_xlabel("deadband"); ax.set_ylabel("switches / Mb")
ax.set_title("Naive argmax (deadband 0) over-fragments"); ax.legend(fontsize=8); fig.tight_layout()
fig.show()
''', label="fig-switchdensity", cap="Inferred switch density vs the deadband: argmax (deadband 0) "
       "over-fragments; a modest deadband recovers the true switch density."),
    md("## Tract-length distribution\n\nHistogram of inferred tract lengths (argmax vs deadband) "
       "against the truth."),
    code('''
def lengths(segs_by):
    return np.concatenate([[r - l for (l, r, _s) in segs_by[q]] for q in queries]) / 1e3
tl_true = lengths(truth)
tl_argmax = lengths({q: hard_segments(p.posteriors[q], 0.0) for q in queries})
tl_db = lengths({q: hard_segments(p.posteriors[q], 0.4) for q in queries})
bins = np.linspace(0, max(tl_true.max(), tl_argmax.max()), 30)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.hist(tl_argmax, bins, alpha=0.5, label="argmax", color="C3")
ax.hist(tl_db, bins, alpha=0.5, label="deadband 0.4", color="C2")
ax.hist(tl_true, bins, histtype="step", lw=2, label="truth", color="k")
ax.set_xlabel("tract length (kb)"); ax.set_ylabel("count")
ax.set_title("Tract-length distribution"); ax.legend(fontsize=8); fig.tight_layout()
fig.show()
''', label="fig-tractlen", cap="Tract-length distribution: naive argmax produces too many short "
       "tracts; the deadband matches the truth."),
])

# -------------------------------------------------------------------------- bp smoother
bp_nb = notebook("Horizontal BP smoother on inferred ARGs", [
    md("""
# Horizontal BP smoother on inferred ARGs

On a **true** ARG the per-tree posteriors are clean and a per-position deadband is near-optimal.
On an **inferred** (tsinfer) ARG, tree inference scatters spurious breakpoints a confidence
threshold cannot filter — the horizontal [BP smoother](../api/bp.bp_paint.html)
(`paint(smooth=True)`) suppresses them. Here we paint the same admixture on a tsinfer ARG with
and without smoothing.
"""),
    code(VIZ, fig=False),
    md("## Paint a tsinfer ARG, with and without smoothing"),
    code('''
ts, labels, queries, truth = admixture(n_admix=8, n_ref=8, L=2e6, T_admix=200, seed=1, infer=True)
plain = tspaint.paint(ts, labels)
smooth = tspaint.paint(ts, labels, smooth=True)
fig = plot_painting(plain, truth, ts, "tsinfer ARG — no smoothing (argmax tracts)",
                    segments=plain.segments())
fig.show()
''', label="fig-bp-plain", cap="Hard tracts from a tsinfer ARG without smoothing: tree-inference "
       "noise fragments the tracts."),
    code('fig = plot_painting(smooth, truth, ts, "tsinfer ARG — BP smoothed (paint(smooth=True))", '
         'segments=smooth.segments())\nfig.show()',
         label="fig-bp-smooth", cap="The same tsinfer ARG with the horizontal BP smoother: spurious "
         "switches are suppressed, recovering the tract structure."),
    md("## Quantified: segmentation F1, true vs inferred ARG\n\nThe BP smoother is redundant on the "
       "true ARG but wins decisively on inferred ARGs."),
    code('''
res = {}
for infer in (False, True):
    r = tspaint.bp.bp_vs_deadband_experiment(T_admix=500, infer=infer, seeds=(1, 2, 3),
                                           n_admix=8, n_ref=8, sequence_length=2e6)
    res[("inferred" if infer else "true") + " ARG"] = r
fig, ax = plt.subplots(figsize=(5.2, 3.8))
x = np.arange(len(res)); w = 0.36
ax.bar(x - w/2, [v["deadband_f1"][0] for v in res.values()], w, label="deadband",
       yerr=[v["deadband_f1"][1] for v in res.values()], capsize=3, color="C0")
ax.bar(x + w/2, [v["bp_f1"][0] for v in res.values()], w, label="BP smoother",
       yerr=[v["bp_f1"][1] for v in res.values()], capsize=3, color="C2")
ax.set_xticks(x); ax.set_xticklabels(list(res)); ax.set_ylabel("breakpoint F1")
ax.set_ylim(0, 1.05); ax.set_title("BP vs deadband (T_admix=500)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.show()
''', label="fig-bp", cap="Breakpoint F1 of the deadband vs the BP smoother on true and inferred "
       "ARGs: BP is redundant on the true ARG but wins on inferred ARGs."),
])

# ------------------------------------------------------------------------------ dating
dating_nb = notebook("Admixture dating: cross-ancestry rate through time", [
    md("""
# Admixture dating: cross-ancestry rate through time

A **separate, optional deliverable** from painting. Making the ancestry CTMC
*time-inhomogeneous* and fitting the cross-ancestry transition rate as a function of (backward)
time gives a profile `q_AB(t)`, `q_BA(t)` that locates **when** the two labelled ancestries
diverged or exchanged genes — directionally. It rides the same EM engine
([`fit_rate_through_time`](../api/fit_rate_through_time.html)) and can reuse a `paint` fit via
`Painting.rate_through_time()`, leaving the painting itself untouched.
"""),
    code(VIZ, fig=False),
    md("## Paint, then date — reusing the fit\n\nA clean split at `T_split`: the cross-ancestry "
       "rate must be ~0 more recently than the split (the two ancestries cannot share ancestry "
       "yet) and rise once `t` exceeds it. `Painting.rate_through_time()` warm-starts from the "
       "painting's fitted `(Q, π, w)` and returns a **new** profile — `painting.posteriors` is "
       "not modified."),
    code('''
T_split = 3000
ts, labels, queries, truth = admixture(n_admix=6, n_ref=8, L=5e5, T_admix=200, T_split=T_split,
                                       Ne=1000, seed=1)
painting = tspaint.paint(ts, labels)
rtt = painting.rate_through_time(n_iter=10)     # reuses the painting's fit; posteriors untouched
rtt
'''),
    md("## The profile\n\n`q_AB(t)` and `q_BA(t)` vs (backward) time on a log axis. The rise marks "
       "the divergence epoch; recent time carries ~no cross-ancestry rate. (Convention: a jump is "
       "parent→child = old→young = forward in time, so a *backward*-time A→B admixture shows in "
       "`q_BA`.)"),
    code('''
fig, ax = plt.subplots(figsize=(6.2, 4))
rtt.plot(ax=ax)
ax.axvline(T_split, color="k", ls="--", lw=1, label=f"true split ({T_split})")
ax.set_title("Cross-ancestry rate through time"); ax.legend(fontsize=8); fig.tight_layout()
fig.show()
''', label="fig-dating-profile", cap="Directional cross-ancestry transition rates q_AB(t), q_BA(t) "
       "vs backward time: ~0 more recent than the population split, rising once cross-ancestry "
       "sharing becomes possible. The dashed line is the true split time."),
    md("## Reading off the onset\n\nThe standalone entry point `tspaint.fit_rate_through_time(ts, "
       "labels)` does the same fit without a painting."),
    code('''
import numpy as np
onset = rtt.centers[np.argmax(rtt.q_AB > 0.5 * np.nanmax(rtt.q_AB))]
print(f"inferred onset ~ {onset:.0f} generations   (true split {T_split})")
'''),
])

import sys

_selected = set(sys.argv[1:])   # optional: regenerate only the named notebooks (default: all)
for fname, nb in [("painting", painting_nb), ("calibration", calibration_nb),
                  ("fragmentation", fragmentation_nb), ("bp_smoother", bp_nb),
                  ("dating", dating_nb)]:
    if _selected and fname not in _selected:
        continue
    path = os.path.join(OUT, f"{fname}.ipynb")
    nbf.write(nb, path)
    print("wrote", path)
print("DONE")
