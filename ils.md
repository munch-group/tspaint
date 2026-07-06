# Can the cross-ancestry rate distinguish admixture from ILS?

*Analysis of whether `tspaint`'s time-resolved cross-ancestry rate (`fit_rate_through_time` /
`Painting.rate_through_time`) can separate admixture (gene flow) from incomplete lineage
sorting (ILS), or whether the method structurally can never represent ILS.*

---

## The question

`tspaint` paints local ancestry by evolving a discrete ancestry state `{A, B}` up the branches
of each marginal tree under a CTMC with generator `Q` (the mugration model). A cross-ancestry
**transition** up a branch is the model's "ancestral migration" event. `fit_rate_through_time`
estimates this as a **time-inhomogeneous, directional** rate — `q_AB(t)` and `q_BA(t)` as
functions of branch-time depth.

Two things share that signature:

- **Admixture** — a past gene-flow event introduces foreign tracts; the donor lineage
  coalesces with the other panel *recently* (below the population split time).
- **ILS** — with no gene flow at all, A-references and B-references are not reciprocally
  monophyletic at every locus; a pure-B query (or reference) can sort with A by deep stochastic
  coalescence in the ancestral population.

So: can the rate tell them apart, or is ILS invisible to the method by construction?

## Short answer

**Both halves are true at once, and they are not in tension.**

1. The model has no ILS *state*, so it never emits a categorical "this is ILS." In that narrow
   sense, **by definition it never infers ILS.**
2. But ILS is **not invisible.** ILS and admixture both surface as cross-ancestry transitions,
   and they are **separable by the *time profile* (and *direction*) of the rate** — exactly what
   `rate_through_time` resolves.

A single time-**homogeneous** `Q` cannot separate them: it averages the two regimes into one
scalar rate. The time-**inhomogeneous** `Q(t)` can.

## Why the timing separates them

Read `q(t)` as an empirical estimate of the **cross-label coalescence intensity** — the rate at
which A-lineages and B-lineages merge at time `t`. A cross-ancestry jump on a branch sits just
below the node where an A-dominated subtree joins a B-dominated one, so the jump *time* ≈ the
cross-label coalescence time. The structured coalescent then pins down where that mass can live:

- **Pure structure / ILS (no gene flow).** A and B lineages **cannot** cross-coalesce until they
  are together in the ancestral population, i.e. not below `T_split`. So

  ```
  q(t) ≈ 0   for t < T_split,    then steps up at/above T_split.
  ```

  "ILS" in the two-panel painting sense *is* trans-`T_split` incomplete sorting: it always lands
  **deep**.

- **Admixture.** Gene flow places cross-label coalescence mass **below** `T_split` — a bump near
  `T_admix`, on top of the deep step.

The discriminator is therefore concrete:

> **The mass of `q(t)` below `T_split`.**
> ≈ 0 ⇒ pure structure / ILS.   A recent bump ⇒ admixture.
> The area-below-split is essentially an admixture estimator.

The **directionality** the EM already estimates adds a second, independent axis: ILS is
symmetric (`q_AB ≈ q_BA`); introgression is directional (asymmetric `q_AB` vs `q_BA`).

### An internal null for ILS, for free

You already have the pieces to calibrate this from the data itself. **Run the rate curve on the
reference panel alone** (no admixed queries):

- Where A-ref and B-ref lineages start merging *is* `T_split`.
- The deep tail above it *is* your empirical ILS / ancestral-structure null.
- Any **query** excess below that onset is admixture — measured against a null you did not have
  to assume.

## The "never infer ILS" framing, precisely

- **No ILS state ⇒ no categorical ILS output.** Correct, and unavoidable: the state space is
  ancestry `{A, B}`, not `{admixed, ILS}`.
- **But ILS surfaces two ways, both observable:**
  1. as **deep cross-ancestry transitions** in `q(t)` (the rate-through-time curve);
  2. as **low-confidence, near-`π` painting** at ILS loci where the local tree genuinely cannot
     discriminate (the "relaxes toward `π`" behaviour) — and distinct from the *missing-info*
     tag.

So you read admixture-vs-ILS off the **depth + confidence + direction** profile, not off a
label. The information is retained; it is simply not categorical.

## The honest caveat — the mugration approximation

The reason this is "read off the curve" rather than a rigorous test: the mugration model treats
ancestry as a trait on a **fixed** genealogy and **does not enforce** the structured-coalescent
constraint. It permits a `Q`-jump on any branch at any time, with no built-in prohibition on
cross-coalescence below `T_split`. Consequences:

1. **It can be fooled.** Tree-inference error, or rare genuine deep-but-not-that-deep
   coalescence, can be read as recent admixture → a false admixture signal. The discrimination
   is only as trustworthy as the branch **times** — squarely the panmictic-prior calibration
   concern. This is a strong argument for coalescent-calibrated times (e.g. SINGER) when the goal
   is actually to *date* the bump.
2. **It degrades exactly as `T_admix → T_split`.** Ancient admixture coalesces at depths
   overlapping ILS, so the recent bump stops separating from the deep step. This is a
   **fundamental coalescent confound** — the same introgression-vs-ILS limit that recurs
   throughout phylogenomics — not a `tspaint`-specific flaw. **`tspaint` distinguishes the two
   to the precise extent that the admixture is more recent than the split.** This is also the
   regime where the query↔reference genealogical link itself decays under present-day admixed
   sampling, so it is doubly the hard limit.

The principled version of this discriminator is **structured-coalescent-aware** — BASTA
([De Maio et al., 2015](https://doi.org/10.1371/journal.pgen.1005421)), MASCOT
([Müller et al., 2018](https://doi.org/10.1093/molbev/msx312)), or **SCAR** on the inferred ARG
([Guo et al., 2022](https://doi.org/10.1371/journal.pcbi.1010422)). There, cross-coalescence
below `T_split` *requires* migration generatively, so admixture-vs-ILS becomes a likelihood
contrast rather than a curve-shape judgment. That is where this points if the curve-shape
diagnostic needs to become a formal test.

## A concrete, falsifiable validation

Directly runnable with the existing simulation harness; it would make a strong figure.

- **Null (ILS only):** a two-population deep split with **no pulse**. Expect `q(t) ≈ 0` below
  `T_split`, a step at/above it, and `q_AB ≈ q_BA` (symmetric).
- **Signal (admixture):** `sim.simulate_admixture_*` at several `T_admix`. Expect a sub-`T_split`
  bump that **shrinks and merges into the deep step as `T_admix → T_split`**, and that vanishes
  at old admixture where the query↔reference link is gone.

Plot `q(t)` for both on one axis. **The separation — and its predicted collapse at old
admixture — is the answer to the question, measured.** A second panel of `q_AB(t) − q_BA(t)`
tests the symmetry axis.

## Bottom line

The cross-ancestry **rate-through-time** can distinguish admixture from ILS; a single
homogeneous `Q` cannot. The method never *labels* ILS (it has no such state), but it does not
need to: ILS is the deep, symmetric, low-confidence baseline of `q(t)`, and admixture is the
recent, directional, confident excess above it. The discrimination is real, it is bounded by
branch-time calibration and by the `T_admix → T_split` coalescent confound, and its rigorous
form is a structured-coalescent model on the same ARG.

---

### References

- De Maio, N., Wu, C.-H., O'Reilly, K. M. & Wilson, D. (2015). New routes to phylogeography: a
  Bayesian structured coalescent approximation (BASTA). *PLoS Genet.* 11, e1005421.
  doi [10.1371/journal.pgen.1005421](https://doi.org/10.1371/journal.pgen.1005421).
- Guo, F., Carbone, I. & Rasmussen, D. A. (2022). Recombination-aware phylogeographic inference
  using the structured coalescent with ancestral recombination (SCAR). *PLoS Comput. Biol.* 18,
  e1010422. doi [10.1371/journal.pcbi.1010422](https://doi.org/10.1371/journal.pcbi.1010422).
- Lemey, P., Rambaut, A., Drummond, A. J. & Suchard, M. A. (2009). Bayesian phylogeography finds
  its roots. *PLoS Comput. Biol.* 5, e1000520. doi
  [10.1371/journal.pcbi.1000520](https://doi.org/10.1371/journal.pcbi.1000520).
- Müller, N. F., Rasmussen, D. A. & Stadler, T. (2018). MASCOT: parameter and state inference
  under the marginal structured coalescent approximation. *Mol. Biol. Evol.* 35, 2307–2321.
  doi [10.1093/molbev/msx312](https://doi.org/10.1093/molbev/msx312).
- Speidel, L., Forest, M., Shi, S. & Myers, S. R. (2019). A method for genome-wide genealogy
  estimation for thousands of samples (Relate). *Nat. Genet.* 51, 1321–1329.
  doi [10.1038/s41588-019-0484-x](https://doi.org/10.1038/s41588-019-0484-x).

*See also `CLAUDE.md` §1 (two Markov structures), §6 (mugration approximation, time
calibration), §9 (signal loss at old admixture), §10 (SCAR, structured-coalescent precedents).*
