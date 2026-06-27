# CLAUDE.md — Tree-Sequence Local Ancestry Inference (working name: `tspaint`)

> Project memory for Claude Code. This file is the authoritative spec. Read it
> fully before writing code. It encodes design decisions, the math, the
> tree-sequence accounting that prevents silent bugs, the open validation
> questions, and a deferred alternative (loopy BP / EP) with criteria for when
> it becomes necessary.

---

## 0. One-paragraph summary

We infer **local ancestry** along individual haplotypes from an inferred
**tree sequence** (e.g. Relate → tskit, or tsinfer). Ancestry is modelled as a
discrete character (2 populations for v1, K-way later) evolving up the branches
of each marginal tree under a continuous-time Markov chain (CTMC) with generator
**Q** ("mugration" / discrete phylogeography). Reference haplotypes provide tip
labels through a **soft** noise model whose per-tip credibility `w_i` is
learned, so admixed references stop anchoring. We fit **(Q, root frequencies,
{w_i})** by **EM**: the E-step is Felsenstein pruning (exact, per marginal tree,
per root); the M-step is closed-form CTMC MLE from expected dwell times and jump
counts. Sufficient statistics are accumulated **per tree-sequence edge,
span-weighted**, so a clade persisting across many trees is counted **once** —
this both removes double-counting and is the channel through which genome-scale
autocorrelation enters. Output: for every haplotype, at every position, a
calibrated posterior `P(ancestry = A)` — soft, not a hard call. The same
machinery, run on reference tips, is a **leave-one-out introgression/mislabel
detector**.

The novelty is the **synthesis** (generative ancestry-CTMC on an inferred tree
sequence + EM-learned Q and per-tip credibility + edge-blocked span-weighted
sufficient statistics → calibrated soft local ancestry). Components individually
exist; this combination, for LAI, appears unoccupied (see §10).

---

## 1. Why this is not the obvious thing

- **Not global ancestry.** A haplotype is a tip in *every* marginal tree. If we
  kept one color per individual and pooled across trees, we'd converge to a
  global ancestry proportion. Local ancestry requires the latent state to be
  **per-tip-per-tree** (equivalently per-edge), free to vary along the genome.
- **Two Markov structures, both modelled.**
  - *Vertical* (up a tree): the ancestry CTMC with generator Q. Felsenstein
    pruning. Estimates the **ancestral-migration timescale**.
  - *Horizontal* (along the genome): ancestry tracts. **Carried by edge span**,
    not by a separate switch-rate parameter ρ. A lineage's ancestry can only
    change where its genealogical context changes — i.e. at recombination
    breakpoints where edges begin/end. Between breakpoints the context is
    identical, so there is nothing to switch against. This removes ρ as a free
    nuisance parameter; the recombination structure Relate/tsinfer already
    inferred supplies it.
- **Decomposition of state change (decided):** recombination (horizontal,
  topology change) vs. ancestral migration (vertical, Q). The edge
  representation separates them cleanly. Q therefore estimates *only* the
  ancestral-migration timescale; tract switching is handled entirely by topology.
  This is a more specific and more falsifiable model than a generic ρ-HMM.

---

## 2. Model

### 2.1 State space and generator
Ancestry state `s ∈ {A, B}` for v1 (generalize to `{1..K}`). CTMC generator on
each branch:

```
Q = [ -q_AB    q_AB  ]
    [  q_BA   -q_BA  ]
```

Transition over a branch of length `t`: `P(t) = expm(Q * t)`.
Root state drawn from root frequencies `π = (π_A, π_B)`.

For K states, `Q` is a `K×K` rate matrix (rows sum to 0); everything below
generalizes by swapping the generator — keep the code generator-agnostic.

### 2.2 Tip emission (soft clamping + learned credibility)
Each tip `i` has a sampled label `ℓ_i ∈ {A, B}` (queries: none / flat). Instead
of a one-hot clamp, the tip's Felsenstein likelihood vector is a noise model
parameterized by credibility `w_i ∈ [0,1]`:

```
emission_i(s) = w_i * 1[s == ℓ_i] + (1 - w_i) * π(s)      # labelled tips
emission_i(s) = π(s)                                       # query tips (flat/root-freq)
```

- `w_i = 1` ⇒ hard clamp. `w_i → 0` ⇒ tip is effectively re-inferred from the
  rest of the tree (like a query). Learned per tip (§4.3).
- **Anchor set:** keep a trusted subset with `w_i ≡ 1` (most-confidently
  unadmixed references). Put a `Beta(α, β)` prior with mass near 1 on the rest
  so labels are believed *unless the genealogy insists otherwise*. **Do not let
  the entire panel go soft simultaneously** — that reintroduces unsupervised
  degeneracy (label switching, Q/ w_i trade-off, collapse). See §6.

### 2.3 Full-introgression-mapping decision (committed)
Reference and query haplotypes are the **same kind of object**: every tip at
every tree has a latent ancestry state and yields a posterior. The only
difference is the strength of the Beta prior on label-emission. Consequence: a
reference's own posterior will dissent from its label over a genuinely foreign
tract — that *is* the introgression map, same machinery, no extra code path.

### 2.4 What the deliverable is
For each tip (haplotype) in each marginal tree (equivalently each tip-edge span),
the **down-pass posterior** `P(s = A | tree, params)`. Spatially smoothed because
the tree topology (shared across the span) and the edge-blocked accounting carry
autocorrelation. When the local tree cannot discriminate (query coalesces with
both source clades at similar depth), the posterior relaxes toward `π` / 50–50
rather than making a confident wrong call — this is emergent, not bolted on.

---

## 3. Inference: structured / blocked EM (the v1 method)

We do **exact pruning per marginal tree** for the E-step, but **accumulate
sufficient statistics per edge, span-weighted**. This captures the
double-counting fix and most of the autocorrelation benefit while keeping each
E-step exact. (The deferred tighter alternative — loopy BP/EP over the full ARG
coupling graph — is §7.)

### 3.1 E-step: Felsenstein pruning (per tree, per root)
Up-pass (post-order, tips→root) computes partial likelihoods `L_u(s)`:

```
L_u(s) = Π_{c ∈ children(u)} [ Σ_{s'} P(t_{u→c})[s, s'] * L_c(s') ]
```

- Tips: `L_u(s) = emission_u(s)` (§2.2).
- **Arbitrary arity (polytomies) handled natively** — product over *all*
  children. Traverse children via `left_child` / `right_sib` (quintuply-linked
  encoding); do **not** assume binarity.
Down-pass computes posterior marginals `γ_u(s) ∝ L_u(s) * M_{parent→u}(s)` and
joint parent-child posteriors `ξ_{(p,c)}(s_p, s_c)` (the K×K expected-transition
object per branch). Standard two-pass message passing.

### 3.2 Per-branch expected sufficient statistics (the CTMC reward / Phasic object)
For a branch of length `t` under `Q`, conditioned on the endpoint-pair posterior
`ξ`, we need **expected dwell time per state** and **expected jump counts per
ordered pair**. These are reward-accumulated CTMC functionals computed via the
**Van Loan block-triangular matrix exponential**:

Expected time spent in state `m` along the branch, integrated against the CTMC,
uses

```
expm( [ Q    E_m ] * t ) = [ P(t)   ∫_0^t P(τ) E_m P(t-τ) dτ ]
     ( [ 0    Q   ]       )   [ 0      P(t)                    ]
```

where `E_m` is the indicator matrix for the reward (for dwell time in state `m`,
`E_m = e_m e_m^T`; for the number of `m→n` jumps, `E = q_{mn} e_m e_n^T`). The
top-right block gives the integral needed to form the expected reward conditioned
on the branch endpoints. For 2 states these are closed-form scalars; **implement
via the block-exponential anyway** so K-way is a generator swap, not a rewrite.

```python
# branch_stats.py  (sketch — verify numerics, especially small/large t)
import numpy as np
from scipy.linalg import expm

def branch_expected_stats(Q, t, xi):
    """
    Q   : (K,K) generator
    t   : branch length (>0; skip root branches, see §3.4)
    xi  : (K,K) posterior over (parent_state s_p, child_state s_c) for this branch,
          normalized to sum to 1 (this branch's contribution weight is applied
          separately as span weight in the accumulator).
    Returns:
      dwell : (K,) expected time in each state along the branch,
              conditioned on endpoint posterior xi
      jumps : (K,K) expected number of m->n transitions, conditioned on xi
    """
    K = Q.shape[0]
    P = expm(Q * t)                       # (K,K) endpoint transition probs
    # Guard against zero entries when conditioning:
    Psafe = np.where(P > 0, P, 1e-300)

    dwell = np.zeros(K)
    for m in range(K):
        Em = np.zeros((K, K)); Em[m, m] = 1.0
        block = np.zeros((2*K, 2*K))
        block[:K, :K] = Q; block[K:, K:] = Q; block[:K, K:] = Em
        top_right = expm(block * t)[:K, K:]     # ∫_0^t P(τ) Em P(t-τ) dτ
        # E[time in m | start=s_p, end=s_c] = top_right[s_p, s_c] / P[s_p, s_c]
        cond = top_right / Psafe
        dwell[m] = np.sum(xi * cond)

    jumps = np.zeros((K, K))
    for m in range(K):
        for n in range(K):
            if m == n or Q[m, n] == 0:
                continue
            Emn = np.zeros((K, K)); Emn[m, n] = Q[m, n]
            block = np.zeros((2*K, 2*K))
            block[:K, :K] = Q; block[K:, K:] = Q; block[:K, K:] = Emn
            top_right = expm(block * t)[:K, K:]
            cond = top_right / Psafe
            jumps[m, n] = np.sum(xi * cond)
    return dwell, jumps
```

> NOTE: This is the natural home for **Phasic**. The per-branch expected
> occupation times and jump counts of a finite-state CTMC are exactly the
> reward-accumulated phase-type / Van Loan objects Phasic computes. Replace the
> `expm`-block calls with Phasic's machinery once the interface is settled;
> Phasic should also give better numerics (symbolic caching of repeated `Q`,
> stable handling of stiff `Q*t`). Keep `branch_expected_stats` as the seam.

### 3.3 The blocking: accumulate per edge, span-weighted (THE correctness core)
A tree-sequence **edge** is `(left, right, parent, child)` over half-open
`[left, right)`. A clade persisting across many marginal trees is **one set of
edges with wide span**, not repeated rows (given `--compress`, see §5/§8). Drive
the loop with `edge_diffs()`:

```python
# accumulate.py  (sketch)
# Sufficient-statistic accumulators (pooled over the whole genome):
#   S_dwell  : (K,) total expected dwell per state (span-weighted)
#   S_jumps  : (K,K) total expected jumps per pair (span-weighted)
#   S_root   : (K,) total expected root-state mass (per root, per span)
#   S_cred[i]: per-tip expected (agree, disagree) counts for Beta update

for (interval, edges_out, edges_in), tree in zip(ts.edge_diffs(), ts.trees()):
    left, right = interval
    span = right - left

    # Maintain an incremental forest. For each ROOT in tree.roots:
    #   run up-pass + down-pass pruning (handles polytomies via sib pointers),
    #   tip messages = emission_i (refs) or flat/root-freq (queries).
    # This yields, per branch active on this interval, the endpoint posterior xi,
    # and per tip the marginal gamma.

    # Bank an edge's contribution ONCE, when it enters (edges_in), weighted by
    # the edge's own span. While it persists, do nothing. (Blocked approx: hold
    # the child's pruning MESSAGE as of entry; see §3.5 for the controlled error.)
    for e in edges_in:
        if tree.parent(e.child) == tskit.NULL:      # root branch -> skip dwell/jumps
            continue                                  # (root handled via S_root)
        t = node_time[e.parent] - node_time[e.child]  # branch length (> 0; see §2 polytomy note)
        xi = endpoint_posterior(e.parent, e.child)    # (K,K), from down-pass
        w_edge = (e.right - e.left)                    # span weight
        dwell, jumps = branch_expected_stats(Q, t, xi)
        S_dwell += w_edge * dwell
        S_jumps += w_edge * jumps
        # credibility evidence for the CHILD tip if it is a labelled sample:
        if is_sample[e.child] and e.child in soft_refs:
            g = tip_marginal(e.child)                  # (K,)
            agree    = g[label[e.child]]
            disagree = 1.0 - agree
            S_cred[e.child][0] += w_edge * agree
            S_cred[e.child][1] += w_edge * disagree

    # Root-state mass for this interval's roots:
    for r in tree.roots:
        S_root += span * root_marginal(r)             # (K,)
```

Why this is exact for the double-counting fix: the edge-table invariant is that
**"the set of intervals on which each node is a child must be disjoint."** So
summing a child-edge's contribution weighted by `(right - left)` partitions the
genome without overlap and cannot double-count that child's branch.

### 3.4 M-step (closed form, O(#edges))
```
q_mn  = S_jumps[m, n] / S_dwell[m]        for m != n   (then set q_mm = -Σ_n q_mn)
π     = S_root / sum(S_root)
w_i   = (α - 1 + S_cred[i].agree) / (α + β - 2 + S_cred[i].agree + S_cred[i].disagree)
        # MAP under Beta(α, β); use posterior mean if preferred
```
All are sums over per-edge accumulators ⇒ M-step is `O(#edges)`, not
`O(#trees × #nodes)`.

### 3.5 What "blocked" approximates (be honest in the paper)
The horizontal Markov structure is **not dropped** — it is encoded *in the
blocking*: holding an edge's contribution constant over its span and keying by
edge asserts the lineage's ancestry state is constant while its genealogical
context (the edge) persists, i.e. the near-identity horizontal coupling, enforced
structurally rather than by BP messages. What is approximated: when a **node
persists but its parent edge is swapped** at a breakpoint, the node-state
posterior can be slightly inconsistent across the breakpoint (uncertainty does
not propagate through that swap). This residual **breakpoint flicker** is the
quantity to measure (§7.3). If small, blocked EM was the right call and loopy BP
is unnecessary.

---

## 4. Edge cases the tskit data model forces (do not skip)

These come straight from the tskit data model and will silently corrupt
sufficient statistics if traversed naively.

1. **Multiple roots / forests.** A marginal tree can have several roots
   (unlinked topologies jointly describing the samples). **Run pruning per root**
   (`for r in tree.roots`), each with its own `π` prior. There may be no single
   root.
2. **Isolated samples = "no information," not "uncertain."** A tip with no parent
   and no children over an interval is a root unto itself; the tree says nothing
   about its ancestry there. Its posterior must fall back to the prior, and the
   output must **tag this span as missing-info, distinct from 50–50 uncertain.**
   Conflating them is a real interpretive error for introgression mapping.
3. **Span accounting must cover the whole sequence per sample.** If you only
   accumulate over `Tree.nodes()`-reachable edges, isolated stretches contribute
   nothing to Q (correct) but must still emit prior-fallback output (else gaps
   look like missing output, not missing data).
4. **Virtual root is bookkeeping.** Its children are the real roots; its time is
   +∞; real roots have parent `tskit.NULL (-1)`, **not** the virtual root. Never
   let Q act on an edge to the virtual root (there are none). `tree.virtual_root`
   is a convenient root enumerator but has no node-table row — accessing its
   attributes throws.
5. **Root branch length is defined as 0** in tskit (definitional, traversal
   artifact). **Skip root branches** for dwell/jump accumulation; handle root
   state purely via `π` (`S_root`). See §3.3.
6. **Polytomies are real and first-class.** Internal nodes with >2 children
   exist. `time[parent] > time[child]` is strictly required, so you will never
   see a literally-zero branch in a valid tree sequence — a polytomy is many
   children at strictly positive lengths, not a zero-time knot. `expm(Q*t)` is
   always well-defined. Soft polytomies are integrated under
   conditional-independence-of-children-given-parent (ordinary product pruning
   with each child's own `t`). The only requirement is sib-pointer traversal.

---

## 5. Input pipeline: Relate → tskit (load-bearing flag)

**Convert with `--compress`. This is not optional.**

```bash
# Relate inferred trees -> tskit, unifying persistent clades.
relate_lib/bin/Convert --mode ConvertToTreeSequence \
    --compress \
    --anc example.anc.gz \
    --mut example.mut.gz \
    -o example
```

- `--compress` (added by Nathaniel S. Pope) assigns the **same age to nodes with
  identical descendant sets across adjacent trees**, unifying a persistent
  subclade into a single long-span set of edges. **This is what makes a subclade
  one node ID across trees**, which is the invariant the edge-blocking and the
  horizontal coupling depend on.
- **Without `--compress`**, Relate's conversion can mint fresh node IDs per local
  tree even for unchanged descendant sets ⇒ short churning edges ⇒ you keep the
  double-counting fix but **lose almost all autocorrelation benefit** (method
  still correct, just no better than per-tree pruning). Don't.
- **Caveat (source-confirmed in `relate_lib`, §8.1):** `--compress` unifies on
  *identical descendant set* — **exact sorted-leaf-set equality, no threshold**
  (`tree_sequence.hpp::FindIdenticalNodes`) — provably **stricter** than Relate's
  own "equivalent branch" notion (Pearson correlation ≥ 0.9, `anc.cpp::Branch‑
  Association`; Relate Supp. §4.1). A clade that persists but gains/loses one tip at
  a breakpoint is **split** into separate node IDs — autocorrelation capture is
  therefore slightly **conservative** (safe direction: never falsely merges
  distinct lineages, but under-links near-equivalent clades). Matters only if
  breakpoint flicker is high.
- **Refinement 1 — the global root is *not* one persistent node.** `--compress`
  deliberately re-IDs the root whenever either of its children changes (avoiding a
  forced genome-wide constant TMRCA), so the top node churns even though its
  descendant set is invariant. Benign — we skip root branches (§3.4, §4.5) — but do
  not expect the global root to be a single long-span node.
- **Refinement 2 — `--compress` reconciles node ages.** Each merged node ID gets a
  **span-weighted-average age**, then a monotonicity constraint (least-squares,
  `--tolerance` 1e-3, `--iterations` 500). Compressed branch lengths are thus *not*
  any single marginal tree's lengths but a smoothed reconciliation — an extra
  calibration layer **on top of** Relate's panmictic-prior branch lengths, which
  compounds the §6 time-calibration concern (the order-only/ranked ablation hedges
  it).

`tsinfer`/`tsdate` tree sequences have native cross-tree node-ID stability by
construction and are an alternative (or even preferable) front end; the method is
front-end agnostic as long as the persistence invariant holds.

### 5.1 First diagnostic, BEFORE any inference code
Put this in `notebooks/00_persistence_check.ipynb`. It tests whether the central
premise is delivered by the input:

```python
import tskit, numpy as np
from collections import Counter

ts = tskit.load("example.trees")

# (a) edge span distribution
spans = ts.tables.edges.right - ts.tables.edges.left
print("edge span: median", np.median(spans), "max", spans.max())

# (b) how many distinct trees each internal node survives  <-- THE key histogram
node_tree_count = Counter()
for (_interval, eout, ein), tree in zip(ts.edge_diffs(), ts.trees()):
    for u in tree.nodes():
        if not tree.is_sample(u):
            node_tree_count[u] += 1
counts = np.array(list(node_tree_count.values()))
print("internal-node persistence across trees: median", np.median(counts),
      "max", counts.max(), "frac==1", np.mean(counts == 1))
```
- If `frac==1` is near 1 (persistence spiked at a single tree), `--compress`
  didn't take / something upstream is wrong — **the method's premise is not met.**
- If there is real mass above 1, horizontal coupling is getting signal. Proceed.

---

## 6. Identifiability and stability (the collapse modes)

Once labels can move, the model can explain disagreement three ways — lower
`w_i` (this haplotype is admixed), inflate Q (everyone switches fast), or relabel
a whole clade. With a hard-clamped core these are pinned; fully soft they trade
off. Guards:

- **Keep a trusted anchor set** with `w_i ≡ 1`. Only the rest float.
- **Informative `Beta(α, β)` prior near 1** on soft refs: labels believed unless
  the genealogy insists otherwise.
- **Never let the whole panel go soft simultaneously** (label switching, unstable
  Q).
- **[MEASURED — softening slightly-impure references]** For references carrying a known
  minority of foreign tracts, **un-clamp them** (soft `w` + strong `Beta` prior) rather than
  hard-clamping — but the payoff is **introgression recovery, not query accuracy**, and it is
  **bound by the genealogical foreign-tract signal**, not the impure:pure ratio. A hard clamp
  makes the emission one-hot, so the impure-ref posterior is pinned (down-pass foreign recall
  **0 by construction**); any `w < 1` restores local override. On strong-structure true-ARG
  sims (`experiments.impure_reference_experiment` / `impure_reference_sweep`,
  `sim.simulate_admixture_impure_refs`): the payoff is **maximal with strong source anchoring +
  recent admixture** (many pure anchors, T_admix≈120) — down-pass recall **0 → 0.76**, the
  leave-one-out introgression map (`output.loo_posterior_table`) **0.64 → 0.92**; **moderate**
  at baseline (down-pass 0.18 but LOO 0.56 — the LOO lens recovers ~3× more, since the down-pass
  hides foreign tracts behind the tip's own soft emission); and it **vanishes at old admixture**
  (T_admix≈1000: LOO Δ≈0.01, down-pass 0) where the query↔reference link is itself gone (§9
  signal-loss). The **query bal-acc gain is small everywhere (+0.01–0.03)** and ceiling-limited —
  the pure anchor core already carries the painting. Benefit is from **un-clamping, not prior
  strength**: learned `w` is flat across `Beta` α∈[2, 2000] (genome-scale span-weighted evidence
  swamps the prior ⇒ `w →` the ref's empirical purity), so the per-tip prior override
  (`em.fit(priors=...)`, `paint(..., priors=...)`) is available but ~inert at genome scale. Keep
  the pure anchors hard-clamped.
- Degenerate fixed points to watch in tests: `Q → 0` freezes initialization;
  `Q → ∞` washes everything to `π`; unsupervised mode label-switches/collapses.
- **Branch-length / time-calibration risk.** Relate estimates branch lengths
  under a *panmictic* coalescent prior — misspecified precisely in the
  structured/admixed regime of interest (Relate Supp. Fig. 3c: TMRCA bias under
  wrong Ne). If Q depends on calibrated `t`, this bias propagates. **Mitigation /
  ablation:** an order-only / ranked-topology variant of the ancestry model
  (trade calibration for robustness, in the spirit of Relate's order-based
  selection test) — `tspaint.ranked.ranked_tree_sequence`, `tspaint_paint(...,
  ranked=True)`.
  - **[MEASURED — order-only ablation is NOT beneficial; do not use for inference.]**
    Dense-ranking node times collapses true-ARG painting ~1.0 → ~0.50. Mechanism:
    rank compresses the timescale → EM fits a much larger Q (~1e-2 vs ~5e-5) → deep/
    root branches wash → **π becomes unidentifiable and drifts to a degenerate extreme**
    (`π≈[0.96,0.04]`) → confident-wrong painting. The §6 worry (mis-calibration hurts)
    was right; this particular cure is wrong — it discards the coalescence-depth
    *magnitudes* the CTMC needs, and worsens the π degeneracy below. Kept as a runnable
    ablation only (`notebooks`/`compare`), not a recommended mode.
- **π-identifiability (the real failure mode; the order-only ablation surfaced it).**
  When deep/root branches wash (large `Q·t`, e.g. sparse/short ARGs or the order-only
  variant), each tree's **root marginal just echoes π**, so π is under-determined and
  the M-step drifts it to a degenerate extreme → the painting collapses to one colour
  with high confidence (the head-to-head's confidently-wrong short-genome row, §9).
  **Fix — hold π fixed (`estimate_pi=False`, the `tspaint_paint` default).** π is a prior
  on the *arbitrary* GMRCA state, so uniform is principled; estimating it from washing
  roots is what breaks. **[MEASURED]** recovers both failures — true-ARG ranked
  0.50→0.94, tsinfer L=5e4 0.50→1.00 — and is harmless on good, long data (the regime
  where π *is* identified: L=2e5 tsinfer 0.984→0.999). The short/sparse regime itself is
  an edge case (real LAI runs on long genomes with many roots, where π is identified and
  tspaint already paints 0.98–1.0); the fix is a free robustness default, not a fix to a
  practical limitation. `estimate_pi=True` remains for π-recovery studies (`em.fit`).
- **Mugration approximation.** Treating ancestry as a trait on a *fixed*
  genealogy ignores that genealogy and ancestry are jointly distributed (lineages
  in different demes cannot coalesce) — the structured-coalescent point.
  BASTA/MASCOT correct it. For a v1 LAI tool the naive version is likely fine but
  **biases Q**; state this limitation explicitly.

---

## 7. DEFERRED ALTERNATIVE: loopy BP / EP over the full ARG coupling graph

Kept for future reference. Build only if §7.3 says blocked EM is insufficient.

> **[DECISION — superseded; see the loopy-bp-ep update below]:** Not needed *on the true ARG*.
> On strong-structure msprime sims (true ARG), breakpoint flicker at persist-but-reparent
> boundaries is ~0.001 vs. ~0.95 discontinuity at true switches, with per-base accuracy ~1.0 and
> good calibration (Rung 8, `notebooks/02`). Blocked EM is sufficient there. The flagged revisit
> on **inferred ARGs** has now been done — and there the verdict flips.

> **[MEASURED — `loopy-bp-ep` branch, BUILT].** Implemented the **single-pass horizontal BP/EP
> smoother** (`tspaint.bp` — `bp_smooth`, `bp_smooth_track`, `bp_paint`): a genome-axis
> forward-backward over each tip's per-tree beliefs with a per-breakpoint switch penalty ``ε``
> (the EP first half of §7.2's schedule, factorised per tip; full loopy = re-feeding into the
> vertical pruning of shared internal nodes, the ``n_sweeps>1`` extension, not yet built).
> Compared head-to-head against the per-position `output.hard_segments` **deadband** for
> *segmentation* fidelity (breakpoint F1 at the switch-density-matched operating point), over
> seeds, true vs inferred ARG (`tspaint.bp.bp_vs_deadband_experiment`):
>
> - **True ARG (4 seeds, T_admix 500/700/900):** the deadband **wins** — F1 0.95–0.99 (±≤0.03)
>   vs BP 0.89–0.93 (±~0.08), and BP's even-max-F1 is below it. The single-seed hint that BP won
>   at T=700 was seed noise. On clean per-tree posteriors a per-position confidence threshold is
>   near-optimal; BP's run-length smoothing trades calibration away and adds variance.
> - **Inferred ARG (tsinfer, 3 seeds): BP wins decisively** — F1 **0.71→0.98** at T_admix=500 and
>   **0.74→0.90** at T_admix=700. Tree inference scatters spurious breakpoints whose per-tree
>   posterior is *not* low-confidence, so the deadband cannot filter them; BP's spatial smoothing
>   recovers the tract structure. BP also nudges per-base balanced accuracy up (+0.005–0.016)
>   in every regime.
>
> **Verdict: §7's horizontal coupling is needed on inferred ARGs — the realistic input — and
> redundant on the true ARG.** `bp_paint` is the recommended segmentation path when painting on a
> tsinfer/Relate ARG; on a true/known ARG, blocked EM + deadband suffices. The single-pass tip
> smoother already captures a large effect; full-loopy (internal-node coupling) is a further
> possible gain, now well-motivated but lower priority. Driver: `bp.bp_vs_deadband_experiment`;
> tests: `tests/test_bp.py`.

### 7.1 Why it exists
The inference object is a graph: nodes are (lineage, genomic-span) edges; couplings are
- **vertical** (within a tree): parent-edge ↔ child-edge via the CTMC `expm(Qt)`;
- **horizontal** (along genome): a node's state across a breakpoint where it
  persists but its parent edge is swapped — near-identity coupling.

This graph **has cycles** (a subclade persisting across a region where another
part of the tree recombines forms a loop in the coupling graph). Exact marginals
are therefore unavailable in general. Blocked EM (§3) sidesteps this by doing
exact pruning per tree and folding the horizontal coupling into the blocking; it
loses only the propagation of *uncertainty* across persist-but-reparent
breakpoints.

### 7.2 What the upgrade is
Run **loopy belief propagation** (or **expectation propagation**) on the full
edge graph:
- Messages along **vertical** edges = the usual CTMC factor `expm(Qt)`.
- Messages along **horizontal** links = near-identity factor (small switch noise)
  between a node's state on the left vs. right of a breakpoint.
- Schedule: forward–backward *along the genome* interleaved with up–down *within
  trees*; iterate to message convergence; then EM on the resulting (approximate)
  marginals. EP if the horizontal factor needs a Gaussian/Dirichlet relaxation
  for stability or for K-way.

Tighter coupling ⇒ better propagation of uncertainty across breakpoints ⇒
smoother, better-calibrated tracts at the cost of approximate (non-exact)
marginals and a convergence loop.

### 7.3 Decision criterion — is loopy BP/EP required?
Measure on blocked-EM output:
1. **Breakpoint flicker.** For tips/queries, quantify posterior discontinuity at
   persist-but-reparent breakpoints: e.g. mean `|P_left(A) - P_right(A)|` across
   such breakpoints, and rate of sign flips of `argmax` across them. Compare to a
   simulated truth (known tracts) discontinuity.
2. **Calibration.** Reliability diagram of `P(A)` vs. empirical correctness on
   simulated admixture (msprime with known local ancestry). If blocked EM is
   under/over-confident specifically near breakpoints, that's the BP-shaped gap.
3. **Tract-boundary accuracy.** Compare inferred vs. true switch points
   (breakpoint localization error) against blocked EM.

**Build loopy BP/EP only if** breakpoint flicker is large relative to true-tract
discontinuity AND it materially degrades tract-boundary accuracy or calibration
near boundaries. If flicker is small, blocked EM is the published method and BP
is a footnote.

> **[MEASURED — Rung 8]:** flicker at non-true boundaries ≈ 0.001 vs. ~0.95
> discontinuity at true switches (ratio ~1e-3); accuracy ~1.0; calibration
> ~diagonal — on strong-structure, true-ARG sims. Criterion **not met ⇒ `bp/`
> deferred.** Metric: `experiments.flicker_vs_true_boundaries`; notebook
> `02_calibration_flicker`. Caveat: favorable regime; the inferred-ARG / ancient
> stress test is the outstanding work.
>
> **[UPDATE — older-admixture stress, via the §9 fragmentation finding]:** at *older*
> admixture (T_admix=200, true ARG) the flicker is no longer negligible — it surfaces as
> ~3× over-fragmentation in the **hard** `argmax` segmentation (the dating-relevant output),
> because the posterior hovers near 0.5 in places and argmax flips there. This is the first
> regime where §7.3's criterion is genuinely approached. **It does not yet trigger `bp/`:** the
> flips are low-confidence, so a post-hoc confidence **deadband** (`output.hard_segments`)
> recovers the true tract-length distribution (switch-density ratio 1.00) without message
> passing. `bp/` becomes the principled fix only in the *weak-signal* regime where real
> short-tract switches also go low-confidence and the deadband can no longer separate them from
> spurious flips (precision↔recall no longer cleanly splittable) — the outstanding trigger to
> watch.

### 7.4 ARG-posterior ensemble merging (SINGER) — the front-line answer to §9

§9 shows painting is **bounded by ARG accuracy**, and §7's `bp/` only addresses
*within-ARG* uncertainty. The larger, binding uncertainty is **which ARG** — the
topology/time error of a point estimate (tsinfer/Relate). The principled fix is to
**marginalise the ARG posterior**: given an ensemble of tree sequences sampling
`P(ARG | genotypes)` — e.g. thinned MCMC samples from **SINGER** (Deng et al., 2024) —
the deliverable is

```
P(s_i(x)=A | data) = E_{G~P(G|data)}[ γ_i^G(x,A;θ) ] ≈ (1/M) Σ_m γ_i^{G_m}(x,A;θ)
```

i.e. paint each member and **average the per-position posteriors**; the spread is an
ARG-uncertainty band. This is a modular ("cut") model: the ARG posterior comes from the
genotypes, not the ancestry labels (sparse tip annotations) — a standard, well-justified
cut that keeps SINGER outside the EM loop. SINGER's coalescent-calibrated times also
mitigate the §6 time-calibration bias and propagate its uncertainty.

**Implemented (prototype):** `ensemble.merge_posterior_tables` (N-way breakpoint
refinement → mean + std + status, duck-compatible with `Segment` so `validate` scores it
directly); θ fit pooled across the ensemble **reuses** `em.fit([G_1..G_M], [labels]*M)`
(scale-invariant M-step). Driver `experiments.arg_ensemble_experiment`. **Surfaced in the
high-level API:** `paint([G_1..G_M], labels)` does the pooled fit → per-member paint → merge
in one call and returns a `Painting` whose `posteriors` carry the mean + `posterior_std` band
(`Painting.introgression_map` likewise merges the leave-one-out map across the ensemble).

> **[MEASURED — prototype].** The merge layer is correct and, on **independent** noisy
> inputs, averaging clearly improves accuracy (synthetic test: ~0.65 → >0.9). But a
> **tsinfer point-estimate ensemble does NOT help** (e.g. 0.73 → 0.71 in a noisy regime):
> its members share tsinfer's bias and the same genealogy, so their errors are
> *correlated* and averaging dilutes confident-correct calls instead of fixing them.
> **That is the point** — the benefit needs genuine *posterior* draws (independent-ish
> errors), which SINGER provides and a point-estimate ensemble does not.

> **[BUILT + MEASURED — SINGER].** `io.singer` (formerly `io_singer.singer_tree_sequences`) runs the SINGER
> binary (haploid VCF → `singer -ploidy 1` with an auto-retry loop → tskit; sample order
> preserved, order-aligned = 1.000; node times in generations). Driver
> `experiments.singer_ensemble_experiment`. The headline **flips the §9 story**: SINGER
> single posterior samples already paint at **~0.99 even at very sparse data (~160–300
> sites) and older admixture** — where tsinfer gives ~0.58 (chance). SINGER's Bayesian SMC
> inference **largely lifts the §9 ARG-quality bound by itself**, so merging adds little
> *accuracy* (single is already near-ceiling) but provides a **calibrated uncertainty
> band** (merged confidence falls 0.88→0.52 as data thins — correctly widening). Takeaway:
> **the ARG inference method dominates the merge; tspaint + SINGER is the strong
> combination.** The merge's accuracy benefit should surface where SINGER's posterior is
> genuinely broad and accuracy-limiting (much larger samples / very low diversity) —
> outstanding. (SINGER's own `convert_to_tskit` omits `compute_mutation_parents()` and
> crashes on recurrent mutations; `io_singer` fixes that.) This direction **supersedes
> `bp/`** for ARG uncertainty and fixes a §6 caveat (calibrated times).

> **[MEASURED — merge-helps hunt].** No regime found where merging SINGER samples improves
> *argmax accuracy* (gain ≤ 0 across weaker structure / more diversity / sparse data;
> single SINGER stays 0.87–0.99, e.g. single 0.87 / merged 0.85 / true 0.97). Averaging
> posteriors then taking argmax dilutes confident-correct calls, so the merge's value is
> the **calibrated uncertainty band**, not accuracy — the **number of samples is a user
> knob for uncertainty/robustness, spent only when needed** (`singer_ensemble_experiment`,
> `n_singer`). A genuine accuracy gain awaits an ARG-uncertainty-limited regime (much
> larger samples / very low diversity / whole-genome) not reached in these sims.

---

## 8. Open issues requiring validation (do these early)

1. **`--compress` semantics (source-level). [RESOLVED — §5 caveat + refinements.]**
   Confirmed against `relate_lib` source (`tree_sequence.hpp::FindIdenticalNodes`,
   `anc.cpp::BranchAssociation`): unifies on **exact identical descendant set, no
   threshold** — stricter than the equivalent-branch notion; ±1-tip clades split
   (conservative). Two refinements found: the global root is re-IDed on any child
   change, and node ages are span-weighted-averaged then monotonicity-constrained.
2. **Node-persistence delivered in practice.** Run §5.1 on a real converted file.
   The persistence histogram is the go/no-go on the method's premise.
3. **Prior-art pass. [DONE — see §10.]** The **mechanism** appears unpublished
   (moderate-to-high confidence), but the **framing** is contested: a live
   ARG-native-LAI cluster exists (ARGMix; Pearson & Durbin), and the
   structured-coalescent-on-inferred-ARG lineage *is* published — **SCAR** (Guo,
   Carbone & Rasmussen, 2022) — but for migration demography, not per-haplotype
   painting. Residual terminology to sweep before the intro: "stochastic mapping /
   Markov jumps & rewards on tree sequences"; phylogeographic painting along a
   chromosome (SCAR citing-lit); tsinfer/tsdate ecosystem (tskit-dev#11 "chromosome
   painting"); archaic-introgression papers formalizing Relate deep-branch labelling.
4. **Time-calibration ablation.** Quantify Q bias from panmictic branch lengths;
   compare time-calibrated vs. order-only ancestry model (§6).
5. **Blocked-EM error (the §3.5 approximation). [MEASURED — Rung 8.]** On
   strong-structure true-ARG sims, flicker at non-true boundaries ≈ 0.001 vs. ~0.95
   at true switches; accuracy ~1.0; calibration ~diagonal ⇒ blocked EM validated,
   §7 (`bp/`) deferred. Still to stress: inferred ARGs (Relate/tsinfer) and
   weak-structure / ancient-admixture regimes.
6. **`edge_diffs()` / `trees()` alignment.** Confirm `zip(ts.edge_diffs(),
   ts.trees())` yields matched `(interval, tree)` pairs in your tskit version,
   and that `edges_in`/`edges_out` reference stable child/parent IDs and
   `left/right` as assumed. Cheap to assert; expensive to discover wrong three
   weeks in.
7. **Numerics of `branch_expected_stats`.** Stress-test small `t` (≈0 after
   compress merges near-coincident times) and large `Q*t` (stiff). This is where
   Phasic should replace `scipy.linalg.expm`.

---

## 9. Validation / experiment plan

- **Simulated truth (primary).** `msprime` admixture scenarios with **known local
  ancestry** (record true ancestry along each haplotype). Vary: admixture age
  (recent → ancient), admixture fraction, reference panel purity, sample size.
  Infer ARG with Relate (`--compress`) and with tsinfer; run `tspaint`.
  - Metrics: per-base ancestry accuracy; calibration (reliability diagram);
    tract-boundary localization error; behaviour vs. admixture age.
- **Headline hypothesis (now a head-to-head, not a novelty claim).** Tract-/
  copying-based methods (RFMix, MOSAIC, FLARE) degrade as admixture ages and tracts
  shorten below their resolution. A tree-native method may reach **older** admixture,
  because the genealogical relationship of a query to source references at a locus
  survives even after recombination shreds the surrounding haplotype — **bounded by
  tree accuracy**, which becomes the binding constraint rather than tract length.
  **Caveat (prior art, §10):** this older-admixture thesis is no longer unoccupied —
  ARGMix (Shanks et al., 2026) and Pearson & Durbin (2023) stake it with ML/
  transformer machinery. So pitch `tspaint` not on *category* ("first tree-native
  LAI") but on what those competitors structurally lack: **generative, calibrated,
  interpretable (an explicit Q + readable per-tip credibility), label-noise-robust**,
  with the **edge-blocked span-weighted sufficient statistics (§3.3) as the lead
  novelty** (found nowhere else in the LAI/ARG literature). Test directly: accuracy +
  calibration vs. admixture age, `tspaint` vs. RFMix/MOSAIC/FLARE **and head-to-head vs.
  ARGMix / Pearson & Durbin**.
- **[MEASURED — Rung 8] Bounded by ARG accuracy, confirmed.** On strong-structure
  sims the true ARG paints at accuracy ~1.0; on a **tsinfer-inferred** ARG accuracy
  is data-density-dependent — ~0.53 at ~650 sites (poor ARG, near chance), ~0.64 at
  ~2100, ~0.88 at ~5400 (good ARG) — for balanced 50/50 admixture (chance 0.5). Tree
  accuracy, not tract length, is the binding constraint, exactly as predicted. Front
  end `io_tsinfer.py`; driver `admixture_experiment(infer=True)`. Outstanding: Relate
  front end, ancient / weak-structure regimes, and the head-to-head vs. comparators.
- **[MEASURED — Rung 8] Discrimination vs. admixture age (true ARG, deep split so
  only tract length varies; balanced 50/50).** Balanced accuracy / mean confidence:
  T_admix=30 → 0.99 / 0.72; 300 → 0.93 / 0.36; 1000 → 0.50 / 0.15; 3000 → 0.50 / 0.18.
  Discriminates well at recent–moderate admixture; the **reference signal is lost at
  old admixture** — here *not* from tree-inference error but from coalescent structure:
  admixed individuals sampled at the present coalesce **among themselves** before the
  old pulse, severing the query→reference genealogical link. **Refines the headline:**
  tract length is not the binding constraint, but the query↔reference link is, and it
  decays with admixture age under present-day admixed sampling. Crossover age is
  scenario-specific (scales with Ne / sampling); the qualitative loss-at-old is robust.
  Methodological note: plain accuracy is misleading on a lopsided truth (argmax on
  P≈0.5) — **report balanced accuracy + confidence** (`validate.balanced_accuracy`,
  `validate.mean_confidence`).
- **Baselines / comparators.** *Segment/copying incumbents:* RFMix (**wired** —
  `io_rfmix.py`, see the RFMix head-to-head below), MOSAIC, FLARE.
  *ARG-native LAI (same task, different machinery — the real head-to-head):* ARGMix
  (Shanks et al., 2026; graph transformer on Relate trees), Pearson & Durbin (2023,
  "AncestralPaths"; NN on inferred tree sequences). *Nearest ARG-native but different
  objects:* ARGformer (embedding+retrieval, unsupervised), `sticcs`+topology
  weighting (topology frequencies), SCAR (Guo et al., 2022; structured-coalescent
  migration rates on inferred ARGs, not painting). See §10.
- **[MEASURED — head-to-head harness] (`compare.py`).** A uniform painter-scoring harness
  (`head_to_head`, `score_painter`); painters = the full method (`tspaint_paint`) and a
  runnable **ARG-native baseline** `nearest_reference_paint` (paint each query by its
  nearest labelled reference in the local tree — no CTMC/EM). External tools (RFMix/…,
  ARGMix, Pearson & Durbin) slot in as painters (not bundled — separate installs). Two
  findings: (i) on **adequate genomes** tspaint matches the baseline's accuracy (~0.98–1.0,
  true & tsinfer) but is **honestly calibrated** (confidence ~0.69–0.85) where the baseline
  is blindly overconfident (1.0) — tspaint's edge is the calibrated soft posterior +
  credibility, not raw accuracy; (ii) on **short/sparse regions** tspaint *was* confidently
  wrong (balanced 0.50 / conf 0.98 at L=5e4) where the topology-only baseline is robust.
  **Diagnosed and fixed:** the cause is **π-identifiability** (§6), not branch-length
  magnitudes per se — sparse ARGs wash the deep branches, π drifts to a degenerate extreme,
  and the painting collapses to one colour. Holding π fixed (`estimate_pi=False`, now the
  `tspaint_paint` default) recovers it: L=5e4 tsinfer **0.50→1.00**, and L=2e5 tsinfer
  0.984→0.999 (harmless on good data). The order-only/ranked variant was the *hypothesised*
  fix but makes it worse (§6 — it inflates Q and deepens the same π degeneracy). Net: tspaint
  now matches the baseline's accuracy on both short and long genomes while staying
  calibrated; the regime is an edge case anyway (real LAI is long-genome, π-identified).
- **[MEASURED — RFMix head-to-head] (`io_rfmix.py`, `rfmix_paint`).** RFMix v2.03 (Maples
  et al., 2013), the field-standard segment / random-forest+CRF incumbent, wired as a painter
  through the same harness. Being **genotype-native** it ignores the ARG: the bridge writes
  phased query/reference VCFs + a reference sample-map + a linear genetic map from the sim, runs
  the binary, and parses its `.fb.tsv` per-marker **posteriors** back to per-haplotype Segments
  (a fair soft-vs-soft comparison). Isolated in the `compare` pixi env (bioconda; binary via
  `TSPAINT_RFMIX` / `.pixi/envs/compare/bin/rfmix`), so the core stack is untouched. Results
  (balanced / mean-confidence; 6 admixed, 6+6 refs, T_admix=30, f_A=0.5): **strong structure**
  (Ne=1e3, T_split=5e3, L=1e6) — rfmix 0.99/1.00, tspaint 0.99/0.68, nearest_ref 0.99/1.00: all
  three tie on accuracy, only tspaint stays soft (RFMix's posteriors **saturate to 0/1**). **Weak
  structure** (Ne=1e4, T_split=2e3) — rfmix 0.78–0.85 / conf 0.92–1.00 vs tspaint 0.77–0.80 / conf
  0.10: balanced accuracy comparable (neither consistently ahead), but RFMix is **overconfident**
  (conf ≫ accuracy) where tspaint's posteriors relax toward 0.5 as the query↔reference genealogical
  signal fades. **Takeaway: against the field standard, tspaint matches RFMix's accuracy across
  regimes and neither dominates on accuracy; tspaint's distinguishing value is the calibrated soft
  posterior + readable Q/credibility** — the same conclusion reached vs. the topology-only
  baseline, now confirmed against RFMix. (Cost: RFMix ~9–125 s/run; tspaint ~7–86 s; nearest_ref
  <1 s.) Outstanding: MOSAIC/FLARE, and the ARG-native ARGMix / Pearson & Durbin (separate
  installs slotting in as painters).
- **[MEASURED — fragmentation / tract-length distribution] (`output.hard_segments`,
  `validate.breakpoint_precision_recall`, `validate.switch_density`).** Downstream
  admixture-pulse dating reads the **segment-length distribution**, so spurious short
  opposite-ancestry calls (fragmenting a long tract) bias the inferred pulse *older*, and
  over-smoothing biases it *younger*. Measured over **6 seeds** (`experiments.fragmentation_experiment`;
  true ARG, strong structure Ne=1e3 / T_split=5e3, T_admix=200, L=5 Mb; true 0.83±0.15 switches/Mb,
  inferred/true switch-density **ratio** as mean±std): **naive `argmax` over-fragments, tspaint worst**
  — ratio **1.95±0.58** (precision 0.66±0.15, median tract 371 kb); nearest_ref **1.53±0.28** (prec
  0.76); **RFMix native (.msp Viterbi) least but still 1.35±0.22** (prec 0.79). *No* method misses
  real switches (recall ~1.0; true tracts here all ≥100 kb, and 100–500 kb tracts recovered 100% by
  all — short-segment *sensitivity* is fine, the problem is false positives). Mechanism: at older
  admixture the posterior hovers near 0.5 in places and argmax flips there — the §7.3
  breakpoint-flicker surfacing in the *segment* output. **But tspaint's flips are exactly the
  low-confidence ones**, so a confidence **deadband** on the calibrated posterior
  (`hard_segments(deadband=c)`) removes them: c≈0.3–0.5 recovers the true distribution almost
  exactly and **tightly** — ratio **0.99±0.01**, precision **0.99±0.01**, recall **0.99±0.01**
  across the 6 seeds; c≥0.9 over-smooths. **Takeaway: for dating, do not argmax tspaint — threshold its
  posterior.** Done so, tspaint *beats* RFMix native on tract-length fidelity (ratio 0.99 vs 1.35): the
  calibrated soft posterior is a tunable precision/recall dial RFMix's hard CRF does not expose.
  Caveat: one regime (6 seeds). At *weak* signal real short-tract switches *also* go low-confidence,
  so the deadband then genuinely trades precision against recall — the regime where §7's uncertainty
  propagation (not a post-hoc threshold) would be the principled fix, and the subject of the
  `loopy-bp-ep` branch.
- **Sanity baseline (free).** Relate's own Neanderthal/Denisovan deep-branch
  labelling (Speidel et al. 2019, Fig. 4b–c) is a hand-rolled 2-color tip-down
  instance of this scheme. `tspaint` with hard clamps should reproduce their
  archaic-tract calls.
- **Leave-one-out / introgression detection.** On reference tips, the learned
  `w_i < 1` (global) and locally-dissenting posterior (per-locus) flag mislabels
  and foreign tracts. Direct application: CASK/DMD-style "is this LoF region
  introgressed or just mislabelled" — inherently a *where on the chromosome*
  question, which is why full local introgression-mapping (§2.3) is the right
  target, not a global scalar.
  - **[MEASURED — the introgression map is leave-one-out.]** A reference's *own* foreign
    tracts are hidden in the down-pass posterior by its (still-confident) tip emission; read
    them instead from the **outside message** (`output.loo_posterior_table`, the LOO marginal
    `PruneResult.loo`). Hard-clamping pins the down-pass (foreign recall 0); softening the
    suspect refs surfaces them (down-pass up to ~0.76, LOO up to ~0.92 with strong source
    anchoring + recent admixture; signal-bound — see the §6 "softening slightly-impure
    references" note). Tools: `experiments.impure_reference_experiment` /
    `impure_reference_sweep`, `sim.simulate_admixture_impure_refs`.
  - **[MEASURED — Plan A: three shipped workflows]** Now user-facing (`tspaint.introgression`;
    top-level `reference_qc` / `foreign_tracts` / `detect_ghost` + `Painting.introgression_map`),
    all on one new primitive `introgression.foreignness_track` (per locus: `loo`;
    `fit = max_s loo[s]`; nearest-ref coalescence `depth`, rank-normalised — they nest:
    QC ⊂ anonymous-foreign ⊂ ghost). (1) **Reference QC** (`reference_qc`, two-pass: hard-clamp
    LOO self-agreement → soften suspects keeping a clean anchor core) — impure refs rank
    least-credible (pure−impure credibility gap 0.18–0.26), their LOO maps recover the foreign
    tracts (~0.9), *assuming the clean refs are the majority*. (2) **Anonymous foreign tracts**
    (`foreign_tracts`) — label-dissent for refs / normalised fits-nothing for queries,
    source-agnostic. (3) **Ghost detection** (`detect_ghost` = low `fit` AND deep) — on an
    archaic-like ghost sim (`sim.simulate_admixture_with_ghost`, deep unsampled outgroup) ghost
    vs other segs separate cleanly (fit 0.51/0.81, depth 0.92/0.42); default recall 0.58 /
    precision 1.0, ghost burden 0.10 vs **no-ghost-control false positive 0.01** (~10×).
    Detection (*that* a tract is foreign-to-panel), not attribution; the depth threshold scales
    with ghost prevalence (tune vs your own no-ghost control). Plans:
    `plans/PLAN_A_workflows.md` (shipped) and `plans/PLAN_B_archaic_detector.md`
    (depth-as-a-generative-state — reference-free archaic). Tests:
    `tests/test_introgression.py`.
  - **[MEASURED — Plan B: reference-free archaic detector is a GO]** The "depth-as-a-generative-state"
    path is **built and beats the Plan A flag** (`tspaint.archaic`, top-level `detect_archaic`): a
    per-sample 2-state genome-axis HMM on nearest-modern-ref `log`-depth — modern emission anchored
    by the panel, **archaic emission learned with no archaic reference** but constrained ABOVE the
    panel's deepest coalescence (a high quantile of the reference depths). That floor — plus widening
    modern to cover its own deep (cross-source) tail and capping the archaic σ tight — is the §6
    identifiability fix; the naive version collapsed (no-ghost control → 0.92 archaic). Head-to-head
    vs `detect_ghost` on the archaic-like ghost sim (`experiments.archaic_detection_experiment`,
    reference-free): the HMM gives **per-locus recall 0.99–1.00 vs the flag's 0.38–0.54 at equal
    precision 1.0**, recovers the ghost **burden near-exactly** (vs the flag's fixed-quantile
    under-detection), keeps a **lower control false-positive** (~0.002 vs ~0.009), is **calibrated**,
    needs **no manual threshold**, and **learns μ_archaic = the true source-divergence depth**
    (recovers `log T_split_ABC` across 6k/10k/20k). Caveats: tested on the msprime true ARG (clean
    times) — raw depth is branch-length-calibration-sensitive (§6; the rank-variant / Relate test is
    outstanding), and the posterior is near-hard in this clean-signal regime. Plan:
    `plans/PLAN_B_archaic_detector.md`. Tests: `tests/test_archaic.py`.
  - **[UPDATE — two-task UI consolidation + rename].** The introgression tools are now organised by
    the two user tasks. **Task 1 (reference QC — *control* panel contamination):** `reference_qc`,
    whose result is **actionable** — `ReferenceQC.soft_refs()` (the suspect set to down-weight) and
    `.mask()` (per-reference foreign spans to drop) feed straight into `paint(..., soft_refs=...)`;
    plus `foreign_tracts`. **Task 2 (dedicated ghost / archaic *search* — accurate segments):** the
    depth-emission HMM is **renamed `detect_ghost`** (the obvious name for the accurate detector;
    `detect_archaic`/`ArchaicResult` kept as deprecated aliases of `detect_ghost`/`GhostResult`). It
    now (a) accepts a **SINGER ensemble** (`detect_ghost([G_1..G_M], labels)` — one pooled fit,
    per-member decode, averaged P(ghost), like `paint(ensemble)`) and (b) takes **`depth="rank"`**, a
    monotonic (calibration-invariant) depth transform that closes the raw-`log`-depth
    calibration-sensitivity caveat above — **[MEASURED]** ×7 node-time rescaling leaves rank-mode
    P(ghost) identical; only the Relate end-to-end test is still outstanding.
  - **[BUGFIX — rank floor overshoot; was silently non-functional].** The above "identical under
    rescaling" was originally **vacuous**: in the bounded `[0,1]` rank space the log-time rule
    `archaic_floor = q_ref + sd_m` overshoots the ceiling (q_ref→1, sd_m~0.25 ⇒ floor>1), parking the
    ghost emission *above every observation* so P(ghost)≈0 — **no detection** — in **both** the
    human-Ne (gf=0.02, Ne=10⁴) and the original gf=0.25/Ne=10³ regimes (the old test passed on a ratio
    of two ≈0 burdens; max P(ghost) was ~0.02). **Fix:** a rank-specific **bounded floor**
    `floor = q_ref + ½(1−q_ref)` with a `ghost_scale = ½(1−q_ref)` decoupled from `sd_m` (new
    `_baum_welch(ghost_scale=…)` arg, used for the ghost init and σ-cap); **`depth="time"` is
    byte-identical**. Rank now actually detects (recall 0→1.0, AUC≈0.97) *and* stays exactly invariant
    (×7 rescale ⇒ max|ΔP|=0); the test now asserts a confident call (max P>0.9) so the vacuous pass
    can't recur. **Caveat:** at human Ne rank's *fixed-threshold* precision is below `depth="time"`
    (the rank transform compresses the modern↔ghost separation, and a ref-anchored floor under-covers
    the deeper admixed-query loci in the separate-ADMIX-pop sim) — threshold higher or prefer SINGER
    calibrated times. Driver `experiments.archaic_detection_experiment`; tests `tests/test_archaic.py`.
    The former Plan-A
    `detect_ghost` *flag* (low fit AND deep) is **folded into `foreign_tracts(mode="fit",
    min_depth=)`** (the fast, deterministic, rank-depth alternative); `archaic_detection_experiment`
    still scores the HMM against it (recall 0.99–1.00 vs ~0.4–0.5). CLI: `tspaint qc` / `ghost`
    (ensemble, `--depth`) / `introgress` (`--min-depth`).

---

## 10. Prior art (as surveyed; see §8.3 for the remaining check)

The exact **mechanism** (generative ancestry-CTMC on an inferred tree sequence +
EM-learned Q and per-tip credibility + edge-blocked span-weighted sufficient
statistics → calibrated soft LAI) was **not found** as published (prior-art pass
§8.3 done; moderate-to-high confidence). The **edge-blocked span-weighted
sufficient statistics (§3.3) in particular appear nowhere** in the LAI/ARG
literature — lead the novelty there. But the *framing* ("tree-native LAI reaching
older admixture") now has direct neighbours; treat the first two below as the real
head-to-head, the rest as distinct objects.

> **[verify-DOI]** All 2026 bioRxiv entries below carry an unusual `10.64898/…`
> prefix as reported by the search pass — confirm on bioRxiv before citing. ARGMix
> author order and whether Pearson & Durbin outputs are calibrated also need PDF
> confirmation.

- **ARGMix** (Shanks, Bonet, Comajoan Cara & Ioannidis, 2026; bioRxiv [verify-DOI])
  — graph transformer doing LAI on the marginal coalescent trees of a Relate ARG,
  using ancient samples as references, **explicitly motivated by reaching older
  admixture where segments are too short** for RFMix/MOSAIC/FLARE. Same task and the
  same headline thesis as `tspaint`, but a black-box transformer — **no generative
  CTMC, no Felsenstein pruning, no EM-learned Q, no per-tip credibility, no
  calibrated posterior**. Chief threat to the novelty *framing*; primary head-to-head
  comparator.
- **Pearson & Durbin** (2023, "AncestralPaths"; bioRxiv 10.1101/2023.03.06.529121) —
  neural-network LAI on *inferred tree sequences* under complex/ancient histories
  ("path ancestry"); the most direct prior tree-native LAI. ML classifier on a
  deterministic population-structure model — **not** a generative mugration
  posterior, EM, or edge-blocked sufficient statistics; calibration not claimed.
  Must-cite head-to-head.
- **ARGformer** (Bonet, Shanks, Comajoan Cara, Abante & Ioannidis, 2026; Ioannidis
  "AI-sandbox" group — **not** Lewanski; bioRxiv [verify-DOI]) — ancestry from ARGs
  via *learned transformer embeddings + clustering/nearest-neighbour retrieval*,
  "without genotype matrices," supporting *unsupervised* ancestry. Black-box
  embedding, global structure/retrieval — **not** a generative mugration posterior,
  **not** calibrated per-locus painting. Comparator + cite (sibling work to ARGMix).
- **`sticcs` + topology weighting** (Martin, *Genetics* 2025, doi
  10.1093/genetics/iyaf181) — model-free genealogical inference + topology
  weighting; keywords "ancestry, introgression, tree sequence." But topology
  weighting summarizes *frequencies of topologies* among predefined groups (à la
  Twisst), not a CTMC-on-branches posterior; no soft per-haplotype assignment, no
  credibility learning. Neighbour, not duplicate.
- **Generalized pruning on a subsplit DAG** (Algorithms Mol Biol 2023, doi
  10.1186/s13015-023-00235-1) — extends Felsenstein pruning to a *multi-tree DAG*
  marginalizing over topologies via DP. Closest methodological precedent for
  "pruning that respects structure shared across many trees," but applied to
  Bayesian tree-space inference, not ancestry on a fixed ARG. Reviewers will know
  it — cite and distinguish.
- **Mugration / discrete phylogeography** (Lemey et al.) — the trait-CTMC-on-tree
  construction we reuse. **Relate Fig. 4b–c** is itself a hand-rolled 2-color
  instance.
- **Structured-coalescent-aware** methods (BASTA — De Maio et al., 2015; MASCOT —
  Müller, Rasmussen & Stadler, 2018) — correct the joint genealogy↔ancestry
  dependence the mugration approximation ignores (§6). The "on inferred ARG" variant
  flagged in §8.3 **exists**: **SCAR** (Guo, Carbone & Rasmussen, 2022, *PLOS Comput.
  Biol.*, doi 10.1371/journal.pcbi.1010422) runs the structured coalescent on an
  inferred ARG — but its deliverable is migration-rate/Ne/recombination demography
  and ancestral lineage locations, **not** per-haplotype painting with credibility.
  Closest "structured-coalescent-on-inferred-ARG" precedent; different object. Cite
  and distinguish.
- **Commercial LAI** (23andMe-style "ancestry painting" patents,
  e.g. US10755805 / US10572831) — haplotype-graph / copying-model / HMM
  constructions. Prior art for the *problem* and the incumbent paradigm we depart
  from, not for the *method*.

---

## 11. Repository layout (proposed)

```
tspaint/
  CLAUDE.md                      # this file (authoritative spec)
  README.md                      # short public-facing summary
  pyproject.toml                 # pixi/conda; deps: tskit, numpy, scipy, msprime, (phasic)
  src/tspaint/
    __init__.py
    model.py                     # Q, pi, emission, K-way generator-agnostic
    branch_stats.py              # branch_expected_stats (Van Loan; Phasic seam)
    pruning.py                   # per-root up/down pass, polytomy-safe (sib pointers)
    accumulate.py                # edge_diffs loop, span-weighted sufficient stats
    em.py                        # E-step orchestration + closed-form M-step
    io_relate.py                 # Relate->tskit (--compress) wrappers + checks
    output.py                    # per-haplotype per-position posterior; missing-info tagging
    bp/                          # DEFERRED loopy BP/EP (empty until §7.3 triggers)
  notebooks/
    00_persistence_check.ipynb   # §5.1 — RUN FIRST
    01_sim_admixture_truth.ipynb # msprime known-truth scenarios
    02_calibration_flicker.ipynb # §7.3 metrics; blocked-EM-vs-BP decision
  tests/
    test_branch_stats.py         # block-exp vs closed-form 2-state; small/large t
    test_pruning_polytomy.py     # arity, multi-root forest, isolated samples
    test_accumulate_nodoublecount.py  # span-weight == sum-over-trees on a toy ts
    test_em_degenerate.py        # Q->0, Q->inf, label-switch guards
```

### 11.1 Build order (so each step is checkable)
1. `00_persistence_check.ipynb` on a real `--compress` file — **go/no-go**.
2. `branch_stats.py` + `test_branch_stats.py` (block-exp vs analytic 2-state).
3. `pruning.py` + polytomy/forest/isolated tests on hand-built toy tree sequences.
4. `accumulate.py` + the no-double-count test: on a toy ts where a clade spans N
   trees, assert span-weighted stats == naive sum-over-trees stats.
5. `em.py`: hard-clamp-only first (no `w_i`); reproduce Relate archaic calls
   (§9 sanity baseline).
6. Add soft credibility `w_i` + anchor set + Beta prior; degenerate-case tests.
7. `01`/`02` notebooks: simulated-truth accuracy, calibration, flicker (§7.3).
8. Only if §7.3 triggers: implement `bp/`.

---

## 12. Key invariants to never violate (quick reference)

- Accumulate **per edge, span-weighted**, banked once on edge entry. Never sum
  per-(tree×branch).
- Prune **per root**; a marginal tree may be a **forest**.
- **Skip root branches** for dwell/jumps (length 0 by definition); root via `π`.
- **Polytomy-safe** traversal via `left_child`/`right_sib`; product over all
  children.
- **Isolated span = missing info ≠ uncertain**; tag separately in output.
- Keep a **hard-clamped anchor set**; never let the whole panel float.
- Keep `branch_expected_stats` **generator-agnostic** (2-state today, K-way by
  generator swap). It is the **Phasic seam**.
- Input **must** be `--compress`d (or tsinfer-native); verify persistence
  histogram first.

---

## 13. References

- Speidel, L., Forest, M., Shi, S. & Myers, S. R. (2019). A method for
  genome-wide genealogy estimation for thousands of samples. *Nature Genetics*
  51, 1321–1329. (Relate; equivalent-branch identification = Supp. §4.1;
  panmictic-prior TMRCA bias = Supp. Fig. 3c; archaic deep-branch labelling =
  Fig. 4b–c; coalescence-order selection test = main text / Supp. Note.)
- Relate documentation & `relate_lib` (`Convert --mode ConvertToTreeSequence
  --compress`): https://myersgroup.github.io/relate/ ;
  https://github.com/leospeidel/relate_lib ; `relater` R package:
  https://github.com/leospeidel/relater
- tskit data model (edges, quintuply-linked encoding, virtual root, multiple
  roots, isolated nodes, root branch length = 0):
  https://tskit.dev/tskit/docs/stable/data-model.html ;
  `edge_diffs` / Python API: https://tskit.dev/tskit/docs/stable/python-api.html
- Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum
  likelihood approach. *J. Mol. Evol.* 17, 368–376. (Pruning.)
- Lemey, P., Rambaut, A., Drummond, A. J. & Suchard, M. A. (2009). Bayesian
  phylogeography finds its roots. *PLoS Comput. Biol.* 5, e1000520. (Mugration /
  discrete trait CTMC on trees.)
- Van Loan, C. F. (1978). Computing integrals involving the matrix exponential.
  *IEEE Trans. Automat. Contr.* 23, 395–404. (Block-triangular exponential for
  CTMC reward integrals — `branch_expected_stats`.)
- Hobolth, A. & Jensen, J. L. (2011). Summary statistics for endpoint-conditioned
  continuous-time Markov chains. *J. Appl. Probab.* 48(4), 911–924. (Expected dwell
  times & jump counts conditioned on branch endpoints; stochastic mapping; EM for
  CTMC on trees. **Verified.**)
- Tataru, P. & Hobolth, A. (2011). Comparison of methods for calculating conditional
  expectations of sufficient statistics for continuous time Markov chains. *BMC
  Bioinformatics* 12, 465, doi 10.1186/1471-2105-12-465. (EXPM/Van Loan vs.
  eigendecomposition vs. uniformization for `branch_expected_stats`; the numerics
  fallback for §8.7 / the Phasic seam.)
- Wong, Y., Ignatieva, A., Koskela, J., Gorjanc, G., Wohns, A. W. & Kelleher, J.
  (2024). A general and efficient representation of ancestral recombination
  graphs. *Genetics* 228(1), iyae100. (ARG/tree-sequence representation.)
- Brandt, D. Y. C., Huber, C. D., Chiang, C. W. K. & Ortega-Del Vecchyo, D.
  (2024). The Promise of Inferring the Past Using the ARG. *Genome Biol. Evol.*
  16(2), evae005.
- Lewanski, A. L., Grundler, M. C. & Bradburd, G. S. (2024). The era of the ARG.
  / Nielsen et al. (2025). Inference and applications of ancestral recombination
  graphs. *Nat. Rev. Genet.* 26(1), 47–58, doi 10.1038/s41576-024-00772-4.
- Martin, S. H. (2025). A model-free method for genealogical inference without
  phasing (`sticcs`) and topology weighting. *Genetics*, doi
  10.1093/genetics/iyaf181. (Nearest tree-sequence ancestry/introgression
  neighbour.)
- Bonet, D., Shanks, C., Comajoan Cara, M., Abante, J. & Ioannidis, A. G. (2026).
  ARGformer: learning on ancestral recombination graphs with transformers. bioRxiv
  [verify-DOI]. (Ioannidis "AI-sandbox" group — **not** Lewanski. ARG embeddings for
  ancestry via retrieval/clustering; embedding, not generative.)
- Shanks, C., Bonet, D., Comajoan Cara, M. & Ioannidis, A. G. (2026). ARGMix: graph
  transformer for ancient ancestry inference. bioRxiv [verify-DOI]. (Tree-native LAI
  on Relate ARGs; primary head-to-head; verify author order on PDF.)
- Pearson, A. & Durbin, R. (2023). Local ancestry inference for complex population
  histories ("AncestralPaths"). bioRxiv, doi 10.1101/2023.03.06.529121. (NN LAI on
  inferred tree sequences; most direct prior tree-native LAI.)
- Guo, F., Carbone, I. & Rasmussen, D. A. (2022). Recombination-aware phylogeographic
  inference using the structured coalescent with ancestral recombination (SCAR).
  *PLOS Comput. Biol.* 18(8), e1010422, doi 10.1371/journal.pcbi.1010422. (Structured
  coalescent on inferred ARGs; migration demography, not painting — §8.3, §10.)
- Generalized phylogenetic pruning on a subsplit DAG (2023). *Algorithms Mol.
  Biol.* doi 10.1186/s13015-023-00235-1. (Pruning across shared multi-tree
  structure.)
- Müller, N. F., Rasmussen, D. & Stadler, T. — MASCOT; De Maio, N. et al. — BASTA.
  (Structured-coalescent-aware approaches; §6, §8.3.)

> Citation hygiene: Hobolth & Jensen (2011) now **verified**; ARGformer
> re-attributed to the Ioannidis "AI-sandbox" group (was wrongly "Lewanski").
> **Still unverified — do not propagate blind:** all 2026 bioRxiv DOIs (unusual
> `10.64898/…` prefix reported; confirm against bioRxiv), ARGMix author order, and
> whether Pearson & Durbin / Medina Tretmanis consume tree sequences vs. genotype
> matrices and whether their outputs are calibrated. Inline citations in write-ups
> use reference-style links (Author–Year, ≤2 authors then "et al.") with verified
> source–claim links.
