# Why `painting.posteriors[ref]` shows a reference's *own* introgression

When you paint a **reference** haplotype (include it in the focal set:
`paint(ts, labels, queries=queries + [ref], mask=mask)`), its per-position posterior can reveal the
reference's *own* foreign tracts — a contaminated "A" reference paints **B** over the span it actually
inherited from B. This note explains the mechanism, why a hard clamp hides it, and why **fragment
masking** (`paint(..., mask=…)`) reveals it. See `introgression_map.py` for the runnable example that
produced the numbers below.

## The down-pass identity

Ancestry is inferred by Felsenstein pruning on each marginal tree. The **down-pass** posterior for any
tip is (see `src/tspaint/pruning.py`, `bc = U[c] * Lnorm[c]`):

```
γ_tip  =  normalize( emission_tip  ·  U_tip )          (elementwise product over states)
```

Two factors multiply:

- **`emission_tip`** — the tip's *own* label evidence (§2.2). A hard-clamped reference with label A has
  the one-hot emission `[1, 0]`; a query (or a masked span) has the flat query emission
  `query_emission(π) = π`.
- **`U_tip`** — the **outside message**: what the *rest of the tree* says about the tip, from its
  genealogical neighbours, **independent of the tip's own label**. This is exactly the leave-one-out
  (LOO) quantity `loo_posterior_table` reads.

The introgression signal lives entirely in **`U_tip`**: over a B-introgressed tract the reference
coalesces (shallowly) with **B** references, so `U_tip` points to B — regardless of the label the
reference carries.

## Hard clamp hides it; masking reveals it

- **Hard clamp** (`emission = [1, 0]`): the product is `[1·U_A, 0·U_B] = [U_A, 0]`. The `0` in the
  emission **annihilates the B component of the outside message**, so `γ` is pinned to `[1, 0]` no
  matter what the tree says. The reference's own certainty overwrites the genealogy — the introgression
  is invisible in the standard painting. (Down-pass foreign-tract recall is **0 by construction**.)

- **Fragment masking** (`emission → π`, flat): the product is `π · U ∝ U`, so
  `γ_tip = U_tip`. The down-pass posterior *becomes* the outside message — the tree's verdict flows
  straight through, and the B tract appears.

Masking does not *compute* anything new: the introgression was always in `U`. Masking removes the tip's
self-certainty so that existing signal reaches the standard `posteriors[ref]` output.

## Measured (from `introgression_map.py`)

An impure `SOURCE_A` reference (node 39, label **A**) carrying a real **~2 Mb `SOURCE_B`** tract, mean
`P(A)` over that true-B tract:

| quantity | `P(A)` over the true-B tract | reading |
|---|---:|---|
| outside message `U` | **0.068** | the tree's verdict — coalesces with **B** here |
| hard down-pass `γ` | **1.000** | `[U_A, 0·U_B]` → pinned to A, introgression **hidden** |
| masked down-pass `γ` | **0.085** | `emission = π` → `γ ∝ U` → matches `U`, introgression **revealed** |

`masked γ == U` (to ~2×10⁻²) — the identity above, in numbers.

## Relation to the LOO introgression map

Because `masked γ = U`, fragment masking makes the **standard down-pass painting** agree with the
**leave-one-out** introgression map (`loo_posterior_table` / `Painting.introgression_map`,
`ReferenceQC`). The difference is ergonomic, not statistical: masking annotates a reference's foreign
tracts *in the ordinary `posteriors[ref]` object* (so `plot`, `segments`, `hard_segments`, `posterior_at`
all apply), and it does so for the whole panel in one coherent down-pass with the contamination
excluded — rather than as a separate per-reference leave-one-out computation.

## Seeing it — the reference-inclusive plot

`Painting.plot` is reference-aware. For any painted row that is a labelled reference (in
`painting.labels`) it labels the row by its nominal ancestry — `ref 39 (A)` instead of `hapl. i` — and,
when the painting was produced with a `mask`, **hatches** the reference's masked (unlabelled) spans over
its soft band. So `ref 39 (A)` renders **blue (P(A)≈0 = B)** across its introgressed tract — nominally A,
painting B — with that span hatched to show where it was un-anchored. Pass `refs=False` /
`mark_masked=False` to suppress either annotation.
