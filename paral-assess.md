# Parallelization — tspaint painting (implemented)

Where painting uses multiple CPUs, and how. **Now built** (`tspaint.parallel`, the `n_jobs` knob,
and the `tspaint` CLI); this file documents the design and the bit-exactness contract it honours.

## Verdict

The expensive part of painting is an **exact map-reduce over thousands of independent marginal
trees**, repeated each EM iteration — near-linear scaling with cores. It needs **process-based**
parallelism (the hot loop is Python-level tree iteration that holds the GIL). There are **two
layers**, the same decomposition at two granularities:

1. **In-process** (`--cores` / `n_jobs`, `tspaint.parallel`): a `ProcessPoolExecutor` over
   tree-index ranges (and ensemble members) **within one job** — saturates a node.
2. **GWF / cluster** (the `tspaint` CLI subcommands): each job is a `tspaint <sub>` call with
   file in/out; the cluster fans out over **SINGER windows** and **ensemble members** — the same
   chunk boundaries, across nodes.

## Parallel axes (coarse → fine), with current `file:line`

| Layer | Where | Parallel? | Grain |
|---|---|---|---|
| Experiment sweeps | `experiments.age_sweep` (`experiments.py:252`), `scaling_sweep` (`:290`) | ✅ trivially | coarsest |
| Ensemble members | `em.fit` E-step (`em.py:284` `estep_parallel`); SINGER windows × members (GWF) | ✅ | M-way / W×M |
| **Genome E-step** | `accumulate_sufficient_statistics` loop (`accumulate.py:114`), ×4–12 EM iters (`em.py:295`) | ✅ **exactly** | dominant per-painting cost |
| Final posterior | `output._paint_tracks` tree loop (`output.py:67`), per-sample | ✅ | one pass |
| Dating E-step | `dating/estep.py:123`, `:270` | ✅ (same shape) | not yet wired to `n_jobs` |
| Archaic Baum–Welch | `archaic.py:247` (per sample, pooled), ×iters (`:240`) | ✅ (over samples) | not yet wired |
| EM outer loop | `em.py:295` | ❌ sequential | N+1 needs N's (Q,π,w); cheap |
| Up/down pruning | `pruning.py` postorder/preorder | ❌ sequential | inherent to message passing |

No parallelism existed before; the only deps were `tskit/msprime/numpy/scipy` (now `+click`).
`em.fit` and `api.paint` take `n_jobs` (default 1); `tspaint.parallel` holds the engine.

## The mechanic — exactly-once banking (the correctness core)

`accumulate_sufficient_statistics` banks each edge **once, at its entry tree** (`edges_in`),
weighted by its own span, and counts each tree's loglik / each root once. **Partition the genome
into contiguous tree-index ranges**: each `edges_in` event — hence each edge's span-weighted
contribution — and each tree/root falls in exactly one range. So a partition reproduces the
whole-genome statistics, summed. **There is no "skip boundary-spanning edges" rule** (an earlier
draft of this file said so — wrong): an edge enters exactly once, period.

Implementation: a single optional `tree_range=(lo, hi)` gate (continue/break) on
`accumulate_sufficient_statistics` and `output._paint_tracks` leaves the per-tree arithmetic
**unchanged**; that invariance is what makes the split bit-exact. `edge_diffs()` is forward-only
(no seek), and its `edges_in` is incremental, so a worker iterates the diff stream from 0 and
skips only the expensive `prune_tree` for trees `< lo` (`parallel._accumulate_range`). Workers get
the ts **by path** (`tskit.load` once, cached in `parallel._TS_CACHE`) plus the small
`(Q, π, w, labels)`; emissions are rebuilt in-worker.

## The float-associativity contract (be honest)

IEEE `+` is not associative, so P>1 cannot be byte-identical to the serial single loop. What holds
(`tests/test_parallel.py`):

- `n_jobs == 1` → one chunk → **byte-identical** to the serial loop (`np.array_equal`) — the
  regression guard.
- `n_jobs == P` → **byte-identical to the same chunk partition reduced serially** (`add_suffstats`
  in chunk order) — depends only on the partition + the parent's in-order fold, not on process
  placement.
- vs the serial single loop, `P > 1` → a few **ULP** (`np.allclose`, `rtol≈1e-12`), reduction
  order only.
- `exact=True` → runs **serially** (byte-identical to the serial loop). A *parallel*-exact mode
  would need per-tree IPC and an arithmetic refactor that would itself perturb the serial bits, so
  we stop at serial — the serial path already gives the guarantee.

**Painting** (`posterior_table_parallel`) is **exactly** equal to serial for any `P`: each
segment's posterior comes from its own tree's pruning, independent of the chunking, so stitching
the per-range tracks and re-merging the seams reproduces the serial segmentation bit-for-bit.

## Overheads & knobs (all manageable)

- **EM is sequential across iterations**, but the per-iteration M-step (`O(#edges)`) is negligible
  vs. the E-step → Amdahl-favorable. `em.fit` builds **one persistent pool** and reuses it across
  iterations, passing only `(Q, π, w)` each time (`em.py` `ExitStack`); the worker ts-cache avoids
  reloading.
- **Cores**: `parallel.resolve_cores()` → explicit `--cores/-j` else `$SLURM_CPUS_PER_TASK` else
  `$SLURM_JOB_CPUS_PER_NODE` (compact `N(xM)` form parsed) else `$TSPAINT_CORES` else **1**
  (serial — the safe library default).
- **Shipping the ts**: workers re-`tskit.load` by path. The CLI passes the known `.trees` path; the
  in-process API has no recoverable path for a loaded ts, so it dumps each member **once per fit**
  to a temp `.trees` (`parallel.as_path`) and reuses it across iterations.
- **BLAS oversubscription**: `make_pool` pins `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1` in
  the **parent** before pool creation (children inherit at `spawn`, before importing numpy). The
  parent's numpy is already initialised, so this does not throttle it. Irrelevant at K=2 (tiny
  matrices); matters for large-K dating.
- **Start method**: Linux (the SLURM cluster) defaults to `fork` → `n_jobs>1` works from any
  context, including the CLI. macOS defaults to `spawn` → a script that spawns workers needs the
  standard `if __name__ == "__main__":` guard (the CLI entry point and pytest both have it; a bare
  `python -` / notebook cell does not).
- **Load balancing**: `genome_chunks` splits into equal tree-count ranges. Edge-count balancing is
  a possible refinement (trees vary in size) but unnecessary while they are similar.

## GWF deployment (the cluster layer)

Each job is a `tspaint <sub>` call; no glue Python. The ensemble pipeline:

```
trees singer-window  × W windows   ──► arg_w{w}_{nodes,branches,muts}_<i>.txt
trees merge-arg       × M members   ──► member_<i>.trees           (stitch windows per member)
fit   member_*.trees --labels       ──► params.npz                 (one pooled fit; -j cores)
paint member_<i>.trees --params     ──► member_<i>.painting.npz     (× M, independent; -j cores)
merge member_*.painting.npz         ──► merged.painting.npz         (marginalise the ARG)
date | qc | introgress | ghost | archaic member_<i>.trees ──► *.npz
```

`fit` is the one coupling point (it pools sufficient statistics across the whole ensemble — the
M-step is scale-invariant), so it is a single job that parallelises *internally* over members ×
tree-ranges; painting then fans out per member. See `examples/workflow.py` for a runnable gwf
`Workflow`.

## Not yet done (follow-on)

- Thread `n_jobs` into `dating.fit_rate_through_time` / `detect_archaic` / `introgression.*` (same
  independent-loop shape; `dating/estep.py:123,270`, `archaic.py:247`). Lower priority — paint/fit
  dominate the cost.
- Edge-count chunk balancing in `genome_chunks` (vs equal tree counts).
- Optional seek-based worker start (precompute per-tree `edges_in` on the parent, ship slices,
  `Tree.seek_index`) to avoid the iterate-and-skip of trees `< lo`. Deferred behind the bit-exact
  test; the skipped iteration is cheap relative to pruning.
