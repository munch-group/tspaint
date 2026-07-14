# external/ — third-party comparators

The tools tspaint benchmarks against are not on conda/PyPI — each is a GitHub repo cloned and built
into its own environment. The tool repos are **not** checked in; what *is* committed is the recipe to
provision them:

- [`tools.ini`](tools.ini) — the pinned manifest (repo + commit + steps per tool);
- [`envs/<tool>/pixi.toml`](envs/) — the per-tool pixi env recipe (dependencies + build tasks).
  These are **our own glue** — they are *not* in the upstream repos, so a fresh clone won't have
  them; `install` copies the tracked recipe into each clone before `pixi install`.

Everything else under `external/` (the clones themselves) is gitignored.

## Provision

```bash
tspaint benchmark install                 # clone + build everything in the manifest
tspaint benchmark install mosaic flare    # just these
tspaint benchmark install --dry-run       # print the plan without doing anything
tspaint benchmark status                  # what is installed, and where
```

`install` clones each tool at its pinned commit, copies in the tracked `envs/<tool>/pixi.toml`, runs
`pixi install`, runs the env's build task(s), and verifies the `check` path. (`setup` is a hidden
alias kept for existing scripts.)

## What is here, and why

**Painters — have a `tspaint benchmark <tool>` runner:**

| Tool | Kind | Why it is here |
|---|---|---|
| `rfmix` | window / random-forest + CRF | The field standard. *Not in the manifest* — bioconda, via the `compare` pixi feature (`pixi install -e compare`). |
| `gnomix` | window / XGBoost | Ioannidis-lab ML incumbent. |
| `salai` | window / cosine matching + conv smoother | Species-agnostic, pre-trained, no genetic map. |
| `recombmix` | site / graph collapse | Best on recent, large-panel, intra-continental admixture. |
| `mosaic` | **R** — nested HMM, Li–Stephens copying | **The strongest genotype-based comparator**: beats RFMix/ELAI/LAMP-LD *even when they are given the panel↔ancestry correspondence and it is not*, and best-in-class on small panels. Also the direct ancestor of GhostBuster, and the origin of the `E[r²]` and `Rst` diagnostics. |
| `flare` | **JVM** — Li–Stephens HMM | The maintained descendant of HAPMIX; one of the three methods that beat Recomb-Mix past 200 generations (our target regime). HAPMIX itself is 2009 C and is not worth reviving. |
| `loter` | graph / dynamic programming | No biological parameters at all — the closest competitor to tspaint's own "no genetic map / non-model species" pitch. |
| `ghostbuster` | genealogy / EM mixture over coalescence rates | **The head-to-head.** It consumes a **tskit tree sequence** — the same object tspaint paints — so it is the one comparator with no VCF bridge and no front-end confound: same ARG in, different model. |

**Not painters — install only (no runner):**

| Tool | Why it is here |
|---|---|
| `clues2` | Downstream ARG-based selection inference. Needed for the uncertainty-propagation experiment: ARGMix inferred an ancestry-stratified selection coefficient from *hard, uncalibrated* ancestry calls with no error bar. Ships `SingerToCLUES.py`, so it reads the same SINGER ARGs we do. |
| `argformer` | Embeddings + nearest-neighbour retrieval. Needed for exactly one experiment: it localises Denisovan segments *by retrieving Denisovan-labelled neighbours*, so withhold the archaic reference and it has nothing left — where `detect_ghost` needs none. **linux-64 only** (see below). |

**ARG front ends are *not* comparators** — they are `tspaint install {relate,singer,argweaver}`.

`argweaver` builds **`mdrasmus/argweaver`**, not the Siepel fork. The Siepel fork *is* ARGweaver-D —
demography-aware sampling with migration bands (`--pop-tree-file`, `--start-mig`) — which would give
us ARGs sampled under a structured demography (no panmictic branch-length prior) and a reference
implementation of generative introgression detection to score `detect_ghost` against. **But its
`.smc` output breaks our converter**: `read_argweaver_smc` keys node identity on `(name, age)`, and
ARGweaver-D reuses a name+age in two different topological positions, so the edges accumulated across
local trees contain a cycle and no node-time assignment can satisfy `time[parent] > time[child]`.
Measured on one 18-sample run: the -D binary produced a cyclic sample that hung the converter at 100%
CPU, where `mdrasmus` was cycle-free. Enabling it needs a node identity that survives an SPR
(descendant-set keying, as Relate's `--compress` does). Opt in at your own risk with
`TSPAINT_ARGWEAVER_REPO` / `_COMMIT` — see `src/tspaint/install.py`.

## Platform notes

Recipes target `osx-arm64` and `linux-64`. Three upstreams needed work to build on Apple Silicon;
all of it is documented in the relevant `envs/<tool>/pixi.toml`:

- **Loter** hardcodes `-msse2` (x86-only) and relies on a pre-C++17 compiler default (its vendored
  Eigen uses `std::binder1st`/`binder2nd`, removed in C++17). We patch the flag to be
  architecture-conditional and pin `-std=c++14`.
- **GhostBuster**'s `requirements.txt` pins `msprime==1.3.0`, which pip builds from sdist on
  osx-arm64 into an extension that cannot resolve GSL at load time. Everything comes from
  conda-forge instead. (Its pinned `pysnptools` is never imported — a stale requirement.)
- **MOSAIC** needs `LaF`, which conda-forge has no osx-arm64 build for; it is fetched from CRAN into
  a clone-local `Rlib/`.
- **ARGformer is linux-64 only.** Upstream's env pins `pytorch-cuda` + `triton`, and its MosaicML
  `composer` dependency does not resolve in a conda env on macOS. It declares `platforms = linux-64`
  in the manifest, so `benchmark install` **skips** it elsewhere with a one-line reason — it does not
  clone the repo and it does not mark the run failed. Provision it on a Linux cluster.

A tool that cannot be built on a given platform must say so with `platforms` (+ an optional `note`)
in `tools.ini`, **not** with a shell guard inside its `build` step: a guard runs *after* the clone,
so it reports `FAILED` and leaves a useless checkout behind, making a documented upstream limitation
look like a broken install. `tests/test_benchmark_setup.py` enforces this for ARGformer.

## Locations / overrides

Clones default to `external/<dir>`. Override the root with `TSPAINT_TOOLS_DIR`, or a single tool with
`TSPAINT_GNOMIX_DIR` / `TSPAINT_SALAI_DIR` / `TSPAINT_RECOMBMIX` (the binary path). To reuse an
existing clone instead of re-cloning, symlink it: `ln -s ~/gnomix external/gnomix`.

## A note on `check`

Every manifest entry's `check` must name an artifact produced by the tool's *last* provisioning step
— a compiled binary (`flare.jar`, `RecombMix_v0.7`), an install tree (`Rlib/MOSAIC/DESCRIPTION`), or
a `.deps-ok` marker written after an import smoke-test. It must **not** be a repo file: those exist
straight after `git clone`, so a clone whose build or pip step failed would still report as
installed. `tests/test_benchmark_setup.py` enforces this.
