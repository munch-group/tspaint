# external/ — third-party benchmark comparators

The local-ancestry tools tspaint benchmarks against (gnomix, SALAI-Net, Recomb-Mix) are not on
conda/PyPI — each is a GitHub repo cloned and built into its own environment. The tool repos are
**not** checked in; what *is* committed is the recipe to provision them:

- [`tools.ini`](tools.ini) — the pinned manifest (repo + commit + steps per tool);
- [`envs/<tool>/pixi.toml`](envs/) — the per-tool pixi env recipe (dependencies + build tasks).
  These are **our own glue** — they are *not* in the upstream repos, so a fresh clone won't have
  them; `setup` copies the tracked recipe into each clone before `pixi install`.

Everything else under `external/` (the clones themselves) is gitignored.

## Provision

```bash
tspaint benchmark setup            # clone + build all three into external/<dir>
tspaint benchmark setup --dry-run  # print the plan without doing anything
tspaint benchmark setup --tools salai,recombmix
tspaint benchmark status           # show which tools are installed and where
```

`setup` clones each tool at the commit pinned in `tools.ini`, copies in its tracked
`envs/<tool>/pixi.toml`, runs `pixi install`, runs the env's build task(s) (e.g. Recomb-Mix's
`build` — the boost/openmp compile), extracts shipped models (SALAI-Net), and verifies the
`check` path. RFMix is not here — it comes from bioconda via the `compare` pixi feature
(`pixi install -e compare`).

The env recipes are osx-arm64 in this checkout (they were authored on a Mac); for Linux/CI add
`linux-64` to `platforms` in `envs/<tool>/pixi.toml`. To keep them in sync with a working clone,
just `cp ~/<tool>/pixi.toml external/envs/<tool>/pixi.toml`.

## Locations / overrides

Clones default to `external/<dir>`. Override the root with `TSPAINT_TOOLS_DIR`, or a single tool
with `TSPAINT_GNOMIX_DIR` / `TSPAINT_SALAI_DIR` / `TSPAINT_RECOMBMIX` (the binary path). To reuse
an existing clone instead of re-cloning, symlink it, e.g. `ln -s ~/gnomix external/gnomix`.
