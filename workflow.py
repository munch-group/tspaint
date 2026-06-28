"""GWF workflow — benchmark tspaint against the external LAI tools (CLAUDE.md §9).

Runs a head-to-head across a coarse demographic grid (human-like ``Ne = 10000``) and two regimes:

* **isolated** — the two source populations are reproductively isolated between split and
  admixture (the clean two-source pulse);
* **migration** — low-level symmetric gene flow between the sources in that interval, so they are
  less differentiated and ancestry is harder to call.

For each ``(model, T_split, T_admix, seed)`` it simulates an admixture with **known local
ancestry**, then paints the same data with every painter and scores three metric families:

1. **overall proportions** — estimated vs true global ancestry fraction (bias);
2. **fragmentation** — inferred vs true ancestry-switch density (tract-length fidelity);
3. **accuracy as a function of true segment size** — per-base accuracy binned by true tract length
   (the headline: which methods hold up as tracts shorten with older admixture).

Painters (all scored against the matching truth, so the aggregate metrics are comparable):

* ``tspaint_true``  — tspaint on the **true** ARG (upper bound), scored vs the node-id truth;
* ``tspaint``       — tspaint on a **tsinfer** ARG built from the VCFs (the fair, realistic case);
* ``rfmix`` / ``gnomix`` / ``salai`` / ``recombmix`` — the external tools, run from the VCFs.

The last six all consume the *identical* query/reference VCFs that ``benchmark export-vcf`` writes
from the simulation, and are scored against its hap-index truth.

Running it
----------
This is a plain gwf workflow (see ``gwf.md``). From the repo root::

    gwf config set backend slurm        # one-time; develop locally with `-b local`
    gwf run                             # submit the whole grid
    gwf status -f summary

Outputs land under ``results/`` (one dir per scenario) and the final tidy CSVs at
``results/summary/{summary_scalar.csv, summary_by_size.csv}`` — ready to plot the three families
across the grid. Edit the grid / sizes / resources in the **CONFIG** block below.

Prerequisites: the external tools must be provisioned (``tspaint benchmark setup`` + the ``compare``
pixi env for RFMix). ``PROVISION_TOOLS = True`` makes that a gwf target the tool jobs depend on; set
it False and provision manually (then ``gwf touch SetupTools``) if you prefer.
"""
import itertools
import os

from gwf import Workflow
from gwf.executors import Pixi

# ============================ CONFIG (edit me) ============================
NE = 10_000                                   # human-like effective size (all populations)
MODELS = {"isolated": 0.0, "migration": 1e-4}  # name -> A<->B migration rate between split & admixture
T_SPLIT = [2_000, 5_000, 10_000]              # source-split times (generations ago)
T_ADMIX = [100, 500, 1_000]                   # admixture-pulse times (generations ago); must be < T_split
SEEDS = [1, 2]                                # replicate simulations per grid point
F_A = 0.5                                     # admixture fraction from source A

N_ADMIX = 20                                  # admixed (query) individuals
N_REF = 20                                    # reference individuals per source
LENGTH = 10_000_000                           # simulated sequence length (bp)
RECOMB = 1e-8                                 # recombination rate (per bp per generation)
MU = 1.25e-8                                  # mutation rate (per bp per generation)
DEADBAND = 0.4                                # confidence dead-band for the fragmentation metric

RESULTS = "results"                           # output root (relative to this file)
PROVISION_TOOLS = True                        # build a SetupTools target the external tools depend on
PAINT_CORES = 4                               # cores per painting job (tspaint uses them via -j)

# Painters: the two tspaint variants plus the external comparators.
EXTERNAL = ["rfmix", "gnomix", "salai", "recombmix"]
PAINTERS = ["tspaint_true", "tspaint"] + EXTERNAL
# =========================================================================

gwf = Workflow(defaults={"cores": 1, "memory": "4g", "walltime": "01:00:00"}, executor=Pixi())


def _name(model, ts, ta, seed):
    """A target-name-safe scenario id (letters/digits/underscores only)."""
    return f"{model}_Ts{ts}_Ta{ta}_s{seed}"


def _dir(model, ts, ta, seed):
    return os.path.join(RESULTS, model, f"Ts{ts}_Ta{ta}", f"s{seed}")


# --- one-time provisioning of the external tools -------------------------------------------
setup_sentinel = os.path.join("external", ".provisioned")
if PROVISION_TOOLS:
    gwf.target("SetupTools", inputs=[], outputs=[setup_sentinel],
               cores=4, memory="8g", walltime="08:00:00") << f"""
pixi install -e compare
tspaint benchmark setup
touch {setup_sentinel}
"""
tool_dep = [setup_sentinel] if PROVISION_TOOLS else []

# --- the grid -----------------------------------------------------------------------------
metrics_jsons = []
for model, ts, ta, seed in itertools.product(MODELS, T_SPLIT, T_ADMIX, SEEDS):
    if ta >= ts:                                  # need T_admix < census < T_split
        continue
    mig = MODELS[model]
    d = _dir(model, ts, ta, seed)
    sid = _name(model, ts, ta, seed)

    trees = f"{d}/sim.trees"
    labels = f"{d}/labels.json"
    truth_nodes = f"{d}/truth_nodes.npz"          # node-id truth (for tspaint_true)
    qvcf, rvcf = f"{d}/query.vcf", f"{d}/reference.vcf"
    smap, truth_hap = f"{d}/sample_map.tsv", f"{d}/truth.npz"   # hap-index truth (for the rest)

    # 1) simulate the admixture with known truth
    gwf.target(f"Sim_{sid}", inputs=[], outputs=[trees, labels, truth_nodes],
               memory="8g", walltime="01:00:00") << f"""
mkdir -p {d}
tspaint simulate --n-admix {N_ADMIX} --n-ref {N_REF} --length {LENGTH} --recomb-rate {RECOMB} \
    --ploidy 2 --seed {seed} --ne {NE} --t-admix {ta} --t-split {ts} --f-a {F_A} \
    --migration-rate {mig} --mutate --mu {MU} \
    -o {trees} --labels-out {labels} --truth {truth_nodes}
"""

    # 2) export query/reference VCFs + sample map + hap-index truth
    gwf.target(f"Export_{sid}", inputs=[trees, labels],
               outputs=[qvcf, rvcf, smap, truth_hap], memory="8g") << f"""
tspaint benchmark export-vcf {trees} --labels {labels} -o {d}
"""

    # 3) paint + score with every painter
    for painter in PAINTERS:
        out_npz = f"{d}/{painter}.npz"
        mjson = f"{d}/{painter}.metrics.json"
        if painter == "tspaint_true":
            # tspaint on the TRUE ARG (upper bound), scored vs the node-id truth; -j for the
            # many marginal trees of the full ARG.
            paint_inputs, truth = [trees, labels], truth_nodes
            paint_cmd = f"tspaint paint {trees} --labels {labels} -j {PAINT_CORES} -o {out_npz}"
            paint_opts = {"cores": PAINT_CORES, "memory": "16g", "walltime": "04:00:00"}
        elif painter == "tspaint":
            # tspaint on a tsinfer ARG built from the VCFs (the fair, realistic case).
            paint_inputs, truth = [qvcf, rvcf, smap], truth_hap
            paint_cmd = (f"tspaint benchmark tspaint --vcf {qvcf} --ref-vcf {rvcf} "
                         f"--sample-map {smap} -o {out_npz}")
            paint_opts = {"cores": PAINT_CORES, "memory": "16g", "walltime": "06:00:00"}
        else:                                      # external comparator (needs the tools provisioned)
            paint_inputs, truth = [qvcf, rvcf, smap] + tool_dep, truth_hap
            gens = f" --generations {ta}" if painter == "rfmix" else ""
            paint_cmd = (f"tspaint benchmark {painter} --vcf {qvcf} --ref-vcf {rvcf} "
                         f"--sample-map {smap}{gens} -o {out_npz}")
            paint_opts = {"cores": PAINT_CORES, "memory": "16g", "walltime": "08:00:00"}

        gwf.target(f"Paint_{painter}_{sid}", inputs=paint_inputs, outputs=[out_npz],
                   **paint_opts) << paint_cmd

        meta = (f"--meta model={model} --meta t_split={ts} --meta t_admix={ta} "
                f"--meta seed={seed} --meta ne={NE} --meta migration={mig}")
        gwf.target(f"Metrics_{painter}_{sid}", inputs=[out_npz, truth], outputs=[mjson],
                   memory="4g") << f"""
tspaint benchmark metrics --truth {truth} --name {painter} {meta} \
    --deadband {DEADBAND} -o {mjson} {out_npz}
"""
        metrics_jsons.append(mjson)

# 4) aggregate every scenario × painter into the two tidy CSVs
summary_dir = os.path.join(RESULTS, "summary")
scalar_csv = f"{summary_dir}/summary_scalar.csv"
size_csv = f"{summary_dir}/summary_by_size.csv"
gwf.target("Aggregate", inputs=metrics_jsons, outputs=[scalar_csv, size_csv], memory="4g") << f"""
mkdir -p {summary_dir}
tspaint benchmark aggregate -o {summary_dir} {' '.join(metrics_jsons)}
"""
