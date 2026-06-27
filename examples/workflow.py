"""Example GWF workflow: SINGER-windowed ensemble → tspaint painting on a cluster.

A reference template (not executed by CI, not imported by the package). It wires the ``tspaint``
CLI subcommands as GWF targets with file in/out — no glue Python. Run with ``gwf run`` after
editing the parameters and pointing ``--ne / --mut-rate / --recomb-rate`` at your data.

Pipeline (see paral-assess.md):

    singer-window × W windows ─► merge-arg × M members ─► fit (1) ─► paint × M ─► merge (1) ─► date

The cluster fans out over windows and members; ``fit`` is the single coupling job (it pools the
EM sufficient statistics across the ensemble) and parallelises internally via ``-j``.

Requires: gwf (``pip install gwf``); the ``tspaint`` CLI on PATH; the SINGER binary
(``TSPAINT_SINGER``) for ``singer-window``; and an interpreter with ``tszip`` for ``merge-arg``
(``--python``), since SINGER's ``merge_ARG.py`` imports it.
"""
from gwf import Workflow

gwf = Workflow(defaults={"cores": 8, "memory": "16g", "walltime": "08:00:00"})

# --- parameters (edit these) ----------------------------------------------------------------
VCF = "cohort.vcf"                 # SINGER reads cohort.vcf (prefix "cohort")
LABELS = "labels.json"             # {"<node-id>": <ancestry-state>} for the reference haplotypes
NE, MU, RHO = 1e4, 1.25e-8, 1e-8   # SINGER demographic / rate parameters
CHROM_LEN = 30_000_000
WINDOW = 5_000_000
N_DRAWS = 20                       # SINGER MCMC samples; BURN_IN discarded
BURN_IN = 5
SKIP_GAPS = ""                     # e.g. "12e6-16e6" to skip a centromere
PY_WITH_TSZIP = "python"           # interpreter that has tszip, for merge_ARG.py

WINDOWS = [(w, s, min(s + WINDOW, CHROM_LEN))
           for w, s in enumerate(range(0, CHROM_LEN, WINDOW))]
MEMBERS = list(range(BURN_IN, N_DRAWS))     # post-burn-in posterior draws


# --- 1. SINGER per genomic window (W jobs) --------------------------------------------------
manifest = "windows.tsv"
with open(manifest, "w") as fh:             # window_index start end out_prefix
    for w, s, e in WINDOWS:
        fh.write(f"{w} {s} {e} arg_w{w}\n")

for w, s, e in WINDOWS:
    last = f"arg_w{w}_nodes_{N_DRAWS - 1}.txt"      # a representative SINGER output for this window
    gwf.target(f"singer_w{w}", inputs=[VCF], outputs=[last], cores=1, walltime="24:00:00") << f"""
    tspaint trees singer-window {VCF} --start {s} --end {e} --out-prefix arg_w{w} \
        --ne {NE} --mut-rate {MU} --recomb-rate {RHO} --n {N_DRAWS} --thin 10 --seed 42
    """

# --- 2. stitch each posterior member across windows (M jobs) --------------------------------
window_markers = [f"arg_w{w}_nodes_{N_DRAWS - 1}.txt" for w, _, _ in WINDOWS]
member_trees = []
for i in MEMBERS:
    out = f"member_{i:03d}.trees"
    member_trees.append(out)
    gaps = f"--skip-gaps {SKIP_GAPS}" if SKIP_GAPS else ""
    gwf.target(f"merge_arg_{i}", inputs=window_markers, outputs=[out], cores=1) << f"""
    tspaint trees merge-arg --manifest {manifest} --member {i} {gaps} \
        --coords local --python {PY_WITH_TSZIP} -o {out}
    """

# --- 3. one pooled fit over the whole ensemble (1 job, internally multi-core) ----------------
gwf.target("fit", inputs=member_trees + [LABELS], outputs=["params.npz"]) << f"""
tspaint fit {' '.join(member_trees)} --labels {LABELS} -j $SLURM_CPUS_PER_NODE -o params.npz
"""

# --- 4. paint each member with the shared params (M jobs, independent) -----------------------
paintings = []
for i in MEMBERS:
    out = f"member_{i:03d}.painting.npz"
    paintings.append(out)
    gwf.target(f"paint_{i}", inputs=[f"member_{i:03d}.trees", "params.npz"], outputs=[out]) << f"""
    tspaint paint member_{i:03d}.trees --params params.npz -j $SLURM_CPUS_PER_NODE -o {out}
    """

# --- 5. marginalise the ARG: mean painting + uncertainty band (1 job) ------------------------
gwf.target("merge", inputs=paintings, outputs=["merged.painting.npz"], cores=1) << f"""
tspaint merge {' '.join(paintings)} -o merged.painting.npz
"""

# --- 6. downstream analyses (fan out per member as needed) -----------------------------------
gwf.target("date", inputs=[f"member_{MEMBERS[0]:03d}.trees", LABELS], outputs=["rtt.npz"]) << f"""
tspaint date member_{MEMBERS[0]:03d}.trees --labels {LABELS} -o rtt.npz
"""
