"""Live end-to-end benchmark runs (external binaries) — opt-in and skippable.

These actually invoke RFMix / gnomix / SALAI-Net / Recomb-Mix, so they are gated behind
``TSPAINT_BENCHMARK_LIVE=1`` (to avoid triggering heavy pixi-env solves on a normal test run)
and additionally skip when a tool is not installed. Each exports diploid VCFs from a known-truth
admixture sim, runs the tool over them, and scores the result.

Run them with, e.g.::

    TSPAINT_BENCHMARK_LIVE=1 pytest tests/test_benchmark_integration.py -q
"""
import os

import pytest

import tspaint
import tspaint.benchmark as bm
from tspaint.benchmark import _common as C
from tspaint.sim import SOURCE_A, SOURCE_B, ADMIXED, admixture_demography

LIVE = os.environ.get("TSPAINT_BENCHMARK_LIVE")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not LIVE, reason="set TSPAINT_BENCHMARK_LIVE=1 to run")]


def _sim_export(tmp_path, seed=2):
    ts = tspaint.simulate_admixture(admixture_demography(Ne=1000, T_admix=30, T_split=5000, f_A=0.5),
                                    n_query=6, n_reference=10, sequence_length=1e6,
                                    recombination_rate=1e-8, random_seed=seed).ts
    npop = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[npop[s]] for s in ts.samples() if npop[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if npop[s] == admix]
    return bm.export_vcf(ts, labels, queries, outdir=str(tmp_path), seed=seed)


@pytest.mark.parametrize("tool", ["rfmix", "gnomix", "salai", "recombmix"])
def test_tool_runs_and_scores(tmp_path, tool):
    if not C.tool_available(tool):
        pytest.skip(f"{tool} not installed")
    paths = _sim_export(tmp_path)
    out = str(tmp_path / f"{tool}.npz")
    bm.run(tool, paths["query_vcf"], paths["ref_vcf"], sample_map=paths["sample_map"], out=out)

    rows = bm.score(paths["truth"], {tool: out})
    assert rows[0]["n_samples"] == 12                        # 6 admixed diploids -> 12 haplotypes
    assert rows[0]["balanced_accuracy"] > 0.6                # clear signal at T_admix=30
