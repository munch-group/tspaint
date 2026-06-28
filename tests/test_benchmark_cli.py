"""`tspaint benchmark` CLI (tspaint.benchmark.cli): help, export-vcf, and the score table.

Offline — runs no external tool binaries. The runner subcommands' wiring is exercised by the
library tests; here we check the CLI surface and the end-to-end export → score path.
"""
import numpy as np
from click.testing import CliRunner

from tspaint.cli import cli
from tspaint import serialize
from tspaint.benchmark.score import load_truth
from tspaint.output import Segment, INFORMATIVE


def _run(*args):
    res = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res


def test_benchmark_help_lists_tools():
    res = CliRunner().invoke(cli, ["benchmark", "--help"])
    assert res.exit_code == 0
    for sub in ("rfmix", "gnomix", "salai", "recombmix", "export-vcf", "score"):
        assert sub in res.output


def test_top_level_lists_benchmark():
    res = CliRunner().invoke(cli, ["--help"])
    assert "benchmark" in res.output


def test_export_vcf_and_score_cli(tmp_path):
    # simulate -> labels + trees
    trees = tmp_path / "sim.trees"
    labels = tmp_path / "labels.json"
    _run("simulate", "--n-admix", 4, "--n-ref", 4, "--length", 3e5, "--ploidy", 2,
         "--seed", 1, "-o", trees, "--labels-out", labels)

    # export-vcf -> query/ref VCFs + sample map + truth
    outdir = tmp_path / "vcf"
    res = _run("benchmark", "export-vcf", trees, "--labels", labels, "-o", outdir)
    for fn in ("query.vcf", "reference.vcf", "sample_map.tsv", "truth.npz"):
        assert (outdir / fn).exists(), res.output

    # build a perfect painting from the truth, then score it through the CLI
    truth = load_truth(str(outdir / "truth.npz"))
    seqlen = max(r for segs in truth.values() for (_l, r, _s) in segs)
    perfect = {k: [Segment(l, r, _one_hot(s), INFORMATIVE) for (l, r, s) in segs]
               for k, segs in truth.items()}
    pp = tmp_path / "perfect.npz"
    serialize.save_painting(str(pp), perfect, seqlen=seqlen)

    res = _run("benchmark", "score", "--truth", str(outdir / "truth.npz"), f"perfect={pp}")
    assert "bal-acc" in res.output and "perfect" in res.output
    assert "1.000" in res.output


def _one_hot(s):
    p = np.zeros(2)
    p[int(s)] = 1.0
    return p
