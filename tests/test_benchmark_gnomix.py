"""Gnomix runner config-fitting (tspaint.benchmark.gnomix): offline, no gnomix binary.

gnomix's shipped configs target whole chromosomes; their smoother asserts ``W >= 2*smooth_size``
windows where ``W ~= total_cM / window_size_cM``. On the benchmark's short (~10 cM) regions the
default (window 0.2, smooth 75 -> needs 150 windows) fails. These tests pin the two pure helpers
that adapt the config to the region length so that assertion holds.
"""
import numpy as np

from tspaint.benchmark.gnomix import _fit_config, _genetic_length_cM


def _write_map(path, lines, header=None):
    with open(path, "w") as f:
        if header:
            f.write(header + "\n")
        for ln in lines:
            f.write(ln + "\n")


def test_genetic_length_cM_plink(tmp_path):
    # plink 3-col: chrom pos cM ; cM is the last column, take the max.
    p = tmp_path / "map.tsv"
    _write_map(p, ["1\t100\t0.0001", "1\t5000000\t5.0", "1\t9999812\t9.999812"],
               header="#chm\tpos\tcM")
    assert _genetic_length_cM(str(p)) == 9.999812


def test_genetic_length_cM_hapmap_header_skipped(tmp_path):
    # hapmap 4-col with a non-numeric header row: cM is still the last column.
    p = tmp_path / "map.txt"
    _write_map(p, ["1\t100\t1.0\t0.0001", "1\t9999812\t1.0\t9.5"],
               header="Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)")
    assert _genetic_length_cM(str(p)) == 9.5


def _assertion_holds(cfg, total_cM):
    """gnomix's smoother check: W = int(total_cM / window) >= 2 * smooth_size."""
    m = cfg["model"]
    W = int(total_cM / m["window_size_cM"])
    return W >= 2 * m["smooth_size"]


def test_fit_config_benchmark_region_keeps_default_smoother():
    # 10 cM region: the default smoother (75) survives, only the window shrinks to fit.
    cfg = {"model": {"window_size_cM": 0.2, "smooth_size": 75}}
    _fit_config(cfg, 10.0)
    assert cfg["model"]["window_size_cM"] <= 0.2        # never coarsened past the config window
    assert cfg["model"]["smooth_size"] == 75            # enough windows -> smoother unchanged
    assert _assertion_holds(cfg, 10.0)


def test_fit_config_short_region_caps_smoother():
    # very short region: window can't shrink enough alone, so smooth_size is capped.
    cfg = {"model": {"window_size_cM": 0.2, "smooth_size": 75}}
    _fit_config(cfg, 0.5)
    assert cfg["model"]["smooth_size"] < 75
    assert cfg["model"]["smooth_size"] >= 2
    assert _assertion_holds(cfg, 0.5)


def test_fit_config_assertion_holds_across_region_sizes():
    for total_cM in [0.4, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]:
        cfg = {"model": {"window_size_cM": 0.2, "smooth_size": 75}}
        _fit_config(cfg, total_cM)
        assert _assertion_holds(cfg, total_cM), total_cM
        assert cfg["model"]["window_size_cM"] > 0
        assert cfg["model"]["smooth_size"] >= 2


def test_fit_config_does_not_coarsen_long_chromosome():
    # On a whole chromosome (>= 30 cM) the default already satisfies the assertion; window stays
    # at the config's value (we only ever shrink it) and the smoother is untouched.
    cfg = {"model": {"window_size_cM": 0.2, "smooth_size": 75}}
    _fit_config(cfg, 100.0)
    assert cfg["model"]["window_size_cM"] == 0.2
    assert cfg["model"]["smooth_size"] == 75
