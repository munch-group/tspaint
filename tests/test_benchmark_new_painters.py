"""Offline tests for the four painters added from the comparator review.

Parsers and input writers only — the live runs are opt-in (see tests/test_benchmark_integration.py).
"""
import gzip
import os

import numpy as np
import pytest

from tspaint.benchmark import _common as C
from tspaint.benchmark._msp import parse_flare_anc_vcf, tracks_from_marker_posteriors
from tspaint.benchmark.mosaic import _write_mosaic_inputs
from tspaint.output import INFORMATIVE


def test_tracks_from_marker_posteriors_tiles_and_merges():
    pos = [10, 20, 30, 40]
    post = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    tr = tracks_from_marker_posteriors(pos, {7: post}, 100.0)
    segs = tr[7]
    assert [(s.left, s.right) for s in segs] == [(0.0, 30.0), (30.0, 100.0)]   # runs merged
    assert all(s.status == INFORMATIVE for s in segs)
    np.testing.assert_allclose(segs[0].posterior, [1.0, 0.0])
    np.testing.assert_allclose(segs[1].posterior, [0.0, 1.0])


def _anc_vcf(tmp_path, ancestry_line, rows):
    p = tmp_path / "out.anc.vcf.gz"
    hdr = ["##fileformat=VCFv4.2", ancestry_line,
           "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tq0"]
    with gzip.open(p, "wt") as f:
        f.write("\n".join(hdr + rows) + "\n")
    return str(p)


def test_parse_flare_maps_ancestry_ids_through_the_meta_line(tmp_path):
    # FLARE's ancestry integers are ITS indices, not our states. Here panel "1" is FLARE ancestry 0
    # and panel "0" is FLARE ancestry 1 — i.e. reversed. A parser that assumed identity would
    # silently produce a perfectly swapped painting, so this is the regression that matters.
    path = _anc_vcf(tmp_path, "##ANCESTRY=<1=0,0=1>", [
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT:AN1:AN2:ANP1:ANP2\t0|1:0:1:0.9,0.1:0.2,0.8",
    ])
    tr = parse_flare_anc_vcf(path, [("q0", (0, 1))], K=2, sequence_length=200.0)
    # FLARE ancestry 0 -> our state 1; ancestry 1 -> our state 0
    np.testing.assert_allclose(tr[0][0].posterior, [0.1, 0.9])
    np.testing.assert_allclose(tr[1][0].posterior, [0.8, 0.2])


def test_parse_flare_hard_calls(tmp_path):
    path = _anc_vcf(tmp_path, "##ANCESTRY=<0=0,1=1>", [
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT:AN1:AN2\t0|1:1:0",
    ])
    tr = parse_flare_anc_vcf(path, [("q0", (0, 1))], K=2, sequence_length=200.0, probs=False)
    np.testing.assert_allclose(tr[0][0].posterior, [0.0, 1.0])
    np.testing.assert_allclose(tr[1][0].posterior, [1.0, 0.0])


def test_parse_flare_requires_the_ancestry_meta_line(tmp_path):
    path = _anc_vcf(tmp_path, "##source=flare", [
        "1\t100\t.\tA\tT\t.\tPASS\t.\tGT:AN1:AN2:ANP1:ANP2\t0|1:0:1:0.9,0.1:0.2,0.8"])
    with pytest.raises(ValueError, match="ANCESTRY"):
        parse_flare_anc_vcf(path, [("q0", (0, 1))], K=2, sequence_length=200.0)


def test_plink_map_format_is_four_column_cm_before_bp(tmp_path):
    # FLARE rejects the 3-column map RFMix calls a genetic map; it needs a real PLINK .map.
    panel = C.Panel(positions=np.array([100, 200]), geno=np.zeros((2, 2), int),
                    alleles=[("A", "T")] * 2, query=[], ref=[], K=2, contig="1",
                    sequence_length=300.0)
    p3, p4 = tmp_path / "a.map", tmp_path / "b.map"
    C.write_genetic_map(str(p3), panel, 1e-8, fmt="plink")
    C.write_genetic_map(str(p4), panel, 1e-8, fmt="plink-map")
    assert p3.read_text().splitlines()[0].split("\t") == ["1", "100", "0.0001000000"]
    c, mid, cm, bp = p4.read_text().splitlines()[0].split("\t")
    assert (c, mid, bp) == ("1", "1:100", "100") and float(cm) == pytest.approx(1e-4)


def test_mosaic_inputs_are_written_in_its_bespoke_format(tmp_path):
    geno = np.array([[0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0]])      # 2 snps x 6 haps
    panel = C.Panel(positions=np.array([100, 200]), geno=geno, alleles=[("A", "T"), ("C", "G")],
                    query=[("q0", (0, 1), (0, 1))],
                    ref=[("r0", (2, 3), 0), ("r1", (4, 5), 1)],
                    K=2, contig="1", sequence_length=300.0)
    _write_mosaic_inputs(str(tmp_path), panel, 1, 1e-8)

    # genofiles: #snps rows, one CHARACTER per haplotype (MOSAIC reads them fixed-width)
    assert (tmp_path / "P0genofile.1").read_text().splitlines() == ["10", "01"]
    assert (tmp_path / "P1genofile.1").read_text().splitlines() == ["01", "10"]
    assert (tmp_path / "TARGETgenofile.1").read_text().splitlines() == ["01", "10"]
    # snpfile: 6 columns, position is column 4
    assert (tmp_path / "snpfile.1").read_text().split("\n")[0].split() == \
        ["rs100", "1", "0.0001000000", "100", "A", "T"]
    # rates: THREE rows — count, positions, cumulative cM (MOSAIC scans past line 1 into 2 columns)
    rates = (tmp_path / "rates.1").read_text().splitlines()
    assert rates[0] == "2" and rates[1].split() == ["100", "200"]
    assert [float(x) for x in rates[2].split()] == pytest.approx([1e-4, 2e-4])
    assert "P0" in (tmp_path / "sample.names").read_text()


def test_new_tools_have_launchers_and_availability_checks():
    for tool in ("flare", "loter", "mosaic", "ghostbuster"):
        argv = C.tool_command(tool, ["--x"])
        assert argv and argv[-1] == "--x"
        assert isinstance(C.tool_available(tool), bool)
