"""Benchmark output parsers (tspaint.benchmark._msp): tool files → Segment tracks.

Fully offline (hand-built tool outputs, no binaries). Posteriors for RFMix/gnomix ``.fb``;
one-hot (0/1) for the ``.msp.tsv`` (RFMix/gnomix/SALAI) and Recomb-Mix segment files; plus a
non-identity population-code remap.
"""
import numpy as np

from tspaint.benchmark._msp import parse_fb, parse_msp, parse_recombmix_segments
from tspaint.output import INFORMATIVE


def test_parse_fb_posteriors(tmp_path):
    fb = tmp_path / "out.fb.tsv"
    fb.write_text(
        "#reference_panel_population:\t0\t1\n"
        "chromosome\tphysical_position\tgenetic_position\tgenetic_marker_index\t"
        "q0:::hap1:::0\tq0:::hap1:::1\tq0:::hap2:::0\tq0:::hap2:::1\n"
        "1\t100\t0.001\t0\t0.9\t0.1\t0.2\t0.8\n"
        "1\t500\t0.005\t1\t0.7\t0.3\t0.4\t0.6\n")
    tracks = parse_fb(str(fb), [("q0", (0, 1))], K=2, sequence_length=1000.0)

    assert set(tracks) == {0, 1}
    np.testing.assert_allclose(tracks[0][0].posterior, [0.9, 0.1])   # hap1 -> key 0
    np.testing.assert_allclose(tracks[0][1].posterior, [0.7, 0.3])
    np.testing.assert_allclose(tracks[1][0].posterior, [0.2, 0.8])   # hap2 -> key 1
    for key in (0, 1):                                               # tile [0, L), all informative
        segs = tracks[key]
        assert segs[0].left == 0.0 and segs[-1].right == 1000.0
        assert [s.right for s in segs][:-1] == [s.left for s in segs][1:]
        assert all(s.status == INFORMATIVE for s in segs)
    assert tracks[0][0].right == 500.0


def test_parse_msp_one_hot(tmp_path):
    msp = tmp_path / "out.msp.tsv"
    msp.write_text(
        "#Subpopulation order/codes: 0=0\t1=1\n"
        "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tq0.0\tq0.1\n"
        "1\t0\t300\t0\t0\t3\t0\t1\n"
        "1\t300\t1000\t0\t0\t5\t1\t1\n")
    tracks = parse_msp(str(msp), [("q0", (0, 1))], K=2, sequence_length=1000.0)

    assert set(tracks) == {0, 1}
    # key 0: state 0 on [0,300), state 1 on [300,1000) — one-hot
    np.testing.assert_array_equal(tracks[0][0].posterior, [1.0, 0.0])
    assert tracks[0][0].left == 0.0 and tracks[0][0].right == 300.0
    np.testing.assert_array_equal(tracks[0][1].posterior, [0.0, 1.0])
    assert tracks[0][1].right == 1000.0
    # key 1: state 1 throughout -> a single merged segment
    assert len(tracks[1]) == 1
    np.testing.assert_array_equal(tracks[1][0].posterior, [0.0, 1.0])
    assert tracks[1][0].left == 0.0 and tracks[1][0].right == 1000.0


def test_parse_recombmix_segments_one_hot(tmp_path):
    txt = tmp_path / "local.txt"
    txt.write_text(
        "#Population label and ID: 0=0\t1=1\n"
        "q0_0\t0\t300\t0\t300\t1000\t1\n"
        "q0_1\t0\t1000\t1\n")
    tracks = parse_recombmix_segments(str(txt), [("q0", (0, 1))], K=2, sequence_length=1000.0)

    assert set(tracks) == {0, 1}
    np.testing.assert_array_equal(tracks[0][0].posterior, [1.0, 0.0])
    assert (tracks[0][0].left, tracks[0][0].right) == (0.0, 300.0)
    np.testing.assert_array_equal(tracks[0][1].posterior, [0.0, 1.0])
    assert tracks[0][1].right == 1000.0
    assert len(tracks[1]) == 1 and tracks[1][0].right == 1000.0


def test_parse_recombmix_remaps_population_codes(tmp_path):
    # Header says population 1 has code 0 and population 0 has code 1 (order-dependent codes).
    txt = tmp_path / "local.txt"
    txt.write_text(
        "#Population label and ID: 1=0\t0=1\n"
        "q0_0\t0\t1000\t0\n")                                # code 0 -> state 1
    tracks = parse_recombmix_segments(str(txt), [("q0", (0, 1))], K=2, sequence_length=1000.0)
    np.testing.assert_array_equal(tracks[0][0].posterior, [0.0, 1.0])   # remapped to state 1


def test_parse_msp_explicit_code_to_state(tmp_path):
    msp = tmp_path / "out.msp.tsv"
    msp.write_text(
        "#chm\tspos\tepos\tsgpos\tegpos\tn snps\tq0.0\tq0.1\n"
        "1\t0\t1000\t0\t0\t9\t0\t1\n")
    # SALAI-style: codes are indices into population_ids; here code 0 -> state 1, 1 -> state 0.
    tracks = parse_msp(str(msp), [("q0", (0, 1))], K=2, sequence_length=1000.0,
                       code_to_state={0: 1, 1: 0})
    np.testing.assert_array_equal(tracks[0][0].posterior, [0.0, 1.0])
    np.testing.assert_array_equal(tracks[1][0].posterior, [1.0, 0.0])
