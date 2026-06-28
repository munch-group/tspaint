"""Benchmark plumbing (tspaint.benchmark._common): VCF panel resolution, writers, npz output.

All offline — no external binaries. Builds tiny VCFs by hand and checks the resolver, the
input writers, state assignment, missing-fill, and the .npz round-trip (incl. hap names).
"""
import numpy as np
import pytest

from tspaint.benchmark import _common as C
from tspaint.benchmark._common import resolve_panel, assign_states, read_sample_map
from tspaint.output import Segment, INFORMATIVE, MISSING_INFO
from tspaint import serialize


def _vcf(path, samples, rows, *, ploidy=2, contig="1"):
    """Write a tiny phased VCF. ``rows`` = list of (pos, [gt-string per sample])."""
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=%s>\n" % contig)
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n")
        for pos, gts in rows:
            f.write(f"{contig}\t{pos}\t.\tA\tT\t.\tPASS\t.\tGT\t" + "\t".join(gts) + "\n")


def _map(path, pairs):
    with open(path, "w") as f:
        for name, lab in pairs:
            f.write(f"{name}\t{lab}\n")


# --- state assignment -----------------------------------------------------------------------

def test_assign_states_integer_labels_literal():
    states, K, lab = assign_states([("r0", "0"), ("r1", "1"), ("r2", "1")])
    assert states == {"r0": 0, "r1": 1, "r2": 1} and K == 2
    assert lab == {0: "0", 1: "1"}


def test_assign_states_string_labels_by_appearance():
    states, K, lab = assign_states([("a", "EUR"), ("b", "AFR"), ("c", "EUR")])
    assert states == {"a": 0, "b": 1, "c": 0} and K == 2
    assert lab == {0: "EUR", 1: "AFR"}


def test_read_sample_map_skips_comments(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text("#Sample\tPanel\nr0\t0\nr1\t1\n")
    assert read_sample_map(str(p)) == [("r0", "0"), ("r1", "1")]


# --- resolve_panel: two-file and combined ---------------------------------------------------

def test_resolve_panel_two_file(tmp_path):
    q = tmp_path / "q.vcf"
    r = tmp_path / "r.vcf"
    m = tmp_path / "m.tsv"
    _vcf(q, ["q0", "q1"], [(100, ["0|1", "1|0"]), (200, ["1|1", "0|0"])])
    _vcf(r, ["r0", "r1", "r2", "r3"],
         [(100, ["0|0", "0|0", "1|1", "1|1"]), (200, ["0|0", "1|0", "1|1", "0|1"])])
    _map(m, [("r0", 0), ("r1", 0), ("r2", 1), ("r3", 1)])

    panel = resolve_panel(str(q), str(r), sample_map=str(m))
    assert panel.K == 2
    assert [name for name, _c, _k in panel.query] == ["q0", "q1"]
    assert panel.query_keys == [0, 1, 2, 3]                  # 2 diploid query samples -> 4 haps
    assert [s for _n, _c, s in panel.ref] == [0, 0, 1, 1]
    np.testing.assert_array_equal(panel.positions, [100, 200])
    assert panel.geno.shape == (2, 4 + 8)                    # 2 query haps*2 + 4 ref samples*2


def test_resolve_panel_combined(tmp_path):
    v = tmp_path / "all.vcf"
    m = tmp_path / "m.tsv"
    _vcf(v, ["q0", "r0", "q1", "r1"],
         [(100, ["0|1", "0|0", "1|0", "1|1"]), (200, ["1|1", "0|0", "0|0", "1|1"])])
    _map(m, [("r0", 0), ("r1", 1)])

    panel = resolve_panel(str(v), None, sample_map=str(m))
    assert [name for name, _c, _k in panel.query] == ["q0", "q1"]
    assert panel.query_keys == [0, 1, 2, 3]
    assert {name: s for name, _c, s in panel.ref} == {"r0": 0, "r1": 1}
    # combined-mode query geno columns are the *original* VCF positions of q0, q1
    assert panel.query[0][1] == (0, 1) and panel.query[1][1] == (4, 5)


def test_resolve_panel_intersects_sites(tmp_path):
    q = tmp_path / "q.vcf"
    r = tmp_path / "r.vcf"
    m = tmp_path / "m.tsv"
    _vcf(q, ["q0"], [(100, ["0|1"]), (200, ["1|0"]), (300, ["1|1"])])
    _vcf(r, ["r0", "r1"], [(200, ["0|0", "1|1"]), (300, ["0|0", "1|1"]), (400, ["0|0", "1|1"])])
    _map(m, [("r0", 0), ("r1", 1)])
    panel = resolve_panel(str(q), str(r), sample_map=str(m))
    np.testing.assert_array_equal(panel.positions, [200, 300])    # shared only


def test_resolve_panel_rejects_haploid(tmp_path):
    v = tmp_path / "all.vcf"
    m = tmp_path / "m.tsv"
    _vcf(v, ["q0", "r0"], [(100, ["0", "1"]), (200, ["1", "0"])], ploidy=1)
    _map(m, [("r0", 0)])
    with pytest.raises(ValueError, match="diploid"):
        resolve_panel(str(v), None, sample_map=str(m))


# --- writers round-trip ---------------------------------------------------------------------

def test_written_inputs_readback(tmp_path):
    q = tmp_path / "q.vcf"
    r = tmp_path / "r.vcf"
    m = tmp_path / "m.tsv"
    _vcf(q, ["q0", "q1"], [(100, ["0|1", "1|0"]), (200, ["1|1", "0|0"])])
    _vcf(r, ["r0", "r1"], [(100, ["0|0", "1|1"]), (200, ["0|0", "1|1"])])
    _map(m, [("r0", 0), ("r1", 1)])
    panel, qv, rv, sm = C.setup_inputs(str(q), str(r), str(m), str(tmp_path / "work"))

    back = C.read_vcf(qv)                                     # query VCF round-trips
    assert back.samples == ["q0", "q1"] and back.ploidy == 2
    np.testing.assert_array_equal(back.positions, [100, 200])
    assert read_sample_map(sm) == [("r0", "0"), ("r1", "1")]

    g = tmp_path / "gmap.tsv"
    C.write_genetic_map(str(g), panel, 1e-8, fmt="plink")
    cols = g.read_text().splitlines()[0].split("\t")
    assert len(cols) == 3 and cols[0] == "1"
    h = tmp_path / "hap.txt"
    C.write_genetic_map(str(h), panel, 1e-8, fmt="hapmap")
    assert h.read_text().splitlines()[0].startswith("Chromosome")


def test_save_tracks_round_trip_with_names(tmp_path):
    q = tmp_path / "q.vcf"
    r = tmp_path / "r.vcf"
    m = tmp_path / "m.tsv"
    _vcf(q, ["q0"], [(100, ["0|1"]), (200, ["1|0"])])
    _vcf(r, ["r0", "r1"], [(100, ["0|0", "1|1"]), (200, ["0|0", "1|1"])])
    _map(m, [("r0", 0), ("r1", 1)])
    panel = resolve_panel(str(q), str(r), sample_map=str(m))

    tracks = {0: [Segment(0.0, panel.sequence_length, np.array([0.8, 0.2]), INFORMATIVE)],
              1: [Segment(0.0, panel.sequence_length, np.array([0.3, 0.7]), INFORMATIVE)]}
    out = tmp_path / "p.npz"
    C.save_tracks(str(out), tracks, panel)

    back = serialize.load_painting(str(out))
    assert set(back) == {0, 1}
    np.testing.assert_allclose(back[0][0].posterior, [0.8, 0.2])
    meta = serialize.load_painting_meta(str(out))
    assert meta["sample_names"] == {0: "q0.0", 1: "q0.1"}
    assert meta["seqlen"] == panel.sequence_length


def test_fill_missing_adds_missing_info(tmp_path):
    q = tmp_path / "q.vcf"
    r = tmp_path / "r.vcf"
    m = tmp_path / "m.tsv"
    _vcf(q, ["q0"], [(100, ["0|1"]), (200, ["1|0"])])
    _vcf(r, ["r0", "r1"], [(100, ["0|0", "1|1"]), (200, ["0|0", "1|1"])])
    _map(m, [("r0", 0), ("r1", 1)])
    panel = resolve_panel(str(q), str(r), sample_map=str(m))
    tracks = {0: [Segment(0.0, panel.sequence_length, np.array([1.0, 0.0]), INFORMATIVE)]}
    C.fill_missing(tracks, panel)
    assert set(tracks) == {0, 1}                             # key 1 was filled
    assert tracks[1][0].status == MISSING_INFO
