"""RFMix painter tests (CLAUDE.md §9, §10 head-to-head comparator).

The parser test is fast and portable (hand-built ``.fb.tsv`` — no binary); the integration
test runs the external ``rfmix`` binary, so it is ``slow`` and skips when rfmix is absent
(install via the isolated ``compare`` env: ``pixi install -e compare``).
"""
import os

import numpy as np
import pytest

import tslai
from tslai.io_rfmix import DEFAULT_RFMIX, _parse_fb, rfmix_paint
from tslai.output import INFORMATIVE

rfmix_missing = not os.path.exists(DEFAULT_RFMIX)
needs_rfmix = pytest.mark.skipif(rfmix_missing, reason="rfmix binary not available")


def test_parse_fb_maps_haplotypes_and_pops(tmp_path):
    # One query individual (nodes 10, 11); two markers; pops "0","1" in state order.
    fb = tmp_path / "out.fb.tsv"
    fb.write_text(
        "#reference_panel_population:\t0\t1\n"
        "chromosome\tphysical_position\tgenetic_position\tgenetic_marker_index\t"
        "ind0:::hap1:::0\tind0:::hap1:::1\tind0:::hap2:::0\tind0:::hap2:::1\n"
        "1\t100\t0.001\t0\t0.9\t0.1\t0.2\t0.8\n"
        "1\t500\t0.005\t1\t0.7\t0.3\t0.4\t0.6\n"
    )
    tracks = _parse_fb(str(fb), [("ind0", (10, 11))], K=2, sequence_length=1000.0)

    assert set(tracks) == {10, 11}
    # hap1 -> node 10; hap2 -> node 11; pop columns recovered in state order [0,1].
    assert np.allclose(tracks[10][0].posterior, [0.9, 0.1])
    assert np.allclose(tracks[10][1].posterior, [0.7, 0.3])
    assert np.allclose(tracks[11][0].posterior, [0.2, 0.8])
    # painting tiles [0, L) with a step at the second marker's position; all INFORMATIVE.
    for node in (10, 11):
        segs = tracks[node]
        assert segs[0].left == 0.0 and segs[-1].right == 1000.0
        assert [s.right for s in segs][:-1] == [s.left for s in segs][1:]   # contiguous
        assert all(s.status == INFORMATIVE for s in segs)
    assert tracks[10][0].right == 500.0       # boundary at the 2nd marker (phys=500)


@pytest.mark.slow
@needs_rfmix
def test_rfmix_paint_runs_and_scores():
    from tslai.sim import local_ancestry_truth, SOURCE_A, SOURCE_B, ADMIXED
    from tslai.validate import balanced_accuracy, map_truth

    ts = tslai.simulate_admixture(n_admix=4, n_ref=6, sequence_length=5e5,
                                  recombination_rate=1e-8, random_seed=1, Ne=1000,
                                  T_admix=30, T_split=5000, f_A=0.5)
    np_ = ts.tables.nodes.population
    names = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, n in names.items() if n == SOURCE_A)
    B = next(p for p, n in names.items() if n == SOURCE_B)
    admix = next(p for p, n in names.items() if n == ADMIXED)
    sop = {A: 0, B: 1}
    labels = {int(s): sop[np_[s]] for s in ts.samples() if np_[s] in (A, B)}
    queries = [int(s) for s in ts.samples() if np_[s] == admix]
    truth, _ = local_ancestry_truth(ts)
    tstates = map_truth({q: truth[q] for q in queries}, sop)

    tracks = rfmix_paint(ts, labels, queries, generations=30)
    assert set(tracks) == set(queries)
    for q in queries:                                   # full [0, L) coverage per haplotype
        assert tracks[q][0].left == 0.0
        assert tracks[q][-1].right == ts.sequence_length
    assert balanced_accuracy(tracks, tstates, samples=queries) > 0.6   # clear signal
