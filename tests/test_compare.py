"""Tests for the head-to-head comparison harness (tslai.compare)."""
import numpy as np
import pytest
import tskit

from tslai.compare import nearest_reference_paint, head_to_head, tslai_paint
from tslai.output import INFORMATIVE


def _build_ts(parents, times, samples, L=10.0):
    t = tskit.TableCollection(sequence_length=L)
    for u in range(len(times)):
        flags = tskit.NODE_IS_SAMPLE if u in samples else 0
        t.nodes.add_row(flags=flags, time=times[u])
    for c, p in parents.items():
        if p != -1:
            t.edges.add_row(0, L, p, c)
    t.sort()
    return t.tree_sequence()


def test_nearest_reference_paint_picks_closest_label():
    # ((query0, refA1)3, refB2)4 : query coalesces with refA at t=1, refB at t=2
    ts = _build_ts({0: 3, 1: 3, 2: 4, 3: 4, 4: -1}, [0, 0, 0, 1.0, 2.0], {0, 1, 2})
    labels = {1: 0, 2: 1}
    segs = nearest_reference_paint(ts, labels, [0])[0]
    assert len(segs) == 1 and segs[0].status == INFORMATIVE
    assert int(np.argmax(segs[0].posterior)) == 0          # nearest ref is label-0 (A)


@pytest.mark.slow
def test_head_to_head_runs_and_scores():
    res = head_to_head({"tslai": tslai_paint, "nearest_ref": nearest_reference_paint},
                       T_admix=300, n_admix=8, n_ref=8, sequence_length=5e4, Ne=1000,
                       T_split=5000, seed=1, substrates=("true",))
    assert set(res) == {"true"}
    for name in ("tslai", "nearest_ref"):
        s = res["true"][name]
        assert 0.0 <= s["balanced_accuracy"] <= 1.0 and 0.0 <= s["confidence"] <= 1.0
