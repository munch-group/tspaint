"""SINGER windowing CLI (tspaint.io_singer + tspaint.cli trees singer-window/merge-arg)."""
import os

import numpy as np
import pytest
from click.testing import CliRunner

import tspaint
from tspaint.cli import cli, _parse_gaps, _read_manifest
from tspaint.io_singer import build_merge_table, DEFAULT_SINGER


# --- unit (no binary) -----------------------------------------------------------------------

def test_build_merge_table_order_naming_blocks():
    windows = [(0, 0.0, 5e6, "A"), (2, 10e6, 15e6, "C"), (1, 5e6, 10e6, "B")]   # unsorted
    rows = build_merge_table(windows, 3)
    assert [r[3] for r in rows] == [0.0, 5e6, 10e6]                  # genome order, local block coords
    assert rows[0] == ("A_nodes_3.txt", "A_branches_3.txt", "A_muts_3.txt", 0.0)
    assert rows[1][0] == "B_nodes_3.txt"                            # member index threaded through


def test_build_merge_table_skip_gaps_and_absolute():
    windows = [(0, 0.0, 5e6, "A"), (1, 5e6, 10e6, "B"), (2, 10e6, 15e6, "C")]
    rows = build_merge_table(windows, 3, skip_gaps=[(5e6, 10e6)])    # drop the centromere window
    assert [r[0] for r in rows] == ["A_nodes_3.txt", "C_nodes_3.txt"]
    rows_abs = build_merge_table(windows, 0, coords="absolute")
    assert all(r[3] == 0.0 for r in rows_abs)


def test_parse_gaps_and_manifest(tmp_path):
    assert _parse_gaps("5e6-8e6, 1e7-1.1e7") == [(5e6, 8e6), (1e7, 1.1e7)]
    assert _parse_gaps(None) == []
    man = tmp_path / "w.tsv"
    man.write_text("# windows\n0 0 5e6 p0\n1 5e6 1e7 p1\n")
    assert _read_manifest(str(man)) == [(0, 0.0, 5e6, "p0"), (1, 5e6, 1e7, "p1")]


def test_merge_arg_dry_run(tmp_path):
    man = tmp_path / "windows.tsv"
    man.write_text("0 0 5000000 argA\n1 5000000 10000000 argB\n")
    res = CliRunner().invoke(cli, ["trees", "merge-arg", "--manifest", str(man),
                                   "--member", "2", "--dry-run"])
    assert res.exit_code == 0, res.output
    lines = res.output.strip().splitlines()
    assert lines[0] == "argA_nodes_2.txt argA_branches_2.txt argA_muts_2.txt 0"
    assert lines[1] == "argB_nodes_2.txt argB_branches_2.txt argB_muts_2.txt 5000000"


# --- real SINGER (slow; needs the binary) ---------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_window_runs_and_is_local(tmp_path):
    """singer-window produces the raw tables; coords are LOCAL (=> coords='local' block=start)."""
    from tspaint.io import write_haploid_vcf
    from tspaint.io_singer import singer_window
    from tspaint.io_tsinfer import add_mutations

    ts = add_mutations(tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=4e4,
                       recombination_rate=1e-8, random_seed=3, Ne=1000, T_admix=30,
                       T_split=5000, f_A=0.5), rate=2e-7, random_seed=3)
    vcf = str(tmp_path / "data.vcf")
    write_haploid_vcf(ts, vcf)
    S, E = 2e4, 4e4
    idxs = singer_window(vcf, start=S, end=E, out_prefix=str(tmp_path / "argW"), Ne=1000,
                         mutation_rate=2e-7, recombination_rate=1e-8, n_samples=5, thin=2, seed=7)
    assert len(idxs) >= 1
    i = idxs[-1]
    for suf in ("nodes", "branches", "muts"):
        assert os.path.exists(tmp_path / f"argW_{suf}_{i}.txt")
    edge = np.loadtxt(tmp_path / f"argW_branches_{i}.txt")
    assert edge[:, 1].max() <= (E - S) * 1.001       # window-LOCAL coords (the coords='local' default)


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_windowed_stitches_across_windows(tmp_path):
    """singer_windowed: write VCF once -> 2 windows in parallel -> merge each member -> ensemble."""
    from tspaint.io import singer_windowed
    from tspaint.io_tsinfer import add_mutations
    import tskit

    ts = add_mutations(tspaint.simulate_admixture(n_admix=4, n_ref=4, sequence_length=6e4,
                       recombination_rate=1e-8, random_seed=5, Ne=1000, T_admix=30,
                       T_split=5000, f_A=0.5), rate=3e-7, random_seed=5)
    ens = singer_windowed(ts, window_size=3e4, Ne=1000, mutation_rate=3e-7,
                          recombination_rate=1e-8, n_samples=4, thin=2, burn_in=1,
                          seed=11, n_jobs=2, workdir=str(tmp_path))

    ens = ens if isinstance(ens, list) else [ens]
    assert 1 <= len(ens) <= 4                                  # members 1..3 survive burn_in=1
    for t in ens:
        assert isinstance(t, tskit.TreeSequence)
        assert t.num_samples == ts.num_samples                 # sample order preserved by the merge
        assert t.sequence_length > 3e4                         # stitched BEYOND the first window
    assert (tmp_path / "data.vcf").exists()                    # the write-once VCF
    assert (tmp_path / "member_1.trees").exists()              # a stitched member on disk
