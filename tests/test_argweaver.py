"""ARGweaver front end (io.argweaver): .sites writer, .smc -> tskit converter, and the run wrapper.

The converter and sites writer are tested offline (no arg-sample binary); a live end-to-end run is
opt-in, gated on the binary being present."""
import inspect
import os

import pytest

import tspaint
from tspaint import io
from tspaint.io_argweaver import (argweaver, write_sites, read_argweaver_smc, argweaver_binary_path,
                                  DEFAULT_ARGWEAVER)


def _smc(names, *tree_lines):
    """A minimal SMC file body: NAMES, REGION [1, 100], then TREE/SPR lines."""
    head = ["NAMES\t" + "\t".join(names), "REGION\tchr\t1\t100"]
    return "\n".join(head + list(tree_lines)) + "\n"


# Two non-recombining intervals with an SPR between them (node 4 re-times and swaps a child).
_T1 = ("TREE\t1\t50\t((0:100[&&NHX:age=0.0],1:100[&&NHX:age=0.0])4:200[&&NHX:age=100.0],"
       "(2:200[&&NHX:age=0.0],3:200[&&NHX:age=0.0])5:100[&&NHX:age=200.0])6[&&NHX:age=300.0];")
_SPR = "SPR\t50\t4\t120.0\t2\t120.0"
_T2 = ("TREE\t51\t100\t((0:120[&&NHX:age=0.0],2:120[&&NHX:age=0.0])4:180[&&NHX:age=120.0],"
       "(1:200[&&NHX:age=0.0],3:200[&&NHX:age=0.0])5:100[&&NHX:age=200.0])6[&&NHX:age=300.0];")


def _write(tmp_path, body, name="out.0.smc"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


# --- .smc -> tskit converter ----------------------------------------------------------------

def test_read_smc_two_intervals_and_topology(tmp_path):
    ts = read_argweaver_smc(_write(tmp_path, _smc(["n0", "n1", "n2", "n3"], _T1, _SPR, _T2)),
                            orig_names=["n0", "n1", "n2", "n3"])
    assert ts.num_samples == 4 and ts.num_trees == 2 and ts.sequence_length == 100.0
    t0, t1 = ts.at(10), ts.at(60)
    assert t0.parent(0) == t0.parent(1) and t0.parent(2) == t0.parent(3)   # interval 1: (0,1)(2,3)
    assert t1.parent(0) == t1.parent(2) and t1.parent(1) == t1.parent(3)   # interval 2: (0,2)(1,3)


def test_read_smc_persists_nodes_across_spr(tmp_path):
    """A node keeping its label+age across the SPR becomes one long-span tskit node (CLAUDE.md §5)."""
    ts = read_argweaver_smc(_write(tmp_path, _smc(["n0", "n1", "n2", "n3"], _T1, _SPR, _T2)))
    full_span = [(e.parent, e.child) for e in ts.edges() if e.left == 0.0 and e.right == 100.0]
    assert len(full_span) >= 2      # node 5's child 3 and root->5 persist unbroken across the SPR


def test_read_smc_restores_original_sample_order(tmp_path):
    """SMC may reorder its NAMES line; leaves are remapped to orig_names order by name."""
    smc = _smc(["n0", "n2", "n1", "n3"],   # SMC order differs from orig
               "TREE\t1\t100\t((0:1[&&NHX:age=0.0],1:1[&&NHX:age=0.0])4:1[&&NHX:age=10.0],"
               "(2:1[&&NHX:age=0.0],3:1[&&NHX:age=0.0])5:1[&&NHX:age=20.0])6[&&NHX:age=30.0];")
    ts = read_argweaver_smc(_write(tmp_path, smc), orig_names=["n0", "n1", "n2", "n3"])
    tree = ts.first()
    # SMC leaf 0=n0, 1=n2 grouped together -> tskit nodes 0 and 2; leaf 2=n1, 3=n3 -> nodes 1 and 3.
    assert tree.parent(0) == tree.parent(2) and tree.parent(1) == tree.parent(3)


def test_read_smc_rejects_malformed(tmp_path):
    with pytest.raises(ValueError):
        read_argweaver_smc(_write(tmp_path, "NAMES\tn0\tn1\n"))   # no REGION / TREE


# --- .sites writer --------------------------------------------------------------------------

def test_write_sites_format(tmp_path):
    ts = io.add_mutations(tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4,
                          recombination_rate=1e-8, random_seed=3, Ne=1000, T_admix=30,
                          T_split=5000, f_A=0.5), rate=1.2e-8, random_seed=3)
    p = tmp_path / "x.sites"
    L = write_sites(ts, str(p))
    lines = p.read_text().splitlines()
    assert lines[0].split("\t")[0] == "NAMES" and len(lines[0].split("\t")) == ts.num_samples + 1
    assert lines[1] == f"REGION\tchr\t1\t{L}"
    for row in lines[2:]:
        pos, bases = row.split("\t")
        assert int(pos) >= 1 and len(bases) == ts.num_samples     # one base per haplotype


# --- run wrapper: Ne / rates required, binary surfaced --------------------------------------

def test_argweaver_requires_ne():
    ts = tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4, random_seed=1)
    with pytest.raises(ValueError, match="requires Ne"):
        argweaver(ts, mutation_rate=1e-8, recombination_rate=1e-8)


def test_argweaver_requires_rates():
    ts = tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4, random_seed=1)
    with pytest.raises(ValueError, match="mutation_rate"):
        argweaver(ts, Ne=1000)


def test_argweaver_missing_binary(tmp_path):
    ts = tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4, random_seed=1)
    with pytest.raises(FileNotFoundError, match="arg-sample"):
        argweaver(ts, Ne=1000, mutation_rate=1e-8, recombination_rate=1e-8,
                  argweaver_bin=str(tmp_path / "nope"))


# --- API surface: ARGweaver flag names + aliases, io export, CLI -----------------------------

def test_signature_exposes_argweaver_flags():
    params = inspect.signature(argweaver).parameters
    for flag in ("Ne", "mutation_rate", "recombination_rate", "ntimes", "maxtime", "compress",
                 "iters", "sample_step", "burn_in", "thin", "seed", "argweaver_args"):
        assert flag in params, flag
    for alias in ("N", "m", "r", "n"):           # arg-sample short names as aliases
        assert alias in params, alias


def test_io_exposes_argweaver():
    assert io.argweaver is argweaver and callable(io.write_sites)
    assert "argweaver" in io.__all__


def test_cli_argweaver_and_install_present():
    from click.testing import CliRunner
    from tspaint.cli import cli
    out = CliRunner().invoke(cli, ["trees", "argweaver", "--help"]).output
    for tok in ("--Ne", "-m", "-r", "--ntimes", "--maxtime", "-c", "-n", "--sample-step",
                "--burnin", "--thin", "--seed", "--argweaver-arg"):
        assert tok in out, tok
    assert "argweaver" in CliRunner().invoke(cli, ["install", "--help"]).output


# --- live end-to-end (opt-in; needs the arg-sample binary) -----------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_ARGWEAVER), reason="arg-sample binary not available")
def test_argweaver_runs_end_to_end():
    # small region + coarse -c so ARGweaver finishes fast; sample_step=10 -> members at iters 0/10/20.
    ts = io.add_mutations(tspaint.simulate_admixture(n_admix=3, n_ref=3, sequence_length=1e4,
                          recombination_rate=1e-8, random_seed=7, Ne=1000, T_admix=30,
                          T_split=5000, f_A=0.5), rate=2e-8, random_seed=7)
    pop = ts.tables.nodes.population
    name = {p: ts.population(p).metadata.get("name", str(p)) for p in range(ts.num_populations)}
    A = next(p for p, nm in name.items() if nm == tspaint.sim.SOURCE_A)
    B = next(p for p, nm in name.items() if nm == tspaint.sim.SOURCE_B)
    labels = {int(s): (0 if pop[s] == A else 1) for s in ts.samples() if pop[s] in (A, B)}
    ens = argweaver(ts, Ne=1000, mutation_rate=2e-8, recombination_rate=1e-8, compress=10,
                    iters=20, sample_step=10, burn_in=0, seed=1)
    ens = ens if isinstance(ens, list) else [ens]
    assert ens and ens[0].num_samples == ts.num_samples
    # the converted ensemble must be a valid, paintable tree-sequence list
    painting = tspaint.paint(ens, labels)
    assert set(painting.posteriors)     # produced per-query posteriors
