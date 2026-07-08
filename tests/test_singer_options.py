"""io.singer / io.argweaver options: the unified ts / mcmc_step / mcmc_burnin sampling knobs, the
underscore-prefixed native terminal flags (_Ne / _m / _r / _n_samples / ...), and the precedence
guard (a plain knob and its ``_``-counterpart cannot both be set)."""
import inspect
import os
import warnings

import pytest

import tspaint
from tspaint import io
from tspaint.io_singer import (_resolve_singer_rates, singer, singer_window, singer_windowed,
                               DEFAULT_SINGER, _singer_sampling, _argweaver_sampling)
from tspaint.io_argweaver import argweaver


# --- rate resolution: -ratio (default 1) -> r = m*ratio; _r / _recomb_map short-circuit ------

def test_ratio_derives_recombination_rate():
    assert _resolve_singer_rates(_m=1e-8, _r=None, _ratio=2.0, _mut_map=None, _recomb_map=None) == (1e-8, 2e-8)


def test_default_ratio_gives_r_equals_m():
    assert _resolve_singer_rates(_m=1.2e-8, _r=None, _ratio=None, _mut_map=None, _recomb_map=None) == (1.2e-8, 1.2e-8)


def test_explicit_r_overrides_ratio():
    assert _resolve_singer_rates(_m=1e-8, _r=5e-8, _ratio=99.0, _mut_map=None, _recomb_map=None)[1] == 5e-8


def test_recomb_map_skips_rate_derivation():
    assert _resolve_singer_rates(_m=1e-8, _r=None, _ratio=3.0, _mut_map=None, _recomb_map="rmap.txt") == (1e-8, None)


def test_missing_rate_raises():
    with pytest.raises(ValueError):
        _resolve_singer_rates(_m=None, _r=None, _ratio=1.0, _mut_map=None, _recomb_map=None)


# --- signatures: unified sampling knobs + underscore native flags -----------------------------

def test_signatures_expose_unified_and_native_flags():
    for fn in (singer, singer_windowed):
        params = inspect.signature(fn).parameters
        for p in ("ts", "mcmc_step", "mcmc_burnin",                     # unified sampling
                  "_Ne", "_m", "_r", "_ratio", "_n_samples", "_thin",   # native terminal flags
                  "_polar", "_ploidy", "_seed", "_recomb_map", "_mut_map"):
            assert p in params, (fn.__name__, p)
        for gone in ("Ne", "mutation_rate", "n_samples", "thin", "burn_in", "nr_samples",
                     "mcmc_sample_spacing"):                            # old names removed
            assert gone not in params, (fn.__name__, gone)
    for fn in (argweaver,):
        params = inspect.signature(fn).parameters
        for p in ("ts", "mcmc_step", "mcmc_burnin", "_N", "_m", "_r", "_iters", "_sample_step",
                  "_ntimes", "_maxtime", "_compress", "_seed"):
            assert p in params, ("argweaver", p)
    # the per-window primitive is native-only (no ts/mcmc_*; it returns raw indices)
    wp = inspect.signature(singer_window).parameters
    for p in ("_Ne", "_m", "_r", "_ratio", "_n_samples", "_thin", "_polar", "_ploidy", "_seed"):
        assert p in wp, p
    assert "ts" not in wp and "mcmc_step" not in wp


# --- Ne required ------------------------------------------------------------------------------

def test_singer_requires_ne():
    ts = tspaint.simulate_admixture(tspaint.sim.admixture_demography(), n_query=2, n_reference=2,
                                    sequence_length=2e4, random_seed=1).ts
    for call in (lambda: io.singer(ts, _m=1e-8),
                 lambda: io.singer_windowed(ts, window_size=1e4, _m=1e-8)):
        with pytest.raises(ValueError, match="requires Ne"):
            call()
    assert callable(io.estimate_ne)


def test_run_singer_passes_every_flag_as_is(monkeypatch, tmp_path):
    """_run_singer (the low-level engine, unchanged) builds the SINGER command from every flag."""
    import tspaint.io_singer as ios
    captured = {}

    class _Res:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Res()

    monkeypatch.setattr(ios.subprocess, "run", fake_run)
    monkeypatch.setattr(ios, "_singer_indices", lambda out: [])
    binp = tmp_path / "singer"; binp.write_text("")
    ios._run_singer("pre", "out", start=0, end=100, Ne=1000, mutation_rate=1e-8,
                    recombination_rate=2e-8, n_samples=5, thin=3, ploidy=2, seed=7, polar=0.9,
                    penalty=0.02, hmm_epsilon=0.1, psmc_bins=0.05, fast=True,
                    singer_args=["-resume"], singer_bin=str(binp))
    cmd = captured["cmd"]
    for flag in ("-Ne", "-m", "-r", "-polar", "-n", "-thin", "-ploidy", "-seed", "-penalty",
                 "-hmm_epsilon", "-psmc_bins", "-fast", "-resume"):
        assert flag in cmd, flag


def test_cli_singer_help_uses_unified_and_native_flags():
    from click.testing import CliRunner
    from tspaint.cli import cli
    out = CliRunner().invoke(cli, ["trees", "singer", "--help"]).output
    for tok in ("--ts", "--mcmc-step", "--mcmc-burnin", "-m", "--Ne", "--ratio", "-r", "--polar",
                "--recomb_map", "--mut_map", "--fast", "--singer-arg"):
        assert tok in out, tok
    for gone in ("--burnin", "--burn-in"):                             # old sampling flags removed
        assert gone not in out, gone


# --- the sampling resolvers: count formula + the plain/underscore precedence guard ------------

def test_singer_sampling_counts():
    # (n, thin, discard, keep) — SINGER writes n samples thin apart, drops discard, keeps keep.
    assert _singer_sampling(None, None, None, None, None) == (24, 50, 4, 20)       # defaults
    n, thin, discard, keep = _singer_sampling(20, 2, 200, None, None)              # spec example
    assert (n, thin, discard, keep) == (120, 2, 100, 20) and n * thin == 20 * 2 + 200
    assert _singer_sampling(4, 2, 4, None, None) == (6, 2, 2, 4)                   # ts=4
    assert _singer_sampling(None, None, None, 10, None)[:2] == (10, 50)            # _n_samples override


def test_argweaver_sampling_counts():
    # (iters, sample_step, discard, keep)
    iters, step, discard, keep = _argweaver_sampling(20, 2, 200, None, None)
    assert (iters, step, discard, keep) == (240, 2, 100, 20) and iters == 20 * 2 + 200
    assert _argweaver_sampling(None, None, None, None, None) == (1200, 50, 4, 20)  # defaults


def test_plain_and_underscore_conflict_raises():
    with pytest.raises(ValueError, match="precedence"):
        _singer_sampling(20, None, None, 10, None)          # ts + _n_samples
    with pytest.raises(ValueError, match="precedence"):
        _singer_sampling(None, 5, None, None, 3)            # mcmc_step + _thin
    with pytest.raises(ValueError, match="precedence"):
        _argweaver_sampling(20, None, None, 100, None)      # ts + _iters
    # and end-to-end through the public function
    ts = tspaint.simulate_admixture(tspaint.sim.admixture_demography(), n_query=2, n_reference=2,
                                    sequence_length=2e4, random_seed=1).ts
    with pytest.raises(ValueError, match="precedence"):
        io.singer(ts, ts=20, _n_samples=10, _Ne=1000, _m=1e-8)


# --- live SINGER runs: exact count via ts, single tree sequence when ts == 1 ------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_ts_count_live():
    ts_in = io.add_mutations(tspaint.simulate_admixture(
                             tspaint.sim.admixture_demography(Ne=1000, T_split=5000),
                             n_query=3, n_reference=3, sequence_length=4e4,
                             recombination_rate=1e-8, random_seed=7).ts,
                             rate=1.2e-8, random_seed=7)
    single = io.singer(ts_in, _Ne=1000, _m=1.2e-8, ts=1, _seed=7)
    assert single.num_samples == ts_in.num_samples                    # ts=1 -> a single tree sequence
    ens = io.singer(ts_in, _Ne=1000, _m=1.2e-8, ts=3, mcmc_step=2, mcmc_burnin=4, _seed=7)
    assert isinstance(ens, list) and len(ens) == 3
