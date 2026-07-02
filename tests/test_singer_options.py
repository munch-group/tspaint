"""io.singer options mirror SINGER's own CLI flags, so recommendations from the SINGER authors or
other SINGER users apply directly (``-m``/``-r``/``-ratio``/``-n``/``-thin``/``-polar``/…), with the
descriptive tspaint names kept as aliases and a ``singer_args`` passthrough for anything else."""
import inspect
import os

import pytest

import tspaint
from tspaint import io
from tspaint.io_singer import (_resolve_singer_rates, singer, singer_window, singer_windowed,
                               DEFAULT_SINGER)


# --- rate resolution: -ratio (default 1) -> r = m*ratio; -m/-r aliases -----------------------

def test_ratio_derives_recombination_rate():
    assert _resolve_singer_rates(mutation_rate=1e-8, m=None, recombination_rate=None, r=None,
                                 ratio=2.0, mut_map=None, recomb_map=None) == (1e-8, 2e-8)


def test_default_ratio_one_gives_r_equals_m():
    assert _resolve_singer_rates(mutation_rate=1.2e-8, m=None, recombination_rate=None, r=None,
                                 ratio=1.0, mut_map=None, recomb_map=None) == (1.2e-8, 1.2e-8)


def test_singer_flag_aliases_win_over_descriptive():
    assert _resolve_singer_rates(mutation_rate=9e-9, m=1e-8, recombination_rate=9e-9, r=2e-8,
                                 ratio=1.0, mut_map=None, recomb_map=None) == (1e-8, 2e-8)


def test_explicit_r_overrides_ratio():
    assert _resolve_singer_rates(mutation_rate=1e-8, m=None, recombination_rate=5e-8, r=None,
                                 ratio=99.0, mut_map=None, recomb_map=None)[1] == 5e-8


def test_recomb_map_skips_rate_derivation():
    assert _resolve_singer_rates(mutation_rate=1e-8, m=None, recombination_rate=None, r=None,
                                 ratio=3.0, mut_map=None, recomb_map="rmap.txt") == (1e-8, None)


def test_missing_rate_raises():
    with pytest.raises(ValueError):
        _resolve_singer_rates(mutation_rate=None, m=None, recombination_rate=None, r=None,
                              ratio=1.0, mut_map=None, recomb_map=None)


# --- the SINGER flags are all exposed as kwargs (plus the descriptive aliases) ---------------

def test_signatures_expose_singer_flags():
    singer_flags = ("m", "r", "n", "ratio", "polar", "burnin", "recomb_map", "mut_map",
                    "penalty", "hmm_epsilon", "psmc_bins", "fast", "singer_args")
    aliases = ("mutation_rate", "recombination_rate", "n_samples", "burn_in")
    for fn in (singer, singer_windowed):
        params = inspect.signature(fn).parameters
        for flag in singer_flags:
            assert flag in params, (fn.__name__, flag)
        for old in aliases:
            assert old in params, (fn.__name__, old)
    # the per-window primitive exposes the run-time flags (no burnin: it returns raw indices)
    wp = inspect.signature(singer_window).parameters
    for flag in ("m", "r", "n", "ratio", "polar", "recomb_map", "mut_map", "singer_args"):
        assert flag in wp, flag


# --- soft_refs are excluded from the Ne estimate --------------------------------------------

def test_singer_exposes_soft_refs():
    for fn in (singer, singer_windowed):
        assert "soft_refs" in inspect.signature(fn).parameters, fn.__name__


def test_singer_soft_refs_forwarded_to_ne_exclude(monkeypatch):
    """singer(Ne=None, soft_refs=...) forwards soft_refs as estimate_ne's exclude (no SINGER run)."""
    import tspaint.io_genotypes as iog
    captured = {}

    def fake_estimate_ne(source, mutation_rate, groups=None, exclude=None):
        captured["groups"], captured["exclude"] = groups, exclude
        return 12345.0

    monkeypatch.setattr(iog, "estimate_ne", fake_estimate_ne)
    ts = tspaint.simulate_admixture(n_admix=2, n_ref=2, sequence_length=2e4, random_seed=1)
    with pytest.raises(FileNotFoundError):            # fails at the (missing) binary, after Ne estimate
        io.singer(ts, m=1e-8, labels={0: 0, 1: 0}, soft_refs={1}, singer_bin="/no/such/singer")
    assert captured["exclude"] == {1} and captured["groups"] == {0: 0, 1: 0}


# --- CLI mirrors SINGER's flags: short single-dash, long two-dash; old names kept ------------

def test_cli_singer_help_uses_singer_flag_names():
    from click.testing import CliRunner
    from tspaint.cli import cli
    out = CliRunner().invoke(cli, ["trees", "singer", "--help"]).output
    for tok in ("-m", "--Ne", "--ratio", "-r", "-n", "--thin", "--burnin", "--polar",
                "--recomb_map", "--mut_map", "--fast", "--singer-arg"):
        assert tok in out, tok
    for old in ("--mut-rate", "--recomb-rate", "--burn-in"):     # aliases retained
        assert old in out, old


# --- live SINGER run driven entirely by SINGER-flag names ------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(DEFAULT_SINGER), reason="SINGER binary not available")
def test_singer_runs_with_singer_flag_aliases():
    ts = io.add_mutations(tspaint.simulate_admixture(n_admix=3, n_ref=3, sequence_length=1e5,
                          recombination_rate=1e-8, random_seed=7, Ne=1000, T_admix=30,
                          T_split=5000, f_A=0.5), rate=1.2e-8, random_seed=7)
    ens = io.singer(ts, m=1.2e-8, ratio=1.0, Ne=1000, n=6, thin=2, burnin=2, polar=0.5, seed=42)
    ens = ens if isinstance(ens, list) else [ens]
    assert 1 <= len(ens) <= 4 and ens[0].num_samples == ts.num_samples
