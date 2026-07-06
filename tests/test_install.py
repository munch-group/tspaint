"""`tspaint install singer` plumbing — offline (no clone/compile).

Covers the install-location resolution that the SINGER front end uses by default, the tracked
toolchain recipe, the pin, and that the CLI subcommand is registered. The actual clone + build is
exercised by running ``tspaint install singer`` (needs network + a C++ toolchain).
"""
import os

import tspaint.io_singer as io_singer
from tspaint.install import SINGER_COMMIT, SINGER_REPO, _env_recipe


def test_singer_install_paths_respect_tools_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TSPAINT_TOOLS_DIR", str(tmp_path))
    assert io_singer.singer_install_dir() == str(tmp_path / "SINGER")
    assert io_singer.singer_binary_path() == \
        str(tmp_path / "SINGER" / "SINGER" / "SINGER" / "singer")


def test_singer_binary_under_install_dir(monkeypatch):
    monkeypatch.delenv("TSPAINT_TOOLS_DIR", raising=False)
    assert io_singer.singer_binary_path().startswith(io_singer.singer_install_dir())
    assert io_singer.singer_binary_path().endswith(os.path.join("SINGER", "SINGER", "singer"))


def test_env_recipe_exists():
    # the tracked toolchain recipe install_singer drops into the clone
    assert os.path.exists(_env_recipe())


def test_singer_pin_is_full_sha():
    assert len(SINGER_COMMIT) == 40 and all(c in "0123456789abcdef" for c in SINGER_COMMIT)
    assert SINGER_REPO.endswith("/SINGER")


def test_install_singer_cli_registered():
    from tspaint.cli import cli
    assert "singer" in cli.commands["install"].commands


# --- `tspaint install relate` plumbing (relate_lib Convert + Relate + EstimatePopulationSize) ----

def test_relate_install_paths_respect_tools_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TSPAINT_TOOLS_DIR", str(tmp_path))
    import tspaint.io_relate as ir
    assert ir.relate_lib_install_dir() == str(tmp_path / "relate_lib")
    assert ir.relate_convert_path() == str(tmp_path / "relate_lib" / "bin" / "Convert")
    assert ir.relate_install_dir() == str(tmp_path / "relate")
    assert ir.relate_binary_path() == str(tmp_path / "relate" / "bin" / "Relate")
    assert ir.relate_file_formats_path() == str(tmp_path / "relate" / "bin" / "RelateFileFormats")
    assert ir.estimate_population_size_path() == str(
        tmp_path / "relate" / "scripts" / "EstimatePopulationSize" / "EstimatePopulationSize.sh")


def test_relate_default_convert_resolution(monkeypatch, tmp_path):
    import tspaint.io_relate as ir
    monkeypatch.setenv("TSPAINT_RELATE_CONVERT", "/custom/Convert")   # env override wins
    assert ir._default_convert() == "/custom/Convert"
    monkeypatch.delenv("TSPAINT_RELATE_CONVERT", raising=False)
    monkeypatch.setenv("TSPAINT_TOOLS_DIR", str(tmp_path))            # no built binary -> PATH lookup
    assert ir._default_convert() == "Convert"
    built = tmp_path / "relate_lib" / "bin"                           # built binary present -> use it
    built.mkdir(parents=True)
    (built / "Convert").write_text("")
    assert ir._default_convert() == str(built / "Convert")


def test_relate_pins_are_full_shas_and_recipe_exists():
    from tspaint.install import (RELATE_COMMIT, RELATE_LIB_COMMIT, RELATE_REPO, RELATE_LIB_REPO,
                                 _relate_env_recipe)
    for sha in (RELATE_COMMIT, RELATE_LIB_COMMIT):
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    assert RELATE_REPO.endswith("/relate") and RELATE_LIB_REPO.endswith("/relate_lib")
    assert os.path.exists(_relate_env_recipe())


def test_install_relate_cli_registered():
    from tspaint.cli import cli
    assert "relate" in cli.commands["install"].commands
