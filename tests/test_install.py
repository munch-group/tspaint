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
