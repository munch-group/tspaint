"""Repo-root / recipe resolution is robust to editable vs non-editable installs.

The ``external/`` paths (SINGER build recipe, benchmark tool clones) must resolve to the repo
whether tspaint is installed editable or as a non-editable copy in ``site-packages`` (which a
``pip install .`` can create, clobbering the editable ``.pth``). Otherwise ``tspaint install
singer`` / ``benchmark setup`` resolve a bogus path inside the env (the reported failure).
"""
import os

import tspaint.io_singer as ios
from tspaint.install import _env_recipe


def test_repo_root_finds_marker_and_recipe():
    r = ios.repo_root()
    assert os.path.exists(os.path.join(r, "external", "tools.ini"))    # the repo marker
    assert os.path.exists(_env_recipe())                              # SINGER build recipe resolves


def test_repo_root_cwd_fallback_when_file_is_non_editable(tmp_path, monkeypatch):
    # Simulate a non-editable install: io_singer.__file__ sits where no marker exists up-tree,
    # but the process runs from the repo -> repo_root() must still find the repo via the cwd search.
    repo = ios.repo_root()
    fake = tmp_path / "env" / "site-packages" / "tspaint" / "io_singer.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("")
    monkeypatch.setattr(ios, "__file__", str(fake))
    monkeypatch.chdir(repo)
    assert ios.repo_root() == os.path.abspath(repo)                   # found via cwd, not __file__


def test_tools_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TSPAINT_TOOLS_DIR", str(tmp_path / "tools"))
    assert ios._tools_dir() == str(tmp_path / "tools")
    assert ios.singer_install_dir() == str(tmp_path / "tools" / "SINGER")
