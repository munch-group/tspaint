"""Build the SINGER ARG sampler from source into the tspaint tree (CLAUDE.md §7.4).

``tspaint install singer`` clones popgenmethods/SINGER at a pinned commit and builds the ``singer``
binary with a pixi-provided C++ toolchain (reproducible — no system compiler needed), leaving it
where tspaint's SINGER front end (:mod:`tspaint.io_singer`) looks by default. It works on both
``linux-64`` and ``osx-arm64``: the compile is upstream's ``$CXX -std=c++17 -O3 *.cpp`` glob (the
mac recipe documented in ``build_singer.md``, generalised to Linux), linked dynamically with an
rpath into the pixi env — no ``-static`` (Apple's linker rejects it and conda's gcc handles it
poorly; the rpath keeps the binary working as long as the build env directory survives, like the
other ``external/`` tools).
"""
from __future__ import annotations

import os
import shutil
import subprocess

from .io_singer import singer_binary_path, singer_install_dir, repo_root

__all__ = ["install_singer", "SINGER_REPO", "SINGER_COMMIT",
           "install_argweaver", "ARGWEAVER_REPO", "ARGWEAVER_COMMIT"]

#: ARGweaver source (mdrasmus/argweaver). Only the C++ ``arg-sample`` binary is built (``make``);
#: the Python-2 ``make install`` step is skipped. Override via $TSPAINT_ARGWEAVER_REPO / _COMMIT.
ARGWEAVER_REPO = os.environ.get("TSPAINT_ARGWEAVER_REPO", "https://github.com/mdrasmus/argweaver")
ARGWEAVER_COMMIT = os.environ.get("TSPAINT_ARGWEAVER_COMMIT", "master")

# TEMPORARY: upstream SINGER (popgenmethods/SINGER @ f88d687, v0.1.9) has a node-write-state
# use-after-free that corrupts the heap and SIGSEGVs on ARGs above ~1 Mb. The fix lives on this
# fork branch; revert to "https://github.com/popgenmethods/SINGER" @ f88d687 once it is merged
# upstream. Override either without editing via $TSPAINT_SINGER_REPO / $TSPAINT_SINGER_COMMIT.
SINGER_REPO = os.environ.get("TSPAINT_SINGER_REPO", "https://github.com/munch-group/SINGER")
SINGER_COMMIT = os.environ.get(
    "TSPAINT_SINGER_COMMIT", "2316f5932acb19eb49773e1f8cd19500df88ec37")  # fork fix branch

_PIXI = os.environ.get("TSPAINT_PIXI", "pixi")


def _env_recipe():
    """The tracked pixi toolchain recipe dropped into the clone (``external/envs/SINGER``).

    Resolved via :func:`tspaint.io_singer.repo_root` so it works whether tspaint is installed
    editable or as a non-editable copy in site-packages (run from the repo either way).
    """
    return os.path.join(repo_root(), "external", "envs", "SINGER", "pixi.toml")


def _run(cmd, *, cwd=None, log):
    log("  " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        shown = cmd if isinstance(cmd, str) else " ".join(cmd)
        raise RuntimeError(f"step failed (exit {res.returncode}): {shown}\n"
                           f"{(res.stderr or res.stdout)[-1500:]}")
    return res


def install_singer(*, commit=None, force=False, tools_dir=None, log=print):
    """Clone + build SINGER where tspaint finds it; return the ``singer`` binary path.

    Parameters
    ----------
    commit : str, optional
        SINGER commit to pin (default :data:`SINGER_COMMIT`).
    force : bool, optional
        Rebuild even if the binary already exists.
    tools_dir : str, optional
        Override the clone root (default: ``$TSPAINT_TOOLS_DIR`` or ``<repo>/external``).
    log : callable, optional
        Progress sink (default ``print``).

    Returns
    -------
    str
        Path to the built ``singer`` binary (:func:`tspaint.io_singer.singer_binary_path`).
    """
    commit = commit or SINGER_COMMIT
    target = singer_install_dir() if tools_dir is None else os.path.join(tools_dir, "SINGER")
    binary = (singer_binary_path() if tools_dir is None
              else os.path.join(target, "SINGER", "SINGER", "singer"))
    if os.path.exists(binary) and not force:
        log(f"singer already built: {binary} (use --force to rebuild)")
        return binary

    recipe = _env_recipe()
    if not os.path.exists(recipe):
        raise FileNotFoundError(f"SINGER pixi toolchain recipe not found: {recipe}")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # 1) partial + sparse clone of just the source (the repo also commits large release binaries
    #    under releases/ that we don't need).
    if not os.path.isdir(os.path.join(target, ".git")):
        log(f"cloning SINGER @ {commit[:10]} -> {target}")
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", SINGER_REPO, target], log=log)
        _run(["git", "-C", target, "sparse-checkout", "init", "--cone"], log=log)
    else:
        # an existing clone may point at a different repo (e.g. after switching SINGER_REPO);
        # retarget origin so the commit below is fetched from the right place.
        _run(["git", "-C", target, "remote", "set-url", "origin", SINGER_REPO], log=log)
    _run(["git", "-C", target, "sparse-checkout", "set", "SINGER"], log=log)
    _run(["git", "-C", target, "fetch", "--filter=blob:none", "origin", commit], log=log)
    _run(["git", "-C", target, "checkout", commit], log=log)

    # 2) drop in the toolchain recipe (our glue; not upstream) and solve the env.
    shutil.copyfile(recipe, os.path.join(target, "pixi.toml"))
    log("solving build env (pixi install)")
    _run([_PIXI, "install", "--manifest-path", target], log=log)

    # 3) compile: one C++17 glob, dynamic-linked with an rpath into the env (mac + linux). Run via
    #    bash so the *.cpp glob and $CXX/$CONDA_PREFIX (set by the pixi env activation) expand.
    compile_cmd = ("$CXX -std=c++17 -O3 -g SINGER/SINGER/*.cpp "
                   "-o SINGER/SINGER/singer -Wl,-rpath,$CONDA_PREFIX/lib")
    log("compiling singer (a couple of minutes)")
    _run([_PIXI, "run", "--manifest-path", target, "bash", "-c", compile_cmd], cwd=target, log=log)

    # 4) verify: the binary exists, is executable, and behaves like singer (no args -> complains
    #    about the missing -r flag and exits non-zero).
    if not os.path.exists(binary):
        raise RuntimeError(f"compile reported success but {binary} is missing")
    os.chmod(binary, 0o755)
    chk = subprocess.run([binary], capture_output=True, text=True)
    if "flag" not in (chk.stdout + chk.stderr).lower():
        raise RuntimeError("built binary did not behave like singer:\n"
                           f"{(chk.stderr or chk.stdout)[-500:]}")
    log(f"singer built: {binary}")
    return binary


def install_argweaver(*, commit=None, force=False, tools_dir=None, log=print):
    """Clone + build ARGweaver's ``arg-sample`` where tspaint finds it; return the binary path.

    Runs the project's own ``make`` (only the C++ sampler is built — the Python-2 ``make install``
    step is skipped, so no Python-2 runtime is needed). Requires a C++ compiler and ``make`` on
    PATH. Override the source via ``$TSPAINT_ARGWEAVER_REPO`` / ``_COMMIT``; relocate the clone with
    ``$TSPAINT_TOOLS_DIR``. tspaint's ARGweaver front end (:mod:`tspaint.io_argweaver`) looks here by
    default (or at ``$TSPAINT_ARGWEAVER``).

    Parameters
    ----------
    commit : str, optional
        ARGweaver commit / ref to build (default :data:`ARGWEAVER_COMMIT`, i.e. ``master``).
    force : bool, optional
        Rebuild even if the binary already exists.
    tools_dir : str, optional
        Override the clone root (default: ``$TSPAINT_TOOLS_DIR`` or ``<repo>/external``).

    Returns
    -------
    str
        Path to the built ``arg-sample`` binary.
    """
    from .io_argweaver import argweaver_binary_path, argweaver_install_dir
    commit = commit or ARGWEAVER_COMMIT
    target = argweaver_install_dir() if tools_dir is None else os.path.join(tools_dir, "argweaver")
    binary = (argweaver_binary_path() if tools_dir is None
              else os.path.join(target, "bin", "arg-sample"))
    if os.path.exists(binary) and not force:
        log(f"argweaver already built: {binary} (use --force to rebuild)")
        return binary
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.exists(os.path.join(target, ".git")):
        log(f"cloning argweaver @ {commit} -> {target}")
        _run(["git", "clone", ARGWEAVER_REPO, target], log=log)
    else:
        _run(["git", "-C", target, "fetch", "origin", commit], log=log)
    _run(["git", "-C", target, "checkout", commit], log=log)
    log("building arg-sample (make)")
    _run(["make", "-C", target], log=log)
    if not os.path.exists(binary):
        raise RuntimeError(
            f"make reported success but {binary} is missing — ARGweaver needs a C++ compiler and "
            f"make; see https://mdrasmus.github.io/argweaver/")
    os.chmod(binary, 0o755)
    log(f"argweaver built: {binary}")
    return binary
