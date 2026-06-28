"""Provision the git-only benchmark comparators from a pinned manifest (CLAUDE.md §9).

gnomix, SALAI-Net and Recomb-Mix are not on conda/PyPI: each is a GitHub repo cloned and built
into its own environment. :func:`setup` reads ``external/tools.ini`` (repo + pinned commit +
build/extract steps per tool), clones each at its pin into ``external/<dir>`` (or
``$TSPAINT_TOOLS_DIR``), solves its env (``pixi install``), runs any build/post step, and verifies
a ``check`` path — the same locations the bridges resolve by default. RFMix is excluded (it is a
bioconda package in the ``compare`` pixi feature).

The manifest is a plain INI (``configparser`` — stdlib, no new dependency) so it is editable
without touching code and adds a fourth tool in a few lines.
"""
from __future__ import annotations

import configparser
import os
import subprocess

from ._common import _REPO_ROOT, _PIXI

__all__ = ["default_manifest_path", "default_tools_dir", "load_manifest", "plan", "setup",
           "tool_status"]


def default_manifest_path():
    """Path to the committed manifest (``<repo>/external/tools.ini``)."""
    return os.path.join(_REPO_ROOT, "external", "tools.ini")


def default_tools_dir():
    """Clone root: ``$TSPAINT_TOOLS_DIR`` or ``<repo>/external`` (matches the bridge defaults)."""
    return os.path.expanduser(
        os.environ.get("TSPAINT_TOOLS_DIR", os.path.join(_REPO_ROOT, "external")))


def load_manifest(path=None):
    """Read the tools manifest INI into ``{tool: {field: value}}``."""
    path = path or default_manifest_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark tools manifest not found: {path}")
    cp = configparser.ConfigParser(interpolation=None)   # URLs/commands may contain '%', '$'
    cp.read(path)
    return {name: dict(cp[name]) for name in cp.sections()}


def _target(spec, name, tools_dir):
    return os.path.join(tools_dir, spec.get("dir", name))


def plan(name, spec, tools_dir, *, env_root=None, force=False):
    """Provisioning plan for one tool → ``(target_dir, check_path, steps)``.

    ``steps`` is empty when the ``check`` path already exists (idempotent). Each step is an argv
    ``list``, a ``("copy", src, dst)`` tuple (drop the tracked env recipe into the clone), or a
    ``("shell", command, cwd)`` tuple (``post``).

    The ``pixi`` field names a tracked env recipe (relative to ``env_root``, the manifest's
    directory) copied into the clone — these recipes are our own glue and are **not** in the
    upstream repos — then ``pixi install``-ed; ``task`` names pixi tasks (e.g. a build) run via
    ``pixi run`` against that recipe.
    """
    env_root = env_root or os.path.dirname(default_manifest_path())
    target = _target(spec, name, tools_dir)
    check_path = os.path.join(target, spec["check"]) if "check" in spec else target
    if os.path.exists(check_path) and not force:
        return target, check_path, []

    steps = []
    if not os.path.isdir(os.path.join(target, ".git")):
        steps.append(["git", "clone", spec["repo"], target])
        if spec.get("commit"):
            steps.append(["git", "-C", target, "checkout", spec["commit"]])
    elif spec.get("commit"):
        steps.append(["git", "-C", target, "fetch", "--all", "--tags"])
        steps.append(["git", "-C", target, "checkout", spec["commit"]])
    if spec.get("pixi"):
        steps.append(("copy", os.path.join(env_root, spec["pixi"]),
                      os.path.join(target, "pixi.toml")))
        steps.append([_PIXI, "install", "--manifest-path", target])
    for task in (t.strip() for t in spec.get("task", "").split(",") if t.strip()):
        steps.append([_PIXI, "run", "--manifest-path", target, task])
    if spec.get("build"):
        steps.append(("shell", spec["build"], target))
    if spec.get("post"):
        steps.append(("shell", spec["post"], target))
    return target, check_path, steps


def _fmt(step):
    if isinstance(step, tuple):
        return f"copy {step[1]} -> {step[2]}" if step[0] == "copy" else f"(cd {step[2]} && {step[1]})"
    return " ".join(step)


def _run(step):
    if isinstance(step, tuple) and step[0] == "copy":
        import shutil
        if not os.path.exists(step[1]):
            raise RuntimeError(f"step failed: {_fmt(step)}\n  tracked env recipe not found")
        shutil.copyfile(step[1], step[2])
        return
    if isinstance(step, tuple):
        res = subprocess.run(step[1], shell=True, cwd=step[2], capture_output=True, text=True)
    else:
        res = subprocess.run(step, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"step failed: {_fmt(step)}\n{res.stderr[-1500:]}")


def setup(tools=None, *, tools_dir=None, manifest=None, force=False, dry_run=False, log=print):
    """Clone + build the git-only comparators listed in the manifest.

    Parameters
    ----------
    tools : iterable[str], optional
        Subset of tool names to provision; default all in the manifest.
    tools_dir : str, optional
        Clone root (default :func:`default_tools_dir`).
    manifest : str, optional
        Manifest path (default :func:`default_manifest_path`).
    force : bool, optional
        Re-provision even if the ``check`` path already exists.
    dry_run : bool, optional
        Print the plan and make no changes.
    log : callable, optional
        Progress sink (default ``print``).

    Returns
    -------
    list[dict]
        One row per tool: ``tool``, ``status`` (``already`` / ``planned`` / ``provisioned`` /
        ``incomplete`` / ``failed`` / ``unknown``), ``path`` and the ``steps`` (formatted).
    """
    env_root = os.path.dirname(manifest or default_manifest_path())
    man = load_manifest(manifest)
    tools_dir = tools_dir or default_tools_dir()
    names = list(man) if tools is None else list(tools)
    if not dry_run:
        os.makedirs(tools_dir, exist_ok=True)

    rows = []
    for name in names:
        if name not in man:
            log(f"  {name}: not in manifest")
            rows.append(dict(tool=name, status="unknown", path="", steps=[]))
            continue
        target, check_path, steps = plan(name, man[name], tools_dir, env_root=env_root, force=force)
        fsteps = [_fmt(s) for s in steps]
        if not steps:
            log(f"  {name}: already provisioned -> {target}")
            rows.append(dict(tool=name, status="already", path=target, steps=[]))
            continue
        if dry_run:
            log(f"  {name}: plan ->")
            for s in fsteps:
                log(f"      {s}")
            rows.append(dict(tool=name, status="planned", path=target, steps=fsteps))
            continue
        try:
            for s in steps:
                log(f"  {name}: {_fmt(s)}")
                _run(s)
            status = "provisioned" if os.path.exists(check_path) else "incomplete"
        except RuntimeError as e:
            log(f"  {name}: FAILED — {e}")
            status = "failed"
        log(f"  {name}: {status} -> {target}")
        rows.append(dict(tool=name, status=status, path=target, steps=fsteps))
    return rows


def tool_status():
    """Resolved path + availability for every benchmark tool (incl. RFMix)."""
    from . import _common as C
    paths = {"rfmix": C.RFMIX_BIN, "gnomix": C.GNOMIX_DIR, "salai": C.SALAI_DIR,
             "recombmix": C.RECOMBMIX_BIN}
    return [dict(tool=t, path=p, available=C.tool_available(t)) for t, p in paths.items()]
