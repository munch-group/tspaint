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
    """Read the tools manifest INI into ``{tool: {field: value}}``.

    Parameters
    ----------
    path : str, optional
        Manifest path. Default ``None`` — uses :func:`default_manifest_path`
        (``<repo>/external/tools.ini``).

    Returns
    -------
    dict[str, dict[str, str]]
        One entry per INI section: ``{tool-name: {field: value}}``. Fields include ``repo``,
        ``commit``, ``dir``, ``sparse``, ``pixi``, ``task``, ``build``, ``post`` and ``check``
        (consumed by :func:`plan`).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = path or default_manifest_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark tools manifest not found: {path}")
    cp = configparser.ConfigParser(interpolation=None)   # URLs/commands may contain '%', '$'
    cp.read(path)
    return {name: dict(cp[name]) for name in cp.sections()}


def _target(spec, name, tools_dir):
    return os.path.join(tools_dir, spec.get("dir", name))


def platform_tag():
    """This machine as a conda/pixi platform string (``osx-arm64``, ``linux-64``, …)."""
    import platform as _p

    sysname = {"Darwin": "osx", "Linux": "linux", "Windows": "win"}.get(_p.system(), _p.system().lower())
    machine = {"x86_64": "64", "AMD64": "64", "arm64": "arm64", "aarch64": "aarch64"}.get(
        _p.machine(), _p.machine())
    return f"{sysname}-{machine}"


def unsupported(spec, tag=None):
    """Why ``spec`` cannot be provisioned on this machine, or ``None`` if it can.

    A tool may declare ``platforms = <space-separated tags>`` in the manifest; if this machine's
    :func:`platform_tag` is not among them, the tool is **skipped, not failed** — it is a known,
    documented limitation of the upstream tool, not a broken install. Checked *before* any cloning,
    so we do not download a repo we cannot build.
    """
    plats = spec.get("platforms", "").split()
    if not plats:
        return None
    tag = tag or platform_tag()
    if tag in plats:
        return None
    note = spec.get("note", "").strip()
    reason = f"only supported on {', '.join(plats)} (this machine is {tag})"
    return f"{reason}. {note}" if note else reason


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

    ref = spec.get("commit", "HEAD")
    sparse = spec.get("sparse", "").split()
    steps = []
    if not os.path.isdir(os.path.join(target, ".git")):
        # Partial clone (no blobs) + cone sparse-checkout: these repos carry large in-repo data
        # and history (Recomb-Mix ~2 GB) we don't need — fetch only the code paths' blobs.
        steps.append(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repo"], target])
        if sparse:
            steps.append(["git", "-C", target, "sparse-checkout", "init", "--cone"])
            steps.append(["git", "-C", target, "sparse-checkout", "set"] + sparse)
        steps.append(["git", "-C", target, "checkout", ref])
    else:
        if sparse:
            steps.append(["git", "-C", target, "sparse-checkout", "set"] + sparse)
        steps.append(["git", "-C", target, "fetch", "--filter=blob:none", "origin", ref])
        steps.append(["git", "-C", target, "checkout", ref])
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
        res = subprocess.run(step[1], shell=True, cwd=step[2], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL)
    else:
        res = subprocess.run(step, capture_output=True, text=True, stdin=subprocess.DEVNULL)
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
        ``skipped`` / ``incomplete`` / ``failed`` / ``unknown``), ``path``, the ``steps``
        (formatted), and — for ``skipped`` — a ``reason``.

        ``skipped`` is **not** a failure: the tool declares a ``platforms`` list in the manifest that
        this machine is not on (see :func:`unsupported`), so nothing is cloned and nothing is built.
    """
    env_root = os.path.dirname(manifest or default_manifest_path())
    man = load_manifest(manifest)
    tools_dir = tools_dir or default_tools_dir()
    names = list(man) if tools is None else list(tools)
    tag = platform_tag()
    if not dry_run:
        os.makedirs(tools_dir, exist_ok=True)

    rows = []
    for name in names:
        if name not in man:
            log(f"  {name}: not in manifest")
            rows.append(dict(tool=name, status="unknown", path="", steps=[]))
            continue
        # Platform check FIRST: never clone a repo we already know we cannot build here.
        why = unsupported(man[name], tag)
        if why:
            log(f"  {name}: skipped — {why.split('. ')[0]}")     # terse here; full reason in the summary
            rows.append(dict(tool=name, status="skipped", path="", steps=[], reason=why))
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


def tool_status(manifest=None, tools_dir=None):
    """Resolved path + availability for every comparator — the four with bridges, plus the manifest.

    The four tools with VCF bridges (``rfmix``, ``gnomix``, ``salai``, ``recombmix``) resolve
    through :mod:`tspaint.benchmark._common`, which honours their ``$TSPAINT_*`` overrides. Every
    *other* tool in ``external/tools.ini`` is reported straight from the manifest, by testing its
    ``check`` path — so newly added comparators (MOSAIC, FLARE, Loter, GhostBuster) and the
    install-only tools (CLUES2, ARGformer) show up here without needing a bridge first.

    Returns
    -------
    list[dict]
        One row per tool: ``tool`` (name), ``path`` (resolved binary / install dir / check path),
        ``available`` (bool), and ``note`` — empty unless the tool is unavailable *by design* on
        this platform, in which case it says why (so a ``--`` in the table is never a mystery).
    """
    from . import _common as C
    rows = [dict(tool=t, path=p, available=C.tool_available(t), note="") for t, p in
            (("rfmix", C.RFMIX_BIN), ("gnomix", C.GNOMIX_DIR), ("salai", C.SALAI_DIR),
             ("recombmix", C.RECOMBMIX_BIN))]

    bridged = {r["tool"] for r in rows}
    tools_dir = tools_dir or default_tools_dir()
    tag = platform_tag()
    for name, spec in load_manifest(manifest).items():
        if name in bridged:
            continue
        target = _target(spec, name, tools_dir)
        check = os.path.join(target, spec["check"]) if "check" in spec else target
        binary = os.path.join(target, spec["binary"]) if "binary" in spec else target
        why = unsupported(spec, tag)
        rows.append(dict(tool=name, path=binary, available=os.path.exists(check),
                         note=why or ""))
    return rows
