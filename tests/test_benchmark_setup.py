"""Benchmark tool provisioning (tspaint.benchmark.setup): manifest, plan, dry-run, status.

Offline — never clones or builds. Exercises the committed manifest, the per-tool plan, the
idempotent already-provisioned short-circuit, and the dry-run (which must touch nothing).
"""
import os

from tspaint.benchmark import _provision as S
from tspaint.benchmark import _common as C


def test_committed_manifest_parses():
    man = S.load_manifest()                                  # external/tools.ini
    assert set(man) == {"gnomix", "salai", "recombmix"}
    for name, spec in man.items():
        assert spec["repo"].startswith("https://github.com/")
        assert len(spec["commit"]) == 40                     # pinned full SHA


def test_tracked_env_recipes_exist():
    # Each tool's pixi env recipe (our glue, not in upstream) is committed under external/envs/.
    man = S.load_manifest()
    env_root = os.path.dirname(S.default_manifest_path())
    for name, spec in man.items():
        recipe = os.path.join(env_root, spec["pixi"])
        assert os.path.exists(recipe), f"{name}: missing tracked env recipe {recipe}"
        assert "[dependencies]" in open(recipe).read()


def test_manifest_dirs_match_bridge_defaults():
    # The manifest's <dir>/<binary> must resolve to the paths the bridges look for.
    man = S.load_manifest()
    root = S.default_tools_dir()
    assert os.path.join(root, man["gnomix"]["dir"]) == C.GNOMIX_DIR
    assert os.path.join(root, man["salai"]["dir"]) == C.SALAI_DIR
    assert os.path.join(root, man["recombmix"]["dir"], man["recombmix"]["binary"]) == C.RECOMBMIX_BIN


def test_plan_fresh_clone_copies_recipe_and_installs(tmp_path):
    spec = {"repo": "https://example.com/x", "commit": "a" * 40, "dir": "x",
            "pixi": "envs/x/pixi.toml", "check": "x.py"}
    target, check, steps = S.plan("x", spec, str(tmp_path), env_root="/REPO/external")
    assert target == str(tmp_path / "x")
    assert check == str(tmp_path / "x" / "x.py")
    # partial clone (no blobs) avoids pulling these repos' large in-repo data + history
    assert steps[0][:2] == ["git", "clone"]
    assert "--filter=blob:none" in steps[0] and spec["repo"] in steps[0]
    assert ["git", "-C", target, "checkout", "a" * 40] in steps
    # the tracked env recipe is copied into the clone, then installed
    assert ("copy", "/REPO/external/envs/x/pixi.toml", os.path.join(target, "pixi.toml")) in steps
    assert [C._PIXI, "install", "--manifest-path", target] in steps


def test_plan_sparse_checkout(tmp_path):
    spec = {"repo": "r", "commit": "a" * 40, "dir": "x", "sparse": "src configs", "check": "x.py"}
    target, _c, steps = S.plan("x", spec, str(tmp_path))
    assert ["git", "-C", target, "sparse-checkout", "init", "--cone"] in steps
    assert ["git", "-C", target, "sparse-checkout", "set", "src", "configs"] in steps
    # sparse is configured before the checkout that populates the tree
    assert steps.index(["git", "-C", target, "sparse-checkout", "set", "src", "configs"]) < \
        steps.index(["git", "-C", target, "checkout", "a" * 40])


def test_plan_runs_pixi_build_task(tmp_path):
    spec = {"repo": "r", "dir": "x", "pixi": "envs/x/pixi.toml", "task": "build", "check": "bin"}
    target, _c, steps = S.plan("x", spec, str(tmp_path), env_root="/e")
    assert [C._PIXI, "run", "--manifest-path", target, "build"] in steps
    # tasks run after the env is installed
    assert steps.index([C._PIXI, "run", "--manifest-path", target, "build"]) > \
        steps.index([C._PIXI, "install", "--manifest-path", target])


def test_plan_already_provisioned(tmp_path):
    spec = {"repo": "r", "commit": "b" * 40, "dir": "x", "check": "bin"}
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "bin").write_text("")                  # check path exists
    _target, _check, steps = S.plan("x", spec, str(tmp_path))
    assert steps == []                                       # idempotent: nothing to do


def test_plan_post_step(tmp_path):
    spec = {"repo": "r", "dir": "x", "check": "bin", "post": "tar xzf m.tgz"}
    _t, _c, steps = S.plan("x", spec, str(tmp_path))
    assert ("shell", "tar xzf m.tgz", str(tmp_path / "x")) in steps


def test_setup_dry_run_touches_nothing(tmp_path):
    out = []
    rows = S.setup(tools_dir=str(tmp_path), dry_run=True, log=out.append)
    assert {r["tool"] for r in rows} == {"gnomix", "salai", "recombmix"}
    assert all(r["status"] == "planned" and r["steps"] for r in rows)
    assert os.listdir(tmp_path) == []                        # dry-run created nothing
    assert any("git clone" in line for line in out)


def test_setup_subset_and_already(tmp_path):
    # Pre-create recombmix's check so it reports "already"; only request that one tool.
    (tmp_path / "Recomb-Mix").mkdir()
    (tmp_path / "Recomb-Mix" / "RecombMix_v0.7").write_text("")
    rows = S.setup(tools=["recombmix"], tools_dir=str(tmp_path), log=lambda *_: None)
    assert len(rows) == 1 and rows[0]["status"] == "already"


def test_tool_status_lists_all_four():
    rows = {r["tool"]: r for r in S.tool_status()}
    assert set(rows) == {"rfmix", "gnomix", "salai", "recombmix"}
    for r in rows.values():
        assert isinstance(r["available"], bool) and r["path"]
