"""Benchmark tool provisioning (tspaint.benchmark.setup): manifest, plan, dry-run, status.

Offline — never clones or builds. Exercises the committed manifest, the per-tool plan, the
idempotent already-provisioned short-circuit, and the dry-run (which must touch nothing).
"""
import os

from tspaint.benchmark import _provision as S
from tspaint.benchmark import _common as C


# Every comparator in external/tools.ini. `argformer` is the one entry with no tracked env recipe:
# it deliberately uses *upstream's own* pixi.toml (linux-64 + CUDA), because its MosaicML/composer
# dependency stack does not resolve in a conda env on macOS — see the manifest comment.
MANIFEST_TOOLS = {"gnomix", "salai", "recombmix", "ghostbuster", "mosaic", "flare", "loter",
                  "clues2", "argformer"}
NO_TRACKED_RECIPE = {"argformer"}


def test_committed_manifest_parses():
    man = S.load_manifest()                                  # external/tools.ini
    assert set(man) == MANIFEST_TOOLS
    for name, spec in man.items():
        assert spec["repo"].startswith("https://github.com/")
        assert len(spec["commit"]) == 40                     # pinned full SHA


def test_tracked_env_recipes_exist():
    # Each tool's pixi env recipe (our glue, not in upstream) is committed under external/envs/.
    man = S.load_manifest()
    env_root = os.path.dirname(S.default_manifest_path())
    for name, spec in man.items():
        if name in NO_TRACKED_RECIPE:
            assert "pixi" not in spec, f"{name}: expected to use upstream's own env recipe"
            continue
        recipe = os.path.join(env_root, spec["pixi"])
        assert os.path.exists(recipe), f"{name}: missing tracked env recipe {recipe}"
        assert "[dependencies]" in open(recipe).read()


def test_every_tool_checks_a_build_artifact_not_a_repo_file():
    # `check` gates idempotency AND what `status` reports, so it must name something produced by the
    # *last* provisioning step. If it named a repo file, a clone whose build or pip step had failed
    # would still report "installed" — which is exactly the bug this guards.
    man = S.load_manifest()
    for name, spec in man.items():
        assert "check" in spec, f"{name}: no check path"
        check = spec["check"]
        assert (check.startswith(".") or check.endswith((".jar", ".pth", "DESCRIPTION"))
                or check == spec.get("binary")), (
            f"{name}: check={check!r} looks like a repo file, not a build artifact")


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
    assert {r["tool"] for r in rows} == MANIFEST_TOOLS
    # Everything installable here is "planned"; a tool this machine's platform is not on is
    # "skipped" even in a dry run — there is no plan to show for something that cannot be built.
    for r in rows:
        if r["status"] == "skipped":
            assert r["reason"] and not r["steps"]
        else:
            assert r["status"] == "planned" and r["steps"]
    assert os.listdir(tmp_path) == []                        # dry-run created nothing
    assert any("git clone" in line for line in out)


def test_setup_subset_and_already(tmp_path):
    # Pre-create recombmix's check so it reports "already"; only request that one tool.
    (tmp_path / "Recomb-Mix").mkdir()
    (tmp_path / "Recomb-Mix" / "RecombMix_v0.7").write_text("")
    rows = S.setup(tools=["recombmix"], tools_dir=str(tmp_path), log=lambda *_: None)
    assert len(rows) == 1 and rows[0]["status"] == "already"


def test_tool_status_lists_bridged_tools_and_the_whole_manifest():
    # RFMix has no manifest entry (it is a bioconda dep in the `compare` pixi feature), and every
    # manifest tool must be reported whether or not it has a VCF bridge — otherwise a newly added
    # comparator is invisible to `tspaint benchmark status` until someone writes a runner for it.
    rows = {r["tool"]: r for r in S.tool_status()}
    assert set(rows) == MANIFEST_TOOLS | {"rfmix"}
    for r in rows.values():
        assert isinstance(r["available"], bool) and r["path"]


# --- platform skips are an outcome, not a failure ---------------------------------------------

def test_platform_tag_looks_like_a_conda_platform():
    tag = S.platform_tag()
    assert "-" in tag and tag.split("-")[0] in ("osx", "linux", "win")


def test_unsupported_is_none_when_no_platforms_declared():
    assert S.unsupported({"repo": "r"}) is None                       # no `platforms` -> universal
    assert S.unsupported({"platforms": "linux-64 osx-arm64"}, tag="osx-arm64") is None


def test_unsupported_explains_why_and_includes_the_note():
    why = S.unsupported({"platforms": "linux-64", "note": "Run it on the cluster."}, tag="osx-arm64")
    assert "linux-64" in why and "osx-arm64" in why and "Run it on the cluster." in why


def test_setup_skips_an_unsupported_tool_without_cloning(tmp_path):
    # The bug this guards: argformer declared its platform limit in a *shell guard*, so `install`
    # cloned the repo, ran the guard, and reported FAILED — making a known, documented limitation
    # look like a broken install (and leaving a useless clone behind). A platform mismatch must be
    # detected BEFORE any step runs.
    man = tmp_path / "tools.ini"
    man.write_text(
        "[nope]\n"
        "repo = https://example.invalid/nope\n"          # would fail loudly if we ever cloned it
        "commit = " + "0" * 40 + "\n"
        "platforms = some-other-platform\n"
        "note = Provision it elsewhere.\n"
        "check = .deps-ok\n")
    out = []
    rows = S.setup(tools_dir=str(tmp_path / "ext"), manifest=str(man), log=out.append)

    assert [r["status"] for r in rows] == ["skipped"]
    assert "Provision it elsewhere." in rows[0]["reason"]
    assert not (tmp_path / "ext" / "nope").exists()                   # nothing cloned
    assert not any("git clone" in line for line in out)               # and no step attempted


def test_argformer_declares_its_platform_in_the_manifest_not_a_shell_guard():
    spec = S.load_manifest()["argformer"]
    assert spec["platforms"] == "linux-64"
    assert spec.get("note")
    assert "uname" not in spec.get("build", "")                       # no shell guard
    assert S.unsupported(spec, tag="linux-64") is None
    assert "linux-64" in S.unsupported(spec, tag="osx-arm64")


def test_tool_status_explains_an_unavailable_by_design_tool():
    rows = {r["tool"]: r for r in S.tool_status()}
    if not rows["argformer"]["available"] and S.platform_tag() != "linux-64":
        assert "linux-64" in rows["argformer"]["note"]
    assert rows["flare"]["note"] == ""                                # universal tools carry no note
