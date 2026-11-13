import os
import re
import subprocess

import pytest

from hub import init
from hub.config import load_config
from hub.init import TEMPLATES, init_vault
from hub.paths import ROOT_REF_RE
from hub.store import Store
from hub.validate import validate


def test_init_creates_skeleton_and_vendors(tmp_path):
    v = tmp_path / "vault"
    created = init_vault(v, name="demo")
    for rel in ["project.yaml", "README.md", "Vision.md", "Locator.md", "Knowledge/README.md", "Figures/Gallery.md",
                "Status/Items.base", ".gitignore", ".gitattributes", "tools/setup.sh",
                "tools/project-hub/pyproject.toml", "tools/project-hub/hub/cli.py",
                "tools/project-hub/USER-GUIDE.md", ".claude/skills/hub-status/SKILL.md",
                ".claude/skills/hub-agenda/SKILL.md"]:
        assert (v / rel).exists(), rel
    assert "project.yaml" in created
    assert not (v / "tools" / "project-hub" / "tests").exists()
    assert not list((v / "tools").rglob("__pycache__"))
    assert os.access(v / "tools" / "setup.sh", os.X_OK)
    assert subprocess.run(["bash", "-n", str(v / "tools" / "setup.sh")]).returncode == 0
    cfg = load_config(v)
    assert cfg.name == "demo" and cfg.people == {} and cfg.series == {}
    assert validate(Store(v), cfg) == []
    assert "tools/.venv/" in (v / ".gitignore").read_text() or ".venv/" in (v / ".gitignore").read_text()


def test_init_is_idempotent_and_keeps_human_files(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    (v / "Vision.md").write_text("# Our vision\nEdited.\n")
    (v / "project.yaml").write_text((v / "project.yaml").read_text().replace("name: demo", "name: renamed"))
    (v / "tools" / "project-hub" / "hub" / "cli.py").write_text("HUB_TEST_SENTINEL_9F3\n")
    created = init_vault(v, name="demo")
    assert (v / "Vision.md").read_text() == "# Our vision\nEdited.\n"
    assert load_config(v).name == "renamed"
    assert "HUB_TEST_SENTINEL_9F3" not in (v / "tools" / "project-hub" / "hub" / "cli.py").read_text()
    assert "Vision.md" not in created and "project.yaml" not in created


def test_init_registers_merge_drivers_in_a_git_repo(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=v, check=True)
    init_vault(v, name="demo")
    out = subprocess.run(["git", "config", "merge.hub-item.driver"], cwd=v, capture_output=True, text=True).stdout
    assert "hub.merge_item" in out


def test_templates_are_generic(tmp_path):
    text = "\n".join([*TEMPLATES.values(), init.PROJECT_TEMPLATE, init.SETUP_SH])
    assert "/home/" not in text and "/Users/" not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), "templates must not hold email addresses"
    init_vault(tmp_path / "v", name="My project")
    assert load_config(tmp_path / "v").name == "My project"


def test_fresh_vault_has_no_broken_links(tmp_path):
    from hub.links import check_links
    v = tmp_path / "v"
    init_vault(v, name="demo")
    assert [r for r in check_links(load_config(v)) if r["ok"] is False] == []
    for rel, body in TEMPLATES.items():
        assert not ROOT_REF_RE.search(body), rel


@pytest.mark.parametrize("name", ["Demo: project", "#x", "yes", "123", "a 'quoted' \"name\"", "- item"])
def test_project_name_round_trips(tmp_path, name):
    init_vault(tmp_path / "v", name=name)
    assert load_config(tmp_path / "v").name == name


@pytest.mark.parametrize("name", ["demo\npeople: oops", "demo\r", "a\x00b", "tab\there"])
def test_project_name_rejects_control_characters(tmp_path, name):
    with pytest.raises(ValueError, match="name"):
        init_vault(tmp_path / "v", name=name)
    assert not (tmp_path / "v" / "project.yaml").exists()


def test_stale_hub_skills_are_removed_but_others_kept(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    skills = v / ".claude" / "skills"
    (skills / "hub-retired").mkdir()
    (skills / "hub-retired" / "SKILL.md").write_text("old\n")
    (skills / "mine").mkdir()
    (skills / "mine" / "SKILL.md").write_text("mine\n")
    init_vault(v, name="demo")
    assert not (skills / "hub-retired").exists()
    assert (skills / "mine" / "SKILL.md").read_text() == "mine\n"
    assert (skills / "hub-status" / "SKILL.md").is_file()


def test_refuses_non_empty_folder_without_project_yaml(tmp_path):
    v = tmp_path / "notes"
    v.mkdir()
    (v / "mine.txt").write_text("keep me\n")
    with pytest.raises(ValueError, match="not empty and has no project.yaml"):
        init_vault(v, name="demo")
    assert sorted(p.name for p in v.iterdir()) == ["mine.txt"]


def test_force_proceeds_into_non_empty_folder(tmp_path):
    v = tmp_path / "notes"
    v.mkdir()
    (v / "mine.txt").write_text("keep me\n")
    init_vault(v, name="demo", force=True)
    assert (v / "project.yaml").exists()
    assert (v / "mine.txt").read_text() == "keep me\n"


def test_existing_vault_and_empty_or_new_folder_are_accepted(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    init_vault(empty, name="demo")
    init_vault(empty, name="demo")  # existing vault: re-run is fine
    init_vault(tmp_path / "brand-new", name="demo")
    assert (tmp_path / "brand-new" / "project.yaml").exists()


def test_foreign_setup_sh_is_not_overwritten(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    (v / "tools" / "setup.sh").write_text("#!/bin/sh\necho my own setup\n")
    with pytest.raises(ValueError, match="setup.sh"):
        init_vault(v, name="demo")
    assert (v / "tools" / "setup.sh").read_text() == "#!/bin/sh\necho my own setup\n"


def test_missing_skills_source_gives_clean_error(tmp_path, monkeypatch):
    fake_pkg = tmp_path / "site-packages"
    fake_pkg.mkdir()
    monkeypatch.setattr(init, "PACKAGE_DIR", fake_pkg)
    v = tmp_path / "vault"
    with pytest.raises(ValueError, match="cannot find the hub skills next to the package"):
        init_vault(v, name="demo")


def test_missing_git_is_treated_as_not_a_repo(tmp_path, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(init.subprocess, "run", no_git)
    assert init._is_git_repo(tmp_path) is False


def test_rerun_does_not_duplicate_ignore_or_attributes_lines(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    init_vault(v, name="demo")
    for f in (".gitignore", ".gitattributes"):
        lines = (v / f).read_text().splitlines()
        assert len(lines) == len(set(lines)), f


def test_user_skill_folder_survives_rerun(tmp_path):
    v = tmp_path / "vault"
    init_vault(v, name="demo")
    (v / ".claude" / "skills" / "my-skill").mkdir(parents=True)
    (v / ".claude" / "skills" / "my-skill" / "SKILL.md").write_text("mine\n")
    init_vault(v, name="demo")
    assert (v / ".claude" / "skills" / "my-skill" / "SKILL.md").read_text() == "mine\n"
