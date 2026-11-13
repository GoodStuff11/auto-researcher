import os
import subprocess

from hub.config import load_config
from hub.init import TEMPLATES, init_vault
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
    (v / "tools" / "project-hub" / "hub" / "cli.py").write_text("broken\n")
    created = init_vault(v, name="demo")
    assert (v / "Vision.md").read_text() == "# Our vision\nEdited.\n"
    assert load_config(v).name == "renamed"
    assert "broken" not in (v / "tools" / "project-hub" / "hub" / "cli.py").read_text()
    assert "Vision.md" not in created and "project.yaml" not in created


def test_init_registers_merge_drivers_in_a_git_repo(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=v, check=True)
    init_vault(v, name="demo")
    out = subprocess.run(["git", "config", "merge.hub-item.driver"], cwd=v, capture_output=True, text=True).stdout
    assert "hub.merge_item" in out


def test_templates_are_generic():
    text = "\n".join(TEMPLATES.values()).lower()
    for word in ("ucc", "hubbard", "fermion", "hva", "clickup list 9"):
        assert word not in text
