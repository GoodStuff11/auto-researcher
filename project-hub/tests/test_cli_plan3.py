import json
import re
import subprocess
import sys

import pytest

from hub.cli import main
from hub.init import SETUP_SH


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_init_runs_outside_a_vault(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out, _ = run(capsys, "init", str(tmp_path / "v"), "--name", "demo")
    assert code == 0 and "project.yaml" in out and "bash tools/setup.sh" in out
    assert (tmp_path / "v" / ".claude" / "skills" / "hub-status" / "SKILL.md").is_file()


def test_init_refuses_non_vault_folder_unless_forced(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "other"
    d.mkdir()
    (d / "notes.txt").write_text("keep me")
    code, out, err = run(capsys, "init", str(d), "--name", "demo")
    assert code == 1 and err.startswith("error:") and sorted(p.name for p in d.iterdir()) == ["notes.txt"]
    code, out, _ = run(capsys, "init", str(d), "--name", "demo", "--force")
    assert code == 0 and (d / "project.yaml").is_file() and (d / "notes.txt").read_text() == "keep me"


@pytest.fixture
def vault(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.png").write_bytes(b"x")
    v = tmp_path / "v"
    v.mkdir()
    (v / "project.yaml").write_text("name: demo\npeople: {alice: {name: Alice}}\nseries: {}\nroots: {SRC: /x}\n")
    (v / "hub.local.yaml").write_text(f"me: alice\nroots:\n  SRC: {src}\n")
    monkeypatch.chdir(v)
    monkeypatch.delenv("HUB_ME", raising=False)
    return v, src


def test_figures_cli(vault, capsys):
    v, src = vault
    code, out, _ = run(capsys, "figures", "add", "f1", "$SRC/f.png", "--title", "T", "--licenses", "L")
    assert code == 0 and (v / "Figures" / "f1" / "figure.png").is_file()
    code, out, _ = run(capsys, "figures", "sync")
    assert code == 0 and "f1  unchanged" in out and (v / "Figures" / "Gallery.md").is_file()
    (src / "f.png").unlink()
    code, out, _ = run(capsys, "figures", "sync")
    assert code == 1 and "f1  PROBLEM" in out
    code, _, err = run(capsys, "figures", "add", "Bad Id", "$SRC/f.png")
    assert code == 1 and "figure id" in err


def test_figures_gallery_cli(vault, capsys):
    v, _ = vault
    code, out, _ = run(capsys, "figures", "gallery")
    assert code == 0 and "Gallery.md" in out and (v / "Figures" / "Gallery.md").is_file()


def test_check_links_cli(vault, capsys):
    v, src = vault
    (v / "Locator.md").write_text("- `$SRC/f.png`\n- `$SRC/gone.txt`\n")
    code, out, _ = run(capsys, "check-links", "--json")
    rows = {r["ref"]: r for r in json.loads(out)}
    assert code == 1 and rows["$SRC/f.png"]["ok"] is True and rows["$SRC/gone.txt"]["ok"] is False
    code, out, _ = run(capsys, "check-links")
    assert code == 1 and "BROKEN" in out and "ok" in out


def test_setup_sh_skips_driver_setup_outside_a_repo(tmp_path):
    block = re.search(r'^if git -C .*?^fi$', SETUP_SH, re.S | re.M).group(0)
    vault = tmp_path / "v"
    vault.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    res = subprocess.run(["bash", "-c", f'VAULT="{vault}"; BIN="{bin_dir}"; set -e; {block}'],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "GIT_CEILING_DIRECTORIES": str(tmp_path)})
    assert res.returncode == 0, res.stderr
    assert "not a git repository" in res.stdout and "Traceback" not in res.stderr
