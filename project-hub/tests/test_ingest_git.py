import subprocess
import textwrap
from datetime import date

import pytest

from hub.config import load_config
from hub.ingest_git import IngestError, git_commits, ingest_git
from hub.sync_state import get_watermark, set_watermark


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def commit_file(repo, name, msg):
    (repo / name).write_text(msg)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.delenv("HUB_ME", raising=False)
    repo = tmp_path / "code"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Tester")
    for i in range(3):
        commit_file(repo, f"f{i}.txt", f"c{i}")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text(textwrap.dedent(f"""\
        name: demo
        people:
          alice: {{name: Alice}}
        series:
          group: {{attendees: [alice]}}
        roots:
          CODE: /nowhere
          NOTGIT: {tmp_path}
        sources:
          repos:
            - {{root: CODE, branch: main}}
        """))
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")
    return load_config(vault), repo, vault


def test_first_run_lists_all_oldest_first_and_advances(setup):
    cfg, repo, vault = setup
    res = ingest_git(cfg, "CODE", today=date.today(), advance=True)
    assert [c["subject"] for c in res["commits"]] == ["c0", "c1", "c2"]
    assert res["commits"][0]["files"] == ["f0.txt"]
    assert res["commits"][0]["author"] == "Tester" and len(res["commits"][0]["short"]) >= 7
    assert res["branch"] == "main" and res["watermark_before"] is None and res["advanced"] is True
    assert get_watermark(vault, "git.CODE") == res["commits"][-1]["sha"]


def test_second_run_lists_only_new_commits(setup):
    cfg, repo, vault = setup
    ingest_git(cfg, "CODE", today=date.today(), advance=True)
    commit_file(repo, "f3.txt", "c3")
    res = ingest_git(cfg, "CODE", today=date.today())
    assert [c["subject"] for c in res["commits"]] == ["c3"]
    assert res["advanced"] is False
    again = ingest_git(cfg, "CODE", today=date.today(), advance=True)
    assert [c["subject"] for c in again["commits"]] == ["c3"]
    assert ingest_git(cfg, "CODE", today=date.today())["commits"] == []


def test_unknown_watermark_falls_back_to_lookback(setup):
    cfg, repo, vault = setup
    set_watermark(vault, "git.CODE", "deadbeef" * 5)
    res = ingest_git(cfg, "CODE", today=date.today())
    assert "not found" in res["warning"]
    assert [c["subject"] for c in res["commits"]] == ["c0", "c1", "c2"]


def test_explicit_since_ignores_watermark(setup):
    cfg, repo, vault = setup
    ingest_git(cfg, "CODE", today=date.today(), advance=True)
    res = ingest_git(cfg, "CODE", today=date.today(), since=date(2000, 1, 1))
    assert len(res["commits"]) == 3


def test_errors(setup):
    cfg, repo, vault = setup
    with pytest.raises(IngestError, match="unknown root"):
        ingest_git(cfg, "NOPE", today=date.today())
    with pytest.raises(IngestError, match="not a git repository"):
        ingest_git(cfg, "NOTGIT", today=date.today())
    cfg.sources["repos"][0]["branch"] = "nosuch"
    with pytest.raises(IngestError, match="branch 'nosuch'"):
        ingest_git(cfg, "CODE", today=date.today())


def test_limit_keeps_oldest_and_warns(setup):
    cfg, repo, vault = setup
    for i in range(3, 7):
        commit_file(repo, f"f{i}.txt", f"c{i}")
    commits, warning = git_commits(repo, "main", limit=5)
    assert [c["subject"] for c in commits] == ["c0", "c1", "c2", "c3", "c4"]
    assert "2 newer commit(s) not listed yet (limit 5)" in warning


def test_advance_with_limit_pages_through_history(setup):
    cfg, repo, vault = setup
    for i in range(3, 7):
        commit_file(repo, f"f{i}.txt", f"c{i}")
    first = ingest_git(cfg, "CODE", today=date.today(), advance=True, limit=5)
    assert [c["subject"] for c in first["commits"]] == ["c0", "c1", "c2", "c3", "c4"]
    assert get_watermark(vault, "git.CODE") == first["commits"][-1]["sha"]
    second = ingest_git(cfg, "CODE", today=date.today(), advance=True, limit=5)
    assert [c["subject"] for c in second["commits"]] == ["c5", "c6"]
    assert second["warning"] is None


def test_subject_with_separator_characters_is_kept(setup):
    cfg, repo, vault = setup
    subject = "x\x1fy\x1ez"
    commit_file(repo, "sep.txt", subject)
    commits, _ = git_commits(repo, "main")
    assert commits[-1]["subject"] == subject
    assert commits[-1]["files"] == ["sep.txt"]


def test_file_names_are_exact_not_quoted(setup):
    cfg, repo, vault = setup
    commit_file(repo, "é fig.pdf", "fig")
    commits, _ = git_commits(repo, "main")
    assert commits[-1]["files"] == ["é fig.pdf"]


def test_watermark_off_the_branch_falls_back(setup):
    cfg, repo, vault = setup
    git(repo, "checkout", "-q", "-b", "other")
    commit_file(repo, "o.txt", "other")
    other_sha = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "checkout", "-q", "main")
    set_watermark(vault, "git.CODE", other_sha)
    res = ingest_git(cfg, "CODE", today=date.today())
    assert "is not an ancestor of main" in res["warning"]
    assert [c["subject"] for c in res["commits"]] == ["c0", "c1", "c2"]
