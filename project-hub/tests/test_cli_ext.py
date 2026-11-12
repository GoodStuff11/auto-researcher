import json
import subprocess

import pytest
import yaml

from hub.cli import main
from hub.store import Store
from hub.sync_state import load_state


@pytest.fixture
def cli(hub_root, monkeypatch):
    monkeypatch.chdir(hub_root)
    monkeypatch.delenv("HUB_ME", raising=False)
    monkeypatch.setenv("HUB_TODAY", "2026-11-19")
    return hub_root


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_seen(cli, capsys):
    run(capsys, "log", "Ran.", "--source", "commit:CODE@abc1234")
    assert run(capsys, "seen", "commit:CODE@abc1234")[1].strip() == "seen log:2026-11"
    assert run(capsys, "seen", "circleback:none")[1].strip() == "new"


def test_ingest_batch(cli, capsys):
    q = Store(cli).create("question", "Why?", owner="alice", author="alice",
                          day=__import__("datetime").date(2026, 11, 1), source="m")
    f = cli / "batch.yaml"
    f.write_text(yaml.safe_dump({"safe": [{"op": "entry", "id": q.id, "text": "Noted.", "source": "circleback:m1"}],
                                 "watermark": {"name": "circleback", "value": "2026-11-19"}}))
    code, out, _ = run(capsys, "ingest", "batch", str(f))
    assert code == 0 and "applied 1, skipped 0, staged 0" in out
    code, out, _ = run(capsys, "ingest", "batch", str(f))
    assert "applied 0, skipped 1" in out
    f.write_text(yaml.safe_dump({"safe": [{"op": "new", "type": "decision", "title": "X", "source": "s"}]}))
    code, _, err = run(capsys, "ingest", "batch", str(f))
    assert code == 1 and "decisions are proposals" in err


def test_ingest_batch_unknown_owner_fails(cli, capsys):
    f = cli / "batch.yaml"
    f.write_text(yaml.safe_dump({"safe": [{"op": "new", "type": "action", "title": "X", "owner": "mallory",
                                           "source": "s"}]}))
    code, _, err = run(capsys, "ingest", "batch", str(f))
    assert code == 1 and "mallory" in err and "not a known person" in err
    assert Store(cli).query(type="action") == []


def _repo(tmp_path, n):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@x"], ["config", "user.name", "T"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    for i in range(n):
        (repo / "a.txt").write_text(str(i))
        subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", ["first", "second", "third"][i]], cwd=repo, check=True)
    return repo


def test_ingest_git(cli, capsys, tmp_path):
    repo = _repo(tmp_path, 1)
    (cli / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")
    code, out, _ = run(capsys, "ingest", "git", "CODE", "--advance")
    res = json.loads(out)
    assert code == 0 and [c["subject"] for c in res["commits"]] == ["first"] and res["advanced"] is True
    code, _, err = run(capsys, "ingest", "git", "NOPE")
    assert code == 1 and "unknown root" in err


def test_ingest_git_limit(cli, capsys, tmp_path):
    repo = _repo(tmp_path, 3)
    (cli / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {repo}\n")
    code, out, err = run(capsys, "ingest", "git", "CODE", "--limit", "2")
    res = json.loads(out)
    assert code == 0 and [c["subject"] for c in res["commits"]] == ["first", "second"]
    assert err.startswith("warning:") and "not listed yet" in err


def test_clickup_plan_and_apply(cli, capsys):
    a = Store(cli).create("action", "Rerun", owner="alice", author="alice",
                          day=__import__("datetime").date(2026, 11, 1), source="m", fields={"clickup_id": "t1"})
    f = cli / "tasks.json"
    f.write_text(json.dumps([{"id": "t1", "name": "Rerun", "status": "complete", "assignees": []}]))
    code, out, _ = run(capsys, "clickup", "plan", "--tasks", str(f))
    assert code == 0 and json.loads(out)["set"][0]["id"] == a.id
    code, out, _ = run(capsys, "clickup", "apply", "--tasks", str(f))
    assert code == 0 and json.loads(out)["updated"] == [a.id]
    assert Store(cli).get(a.id).status == "done"


def test_meeting_next_and_decided(cli, capsys):
    assert run(capsys, "meeting", "next", "group", "2026-11-26")[0] == 0
    assert str(load_state(cli)["next_meeting"]["group"]) == "2026-11-26"
    code, _, err = run(capsys, "meeting", "next", "nosuch", "2026-11-26")
    assert code == 1 and "unknown series" in err
    s = Store(cli)
    d = s.create("decision", "Adopt Y", owner="alice", author="alice",
                 day=__import__("datetime").date(2026, 11, 1), source="m",
                 fields={"status": "made", "meeting": "group/2026-11-19", "as_decided": "Only 4x3"})
    s.create("action", "Do it", owner="bob", author="alice", day=__import__("datetime").date(2026, 11, 19),
             source="m", links={"spawned_by": [d.id]})
    code, out, _ = run(capsys, "meeting", "decided", "group", "2026-11-19", "--json")
    rows = json.loads(out)
    assert rows == [{"id": d.id, "title": "Adopt Y", "status": "made", "as_decided": "Only 4x3",
                     "actions": [s.query(type="action")[0].id]}]


def test_sources(cli, capsys):
    (cli / "project.yaml").write_text((cli / "project.yaml").read_text() + (
        "sources:\n  clickup: {list_id: '900'}\n  circleback: {folder: Proj}\n"
        "  repos: [{root: CODE, branch: main}]\n  calendar: {}\n"))
    run(capsys, "sync-state", "set", "clickup", "2026-11-18T10:00")
    code, out, _ = run(capsys, "sources", "--json")
    rows = json.loads(out)
    assert [r["source"] for r in rows] == ["clickup", "circleback", "git.CODE", "calendar"]
    assert rows[0]["watermark"] == "2026-11-18T10:00" and rows[1]["watermark"] is None
    code, out, _ = run(capsys, "sources")
    assert code == 0 and "git.CODE" in out


def test_sources_json_with_date_watermark(cli, capsys):
    (cli / "project.yaml").write_text((cli / "project.yaml").read_text() + "sources:\n  clickup: {list_id: '9'}\n")
    d = cli / "Status" / "sync-state"
    d.mkdir(parents=True, exist_ok=True)
    (d / "watermark.clickup.yaml").write_text("value: 2026-11-19\n")
    code, out, _ = run(capsys, "sources", "--json")
    assert code == 0 and "2026-11-19" in out


def test_bad_yaml_batch_names_file(cli, capsys):
    f = cli / "bad.yaml"
    f.write_text("safe: [unclosed\n")
    code, _, err = run(capsys, "ingest", "batch", str(f))
    assert code == 1 and "bad.yaml" in err and "cannot parse" in err


def test_bad_tasks_file_names_file(cli, capsys):
    f = cli / "tasks.json"
    f.write_text("[{unclosed\n")
    code, _, err = run(capsys, "clickup", "plan", "--tasks", str(f))
    assert code == 1 and "tasks.json" in err and "cannot parse" in err


@pytest.mark.parametrize("n", ["0", "-3"])
def test_limit_must_be_positive(cli, capsys, n):
    with pytest.raises(SystemExit) as e:
        main(["ingest", "git", "CODE", "--limit", n])
    assert e.value.code == 2
