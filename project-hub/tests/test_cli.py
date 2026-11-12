# project-hub/tests/test_cli.py
import json
import pytest
from hub.cli import main
from hub.store import Store


@pytest.fixture
def cli(hub_root, monkeypatch):
    monkeypatch.chdir(hub_root)
    monkeypatch.delenv("HUB_ME", raising=False)
    monkeypatch.setenv("HUB_TODAY", "2026-11-12")
    return hub_root


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_new_list_show_set(cli, capsys):
    code, qid, _ = run(capsys, "new", "q", "Why?", "--source", "manual", "--series", "group", "--priority", "1")
    qid = qid.strip()
    assert code == 0 and qid.startswith("Q-20261112-")
    code, did, _ = run(capsys, "new", "d", "Do X", "--source", "manual", "--link", f"answers:{qid}")
    did = did.strip()
    code, out, _ = run(capsys, "list", "--type", "question", "--json")
    assert json.loads(out)[0]["id"] == qid
    code, out, _ = run(capsys, "show", qid, "--chain")
    assert qid in out and f"  {did}" in out
    assert run(capsys, "entry", qid, "More info.", "--source", "commit:x@1")[0] == 0
    assert run(capsys, "set", qid, "status=answered", "closed=2026-11-12")[0] == 0
    assert Store(cli).get(qid).status == "answered"


def test_validate_exit_codes(cli, capsys):
    assert run(capsys, "validate")[0] == 0
    run(capsys, "new", "d", "Do X", "--source", "manual")
    did = Store(cli).query(type="decision")[0].id
    run(capsys, "set", did, "status=made")
    code, out, _ = run(capsys, "validate")
    assert code == 1 and "decision-answers" in out


def test_user_error_is_clean(cli, capsys):
    code, _, err = run(capsys, "set", "Q-20990101-zzz", "status=open")
    assert code == 1 and "Q-20990101-zzz" in err


def test_writes_need_me(cli, capsys):
    (cli / "hub.local.yaml").unlink()
    code, _, err = run(capsys, "new", "q", "Why?", "--source", "m")
    assert code == 1 and "hub.local.yaml" in err


def test_agenda_flow_and_pending(cli, capsys):
    run(capsys, "new", "q", "Why?", "--source", "m", "--series", "group")
    code, out, _ = run(capsys, "agenda", "scaffold", "group", "--date", "2026-11-19")
    assert code == 0 and "group.draft.yaml" in out
    code, _, err = run(capsys, "agenda", "build", "group")
    assert code == 1 and "summary is empty" in err
    draft = cli / "Agenda" / "group.draft.yaml"
    draft.write_text(draft.read_text().replace("summary: ''", "summary: Status."))
    assert run(capsys, "agenda", "build", "group")[0] == 0
    code, out, _ = run(capsys, "agenda", "read", "group")
    assert code == 0 and "applied 0" in out
    code, out, _ = run(capsys, "pending", "list")
    assert code == 0 and "No pending" in out


def test_snapshot_and_sync_state(cli, capsys):
    assert run(capsys, "snapshot")[0] == 0
    assert (cli / "Status" / "Snapshot.md").is_file()
    assert run(capsys, "sync-state", "set", "clickup", "2026-11-12T09:00")[0] == 0
    assert run(capsys, "sync-state", "get", "clickup")[1].strip() == "2026-11-12T09:00"


def test_help_documents_every_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("new", "entry", "set", "link", "list", "show", "since", "log", "validate", "snapshot",
                "pending", "agenda", "sync-state", "commit"):
        assert cmd in out


def test_scaffold_refuses_to_overwrite(cli, capsys):
    assert run(capsys, "agenda", "scaffold", "group", "--date", "2026-11-19")[0] == 0
    code, _, err = run(capsys, "agenda", "scaffold", "group", "--date", "2026-11-19")
    assert code == 1 and "already exists" in err and "--force" in err
    assert run(capsys, "agenda", "scaffold", "group", "--date", "2026-11-19", "--force")[0] == 0


def test_garbage_item_file_is_clean_error(cli, capsys):
    items = cli / "Status" / "Items"
    items.mkdir(parents=True, exist_ok=True)
    (items / "Q-20260101-bad.md").write_text("not an item at all\n")
    for argv in (("validate",), ("list",)):
        code, _, err = run(capsys, *argv)
        assert code == 1 and "error:" in err


def test_ops_file_problems_are_clean(cli, capsys):
    base = ("pending", "add", "--title", "T", "--source", "m", "--reason", "r", "--ops-file")
    code, _, err = run(capsys, *base, str(cli / "missing.yaml"))
    assert code == 1 and "error:" in err
    bad = cli / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    code, _, err = run(capsys, *base, str(bad))
    assert code == 1 and "error:" in err


def test_malformed_project_yaml_is_clean(cli, capsys):
    (cli / "project.yaml").write_text("name: [unclosed\n")
    code, _, err = run(capsys, "list")
    assert code == 1 and "error:" in err


def test_bad_root_is_clean(cli, capsys):
    code, _, err = run(capsys, "--root", "/nonexistent/hub", "list")
    assert code == 1 and "error:" in err


def _walk(parser, path="hub"):
    import argparse
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            helps = {a.dest: a.help for a in act._choices_actions}
            for name, sp in act.choices.items():
                assert helps.get(name) or any(c.dest == name and c.help for c in act._choices_actions), \
                    f"{path} {name}: no help="
                assert sp.description, f"{path} {name}: no description"
                yield from _walk(sp, f"{path} {name}")
        elif not isinstance(act, argparse._HelpAction):
            assert act.help, f"{path}: argument {act.dest} has no help"


def test_every_subparser_and_argument_is_documented():
    from hub.cli import build_parser
    list(_walk(build_parser()))


def test_owner_me_needs_local_config(cli, capsys):
    (cli / "hub.local.yaml").unlink()
    code, _, err = run(capsys, "list", "--owner", "me")
    assert code == 1 and "hub.local.yaml" in err


def test_bad_since_date_is_exit_2(cli, capsys):
    with pytest.raises(SystemExit) as ei:
        main(["list", "--since", "nope"])
    assert ei.value.code == 2


def test_sync_conflict_exit_3(cli, capsys, monkeypatch):
    from hub.gitsync import SyncConflict

    def boom(*a, **k):
        raise SyncConflict(["Status/Items/x.md"], "detail")

    monkeypatch.setattr("hub.cli.commit", boom)
    code, _, err = run(capsys, "commit", "msg")
    assert code == 3 and "x.md" in err
