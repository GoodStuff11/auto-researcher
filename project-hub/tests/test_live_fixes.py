import json
import textwrap
from datetime import date

import pytest

from hub.batch import check_batch, run_batch
from hub.cli import main
from hub.config import load_config
from hub.links import check_links
from hub.pending import add_proposal, apply_proposals
from hub.store import Store

D = date(2026, 11, 19)


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


def test_rerun_with_clickup_id_proposal_is_a_no_op(hub_root):
    store = Store(hub_root)
    batch = {"proposals": [{"title": "Import: Ship it", "reason": "r", "source": "doc:notes#todo",
                            "ops": [{"op": "new", "type": "action", "title": "Ship it", "source": "doc:notes#todo",
                                     "fields": {"clickup_id": "86abc"}}]}]}
    pid = run_batch(hub_root, store, batch, author="alice", day=D)["staged"][0]
    rep = run_batch(hub_root, store, batch, author="alice", day=D)  # still staged
    assert rep["staged"] == [] and rep["skipped_proposals"] == 1
    apply_proposals(hub_root, store, [pid], author="alice", day=D)
    assert [a.meta.get("clickup_id") for a in store.query(type="action")] == ["86abc"]
    # applied: the id is now linked, yet the re-run is a no-op and not an error
    rep = run_batch(hub_root, store, batch, author="alice", day=D)
    assert rep["staged"] == [] and rep["skipped_proposals"] == 1
    assert check_batch(store, batch) == []


def test_new_proposal_with_taken_clickup_id_is_still_rejected(hub_root):
    store = Store(hub_root)
    store.create("action", "Existing", owner="bob", author="alice", day=D, source="m", fields={"clickup_id": "86abc"})
    batch = {"proposals": [{"title": "Other", "reason": "r", "source": "s",
                            "ops": [{"op": "new", "type": "action", "title": "Other", "source": "s",
                                     "fields": {"clickup_id": "86abc"}}]}]}
    assert any("already linked" in e for e in check_batch(store, batch))


def test_pending_list_source_filter_and_order(cli, capsys):
    ops = [{"op": "new", "type": "question", "title": "A"}]
    ids = {}
    for title, src in [("T1", "doc:notes#b"), ("T2", "doc:notes#a"), ("T3", "doc:other#a"), ("T4", "doc:notes#a")]:
        ids[title] = add_proposal(cli, title, ops, proposer="alice", source=src, reason="r", day=date(2026, 11, 12))
    code, out, _ = run(capsys, "pending", "list", "--source", "doc:notes#a")
    got = [ln.split()[0] for ln in out.splitlines()]
    assert code == 0 and got == sorted([ids["T2"], ids["T4"]])
    code, out, _ = run(capsys, "pending", "list", "--source", "doc:nothing")
    assert code == 0 and "No pending proposals" in out
    code, out, _ = run(capsys, "pending", "list")
    got = [ln.split()[0] for ln in out.splitlines()]
    by_src = {ids["T1"]: "doc:notes#b", ids["T2"]: "doc:notes#a", ids["T3"]: "doc:other#a", ids["T4"]: "doc:notes#a"}
    assert got == sorted(got, key=lambda p: (by_src[p], p))


def test_sources_none_configured(cli, capsys):
    code, out, _ = run(capsys, "sources")
    assert code == 0 and out.strip() == "no sources configured in project.yaml"
    code, out, _ = run(capsys, "sources", "--json")
    assert code == 0 and json.loads(out) == []


def _vault(tmp_path, root_target, locator):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "project.yaml").write_text("name: demo\npeople: {alice: {name: Alice}}\nseries: {}\nroots: {CODE: /x}\n")
    (vault / "hub.local.yaml").write_text(f"me: alice\nroots:\n  CODE: {root_target}\n")
    (vault / "Locator.md").write_text(textwrap.dedent(locator))
    return load_config(vault)


def test_bare_root_reference_is_checked(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    cfg = _vault(tmp_path, code, "# Locator\n- Code: `$CODE`\n- Missing: `$NOPE`\n- Maths $x$.\n")
    by_ref = {r["ref"]: r for r in check_links(cfg)}
    assert set(by_ref) == {"$CODE", "$NOPE"}
    assert by_ref["$CODE"]["ok"] is True
    assert by_ref["$NOPE"]["ok"] is False and "unknown root" in by_ref["$NOPE"]["problem"]


def test_bare_root_reference_must_be_a_directory(tmp_path):
    f = tmp_path / "afile"
    f.write_text("x")
    cfg = _vault(tmp_path, f, "- Code: `$CODE`\n")
    row = check_links(cfg)[0]
    assert row["ok"] is False and "not a directory" in row["problem"]
