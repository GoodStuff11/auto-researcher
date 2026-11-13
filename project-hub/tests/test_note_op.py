from datetime import date

import pytest

from hub.batch import BatchError, run_batch
from hub.pending import PendingError, add_proposal, apply_proposals, check_ops, load_proposals
from hub.store import Store

D = date(2026, 11, 13)


def test_note_replace_and_append(hub_root):
    s = Store(hub_root)
    pid = add_proposal(hub_root, "Vision draft", [{"op": "note", "path": "Vision.md", "text": "# Vision\nFirst."}],
                       proposer="alice", source="manual", reason="r", day=D)
    apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Vision.md").read_text() == "# Vision\nFirst.\n"
    pid = add_proposal(hub_root, "Add a fact", [{"op": "note", "path": "Knowledge/Code map.md",
                                                 "text": "- solver lives in src/", "mode": "append"},
                                                {"op": "note", "path": "Knowledge/Code map.md",
                                                 "text": "- tests in tests/", "mode": "append"}],
                       proposer="alice", source="manual", reason="r", day=D)
    apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Knowledge" / "Code map.md").read_text() == "- solver lives in src/\n- tests in tests/\n"


@pytest.mark.parametrize("path", ["../x.md", "Status/Items/Q-20261113-abc.md", "Knowledge/sub/x.md",
                                  "Knowledge/x.txt", "vision.md", "/etc/passwd", "Knowledge/.md"])
def test_note_paths_are_restricted(hub_root, path):
    errs = check_ops(Store(hub_root), [{"op": "note", "path": path, "text": "x"}])
    assert any("note path" in e for e in errs)


def test_note_rejects_bad_text_and_mode(hub_root):
    s = Store(hub_root)
    assert any("text" in e for e in check_ops(s, [{"op": "note", "path": "Vision.md", "text": 3}]))
    assert any("mode" in e for e in check_ops(s, [{"op": "note", "path": "Vision.md", "text": "x", "mode": "x"}]))


def test_failed_proposal_restores_the_note(hub_root, monkeypatch):
    (hub_root / "Locator.md").write_text("original\n")
    s = Store(hub_root)
    pid = add_proposal(hub_root, "Locator and a bad link",
                       [{"op": "note", "path": "Locator.md", "text": "new"},
                        {"op": "new", "type": "question", "title": "Q", "source": "manual"}],
                       proposer="alice", source="manual", reason="r", day=D)
    monkeypatch.setattr(Store, "create", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(PendingError):
        apply_proposals(hub_root, s, [pid], author="alice", day=D)
    assert (hub_root / "Locator.md").read_text() == "original\n"
    assert [p.pid for p in load_proposals(hub_root)] == [pid]


def test_note_is_never_a_safe_batch_op(hub_root):
    with pytest.raises(BatchError):
        run_batch(hub_root, Store(hub_root), {"safe": [{"op": "note", "path": "Vision.md", "text": "x",
                                                        "source": "manual"}]}, author="alice", day=D)
    rep = run_batch(hub_root, Store(hub_root),
                    {"proposals": [{"title": "Knowledge: setup", "reason": "from doc", "source": "doc:readme#setup",
                                    "ops": [{"op": "note", "path": "Knowledge/Setup.md", "text": "Run make."}]}]},
                    author="alice", day=D)
    assert len(rep["staged"]) == 1
