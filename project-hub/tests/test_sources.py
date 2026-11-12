from datetime import date

import pytest

from hub.pending import add_proposal
from hub.sources import source_seen
from hub.store import Store

D = date(2026, 11, 12)


def test_source_seen(hub_root):
    s = Store(hub_root)
    q = s.create("question", "Q", owner="alice", author="alice", day=D, source="circleback:m1")
    s.log("Ran it.", author="alice", source="commit:ML@abc1234", day=D)
    pid = add_proposal(hub_root, "P", [{"op": "entry", "id": q.id, "text": "x", "source": "clickup:t9"}],
                       proposer="alice", source="clickup:t9", reason="r", day=D)
    assert source_seen(hub_root, s, "circleback:m1") == [q.id]
    assert source_seen(hub_root, s, "commit:ML@abc1234") == ["log:2026-11"]
    assert source_seen(hub_root, s, "clickup:t9") == [pid]
    assert source_seen(hub_root, s, "circleback:zzz") == []


def test_source_seen_does_not_match_prefixes(hub_root):
    s = Store(hub_root)
    s.log("Ran it.", author="alice", source="commit:ML@abc12345", day=D)
    assert source_seen(hub_root, s, "commit:ML@abc1234") == []


def test_unreadable_item_file_is_reported_not_silently_new(hub_root):
    s = Store(hub_root)
    (hub_root / "Status" / "Items").mkdir(parents=True, exist_ok=True)
    (hub_root / "Status" / "Items" / "I-zzzz.md").write_text("garbage, no frontmatter\n", encoding="utf-8")
    assert source_seen(hub_root, s, "circleback:never") == ["unreadable:Status/Items/I-zzzz.md"]


def test_unreadable_pending_file_is_reported(hub_root):
    s = Store(hub_root)
    (hub_root / "Status" / "Pending").mkdir(parents=True, exist_ok=True)
    (hub_root / "Status" / "Pending" / "P-zzzz.md").write_text("garbage\n", encoding="utf-8")
    assert "unreadable:Status/Pending/P-zzzz.md" in source_seen(hub_root, s, "circleback:never")


def test_log_with_invalid_utf8_raises(hub_root):
    s = Store(hub_root)
    (hub_root / "Status" / "Log").mkdir(parents=True, exist_ok=True)
    (hub_root / "Status" / "Log" / "2026-11.md").write_bytes(b"# Log\n\xff\xfe bad\n")
    with pytest.raises(ValueError, match="cannot read log"):
        source_seen(hub_root, s, "circleback:never")


def test_string_sources_do_not_substring_match(hub_root):
    s = Store(hub_root)
    q = s.create("question", "Q", owner="alice", author="alice", day=D, source="circleback:m12")
    item = s.get(q.id)
    item.meta["sources"] = "circleback:m12"
    s.save(item)
    assert q.id not in source_seen(hub_root, s, "circleback:m1")
