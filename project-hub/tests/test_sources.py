from datetime import date

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
