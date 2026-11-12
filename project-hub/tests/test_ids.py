import re
from datetime import date
from hub.ids import new_id, new_pending_id


class FixedRng:
    def __init__(self, seq):
        self.seq = list(seq)

    def choice(self, _alphabet):
        return self.seq.pop(0)


def test_format():
    i = new_id("question", date(2026, 11, 12), set())
    assert re.fullmatch(r"Q-20261112-[a-z0-9]{3}", i)


def test_prefixes():
    d = date(2026, 1, 2)
    assert new_id("decision", d, set()).startswith("D-20260102-")
    assert new_id("action", d, set()).startswith("A-20260102-")


def test_avoids_existing():
    rng = FixedRng("aaabbb")
    assert new_id("question", date(2026, 11, 12), {"Q-20261112-aaa"}, rng) == "Q-20261112-bbb"


def test_pending_id():
    assert re.fullmatch(r"P-[a-z0-9]{4}", new_pending_id(set()))
    assert new_pending_id({"P-aaaa"}, FixedRng("aaaabbbb")) == "P-bbbb"
