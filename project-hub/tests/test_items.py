# project-hub/tests/test_items.py
from datetime import date
import pytest
from hub.items import (Entry, Item, ItemFormatError, default_meta, parse_item, render_item)

CANONICAL = """---
id: Q-20261112-a3f
type: question
title: Does method A beat method B?
status: open
owner: alice
series:
- group
raised: 2026-11-12
closed:
priority: 2
links:
  answers: []
  spawned_by: []
  pivoted_to: []
sources:
- circleback:123
custom_key: kept
---
Why this matters.
## Entries
- 2026-11-12 · alice · circleback:123 — Raised at group meeting.
- 2026-11-15 · bob · commit:ML@ab12cd3 — Run finished: F=0.97.
"""


def test_round_trip_is_exact():
    assert render_item(parse_item(CANONICAL)) == CANONICAL


def test_parsed_fields():
    it = parse_item(CANONICAL)
    assert it.id == "Q-20261112-a3f"
    assert it.type == "question" and it.status == "open" and it.owner == "alice"
    assert it.meta["raised"] == date(2026, 11, 12)
    assert it.meta["closed"] is None
    assert it.meta["custom_key"] == "kept"
    assert it.description == "Why this matters."
    assert it.entries[1] == Entry(date(2026, 11, 15), "bob", "commit:ML@ab12cd3", "Run finished: F=0.97.")
    assert it.links("answers") == []


def test_last_activity():
    assert parse_item(CANONICAL).last_activity() == date(2026, 11, 15)


def test_entry_render_collapses_newlines():
    e = Entry(date(2026, 1, 1), "alice", "manual", "two\nlines")
    assert e.render() == "- 2026-01-01 · alice · manual — two lines"


def test_malformed_entry_rejected():
    bad = CANONICAL.replace("— Raised", "- Raised")
    with pytest.raises(ItemFormatError, match="malformed entry"):
        parse_item(bad)


def test_missing_frontmatter_rejected():
    with pytest.raises(ItemFormatError):
        parse_item("no frontmatter\n")


def test_default_meta_by_type():
    d = default_meta("D-20261112-aaa", "decision", "Pick X", "alice", date(2026, 11, 12), ["group"], 1, ["manual"])
    assert d["status"] == "proposed" and d["rationale"] == "" and d["meeting"] is None and d["as_decided"] == ""
    a = default_meta("A-20261112-aaa", "action", "Do X", "bob", date(2026, 11, 12), [], 2, [])
    assert a["status"] == "open" and a["clickup_id"] is None and a["due"] is None
    q = default_meta("Q-20261112-aaa", "question", "Why?", "bob", date(2026, 11, 12), [], 3, [])
    assert q["status"] == "open" and "clickup_id" not in q
    assert q["links"] == {"answers": [], "spawned_by": [], "pivoted_to": []}


def test_no_entries_section():
    text = CANONICAL.split("## Entries")[0]
    it = parse_item(text)
    assert it.entries == [] and it.description == "Why this matters."


def test_long_title_not_wrapped():
    title = " ".join(["word"] * 30)
    assert len(title) == 149
    item = parse_item(CANONICAL)
    item.meta["title"] = title
    out = render_item(item)
    assert [ln for ln in out.splitlines() if ln.startswith("title: ")] == [f"title: {title}"]
    assert render_item(parse_item(out)) == out


def test_invalid_yaml_frontmatter_rejected():
    bad = CANONICAL.replace("id: Q-20261112-a3f", "id: [unclosed")
    with pytest.raises(ItemFormatError, match="invalid frontmatter YAML"):
        parse_item(bad)


def test_non_mapping_frontmatter_rejected():
    with pytest.raises(ItemFormatError, match="frontmatter must be a mapping"):
        parse_item("---\n- a\n- b\n---\nbody\n")


def test_bad_entry_date_rejected():
    bad = CANONICAL.replace("- 2026-11-15 ·", "- 2026-13-45 ·")
    with pytest.raises(ItemFormatError, match="malformed entry date"):
        parse_item(bad)


@pytest.mark.parametrize("author, source", [("", "manual"), ("ali ce", "manual"), ("alice", ""), ("alice", "a b")])
def test_entry_render_rejects_whitespace_fields(author, source):
    with pytest.raises(ValueError):
        Entry(date(2026, 1, 1), author, source, "text").render()


def test_crlf_parses_like_lf():
    crlf = CANONICAL.replace("\n", "\r\n")
    it = parse_item(crlf)
    assert render_item(it) == CANONICAL
    assert it.entries[1].text == "Run finished: F=0.97."


def test_datetime_dates_coerced_to_date():
    text = CANONICAL.replace("raised: 2026-11-12", "raised: 2026-11-12 10:30:00").replace(
        "closed:\n", "closed: 2026-11-14T08:00:00\n").replace("custom_key: kept", "custom_key: kept\ndue: 2026-11-20 09:00:00")
    it = parse_item(text)
    assert type(it.meta["raised"]) is date and it.meta["raised"] == date(2026, 11, 12)
    assert type(it.meta["closed"]) is date and it.meta["closed"] == date(2026, 11, 14)
    assert type(it.meta["due"]) is date
