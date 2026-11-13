from datetime import date

from hub.items import Entry, parse_item, render_item
from hub.merge_item import main
from hub.store import Store

D = date(2026, 11, 12)


def test_conflict_writes_merged_item_keeping_both_sides(hub_root, tmp_path, capsys):
    q = Store(hub_root).create("question", "Base title", owner="alice", author="alice", day=D, source="m")
    base = Store(hub_root).get(q.id)
    a = parse_item(render_item(base))
    b = parse_item(render_item(base))
    a.meta["title"] = "Title A"
    b.meta["title"] = "Title B"
    a.entries.append(Entry(date(2026, 11, 13), "alice", "manual", "From A."))
    b.entries.append(Entry(date(2026, 11, 14), "bob", "manual", "From B."))
    b.meta["sources"] = b.meta["sources"] + ["circleback:x1"]
    paths = []
    for name, item in (("o", base), ("a", a), ("b", b)):
        p = tmp_path / f"{name}.md"
        p.write_text(render_item(item))
        paths.append(str(p))
    code = main(paths)
    err = capsys.readouterr().err
    merged = parse_item((tmp_path / "a.md").read_text())
    assert code == 1
    assert "title: A=Title A B=Title B" in err
    assert merged.title == "Title A"                       # conflicting key: A kept for the human to fix
    assert [e.text for e in merged.entries] == ["Raised.", "From A.", "From B."]
    assert "circleback:x1" in merged.meta["sources"]
