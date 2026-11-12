from datetime import date

from hub.items import Entry, default_meta, Item, parse_item, render_item
from hub.merge_item import main

D = date(2026, 11, 12)


def base_item():
    meta = default_meta("Q-20261112-abc", "question", "Why?", "alice", D, ["group"], 2, ["manual"])
    return Item(meta, "Because.", [Entry(D, "alice", "manual", "Raised.")])


def write(tmp_path, name, item):
    p = tmp_path / name
    p.write_text(item if isinstance(item, str) else render_item(item))
    return p


def run(tmp_path, o, a, b):
    paths = [write(tmp_path, n, it) for n, it in (("O", o), ("A", a), ("B", b))]
    code = main([str(p) for p in paths])
    return code, paths[1].read_text()


def test_entries_union_sorted_by_date(tmp_path):
    o, a, b = base_item(), base_item(), base_item()
    a.entries.append(Entry(date(2026, 11, 14), "alice", "manual", "A later."))
    b.entries.append(Entry(date(2026, 11, 13), "bob", "commit:x@1", "B earlier."))
    b.entries.append(Entry(date(2026, 11, 14), "bob", "manual", "B same day."))
    code, out = run(tmp_path, o, a, b)
    assert code == 0
    texts = [e.text for e in parse_item(out).entries]
    assert texts == ["Raised.", "B earlier.", "A later.", "B same day."]


def test_duplicate_entries_kept_once(tmp_path):
    o, a, b = base_item(), base_item(), base_item()
    same = Entry(date(2026, 11, 13), "alice", "manual", "Same.")
    a.entries.append(same)
    b.entries.append(same)
    code, out = run(tmp_path, o, a, b)
    assert code == 0 and [e.text for e in parse_item(out).entries] == ["Raised.", "Same."]


def test_lists_union_and_one_sided_scalar_changes(tmp_path):
    o, a, b = base_item(), base_item(), base_item()
    a.meta["sources"].append("circleback:1")
    a.meta["links"]["spawned_by"].append("D-20261112-aaa")
    a.meta["status"] = "answered"
    b.meta["sources"].append("clickup:2")
    b.meta["series"].append("theory")
    b.meta["links"]["spawned_by"].append("D-20261112-bbb")
    b.meta["priority"] = 1
    b.meta["new_key"] = "from b"
    b.description = "Because, explained."
    code, out = run(tmp_path, o, a, b)
    assert code == 0
    m = parse_item(out)
    assert m.meta["sources"] == ["manual", "circleback:1", "clickup:2"]
    assert m.meta["series"] == ["group", "theory"]
    assert m.links("spawned_by") == ["D-20261112-aaa", "D-20261112-bbb"]
    assert m.status == "answered" and m.meta["priority"] == 1 and m.meta["new_key"] == "from b"
    assert m.description == "Because, explained."


def test_same_change_both_sides_is_fine(tmp_path):
    o, a, b = base_item(), base_item(), base_item()
    a.meta["title"] = b.meta["title"] = "Why, really?"
    code, out = run(tmp_path, o, a, b)
    assert code == 0 and parse_item(out).title == "Why, really?"


def test_real_conflict_keeps_ours_and_reports(tmp_path, capsys):
    o, a, b = base_item(), base_item(), base_item()
    a.meta["title"] = "Title A"
    b.meta["title"] = "Title B"
    a.description = "Desc A"
    b.description = "Desc B"
    a_text = render_item(a)
    code, out = run(tmp_path, o, a, b)
    err = capsys.readouterr().err
    assert code == 1 and out == a_text
    assert "hub-item conflict in Q-20261112-abc: title: ours=Title A theirs=Title B" in err
    assert "description: ours=Desc A theirs=Desc B" in err


def test_empty_base_add_add(tmp_path):
    a, b = base_item(), base_item()
    b.entries.append(Entry(D, "bob", "manual", "Raised."))
    code, out = run(tmp_path, "", a, b)
    assert code == 0 and len(parse_item(out).entries) == 2


def test_unparseable_input_exits_1(tmp_path, capsys):
    a = base_item()
    code, out = run(tmp_path, base_item(), a, "not an item\n")
    assert code == 1 and out == render_item(a)
    assert "cannot merge" in capsys.readouterr().err
