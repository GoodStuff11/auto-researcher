import json
from pathlib import Path

import pytest

from auto_researcher.models import Paper
from auto_researcher.store import (
    append_paper_note,
    append_score,
    chunk_paper,
    get_candidates_by_ids,
    get_paper_notes,
    get_unscored_ids,
    has_fulltext,
    list_candidate_ids,
    list_chunks,
    list_scores,
    load_paper,
    load_scores,
    record_fulltext,
    safe_paper_id,
    upsert_paper,
)


def _paper(**kwargs):
    base = dict(
        id="arxiv:1234.5678", title="A Paper", authors=["A. Author"], year=2024,
        venue=None, abstract="An abstract.", doi=None, arxiv_id="1234.5678",
        source="arxiv", oa_pdf_url="https://arxiv.org/pdf/1234.5678", landing_url=None,
    )
    base.update(kwargs)
    return Paper(**base)


def test_safe_paper_id_replaces_colons_and_slashes():
    assert safe_paper_id("doi:10.1000/xyz") == "doi_10.1000_xyz"


def test_upsert_paper_writes_meta_and_abstract(tmp_path):
    paper = _paper()
    paper_dir = upsert_paper(tmp_path, paper)

    meta = json.loads((paper_dir / "meta.json").read_text())
    assert meta["id"] == "arxiv:1234.5678"
    assert meta["title"] == "A Paper"
    assert (paper_dir / "abstract.txt").read_text() == "An abstract."


def test_upsert_paper_merges_first_seen_wins_on_conflict(tmp_path):
    first = _paper(abstract="First abstract.", venue=None)
    second = _paper(abstract="Second abstract.", venue="NeurIPS")

    upsert_paper(tmp_path, first)
    upsert_paper(tmp_path, second)

    loaded = load_paper(tmp_path, "arxiv:1234.5678")
    assert loaded["abstract"] == "First abstract."  # first-seen wins when both present
    assert loaded["venue"] == "NeurIPS"  # gap-fill: first had no venue, second's is kept


def test_has_fulltext_false_before_record_true_after(tmp_path):
    paper = _paper()
    upsert_paper(tmp_path, paper)
    assert has_fulltext(tmp_path, paper.id) is False

    record_fulltext(tmp_path, paper.id, text="full text", status="open_access")
    assert has_fulltext(tmp_path, paper.id) is True


def test_record_fulltext_writes_pdf_bytes_when_given(tmp_path):
    paper = _paper()
    upsert_paper(tmp_path, paper)
    record_fulltext(
        tmp_path, paper.id, text="full text", status="open_access",
        source_url="https://arxiv.org/pdf/1234.5678", pdf_bytes=b"%PDF-fake-bytes",
    )
    paper_dir = tmp_path / "papers" / safe_paper_id(paper.id)
    assert (paper_dir / "fulltext.pdf").read_bytes() == b"%PDF-fake-bytes"
    assert (paper_dir / "fulltext.txt").read_text() == "full text"
    status = json.loads((paper_dir / "fetch_status.json").read_text())
    assert status["status"] == "open_access"
    assert status["source_url"] == "https://arxiv.org/pdf/1234.5678"


def test_load_paper_returns_none_when_never_upserted(tmp_path):
    assert load_paper(tmp_path, "doi:10.1/nonexistent") is None


def test_load_paper_includes_fulltext_and_status_once_fetched(tmp_path):
    paper = _paper()
    upsert_paper(tmp_path, paper)
    record_fulltext(tmp_path, paper.id, text="full text", status="proxy", source_url="https://x.example/y")

    loaded = load_paper(tmp_path, paper.id)
    assert loaded["fulltext"] == "full text"
    assert loaded["fetch_status"]["status"] == "proxy"


from auto_researcher.store import list_queries, load_query, record_query, record_scores, record_synthesis


def test_record_query_then_load_query_round_trips(tmp_path):
    papers = [_paper(id="arxiv:1", title="Paper One"), _paper(id="arxiv:2", title="Paper Two")]
    record_query(tmp_path, "fermion-sign-nqs", "Has X been done?", papers)

    loaded = load_query(tmp_path, "fermion-sign-nqs")
    assert loaded["question"] == "Has X been done?"
    assert [c["id"] for c in loaded["candidates"]] == ["arxiv:1", "arxiv:2"]
    assert loaded["relevant_ids"] == []
    assert loaded["synthesis"] is None


def test_record_scores_merges_relevance_into_existing_candidates(tmp_path):
    papers = [_paper(id="arxiv:1", title="Paper One"), _paper(id="arxiv:2", title="Paper Two")]
    record_query(tmp_path, "fermion-sign-nqs", "Has X been done?", papers)

    record_scores(tmp_path, "fermion-sign-nqs", [
        {"id": "arxiv:1", "relevance": 9, "reason": "directly on-topic"},
        {"id": "arxiv:2", "relevance": 1, "reason": "unrelated"},
    ])

    loaded = load_query(tmp_path, "fermion-sign-nqs")
    by_id = {c["id"]: c for c in loaded["candidates"]}
    assert by_id["arxiv:1"]["relevance"] == 9
    assert by_id["arxiv:1"]["reason"] == "directly on-topic"
    assert by_id["arxiv:2"]["relevance"] == 1


def test_record_synthesis_writes_relevant_ids_and_synthesis(tmp_path):
    papers = [_paper(id="arxiv:1", title="Paper One")]
    record_query(tmp_path, "fermion-sign-nqs", "Has X been done?", papers)

    record_synthesis(tmp_path, "fermion-sign-nqs", ["arxiv:1"], "# Direct answer\n\nYes, partially.")

    loaded = load_query(tmp_path, "fermion-sign-nqs")
    assert loaded["relevant_ids"] == ["arxiv:1"]
    assert loaded["synthesis"] == "# Direct answer\n\nYes, partially."


def test_load_query_returns_none_for_unknown_slug(tmp_path):
    assert load_query(tmp_path, "never-ran-this") is None


def test_list_queries_lists_every_recorded_query(tmp_path):
    record_query(tmp_path, "topic-a", "Question A?", [_paper(id="arxiv:1")])
    record_query(tmp_path, "topic-b", "Question B?", [_paper(id="arxiv:2")])

    listed = list_queries(tmp_path)
    slugs = {q["topic_slug"] for q in listed}
    assert slugs == {"topic-a", "topic-b"}
    by_slug = {q["topic_slug"]: q for q in listed}
    assert by_slug["topic-a"]["question"] == "Question A?"


def test_list_queries_empty_before_any_query_recorded(tmp_path):
    assert list_queries(tmp_path) == []


def test_list_candidate_ids_and_get_candidates_by_ids(tmp_path):
    papers = [_paper(id="arxiv:1", title="Paper One"), _paper(id="arxiv:2", title="Paper Two")]
    record_query(tmp_path, "topic-a", "Question?", papers)

    assert list_candidate_ids(tmp_path, "topic-a") == ["arxiv:1", "arxiv:2"]
    fetched = get_candidates_by_ids(tmp_path, "topic-a", ["arxiv:2"])
    assert [c["id"] for c in fetched] == ["arxiv:2"]


def test_append_score_then_load_scores_keeps_latest_per_id(tmp_path):
    papers = [_paper(id="arxiv:1"), _paper(id="arxiv:2")]
    record_query(tmp_path, "topic-a", "Question?", papers)

    append_score(tmp_path, "topic-a", "arxiv:1", relevance=6, read_priority=2, reason="ok")
    append_score(tmp_path, "topic-a", "arxiv:2", relevance=1, read_priority=0, reason="unrelated")
    append_score(tmp_path, "topic-a", "arxiv:1", relevance=9, read_priority=8, reason="re-scored higher")

    scores = load_scores(tmp_path, "topic-a")
    assert scores["arxiv:1"] == {
        "id": "arxiv:1", "relevance": 9, "read_priority": 8, "reason": "re-scored higher",
    }
    assert scores["arxiv:2"]["relevance"] == 1


def test_get_unscored_ids_excludes_scored_papers(tmp_path):
    papers = [_paper(id="arxiv:1"), _paper(id="arxiv:2"), _paper(id="arxiv:3")]
    record_query(tmp_path, "topic-a", "Question?", papers)
    append_score(tmp_path, "topic-a", "arxiv:2", relevance=5)

    assert get_unscored_ids(tmp_path, "topic-a") == ["arxiv:1", "arxiv:3"]
    assert get_unscored_ids(tmp_path, "topic-a", limit=1) == ["arxiv:1"]


def test_list_scores_filters_and_sorts(tmp_path):
    papers = [_paper(id=f"arxiv:{i}") for i in range(3)]
    record_query(tmp_path, "topic-a", "Question?", papers)
    append_score(tmp_path, "topic-a", "arxiv:0", relevance=9, read_priority=2)
    append_score(tmp_path, "topic-a", "arxiv:1", relevance=3, read_priority=8)
    append_score(tmp_path, "topic-a", "arxiv:2", relevance=1, read_priority=0)

    by_relevance = list_scores(tmp_path, "topic-a", min_relevance=3)
    assert [s["id"] for s in by_relevance] == ["arxiv:0", "arxiv:1"]

    by_read_priority = list_scores(tmp_path, "topic-a", sort_by="read_priority", top=1)
    assert [s["id"] for s in by_read_priority] == ["arxiv:1"]


def test_load_query_merges_scores_jsonl_into_candidates(tmp_path):
    papers = [_paper(id="arxiv:1")]
    record_query(tmp_path, "topic-a", "Question?", papers)
    append_score(tmp_path, "topic-a", "arxiv:1", relevance=7, read_priority=9, reason="core")

    loaded = load_query(tmp_path, "topic-a")
    candidate = loaded["candidates"][0]
    assert candidate["relevance"] == 7
    assert candidate["read_priority"] == 9
    assert candidate["reason"] == "core"


def test_chunk_paper_splits_fulltext_into_word_bounded_files(tmp_path):
    paper = _paper(id="arxiv:1")
    upsert_paper(tmp_path, paper)
    paper_dir = tmp_path / "papers" / safe_paper_id("arxiv:1")
    (paper_dir / "fulltext.txt").write_text(" ".join(f"word{i}" for i in range(2500)))

    index = chunk_paper(tmp_path, "arxiv:1", chunk_words=1000)

    assert [c["n"] for c in index] == [1, 2, 3]
    assert index[0]["word_count"] == 1000
    assert index[2]["word_count"] == 500
    assert Path(index[0]["path"]).read_text().split()[0] == "word0"
    assert list_chunks(tmp_path, "arxiv:1") == index


def test_chunk_paper_raises_when_no_fulltext_stored(tmp_path):
    paper = _paper(id="arxiv:1")
    upsert_paper(tmp_path, paper)
    with pytest.raises(FileNotFoundError):
        chunk_paper(tmp_path, "arxiv:1")


def test_append_paper_note_then_get_paper_notes_accumulates(tmp_path):
    paper = _paper(id="arxiv:1")
    upsert_paper(tmp_path, paper)

    assert get_paper_notes(tmp_path, "arxiv:1") is None

    append_paper_note(tmp_path, "arxiv:1", 1, "First insight.")
    append_paper_note(tmp_path, "arxiv:1", 2, "Second insight.")

    notes = get_paper_notes(tmp_path, "arxiv:1")
    assert "## Chunk 1" in notes
    assert "First insight." in notes
    assert "## Chunk 2" in notes
    assert "Second insight." in notes


def test_get_topic_summary_and_build_bibliography(tmp_path):
    from auto_researcher.store import build_bibliography, get_topic_summary

    papers = [
        _paper(id="arxiv:1", title="Paper 1"),
        _paper(id="arxiv:2", title="Paper 2"),
        _paper(id="arxiv:3", title="Paper 3"),
    ]
    for p in papers:
        upsert_paper(tmp_path, p)
    record_query(tmp_path, "test-topic", "What is X?", papers)
    append_score(tmp_path, "test-topic", "arxiv:1", relevance=9, read_priority=8, reason="key")
    append_score(tmp_path, "test-topic", "arxiv:2", relevance=6, read_priority=3, reason="medium")

    summary = get_topic_summary(tmp_path, "test-topic", min_relevance=5.0, min_read_priority=7.0)
    assert summary["total_candidates"] == 3
    assert summary["total_scored"] == 2
    assert summary["unscored_count"] == 1
    assert summary["unscored_ids"] == ["arxiv:3"]
    assert summary["qualified_count"] == 2
    assert summary["deep_read_count"] == 1
    assert summary["deep_read_ids"] == ["arxiv:1"]

    bib = build_bibliography(tmp_path, "test-topic", max_read=2)
    assert len(bib) == 2
    assert bib[0]["id"] == "arxiv:1"
    assert bib[0]["read_status"] == "abstract_only"


def test_generate_report_markdown(tmp_path):
    from auto_researcher.store import generate_report_markdown

    papers = [_paper(id="arxiv:1", title="Paper 1")]
    upsert_paper(tmp_path, papers[0])
    record_query(tmp_path, "test-topic", "What is X?", papers)
    append_score(tmp_path, "test-topic", "arxiv:1", relevance=9, read_priority=8, reason="key")

    md = generate_report_markdown(
        root=tmp_path,
        topic_slug="test-topic",
        direct_answer="Direct answer here.",
        synthesis_body="Synthesis text here.",
        references=[{"n": 1, "authors": ["A. Author"], "year": 2024, "title": "Paper 1", "link": "https://arxiv.org/abs/arxiv:1"}],
        gaps_body="Explicit gaps here.",
    )

    assert "# What is X?" in md
    assert "## Direct answer" in md
    assert "Direct answer here." in md
    assert "## References" in md
    assert "[1] A. Author (2024)" in md
    assert "## Bibliography" in md

