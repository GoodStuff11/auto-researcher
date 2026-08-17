import json
import time
from unittest.mock import patch

from auto_researcher.cli import run_fetch, run_search
from auto_researcher.fetch import FullTextResult
from auto_researcher.models import Paper


def _paper(id_, title):
    return Paper(
        id=id_, title=title, authors=[], year=2021, venue=None,
        abstract=None, doi=None, arxiv_id=None, source="test",
    )


def test_run_search_writes_deduped_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    out_path = tmp_path / "candidates.json"

    with patch(
        "auto_researcher.cli.search_arxiv", return_value=[_paper("arxiv:1", "Paper One")]
    ), patch(
        "auto_researcher.cli.search_semantic_scholar",
        return_value=[_paper("arxiv:1", "Paper One")],
    ):
        run_search(["test query"], limit=10, out_path=out_path)

    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["title"] == "Paper One"


def test_run_search_skips_a_failing_source_without_crashing(tmp_path, monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    out_path = tmp_path / "candidates.json"

    with patch(
        "auto_researcher.cli.search_arxiv", side_effect=RuntimeError("arxiv is down")
    ), patch(
        "auto_researcher.cli.search_semantic_scholar",
        return_value=[_paper("arxiv:1", "Paper One")],
    ):
        run_search(["test query"], limit=10, out_path=out_path)

    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["title"] == "Paper One"


def test_run_search_persists_candidates_to_store_when_topic_given(tmp_path, monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    out_path = tmp_path / "candidates.json"
    store_root = tmp_path / "store"

    with patch(
        "auto_researcher.cli.search_arxiv", return_value=[_paper("arxiv:1", "Paper One")]
    ), patch(
        "auto_researcher.cli.search_semantic_scholar", return_value=[]
    ):
        run_search(
            ["test query"], limit=10, out_path=out_path,
            topic="my-topic", question="Has X been done?", store_root=store_root,
        )

    from auto_researcher.store import load_paper, load_query
    assert load_paper(store_root, "arxiv:1")["title"] == "Paper One"
    query = load_query(store_root, "my-topic")
    assert query["question"] == "Has X been done?"
    assert [c["id"] for c in query["candidates"]] == ["arxiv:1"]


def test_run_search_skips_store_when_topic_not_given(tmp_path, monkeypatch):
    monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    out_path = tmp_path / "candidates.json"
    store_root = tmp_path / "store"

    with patch(
        "auto_researcher.cli.search_arxiv", return_value=[_paper("arxiv:1", "Paper One")]
    ), patch(
        "auto_researcher.cli.search_semantic_scholar", return_value=[]
    ):
        run_search(["test query"], limit=10, out_path=out_path, store_root=store_root)

    assert not store_root.exists()


def test_run_fetch_marks_failing_paper_unavailable_without_crashing(tmp_path):
    candidates = [
        {
            "id": "arxiv:1",
            "title": "Paper One",
            "authors": [],
            "year": 2021,
            "venue": None,
            "abstract": None,
            "doi": None,
            "arxiv_id": None,
            "source": "test",
            "oa_pdf_url": None,
            "landing_url": None,
        },
        {
            "id": "arxiv:2",
            "title": "Paper Two",
            "authors": [],
            "year": 2021,
            "venue": None,
            "abstract": None,
            "doi": None,
            "arxiv_id": None,
            "source": "test",
            "oa_pdf_url": None,
            "landing_url": None,
        },
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))
    out_dir = tmp_path / "out"
    cookies_path = tmp_path / "cookies.txt"

    def fake_fetch_full_text(paper, cookie_store=None, email=None):
        if paper.id == "arxiv:1":
            raise RuntimeError("network exploded")
        return FullTextResult(paper.id, "open_access", "some text")

    with patch("auto_researcher.cli.fetch_full_text", side_effect=fake_fetch_full_text):
        run_fetch(candidates_path, ["arxiv:1", "arxiv:2"], out_dir, cookies_path, store_root=tmp_path / "store")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["arxiv:1"] == "unavailable"
    assert manifest["arxiv:2"] == "open_access"

    assert (out_dir / "arxiv_2.txt").read_text() == "some text"
    assert not (out_dir / "arxiv_1.txt").exists()


def test_run_fetch_reuses_cached_fulltext_without_calling_fetch_full_text(tmp_path):
    from auto_researcher import store

    candidates = [
        {
            "id": "arxiv:1", "title": "Paper One", "authors": [], "year": 2021,
            "venue": None, "abstract": None, "doi": None, "arxiv_id": "1",
            "source": "test", "oa_pdf_url": None, "landing_url": None,
        }
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))
    out_dir = tmp_path / "out"
    store_root = tmp_path / "store"

    store.upsert_paper(store_root, Paper(**candidates[0]))
    store.record_fulltext(store_root, "arxiv:1", text="cached text", status="open_access")

    with patch("auto_researcher.cli.fetch_full_text") as mock_fetch:
        run_fetch(candidates_path, ["arxiv:1"], out_dir, tmp_path / "missing-cookies.txt", store_root=store_root)

    mock_fetch.assert_not_called()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["arxiv:1"] == "open_access"
    assert (out_dir / "arxiv_1.txt").read_text() == "cached text"


def test_run_fetch_records_new_fetch_into_store(tmp_path):
    from auto_researcher import store
    from auto_researcher.fetch import FullTextResult

    candidates = [
        {
            "id": "arxiv:2", "title": "Paper Two", "authors": [], "year": 2021,
            "venue": None, "abstract": None, "doi": None, "arxiv_id": "2",
            "source": "test", "oa_pdf_url": "https://example.com/2.pdf", "landing_url": None,
        }
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))
    out_dir = tmp_path / "out"
    store_root = tmp_path / "store"

    with patch(
        "auto_researcher.cli.fetch_full_text",
        return_value=FullTextResult("arxiv:2", "open_access", "fresh text", b"%PDF-bytes"),
    ):
        run_fetch(candidates_path, ["arxiv:2"], out_dir, tmp_path / "missing-cookies.txt", store_root=store_root)

    loaded = store.load_paper(store_root, "arxiv:2")
    assert loaded["fulltext"] == "fresh text"
    assert loaded["fetch_status"]["status"] == "open_access"
    paper_dir = store_root / "papers" / store.safe_paper_id("arxiv:2")
    assert (paper_dir / "fulltext.pdf").read_bytes() == b"%PDF-bytes"


def test_run_fetch_filters_extra_keys_from_store_show_enriched_candidates(tmp_path):
    from auto_researcher.fetch import FullTextResult

    candidates = [
        {
            "id": "arxiv:1", "title": "Paper One", "authors": [], "year": 2021,
            "venue": None, "abstract": None, "doi": None, "arxiv_id": "1",
            "source": "test", "oa_pdf_url": None, "landing_url": None,
            "relevance": 8, "reason": "on-topic",
        }
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps(candidates))
    out_dir = tmp_path / "out"
    cookies_path = tmp_path / "cookies.txt"
    store_root = tmp_path / "store"

    with patch(
        "auto_researcher.cli.fetch_full_text",
        return_value=FullTextResult("arxiv:1", "open_access", "some text"),
    ):
        run_fetch(candidates_path, ["arxiv:1"], out_dir, cookies_path, store_root=store_root)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["arxiv:1"] == "open_access"
    assert (out_dir / "arxiv_1.txt").read_text() == "some text"


def test_run_store_record_scores_merges_into_candidates(tmp_path):
    from auto_researcher.cli import run_store_record_scores
    from auto_researcher.store import load_query, record_query

    store_root = tmp_path / "store"
    record_query(store_root, "my-topic", "Has X?", [_paper("arxiv:1", "Paper One")])

    run_store_record_scores(store_root, "my-topic", [{"id": "arxiv:1", "relevance": 8, "reason": "on-topic"}])

    loaded = load_query(store_root, "my-topic")
    assert loaded["candidates"][0]["relevance"] == 8


def test_run_store_record_synthesis_writes_relevant_ids_and_synthesis(tmp_path):
    from auto_researcher.cli import run_store_record_synthesis
    from auto_researcher.store import load_query, record_query

    store_root = tmp_path / "store"
    record_query(store_root, "my-topic", "Has X?", [_paper("arxiv:1", "Paper One")])

    run_store_record_synthesis(store_root, "my-topic", ["arxiv:1"], "# Answer\n\nYes.")

    loaded = load_query(store_root, "my-topic")
    assert loaded["relevant_ids"] == ["arxiv:1"]
    assert loaded["synthesis"] == "# Answer\n\nYes."


def test_run_store_show_prints_query_json(tmp_path, capsys):
    from auto_researcher.cli import run_store_show
    from auto_researcher.store import record_query

    store_root = tmp_path / "store"
    record_query(store_root, "my-topic", "Has X?", [_paper("arxiv:1", "Paper One")])

    run_store_show(store_root, "my-topic")

    printed = json.loads(capsys.readouterr().out)
    assert printed["question"] == "Has X?"
    assert printed["candidates"][0]["id"] == "arxiv:1"


def test_run_store_show_paper_prints_paper_json(tmp_path, capsys):
    from auto_researcher.cli import run_store_show_paper
    from auto_researcher.store import upsert_paper

    store_root = tmp_path / "store"
    upsert_paper(store_root, _paper("arxiv:1", "Paper One"))

    run_store_show_paper(store_root, "arxiv:1")

    printed = json.loads(capsys.readouterr().out)
    assert printed["title"] == "Paper One"


def test_run_cookies_status_true_for_domain_with_fresh_proxy_cookie(tmp_path):
    from auto_researcher.cli import run_cookies_status

    cookies_path = tmp_path / "cookies.txt"
    cookies_path.write_text(
        "# Netscape HTTP Cookie File\n"
        f".proxy.library.cornell.edu\tTRUE\t/\tTRUE\t{int(time.time()) + 3600}\tsession\tabc\n"
    )

    result = run_cookies_status(["ieeexplore.ieee.org", "dl.acm.org"], cookies_path)

    assert result == {"ieeexplore.ieee.org": True, "dl.acm.org": True}


def test_run_cookies_status_false_when_cookies_file_missing(tmp_path):
    from auto_researcher.cli import run_cookies_status

    result = run_cookies_status(["ieeexplore.ieee.org"], tmp_path / "missing.txt")

    assert result == {"ieeexplore.ieee.org": False}


def test_run_cookies_refresh_prints_manual_instructions_without_local_browser(tmp_path, capsys):
    from auto_researcher.cli import run_cookies_refresh

    with patch("auto_researcher.cli.is_local_browser_available", return_value=False):
        run_cookies_refresh(["ieeexplore.ieee.org"], tmp_path / "cookies.txt")

    out = capsys.readouterr().out
    assert "No local browser is available" in out
    assert "ieeexplore-ieee-org.proxy.library.cornell.edu" in out


def test_run_cookies_refresh_delegates_to_interactive_flow_with_local_browser(tmp_path):
    from auto_researcher.cli import run_cookies_refresh

    with patch("auto_researcher.cli.is_local_browser_available", return_value=True), patch(
        "auto_researcher.cli.refresh_cookies_interactive"
    ) as mock_refresh:
        run_cookies_refresh(["ieeexplore.ieee.org"], tmp_path / "cookies.txt")

    mock_refresh.assert_called_once_with(["ieeexplore.ieee.org"], tmp_path / "cookies.txt")


def test_run_store_list_prints_all_queries(tmp_path, capsys):
    from auto_researcher.cli import run_store_list
    from auto_researcher.store import record_query

    store_root = tmp_path / "store"
    record_query(store_root, "topic-a", "Question A?", [_paper("arxiv:1", "Paper One")])
    record_query(store_root, "topic-b", "Question B?", [_paper("arxiv:2", "Paper Two")])

    run_store_list(store_root)

    printed = json.loads(capsys.readouterr().out)
    assert {q["topic_slug"] for q in printed} == {"topic-a", "topic-b"}
