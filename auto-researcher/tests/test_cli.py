import json
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

    def fake_fetch_full_text(paper, cookie_store=None):
        if paper.id == "arxiv:1":
            raise RuntimeError("network exploded")
        return FullTextResult(paper.id, "open_access", "some text")

    with patch("auto_researcher.cli.fetch_full_text", side_effect=fake_fetch_full_text):
        run_fetch(candidates_path, ["arxiv:1", "arxiv:2"], out_dir, cookies_path)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["arxiv:1"] == "unavailable"
    assert manifest["arxiv:2"] == "open_access"

    assert (out_dir / "arxiv_2.txt").read_text() == "some text"
    assert not (out_dir / "arxiv_1.txt").exists()
