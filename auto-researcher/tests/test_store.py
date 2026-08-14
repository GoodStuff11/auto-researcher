import json

from auto_researcher.models import Paper
from auto_researcher.store import (
    has_fulltext,
    load_paper,
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
