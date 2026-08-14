from unittest.mock import MagicMock, patch

from auto_researcher.search.arxiv import search_arxiv

SAMPLE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <title>A Sample Paper Title</title>
    <summary>This is the abstract.</summary>
    <published>2021-01-01T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2101.00001v1" rel="related"/>
  </entry>
</feed>"""


def test_search_arxiv_parses_entries():
    fake_resp = MagicMock(status_code=200, text=SAMPLE_ATOM)
    with patch("auto_researcher.search.arxiv.request_with_retry", return_value=fake_resp):
        papers = search_arxiv("sample query", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "A Sample Paper Title"
    assert p.abstract == "This is the abstract."
    assert p.arxiv_id == "2101.00001v1"
    assert p.authors == ["Jane Doe", "John Smith"]
    assert p.oa_pdf_url == "http://arxiv.org/pdf/2101.00001v1"
    assert p.year == 2021
    assert p.source == "arxiv"
