# Auto-Researcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tool that exhaustively searches physics/CS/chemistry/engineering literature to answer "has X been done before?" questions with a cited, synthesized report.

**Architecture:** A mechanical Python package (`auto-researcher/auto_researcher/`) queries free literature-search APIs (arXiv, OpenAlex, Semantic Scholar, CrossRef, CORE, Unpaywall), dedupes results, and fetches full text (open-access first, Cornell-proxy-with-cookies second). A Claude Code Workflow script and skill sit on top, doing all LLM-requiring work (query generation, relevance scoring, reading, synthesis) inside the session — no separate metered API billing.

**Tech Stack:** Python 3.12, `requests`, stdlib only otherwise (`xml.etree.ElementTree` for arXiv's Atom feed, `http.cookiejar` for cookie handling, `difflib` for fuzzy dedup matching), `pytest` for tests. Claude Code `Workflow` tool (JS) for the scoring/reading/synthesis pipeline.

**Spec:** `docs/superpowers/specs/2026-08-14-auto-researcher-design.md`

## Global Constraints

- No separate LLM API billing — all reading/reasoning happens via Claude Code subagents/Workflow in-session; the Python package makes zero LLM calls.
- Sources: arXiv, OpenAlex, Semantic Scholar, CrossRef, CORE, Unpaywall. No PubMed, no Google Scholar scraping.
- Paywalled full text (IEEE/ACM/Elsevier via Cornell) uses a manually-exported `cookies.txt` (Netscape format) — no interactive/headless login automation, since the tool runs over SSH with no browser available.
- Any single search source failing/rate-limiting must not fail the whole run; skip and continue.
- Missing/expired cookies degrade a paywalled paper to abstract-only, never blocks the run.
- Reports are saved to `auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `auto-researcher/requirements.txt`
- Create: `auto-researcher/.env.example`
- Create: `auto-researcher/.gitignore`
- Create: `auto-researcher/auto_researcher/__init__.py`
- Create: `auto-researcher/auto_researcher/search/__init__.py`
- Create: `auto-researcher/tests/__init__.py`
- Create: `auto-researcher/tests/search/__init__.py`

**Interfaces:**
- Produces: a `auto-researcher/.venv` virtualenv with `requests` and `pytest` installed, and an importable `auto_researcher` package (empty stub) that later tasks fill in.

- [ ] **Step 1: Create directory structure and empty package files**

```bash
mkdir -p auto-researcher/auto_researcher/search
mkdir -p auto-researcher/tests/search
touch auto-researcher/auto_researcher/__init__.py
touch auto-researcher/auto_researcher/search/__init__.py
touch auto-researcher/tests/__init__.py
touch auto-researcher/tests/search/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
requests>=2.31
pytest>=8.0
```

- [ ] **Step 3: Write .env.example**

```
OPENALEX_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
CORE_API_KEY=
CROSSREF_MAILTO=you@example.com
```

- [ ] **Step 4: Write .gitignore**

```
.venv/
.env
.cookies.txt
reports/
__pycache__/
*.pyc
```

- [ ] **Step 5: Create the virtualenv and install dependencies**

```bash
cd auto-researcher
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- [ ] **Step 6: Verify pytest runs (no tests yet, should report "no tests ran")**

Run: `cd auto-researcher && .venv/bin/pytest`
Expected: exits with "no tests ran" message, not an error.

- [ ] **Step 7: Commit**

```bash
git add auto-researcher/requirements.txt auto-researcher/.env.example auto-researcher/.gitignore auto-researcher/auto_researcher auto-researcher/tests
git commit -m "chore: scaffold auto-researcher package"
```

---

## Task 2: Paper model

**Files:**
- Create: `auto-researcher/auto_researcher/models.py`
- Test: `auto-researcher/tests/test_models.py`

**Interfaces:**
- Produces: `Paper` dataclass with fields `id, title, authors, year, venue, abstract, doi, arxiv_id, source, oa_pdf_url=None, landing_url=None`; `make_id(doi, arxiv_id, title, year) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/test_models.py
from auto_researcher.models import Paper, make_id


def test_make_id_prefers_doi():
    assert make_id("10.1000/xyz", "2101.00001", "Title", 2021) == "doi:10.1000/xyz"


def test_make_id_falls_back_to_arxiv():
    assert make_id(None, "2101.00001", "Title", 2021) == "arxiv:2101.00001"


def test_make_id_falls_back_to_title_hash():
    id1 = make_id(None, None, "Some Title", 2021)
    id2 = make_id(None, None, "Some Title", 2021)
    id3 = make_id(None, None, "Different Title", 2021)
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("title:")


def test_paper_dataclass_defaults():
    p = Paper(
        id="x", title="T", authors=[], year=2020, venue=None,
        abstract=None, doi=None, arxiv_id=None, source="test",
    )
    assert p.oa_pdf_url is None
    assert p.landing_url is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auto_researcher.models'`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/models.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Paper:
    id: str
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    abstract: Optional[str]
    doi: Optional[str]
    arxiv_id: Optional[str]
    source: str
    oa_pdf_url: Optional[str] = None
    landing_url: Optional[str] = None


def make_id(
    doi: Optional[str],
    arxiv_id: Optional[str],
    title: str,
    year: Optional[int],
) -> str:
    if doi:
        return f"doi:{doi.strip().lower()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip().lower()}"
    basis = f"{title.strip().lower()}|{year or ''}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"title:{digest}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/models.py auto-researcher/tests/test_models.py
git commit -m "feat: add Paper model and make_id helper"
```

---

## Task 3: HTTP retry helper

**Files:**
- Create: `auto-researcher/auto_researcher/http_utils.py`
- Test: `auto-researcher/tests/test_http_utils.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `request_with_retry(method, url, *, max_retries=3, backoff_seconds=1.0, **kwargs) -> requests.Response`, used by every search adapter and `fetch.py`.

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/test_http_utils.py
from unittest.mock import MagicMock, patch

from auto_researcher.http_utils import request_with_retry


def test_returns_successful_response():
    fake_resp = MagicMock(status_code=200)
    with patch("auto_researcher.http_utils.requests.request", return_value=fake_resp) as mock_req:
        resp = request_with_retry("GET", "https://example.com")
    assert resp.status_code == 200
    mock_req.assert_called_once()


def test_retries_on_429_then_succeeds():
    fail_resp = MagicMock(status_code=429)
    ok_resp = MagicMock(status_code=200)
    with patch(
        "auto_researcher.http_utils.requests.request",
        side_effect=[fail_resp, ok_resp],
    ):
        with patch("auto_researcher.http_utils.time.sleep"):
            resp = request_with_retry(
                "GET", "https://example.com", max_retries=3, backoff_seconds=0.01
            )
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_http_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auto_researcher.http_utils'`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/http_utils.py
from __future__ import annotations

import time

import requests


def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    **kwargs,
) -> requests.Response:
    timeout = kwargs.pop("timeout", 20)
    last_exc: Exception | None = None
    resp: requests.Response | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff_seconds * (2**attempt))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(backoff_seconds * (2**attempt))
            continue
        return resp
    if resp is not None:
        return resp
    assert last_exc is not None
    raise last_exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_http_utils.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/http_utils.py auto-researcher/tests/test_http_utils.py
git commit -m "feat: add HTTP retry helper for search adapters"
```

---

## Task 4: arXiv search adapter

**Files:**
- Create: `auto-researcher/auto_researcher/search/arxiv.py`
- Test: `auto-researcher/tests/search/test_arxiv.py`

**Interfaces:**
- Consumes: `Paper`, `make_id` from `auto_researcher.models` (Task 2); `request_with_retry` from `auto_researcher.http_utils` (Task 3).
- Produces: `search_arxiv(query: str, limit: int = 25) -> List[Paper]`.

- [ ] **Step 1: Write the failing test**

```python
# auto-researcher/tests/search/test_arxiv.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_arxiv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auto_researcher.search.arxiv'`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/arxiv.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def search_arxiv(query: str, limit: int = 25) -> List[Paper]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 100)}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers: List[Paper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
        if not title:
            continue
        summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip()
        id_url = entry.findtext(f"{ATOM_NS}id") or ""
        arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url.rsplit("/", 1)[-1]
        published = entry.findtext(f"{ATOM_NS}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext(f"{ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        pdf_url = None
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        papers.append(
            Paper(
                id=make_id(None, arxiv_id, title, year),
                title=title,
                authors=[a for a in authors if a],
                year=year,
                venue="arXiv",
                abstract=summary or None,
                doi=None,
                arxiv_id=arxiv_id,
                source="arxiv",
                oa_pdf_url=pdf_url,
                landing_url=id_url or None,
            )
        )
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_arxiv.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/arxiv.py auto-researcher/tests/search/test_arxiv.py
git commit -m "feat: add arXiv search adapter"
```

---

## Task 5: OpenAlex search adapter

**Files:**
- Create: `auto-researcher/auto_researcher/search/openalex.py`
- Test: `auto-researcher/tests/search/test_openalex.py`

**Interfaces:**
- Consumes: `Paper`, `make_id` (Task 2); `request_with_retry` (Task 3).
- Produces: `search_openalex(query: str, api_key: str, limit: int = 25) -> List[Paper]`; `_reconstruct_abstract(inverted_index: dict | None) -> str | None` (internal helper, also unit tested directly).

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/search/test_openalex.py
from unittest.mock import MagicMock, patch

from auto_researcher.search.openalex import _reconstruct_abstract, search_openalex

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Sample OpenAlex Paper",
            "publication_year": 2022,
            "doi": "https://doi.org/10.1000/sample",
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {"source": {"display_name": "Journal of Sample Studies"}},
            "open_access": {"oa_url": "https://example.com/sample.pdf"},
            "abstract_inverted_index": {"This": [0], "is": [1], "an": [2], "abstract": [3]},
            "id": "https://openalex.org/W123",
        }
    ]
}


def test_reconstruct_abstract_orders_words():
    assert (
        _reconstruct_abstract({"This": [0], "is": [1], "an": [2], "abstract": [3]})
        == "This is an abstract"
    )


def test_reconstruct_abstract_handles_none():
    assert _reconstruct_abstract(None) is None


def test_search_openalex_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.openalex.request_with_retry", return_value=fake_resp):
        papers = search_openalex("sample query", api_key="fake-key", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample OpenAlex Paper"
    assert p.doi == "10.1000/sample"
    assert p.authors == ["Ada Lovelace"]
    assert p.venue == "Journal of Sample Studies"
    assert p.abstract == "This is an abstract"
    assert p.oa_pdf_url == "https://example.com/sample.pdf"
    assert p.source == "openalex"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_openalex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auto_researcher.search.openalex'`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/openalex.py
from __future__ import annotations

from typing import List, Optional

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def search_openalex(query: str, api_key: str, limit: int = 25) -> List[Paper]:
    params = {"search": query, "per-page": min(limit, 200), "api_key": api_key}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("results", []):
        title = item.get("title") or ""
        if not title:
            continue
        authors = [
            a["author"]["display_name"]
            for a in item.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ]
        doi = item.get("doi")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        oa = item.get("open_access") or {}
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}

        papers.append(
            Paper(
                id=make_id(doi, None, title, item.get("publication_year")),
                title=title,
                authors=authors,
                year=item.get("publication_year"),
                venue=source.get("display_name"),
                abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
                doi=doi,
                arxiv_id=None,
                source="openalex",
                oa_pdf_url=oa.get("oa_url"),
                landing_url=item.get("id"),
            )
        )
    return papers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_openalex.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/openalex.py auto-researcher/tests/search/test_openalex.py
git commit -m "feat: add OpenAlex search adapter"
```

---

## Task 6: Semantic Scholar search adapter

**Files:**
- Create: `auto-researcher/auto_researcher/search/semantic_scholar.py`
- Test: `auto-researcher/tests/search/test_semantic_scholar.py`

**Interfaces:**
- Consumes: `Paper`, `make_id` (Task 2); `request_with_retry` (Task 3).
- Produces: `search_semantic_scholar(query: str, api_key: str | None = None, limit: int = 25) -> List[Paper]`.

- [ ] **Step 1: Write the failing test**

```python
# auto-researcher/tests/search/test_semantic_scholar.py
from unittest.mock import MagicMock, patch

from auto_researcher.search.semantic_scholar import search_semantic_scholar

SAMPLE_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Sample S2 Paper",
            "abstract": "An S2 abstract.",
            "year": 2020,
            "venue": "S2 Conference",
            "authors": [{"name": "Grace Hopper"}],
            "externalIds": {"DOI": "10.1000/s2", "ArXiv": "2001.00001"},
            "openAccessPdf": {"url": "https://example.com/s2.pdf"},
        }
    ]
}


def test_search_semantic_scholar_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch(
        "auto_researcher.search.semantic_scholar.request_with_retry", return_value=fake_resp
    ):
        papers = search_semantic_scholar("sample query", api_key=None, limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample S2 Paper"
    assert p.doi == "10.1000/s2"
    assert p.arxiv_id == "2001.00001"
    assert p.authors == ["Grace Hopper"]
    assert p.oa_pdf_url == "https://example.com/s2.pdf"
    assert p.landing_url == "https://www.semanticscholar.org/paper/abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_semantic_scholar.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/semantic_scholar.py
from __future__ import annotations

from typing import List, Optional

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,externalIds,venue,openAccessPdf"


def search_semantic_scholar(
    query: str, api_key: Optional[str] = None, limit: int = 25
) -> List[Paper]:
    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}
    resp = request_with_retry("GET", BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("data", []):
        title = item.get("title") or ""
        if not title:
            continue
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        oa_pdf = (item.get("openAccessPdf") or {}).get("url")
        paper_id = item.get("paperId")

        papers.append(
            Paper(
                id=make_id(doi, arxiv_id, title, item.get("year")),
                title=title,
                authors=[a.get("name") for a in item.get("authors", []) if a.get("name")],
                year=item.get("year"),
                venue=item.get("venue") or None,
                abstract=item.get("abstract"),
                doi=doi,
                arxiv_id=arxiv_id,
                source="semantic_scholar",
                oa_pdf_url=oa_pdf,
                landing_url=(
                    f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None
                ),
            )
        )
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_semantic_scholar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/semantic_scholar.py auto-researcher/tests/search/test_semantic_scholar.py
git commit -m "feat: add Semantic Scholar search adapter"
```

---

## Task 7: CrossRef search adapter

**Files:**
- Create: `auto-researcher/auto_researcher/search/crossref.py`
- Test: `auto-researcher/tests/search/test_crossref.py`

**Interfaces:**
- Consumes: `Paper`, `make_id` (Task 2); `request_with_retry` (Task 3).
- Produces: `search_crossref(query: str, mailto: str, limit: int = 25) -> List[Paper]`.

- [ ] **Step 1: Write the failing test**

```python
# auto-researcher/tests/search/test_crossref.py
from unittest.mock import MagicMock, patch

from auto_researcher.search.crossref import search_crossref

SAMPLE_RESPONSE = {
    "message": {
        "items": [
            {
                "title": ["Sample Crossref Paper"],
                "author": [{"given": "Alan", "family": "Turing"}],
                "published": {"date-parts": [[2019, 5]]},
                "container-title": ["Journal of Computation"],
                "DOI": "10.1000/crossref-sample",
                "URL": "https://doi.org/10.1000/crossref-sample",
            }
        ]
    }
}


def test_search_crossref_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.crossref.request_with_retry", return_value=fake_resp):
        papers = search_crossref("sample query", mailto="you@example.com", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample Crossref Paper"
    assert p.authors == ["Alan Turing"]
    assert p.year == 2019
    assert p.venue == "Journal of Computation"
    assert p.doi == "10.1000/crossref-sample"
    assert p.landing_url == "https://doi.org/10.1000/crossref-sample"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_crossref.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/crossref.py
from __future__ import annotations

from typing import List

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.crossref.org/works"


def search_crossref(query: str, mailto: str, limit: int = 25) -> List[Paper]:
    params = {"query": query, "rows": min(limit, 100), "mailto": mailto}
    resp = request_with_retry("GET", BASE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title") or []
        title = title_list[0] if title_list else ""
        if not title:
            continue

        authors = []
        for a in item.get("author", []) or []:
            name = " ".join(part for part in [a.get("given"), a.get("family")] if part)
            if name:
                authors.append(name)

        year = None
        date_parts = (item.get("published") or {}).get("date-parts")
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        venue_list = item.get("container-title") or []
        venue = venue_list[0] if venue_list else None
        doi = item.get("DOI")

        papers.append(
            Paper(
                id=make_id(doi, None, title, year),
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                abstract=None,
                doi=doi,
                arxiv_id=None,
                source="crossref",
                oa_pdf_url=None,
                landing_url=item.get("URL"),
            )
        )
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_crossref.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/crossref.py auto-researcher/tests/search/test_crossref.py
git commit -m "feat: add CrossRef search adapter"
```

---

## Task 8: CORE search adapter

**Files:**
- Create: `auto-researcher/auto_researcher/search/core.py`
- Test: `auto-researcher/tests/search/test_core.py`

**Interfaces:**
- Consumes: `Paper`, `make_id` (Task 2); `request_with_retry` (Task 3).
- Produces: `search_core(query: str, api_key: str, limit: int = 25) -> List[Paper]`.

- [ ] **Step 1: Write the failing test**

```python
# auto-researcher/tests/search/test_core.py
from unittest.mock import MagicMock, patch

from auto_researcher.search.core import search_core

SAMPLE_RESPONSE = {
    "results": [
        {
            "title": "Sample CORE Paper",
            "abstract": "A CORE abstract.",
            "authors": [{"name": "Marie Curie"}],
            "yearPublished": 2018,
            "doi": "10.1000/core-sample",
            "downloadUrl": "https://core.ac.uk/download/sample.pdf",
            "publisher": "Sample Press",
        }
    ]
}


def test_search_core_parses_results():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    with patch("auto_researcher.search.core.request_with_retry", return_value=fake_resp):
        papers = search_core("sample query", api_key="fake-key", limit=5)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Sample CORE Paper"
    assert p.abstract == "A CORE abstract."
    assert p.authors == ["Marie Curie"]
    assert p.doi == "10.1000/core-sample"
    assert p.oa_pdf_url == "https://core.ac.uk/download/sample.pdf"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_core.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/core.py
from __future__ import annotations

from typing import List

from ..http_utils import request_with_retry
from ..models import Paper, make_id

BASE_URL = "https://api.core.ac.uk/v3/search/works/"


def search_core(query: str, api_key: str, limit: int = 25) -> List[Paper]:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "limit": min(limit, 100)}
    resp = request_with_retry("GET", BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    papers: List[Paper] = []
    for item in data.get("results", []):
        title = item.get("title") or ""
        if not title:
            continue
        authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

        papers.append(
            Paper(
                id=make_id(item.get("doi"), None, title, item.get("yearPublished")),
                title=title,
                authors=authors,
                year=item.get("yearPublished"),
                venue=item.get("publisher"),
                abstract=item.get("abstract"),
                doi=item.get("doi"),
                arxiv_id=None,
                source="core",
                oa_pdf_url=item.get("downloadUrl"),
                landing_url=item.get("downloadUrl"),
            )
        )
    return papers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/core.py auto-researcher/tests/search/test_core.py
git commit -m "feat: add CORE search adapter"
```

---

## Task 9: Unpaywall lookup

**Files:**
- Create: `auto-researcher/auto_researcher/search/unpaywall.py`
- Test: `auto-researcher/tests/search/test_unpaywall.py`

**Interfaces:**
- Consumes: `request_with_retry` (Task 3). Note: this is a per-DOI lookup, not a search adapter, so it doesn't build a `Paper`.
- Produces: `find_oa_location(doi: str, email: str) -> str | None`.

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/search/test_unpaywall.py
from unittest.mock import MagicMock, patch

from auto_researcher.search.unpaywall import find_oa_location


def test_find_oa_location_returns_pdf_url():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "best_oa_location": {
            "url_for_pdf": "https://example.com/oa.pdf",
            "url": "https://example.com/oa",
        }
    }
    with patch("auto_researcher.search.unpaywall.request_with_retry", return_value=fake_resp):
        url = find_oa_location("10.1000/sample", email="you@example.com")
    assert url == "https://example.com/oa.pdf"


def test_find_oa_location_returns_none_on_404():
    fake_resp = MagicMock(status_code=404)
    with patch("auto_researcher.search.unpaywall.request_with_retry", return_value=fake_resp):
        url = find_oa_location("10.1000/missing", email="you@example.com")
    assert url is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_unpaywall.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/search/unpaywall.py
from __future__ import annotations

from typing import Optional

from ..http_utils import request_with_retry

BASE_URL = "https://api.unpaywall.org/v2/{doi}"


def find_oa_location(doi: str, email: str) -> Optional[str]:
    url = BASE_URL.format(doi=doi)
    resp = request_with_retry("GET", url, params={"email": email})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    best = data.get("best_oa_location") or {}
    return best.get("url_for_pdf") or best.get("url")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/search/test_unpaywall.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/search/unpaywall.py auto-researcher/tests/search/test_unpaywall.py
git commit -m "feat: add Unpaywall OA lookup"
```

---

## Task 10: Dedup

**Files:**
- Create: `auto-researcher/auto_researcher/dedup.py`
- Test: `auto-researcher/tests/test_dedup.py`

**Interfaces:**
- Consumes: `Paper` from `auto_researcher.models` (Task 2).
- Produces: `dedupe(papers: List[Paper], title_similarity_threshold: float = 0.92) -> List[Paper]`.

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/test_dedup.py
from auto_researcher.dedup import dedupe
from auto_researcher.models import Paper


def _paper(**kwargs):
    base = dict(
        id="x", title="T", authors=[], year=2020, venue=None,
        abstract=None, doi=None, arxiv_id=None, source="s",
    )
    base.update(kwargs)
    return Paper(**base)


def test_dedupe_merges_exact_id_matches():
    a = _paper(id="doi:10.1/x", title="A Paper", abstract=None, source="crossref")
    b = _paper(id="doi:10.1/x", title="A Paper", abstract="An abstract", source="semantic_scholar")
    result = dedupe([a, b])
    assert len(result) == 1
    assert result[0].abstract == "An abstract"


def test_dedupe_merges_fuzzy_title_matches_same_year():
    a = _paper(id="title:aaa", title="Neural Quantum States for Fermions", year=2022, source="arxiv")
    b = _paper(
        id="title:bbb", title="Neural Quantum States for Fermions.", year=2022, source="openalex"
    )
    result = dedupe([a, b])
    assert len(result) == 1


def test_dedupe_keeps_distinct_papers():
    a = _paper(id="title:aaa", title="Neural Quantum States for Fermions", year=2022)
    b = _paper(id="title:bbb", title="Coupled Cluster Methods for Hubbard Models", year=2022)
    result = dedupe([a, b])
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_dedup.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/dedup.py
from __future__ import annotations

import difflib
from typing import List

from .models import Paper


def _score(p: Paper) -> tuple:
    return (p.abstract is not None, p.doi is not None, p.oa_pdf_url is not None)


def _merge(a: Paper, b: Paper) -> Paper:
    winner = a if _score(a) >= _score(b) else b
    loser = b if winner is a else a
    return Paper(
        id=winner.id,
        title=winner.title,
        authors=winner.authors or loser.authors,
        year=winner.year or loser.year,
        venue=winner.venue or loser.venue,
        abstract=winner.abstract or loser.abstract,
        doi=winner.doi or loser.doi,
        arxiv_id=winner.arxiv_id or loser.arxiv_id,
        source=f"{winner.source}+{loser.source}" if winner.source != loser.source else winner.source,
        oa_pdf_url=winner.oa_pdf_url or loser.oa_pdf_url,
        landing_url=winner.landing_url or loser.landing_url,
    )


def dedupe(papers: List[Paper], title_similarity_threshold: float = 0.92) -> List[Paper]:
    by_id: dict[str, Paper] = {}
    for p in papers:
        by_id[p.id] = _merge(by_id[p.id], p) if p.id in by_id else p

    result: List[Paper] = []
    for p in by_id.values():
        match_idx = None
        for i, existing in enumerate(result):
            if existing.year != p.year:
                continue
            ratio = difflib.SequenceMatcher(
                None, existing.title.lower(), p.title.lower()
            ).ratio()
            if ratio >= title_similarity_threshold:
                match_idx = i
                break
        if match_idx is None:
            result.append(p)
        else:
            result[match_idx] = _merge(result[match_idx], p)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_dedup.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/dedup.py auto-researcher/tests/test_dedup.py
git commit -m "feat: add cross-source paper dedup"
```

---

## Task 11: Cookie store

**Files:**
- Create: `auto-researcher/auto_researcher/cookies.py`
- Test: `auto-researcher/tests/test_cookies.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (stdlib `http.cookiejar` only).
- Produces: `CookieStore(cookies_path: Path)` with `.is_fresh(domain: str) -> bool` and `.as_requests_cookies() -> dict[str, str]`. Used by Task 12 (`fetch.py`).

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/test_cookies.py
import time
from pathlib import Path

from auto_researcher.cookies import CookieStore

NETSCAPE_HEADER = "# Netscape HTTP Cookie File\n"


def _write_cookie_file(path: Path, domain: str, expires: int) -> None:
    line = f"{domain}\tTRUE\t/\tTRUE\t{expires}\tsession\tabc123\n"
    path.write_text(NETSCAPE_HEADER + line)


def test_is_fresh_true_for_unexpired_cookie(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", int(time.time()) + 3600)
    store = CookieStore(cookie_path)
    assert store.is_fresh("proxy.library.cornell.edu") is True


def test_is_fresh_false_for_expired_cookie(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    _write_cookie_file(cookie_path, ".proxy.library.cornell.edu", int(time.time()) - 3600)
    store = CookieStore(cookie_path)
    assert store.is_fresh("proxy.library.cornell.edu") is False


def test_missing_cookie_file_has_no_fresh_cookies(tmp_path):
    store = CookieStore(tmp_path / "missing.txt")
    assert store.is_fresh("anything.com") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cookies.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/cookies.py
from __future__ import annotations

import http.cookiejar
import time
from pathlib import Path
from typing import Dict


class CookieStore:
    def __init__(self, cookies_path: Path):
        self.cookies_path = Path(cookies_path)
        self.jar = http.cookiejar.MozillaCookieJar()
        if self.cookies_path.exists():
            self.jar.load(str(self.cookies_path), ignore_discard=True, ignore_expires=True)

    def is_fresh(self, domain: str) -> bool:
        now = time.time()
        for cookie in self.jar:
            if domain in cookie.domain and cookie.expires and cookie.expires > now:
                return True
        return False

    def as_requests_cookies(self) -> Dict[str, str]:
        return {c.name: c.value for c in self.jar if c.value is not None}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cookies.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/cookies.py auto-researcher/tests/test_cookies.py
git commit -m "feat: add cookie store for Cornell proxy sessions"
```

---

## Task 12: Full-text fetch

**Files:**
- Create: `auto-researcher/auto_researcher/fetch.py`
- Test: `auto-researcher/tests/test_fetch.py`

**Interfaces:**
- Consumes: `Paper` (Task 2); `request_with_retry` (Task 3); `CookieStore` (Task 11).
- Produces: `FullTextResult(paper_id, status, text=None)` dataclass with `status` in `{"open_access", "proxy", "unavailable"}`; `to_proxy_url(url: str) -> str`; `fetch_full_text(paper: Paper, cookie_store: CookieStore | None = None) -> FullTextResult`. Used by Task 13 (`cli.py`).

- [ ] **Step 1: Write the failing tests**

```python
# auto-researcher/tests/test_fetch.py
import time
from unittest.mock import MagicMock, patch

from auto_researcher.cookies import CookieStore
from auto_researcher.fetch import fetch_full_text, to_proxy_url
from auto_researcher.models import Paper


def _paper(**kwargs):
    base = dict(
        id="x", title="T", authors=[], year=2020, venue=None, abstract=None,
        doi=None, arxiv_id=None, source="s", oa_pdf_url=None, landing_url=None,
    )
    base.update(kwargs)
    return Paper(**base)


def test_to_proxy_url_rewrites_host():
    assert to_proxy_url("https://ieeexplore.ieee.org/document/123") == (
        "https://ieeexplore-ieee-org.proxy.library.cornell.edu/document/123"
    )


def test_fetch_full_text_prefers_open_access():
    paper = _paper(oa_pdf_url="https://example.com/oa.pdf")
    fake_resp = MagicMock(status_code=200, text="full text here")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.status == "open_access"
    assert result.text == "full text here"


def test_fetch_full_text_unavailable_without_oa_or_cookies():
    paper = _paper(landing_url="https://ieeexplore.ieee.org/document/123")
    result = fetch_full_text(paper, cookie_store=None)
    assert result.status == "unavailable"
    assert result.text is None


def test_fetch_full_text_uses_proxy_when_cookies_fresh(tmp_path):
    cookie_path = tmp_path / "cookies.txt"
    cookie_path.write_text(
        "# Netscape HTTP Cookie File\n"
        f"ieeexplore-ieee-org.proxy.library.cornell.edu\tTRUE\t/\tTRUE\t"
        f"{int(time.time()) + 3600}\tsession\tabc\n"
    )
    store = CookieStore(cookie_path)
    paper = _paper(landing_url="https://ieeexplore.ieee.org/document/123")
    fake_resp = MagicMock(status_code=200, text="proxied full text")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper, cookie_store=store)
    assert result.status == "proxy"
    assert result.text == "proxied full text"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/fetch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from .cookies import CookieStore
from .http_utils import request_with_retry
from .models import Paper

CORNELL_PROXY_SUFFIX = ".proxy.library.cornell.edu"


@dataclass
class FullTextResult:
    paper_id: str
    status: str  # "open_access", "proxy", "unavailable"
    text: Optional[str] = None


def to_proxy_url(url: str) -> str:
    parts = urlsplit(url)
    proxied_host = parts.netloc.replace(".", "-") + CORNELL_PROXY_SUFFIX
    return urlunsplit((parts.scheme, proxied_host, parts.path, parts.query, parts.fragment))


def fetch_full_text(
    paper: Paper, cookie_store: Optional[CookieStore] = None
) -> FullTextResult:
    if paper.oa_pdf_url:
        resp = request_with_retry("GET", paper.oa_pdf_url)
        if resp.status_code == 200:
            return FullTextResult(paper.id, "open_access", resp.text)

    if paper.landing_url and cookie_store is not None:
        proxy_url = to_proxy_url(paper.landing_url)
        proxy_domain = urlsplit(proxy_url).netloc
        if cookie_store.is_fresh(proxy_domain):
            resp = request_with_retry(
                "GET", proxy_url, cookies=cookie_store.as_requests_cookies()
            )
            if resp.status_code == 200:
                return FullTextResult(paper.id, "proxy", resp.text)

    return FullTextResult(paper.id, "unavailable", None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_fetch.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add auto-researcher/auto_researcher/fetch.py auto-researcher/tests/test_fetch.py
git commit -m "feat: add full-text fetch with OA-first, proxy-fallback logic"
```

---

## Task 13: CLI

**Files:**
- Create: `auto-researcher/auto_researcher/cli.py`
- Test: `auto-researcher/tests/test_cli.py`

**Interfaces:**
- Consumes: `Paper` (Task 2); `dedupe` (Task 10); `CookieStore` (Task 11); `fetch_full_text` (Task 12); `search_arxiv` (Task 4); `search_openalex` (Task 5); `search_semantic_scholar` (Task 6); `search_crossref` (Task 7); `search_core` (Task 8).
- Produces: `run_search(queries: List[str], limit: int, out_path: Path) -> None`; `run_fetch(candidates_path: Path, ids: List[str], out_dir: Path, cookies_path: Path) -> None`; a `python -m auto_researcher {search,fetch}` CLI. This is what the skill (Task 15) shells out to.

- [ ] **Step 1: Write the failing test**

```python
# auto-researcher/tests/test_cli.py
import json
from unittest.mock import patch

from auto_researcher.cli import run_search
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# auto-researcher/auto_researcher/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List

from .cookies import CookieStore
from .dedup import dedupe
from .fetch import fetch_full_text
from .models import Paper
from .search.arxiv import search_arxiv
from .search.core import search_core
from .search.crossref import search_crossref
from .search.openalex import search_openalex
from .search.semantic_scholar import search_semantic_scholar


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id, "title": p.title, "authors": p.authors, "year": p.year,
        "venue": p.venue, "abstract": p.abstract, "doi": p.doi,
        "arxiv_id": p.arxiv_id, "source": p.source,
        "oa_pdf_url": p.oa_pdf_url, "landing_url": p.landing_url,
    }


def _run_source(name: str, call: Callable[[], List[Paper]]) -> List[Paper]:
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - any source failure must not abort the run
        print(f"warning: {name} search failed, skipping: {exc}", file=sys.stderr)
        return []


def run_search(queries: List[str], limit: int, out_path: Path) -> None:
    papers: List[Paper] = []
    for query in queries:
        papers.extend(_run_source("arxiv", lambda: search_arxiv(query, limit=limit)))
        papers.extend(
            _run_source(
                "semantic_scholar",
                lambda: search_semantic_scholar(
                    query, api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"), limit=limit
                ),
            )
        )
        if os.environ.get("CROSSREF_MAILTO"):
            papers.extend(
                _run_source(
                    "crossref",
                    lambda: search_crossref(
                        query, mailto=os.environ["CROSSREF_MAILTO"], limit=limit
                    ),
                )
            )
        if os.environ.get("OPENALEX_API_KEY"):
            papers.extend(
                _run_source(
                    "openalex",
                    lambda: search_openalex(
                        query, api_key=os.environ["OPENALEX_API_KEY"], limit=limit
                    ),
                )
            )
        if os.environ.get("CORE_API_KEY"):
            papers.extend(
                _run_source(
                    "core",
                    lambda: search_core(query, api_key=os.environ["CORE_API_KEY"], limit=limit),
                )
            )

    deduped = dedupe(papers)
    out_path.write_text(json.dumps([_paper_to_dict(p) for p in deduped], indent=2))


def run_fetch(
    candidates_path: Path, ids: List[str], out_dir: Path, cookies_path: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(candidates_path.read_text())
    id_set = set(ids)
    cookie_store = CookieStore(cookies_path) if cookies_path.exists() else None

    manifest = {}
    for item in candidates:
        if item["id"] not in id_set:
            continue
        paper = Paper(**item)
        result = fetch_full_text(paper, cookie_store)
        manifest[paper.id] = result.status
        if result.text:
            safe_name = paper.id.replace(":", "_").replace("/", "_")
            (out_dir / f"{safe_name}.txt").write_text(result.text)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto_researcher")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search")
    search_p.add_argument("--query", action="append", required=True, dest="queries")
    search_p.add_argument("--limit", type=int, default=25)
    search_p.add_argument("--out", type=Path, required=True)

    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--in", type=Path, required=True, dest="candidates_path")
    fetch_p.add_argument("--ids", required=True)
    fetch_p.add_argument("--out-dir", type=Path, required=True)
    fetch_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"))

    args = parser.parse_args()
    if args.command == "search":
        run_search(args.queries, args.limit, args.out)
    elif args.command == "fetch":
        run_fetch(args.candidates_path, args.ids.split(","), args.out_dir, args.cookies)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to confirm nothing regressed**

Run: `cd auto-researcher && .venv/bin/pytest -v`
Expected: PASS (all tests across all tasks)

- [ ] **Step 6: Commit**

```bash
git add auto-researcher/auto_researcher/cli.py auto-researcher/tests/test_cli.py
git commit -m "feat: add CLI wiring search adapters, dedup, and fetch together"
```

---

## Task 14: Synthesis Workflow script

**Files:**
- Create: `.claude/workflows/research-synthesis.js`

**Interfaces:**
- Consumes: `args = { question: string, candidates: Array<{id, title, abstract, full_text_excerpt?}> }` (candidates come from Task 13's `search` output, optionally enriched with full text from Task 13's `fetch` output).
- Produces: `{ synthesis: string, extracts: Array<object>, totalCandidates: number, totalRanked: number }`. Consumed by the skill (Task 15).

This task has no automated test — `Workflow` scripts run subagents and can't be unit tested the way the Python package can. Verification is a manual dry run in Step 2.

- [ ] **Step 1: Write the workflow script**

```javascript
// .claude/workflows/research-synthesis.js
export const meta = {
  name: 'research-synthesis',
  description: 'Score, read, and synthesize a candidate paper pool into an answer to a research question',
  phases: [
    { title: 'Score' },
    { title: 'Read' },
    { title: 'Synthesize' },
  ],
}

const { question, candidates } = args

phase('Score')

const BATCH_SIZE = 10
const batches = []
for (let i = 0; i < candidates.length; i += BATCH_SIZE) {
  batches.push(candidates.slice(i, i + BATCH_SIZE))
}

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          relevance: { type: 'number' },
          reason: { type: 'string' },
        },
        required: ['id', 'relevance'],
      },
    },
  },
  required: ['scores'],
}

const scoredBatches = await parallel(
  batches.map(batch => () =>
    agent(
      `Question: "${question}"\n\nScore each paper's relevance to this question from 0 ` +
      `(irrelevant) to 10 (directly on-topic). Consider tangential/adjacent-field relevance ` +
      `too, not just exact keyword matches.\n\nPapers:\n` +
      JSON.stringify(batch.map(p => ({ id: p.id, title: p.title, abstract: p.abstract })), null, 2),
      { phase: 'Score', schema: SCORE_SCHEMA }
    )
  )
)

const scoreById = {}
for (const result of scoredBatches.filter(Boolean)) {
  for (const s of result.scores) {
    scoreById[s.id] = s
  }
}

const ranked = candidates
  .filter(p => scoreById[p.id] && scoreById[p.id].relevance >= 5)
  .sort((a, b) => scoreById[b.id].relevance - scoreById[a.id].relevance)
  .slice(0, 50)

log(`${ranked.length} papers selected out of ${candidates.length} candidates`)

phase('Read')

const EXTRACT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    summary: { type: 'string' },
    approach: { type: 'string' },
    relation_to_question: { type: 'string' },
  },
  required: ['id', 'summary', 'relation_to_question'],
}

const extracts = await pipeline(
  ranked,
  paper =>
    agent(
      `Question: "${question}"\n\nRead this paper and extract what's relevant to the question.\n\n` +
      `Title: ${paper.title}\nAbstract: ${paper.abstract || '(no abstract available)'}\n` +
      `${paper.full_text_excerpt ? `Full text excerpt:\n${paper.full_text_excerpt}` : '(no full text available)'}\n\n` +
      `Return a summary of what the paper does, its approach/method, and specifically how it ` +
      `relates to the question (does it answer it, partially address it, or is it just adjacent).`,
      { phase: 'Read', schema: EXTRACT_SCHEMA, label: `read:${paper.id}` }
    ).then(extract => ({ ...extract, paper }))
)

phase('Synthesize')

const validExtracts = extracts.filter(Boolean)

const synthesis = await agent(
  `Question: "${question}"\n\nBased on these paper extracts, write a synthesis answering the ` +
  `question directly. Structure your response as:\n\n` +
  `1. A direct answer (has this been done? fully / partially / not found) with a confidence ` +
  `level and why.\n` +
  `2. What's been done, organized by approach or theme.\n` +
  `3. Explicit gaps: what the question asks that nothing in this set addresses.\n\n` +
  `Extracts:\n` +
  JSON.stringify(
    validExtracts.map(e => ({
      id: e.id, summary: e.summary, approach: e.approach, relation: e.relation_to_question,
    })),
    null, 2
  ),
  { phase: 'Synthesize' }
)

return {
  synthesis,
  extracts: validExtracts,
  totalCandidates: candidates.length,
  totalRanked: ranked.length,
}
```

- [ ] **Step 2: Dry-run with a tiny synthetic candidate set to confirm the script parses and runs**

Run this via the `Workflow` tool with:
```json
{
  "scriptPath": ".claude/workflows/research-synthesis.js",
  "args": {
    "question": "Has anyone used gradient boosting for weather forecasting?",
    "candidates": [
      {"id": "test:1", "title": "Gradient Boosted Trees for Short-Term Weather Prediction", "abstract": "We apply XGBoost to hourly temperature forecasting."},
      {"id": "test:2", "title": "A Survey of Unrelated Topic", "abstract": "This paper is about knitting patterns."}
    ]
  }
}
```
Expected: the run completes, `totalRanked` is 1 (the knitting paper scores low and is filtered out), and `synthesis` is a non-empty string discussing the gradient boosting paper.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/research-synthesis.js
git commit -m "feat: add research-synthesis Workflow for score/read/synthesize stages"
```

---

## Task 15: Research-question skill

**Files:**
- Create: `.claude/skills/research-question/SKILL.md`

**Interfaces:**
- Consumes: `auto_researcher` CLI (Task 13, via Bash); `research-synthesis` Workflow (Task 14, via the `Workflow` tool).
- Produces: the end-user-facing entry point — a skill invoked to answer a research question and produce a report in `auto-researcher/reports/`.

This task has no automated test — it's an instructions file read by Claude, not executable code. Verification is a manual dry run in Step 2.

- [ ] **Step 1: Write the skill file**

```markdown
---
name: research-question
description: Exhaustively search physics/CS/chemistry/engineering literature to answer whether something has been done before, and write a cited report
---

# Research Question

Given a research question (e.g. "has X been applied to Y before?"), search
academic literature broadly, score and read the most relevant papers, and
write a synthesized, cited answer.

## Setup (one-time, per machine)

From `auto-researcher/`:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# fill in OPENALEX_API_KEY, CORE_API_KEY, SEMANTIC_SCHOLAR_API_KEY, CROSSREF_MAILTO in .env
```

## Steps

1. **Generate query variants.** From the user's question, write 4-8 distinct
   search query strings covering: the literal question's key terms, likely
   synonyms, adjacent subfields, and alternate phrasings an author in a
   different field might use. Don't just reuse the user's exact wording.

2. **Gather candidates.** Load `auto-researcher/.env` into the environment,
   then run (from `auto-researcher/`):
   ```bash
   .venv/bin/python -m auto_researcher search --query "q1" --query "q2" --limit 40 --out /tmp/candidates.json
   ```
   This writes a deduped JSON list of candidate papers (title/abstract/metadata,
   no full text yet) to `/tmp/candidates.json`.

3. **Run the synthesis workflow.** Read `/tmp/candidates.json`, then call the
   `Workflow` tool with:
   - `scriptPath`: `.claude/workflows/research-synthesis.js`
   - `args`: `{ "question": "<the user's question>", "candidates": <the JSON array from step 2> }`

   This scores relevance, reads the top ~50, and returns
   `{ synthesis, extracts, totalCandidates, totalRanked }`.

4. **Fetch full text for the highest-relevance papers (optional deepening).**
   If the synthesis flags specific papers as critical but abstract-only, run:
   ```bash
   .venv/bin/python -m auto_researcher fetch --in /tmp/candidates.json --ids id1,id2 --out-dir /tmp/fulltext --cookies .cookies.txt
   ```
   then attach the fetched text as `full_text_excerpt` on those candidates and
   re-run the synthesis workflow.

5. **Write the report.** Save the synthesis to
   `auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md` with the direct
   answer up top, the synthesis body, and a full bibliography (title, authors,
   year, venue, link — OA PDF link if available, otherwise DOI or landing
   page, flagged `abstract-only` if no full text was fetched). Also give the
   user the direct answer in chat.

## Notes

- If `auto-researcher/.cookies.txt` is missing or stale, `fetch.py` marks
  paywalled papers abstract-only automatically — this is expected, not an
  error. Tell the user which papers this affected if any looked important.
- To refresh `.cookies.txt`: log into the Cornell library proxy in a normal
  browser, export cookies for the relevant domains with a browser extension
  (e.g. "Get cookies.txt LOCALLY"), and `scp` the file to
  `auto-researcher/.cookies.txt` on the cluster. Sessions typically last
  hours, not days.
```

- [ ] **Step 2: Dry-run the skill end-to-end with a real question**

Invoke the skill with a concrete question you actually want answered, e.g.
`/research-question "Has anyone applied neural quantum states to the fermionic sign problem?"`,
and confirm: candidates.json is produced and non-empty, the Workflow
completes and returns a synthesis, and a report file is written to
`auto-researcher/reports/`.
Expected: report file exists, contains a direct answer, a bibliography with
working links, and reads as a coherent answer to the question asked.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/research-question/SKILL.md
git commit -m "feat: add research-question skill orchestrating search, fetch, and synthesis"
```
