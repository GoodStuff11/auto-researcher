# Auto-Researcher Persistent Local Store + Follow-Up Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `auto-researcher` a persistent local store — a global cache of every paper ever fetched (never refetch the same paper across questions) and a durable per-question record of what was found/scored/read — plus a new `research-followup` skill to dig into a past question's papers long after the original run finished.

**Architecture:** Add one new plain-Python module, `auto_researcher/store.py`, that `search`/`fetch` write through and that a new `store` CLI subcommand group exposes for the skill layer to read/write score and synthesis results (which only exist after an LLM step, so the CLI can't produce them itself). No database — plain JSON/text files under `auto-researcher/store/`, matching this package's existing dependency-light, file-based style.

**Tech Stack:** Python stdlib (`json`, `pathlib`, `datetime`) — no new dependencies. JavaScript (existing Workflow script, one-line addition). Markdown (skill files).

**Spec:** `docs/superpowers/specs/2026-08-14-auto-researcher-persistent-store-design.md` (extends `docs/superpowers/specs/2026-08-14-auto-researcher-design.md`)

## Global Constraints

- Single user, sequential use — no concurrent-writer locking, no database server; plain files on disk.
- The store is additive: existing `search`/`fetch` CLI behavior (JSON in/out via `--out`/`--out-dir`) keeps working for direct/manual use; the store is a side effect, not a replacement interface.
- A paper already fetched (for any past question) must never be refetched — full-text retrieval must check the cache first.
- Zero LLM calls anywhere in the Python package (`auto_researcher/`) — inherited from the original design.
- No separate metered LLM API billing — all reasoning happens via Claude Code subagents/Workflow within an existing session.

---

## File Structure

- Create: `auto-researcher/auto_researcher/store.py` — the store module (paper cache + query records).
- Create: `auto-researcher/tests/test_store.py` — its tests.
- Modify: `auto-researcher/auto_researcher/fetch.py` — `FullTextResult` gains a `pdf_bytes` field.
- Modify: `auto-researcher/auto_researcher/cli.py` — `run_search`/`run_fetch` gain store side effects; new `run_store_*` functions + `store` subcommand group.
- Modify: `auto-researcher/tests/test_fetch.py`, `auto-researcher/tests/test_cli.py` — extended for the above.
- Modify: `.claude/workflows/research-synthesis.js` — return raw per-candidate scores (not just the ranked/read subset), so the skill can persist a full record.
- Modify: `.claude/skills/research-question/SKILL.md` — pick a topic slug up front, pass `--topic`/`--question` to `search`, persist scores/synthesis via the new `store` CLI subcommands.
- Create: `.claude/skills/research-followup/SKILL.md` — the new "dig deeper later" skill.
- Modify: `auto-researcher/README.md` — document the store, new CLI flags, and the follow-up skill.
- Modify: `auto-researcher/.gitignore` — ignore `store/` (local cache, not part of the repo).

---

### Task 1: Store module — global paper cache

**Files:**
- Create: `auto-researcher/auto_researcher/store.py`
- Test: `auto-researcher/tests/test_store.py`
- Modify: `auto-researcher/.gitignore`

**Interfaces:**
- Consumes: `auto_researcher.models.Paper` (existing dataclass with fields `id, title, authors, year, venue, abstract, doi, arxiv_id, source, oa_pdf_url, landing_url`).
- Produces: `safe_paper_id(paper_id: str) -> str`, `upsert_paper(root: Path, paper: Paper) -> Path`, `has_fulltext(root: Path, paper_id: str) -> bool`, `record_fulltext(root: Path, paper_id: str, text: Optional[str], status: str, source_url: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> None`, `load_paper(root: Path, paper_id: str) -> Optional[dict]` — all consumed by Task 3 (`fetch.py`/`cli.py` changes) and Task 5 (`cli.py`'s `run_fetch`).

- [ ] **Step 1: Write the failing tests**

Create `auto-researcher/tests/test_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'auto_researcher.store'`

- [ ] **Step 3: Implement the module**

Create `auto-researcher/auto_researcher/store.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import Paper


def safe_paper_id(paper_id: str) -> str:
    return paper_id.replace(":", "_").replace("/", "_")


def _paper_dir(root: Path, paper_id: str) -> Path:
    return Path(root) / "papers" / safe_paper_id(paper_id)


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id, "title": p.title, "authors": p.authors, "year": p.year,
        "venue": p.venue, "abstract": p.abstract, "doi": p.doi,
        "arxiv_id": p.arxiv_id, "source": p.source,
        "oa_pdf_url": p.oa_pdf_url, "landing_url": p.landing_url,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_paper(root: Path, paper: Paper) -> Path:
    paper_dir = _paper_dir(root, paper.id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    meta_path = paper_dir / "meta.json"

    new = _paper_to_dict(paper)
    if meta_path.exists():
        merged = json.loads(meta_path.read_text())
        for key, value in new.items():
            if not merged.get(key) and value:
                merged[key] = value
    else:
        merged = new
    meta_path.write_text(json.dumps(merged, indent=2))

    abstract_path = paper_dir / "abstract.txt"
    if paper.abstract and not abstract_path.exists():
        abstract_path.write_text(paper.abstract)

    return paper_dir


def has_fulltext(root: Path, paper_id: str) -> bool:
    return (_paper_dir(root, paper_id) / "fulltext.txt").exists()


def record_fulltext(
    root: Path,
    paper_id: str,
    text: Optional[str],
    status: str,
    source_url: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
) -> None:
    paper_dir = _paper_dir(root, paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    if pdf_bytes:
        (paper_dir / "fulltext.pdf").write_bytes(pdf_bytes)
    if text:
        (paper_dir / "fulltext.txt").write_text(text)
    (paper_dir / "fetch_status.json").write_text(
        json.dumps({"status": status, "source_url": source_url, "fetched_at": _now()}, indent=2)
    )


def load_paper(root: Path, paper_id: str) -> Optional[dict]:
    paper_dir = _paper_dir(root, paper_id)
    meta_path = paper_dir / "meta.json"
    if not meta_path.exists():
        return None

    result: dict = json.loads(meta_path.read_text())

    abstract_path = paper_dir / "abstract.txt"
    if abstract_path.exists():
        result["abstract"] = abstract_path.read_text()

    fulltext_path = paper_dir / "fulltext.txt"
    if fulltext_path.exists():
        result["fulltext"] = fulltext_path.read_text()

    status_path = paper_dir / "fetch_status.json"
    if status_path.exists():
        result["fetch_status"] = json.loads(status_path.read_text())

    return result
```

Add `store/` to `auto-researcher/.gitignore` (it's a local cache, not repo content):

```
.venv/
.env
.cookies.txt
reports/
store/
__pycache__/
*.pyc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_store.py -v`
Expected: PASS (7/7)

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/store.py tests/test_store.py .gitignore
git commit -m "feat: add global paper cache to the local store"
```

---

### Task 2: Store module — per-query records

**Files:**
- Modify: `auto-researcher/auto_researcher/store.py`
- Modify: `auto-researcher/tests/test_store.py`

**Interfaces:**
- Consumes: `_paper_to_dict` (private helper from Task 1, same file).
- Produces: `record_query(root: Path, topic_slug: str, question: str, candidates: List[Paper]) -> None`, `record_scores(root: Path, topic_slug: str, scores: List[dict]) -> None`, `record_synthesis(root: Path, topic_slug: str, relevant_ids: List[str], synthesis_md: str) -> None`, `load_query(root: Path, topic_slug: str) -> Optional[dict]`, `list_queries(root: Path) -> List[dict]` — all consumed by Task 4 (`run_search`), Task 6 (`store` CLI subcommands), and Task 9 (`research-followup` skill, via those subcommands).

- [ ] **Step 1: Write the failing tests**

Append to `auto-researcher/tests/test_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'record_query' from 'auto_researcher.store'`

- [ ] **Step 3: Implement**

Append to `auto-researcher/auto_researcher/store.py`:

```python
def _query_dir(root: Path, topic_slug: str) -> Path:
    return Path(root) / "queries" / topic_slug


def record_query(root: Path, topic_slug: str, question: str, candidates: List[Paper]) -> None:
    query_dir = _query_dir(root, topic_slug)
    query_dir.mkdir(parents=True, exist_ok=True)

    (query_dir / "question.txt").write_text(question)
    (query_dir / "candidates.json").write_text(
        json.dumps([_paper_to_dict(p) for p in candidates], indent=2)
    )

    created_path = query_dir / "created_at.txt"
    if not created_path.exists():
        created_path.write_text(_now())
    (query_dir / "updated_at.txt").write_text(_now())


def record_scores(root: Path, topic_slug: str, scores: List[dict]) -> None:
    query_dir = _query_dir(root, topic_slug)
    candidates_path = query_dir / "candidates.json"
    candidates = json.loads(candidates_path.read_text())

    score_by_id = {s["id"]: s for s in scores}
    for c in candidates:
        s = score_by_id.get(c["id"])
        if s:
            c["relevance"] = s.get("relevance")
            c["reason"] = s.get("reason")

    candidates_path.write_text(json.dumps(candidates, indent=2))
    (query_dir / "updated_at.txt").write_text(_now())


def record_synthesis(root: Path, topic_slug: str, relevant_ids: List[str], synthesis_md: str) -> None:
    query_dir = _query_dir(root, topic_slug)
    query_dir.mkdir(parents=True, exist_ok=True)
    (query_dir / "relevant_ids.json").write_text(json.dumps(relevant_ids, indent=2))
    (query_dir / "synthesis.md").write_text(synthesis_md)
    (query_dir / "updated_at.txt").write_text(_now())


def load_query(root: Path, topic_slug: str) -> Optional[dict]:
    query_dir = _query_dir(root, topic_slug)
    question_path = query_dir / "question.txt"
    if not question_path.exists():
        return None

    result: dict = {"topic_slug": topic_slug, "question": question_path.read_text()}

    candidates_path = query_dir / "candidates.json"
    result["candidates"] = json.loads(candidates_path.read_text()) if candidates_path.exists() else []

    relevant_path = query_dir / "relevant_ids.json"
    result["relevant_ids"] = json.loads(relevant_path.read_text()) if relevant_path.exists() else []

    synthesis_path = query_dir / "synthesis.md"
    result["synthesis"] = synthesis_path.read_text() if synthesis_path.exists() else None

    return result


def list_queries(root: Path) -> List[dict]:
    queries_dir = Path(root) / "queries"
    if not queries_dir.exists():
        return []

    result = []
    for slug_dir in sorted(queries_dir.iterdir()):
        question_path = slug_dir / "question.txt"
        if not question_path.exists():
            continue
        created_path = slug_dir / "created_at.txt"
        result.append({
            "topic_slug": slug_dir.name,
            "question": question_path.read_text(),
            "created_at": created_path.read_text() if created_path.exists() else None,
        })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_store.py -v`
Expected: PASS (13/13 — 7 from Task 1 + 6 new)

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/store.py tests/test_store.py
git commit -m "feat: add per-query records to the local store"
```

---

### Task 3: `fetch.py` — capture raw PDF bytes alongside extracted text

**Files:**
- Modify: `auto-researcher/auto_researcher/fetch.py`
- Modify: `auto-researcher/tests/test_fetch.py`

**Interfaces:**
- Consumes: existing `_is_pdf_response(resp) -> bool` (private helper, same file, unchanged).
- Produces: `FullTextResult` gains a 4th field `pdf_bytes: Optional[bytes] = None`, populated whenever a PDF response was successfully parsed. Consumed by Task 5 (`run_fetch` passes `result.pdf_bytes` to `store.record_fulltext`).

- [ ] **Step 1: Write the failing test**

Add to `auto-researcher/tests/test_fetch.py` (near `test_fetch_full_text_extracts_text_from_pdf_response`):

```python
def test_fetch_full_text_captures_raw_pdf_bytes():
    paper = _paper(oa_pdf_url="https://arxiv.org/pdf/1234.5678")
    pdf_bytes = _minimal_pdf_bytes()
    fake_resp = MagicMock(
        status_code=200,
        content=pdf_bytes,
        headers={"content-type": "application/pdf"},
    )
    fake_resp.text = pdf_bytes.decode("latin-1")
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.pdf_bytes == pdf_bytes


def test_fetch_full_text_html_response_has_no_pdf_bytes():
    paper = _paper(oa_pdf_url="https://example.com/oa-landing")
    fake_resp = MagicMock(
        status_code=200,
        content=b"<html>full text here</html>",
        text="full text here",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    with patch("auto_researcher.fetch.request_with_retry", return_value=fake_resp):
        result = fetch_full_text(paper)
    assert result.pdf_bytes is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_fetch.py -k pdf_bytes -v`
Expected: FAIL with `AttributeError: 'FullTextResult' object has no attribute 'pdf_bytes'`

- [ ] **Step 3: Implement**

In `auto-researcher/auto_researcher/fetch.py`, update the dataclass and both success-return sites:

```python
@dataclass
class FullTextResult:
    paper_id: str
    status: str  # "open_access", "proxy", "unavailable"
    text: Optional[str] = None
    pdf_bytes: Optional[bytes] = None
```

```python
    if oa_url:
        resp = request_with_retry("GET", oa_url)
        if resp.status_code == 200:
            text = _extract_text(resp)
            if text is not None:
                pdf_bytes = resp.content if _is_pdf_response(resp) else None
                return FullTextResult(paper.id, "open_access", text, pdf_bytes)

    if paper.landing_url and cookie_store is not None:
        proxy_url = to_proxy_url(paper.landing_url)
        proxy_domain = urlsplit(proxy_url).netloc
        if cookie_store.is_fresh(proxy_domain):
            resp = request_with_retry(
                "GET", proxy_url, cookies=cookie_store.as_requests_cookies()
            )
            if resp.status_code == 200:
                text = _extract_text(resp)
                if text is not None:
                    pdf_bytes = resp.content if _is_pdf_response(resp) else None
                    return FullTextResult(paper.id, "proxy", text, pdf_bytes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_fetch.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/fetch.py tests/test_fetch.py
git commit -m "feat: capture raw PDF bytes in FullTextResult"
```

---

### Task 4: `cli.py` — `run_search` persists into the store

**Files:**
- Modify: `auto-researcher/auto_researcher/cli.py`
- Modify: `auto-researcher/tests/test_cli.py`

**Interfaces:**
- Consumes: `store.upsert_paper`, `store.record_query` (Tasks 1-2).
- Produces: `run_search(queries, limit, out_path, topic=None, question=None, store_root=Path("store"))` — the two new keyword params default to `None`/`Path("store")` so existing direct calls (and Task 4's own new tests that don't care about the store) keep working unchanged. Task 4 only changes this function; the `search` subcommand's argparse wiring (making `--topic`/`--question` required on the actual CLI) is added in Task 6 alongside the new `store` subcommand group, at which point `main()` passes them through. Until Task 6 lands, the CLI still runs (just without store side effects, same as today) — only Task 4's direct-call tests exercise the new parameters.

- [ ] **Step 1: Write the failing test**

Add to `auto-researcher/tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -k "persists_candidates_to_store or skips_store" -v`
Expected: FAIL with `TypeError: run_search() got an unexpected keyword argument 'topic'`

- [ ] **Step 3: Implement**

In `auto-researcher/auto_researcher/cli.py`, add the import and update `run_search`:

```python
from . import store
```

(add alongside the existing `from .cookies import CookieStore` etc. imports)

```python
def run_search(
    queries: List[str],
    limit: int,
    out_path: Path,
    topic: str | None = None,
    question: str | None = None,
    store_root: Path = Path("store"),
) -> None:
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

    if topic:
        for p in deduped:
            store.upsert_paper(store_root, p)
        store.record_query(store_root, topic, question or "", deduped)
```

(only the final `if topic:` block and the new parameters are additions — the loop body above is unchanged, reproduced in full so the task is self-contained for whoever implements it)

Add `from __future__ import annotations` is already present at the top of `cli.py` — the `str | None` annotation is valid as written.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (all existing + 2 new)

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/cli.py tests/test_cli.py
git commit -m "feat: persist search candidates to the local store"
```

---

### Task 5: `cli.py` — `run_fetch` reuses cached full text and populates the store

**Files:**
- Modify: `auto-researcher/auto_researcher/cli.py`
- Modify: `auto-researcher/tests/test_cli.py`

**Interfaces:**
- Consumes: `store.has_fulltext`, `store.load_paper`, `store.record_fulltext`, `store.safe_paper_id` (Task 1); `FullTextResult.pdf_bytes` (Task 3).
- Produces: `run_fetch(candidates_path, ids, out_dir, cookies_path, store_root=Path("store"))` — behavior unchanged for callers that don't care about the store (same `--out-dir`/manifest.json output); a paper with `has_fulltext(store_root, id) == True` is served from the store without a network call.

- [ ] **Step 1: Update the existing failing-fetch test's store isolation, and write the failing tests**

`run_fetch`'s new `store_root` parameter defaults to the *relative* path
`Path("store")`, matching this package's existing convention for relative
defaults (`CookieStore`'s `.cookies.txt`). That means any test that calls
`run_fetch` without passing `store_root` explicitly will read/write a real
`store/` directory relative to wherever pytest's cwd is (i.e., inside the
actual `auto-researcher/` checkout) — polluting the repo with test
artifacts and leaking state between test runs. Every test that exercises
`run_fetch` must pass an explicit `store_root=tmp_path / "store"`.

Update the existing test in `auto-researcher/tests/test_cli.py`,
`test_run_fetch_marks_failing_paper_unavailable_without_crashing`, changing
its `run_fetch` call from:

```python
    with patch("auto_researcher.cli.fetch_full_text", side_effect=fake_fetch_full_text):
        run_fetch(candidates_path, ["arxiv:1", "arxiv:2"], out_dir, cookies_path)
```

to:

```python
    with patch("auto_researcher.cli.fetch_full_text", side_effect=fake_fetch_full_text):
        run_fetch(candidates_path, ["arxiv:1", "arxiv:2"], out_dir, cookies_path, store_root=tmp_path / "store")
```

Then add these two new tests to the same file (both already pass an
explicit `store_root`):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -k "reuses_cached_fulltext or records_new_fetch" -v`
Expected: FAIL with `TypeError: run_fetch() got an unexpected keyword argument 'store_root'`

- [ ] **Step 3: Implement**

In `auto-researcher/auto_researcher/cli.py`, replace `run_fetch` with:

```python
def run_fetch(
    candidates_path: Path,
    ids: List[str],
    out_dir: Path,
    cookies_path: Path,
    store_root: Path = Path("store"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(candidates_path.read_text())
    id_set = set(ids)
    cookie_store = CookieStore(cookies_path) if cookies_path.exists() else None
    email = os.environ.get("CROSSREF_MAILTO")

    manifest = {}
    for item in candidates:
        if item["id"] not in id_set:
            continue
        paper = Paper(**item)

        if store.has_fulltext(store_root, paper.id):
            cached = store.load_paper(store_root, paper.id)
            manifest[paper.id] = cached["fetch_status"]["status"]
            if cached.get("fulltext"):
                safe_name = store.safe_paper_id(paper.id)
                (out_dir / f"{safe_name}.txt").write_text(cached["fulltext"])
            continue

        try:
            result = fetch_full_text(paper, cookie_store, email=email)
        except Exception as exc:  # noqa: BLE001 - a single paper's fetch failure must not abort the batch
            print(
                f"warning: fetch failed for {paper.id}, marking unavailable: {exc}",
                file=sys.stderr,
            )
            manifest[paper.id] = "unavailable"
            store.record_fulltext(store_root, paper.id, text=None, status="unavailable")
            continue

        manifest[paper.id] = result.status
        if result.text:
            safe_name = store.safe_paper_id(paper.id)
            (out_dir / f"{safe_name}.txt").write_text(result.text)
        store.record_fulltext(
            store_root, paper.id, text=result.text, status=result.status,
            source_url=paper.oa_pdf_url or paper.landing_url, pdf_bytes=result.pdf_bytes,
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -v`
Expected: PASS (all existing + 2 new). The pre-existing
`test_run_fetch_marks_failing_paper_unavailable_without_crashing` test
(updated in Step 1 above to pass an explicit `store_root`) should still
pass with its original assertions unchanged — it exercises a real fetch
failure (not a cache hit), so `store.has_fulltext` returns `False` for it
and its behavior is otherwise identical to before this task.

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/cli.py tests/test_cli.py
git commit -m "feat: cache full-text fetches in the local store, skip refetching"
```

---

### Task 6: `cli.py` — `store` subcommand group (record-scores, record-synthesis, show, show-paper, list)

**Files:**
- Modify: `auto-researcher/auto_researcher/cli.py`
- Modify: `auto-researcher/tests/test_cli.py`

**Interfaces:**
- Consumes: `store.record_scores`, `store.record_synthesis`, `store.load_query`, `store.load_paper`, `store.list_queries` (Task 2).
- Produces: 5 new thin wrapper functions in `cli.py` — `run_store_record_scores(store_root, topic, scores)`, `run_store_record_synthesis(store_root, topic, relevant_ids, synthesis_md)`, `run_store_show(store_root, topic)`, `run_store_show_paper(store_root, paper_id)`, `run_store_list(store_root)` (the latter three `print(json.dumps(...))` to stdout) — and a `store` subcommand in `main()`'s argparse tree that these are consumed by (Task 8's `research-question` skill update, Task 9's new `research-followup` skill).

- [ ] **Step 1: Write the failing tests**

Add to `auto-researcher/tests/test_cli.py`:

```python
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


def test_run_store_list_prints_all_queries(tmp_path, capsys):
    from auto_researcher.cli import run_store_list
    from auto_researcher.store import record_query

    store_root = tmp_path / "store"
    record_query(store_root, "topic-a", "Question A?", [_paper("arxiv:1", "Paper One")])
    record_query(store_root, "topic-b", "Question B?", [_paper("arxiv:2", "Paper Two")])

    run_store_list(store_root)

    printed = json.loads(capsys.readouterr().out)
    assert {q["topic_slug"] for q in printed} == {"topic-a", "topic-b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd auto-researcher && .venv/bin/pytest tests/test_cli.py -k run_store -v`
Expected: FAIL with `ImportError: cannot import name 'run_store_record_scores' from 'auto_researcher.cli'`

- [ ] **Step 3: Implement**

In `auto-researcher/auto_researcher/cli.py`, add after `run_fetch`:

```python
def run_store_record_scores(store_root: Path, topic: str, scores: List[dict]) -> None:
    store.record_scores(store_root, topic, scores)


def run_store_record_synthesis(
    store_root: Path, topic: str, relevant_ids: List[str], synthesis_md: str
) -> None:
    store.record_synthesis(store_root, topic, relevant_ids, synthesis_md)


def run_store_show(store_root: Path, topic: str) -> None:
    print(json.dumps(store.load_query(store_root, topic), indent=2))


def run_store_show_paper(store_root: Path, paper_id: str) -> None:
    print(json.dumps(store.load_paper(store_root, paper_id), indent=2))


def run_store_list(store_root: Path) -> None:
    print(json.dumps(store.list_queries(store_root), indent=2))
```

Then update `main()` to add the `store` subcommand tree and dispatch. Replace the existing `main()` function with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(prog="auto_researcher")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search")
    search_p.add_argument("--query", action="append", required=True, dest="queries")
    search_p.add_argument("--limit", type=int, default=25)
    search_p.add_argument("--out", type=Path, required=True)
    search_p.add_argument("--topic", required=True)
    search_p.add_argument("--question", required=True)
    search_p.add_argument("--store-root", type=Path, default=Path("store"))

    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--in", type=Path, required=True, dest="candidates_path")
    fetch_p.add_argument("--ids", required=True)
    fetch_p.add_argument("--out-dir", type=Path, required=True)
    fetch_p.add_argument("--cookies", type=Path, default=Path(".cookies.txt"))
    fetch_p.add_argument("--store-root", type=Path, default=Path("store"))

    store_p = sub.add_parser("store")
    store_sub = store_p.add_subparsers(dest="store_command", required=True)

    rs_p = store_sub.add_parser("record-scores")
    rs_p.add_argument("--topic", required=True)
    rs_p.add_argument("--scores", type=Path, required=True)
    rs_p.add_argument("--store-root", type=Path, default=Path("store"))

    rsyn_p = store_sub.add_parser("record-synthesis")
    rsyn_p.add_argument("--topic", required=True)
    rsyn_p.add_argument("--relevant-ids", required=True)
    rsyn_p.add_argument("--synthesis", type=Path, required=True)
    rsyn_p.add_argument("--store-root", type=Path, default=Path("store"))

    show_p = store_sub.add_parser("show")
    show_p.add_argument("--topic", required=True)
    show_p.add_argument("--store-root", type=Path, default=Path("store"))

    show_paper_p = store_sub.add_parser("show-paper")
    show_paper_p.add_argument("--id", required=True, dest="paper_id")
    show_paper_p.add_argument("--store-root", type=Path, default=Path("store"))

    list_p = store_sub.add_parser("list")
    list_p.add_argument("--store-root", type=Path, default=Path("store"))

    args = parser.parse_args()
    if args.command == "search":
        run_search(
            args.queries, args.limit, args.out,
            topic=args.topic, question=args.question, store_root=args.store_root,
        )
    elif args.command == "fetch":
        run_fetch(
            args.candidates_path, args.ids.split(","), args.out_dir, args.cookies,
            store_root=args.store_root,
        )
    elif args.command == "store":
        if args.store_command == "record-scores":
            scores = json.loads(args.scores.read_text())
            run_store_record_scores(args.store_root, args.topic, scores)
        elif args.store_command == "record-synthesis":
            run_store_record_synthesis(
                args.store_root, args.topic, args.relevant_ids.split(","), args.synthesis.read_text()
            )
        elif args.store_command == "show":
            run_store_show(args.store_root, args.topic)
        elif args.store_command == "show-paper":
            run_store_show_paper(args.store_root, args.paper_id)
        elif args.store_command == "list":
            run_store_list(args.store_root)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd auto-researcher && .venv/bin/pytest tests/ -v`
Expected: PASS (full suite, no regressions)

- [ ] **Step 5: Commit**

```bash
git add auto_researcher/cli.py tests/test_cli.py
git commit -m "feat: add store subcommand group for scores/synthesis/show/list"
```

---

### Task 7: `research-synthesis.js` — return raw per-candidate scores

**Files:**
- Modify: `.claude/workflows/research-synthesis.js`

**Interfaces:**
- Consumes: existing `scoreById` object (already built in the Score phase, keyed by candidate id, values `{id, relevance, reason}`).
- Produces: the Workflow's return object gains a `scores` field: `Object.values(scoreById)`, an array covering every scored candidate (not just the ranked/read subset already returned as `extracts`). Consumed by Task 8 (the `research-question` skill persists this via `store record-scores`).

This task has no automated test — Workflow scripts aren't unit-testable, same as the rest of this file (established in the original implementation plan's Task 14). Verified by a live dry-run in Task 9's manual verification step, and again by the controller before merge.

- [ ] **Step 1: Make the change**

In `.claude/workflows/research-synthesis.js`, change the final `return` statement from:

```js
return {
  synthesis,
  extracts: validExtracts,
  totalCandidates: candidates.length,
  totalRanked: ranked.length,
}
```

to:

```js
return {
  synthesis,
  extracts: validExtracts,
  scores: Object.values(scoreById),
  totalCandidates: candidates.length,
  totalRanked: ranked.length,
}
```

- [ ] **Step 2: Sanity-check with a syntax check**

Run: `node --check .claude/workflows/research-synthesis.js`
Expected: no output (valid syntax). Full behavioral verification happens via a live `Workflow` tool invocation in Task 9's verification step, since this file has no test runner of its own.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/research-synthesis.js
git commit -m "feat: return raw per-candidate scores from the synthesis workflow"
```

---

### Task 8: `research-question` skill — pick a topic slug, persist scores and synthesis

**Files:**
- Modify: `.claude/skills/research-question/SKILL.md`

**Interfaces:**
- Consumes: `auto_researcher search --topic --question` (Task 4), `auto_researcher store record-scores`/`record-synthesis` (Task 6), the Workflow's new `scores` field (Task 7).
- Produces: the updated skill instructions — no code interface, this is the file a future Claude session reads to drive the pipeline. Consumed by Task 9's `research-followup` skill only insofar as it documents where records end up (`store/queries/<slug>/`).

This task has no automated test — same as the original plan's Task 15 for this same file. Verification is a manual dry run (Task 9's Step 2, after the follow-up skill exists to actually revisit what this run produces).

- [ ] **Step 1: Rewrite the skill file**

Replace the full contents of `.claude/skills/research-question/SKILL.md` with:

```markdown
---
name: research-question
description: Exhaustively search physics/CS/chemistry/engineering literature to answer whether something has been done before, and write a cited report
---

# Research Question

Given a research question (e.g. "has X been applied to Y before?"), search
academic literature broadly, score and read the most relevant papers, and
write a synthesized, cited answer. Everything found, scored, and read is
persisted to a local store (`auto-researcher/store/`) so it never needs to
be requeried — revisit it anytime later with the `research-followup` skill.

## Setup (one-time, per machine)

From `auto-researcher/`:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# fill in OPENALEX_API_KEY, CORE_API_KEY, SEMANTIC_SCHOLAR_API_KEY, CROSSREF_MAILTO in .env
```

## Steps

1. **Generate query variants and a topic slug.** From the user's question,
   write 4-8 distinct search query strings covering: the literal question's
   key terms, likely synonyms, adjacent subfields, and alternate phrasings
   an author in a different field might use. Also pick a short, filesystem-safe
   topic slug for this question (lowercase, hyphenated, e.g.
   `nqs-fermionic-sign-problem`) — it names this question's record for the
   rest of this run and for any later `research-followup` session.

2. **Gather candidates.** Load `auto-researcher/.env` into the environment,
   then run (from `auto-researcher/`):
   ```bash
   .venv/bin/python -m auto_researcher search \
     --query "q1" --query "q2" --limit 40 --out /tmp/candidates.json \
     --topic "<topic-slug>" --question "<the user's original question, verbatim>"
   ```
   This writes a deduped JSON list of candidate papers (title/abstract/metadata,
   no full text yet) to `/tmp/candidates.json`, AND persists every candidate
   plus this query's record into `auto-researcher/store/`.

3. **Run the synthesis workflow.** Read `/tmp/candidates.json`, then call the
   `Workflow` tool with:
   - `scriptPath`: `.claude/workflows/research-synthesis.js`
   - `args`: `{ "question": "<the user's question>", "candidates": <the JSON array from step 2> }`

   This scores relevance, reads the top ~50, and returns
   `{ synthesis, extracts, scores, totalCandidates, totalRanked }`.

4. **Persist the scores.** Write the `scores` array from step 3 to a temp
   file and run:
   ```bash
   .venv/bin/python -m auto_researcher store record-scores \
     --topic "<topic-slug>" --scores /tmp/scores.json
   ```

5. **Fetch full text for the highest-relevance papers (optional deepening).**
   If the synthesis flags specific papers as critical but abstract-only, run:
   ```bash
   .venv/bin/python -m auto_researcher fetch --in /tmp/candidates.json --ids id1,id2 --out-dir /tmp/fulltext --cookies .cookies.txt
   ```
   then attach the fetched text as `full_text_excerpt` on those candidates and
   re-run the synthesis workflow. A paper fetched here (or in any past run)
   is served from the store instantly on a future fetch — it is never
   downloaded twice.

6. **Persist the synthesis.** Write the final synthesis body to a temp
   markdown file, then run:
   ```bash
   .venv/bin/python -m auto_researcher store record-synthesis \
     --topic "<topic-slug>" --relevant-ids id1,id2,id3 --synthesis /tmp/synthesis.md
   ```
   (`--relevant-ids` is the comma-separated list of candidate ids from the
   workflow's `extracts` — the papers actually read and included.)

7. **Write the report.** Save the synthesis to
   `auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md` with the direct
   answer up top, the synthesis body, and a full bibliography (title, authors,
   year, venue, link — OA PDF link if available, otherwise DOI or landing
   page, flagged `abstract-only` if no full text was fetched). Also give the
   user the direct answer in chat, and mention that the full record is
   durably stored under `store/queries/<topic-slug>/` for later follow-up.

## Notes

- If `auto-researcher/.cookies.txt` is missing or stale, `fetch.py` marks
  paywalled papers `unavailable` (abstract-only) automatically — this is
  expected, not an error. Tell the user which papers this affected if any
  looked important.
- To refresh `.cookies.txt`: log into the Cornell library proxy in a normal
  browser, export cookies for the relevant domains with a browser extension
  (e.g. "Get cookies.txt LOCALLY"), and `scp` the file to
  `auto-researcher/.cookies.txt` on the cluster. Sessions typically last
  hours, not days.
- To dig into a specific paper from this question later, or ask something
  the original synthesis didn't cover, use the `research-followup` skill
  with this question's topic slug.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/research-question/SKILL.md
git commit -m "feat: persist scores and synthesis to the store from research-question"
```

---

### Task 9: `research-followup` skill — dig into a past query

**Files:**
- Create: `.claude/skills/research-followup/SKILL.md`

**Interfaces:**
- Consumes: `auto_researcher store list`/`store show`/`store show-paper` (Task 6), `auto_researcher fetch` (Task 5, for on-demand fetch of a previously-unread candidate).
- Produces: the new skill file. Terminal — nothing downstream depends on it within this plan.

This task has no automated test — same as `research-question`. This step includes the plan's manual verification of the *whole* feature (Tasks 1-9 together), since a follow-up skill is only meaningful once a real query record exists to follow up on.

- [ ] **Step 1: Write the skill file**

Create `.claude/skills/research-followup/SKILL.md`:

```markdown
---
name: research-followup
description: Dig deeper into a past research-question run - reread a specific paper, ask something the original synthesis didn't cover, or fetch a paper that wasn't read the first time
---

# Research Follow-Up

Revisits a research question answered earlier by the `research-question`
skill, using its stored record instead of rerunning the search from scratch.

## Steps

1. **Find the query.** If given a topic slug directly, use it. Otherwise,
   list past queries and match by question similarity:
   ```bash
   cd auto-researcher && .venv/bin/python -m auto_researcher store list
   ```
   This prints every stored query's `topic_slug`, `question`, and
   `created_at`. If more than one plausibly matches what the user is asking
   about, ask them which one they mean before proceeding.

2. **Load the record.**
   ```bash
   .venv/bin/python -m auto_researcher store show --topic "<topic-slug>"
   ```
   Returns `{ topic_slug, question, candidates, relevant_ids, synthesis }` —
   `candidates` includes every paper originally found (with `relevance`/
   `reason` if scored), `relevant_ids` are the ones read in depth, and
   `synthesis` is the original answer. Load this into context.

3. **Converse.** Answer the user's follow-up question against what's
   already loaded — the synthesis, the candidate list, and the relevance
   scores/reasons — without refetching anything by default.

4. **Look at a specific paper more closely, on request.** If the user
   asks about a paper by id or title:
   - If it's in `relevant_ids`, its full text may already be cached. Check:
     ```bash
     .venv/bin/python -m auto_researcher store show-paper --id "<paper-id>"
     ```
     If `fulltext` is present in the output, read and discuss it directly.
   - If it was only in `candidates` (found but never read in depth), offer
     to fetch and read it now:
     ```bash
     .venv/bin/python -m auto_researcher fetch --in <(python3 -c "import json,sys; print(json.dumps(json.load(open('/dev/stdin'))['candidates']))" < /tmp/query.json) --ids "<paper-id>" --out-dir /tmp/fulltext --cookies .cookies.txt
     ```
     (simpler in practice: write the `candidates` array from step 2 to
     `/tmp/candidates.json` first, then run
     `fetch --in /tmp/candidates.json --ids "<paper-id>" --out-dir /tmp/fulltext --cookies .cookies.txt`)
     This call persists the result into the store automatically — it will
     never need to be fetched again, even for a completely different
     future question. Read `/tmp/fulltext/<safe-id>.txt` and discuss it.
   - If the user wants this paper considered part of the record going
     forward, append its id to the relevant list and re-persist (reusing
     the *existing* synthesis text unchanged, since this skill does not
     regenerate the synthesis itself):
     ```bash
     .venv/bin/python -m auto_researcher store record-synthesis \
       --topic "<topic-slug>" \
       --relevant-ids "<old-relevant-ids-comma-joined>,<paper-id>" \
       --synthesis /tmp/original-synthesis.md
     ```
     (write the `synthesis` string from step 2's output to
     `/tmp/original-synthesis.md` first, unmodified.)

5. **If the user wants the full report redone** with new information
   (not just a targeted question answered), say so explicitly and defer to
   `research-question`'s own re-run flow (its Step 5) rather than
   regenerating a synthesis here — this skill is for targeted digging, not
   full re-synthesis.

## Notes

- Nothing here makes a new search API call — only `fetch` (step 4, and
  only when explicitly asked for a specific paper) touches the network,
  and even then only for a paper not already cached.
- If `store list` returns nothing, no `research-question` run has happened
  yet on this machine — there's nothing to follow up on.
```

- [ ] **Step 2: Live dry-run verification (manual, not automated)**

This step verifies Tasks 1-9 together, end to end, since the store and
follow-up skill are only meaningful in combination with a real prior run.

1. Run the full test suite one more time to confirm no regressions:
   `cd auto-researcher && .venv/bin/pytest -v` — expect all tests passing.
2. Invoke `/research-question` with a real question you want answered
   (reuse a question from a past dry-run if convenient) and confirm:
   - `auto-researcher/store/queries/<slug>/` now exists with `question.txt`,
     `candidates.json` (with `relevance`/`reason` fields merged in),
     `relevant_ids.json`, and `synthesis.md`.
   - `auto-researcher/store/papers/<id>/` exists for at least one candidate,
     with `meta.json` and `abstract.txt`.
   - If step 5 (optional fetch) was exercised, confirm the fetched paper's
     directory also has `fulltext.txt` and (if it was a PDF) `fulltext.pdf`.
3. Invoke `/research-followup` naming that same topic slug and confirm:
   - `store list` includes it.
   - `store show` returns the expected record.
   - Asking about a specific candidate not in `relevant_ids` triggers a
     fetch, and a second identical ask (or a completely unrelated future
     `research-question` run touching the same paper) does not refetch it
     — confirm by checking `store/papers/<id>/fetch_status.json`'s
     `fetched_at` timestamp doesn't change on the second ask.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/research-followup/SKILL.md
git commit -m "feat: add research-followup skill for revisiting past queries"
```

---

### Task 10: Update `auto-researcher/README.md`

**Files:**
- Modify: `auto-researcher/README.md`

**Interfaces:**
- Consumes: nothing new — documents Tasks 1-9's interfaces as built.
- Produces: nothing consumed by other tasks — this is the terminal documentation task.

No test — documentation only.

- [ ] **Step 1: Update the README**

Make these edits to `auto-researcher/README.md`:

1. In the **Architecture** diagram, add a line after the `search` step and
   before/around the `fetch` step noting persistence, e.g.:
   ```
     -> CLI `search`: queries all configured sources per variant, dedupes,
        persists every candidate + this query's record to the local store
        -> candidates.json (title/authors/year/venue/abstract/DOI/links,
           no full text yet)
   ```
   and after the optional `fetch` step:
   ```
     -> CLI `fetch` (optional, for papers flagged critical but abstract-only):
        fetches full text (OA link, then Unpaywall, then Cornell proxy,
        else stays abstract-only) - checks the local store first and never
        refetches a paper already retrieved for any past question ->
        re-run Score/Read/Synthesize with the added text
   ```

2. In **`auto_researcher search`**'s flag list, document the two new
   required flags and the store side effect:
   ```
   - `--topic` (required) - a short slug identifying this question; also
     names its record under `store/queries/<topic>/`.
   - `--question` (required) - the literal research question, stored
     verbatim alongside the record for later reference.
   - `--store-root` (default `store`) - where the local store lives.
   ```
   And add a short paragraph noting that `search` now also writes every
   candidate into the global paper cache (`store/papers/`) and this
   query's own record (`store/queries/<topic>/`), in addition to `--out`.

3. In **`auto_researcher fetch`**'s section, add:
   ```
   Before fetching, each requested id is checked against the local store
   (`store/papers/<id>/fulltext.txt`) - a paper already fetched for any
   past question is served from disk instantly, never refetched.
   ```

4. Add a new section, **"Persistent local store"**, after the "Library
   modules" section:
   ```markdown
   ## Persistent local store

   Everything `search` and `fetch` touch is cached under
   `auto-researcher/store/` (gitignored - local data, not repo content):

   - `store/papers/<safe-id>/` - one directory per paper ever seen, with
     `meta.json`, `abstract.txt` (if available), and once fetched,
     `fulltext.txt` + `fulltext.pdf` (if the source was a PDF) +
     `fetch_status.json`. A paper is fetched at most once, ever, regardless
     of how many different questions later reference it.
   - `store/queries/<topic-slug>/` - one directory per research question
     ever run, with `question.txt`, `candidates.json` (every candidate
     found, with relevance score/reason once scored), `relevant_ids.json`
     (which candidates were read in depth), and `synthesis.md` (the final
     answer).

   New CLI subcommands expose this to the skill layer for anything that
   requires an LLM step the CLI itself can't do:

   ```bash
   auto_researcher store record-scores --topic <slug> --scores <path-to-json>
   auto_researcher store record-synthesis --topic <slug> --relevant-ids id1,id2 --synthesis <path-to-md>
   auto_researcher store show --topic <slug>        # prints the full query record as JSON
   auto_researcher store show-paper --id <paper-id>  # prints one paper's cached record as JSON
   auto_researcher store list                        # prints every stored query (slug, question, created_at)
   ```

   To revisit a past question long after the original run - reread a
   specific paper, ask something the synthesis didn't cover, or fetch a
   candidate that wasn't read the first time - use the `research-followup`
   skill (`.claude/skills/research-followup/SKILL.md`) rather than rerunning
   `research-question` from scratch.
   ```

5. In the **Full workflow** section, add a closing sentence after the
   numbered manual-drive steps: "Every run's full record is saved to
   `store/queries/<topic-slug>/` — see 'Persistent local store' above to
   revisit it later without rerunning the search."

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the persistent local store and research-followup skill"
```
