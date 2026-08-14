# Auto-Researcher: Exhaustive Literature Question-Answering Tool

**Date:** 2026-08-14
**Status:** Approved design, pending implementation plan

## Purpose

Given a research question ("has X been done before?", "what progress exists on Y?"),
exhaustively search academic literature across physics and adjacent fields (CS,
chemistry, engineering) — including work never posted to arXiv — and produce a
direct, well-cited answer: what's been done, by whom, how it relates to the
question, and what gaps remain.

Lives in `auto-researcher/` in this repo.

## Constraints that shaped this design

- Must not be limited to arXiv/Semantic Scholar alone — many engineering papers
  are never posted to preprint servers.
- Full text matters for papers behind subscriptions (IEEE Xplore, ACM DL,
  ScienceDirect), accessed via Cornell's institutional library subscription.
- The tool runs from an SSH session on a cluster — no local browser available,
  so paywalled-paper authentication cannot use an interactive login popup.
- No separate LLM API billing. All reading/reasoning must happen inside a
  Claude Code session (subagents), not via a metered Anthropic/Gemini/OpenAI
  API key. Only the mechanical parts (HTTP calls to search APIs) run as plain
  code outside the agent.
- Push as much of the work as reasonable into non-agent (plain Python) code;
  reserve agent calls for judgment calls a script can't make (relevance
  scoring, reading comprehension, synthesis).

## Architecture

Two layers with a clean split of responsibility:

1. **Python package** (`auto_researcher/`) — purely mechanical. Talks to
   search APIs, normalizes results, dedupes, fetches full text. Makes zero
   LLM calls. No separate billing.
2. **Claude Code skill** — orchestrates the whole run inside a session
   (existing subscription, no metered API key). Generates search queries,
   invokes the Python CLI for the mechanical retrieval, then runs a
   `Workflow` to score relevance, read papers, and synthesize the answer.

## Components (Python package)

- `search/` — one adapter per source, each normalizing results into a common
  `Paper` record (title, authors, year, venue, abstract, DOI, arXiv ID, OA PDF
  URL if any):
  - `openalex.py` — broadest coverage (250M+ works, all fields). Requires a
    free API key ($1/day free credit, effectively unlimited at this scale).
  - `semantic_scholar.py` — strong CS/physics coverage, citation graph, TL;DRs.
    Free API key optional (raises rate limits).
  - `crossref.py` — DOI metadata for nearly everything published, including
    IEEE/ACM/Elsevier journal articles we can't get full text for. No key
    needed.
  - `arxiv.py` — physics/CS/math preprints. No key needed.
  - `core.py` — largest open-access full-text aggregator (40M+ full texts).
    Free API key required.
  - `unpaywall.py` — given a DOI, finds a legal free full-text link anywhere
    it exists. No key needed (just a contact email per their terms).
- `dedup.py` — merges duplicate records across sources: DOI match first,
  fuzzy title+author match as fallback.
- `fetch.py` — full-text retrieval, in priority order:
  1. Open-access link from arXiv/CORE/Unpaywall.
  2. Cornell EZproxy URL + cookie-authenticated session, for papers behind
     IEEE Xplore/ACM DL/ScienceDirect/etc.
  3. Neither available → paper is marked `full_text: none, abstract-only`,
     never blocks the run.
- `cookies.py` — loads a `cookies.txt` (Netscape format, exported from your
  local browser's Cornell library session and copied to the cluster via
  scp), checks per-domain freshness against each cookie's expiry, and warns
  (does not error) when a domain's session looks stale.
- `cli.py` — thin entrypoint the skill shells out to (e.g.
  `python -m auto_researcher search --queries q1.json --out candidates.json`,
  `python -m auto_researcher fetch --ids id1,id2 --out fulltext/`). All
  input/output is JSON/files — no interactive prompts, since it's driven by
  the skill layer.

## Cookie refresh workflow

1. On your laptop, log into the Cornell library proxy normally in your
   regular browser.
2. Export cookies for the relevant domains (ieeexplore-ieee-org.proxy.
   library.cornell.edu, dl-acm-org.proxy.library.cornell.edu, etc.) to
   `cookies.txt` using a browser extension (e.g. "Get cookies.txt LOCALLY").
3. `scp cookies.txt` to `auto-researcher/.cookies.txt` on the cluster
   (gitignored).
4. Re-run whenever `fetch.py` reports a domain's session as expired —
   Shibboleth/SSO sessions typically last hours, not days.

## Data flow (a single research run)

```
question
  -> skill generates several query variants (adjacent fields, synonyms,
     alternate phrasings — not just the literal question)
  -> CLI queries all sources per variant -> raw results (JSON)
  -> dedup.py merges into a candidate pool (~100-300 papers)
  -> [Workflow] relevance-scoring stage: batched agent calls score
     title+abstract against the question -> top ~20-50 selected
  -> CLI fetches full text for the selected papers (OA first, cookie
     fallback, else abstract-only)
  -> [Workflow] pipeline: one read/extract agent per paper, pulling out
     findings relevant to the question, method, and how it relates
  -> [Workflow] synthesis stage: merges all extracts into:
       - direct answer (done / partially done / not done + confidence)
       - organized by approach/theme
       - explicit gaps: what hasn't been done
       - full bibliography with links (OA PDF / DOI / abstract-only flag)
  -> report written to auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md
  -> short answer also echoed directly in chat
```

The read/score/synthesize stage runs as a `Workflow` script (pipeline-style:
each paper's read stage isn't blocked on every other paper finishing), since
this is exactly the deterministic multi-stage fan-out/fan-in shape Workflow
is for. The mechanical retrieval stages (query fan-out, dedup, fetch) are
plain Python — no agent involvement — to keep agent usage focused on
judgment calls rather than HTTP plumbing.

## Error handling

- Any single search source being down, rate-limited, or erroring is
  skipped with a warning; the run continues with whatever sources
  succeeded. A run never hard-fails because of one source.
- Missing or expired cookies degrade a paywalled paper to abstract-only
  rather than blocking the run.
- Duplicate detection is best-effort (DOI match is reliable; fuzzy
  title/author match is not perfect) — some near-duplicates may slip
  through; this is an acceptable tradeoff over false-merging distinct
  papers.
- If the relevance-scoring stage finds nothing genuinely relevant, the
  final report says so explicitly rather than forcing an answer out of
  weak matches.
- Per-source rate limits (documented by each API) are respected with
  basic backoff/retry in the adapter layer.

## Testing

- Unit tests per search adapter against mocked HTTP responses, verifying
  correct parsing into normalized `Paper` records.
- Unit tests for `dedup.py` against known duplicate/non-duplicate pairs.
- One optional, slow/live integration test exercising the full mechanical
  pipeline (query -> candidates -> fetch) against real APIs, skipped in
  normal test runs.
- Synthesis quality is not unit-testable; validated by spot-checking a
  handful of questions with known ground-truth answers ("yes, group X did
  this in year Y") once implemented.

## Setup requirements

- Free API keys: OpenAlex, CORE, (optionally) Semantic Scholar — stored in
  a local, gitignored `.env` in `auto-researcher/`.
- Periodic manual cookie export for Cornell-proxied paywalled access (see
  above).

## Explicitly out of scope for this design

- Fully automated headless login to Cornell SSO (blocked by Duo 2FA;
  cookie export/import is the agreed workaround).
- Google Scholar scraping (no official API, fragile, ToS risk) — not
  included as a source.
- PubMed (user doesn't care about biomedical literature for this tool).
- Any separate metered LLM API billing — all reasoning happens via Claude
  Code subagents/Workflow within an existing session.
