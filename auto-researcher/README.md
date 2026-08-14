# auto-researcher

Given a research question ("has X been done before?", "what progress exists
on Y?"), exhaustively searches academic literature across physics and
adjacent fields (CS, chemistry, engineering) and produces a direct,
cited answer: what's been done, by whom, how it relates to the question, and
what gaps remain.

Two layers:

- **`auto_researcher/`** — a plain Python package. Talks to search APIs,
  normalizes results, dedupes, fetches full text. Makes zero LLM calls, so
  it costs nothing beyond the (free) API keys below.
- **`.claude/workflows/research-synthesis.js`** + **`.claude/skills/research-question/SKILL.md`**
  — a Claude Code Workflow and skill that do the LLM-requiring work
  (query generation, relevance scoring, reading, synthesis) inside a Claude
  Code session, using Claude Code subagents rather than a separately
  metered API key.

Design rationale: `docs/superpowers/specs/2026-08-14-auto-researcher-design.md`
Full implementation plan: `docs/superpowers/plans/2026-08-14-auto-researcher.md`

## Setup

```bash
cd auto-researcher
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Then fill in `.env`:

| Variable | Required? | What it's for | Where to get it |
|---|---|---|---|
| `CROSSREF_MAILTO` | Recommended | Contact email sent to CrossRef and Unpaywall for their "polite pool" (higher, more reliable rate limits) — no signup needed, just an email. | Your own email address. |
| `OPENALEX_API_KEY` | Optional | Enables the OpenAlex source (250M+ works, broadest metadata coverage). OpenAlex actually works without a key via `mailto`, but this package's CLI currently gates the source on this variable being set. | Free — https://openalex.org/pricing (the "$1/day free credit" tier is effectively unlimited at this tool's usage scale). |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional | Raises Semantic Scholar's rate limit. The source runs even without a key (unauthenticated), just more likely to get rate-limited (429) on a run with many query variants — which is handled gracefully (see Error Handling below), not fatal. | Free — https://www.semanticscholar.org/product/api |
| `CORE_API_KEY` | Optional | Enables the CORE source — the largest open-access full-text aggregator (40M+ full texts). Without it, CORE is skipped entirely. | Free, instant — https://core.ac.uk/services/api |

No key is required for **arXiv** or **CrossRef** search, or for **Unpaywall**
full-text lookups (Unpaywall just requires `CROSSREF_MAILTO` to be set, reused
as its contact email).

### Paywalled full text (optional)

For papers behind IEEE Xplore / ACM DL / ScienceDirect etc., accessed via
Cornell's library proxy:

1. Log into the Cornell library proxy in a normal browser.
2. Export cookies for the relevant domains (e.g.
   `ieeexplore-ieee-org.proxy.library.cornell.edu`) to a `cookies.txt` file
   (Netscape format) using a browser extension such as "Get cookies.txt
   LOCALLY".
3. `scp cookies.txt` to `auto-researcher/.cookies.txt` (already
   `.gitignore`d).
4. Re-export whenever a run reports a paper as `unavailable` that you
   expected to be proxy-accessible — these sessions typically last hours,
   not days.

This is entirely optional. A paper with no open-access link and no fresh
proxy cookie is simply marked `unavailable` (abstract-only) — it never
blocks a run.

## Architecture

```
question
  -> generate 4-8 search query variants (adjacent fields, synonyms,
     alternate phrasings — done by the skill, not the CLI)
  -> CLI `search`: queries all configured sources per variant, dedupes,
     persists every candidate + this query's record to the local store
     -> candidates.json (title/authors/year/venue/abstract/DOI/links,
        no full text yet)
  -> [Workflow] Score phase: batched agent calls score title+abstract
     against the question -> top ~50 selected
  -> [Workflow] Read phase: one agent per selected paper, extracting
     findings relevant to the question
  -> [Workflow] Synthesize phase: merges extracts into a direct answer,
     organized by theme, with explicit gaps
  -> CLI `fetch` (optional, for papers flagged critical but abstract-only):
     fetches full text (OA link, then Unpaywall, then Cornell proxy,
     else stays abstract-only) - checks the local store first and never
     refetches a paper already retrieved for any past question ->
     re-run Score/Read/Synthesize with the added text
  -> report written to auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md
     (gitignored — local output, not part of the repo)
```

## Python package interface

### `auto_researcher search`

```bash
.venv/bin/python -m auto_researcher search \
  --query "neural quantum states fermionic sign problem" \
  --query "neural network wavefunction fermion sign problem" \
  --topic nqs-sign-problem \
  --question "Have neural quantum states been applied to the fermionic sign problem?" \
  --limit 40 \
  --out candidates.json
```

- `--query` (repeatable, required) — one search string per distinct angle
  on the question. Runs against every configured source *per query*, so
  N queries × ~5 sources each contributing up to `--limit` results before
  dedup.
- `--topic` (required) - a short slug identifying this question; also
  names its record under `store/queries/<topic>/`.
- `--question` (required) - the literal research question, stored
  verbatim alongside the record for later reference.
- `--limit` (default 25) — max results requested per source, per query.
- `--out` (required) — path to write the deduped candidate list as JSON.
- `--store-root` (default `store`) - where the local store lives.

Behavior:
- Runs arXiv and Semantic Scholar unconditionally (no key required);
  CrossRef, OpenAlex, and CORE only if their respective env vars are set.
- Any single source failing or rate-limiting (e.g. a 429) is caught,
  logged to stderr, and skipped — the run continues with whatever
  sources succeeded. A run never hard-fails because of one source.
- Results are deduped across sources: exact match on DOI/arXiv ID first,
  fuzzy title+year match as a fallback (`auto_researcher/dedup.py`).
- In addition to writing `--out`, every candidate found is persisted into
  the global paper cache (`store/papers/`) and this query's own record
  (`store/queries/<topic>/`), enabling future fetches to check the cache
  and reruns to skip re-searching the same question.

Output JSON shape (one object per deduped paper):
```json
{
  "id": "arxiv:2607.15060v2",
  "title": "...", "authors": ["..."], "year": 2024, "venue": "...",
  "abstract": "...", "doi": null, "arxiv_id": "2607.15060v2",
  "source": "arxiv", "oa_pdf_url": "https://...", "landing_url": "https://..."
}
```

### `auto_researcher fetch`

```bash
.venv/bin/python -m auto_researcher fetch \
  --in candidates.json \
  --ids arxiv:2607.15060v2,doi:10.1000/xyz \
  --out-dir fulltext/ \
  --cookies .cookies.txt
```

- `--in` (required) — the `candidates.json` from `search`.
- `--ids` (required) — comma-separated paper `id`s to fetch (usually the
  handful the synthesis flagged as important).
- `--out-dir` (required) — directory for `<safe-id>.txt` files plus a
  `manifest.json` mapping each id to its fetch status.
- `--cookies` (default `.cookies.txt`) — Netscape-format cookie file for
  the Cornell proxy fallback. Missing/expired cookies degrade that paper
  to abstract-only rather than failing the run.
- `--store-root` (default `store`) — where the local store lives.

Before fetching, each requested id is checked against the local store
(`store/papers/<id>/fulltext.txt`) - a paper already fetched for any
past question is served from disk instantly, never refetched.

Fetch priority order per paper, first success wins:
1. Direct open-access link (`oa_pdf_url` from arXiv/CORE/OpenAlex/S2).
2. Unpaywall lookup by DOI, if no direct OA link exists.
3. Cornell EZproxy + cookie session, for a paywalled `landing_url`.
4. Otherwise: `unavailable` (abstract-only) — never blocks the run.

PDF responses (the common case for OA links) are automatically detected
and their text extracted with `pypdf`; a corrupt or unparseable PDF falls
through this priority order rather than storing garbage.

`manifest.json` shape: `{"<paper-id>": "open_access" | "proxy" | "unavailable"}`

### Library modules (if scripting against the package directly)

- `auto_researcher.models.Paper` / `make_id(doi, arxiv_id, title, year)`
- `auto_researcher.search.{arxiv,openalex,semantic_scholar,crossref,core}.search_*(query, ..., limit=25) -> List[Paper]`
- `auto_researcher.search.unpaywall.find_oa_location(doi, email) -> str | None`
- `auto_researcher.dedup.dedupe(papers, title_similarity_threshold=0.92) -> List[Paper]`
- `auto_researcher.cookies.CookieStore(cookies_path)` — `.is_fresh(domain)`, `.as_requests_cookies()`
- `auto_researcher.fetch.fetch_full_text(paper, cookie_store=None, email=None) -> FullTextResult`

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

## Full workflow (via the Claude Code skill)

The intended way to use this tool is inside a Claude Code session, via the
skill at `.claude/skills/research-question/SKILL.md`:

```
/research-question "Has anyone applied neural quantum states to the fermionic sign problem?"
```

This drives the whole pipeline above end-to-end: generates query variants,
runs `search`, calls the `research-synthesis` Workflow
(`.claude/workflows/research-synthesis.js`) to score/read/synthesize the
candidate pool, optionally runs `fetch` to deepen on the most important
abstract-only papers and re-synthesizes, then writes the report to
`auto-researcher/reports/` and gives you the direct answer in chat.

You can also drive the two layers manually:

1. Run `auto_researcher search` yourself (see above) to get `candidates.json`.
2. Ask Claude Code to run the `Workflow` tool with
   `scriptPath: .claude/workflows/research-synthesis.js` and
   `args: {"question": "...", "candidates": <the JSON array from step 1>}`.
   Returns `{synthesis, extracts, totalCandidates, totalRanked}`.
3. Optionally run `auto_researcher fetch` on the papers the synthesis
   flagged as important but abstract-only, attach the fetched text as
   `full_text_excerpt` on those candidate objects, and re-run step 2.

Every run's full record is saved to `store/queries/<topic-slug>/` — see
'Persistent local store' above to revisit it later without rerunning the
search.

## Error handling / guarantees

- Any single search source down, rate-limited, or erroring is skipped
  with a warning; the run continues with whatever sources succeeded.
- Any single paper's full-text fetch failing (network error, corrupt
  PDF, expired cookies) is caught per-paper; that paper is marked
  `unavailable` in the manifest and the batch continues.
- Missing or expired proxy cookies degrade a paywalled paper to
  abstract-only — never blocks a run.
- Dedup is best-effort (DOI match is reliable; fuzzy title/author match
  is not perfect) — an acceptable tradeoff over false-merging distinct
  papers.
- If nothing in the candidate pool scores as genuinely relevant, the
  synthesis says so explicitly rather than forcing an answer from weak
  matches.

## Testing

```bash
cd auto-researcher
.venv/bin/pytest -v
```

36 unit tests across the search adapters, dedup, cookies, fetch, and CLI —
all against mocked HTTP responses. The Workflow/skill layer has no
automated tests (subagent orchestration isn't unit-testable); it's
validated by live dry-runs against real questions.

## Explicitly out of scope

- Google Scholar (no official API, ToS risk) and PubMed (not relevant to
  this tool's target fields) are not sources.
- No headless/automated login to Cornell SSO — blocked by Duo 2FA; manual
  cookie export/import is the deliberate workaround for the SSH-only,
  no-browser environment this tool runs in.
- No separate metered LLM API billing — all reasoning happens via Claude
  Code subagents/Workflow within an existing session.
