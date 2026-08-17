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
- **`.claude/workflows/research-synthesis.js`** + skills in `.claude/skills/`
  — Claude Code Workflow and skills that do the LLM-requiring work
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

### Paywalled full text: setting up `.cookies.txt` (optional)

Papers behind a publisher paywall (IEEE Xplore, ACM DL, ScienceDirect, etc.)
with no open-access copy can still be fetched through Cornell's library
proxy, if you give the tool a valid, logged-in session cookie for it. This
is entirely optional — a paper with no OA link and no fresh proxy cookie is
just marked `unavailable` (abstract-only); it never blocks a run.

**What you need:** a `cookies.txt` file in the Netscape cookie format
(one cookie per line, tab-separated: `domain, include_subdomains, path,
secure, expiry, name, value`) placed at `auto-researcher/.cookies.txt`.
This is a plain file, not a browser session — the tool reads it once per
run and sends the cookies as headers, it never drives a real browser.

Step by step:

1. **Log into the Cornell library proxy in a normal desktop browser** —
   go to a Cornell library-proxied link (e.g. through
   `https://library.cornell.edu`, search for a paywalled article and click
   through to it) and complete the Cornell NetID + Duo 2FA login when
   prompted. This is unavoidably interactive; there's no headless/scripted
   way to pass Duo (see "Explicitly out of scope" below).

2. **Visit at least one page on each publisher domain you care about,
   through the proxy**, so the session cookie actually gets set for that
   domain. The proxied domain looks like the original with
   `.proxy.library.cornell.edu` appended, e.g.:
   - `ieeexplore-ieee-org.proxy.library.cornell.edu`
   - `www-sciencedirect-com.proxy.library.cornell.edu`
   - `dl-acm-org.proxy.library.cornell.edu`

   If you only ever need one publisher, one visit is enough; the cookie
   file only needs to cover domains you'll actually fetch from.

3. **Export cookies to a file** using a browser extension that writes the
   Netscape format — e.g. "Get cookies.txt LOCALLY" (Chrome/Firefox). Open
   the extension while on the proxied page and export/download
   `cookies.txt`. Don't hand-edit this file; the format is picky about tab
   separation.

4. **Copy it to the machine running `auto-researcher`**, naming it exactly
   `.cookies.txt` in the `auto-researcher/` directory:
   ```bash
   scp cookies.txt <user>@<host>:/path/to/research/auto-researcher/.cookies.txt
   ```
   It's already listed in `.gitignore` — never commit it (it's a live
   authenticated session for your Cornell account). Consider
   `chmod 600 auto-researcher/.cookies.txt` since it grants proxy access
   to anyone who can read it.

5. **Use it.** Both the `fetch` CLI command and the skills pass
   `--cookies .cookies.txt` by default (relative to `auto-researcher/`) —
   nothing else to configure.

6. **Re-export when it goes stale.** Cornell proxy sessions typically last
   hours, not days — cookies collected either manually or via the automated
   flow below are equally temporary, so this is never a one-time setup step.
   Check freshness for the domains you need before relying on it:
   ```bash
   .venv/bin/python -m auto_researcher cookies status --domains ieeexplore.ieee.org,dl.acm.org
   ```
   prints `true`/`false` per domain without making any network request (it
   only checks the cookie's stored expiry). `false` means repeat steps 1–4
   (or run the automated refresh below) before fetching from that domain.

#### Automated refresh (when a local browser is available)

Manually exporting cookies through a browser extension every few hours is
tedious. `auto_researcher cookies refresh` automates everything *except* the
NetID + Duo tap itself — it opens a real, visible browser tab per publisher
domain, waits for you to log in, and saves the resulting session cookies
straight to `.cookies.txt`:

```bash
.venv/bin/pip install playwright && .venv/bin/playwright install chromium  # one-time
.venv/bin/python -m auto_researcher cookies refresh --domains ieeexplore.ieee.org,dl.acm.org
```

It only runs this way when it can plausibly show you a browser window —
`is_local_browser_available()` (`auto_researcher/browser_login.py`) checks
for an SSH session (`SSH_CONNECTION`/`SSH_TTY`) or a headless Linux display
(no `DISPLAY`/`WAYLAND_DISPLAY`) and, if either applies, prints the manual
instructions above instead of trying to launch a browser that can't be seen.
This means it's genuinely a no-op over an SSH-only research server — the
manual export/`scp` flow above is still how you get cookies onto a headless
machine; the automated path is for when you're running `auto_researcher`
directly on your own desktop.

Since cookies always expire regardless of how they were collected, the
`research-question` and `research-followup` skills check
`cookies status` before any `fetch` and ask you how to refresh (automated,
manual, or skip and stay abstract-only) rather than assuming an old
`.cookies.txt` is still good.

#### Covering multiple publishers in one `.cookies.txt`

A single `.cookies.txt` can hold cookies for as many proxied publisher
domains as you like at once — `CookieStore` (`auto_researcher/cookies.py`)
loads the whole file into one cookie jar and, per fetch, picks out whatever
cookies match the domain being requested. There's no per-publisher
configuration; you only need to get all the relevant cookies into that one
file. Two ways to do that:

- **Easiest: export your whole browser's cookie jar at once**, not just the
  current tab. Most cookie-export extensions have an "export all cookies"
  mode as well as a "current site only" one (in "Get cookies.txt LOCALLY",
  this is a toggle in the extension popup, not the default). Do this after
  visiting the proxy login page for every publisher you need (step 2 above)
  — one export at the end captures all of them in a single valid file, and
  you can skip the merge step below entirely.

- **If your extension only exports the current tab's domain**, visit and
  export each publisher separately (step 2, then step 3, once per
  publisher), then merge the resulting files by concatenating their cookie
  lines into one `.cookies.txt`:
  ```bash
  # keep the header + cookie lines from the first file, then append only
  # the cookie lines (skip the "# Netscape..." / "# This is a generated
  # file" comment lines) from every subsequent export
  cat ieee-cookies.txt > .cookies.txt
  grep -v '^#' sciencedirect-cookies.txt >> .cookies.txt
  grep -v '^#' acm-cookies.txt >> .cookies.txt
  ```
  Order doesn't matter and duplicate domains aren't a problem — each cookie
  line is independent and `CookieStore` matches by domain per-request.

Either way, once merged, re-run the test below — it should report `True`
for every domain you added.

#### Testing whether your `.cookies.txt` works

`cookies status` (above) confirms the file parses and a cookie for the
domain hasn't *expired*, but that's a local check — it can't tell you the
session was actually revoked server-side. For that, a short inline check
makes a real request through the proxy:

```bash
cd auto-researcher
.venv/bin/python3 -c "
from pathlib import Path
from auto_researcher.cookies import CookieStore
from auto_researcher.fetch import to_proxy_url
from auto_researcher.http_utils import request_with_retry

cs = CookieStore(Path('.cookies.txt'))

# swap in a real paywalled article URL from the publisher you're testing
test_url = 'https://www.nature.com/articles/<some-article-id>'
proxy_url = to_proxy_url(test_url)

resp = request_with_retry('GET', proxy_url, cookies=cs.as_requests_cookies())
print('status:', resp.status_code, '| stayed on proxy domain:', resp.url == proxy_url)
"
```

`status: 200` with `stayed on proxy domain: True` means the session is
live — a request through the proxy without a valid cookie gets bounced off
the proxy hostname back to the original publisher domain instead (test this
by rerunning `request_with_retry` without the `cookies=` argument for
comparison), so "stayed on proxy domain" is the reliable signal, not just
the status code — a paywalled article can still return 200 to a logged-out
request for its (paywalled) landing page.

If the script raises `LoadError: '.cookies.txt' does not look like a
Netscape format cookies file`, the file is empty, corrupt, or was
hand-edited incorrectly — re-export it from scratch (step 3 above).

**If you get `status: 403` with a body containing `Just a moment...`**, that's
a Cloudflare bot challenge, not a cookie problem — some publisher platforms
(e.g. `journals.aps.org`) sit behind Cloudflare and block plain HTTP clients
by their request fingerprint before your cookies are ever checked. This
tool's HTTP layer (`request_with_retry` in `auto_researcher/http_utils.py`)
already sends a browser-like `User-Agent` for exactly this reason, so a
correctly-loaded, still-fresh cookie should get through; if you still see
the challenge page, the site's Cloudflare tier is likely doing deeper
fingerprinting (TLS/JS) that a plain `requests` call can't satisfy — that
publisher will stay `unavailable` via the proxy path regardless of cookie
freshness, and the paper falls back to OA/abstract-only.

## Usage

The intended way to use this tool is through Claude Code skills — you
generally never need to type Python commands yourself. Three entry points,
depending on what you're doing:

### Starting a new research question

```
/research-question "Has anyone applied neural quantum states to the fermionic sign problem?"
```

Runs `.claude/skills/research-question/SKILL.md`. Drives the whole pipeline
end-to-end: generates query variants, runs `search`, calls the
`research-synthesis` Workflow to score/read/synthesize the candidate pool,
optionally runs `fetch` to deepen on the most important abstract-only papers
and re-synthesizes, then writes a report to `auto-researcher/reports/` and
gives you the direct answer in chat. Everything is persisted to the store
along the way — see "What gets generated" below.

### Following up on a question you already asked

```
/research-followup "what did we find about the J1-J2 model specifically?"
```
or naming a topic slug directly if you remember it. Runs
`.claude/skills/research-followup/SKILL.md`. Loads the stored record for a
past question — synthesis, full candidate list with scores, which papers
were read in depth — and answers against it *without* re-searching. Can, on
request, fetch and read one specific paper that was found but never read
the first time, or fold it into the record permanently. This is the right
tool for "dig deeper into paper X" or "did the original search cover Z?" —
it's cheap because it makes no new search-API calls unless you explicitly
ask it to fetch something.

### Driving it manually (no skill)

Useful for scripting, debugging, or a one-off `search`/`fetch` outside the
skill flow. Both CLI subcommands and the Workflow can be invoked directly;
see "Python commands run at each stage" below for exact syntax. In short:

1. Run `auto_researcher search` yourself to get `candidates.json`.
2. Ask Claude Code to run the `Workflow` tool with
   `scriptPath: .claude/workflows/research-synthesis.js` and
   `args: {"question": "...", "candidates": <the JSON array from step 1>}`.
3. Persist the result with `auto_researcher store record-scores` and
   `auto_researcher store record-synthesis`.
4. Optionally `auto_researcher fetch` the papers flagged important-but-
   abstract-only, attach the text, and re-run step 2.

This is exactly what the `research-question` skill does on your behalf —
drive it by hand only when you need to change something the skill doesn't
expose (e.g. a custom scoring prompt, or debugging one stage in isolation).

## What gets generated: the persistent local store

Everything `search` and `fetch` touch is cached under
`auto-researcher/store/` (gitignored — local data, not repo content):

- `store/papers/<safe-id>/` — one directory per paper ever seen, with
  `meta.json`, `abstract.txt` (if available), and once fetched,
  `fulltext.txt` + `fulltext.pdf` (if the source was a PDF) +
  `fetch_status.json`. A paper is fetched at most once, ever, regardless
  of how many different questions later reference it.
- `store/queries/<topic-slug>/` — one directory per research question ever
  run, with `question.txt`, `candidates.json` (every candidate found, with
  relevance score/reason once scored), `relevant_ids.json` (which
  candidates were read in depth), `synthesis.md` (the final answer), and
  `created_at.txt`/`updated_at.txt` (bookkeeping timestamps, also surfaced
  by `store list`).

Reports are also written per run to
`auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md` (also gitignored —
local output, not part of the repo) — the human-readable version of a
query's store record, with the answer up top and a full bibliography.

CLI subcommands exposing the store to the skill layer:

```bash
auto_researcher store record-scores --topic <slug> --scores <path-to-json>
auto_researcher store record-synthesis --topic <slug> --relevant-ids id1,id2 --synthesis <path-to-md>
auto_researcher store show --topic <slug>        # prints the full query record as JSON
auto_researcher store show-paper --id <paper-id>  # prints one paper's cached record as JSON
auto_researcher store list                        # prints every stored query (slug, question, created_at)
```

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
```

Everything left of the Workflow arrow is plain, testable Python
(`auto_researcher/`) with no LLM involvement and no metered API cost beyond
the free keys above. Everything at/after the Workflow arrow runs as Claude
Code subagents inside your session — no separate LLM billing.

## Python commands run at each stage

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
- `--topic` (required) — a short slug identifying this question; also
  names its record under `store/queries/<topic>/`.
- `--question` (required) — the literal research question, stored
  verbatim alongside the record for later reference.
- `--limit` (default 25) — max results requested per source, per query.
- `--out` (required) — path to write the deduped candidate list as JSON.
- `--store-root` (default `store`) — where the local store lives.

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

### The `research-synthesis` Workflow

Not a Python command — invoked as a Claude Code `Workflow` tool call with
`scriptPath: .claude/workflows/research-synthesis.js` and
`args: {"question": "...", "candidates": [...]}`. Scores every candidate's
title+abstract against the question, reads the top ~50 in parallel
subagents, and synthesizes their extracts into one answer. Returns
`{synthesis, extracts, scores, totalCandidates, totalRanked}` — write
`scores` and `synthesis` to files and hand them to the `store` subcommands
below.

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
  the Cornell proxy fallback (see Setup above for how to produce this).
  Missing/expired cookies degrade that paper to abstract-only rather than
  failing the run.
- `--store-root` (default `store`) — where the local store lives.

Before fetching, each requested id is checked against the local store
(`store/papers/<id>/fulltext.txt`) — a paper already fetched for any
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

### `auto_researcher store ...`

See "What gets generated" above for the four subcommands
(`record-scores`, `record-synthesis`, `show`, `show-paper`, `list`) and the
on-disk shape they read/write.

### Library modules (if scripting against the package directly)

- `auto_researcher.models.Paper` / `make_id(doi, arxiv_id, title, year)`
- `auto_researcher.search.{arxiv,openalex,semantic_scholar,crossref,core}.search_*(query, ..., limit=25) -> List[Paper]`
- `auto_researcher.search.unpaywall.find_oa_location(doi, email) -> str | None`
- `auto_researcher.dedup.dedupe(papers, title_similarity_threshold=0.92) -> List[Paper]`
- `auto_researcher.cookies.CookieStore(cookies_path)` — `.is_fresh(domain)`, `.as_requests_cookies()`
- `auto_researcher.fetch.fetch_full_text(paper, cookie_store=None, email=None) -> FullTextResult`

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

61 unit tests across the search adapters, dedup, cookies, fetch, the local
store, and the CLI — all against mocked HTTP responses and `tmp_path`-isolated
filesystem state. The Workflow/skill layer has no automated tests (subagent
orchestration isn't unit-testable); it's validated by live dry-runs against
real questions, including the full search → score → read → synthesize →
persist → follow-up sequence.

## Explicitly out of scope

- Google Scholar (no official API, ToS risk) and PubMed (not relevant to
  this tool's target fields) are not sources.
- No way to script past Duo 2FA itself — that's the point of 2FA. When a
  real display is available (`auto_researcher cookies refresh`), a visible
  browser automates the login *flow* but still requires you to tap Duo
  yourself; over SSH or on a headless machine, manual cookie export/import
  (see above) is the deliberate fallback, since there's no way to show you
  a browser window to log into in the first place.
- No separate metered LLM API billing — all reasoning happens via Claude
  Code subagents/Workflow within an existing session.
