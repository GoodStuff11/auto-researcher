# Auto-Researcher: Persistent Local Store + Follow-Up Skill

**Date:** 2026-08-14
**Status:** Approved design, pending implementation plan
**Extends:** `docs/superpowers/specs/2026-08-14-auto-researcher-design.md` (original design; read first)

## Purpose

The original design ran each research question as a self-contained, disposable
pipeline: search → score → read → synthesize → report, with all intermediate
data (candidates.json, fetched text) written to scratch paths and discarded
after the run. Two gaps that surfaced from actual use:

1. **Nothing is reused across runs.** The same paper found again for a
   different, later question gets refetched from scratch — wasted API calls
   and paywall-proxy fetches.
2. **The synthesis is the only surviving artifact.** Once a run finishes, you
   only have the AI's written answer. You can't go back and ask "what did
   paper X actually say," "why was paper Y excluded," or "read that one more
   carefully" without rerunning the whole search.

This design adds a persistent local store that both problems fall out of
naturally: a global cache of every paper ever touched, and a durable
per-question record of what was found, what was read, and why — plus a new
skill to converse with that record long after the original run.

## Constraints (inherited + new)

Everything in the original spec's Constraints section still applies. New:

- Single user, sequential use — no concurrent-writer locking, no database
  server. Plain files on disk, matching the existing package's
  dependency-light style (`requests` + `pypdf` + stdlib).
- The store is additive: existing `search`/`fetch` CLI behavior (JSON
  in/out via `--out`/`--out-dir`) keeps working for direct/manual use; the
  store is a side effect of those commands, not a replacement interface.
- A paper already fetched (anywhere, for any past question) must never be
  refetched — full-text retrieval is the most rate-limited, most fragile
  step in the pipeline (paywall cookies, PDF parsing) and its result should
  be treated as durable once obtained.

## Architecture

Adds one new component to the two-layer split from the original design:

3. **Local store** (`auto_researcher/store.py` + `store/` directory in
   `auto-researcher/`) — a plain-filesystem cache and per-query index that
   `search`, `fetch`, and the Workflow all read and write through. Owned by
   the mechanical Python layer (zero LLM calls), consumed by the agent layer.

## Storage layout

```
auto-researcher/store/
  papers/
    <safe-paper-id>/
      meta.json          # Paper fields: title, authors, year, venue, doi,
                          # arxiv_id, source, oa_pdf_url, landing_url
      abstract.txt        # present whenever the source provided one
      fulltext.pdf          # raw bytes, present only once fetched
      fulltext.txt            # pypdf-extracted text, present only once fetched
      fetch_status.json        # {"status": "open_access"|"proxy"|"unavailable",
                                #  "fetched_at": ISO8601, "source_url": "..."}
  queries/
    <topic-slug>/
      question.txt         # the original research question, verbatim
      candidates.json        # every deduped candidate found for this query,
                              # each with its relevance score+reason once scored
      relevant_ids.json        # candidate ids that were read in depth /
                                # included in the synthesis
      synthesis.md                # the full synthesized answer (same content
                                    # that reports/*.md holds today)
      created_at.txt / updated_at.txt
```

- `<safe-paper-id>` reuses the existing scheme from `fetch.py`
  (`paper.id.replace(":", "_").replace("/", "_")`).
- `<topic-slug>` reuses the slug scheme already used for
  `reports/YYYY-MM-DD-<topic-slug>.md` filenames. `reports/*.md` becomes a
  convenience copy generated from `queries/<slug>/synthesis.md` — the store
  is the source of truth going forward; `reports/` stays for a quick
  human-readable read without touching the store's internals.
- `store/` is created lazily on first use — no separate init step, no
  schema migration to manage.

## Components

### `auto_researcher/store.py` (new)

Plain functions, no classes needed beyond what's already idiomatic in this
package (module-level functions, following `dedup.py`/`fetch.py`'s style):

- `upsert_paper(paper: Paper) -> Path` — writes/merges `meta.json` and
  `abstract.txt` for a paper into `store/papers/<id>/`. If the paper already
  exists, merges non-null fields (first-seen wins for conflicting values,
  same tie-break already used in `dedup.py`) rather than overwriting.
- `has_fulltext(paper_id: str) -> bool` — checks whether `fulltext.txt`
  already exists for this paper, anywhere, from any past query.
- `record_fulltext(paper_id: str, pdf_bytes: bytes | None, text: str, status: str, source_url: str) -> None`
  — writes `fulltext.pdf` (if bytes given — a proxy/HTML-only fetch may have
  no PDF bytes), `fulltext.txt`, and `fetch_status.json`.
- `load_paper(paper_id: str) -> dict` — reads back everything stored for one
  paper (metadata + abstract + fulltext if present) as a plain dict.
- `record_query(topic_slug: str, question: str, candidates: list[Paper]) -> None`
  — writes `question.txt` and `candidates.json` for a query, creating the
  query directory if needed. Called by `search`.
- `record_scores(topic_slug: str, scores: list[dict]) -> None` — merges
  relevance scores/reasons into the existing `candidates.json` entries.
  Called after the Workflow's Score phase.
- `record_synthesis(topic_slug: str, relevant_ids: list[str], synthesis_md: str) -> None`
  — writes `relevant_ids.json` and `synthesis.md`. Called after the
  Workflow's Synthesize phase.
- `load_query(topic_slug: str) -> dict` — reads back everything stored for
  one query (question, candidates with scores, relevant ids, synthesis) as
  a plain dict. This is what the follow-up skill loads.
- `list_queries() -> list[dict]` — lightweight listing (slug, question,
  created_at) of every stored query, for the follow-up skill to search
  against when the user describes a question instead of naming a slug.

### `.claude/skills/research-followup/SKILL.md` (new)

Given either a topic slug or a description of a past question:

1. If given a slug, `store.load_query(slug)` directly. If given a
   description, call `store.list_queries()`, and either match by
   slug/question similarity or ask the user which past query they mean if
   more than one plausibly matches.
2. Load the query's `synthesis.md`, `candidates.json` (with scores), and
   `relevant_ids.json` into context.
3. Converse from there: answer questions against what's already loaded; for
   a candidate mentioned by id that wasn't in `relevant_ids` (found but not
   read in depth), offer to fetch and read it now via `auto_researcher fetch`
   + a single read-agent call, then persist the result back into the store
   (`record_fulltext`, and append to `relevant_ids.json` if it becomes part
   of the ongoing discussion) so a *third* future session benefits too.
4. No automated re-synthesis loop by default — this skill is for targeted
   digging, not regenerating the whole report. If the user wants the full
   synthesis redone with new information, say so explicitly and re-invoke
   `research-question`'s existing re-run flow (Step 4 in that skill) rather
   than duplicating that logic here.

### Changes to existing components

- `cli.py`'s `run_search`: after `dedup.py` produces the candidate pool, call
  `store.upsert_paper(p)` for each candidate, then `store.record_query(slug, question, candidates)`.
  Requires a new `--topic <slug>` argument (the skill already generates a
  slug for the report filename today — reuse it) and a `--question <text>`
  argument (the literal question, for `question.txt`). `--out` keeps working
  unchanged for anyone scripting against the CLI directly; the store write
  is an additional side effect, not a replacement.
- `cli.py`'s `run_fetch`: before calling `fetch_full_text`, check
  `store.has_fulltext(paper.id)` — if already present, reuse the stored
  `fulltext.txt` instead of hitting the network. After a successful fetch,
  call `store.record_fulltext(...)` instead of (or in addition to) writing
  to `--out-dir`, which remains for direct/manual CLI use.
- `.claude/workflows/research-synthesis.js`: after the Score phase, the
  skill (which owns the CLI/Workflow orchestration, not the Workflow script
  itself) persists scores via `store.record_scores`; after Synthesize, via
  `store.record_synthesis`. The Read phase's behavior is unchanged — it
  still reads from whatever's already available (abstract, or full text
  only if already cached from a past query) and does not force a new fetch
  for every selected paper; that stays an explicit, optional deepening step
  (the existing skill's Step 4), same as the original design. The benefit
  of the cache applies there: when that optional step *is* invoked, a paper
  already fetched for a past query costs nothing, since `run_fetch` now
  checks `has_fulltext` before hitting the network (see cli.py change
  above).
- `.claude/skills/research-question/SKILL.md`: Step 2 (`search`) gains
  `--topic`/`--question`; a new final note tells the user the full record now
  lives in `store/queries/<slug>/` and can be revisited later with
  `research-followup`.

## Data flow (updated)

```
question
  -> skill generates query variants, picks a topic slug
  -> CLI `search --topic <slug> --question "..."`: queries sources, dedups
     -> store.upsert_paper (x N) + store.record_query
  -> [Workflow] Score phase -> store.record_scores
  -> [Workflow] Read phase: reads abstract, or cached full text if already
     stored from a past query (no new fetch triggered here, unchanged
     from the original design)
  -> [Workflow] Synthesize phase -> store.record_synthesis
  -> (optional, unchanged Step 4) `fetch` on papers flagged critical but
     abstract-only -> cached papers return instantly, new ones are fetched
     once and cached for every future query
  -> reports/YYYY-MM-DD-<slug>.md written as a copy of the stored synthesis
  -> (any time later) `/research-followup <slug or question>` reloads
     everything from store/queries/<slug>/ and store/papers/*, converses,
     optionally fetches+persists a previously-unread candidate
```

## Error handling

All error-handling guarantees from the original design still hold
(per-source and per-paper failures never block a run; missing/expired
cookies degrade to abstract-only). New:

- Store writes are simple file writes with no locking — acceptable given
  single-user, sequential use. If a run is interrupted mid-write, the
  worst case is a partially-written query record, which the next run of
  the same topic slug simply overwrites/extends; nothing corrupts silently
  since each file is a self-contained JSON/text write, not an append.
- `has_fulltext` false negatives (e.g. corrupted `fulltext.txt`) just result
  in a refetch, not a crash — cost is a wasted API call, not a failure.

## Testing

- `store.py` gets unit tests following the existing pytest style
  (`tmp_path` fixture for a scratch store directory): upsert-merge
  behavior, has_fulltext idempotency across two upserts of the same paper,
  round-trip of `record_query`/`load_query` and `record_fulltext`/`load_paper`.
- `run_search`/`run_fetch` CLI tests get extended to assert the store
  side-effects (a paper written under `store/papers/...`, a query record
  under `store/queries/...`) in addition to their existing `--out`/`--out-dir`
  assertions.
- `research-followup` has no automated test (same as `research-question`
  today, since it's an instructions file, not code) — verified by a live
  dry-run against a query the original `research-question` dry-run created.

## Explicitly out of scope for this design

- No search/query interface *within* the store beyond `list_queries()` —
  finding a past query by fuzzy question match is the follow-up skill's job
  (an LLM judgment call), not a feature of `store.py` itself.
- No automatic pruning/expiry of the store. At personal-research scale this
  isn't worth the complexity; if it ever becomes a real disk-usage problem,
  that's a follow-up decision, not part of this design.
- No re-synthesis automation inside `research-followup` — deepening a
  specific paper is in scope; regenerating the whole report is explicitly
  routed back to `research-question`'s existing flow instead of duplicated.
- No SQLite/database index. Revisit only if directory-scan performance
  becomes a real problem at a scale this design doesn't anticipate.
