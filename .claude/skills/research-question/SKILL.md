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
