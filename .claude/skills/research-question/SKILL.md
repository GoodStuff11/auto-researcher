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
