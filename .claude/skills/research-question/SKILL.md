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

3. **If the candidate pool is too large to pass in one call, pre-filter
   before scoring.** A broad question can pull back hundreds of candidates
   (900+ is not unusual) — more than fits in a single `Workflow` tool call
   (the whole `candidates` array has to be typed into that one call, and
   `Read`-ing `/tmp/candidates.json` back into context to do that fails
   outright above 256KB, well below what hundreds of full abstracts costs).
   Check the file size first:
   ```bash
   wc -c /tmp/candidates.json
   ```
   If it's above roughly 150-200KB (a rough proxy for "won't fit"), pre-filter
   down to a manageable pool — a few hundred candidates at most, and lower
   still if abstracts are long — with a cheap keyword-relevance heuristic
   *before* handing anything to the LLM scorer:
   - Score each candidate by counting occurrences of question-derived
     keywords/synonyms (the same terms used to build the search queries in
     step 1, plus domain jargon) in its title+abstract.
   - Sort descending, truncate long abstracts (~400-700 chars) to control
     token size, and keep enough top-scoring candidates to comfortably fit
     (verify with `Read` — it should return without a truncation warning).
   - **This pre-filter is a coverage tradeoff, not free** — it's a blunter
     signal than the LLM relevance scoring in step 4, so it can drop
     genuinely relevant papers that just don't share surface keywords with
     the question. Disclose in step 8's report exactly how many raw
     candidates existed and how many survived this pass, and mention that
     `research-followup` (or a re-run with a different keyword set) can pull
     in more of the untouched pool later.
   - If the pool already fits without pre-filtering, skip this step entirely.

4. **Run the synthesis workflow.** Read the (possibly pre-filtered)
   candidates file, then call the `Workflow` tool with:
   - `scriptPath`: `.claude/workflows/research-synthesis.js`
   - `args`: `{ "question": "<the user's question>", "candidates": <the JSON array>, "maxRead": 50 }`
     (`maxRead` is optional, defaults to 50 — how many of the papers that
     score >=5/10 relevance actually get read in depth; raise it, e.g. to
     100-150, up front for a question you already expect to be broad, at
     the cost of more agent calls/time.)

   This scores relevance, reads up to `maxRead`, and returns
   `{ synthesis, extracts, scores, citations, totalCandidates, totalRanked, totalQualified, unreadQualifiedIds }`.
   `synthesis` already contains inline citations — every claim that draws on a
   specific paper is followed by a markdown link like `[3](https://arxiv.org/...)`,
   clickable straight to that paper. `citations` is the numbered list those
   markers refer to (`{ n, id, title, authors, year, venue, link }`), in the
   same order as the `[n]` markers — this is what you'll render as the
   References section in step 7, so the numbers line up exactly.

   **If `totalQualified` is noticeably larger than `totalRanked`**, more
   papers scored relevant than were actually read — the workflow logs this
   but does not stop for you. Don't just silently accept the top `maxRead`
   and move on:
   - Always disclose the gap in the report (step 8) — how many scored
     relevant, how many were read, and that `unreadQualifiedIds` names the
     rest (they're still in the store, scored, just not read in depth).
   - If the gap is large (say, `totalQualified` is more than ~1.5x
     `totalRanked`, or the absolute gap is more than ~25 papers), ask the
     user (`AskUserQuestion`) whether to do a deeper pass — re-run this step
     with a higher `maxRead` — before finalizing, rather than assuming the
     default depth was enough for how broad the topic turned out to be.
     Re-running reuses the same `candidates`/scores; only the extra papers
     beyond the previous `maxRead` need to be read and re-synthesized.

5. **Persist the scores.** Write the `scores` array from step 4 to a temp
   file and run:
   ```bash
   .venv/bin/python -m auto_researcher store record-scores \
     --topic "<topic-slug>" --scores /tmp/scores.json
   ```

6. **Fetch full text for the highest-relevance papers (optional deepening).**
   If the synthesis flags specific papers as critical but abstract-only:

   a. Derive the publisher domain (from each candidate's `landing_url`) for
      every flagged paper that has no OA PDF, and check whether proxy access
      to those domains is currently usable:
      ```bash
      .venv/bin/python -m auto_researcher cookies status --domains d1.com,d2.com
      ```
      This never makes a network call — it just checks whether a stored
      cookie for that domain hasn't expired yet. Cookies are always
      temporary (a few hours), so re-check this every time, even if a prior
      run in this same session already fetched successfully.

   b. If any domain comes back `false`, **ask the user before proceeding**
      (with `AskUserQuestion`) rather than silently degrading those papers
      to abstract-only — don't assume a missing/stale `.cookies.txt` means
      the user doesn't want full text. Offer:
      - **Automated browser login (default)** — run
        `.venv/bin/python -m auto_researcher cookies refresh --domains d1.com,d2.com`.
        This only actually opens a browser when one is available (checked
        internally); over SSH/headless it just prints the same manual
        instructions as the next option, so it's always safe to try first.
      - **Manual export** — walk the user through the steps in
        `auto-researcher/README.md`'s "Paywalled full text" section (log
        into the Cornell proxy, export cookies with a browser extension,
        `scp` the file over) and wait for them to confirm it's done.
      - **Skip and continue abstract-only** — proceed to step 7 without
        fetching; note in the final report which papers this affected.

      If the user picks automated or manual and completes it, re-run
      `cookies status` to confirm before calling `fetch`.

   c. Once proxy access is confirmed (or the papers have OA PDF links that
      don't need it at all), run:
      ```bash
      .venv/bin/python -m auto_researcher fetch --in /tmp/candidates.json --ids id1,id2 --out-dir /tmp/fulltext --cookies .cookies.txt
      ```
      then attach the fetched text as `full_text_excerpt` on those candidates
      and re-run the synthesis workflow. A paper fetched here (or in any past
      run) is served from the store instantly on a future fetch — it is
      never downloaded twice.

7. **Build the References section and persist the synthesis.** From the
   workflow's `citations` array (already in `n` order), build one line per
   entry:
   `[3] Bukov, Schmitt, Dupont (2020) — "Learning the ground state..." SciPost Physics. https://arxiv.org/pdf/2011.11214`
   (repeat the link as plain text too, not just as a markdown target, so it's
   visible/copyable even where the renderer strips links). This is the
   backing citation for every inline `[n](link)` marker already present in
   `synthesis` — the single place a reader lands to see full source detail
   for any claim, right after the summary as requested, no separate
   full-bibliography hunt required.

   Append this References section to the `synthesis` body, write the
   combined markdown to a temp file, and run:
   ```bash
   .venv/bin/python -m auto_researcher store record-synthesis \
     --topic "<topic-slug>" --relevant-ids id1,id2,id3 --synthesis /tmp/synthesis.md
   ```
   (`--relevant-ids` is the comma-separated list of candidate ids from the
   workflow's `extracts` — the papers actually read and included.) Persisting
   References alongside the body means a later `research-followup` session
   still has working inline citation links without re-deriving them.

8. **Write the report.** Save to
   `auto-researcher/reports/YYYY-MM-DD-<topic-slug>.md`:
   - the direct answer up top,
   - a caveats line if step 3's pre-filter ran (raw candidate count vs. how
     many were scored) and/or if step 4 flagged a `totalQualified` vs.
     `totalRanked` gap (how many scored relevant vs. how many were read,
     pointing at `unreadQualifiedIds` for what's still unread but scored),
   - the synthesis body + References section from step 7, verbatim,
   - a full **Bibliography** section below that, covering every candidate
     considered (not just cited ones): title, authors, year, venue, link — OA
     PDF link if available, otherwise DOI or landing page, flagged
     `abstract-only` if no full text was fetched. This is the complete record;
     References is the subset actually cited inline.
   Also give the user the direct answer in chat, and mention that the full
   record is durably stored under `store/queries/<topic-slug>/` for later
   follow-up.

## Notes

- Proxy cookies are always temporary (a few hours), whether collected
  manually or via `cookies refresh` — never treat an existing
  `.cookies.txt` as a permanent, one-time setup. Step 6b is the checkpoint:
  ask before degrading a flagged paper to abstract-only, don't assume it.
- Two separate truncation points can silently under-cover a broad question:
  the pre-filter in step 3 (raw candidates too numerous to even score) and
  the `maxRead` cap in step 4 (more papers score relevant than get read).
  Both are cheap to disclose and only sometimes worth re-running for —
  don't skip the disclosure even when you decide the current depth is fine.
- `cookies refresh` picks automated-vs-manual for you based on whether a
  local browser is actually available (SSH sessions and headless machines
  always fall back to manual instructions) — see `auto-researcher/README.md`
  for how that detection works and one-time `playwright install` setup.
- If the user chooses to skip fetching, `fetch.py` still marks those papers
  `unavailable` (abstract-only) automatically and the run continues — that
  degradation itself isn't an error, only skipping the *prompt* before it
  would be. Tell the user which papers this affected if any looked
  important.
- To dig into a specific paper from this question later, or ask something
  the original synthesis didn't cover, use the `research-followup` skill
  with this question's topic slug.
