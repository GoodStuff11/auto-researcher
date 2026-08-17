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
     to fetch and read it now. If the paper has no OA PDF link, check proxy
     access before assuming it's usable — cookies expire in hours and a
     stale `.cookies.txt` shouldn't silently produce an abstract-only result:
     ```bash
     .venv/bin/python -m auto_researcher cookies status --domains <paper's landing_url domain>
     ```
     If that comes back `false`, **ask the user** (`AskUserQuestion`) how to
     proceed rather than fetching straight away and letting it degrade
     quietly: automated refresh (`auto_researcher cookies refresh --domains ...`
     — safe to try first, it only opens a browser when one's actually
     available and otherwise prints manual instructions), manual export per
     `auto-researcher/README.md`, or skip and stay abstract-only. Once
     resolved (or if the paper already has an OA link and none of this is
     needed), write the `candidates` array from step 2 to
     `/tmp/candidates.json` and run:
     ```bash
     .venv/bin/python -m auto_researcher fetch --in /tmp/candidates.json --ids "<paper-id>" --out-dir /tmp/fulltext --cookies .cookies.txt
     ```
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
- Proxy cookies are always temporary (a few hours) regardless of how they
  were collected — never assume an existing `.cookies.txt` is still good;
  check with `cookies status` and ask the user before a fetch silently
  degrades to abstract-only for lack of a fresh one.
