# Project hub Plan 1 — follow-ups for Plan 2
*From the subagent-driven execution ledger, 2026-11-12. Parked at the final review; none block Plan 1 use.*
## Parked findings
- untitled-row title can contain newlines — Ruling: real, deferred to Plan 2 (one-line normalise); cost if wrong: an ugly Snapshot row until fixed
- two-word owner regex reads "(due Friday)" as an owner — Ruling: real, validate flags it so nothing is silent; revisit with Plan 2 ingesters
- watermark names now restricted ([A-Za-z0-9._-]) and save_state drops unknown keys — Ruling: acceptable path-safety; Plan 2 ingesters name watermarks like repo-<name>; cost if wrong: rename keys
- merge-driver ours/theirs labels inverted during rebase and manual-rebase leaves no markers — Ruling: real, deferred to Plan 2; the hub commit path aborts and reports both values, which is what agents use
- wrong-shaped frontmatter (links: [a]) still crashes validate/snapshot; same-proposal applied on different days still duplicates; log OSError window — Ruling: deferred to Plan 2 hardening
## Deferred minors (per task)
- Task 2: minor (deferred): load_config lets FileNotFoundError/YAMLError/TypeError escape instead of ConfigError (CLI traceback on malformed project.yaml)
- Task 2: minor (deferred): .pytest_cache/ not in project-hub/.gitignore; no test for missing hub.local.yaml / RuntimeError exhaustion
- Task 3: minor (deferred): brittle frontmatter delimiters (CRLF, empty block, EOF '---'); default_meta KeyError on unknown type; unused Item import in test_items.py; blank line under frontmatter when description empty
- Task 3: minor (deferred): invalid YAML date scalar (raised: 2026-02-30) raises ValueError, not ItemFormatError
- Task 4: minor (deferred): non-atomic save/log; link allows self-links/odd rel per type; all() aborts on one malformed file while ids() counts it; extra test gaps (combined filters, create fields)
- Task 4: minor (deferred): ID_RE uses $ not fullmatch (harmless); stray non-ID *.md in Items appears in ids()/all(); create(series="str") splits chars via list() (CLI always passes lists)
- Task 5: minor (deferred): pivot-target fires for non-question pivoted; dropped actions count as spawned; whitespace no_action passes; action-owner+unknown-owner double report; Violation order unsorted
- Task 5: minor (deferred): closing-entry relies on entries[0] being creation entry (hand-edited files); filename vs frontmatter id mismatch not flagged
- Task 6: minor (deferred): sync_state setdefault doesn't replace explicit null; '|'/newline in titles breaks snapshot tables; priority None prints 'None'; last_meeting not bounded by today; snapshot happy-path-only tests
- Task 7: minor (deferred): apply_ops untested directly; ref values not validated (no $, duplicates); bare-string links misleading error; KeyError message unclear
- Task 7: minor (deferred): _check_author duplicates Entry.render check; source validated only at apply time
- Task 8: minor (deferred): Snapshot conflicts for people pulling with plain git before ever running hub (driver unset); extra regenerate commits per retry; no resolution guidance in SyncConflict message; test polish
- Task 8: minor (deferred): push retry matches "rejected" incl. hook-declined (hides stderr after 3 tries); English-only stderr matching; .git-as-file rebase check; push-failure branch untested
- Task 9: minor (deferred): scaffold silently drops >7 decisions and ranks by decision priority not question priority; one proposal per question on Sheet 2; save_draft overwrites a filled draft (guard in CLI); series not path-validated
- Task 9: minor (deferred): load_draft doesn't wrap OSError/UnicodeDecodeError; question rows' fields unvalidated
- Task 10: minor (deferred): missing Decisions sheet → bare KeyError; TITLE_RE \S+ breaks series with spaces; dead branch read.py:100-102; build KeyError on missing draft row keys; set_read_back rewrites file non-atomically; frozen tall summary row
- Task 10: minor (deferred): "approved ." (space before period) → None; title series overrides valid _meta series
- Task 11: minor (deferred): duplicate-title unplanned rows; test gaps (wrong-series, defer path, "as proposed", empty sections)
- Task 11: minor (deferred): raised row with only new_actions stages decision status "made"; by_title only matches proposed
- Task 12: minor (deferred): bare except KeyError labels any KeyError "no such item"; ValueError in catch tuple can mask bugs; show --chain partial stdout on error; --link without colon rejected only downstream
- Task 12: minor (deferred): bad --root shows raw errno text, not "no vault at <path>"
- Task 13: minor (deferred): item-format example lacks question/action examples; meeting-field doc incomplete; snapshot proposed-decisions columns undocumented; spec CLI table omits agenda scaffold; office-viewer-spike.md pending Task 1
