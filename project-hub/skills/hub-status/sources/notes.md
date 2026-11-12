# Personal notes (opt-in)
Only the folders listed under `notes:` in **your own** `hub.local.yaml` (e.g. `notes: [/home/me/obsidian/personal/research]`) — never another person's, never anything not listed. Watermark `notes.<me>` = last date ingested.
1. List files in those folders modified after the watermark.
2. Keep only facts about this project: results, decisions you took, questions you opened. Summarise in your own words; never copy text verbatim; skip anything personal, career-related or about other people's performance.
3. Batch file: `safe` entries/log lines with source `notes:<me>@<YYYY-MM-DD of the note>`; anything that closes or decides → `proposals`. `owner` of any new item must be a people key (name unmapped people in the text instead). `watermark: {name: notes.<me>, value: <today>}`. Then `hub ingest batch <file>`.
