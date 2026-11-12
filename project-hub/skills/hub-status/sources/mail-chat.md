# Gmail and Slack (opt-in per project)
Only when `project.yaml` `sources.gmail.enabled` / `sources.slack.enabled` is true. Watermarks `gmail` / `slack` = last date ingested.
- **Gmail:** `search_threads` with the configured `terms` and `after:<watermark>`; read matching threads (`get_thread`). Convert the watermark to Gmail's format, `after:YYYY/MM/DD`. Source `gmail:<thread id>`.
- **Slack:** read the configured `channels` since the watermark (if the Slack connector needs authentication, report the source as not reachable and stop). Source `slack:<channel id>-<message ts>`.

Same extraction rules as `sources/circleback.md` (decisions and every action → proposals, never safe `new` actions; closures use the shape in `sources/circleback.md`; facts → entries/log lines; `owner` must be a people key, unmapped people go in the text), same privacy rules (no personal matters, no verbatim private messages). One batch file in the scratch directory (outside the vault); `watermark: {name: gmail|slack, value: <today>}`; `hub ingest batch <file>`.
