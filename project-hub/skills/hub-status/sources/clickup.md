# ClickUp
ClickUp owns task status, assignee and due date. The hub mirrors them onto actions that carry `clickup_id`, stages tasks created directly in ClickUp, and gets a ClickUp task created for each new owned action.
## Configuration
`project.yaml`: `sources.clickup.list_id` (and optional `done_statuses`, default complete/completed/closed/done); `people.<key>.clickup` = that person's ClickUp user id.
## Export
1. `clickup_get_user` → note `current_time` (used for the watermark and for any relative date).
2. `clickup_filter_tasks` with `list_ids: [<list_id>]`, `include_closed: true`, `fields: ["status", "assignees", "due_date"]`, `page: 0`; repeat with `page: next_page` while `has_more` is true.
3. Write every task to a scratch file `tasks.json` as `{"id", "name", "status", "assignees", "due_date"}`, passing the connector's values through unchanged: `status` as returned (a name string, or `{status, type}`), `assignees` as returned (ids, usernames or `{id, username, email}`), `due_date` as epoch milliseconds (or null). `hub clickup plan` and `apply` also accept `due` (`YYYY-MM-DD`) instead.
## Reconcile
1. `hub clickup plan --tasks tasks.json` — read `set`, `proposals`, `flags`, `push` and sanity-check (a mass of unexpected changes means a wrong list id: stop and ask).
2. `hub clickup apply --tasks tasks.json` (skip in dry run).
3. `hub sync-state set clickup <current_time>`.
## Push (new actions → tasks)
For each entry of `push` (open actions with an owner and no ClickUp task), unless dry run: `clickup_create_task` with `list_id`, `name` = the action title, `assignees: [<assignee>]` when present, `due_date` when present, `markdown_description` = "Hub item <id>: <title>". Then record the link: `hub set <id> clickup_id=<task id>` and `hub entry <id> "ClickUp task created." --source clickup:<task id>`. Do this after the approval batch, so actions approved in this run get tasks too.
## Never
Change a task's status, assignee or due date from the hub (ClickUp owns them); delete tasks. Mismatches are flags for a human.
