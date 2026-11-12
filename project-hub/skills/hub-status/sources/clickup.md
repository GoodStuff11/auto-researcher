# ClickUp
ClickUp owns task status, assignee and due date. The hub mirrors them onto actions that carry `clickup_id`, stages tasks created directly in ClickUp, and gets a ClickUp task created for each new owned action.
## Configuration
`project.yaml`: `sources.clickup.list_id` (and optional `done_statuses`, default complete/completed/closed/done); `people.<key>.clickup` = that person's ClickUp user id.
## Export
1. `clickup_get_user` → note `current_time` (used for the watermark and for any relative date).
2. `clickup_filter_tasks` with `list_ids: [<list_id>]`, `include_closed: true`, `fields: ["status", "assignees", "due_date"]`, `page: 0`; repeat with `page: next_page` while `has_more` is true.
3. Write every task to `$SCRATCH/tasks.json` (the scratch directory outside the vault, see SKILL.md) as `{"id", "name", "status", "assignees", "due_date"}`, passing the connector's values through unchanged: `status` as returned (a name string, or `{status, type}`), `assignees` as strings (the ids, usernames or emails the connector returns), `due_date` as epoch milliseconds (or null). `hub clickup plan` and `apply` also accept `due` (`YYYY-MM-DD`) instead.
## Reconcile
1. `hub clickup plan --tasks "$SCRATCH/tasks.json"` — read `set`, `proposals`, `flags`, `push` and sanity-check (a mass of unexpected changes means a wrong list id: stop and ask).
2. `hub clickup apply --tasks "$SCRATCH/tasks.json"` (skip in dry run).
3. Only if `clickup apply` succeeded: `hub sync-state set clickup <current_time>`, then `hub commit "status: clickup (<YYYY-MM-DD>)"` (SKILL.md, update step 3). If it failed, show the error and leave the watermark alone.
4. `apply` stages a `New ClickUp task: <name>` proposal for each open task no action carries yet, and flags `clickup: actions A, B share task <id>` if two actions carry one task (`hub validate` reports it as `duplicate-clickup-id`; tell the user, who decides which action to keep). Before approving a `New ClickUp task` proposal, follow SKILL.md update step 6 (an open action with the same title gets the link instead).
## Push (new actions → tasks)
Run this after the approval batch and `hub validate` (see SKILL.md): re-run `hub clickup plan --tasks "$SCRATCH/tasks.json"` and take its `push` list. For each entry (open actions with an owner and no ClickUp task), unless dry run, first look in `tasks.json` for an open task whose `name` equals the action title (whitespace and case-insensitive; the export has no descriptions). If one exists, call `clickup_get_task` on it: only if its description contains "Hub item <id>" (an earlier push that died before `hub set`), link it (`hub set <id> clickup_id=<task id>` and `hub entry <id> "Linked to existing ClickUp task." --source clickup:<task id>`) and skip creation. Otherwise `clickup_create_task` with `list_id`, `name` = the action title, `assignees: ["<assignee>"]` (a string) when present, `due_date` when present, `markdown_description` = "Hub item <id>: <title>". Then record the link: `hub set <id> clickup_id=<task id>` and `hub entry <id> "ClickUp task created." --source clickup:<task id>`.
## Never
Change a task's status, assignee or due date from the hub (ClickUp owns them); delete tasks. Mismatches are flags for a human.
