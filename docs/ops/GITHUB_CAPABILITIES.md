# GitHub capabilities inventory (agent fleet)

Status: ops reference, NOT A RESULT. Verified 2026-10-23 against live tool schemas (read-only probes; main at be5a458e).

Three tool namespaces exist. Only two reach GitHub:

| Namespace | Reaches | Notes |
|---|---|---|
| `GitHub-xai` (GitHub's official MCP server) | github.com | Widest coverage: Actions writes, threads, Copilot review, security, projects, rulesets |
| `cursor-github` | github.com | REST + a few GraphQL mutations; has `set_pull_request_draft` |
| `cursor-origin` | **Origin-hosted repos only** | NOT usable on d6g8k5htny-coder/* GitHub repos |

## Top 10 efficiency wins

1. **Resolve review threads ourselves**: `GitHub-xai pull_request_review_write method=resolve_thread threadId=PRRT_...` (get ids from `pull_request_read method=get_review_comments`). Unblocks 405s from "conversation resolution required". Also `unresolve_thread`.
2. **Mark Ready / back to Draft**: `cursor-github set_pull_request_draft {draft:false}` (GraphQL markPullRequestReadyForReview), or `GitHub-xai update_pull_request draft:false`. Capability exists; *policy* still says only the author (or Dylan) undrafts someone else's PR.
3. **One-call sync with main**: `update_pull_request_branch` (both namespaces) with `expectedHeadSha` / `expected_head_sha`. GitHub creates the two-parent merge commit server-side, so we do not need create-tree/create-commit. Async (202): re-read head after.
4. **SHA-guarded merge**: `merge_pull_request` with `expectedHeadSha` (xai) or `sha` (cursor-github); 409 if head moved. Methods: merge/squash/rebase.
5. **Rerun flaky/failed CI**: `GitHub-xai actions_run_trigger method=rerun_failed_jobs run_id=...` (also `rerun_workflow_run`, `cancel_workflow_run`, `run_workflow` for workflow_dispatch).
6. **Failed-job logs in one call**: `get_job_logs run_id=... failed_only=true return_content=true tail_lines=200` (xai) or cursor-github `get_job_logs run_id` (tail_bytes).
7. **Extra reviewer on demand**: `GitHub-xai request_copilot_review` adds a Copilot code review (advisory; does not count as our nonauthor PASS).
8. **Check status precisely**: `pull_request_read method=get_check_runs` / `get_status`, or cursor-github `list_check_runs_for_ref ref=<sha>`. Use instead of guessing "queued".
9. **Byte-identity checks without cloning**: `compare_commits base head` (cursor-github, gives ahead/behind + patches), `get_git_tree`/`get_repository_tree` for blob SHAs, `get_commit detail=stats`.
10. **Pre-flight what blocks a merge**: `repository_ruleset_read method=get_rules_for_branch branch=main` shows required checks/threads/reviews; plus `run_secret_scanning` on diffs before pushing.

## Inventory by job

### PR lifecycle
- Create: `create_pull_request` (xai: `draft`, `reviewers`; cursor-github: `draft`).
- Update title/body/base/state: `update_pull_request` (both); xai also `draft`, `reviewers`.
- Draft<->Ready: cursor-github `set_pull_request_draft`; xai `update_pull_request draft`.
- Sync: `update_pull_request_branch` (merge-commit style only; no rebase option).
- Merge: `merge_pull_request` (merge|squash|rebase, head-SHA guard).
- Read: xai `pull_request_read` methods get, get_diff, get_status, get_files, get_commits, get_review_comments (threads + node ids + resolved state), get_reviews, get_comments, get_check_runs. cursor-github: get_pull_request, get_pull_request_diff, list_pull_request_files/commits/reviews/review_comments/comments.

### Reviews and threads
- Submit review: xai `pull_request_review_write create` (event APPROVE/REQUEST_CHANGES/COMMENT, `commitID` pins head); cursor-github `create_pull_request_review` (inline comments array, `commit_id`).
- Pending review flow: create without event -> `add_comment_to_pending_review` -> `submit_pending` / `delete_pending`.
- Threads: xai `resolve_thread` / `unresolve_thread`; replies via xai `add_reply_to_pull_request_comment` or cursor-github `reply_to_pull_request_review_comment`.
- Copilot: xai `request_copilot_review`.
- Not available: dismissing a review on GitHub, removing requested reviewers on GitHub (Origin-only tools).

### Checks / Actions
- Read: xai `actions_list` (list_workflows, list_workflow_runs w/ filters, list_workflow_jobs, list_workflow_run_artifacts), `actions_get` (get_workflow, get_workflow_run, get_workflow_job, download_workflow_run_artifact, get_workflow_run_usage, get_workflow_run_logs_url), `get_job_logs`. cursor-github: list_workflows, list_workflow_runs, list_workflow_run_jobs, get_job, get_workflow(_run), get_job_logs, list_check_runs_for_ref, get_commit_combined_status.
- Write (xai only): `actions_run_trigger` run_workflow, rerun_workflow_run, rerun_failed_jobs, cancel_workflow_run, delete_workflow_run_logs (do not use the last without Dylan).

### Branches, commits, files
- Branch: `create_branch` (cursor-github accepts exact `sha`). No branch delete / force-update tool.
- Commit: xai `push_files` (many files, one single-parent commit on branch tip), `create_or_update_file` (needs blob sha for updates), `delete_file`.
- Read: get_file_contents, get_repository_tree / get_git_tree, list_commits, get_commit, compare_commits, list_branches, get_branch, tags.
- **Limit:** no git create-tree / create-commit / update-ref, so no hand-built multi-parent commits, no force-push, no rebase. Workaround: `update_pull_request_branch` for merges with base.

### Issues, labels, projects, discussions
- Issues: xai `issue_read` (get, get_comments, get_sub_issues, get_parent, get_labels), `issue_write` (create/update, labels, assignees, state_reason, issue_fields, parent), `sub_issue_write`; cursor-github create_issue/update_issue/sub-issue tools.
- Comments: `add_issue_comment` (xai also supports emoji reactions), xai `update_issue_comment`.
- Labels: xai `label_write` create/update/delete; cursor-github create/update/delete_label.
- Projects v2: xai `projects_list/get/write` (items, bulk updates, views, status updates). Discussions: both namespaces.

### Search
- search_code, search_commits, search_issues (xai is semantic), search_pull_requests, search_repositories/users/orgs; xai `semantic_issue_similarity_search` for duplicates.

### Security
- xai: code scanning, Dependabot, secret scanning alerts (list/get), `run_secret_scanning` on content, `check_dependency_vulnerabilities`, global/repo advisories, `get_code_quality_finding`.

### Releases, repo admin, misc
- Releases/tags: read-only (no create_release tool).
- Admin-scope writes exist (`create_repository_ruleset`, `custom_properties_write`, `create_repository`, `fork_repository`): **settings changes, Dylan-only.**
- Notifications, gists, stars, Copilot Spaces, `github_support_docs_search`.

## Other GitHub resources (outside our tools)
- **Copilot coding agent**: assign an issue to Copilot (or @copilot in a PR) and it opens a draft PR from a GitHub Actions sandbox. Needs a Copilot plan + repo enablement.
- **Codex cloud / Codex GitHub review**: @codex review on PRs; leaves review threads (the kind that blocked #552).
- **Claude Code GitHub Actions** (`anthropics/claude-code-action`): workflow triggered by @claude mentions. Needs a workflow file + API secret.
- **Auto-merge** (`enablePullRequestAutoMerge` GraphQL / `gh pr merge --auto`): merges once checks + reviews pass. Repo setting "Allow auto-merge" must be on; no tool in our namespaces exposes it.
- **Merge queue**: serializes merges and tests each against latest main, would replace our MAIN-LOCK protocol. Ruleset change.
- **CODEOWNERS**: auto-request reviewers by path; enforcement requires a ruleset.
- **gh CLI**: `gh pr ready`, `gh pr merge --auto`, `gh run rerun --failed`, `gh api graphql`. Needs a token on the box; we do not mint or read tokens.

## Available now vs needs Dylan

Available now: everything listed under Top 10 and the inventory, within fleet policy (nonauthor review, SHA guards, MAIN-LOCK).

## Needs Dylan
1. Policy call: may integrators undraft PRs whose author has gone quiet (tool exists: `set_pull_request_draft`)? Currently 5 PRs wait on GPT-6.
2. Enable "Allow auto-merge" and/or a **merge queue** on main (would remove MAIN-LOCK races like #540/#554).
3. Optional: CODEOWNERS, Copilot coding agent, Claude/Codex Actions (workflow files + secrets).
4. Any ruleset, custom-property, workflow-log deletion, or repo create/fork action.

— Grok Bot Chief of Staff (Cursor-hosted Grok Bot, not xAI Grok)
