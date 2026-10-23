# GitHub capabilities inventory (agent fleet)

Status: ops reference, NOT A RESULT. Verified 2026-10-23 against live tool schemas (read-only probes; main at be5a458e).

Three GitHub namespaces exist (namespace ids as exposed to agents). Only two reach github.com:

| Namespace | Reaches | Notes |
|---|---|---|
| `user-GitHub-xai` | github.com | Widest coverage: Actions writes, threads, Copilot review, security, projects, rulesets |
| `cursor-github` | github.com | REST + a few GraphQL mutations; has `set_pull_request_draft` |
| `cursor-origin` | **Origin-hosted repos only** | NOT usable on d6g8k5htny-coder/* GitHub repos |

## Top 10 efficiency wins

1. **Resolve review threads ourselves**: `user-GitHub-xai pull_request_review_write method=resolve_thread threadId=PRRT_...` (get ids from `pull_request_read method=get_review_comments`). Unblocks 405s from "conversation resolution required". Also `unresolve_thread`.
2. **Mark Ready / back to Draft**: `cursor-github set_pull_request_draft {draft:false}` (GraphQL markPullRequestReadyForReview), or `user-GitHub-xai update_pull_request draft:false`. Capability exists; *policy* still says only the author (or Dylan) undrafts someone else's PR.
3. **One-call sync with main**: `update_pull_request_branch` (both namespaces) with `expectedHeadSha` / `expected_head_sha`. GitHub creates the two-parent merge commit server-side, so we do not need create-tree/create-commit. Async (202): re-read head after.
4. **SHA-guarded merge**: `merge_pull_request` with `expectedHeadSha` (xai) or `sha` (cursor-github); 409 if head moved. Methods: merge/squash/rebase.
5. **Rerun flaky/failed CI**: `user-GitHub-xai actions_run_trigger method=rerun_failed_jobs run_id=...` (also `rerun_workflow_run`, `cancel_workflow_run`, `run_workflow` for workflow_dispatch).
6. **Failed-job logs in one call**: `get_job_logs run_id=... failed_only=true return_content=true tail_lines=200` (xai) or cursor-github `get_job_logs` (tail_bytes). Limit: with only `run_id`, it returns at most 3 failed jobs and the last 64 KiB; pass a `job_id` for anything beyond that.
7. **Extra reviewer on demand**: `user-GitHub-xai request_copilot_review` adds a Copilot code review (advisory; does not count as our nonauthor PASS).
8. **Check status precisely**: `pull_request_read method=get_check_runs` / `get_status`, or cursor-github `list_check_runs_for_ref ref=<sha>`. Use instead of guessing "queued".
9. **Byte-identity checks without cloning**: `compare_commits base head` (cursor-github, gives ahead/behind + patches), `get_repository_tree` (user-GitHub-xai) / `get_git_tree` (cursor-github) for blob SHAs, `get_commit detail=stats`.
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
- Read: get_file_contents, get_repository_tree (xai) / get_git_tree (cursor-github), list_commits, get_commit, compare_commits, list_branches, get_branch, tags.
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
Sources checked 2026-10-23.
- **Copilot coding agent**: assign an issue to Copilot (or mention @copilot on a PR); it works in a GitHub Actions sandbox and opens a draft PR. Needs a Copilot plan + enablement. [docs](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent)
- **Codex cloud / Codex GitHub review**: `@codex review` on PRs; it leaves review threads. Example: main#552 thread `PRRT_kwDOQpeC2s6vvOIw` ([r4363242632](https://github.com/d6g8k5htny-coder/main/pull/552#discussion_r4363242632)), resolved by 20:40Z on 2026-10-23. [Codex cloud](https://developers.openai.com/codex/cloud), [GitHub integration](https://developers.openai.com/codex/integrations/github)
- **Claude Code GitHub Actions**: [anthropics/claude-code-action](https://github.com/anthropics/claude-code-action), triggered by @claude mentions. Needs a workflow file + API secret (Dylan-only).
- **Auto-merge** (`enablePullRequestAutoMerge` [GraphQL mutation](https://docs.github.com/en/graphql/reference/mutations) / `gh pr merge --auto`): merges once required checks + reviews pass. Repo setting "Allow auto-merge" must be on; none of our namespaces exposes it. [docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request). **It does not remove the MAIN-LOCK race**: auto-merge does not update a PR branch that has fallen behind main, so with "require branches up to date" it just stalls, and without it, it merges untested combinations.
- **Merge queue**: tests each PR against latest main + PRs ahead of it. **Not available on this repo**: GitHub gates it to public repos owned by an *organization* (or Enterprise Cloud private repos); d6g8k5htny-coder is a personal account. Sources: [GitHub docs gated-feature text](https://github.com/github/docs/blob/main/data/reusables/gated-features/merge-queue.md), [GA changelog](https://github.blog/changelog/2023-07-12-pull-request-merge-queue-is-now-generally-available/), [Managing a merge queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue); third-party report of a 422 on a user-owned repo: [Aureliolo/synthorg 3b78f93](https://github.com/Aureliolo/synthorg/commit/3b78f93acac81cad3e9e941f7b64b84311baf55f). Using it would also require adding an `on: merge_group` trigger to **every required workflow** ([events reference](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)), which is a Dylan-only workflow edit.
- **CODEOWNERS**: auto-requests reviewers by path; enforcement needs a ruleset. [docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- **gh CLI**: `gh pr ready`, `gh pr merge --auto`, `gh run rerun --failed`, `gh api graphql`. Needs a token on the box; we do not mint or read tokens. [manual](https://cli.github.com/manual/)

## Available now vs needs Dylan

Available now: everything listed under Top 10 and the inventory, within fleet policy (nonauthor review, SHA guards, MAIN-LOCK).

## Needs Dylan
1. Policy call: may integrators undraft PRs whose author has gone quiet (tool exists: `set_pull_request_draft`)?
   - Source: GitHub search `repo:d6g8k5htny-coder/main is:pr is:open draft:true GPT-6`, as of 2026-10-23 20:50Z (15:50 CDT). It returned 24 open drafts that mention "GPT-6" anywhere (mentions, not verified authorship): #315 #343 #350 #358 #385 #389 #399 #405 #407 #408 #447 #460 #471 #478 #484 #485 #518 #519 #530 #564 #568 #572 #573 #575.
   - Agent 1's review of #575 identified about 16 of those as GPT-6-authored drafts: #315 #350 #358 #385 #405 #407 #408 #460 #471 #484 #518 #519 #530 #568 #572 #573. All 16 appear in the search result. Separately, #485 and #447 have GPT-6/6.1 reviewers.
   - Review-passed and waiting on an author/Ready step, per agent 1 (not re-verified here): #566, #572, #573, Math-#500, query-#50.
2. MAIN-LOCK race, sourced: #554 merged at 20:03:16Z while #540 held MAIN-LOCK. #540 then got a 405 at 20:16Z ([540/6373876826](https://github.com/d6g8k5htny-coder/main/pull/540#issuecomment-6373876826)). The fix was posted at [539/6373887135](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6373887135). Auto-merge alone would **not** prevent this (see above). A merge queue would, but it is unavailable on a personal-account repo; it needs a transfer to an organization plus `merge_group` workflow edits. Dylan decides.
3. Optional: CODEOWNERS, Copilot coding agent, Claude/Codex Actions (workflow files + secrets).
4. Any ruleset, custom-property, workflow-log deletion, or repo create/fork action.

— Grok Bot Chief of Staff (Cursor-hosted Grok Bot, not xAI Grok)
