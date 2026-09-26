# Public shop deployment and contribution board

Scientific effect: **NONE**. This page records the remaining GitHub configuration and the exact reviewable repository implementation. It does not claim settings are already installed.

## Pages

In [main Pages settings](https://github.com/d6g8k5htny-coder/main/settings/pages), choose **Deploy from a branch**, branch **main**, folder **/docs**, then Save. The entry file and `.nojekyll` are committed. Verify the deployment before advertising the public Pages URL as live. Do not enable Pages on Math-, trial, or sandbox for this shop.

## Profile pins

On the [owner profile](https://github.com/d6g8k5htny-coder), use Customize your pins. Select **main**, **Math-**, **query-**, and **Universal-Law-Workspace**. Leave trial and sandbox unpinned.

## Issues and Project

Issues are already in use on main. Keep tasks in [main Issues](https://github.com/d6g8k5htny-coder/main/issues), using the public task form. Create these labels if absent:

`good-first-task`, `needs-replay`, `needs-chart`, `eng-only`, `math-review`, `do-not-merge`, `results-for-review`.

Create one [main Project](https://github.com/d6g8k5htny-coder/main/projects), **Public contributions**, with Status options:

`open → claimed → results-pr → review → parked / landed-eng`.

Add the public contribution tasks to that project. `/claim` is a request: a maintainer assigns it after checking one active claim per person. This is a human/agent coordination convention, not an installed assignment bot. Keep deep open theorems off this introductory board.

## Branch protection and review

In [main branch settings](https://github.com/d6g8k5htny-coder/main/settings/branches) and [Math- branch settings](https://github.com/d6g8k5htny-coder/Math-/settings/branches), protect the default `main` branch: require a pull request, a designated review, and the applicable passing checks. For main, the external results guard is **public-intake**; its tests are a different check. For Math-, retain the existing replay/scientific gates and require review rather than treating a green replay as acceptance. Do not grant outsiders direct write access.

A check name alone does not authenticate the workflow. Before merging an intake PR, inspect the **trusted public-intake workflow**, its `pull_request_target` event, the precise reviewed PR head, and the pass result. Administrative override must not be treated as a scientific review.

Do not merge Math #60/#64/#69 or main #122/#128 to populate this shop. Landed custody copies, results intake, and notebook figures never adopt source status labels or promote a theorem.
