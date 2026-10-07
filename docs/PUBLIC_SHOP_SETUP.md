# Public shop deployment and contribution board

Scientific effect: **NONE**. This page records verified public deployment and review-settings installation. A live site or installed setting does not promote mathematical status.

## Pages

**Verified live on 2026-09-26:** [https://d6g8k5htny-coder.github.io/main/site/](https://d6g8k5htny-coder.github.io/main/site/). [main Pages settings — owner-only](https://github.com/d6g8k5htny-coder/main/settings/pages) use **Deploy from a branch**, branch **main**, folder **/docs**. The entry file redirects to `site/`, and `.nojekyll` serves the static app. A real-browser check loaded all 2,138 original inventory records and verified the exact SIDE24 source bytes. Pages is the public viewer; it does not write scientific status. Do not enable Pages on Math-, trial, or sandbox for this shop.

## Profile pins

The [owner profile](https://github.com/d6g8k5htny-coder) now pins **main**, **Math-**, **query-**, and **Universal-Law-Workspace**. Trial and sandbox are not shop pillars.

## Issues and Project

Issues are already in use on main. Keep tasks in [main Issues](https://github.com/d6g8k5htny-coder/main/issues), using the public task form. Create these labels if absent:

`good-first-task`, `needs-replay`, `needs-chart`, `eng-only`, `math-review`, `do-not-merge`, `results-for-review`.

The public [Public contributions Project](https://github.com/users/d6g8k5htny-coder/projects/1/views/1) is installed with Status options:

`open → claimed → results-pr → review → parked / landed-eng`.

The first five bounded tasks are [#141](https://github.com/d6g8k5htny-coder/main/issues/141), [#142](https://github.com/d6g8k5htny-coder/main/issues/142), [#143](https://github.com/d6g8k5htny-coder/main/issues/143), [#144](https://github.com/d6g8k5htny-coder/main/issues/144), and [#146](https://github.com/d6g8k5htny-coder/main/issues/146). `/claim` is a request: a maintainer assigns it after checking one active claim per person. This is a human/agent coordination convention, not an installed assignment bot. Keep deep open theorems off this introductory board.

## Branch protection and review

**Math- verified active:** [Proof vault: reviewed changes](https://github.com/d6g8k5htny-coder/Math-/rules/24045351) protects its default `main` branch. It requires a pull request, resolved conversations, and an up-to-date **math-downstream-gates** check from GitHub Actions. Force pushes and deletion are blocked; the bypass list is empty.

**Main verified active:** [Verified pull requests](https://github.com/d6g8k5htny-coder/main/rules/23798639) requires a pull request, resolved conversations, and up-to-date **verify**, **public-shop**, and **public-intake** checks from GitHub Actions. Force pushes and deletion are blocked. The existing repository-admin bypass is restricted to pull requests; it was not expanded. See the [installation receipt](https://github.com/d6g8k5htny-coder/main/pull/149#issuecomment-5848091891), recorded separately from the deployment PR. *Correction, 2026-10-07 (live read):* the bypass list of that ruleset holds the ChatGPT Codex Connector integration (1144995) with bypass mode `always`, which exempts it from the pull-request requirement and the checks on `main`; the ruleset was last updated 2026-09-28T06:22Z, after the receipt. The sentence before this one is the 26 September snapshot, superseded; see [OP-ACCESS-20261007](../governance/OP-ACCESS-20261007.md), where the disposition (remove, or record as an accepted exception) is pending. The external results guard is **public-intake**; its tests are a different check. A green replay is not mathematical acceptance. Do not grant outsiders direct write access.

**Single-account autonomous operation (owner-directed correction, 2026-09-26):** both rulesets require **zero GitHub approving reviews**, and approval of the latest reviewable push is disabled. This removes the second-account dependency: authorized agents may review and merge eligible PRs without requesting routine permission from Dylan. Do not reinstate account-count approval gates as a substitute for agent review. Preserve nonauthor technical reviews as exact-source comments or artifacts, disclose authorship and provider, and give same-provider reviews zero organizational-independence credit. Required theorem-specific review predicates and unresolved scientific holds remain separate from engineering merge eligibility. Dismissal of stale optional approvals remains enabled; the Copilot extra-approval option applies only when a nonzero approval count is required. Neither repository has classic branch protection configured. This supersedes the approval-count/latest-push settings in the historical installation receipt; all other protections described above remain.

A check name alone does not authenticate the workflow. Before merging an intake PR, inspect the **trusted public-intake workflow**, its `pull_request_target` event, the precise reviewed PR head, and the pass result. Administrative override must not be treated as a scientific review.

Do not merge Math #60/#64/#69/#71–#74 or main #122/#128 to populate this shop. Landed custody copies, results intake, and notebook figures never adopt source status labels or promote a theorem.

## Public account and repository baseline

Installed and read back on **2026-09-26**, under the owner's release-architecture directive:

| Surface | Recorded state |
|---|---|
| Four profile pins | `main`, `Math-`, `query-`, `Universal-Law-Workspace`, in that order; trial and sandbox excluded |
| Profile bio and website | Gaussian random fields, persistent homology, reproducible research; links to the live shop |
| Four repositories' About panels | Plain-language front door / vault / lookup / federation descriptions and shop links |
| Topics on all four | `mathematics`, `random-fields`, `persistent-homology`, `research`, `reproducible-research` |
| Wikis | Off on all four; documentation stays in versioned files |
| Automatic head-branch deletion | On on all four; this does not delete open branches or research history |
| main Issues | On; public task form offers replay, chart, review-comment, catalog-stub, docs |
| main and Math- private vulnerability reporting | Enabled; see each repository's `SECURITY.md` for its private disclosure URL |
| main and Math- secret scanning / push protection | Already enabled; verified and retained |
| main and Math- CODEOWNERS | Owner routing, including STATUS, PROOF_INDEX, claims, and shop config; **no required code-owner approval** |
| main and Math- Dependabot | GitHub Actions only, weekly grouped updates, one open update PR maximum |

[Math- #78](https://github.com/d6g8k5htny-coder/Math-/pull/78) landed these three operational files at `66e39d1894d5c11bb5930b76692b5d0c71939efd`. Its comparison with `d6628da09384728992dcbe6e921cc28ba85aebb0` changes only `SECURITY.md`, `.github/CODEOWNERS`, and `.github/dependabot.yml`. Source pins stay immutable; a documentation-only tip move is not a reason to rewrite proof identities. Before changing any query pin, run its exact payload compatibility check, `python -B -S verify_portable_stubs.py --check-math-tip`, from the pinned query checkout. A passing check is engineering evidence only.

The unchanged query `check_math_tip_drift` function was executed from `c88768bb11efd1f7d6bda188f13064bedec54a06` using seven downloaded, Git-blob-verified payloads at Math `66e39d1894d5c11bb5930b76692b5d0c71939efd`: all seven checked, `drifted: []`. This was a captured-byte compatibility execution, not a full network CLI or unrelated package-suite replay. No query issue, PR, or pin churn was needed.

No new CodeQL workflow or required check was added: none existed on the four audited pillars. No package dependencies, npm lockfile, funding configuration, theorem release, or new license terms were introduced. Main's existing LICENSE and CITATION.cff remain. Query has no LICENSE and remains release-ineligible. Social-preview artwork and Codespaces configuration are optional follow-ups, not claimed installed controls; the existing pinned notebook/Colab route remains.

## Profile banner and remaining owner-only steps

The explicitly requested [profile README repository](https://github.com/d6g8k5htny-coder/d6g8k5htny-coder) was created publicly. This special repository places a banner on the account page; all exhibits still live in `main/docs/site`. Its generated starter README is **not yet the prepared research banner**. The connected GitHub integration returned `403 Resource not accessible by integration` for writes to this newly created repository. The banner remains UNDELIVERED; no alternate write route or permission change was used.

To enable the authorized banner delivery, the owner can open [GitHub installed applications](https://github.com/settings/installations), choose **Configure** for the connected GitHub app, and include `d6g8k5htny-coder/d6g8k5htny-coder` in its repository access. The agent can then submit the prepared README through a normal PR. Do not grant broader permissions than needed. This account-level access step is not a new recurring approval requirement for ordinary research or engineering work.

For account security, open [Password and authentication](https://github.com/settings/security), review **Two-factor authentication**, and enable it if absent. Follow GitHub's device/authenticator prompts and store recovery codes privately. Review passkeys/security keys and recovery methods there. Account 2FA state was **not inspected or changed** in this release; no credentials or recovery material were collected.

For later repository audits: **Settings → Advanced Security** shows private vulnerability reporting, Secret Protection, and Push protection. **Settings → General → Features** controls Wikis; **Pull Requests → Automatically delete head branches** controls post-merge cleanup. Existing rule settings are described above. These paths are an operations checklist, not a second proof or status register.

## Intake and factory boundaries

[PR #153](https://github.com/d6g8k5htny-coder/main/pull/153) delivered the reviewed source-reachability and PNG structural checks. Its 51 tests passed in normal and optimized Python during nonauthor review. Out-of-tree, renamed-outside, second-page, and guard-edit negative controls passed. The trusted `public-intake` workflow continues to read default-branch code and treats submitted files as data; do not execute packet code. This is a tested boundary, not a claim that every malicious input is detectable. JSON overflow and broader catalog-path grammar concerns were left explicitly for the author's bounded successor.

[PR #150](https://github.com/d6g8k5htny-coder/main/pull/150) landed at `a12c178c0f857a130cf434e9efd44233a038195b` after its fresh-base check. It remains an illustrative `REVIEW_REQUIRED` packet with `scientific_acceptance: false`. The chart generator is not in that packet; the release reviewer verified its bytes and source correspondence, not independent chart regeneration. The museum's separately pinned packet index is display only.

Public shop work belongs in `main/docs/site`; submissions belong in `main/incoming/<id>/`. Trial is an internal load-test/workbench lane. Do not launch a new Batch NNN or trial PR to implement this shop, and do not bump meta-framework v55 for it. An additive trial `AGENTS.md` note was [requested from the existing #142 writer](https://github.com/d6g8k5htny-coder/trial/pull/142#issuecomment-5848446686); no competing edit was made, and requested is not delivered. Sandbox remains a workbench and is not featured on the profile.

See the [release dispatch log](RELEASE_DISPATCH_20260926.md) for exact lane ownership, write scopes, reviews, and outstanding actions. Integration and publication receipts belong on [main #154](https://github.com/d6g8k5htny-coder/main/issues/154).
