**CONDITIONS RECORDED BEFORE MERGE — `M466-INTEGRATION-017-20261020`, under the standing usage-limit route in `governance/OP-WORKFLOW-20260930.md`.** Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017), the integrator Dylan named ([466/6301931398](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6301931398)). Credit 0. Scientific effect NONE. This record is not a review.

**1. Identities**
- **Head:** `3dd0be245634f3b9f75bd90709993183c98b7f1c`, tree `7af68e5ba6b8cb2cd32fe584c332f5df85a43434`. It is my guarded update-branch merge at 21:31:38Z (expected head `8ba9bf2a`), with parents `8ba9bf2a`, the reviewed head, and `4c7354e5`.
- **Base:** current `main` `4c7354e5e629e8ded3cdc9b7ad33b47c22bfe49d`.
- **Tested merge:** GitHub's `refs/pull/466/merge` is `abaed7d3`, with parents `4c7354e5` and `3dd0be24`. Its tree, `7af68e5b`, equals the head tree, and that is also the tree my local `merge-tree` predicted before the refresh.
- **Diff from `main`:** only `docs/WORKSPACE.md` (blob `1da250d1`), one hunk of +8/−6.
- **Mergeability:** `mergeable` true; `mergeable_state` clean at 21:43:37Z.

**2. Scope: covered.** This is documentation only: the workspace-guide paragraph on comparing a merge base with the full SHA. It touches no scientific statement, finding, review disposition, register or index, workflow, permission, credential, ruleset, protection or decision authority.

**3. The usage-limit failure, established at the exact head**
- I marked the PR Ready at 21:31:50Z, and the summary records the trigger "Draft marked ready".
- The connector's explicit notice is [6301950316](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6301950316) (21:31:55Z): "You have reached your Codex usage limits for code reviews."
- The summary [6301949530](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6301949530) binds `3dd0be24` with `status: failed`.
- There are no native review objects, review threads or inline comments, so the failure is quota-only and produced no output.
- The summary's failed-state flags (`has_any_findings: true`) are not treated as findings, nor as evidence that there are none. The same flags sit on #465 (`b7b80650`) and #467 (`72de805c`), neither of which has a single Codex review object.

**4. Condition 1: the reviews cover the current head and base. Met.**
- **The review.** 014K's PASS_SCOPED at `8ba9bf2a` on `a1d79d72` ([466/6295260399](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6295260399)) is same-provider, so it carries credit 0.
- **The composition assessment** for the base-only refresh, from an eligible nonauthor who is not the integrator: xAI Grok 4.7 via Cursor `bc-fbed80de`, PASS_SCOPED_COMPOSITION and release, [466/6302069899](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6302069899). I summoned it ([466/6302010113](https://github.com/d6g8k5htny-coder/main/pull/466#issuecomment-6302010113)). It reports:
  - the reviewed hunk body is byte-identical: both full-index patches are 1,635 B, `43e6dd63…` reviewed and `263966d9…` refreshed, and they differ only in the index line and the hunk position;
  - the base change is compatible: main#465 changed the zeta-fold paragraph at `@@ -20`, and main#467 does not touch the file;
  - the validations pass at the head: `workspace_landing_check.py` gives `"problems": []` over 16 pages and 188 links, and `tests.test_workspace_landing` gives 15 OK, also under `-O`.

  I re-ran each of these points locally, and they reproduce.
- **My integrator data agrees.**
  - The PR's changed lines in `docs/WORKSPACE.md` at `3dd0be24` are identical to those of `a1d79d72..8ba9bf2a`.
  - main#465 edited a different hunk of the same file, and the merge is clean.
- **Review threads:** 0.

**5. Condition 2: the required checks pass on the current tested merge. Met.** The head contains `main`, so the tested tree is the head tree.
- `verify`: success, completed 21:42:56Z (run [39501074196](https://github.com/d6g8k5htny-coder/main/actions/runs/39501074196), `workspace-landing`, `pull_request`).
- `public-shop`: success (run 39501074130).
- `public-intake`: success in runs [39501070672](https://github.com/d6g8k5htny-coder/main/actions/runs/39501070672) (from the refresh) and [39501105419](https://github.com/d6g8k5htny-coder/main/actions/runs/39501105419) (from Ready). Both are event `pull_request_target`, both use the trusted default-branch workflow `.github/workflows/public-intake.yml`, and both bind head `3dd0be24`, as CONTRIBUTING requires.
- The other check runs at the head: `landing-checks`, `architecture / research-architecture` and `formal / formal-evidence` (run 39501074196), `browser-smoke` (run 39501074130) and `navigation` (run 39501074022) all succeeded. That makes 9 of 9 check runs success and none pending.

**6. Condition 3: protections remain enforced. Met.**
- Ruleset 23798639, "Verified pull requests", is active on `main`. It requires:
  - a pull request with 0 required approvals and review-thread resolution;
  - strict required checks: `verify`, `public-intake` and `public-shop`.

  `main` is also protected against deletion and non-fast-forward updates.
- The ruleset lists one bypass actor (an integration, mode "always"). This merge does not rely on it: every rule is satisfied at the exact head. The branch is up to date, the required checks are green and there are 0 threads.
- I did not make the classic-protection read, which Dylan chose to check himself.

**Next:** one merge-commit merge at expected head `3dd0be245634f3b9f75bd90709993183c98b7f1c`, then the landed readback, the post-merge checks and my release.

---
_Generated by [Claude Code](https://claude.ai/code)_
