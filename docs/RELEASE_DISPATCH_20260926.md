# Release architecture dispatch — 2026-09-26

Scientific effect: **NONE**. This log records engineering work, not theorem status. Main owns the only integrated shop; no exhibit repository or second status register was created.

| Lane / issue | Assignee and branch | Allowed paths | Status at this checkpoint |
|---|---|---|---|
| [Main integration #154](https://github.com/d6g8k5htny-coder/main/issues/154) | OpenAI Codex root; `codex/release-architecture-20260926` | `docs/site/**`, museum/export tools and tests explicitly listed in issue, setup/dispatch docs, `SECURITY.md`, `.github/CODEOWNERS`, `.github/dependabot.yml` | One remote writer; display worker edits locally only; review and complete-fixture CI required before merge |
| [Vault operations #77](https://github.com/d6g8k5htny-coder/Math-/issues/77) / [PR #78](https://github.com/d6g8k5htny-coder/Math-/pull/78) | OpenAI Codex root; `codex/vault-release-ops-20260926` | `SECURITY.md`, `.github/CODEOWNERS`, `.github/dependabot.yml` only | Merged at `66e39d1894d5c11bb5930b76692b5d0c71939efd`; nonauthor engineering review and exact-head math-downstream-gates success |
| [Intake PR #153](https://github.com/d6g8k5htny-coder/main/pull/153) | Existing Claude writer; branch owned by that PR | Existing intake checker/tests scope | Merged at `1b0d4eefbe5b20dda846a9575d475ed9aa45597e`; OpenAI reviewed 51 controls normal/-O; deferred overflow/path concerns remain explicit |
| [Packet PR #150](https://github.com/d6g8k5htny-coder/main/pull/150) | Existing Claude writer; branch owned by that PR | `incoming/side24-chart-claude-20260926/**` | Merged at `a12c178c0f857a130cf434e9efd44233a038195b` after conditional OpenAI review and fresh-base trusted intake; REVIEW_REQUIRED, not scientific acceptance |
| Query compatibility | OpenAI read-only support under #154 | Read exact pinned verifier and seven current Math payloads; no write assignment | Completed: unchanged `check_math_tip_drift` executed with seven captured, Git-blob-verified payloads at Math `66e39d18`; `drifted: []`; no full network CLI claimed, no issue/PR needed |
| Profile banner under #154 | OpenAI root; planned `codex/profile-front-door-20260926` | Profile repository `README.md` only | Public repository created; prepared banner UNDELIVERED because connector write access is missing; no branch or PR created |
| [Trial #142 note request](https://github.com/d6g8k5htny-coder/trial/pull/142#issuecomment-5848446686) | Existing #142 writer; no new assignee/branch | Additive `AGENTS.md` operations note only | Requested, not delivered; root does not race the active writer |

All lanes forbid proof bodies, `lemma_closed`, prize flags, `LANDING_CLAIMS.json`, PROOF_INDEX verdicts, and STATUS ACCEPT/AMEND wording. Stop on an overlapping writer or changed scientific scope. The integration does not merge forbidden mathematical PRs. Reviewers disclose source exposure and authorship; different sessions/accounts do not establish provider independence. Local reviewers are OpenAI and receive zero organizational-independence credit.

## Validation contract

- Main: `node --test tests/test_museum_frontend.mjs tests/test_museum_geometry.mjs tests/test_public_shop_frontend.mjs`; `python3 -B -m unittest discover -s tests -p 'test_museum_data.py'`; exporter regression; complete checkout `python3 -B tools/public_shop_check.py` and `python3 -B tools/museum_check.py` in hosted CI; `git diff --check`.
- Compare remote changed-path lists and every touched file SHA-256 with the PR body. Confirm README Math/query checkout pins match the single shop config and museum source. Preserve exact-source failures and fixture limitations.
- Check current branch refs, ancestry, trusted workflow/event, and both available push/PR run collections at the actual head before merging. Never infer new-tree success from an older run or use an admin override as review evidence.
- Browser: verify deployed cards, exact SIDE24 bytes, geometry fallback, visibly hatched open complements, catalog, and packet contracts. A canvas explains a pinned source; it is not a proof.
- Query: `python -B -S verify_portable_stubs.py --check-math-tip`, with real fetched payloads. No extra coefficient digits or content-free tip refresh.
- Settings: read back About/topics, wikis off, auto-delete on, private disclosure enabled, and existing secret/push protection. No credentials, sharing changes, new required approval gate, or funding setup.

Final delivery receipts and claim release are appended to the existing issue threads. This dated checkpoint is retained as history, rather than rewritten to imply work had already completed.

## Follow-through checkpoint

The original integration landed through [PR #155](https://github.com/d6g8k5htny-coder/main/pull/155)
at `2ed91b777217fab50623402a3538fa5478076e6c`. Its PR checks, merged-tree
public-shop/verify checks and Pages deployment succeeded. The exact receipt is
[comment 5848632593](https://github.com/d6g8k5htny-coder/main/issues/154#issuecomment-5848632593).

| Lane / claim | Assignee and allowed paths | Current disposition |
|---|---|---|
| [Cache follow-through](https://github.com/d6g8k5htny-coder/main/issues/154#issuecomment-5848689223), `codex/museum-cache-followthrough-20260926` | OpenAI root is sole remote writer; local display worker owns `docs/site/museum.mjs` and `tests/test_museum_frontend.mjs`; root owns `docs/site/README.md` and this log | One successor integration: local config/manifest cache policy and regression controls; no config, source pin, intake-checker or scientific edits |
| [Claude intake #156](https://github.com/d6g8k5htny-coder/main/pull/156) | Existing Claude writer owns checker/tests; OpenAI support is read-only review/comments | Bounded successor for JSON overflow and exact catalog source paths; completion and exact-head review are recorded on the PR, not inferred from this assignment |
| Trial #142 | Existing open PR retains `AGENTS.md`; no competing writer | Fresh head `4499e2314011808e66b39be045657a52996eab94` still lacks the requested factory note; no reply or delivery observed |
| Profile banner | Prepared profile README only; no alternate write route | Still UNDELIVERED: profile default `c2c2d91de670d91c0365f515158fe0a72adcd9c0` contains the starter README; connector access remains the concrete missing capability |

Fresh live browser inspection now displays both packet cards with exact RESULT
identities and REVIEW_REQUIRED labels, correcting the earlier cached one-packet
observation. This is public-access verification, not scientific acceptance. The
cache repair prevents reuse through the browser's local cache; it does not promise
an atomic CDN release. Mixed config/manifest identities remain fail-closed.

Math ruleset 24045351 was read back with zero required approvals, no code-owner
or latest-push approval, strict `math-downstream-gates`, and an empty bypass list.
No second account or recurring owner approval is required for eligible merges.
The retained check, PR, conversation-resolution and force-push protections still
apply. Provider independence and theorem-specific predicates remain separate.
