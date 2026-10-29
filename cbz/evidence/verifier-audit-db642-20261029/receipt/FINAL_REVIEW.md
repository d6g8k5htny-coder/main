# Main receipt repair: final whole-branch review

Reviewer: OpenAI / Codex `/root/receipt_final_review`, 2026-10-29. Engineering source/evidence review, not scientific alignment review or a claim of organizational independence.

Reviewed worktree: `/workspace/scratch/db6428847655/cbz/worktrees/main-cbz`.
Base: `81acc060341e8a31cc4cd4f9523ef71fb6c06175`.
Head: `5a3f4cddbb7b285e62312a8e5265aaa010d11fb5`.
Scope: full 44-file branch delta, accepted receipt spec/plan, source integration, retained evidence and new reproduction documentation. Generated inventories/logs were inspected directly as needed rather than treated as implementation source. No remote actions, checkout/index/HEAD changes, suite reruns or subagents. Only this report was written outside the checkout.

## Ready-to-merge verdict

**NO for the whole branch at this head: correct the two new tool-invocation documentation defects below.** The receipt production/binding implementation itself is acceptable on source review; no production correctness, proof-preservation or test-weakening defect was found. Both requested fixes are documentation-only. There is no reason to change the strict validator or historic manifest boundary to make the incorrect commands pass.

After those documentation fixes and regeneration of the affected `SHA256SUMS` entry, this review identifies no remaining source blocker. Actual required hosted checks at the integration source remain a separate mandatory condition. The recorded local Python results and this source verdict establish no fresh Lean/kernel or hosted PASS.

## Actionable findings

### P2 / Important — Unsupported source-only flag in registration command

Location: `cbz/evidence/verifier-audit-db642-20261029/TOOL_REGISTRATION_ADDENDUM.md:7`.

The newly advertised command is `python3 -B -S tools/formal_gate_check.py --source-only`. That option does not exist in the parser (`tools/formal_gate_check.py:362` through `:369`). Source-only checking is already the default. A contributor copying the admission/registration command gets argparse exit 2 and performs no source check. This conflicts with the addendum's role as an exact invocation input to canonical tool documentation.

Focused reproduction at the reviewed head: exit 2, `formal_gate_check.py: error: unrecognized arguments: --source-only`.

Fix: document `python3 -B -S tools/formal_gate_check.py`, matching both the retained `receipt/focused-results.json` command and `.github/workflows/formal-verification.yml:63`. Do not add a redundant CLI mode just to accommodate this typo.

### P2 / Important — Historical builder invocation omits mandatory input and historical root

Location: `cbz/evidence/verifier-audit-db642-20261029/TOOL_REGISTRATION_ADDENDUM.md:9`.

`python3 -B -S tools/atlas_formal_evidence_build.py --check` cannot reproduce the promised evidence: `--snapshot` is mandatory (`tools/atlas_formal_evidence_build.py:164`), and the default current checkout cannot satisfy the fixed historical manifest (`:71`–`:72`). Adding only a snapshot still fails by design. A reader following this new registration row cannot distinguish incomplete documentation from a broken historical route.

Focused reproductions at the reviewed head:

- Exact documented command: exit 2, `the following arguments are required: --snapshot`.
- Same command plus `--snapshot tests/fixtures/atlas_formal_projection.json`: exit 1, `ValueError: formal manifest changed`.

Fix: explicitly document the qualified historical projection and snapshot arguments, their preparation/prerequisites, and the intentional current-root refusal, consistent with `docs/site/README.md:463`. Alternatively label the CLI as requiring an externally prepared pinned historical projection and give the existing focused fixture test as the runnable current-checkout reproduction route. Do not loosen the manifest check or suggest that an arbitrary snapshot grants historical identity.

## Source and integration assessment

- `tools/formal_gate_check.py:304` emits the actual `GITHUB_RUN_ATTEMPT` value, with null when absent. There is no inferred/default attempt. `tests/test_formal_gate.py:444` verifies non-default attempt 7 in returned and serialized receipt; `:455` verifies absent hosted fields stay null.
- `tools/required_formal_check.py:46` already validates the consumer attempt as a positive decimal string. The added comparison at `:60` requires exact equality with that string. Missing, null, numeric, boolean, zero, padded, whitespace-bearing and stale attempts cannot pass. Existing commit, repository, run, manifest and log checks remain intact.
- The live CLI at `tools/required_formal_check.py:99` still exposes only `receipt` and `aggregate`; receipt binding at `:106` cannot dispatch to a historical mode. The historical builder imports only `strict_json` from this module. Its separate helper is not a live fallback.
- `tools/atlas_formal_evidence_build.py:54`–`:66` pins archive, manifest and original receipt SHA-256 before parsing, verifies actual log bytes, and returns only the original fixed identity. The assembler compares that result to the original archive binding at `:105`. Omitted repeated schema/hosted-context validation is safe for this exact digest-pinned historical record; it does not authorize arbitrary legacy data.
- Added historical tests exercise original byte/binding preservation, actual legacy receipt refusal by the live binder, altered identities/attempt insertion, changed archive/manifest/logs, and exact reproduction of published evidence. Existing assertions remain. The positive fixture update in `tests/test_required_formal_binding_control.py:38` supplies its already-established attempt 2 rather than weakening acceptance.
- Frozen bridge profile controls are literal `.txt` evidence and are not executed. Their bytes equal the base producer/test Git objects. Current validators still run against those historical fixtures.
- Direct byte comparison confirms all 12 implementation blobs at original tested commit `bae3c5866656ad99e88b5839f585e47d8a1fce40` equal rebased commit `d75c27c2a97ee21e875db3819a057560e3f7f1e5` and current worktree bytes. The intervening base changes from `4ef1867` to `81acc060` affect only `AGENTS.md`, `DECISIONS.md`, and `GOVERNANCE.md`.
- Direct manifest comparison confirms precisely two existing control digest changes, both equal the current file SHA-256. Non-file metadata and every other bound digest are unchanged. The original execution archive, receipt and published evidence module remain byte-identical to base. No mathematical source was changed.

## Evidence assessment and verification performed

- All 29 entries in the contribution's `SHA256SUMS` pass. This establishes package integrity, not semantic truth or a fresh run.
- The retained final full-run log agrees with `full-results.json`: 1551 tests, exit 0, 24 explicit skips. The retained first full-run log agrees with `full-results-first.json` and preserves the positive-fixture failure. Focused summaries total 130 tests per mode, one Lean skip per mode. These are inspected prior results, not reruns by this reviewer.
- Mutation summary records removal of only the attempt comparison, three adverse methods and 11 expected assertion failures, exit 1. The prior accepted task review separately inspected the isolated mutant and raw mutation output. This final review does not claim to have rerun that mutation.
- README source/ref counts agree with the retained census/snapshot fields: ten public repositories, Main 276 branches, Math 149, and the stated remaining branch counts; Math 4019 blobs with no mismatches, 162 Lean files, eight live roots and 43 files outside those roots. Inventories identify their source cuts; their statements about open gates and historical runtime evidence are not promoted to current integration results.
- The contribution clearly distinguishes original observations, repaired Python results, unrun hosted work, incomplete federation gates and absent STATUS-only cold-start acceptance. Existing task review predates final discovery; README and implementation report correctly describe that chronology. No inference of scientific acceptance or current kernel closure was found.
- I ran only the three short CLI checks needed to establish the named documentation defects, direct hash/blob/log comparisons, package checksum validation, and Git status/diff inspection. Full and focused Python suites were not rerun.
- `git diff --check` reports raw trailing whitespace in preserved `receipt/red-binder.log:14` and final blank lines in the new plan/spec. These are cosmetic, not correctness findings. Preserve raw failure-log bytes; do not rewrite that historical output to silence a whitespace check.
- Worktree was clean and HEAD remained the specified review head on final source inspection.

## Outside-scope considerations and reasons

1. Hosted merge/required checks, fresh Lean build/kernel/axiom verification, and latest remote integration heads: explicitly delegated to the canonical integration writer; no remote actions or local kernel claims are part of this review.
2. Math packet repairs and federation-wide G0–G11 closure: this Main branch retains bounded discovery evidence only. Their independent implementations and semantic coverage cannot be certified from the Main receipt fix.
3. Current status/ownership of PR347, PR358 and all repository refs: snapshots are explicitly interval/source-pinned observations. Remote freshness must be reconciled by integration, not inferred from the inventory.
4. Completeness of full discovery payload retention: copied worker reports mention additional local files such as individual focused logs, `mutation.log`, and `completion.json` that are not all in the published subset. This review uses the named retained summaries/full logs and prior review, not those unavailable paths; the package does not establish a complete raw discovery archive. An explicit retained-versus-local index would improve portability but is not needed to establish the bounded source fix.
5. General signed-artifact authentication, concurrent filesystem adversaries, or redesigning the binder: no new such exposure is introduced; the task requires exact current-run attempt equality within existing workflow custody. The historical helper's fixed hashes remain its intentional boundary.
6. Existing current-root historical-builder refusal: intentional and already tested, not a production regression. Only the new incomplete command is actionable here.
