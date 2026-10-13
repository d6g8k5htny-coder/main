# Population certificate A1 repair: source-bound handoff

This is the additive engineering custody record for the isolated A1 repair of main #343. It preserves an offered successor, not an adoption into the original branch, an integration decision, or a scientific acceptance declaration.

## Identity and disposition

- Frozen upstream: `5bf0bcd461b4b7de49052680dd6d4c6e8ee11e1f`
- S v2 adoption: `aca0f9e4dfe54dca3607f6b14773cca8dbd4107a`
- Initial A1 implementation: `1b7a54c58f7958cbb0db212f067199f16064c5ea`
- Reviewed source successor: `10de976214f9cfc36d50fc4bbee7ceba3683cafb`, tree `0e0108007ee1cb893739ab32c80f350cd8be0acc`
- Isolated branch: `codex/population-a1-controls-20261013`
- Source repair author: OpenAI/Codex invocation `repair_population_certificate_controls`, coordinated by dot
- Distinct nonauthor reviewer: OpenAI/Codex `review_population_control_repair`; source-exposed, same provider/team, organizational-independence credit 0
- Reviewer disposition: scoped A1 engineering PASS, no remaining must-fix within that review; see [exact review](review/SOURCE_REVIEW.md)
- Scientific effect: NONE; Dylan's personal reading is PENDING

[Claim](https://github.com/d6g8k5htny-coder/main/pull/343#issuecomment-6144767238), [activation](https://github.com/d6g8k5htny-coder/main/pull/343#issuecomment-6144776429), and [additive evidence scope](https://github.com/d6g8k5htny-coder/main/pull/343#issuecomment-6145038160) remain the coordination records. The enclosing commit of this directory identifies its custody publication; it does not change the reviewed source identities.

## Repair and preserved attribution

Only `population_certificate.py` and `test_population_certificate.py` change relative to upstream before adding this evidence directory. The production delta is confined to `rectangle_mass_control`, `cdf_mass`, and `negative_controls`.

The CDF now evaluates its existing primitive at the support endpoint's exact root `s=1`, using the same normalization as the interior route. Above-support input is clamped to `9/4`. The expected rectangle mass `1/5` remains an independent analytic oracle from the existing contract/proof. The missing-sign control halves the computed route mass; conditioning multiplies it by 5, explicitly dividing by the contract's finite-bar mass. Their expected masses are checked rather than supplied as outputs. The rectangle predicate accepts an enclosure containing its oracle. Function names, signatures, control keys, and the complete unmutated control payload remain unchanged.

The original S v2 patch is adopted byte-for-byte, with its authorship preserved: Anthropic Claude session `session_01S16677s1SGeAJmN3QMz4ip`, recovery commit `0f38a593c5f54cacdde7b841aae032fc14f1fb60`, patch SHA256 `f49e503741a315250c250e2cc0beabf5cfed110e4f3f7e7b716de9d79836258d`. Its 255 added lines are not attributed to this repair author. Seven new A1 tests and the later strengthening of one of them are the A1 author's additions.

Final source SHA256s:

- `population_certificate.py`: `be1d135b838733e4549b2519bae30d02377c97561a8876a16d9b7ea198305edf`
- `test_population_certificate.py`: `3d653a0646a8121b44982a65c14d8a2a82c61343522e4b4ab76fd64166810d4e`
- Initial A1 test file, superseded by the row-level regression strengthening: `2835315e6866b21f2d8479bfd464cbc2ef9b65fab10fe9eede1c3c8cdb21ccf4`

[Candidate identity](candidate_identity.json) binds the ten protected proof/contract/runner/dependency/result files to unchanged upstream bytes. The historical exact-population RUN still binds its original test source; it is not a receipt for this successor.

## Observed verification

Author interpreter: CPython 3.12.14. Actual interpreter and source identities, commands, exits and durations are preserved in [candidate identity](candidate_identity.json) and [verification results](verification_results.json).

- Pristine population baseline: 23 tests PASS; exact S v2 adoption: 42 tests PASS
- Failing-first A1: six of seven added tests fail by intended assertions on unchanged production; the seventh checks preserved valid behavior. [RED](a1_red.log), [GREEN](a1_green.log)
- Initial source-stable A1 candidate: population 49, universality 71, repository 519 tests PASS in both normal and optimized modes. The 519-test suite has six explicit skips below. The before/after 823-file source manifests match exactly
- Existing workflow exploratory pilot: PASS; retained-output verification: PASS; committed published-artifact verification: PASS. These are existing finite-grid software controls, not the full exact-population campaign
- After the review-driven test-only refinement: affected population 49 tests PASS in both modes, [normal](refined/population_normal.log) and [optimized](refined/population_optimized.log). Production and other test/dependency slices are unchanged; the earlier unaffected aggregate results are retained under their actual source cut
- Eleven targeted A1 mutants are killed in both modes on the successor, with seven tests per subprocess, assertion failures, no errors or skips. [Successor mutation results](refined/a1_mutation_results.json) retain exact transformations and mutant hashes
- The twelfth distinct fault deletes the runner's per-row soundness guard. The initial zero-budget regression incorrectly passes that mutant; the strengthened otherwise-certified synthetic-row regression fails it in both modes and passes the correct runner. The reviewer independently repeated this bounded check. [Reviewer normal receipt](review/delta-normal.json), [optimized receipt](review/delta-optimized.json)
- Author comparison of 14 CDF inputs and complete unmutated control payload: exact equality to upstream. The distinct source review records its additional 141-input comparison

The reviewer caught the initial row-level coverage gap before publication. It is preserved, not erased: [old regression survives](row_sound_gate_old_test.log), [successor rejects mutant](row_sound_gate_new_test_red.log), [optimized rejection](row_sound_gate_new_test_red_O.log), [exact test refinement](row_sound_gate_test_refinement.patch).

The six repository skips are five `test_architecture_sandbox.ArchitectureSandbox` tests requiring a hosted restricted container (`test_kernel_enforces_cpu_memory_and_process_limits`, `test_network_namespace_has_no_external_interface_or_route`, `test_nonroot_and_no_new_privileges`, `test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space`, `test_source_mount_is_read_only_and_environment_has_no_publishing_secret`) and `test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt`, because the Lean toolchain is not installed. Skips are not passes for those capabilities.

## Explicit remaining boundaries

- S retains the separate C270-A4/v3 tail-upper regression task. Its unresolved disposition is not closed by this A1 review, nor is this a replay of S's 22/23-mutant campaign
- The new full eight-row exact-population campaign is NOT_RUN in this delivery. No fresh RUN/CERTIFICATE/RESULTS is asserted. Its separate source-bound execution and required review remain a later step
- Hosted checks for this isolated amendment, branch adoption, readiness and integration are NOT_RUN. No original branch was rewritten or merged by this repair
- Original proof, frozen contract, numerical budgets, thresholds, downstream pins and saved exact-population results are unchanged
- No new mathematics, independent whole-field selector, Gaussian universality, Lean evidence, human review, organizational independence or scientific promotion is claimed

## Custody and replay

`MANIFEST.json` is the canonical payload inventory. It binds every exported evidence file to its original byte count/SHA256 and public-export byte count/SHA256. Cloud execution locations are normalized as documented there; results, endpoints, assertions, flags and exit codes are not edited. Raw originals remain retained locally. The exact review and its three replay artifacts contain no such paths and are preserved unchanged.

The public set includes complete relevant test/fault logs, intentional mutant source files, author drivers, original RED/GREEN evidence, source manifests, exploratory outputs and exact patches. Mutant `.py` files are deliberately incorrect test fixtures, not proposed production modules. Duplicate intermediate patch cuts are omitted because their commits remain in this branch; the final three-commit mbox, combined patch, A1-only patch and exact refinement are retained.

From the repository root, the affected suite is:

```
python3 -B -E -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p test_population_certificate.py -v
python3 -B -E -O -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p test_population_certificate.py -v
```

The read-only reviewer probe takes the checkout path and verifies the exact three consumed file hashes before running its correct/mutated one-test comparison:

```
python3 -B -E -S reviews/population_a1_controls_20261013/review/check_delta.py .
python3 -B -E -O -S reviews/population_a1_controls_20261013/review/check_delta.py .
```

To replay an author driver, copy that driver into a new empty output directory and invoke it from this repository root. Its public form locates source with `Path.cwd()` and writes outputs beside its copied script. Do not execute output-writing drivers in this retained evidence directory or overwrite historical logs. The reviewer probe above writes only its JSON receipt to stdout.
