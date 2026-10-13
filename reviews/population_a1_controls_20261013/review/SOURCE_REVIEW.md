# Scoped A1 engineering source/delta review

Dylan Roy — delegated AI review. Actual performer: OpenAI/Codex AI agent `review_population_control_repair`, a distinct nonauthor engineering reviewer. Fresh source/delta review with bounded local execution; source- and route-exposed, same provider/team, zero organizational-independence credit. Dylan's personal reading is PENDING.

## Disposition and exact scope

PASS, with no unresolved must-fix in the A1 normalization/control repair reviewed here. The one test-coverage weakness found during review was corrected and independently mutation-verified in the successor below. This is not an A1–A5 closure declaration, a full scientific campaign, mathematical acceptance, hosted-check approval, publication approval, or integration verdict.

Frozen upstream: `5bf0bcd461b4b7de49052680dd6d4c6e8ee11e1f`.
S v2 adoption: `aca0f9e4dfe54dca3607f6b14773cca8dbd4107a`.
Initial A1 candidate: `1b7a54c58f7958cbb0db212f067199f16064c5ea`.
Reviewed successor: `10de976214f9cfc36d50fc4bbee7ceba3683cafb`.
Successor tree: `0e0108007ee1cb893739ab32c80f350cd8be0acc`.

Within `experiments/universality/two_parameter_cusp_population/`:

- `population_certificate.py`: SHA256 `be1d135b838733e4549b2519bae30d02377c97561a8876a16d9b7ea198305edf`
- `test_population_certificate.py`: SHA256 `3d653a0646a8121b44982a65c14d8a2a82c61343522e4b4ab76fd64166810d4e`
- Initial A1 test bytes, superseded only by the reviewed regression strengthening: SHA256 `2835315e6866b21f2d8479bfd464cbc2ef9b65fab10fe9eede1c3c8cdb21ccf4`

Only these two tracked paths differ from the frozen upstream. The final successor changes only the single A1 campaign regression, 11 added lines and one removed line. It does not change production, runner, contract, proof, budgets, thresholds, dependencies, or historical results.

## Source findings

1. At the support endpoint, the existing split identity has exact root s=1. Evaluating the existing primitive gives first term 3 and second term 0; the actual route normalization produces 3/15. The irrational elementary enclosures multiply exact zero terms, so preserving the exact Fraction endpoint is sound. Above-support tau is clamped to 9/4. Interior operation order and results are preserved.
2. The expected whole-rectangle 1/5 has an analytic source separate from the tested numerical route: PROOF.md equation (26), with the frozen rectangle parameters. The endpoint calculation therefore consumes the route's normalization instead of returning its oracle. Missing-sign and conditioned masses are computed from that route as one-half and five times the resulting mass. Their expected 1/10 and 1 values are comparisons, not supplied outputs. The tested missing-factor fault actually changes these values and fails their gates.
3. The rectangle predicate now checks oracle containment in an ordered Fraction probability interval. This intentionally replaces literal interval equality; it does not assume every valid enclosure is a point. Input validation remains in force. In the unmodified route the endpoint remains exactly (1/5, 1/5).
4. All function names and signatures are unchanged. AST comparison confines production changes to `rectangle_mass_control`, `cdf_mass`, and `negative_controls`. The control keys and full unmutated returned value are unchanged. Independent baseline/candidate comparison of 141 CDF inputs, including all eight schedule values, zero, the endpoint, above-support values, a near-endpoint value and additional interior rationals, returned exactly equal Fraction intervals.
5. These controls specifically address route normalization and rejection of the declared mass faults. They are not an independent whole-field selector, numerical second implementation, or exhaustive test of interior terms that vanish at saturation.

## Verification and resolved coverage observation

Reviewer execution on initial candidate 1b7a54c5:

- Population suite: 49 tests, no failures/errors, normal and optimized Python modes
- Exact frozen upstream production with the initial seven A1 tests: six assertion failures, zero errors, in both modes; the literal-control defect is reproducible
- S v2 patch applied to upstream bytes in an isolated temporary directory exactly reproduces adoption commit aca0f9e4's test file, SHA256 `c43a929d86fda0e6ea0f75067a80ab304887ab1226a21f17e76b5e2dfe95ae0f`

The initial campaign regression used zero direct-evaluation budget. Removing the runner's per-row `elif not sound` guard still passed that test. An otherwise certified synthetic route fixture exposed the difference: the correct runner left all coefficient tests NOT_RUN, while the mutant reported eight coefficient PASS rows despite failed controls. Overall failure alone did not test that row-level guard.

The successor captures actual failed controls under the route-normalization fault, then supplies them to the existing explicitly synthetic accepting route fixture. It asserts eight CERTIFIED direct results, eight FAIL_IMPLEMENTATION rows and eight NOT_RUN coefficient dispositions. Reviewer execution of this exact regression in both modes now passes unmodified and fails the guard-removal mutant with one expected assertion failure and zero errors. This resolves the observation without production changes or a full campaign.

Author evidence was inspected, not recast as reviewer reruns: 49 population, 71 universality and 519 repository tests in both modes on initial candidate, with six documented repository skips; affected 49-test suites rerun on the successor. The retained successor mutation record contains 11 distinct A1 mutants in both modes, all seven tests executed, assertion failures present, no errors or skips, and the saved mutant bytes match their recorded hashes. This is separate from S's other mutation campaigns and pending A4 work.

## Historical evidence preservation

All 823 author-manifest files matched the recorded initial candidate through that review phase. The only later source change is the identified A1 regression refinement. Original exact_population_1 files were compared directly with frozen upstream bytes and remain unchanged:

- CERTIFICATE.json: 71,881 bytes, SHA256 `8df03e0bfc338f27796c25816e319ee2304222d325540fe3372c2308a2d1a015`
- RESULTS.md: 6,290 bytes, SHA256 `8fe63fb42cd05fc7438e54e235b46e7fc74bb42b2ca8d5007d0011e62808c4c9`
- RUN.json: 5,129 bytes, SHA256 `ee7014fc73ed2b86558fb1ad6cbece484788fdfae5748a9d4ed75dc43acd50cd`

No source was edited by this reviewer. No full eight-row numerical campaign, public review, branch push, merge, S A4 repair, or scientific-status transition was performed.
