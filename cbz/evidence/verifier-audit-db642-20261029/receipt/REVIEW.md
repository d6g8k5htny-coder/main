# Receipt-attempt engineering review

Reviewer: OpenAI / Codex task reviewer `/root/receipt_review`, 2026-10-29. This is code/spec review under the explicit owner CBZ task, not scientific alignment or an organizational-independence claim.

Base: `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`. Reviewed the supplied staged `review.diff`, design/spec, plan, implementer report and focused verification/mutation records. No repository changes, git mutations, remote writes, or test reruns were performed. This report is the only file written.

## Verdict

**ACCEPT the implementation on code/spec review; final verification and hosted integration remain pending.** No critical, important, or minor actionable findings were identified in the reviewed change. This is not a claim that final full discovery, Lean execution, or hosted required checks have passed.

## Findings by severity

- Critical: none.
- Important: none.
- Minor: none.

## Basis

- `tools/required_formal_check.py:39` strictly parses the current hosted attempt as a positive decimal string. At `:60`, the receipt must equal that exact validated string. Missing, null, numeric, boolean, zero, padded, whitespace-bearing and stale attempts cannot match. `:99` and `:106` retain only the existing live CLI operations and strict binder; no historical mode or fallback was added.
- `tools/formal_gate_check.py:304` records the actual environment attempt without an invented default. `tests/test_formal_gate.py:444` checks attempt 7 in both the returned receipt and serialized JSON; `:455` checks that absent hosted identity is explicitly null. These are serialization/control tests with simulated runtime behavior, not evidence of a real kernel run.
- `tools/atlas_formal_evidence_build.py:47` confines historical acceptance to all three fixed archive, manifest and receipt SHA-256 identities, then checks actual log bytes. It takes no caller-selected run identity or environment. The returned binding is fixed to the original commit/run/attempt and compared with the archived binding at `:105`. Removing repeated schema checks after an exact receipt digest is safe here because only the immutable known receipt can reach the parser. The live CLI does not import or call this route.
- `tests/test_atlas_formal_evidence_build.py:82`, `:94`, `:102`, `:112` and `:128` cover preservation of original bytes/binding, strict live refusal of the real legacy receipt, altered legacy identities and inserted attempt refusal, changed manifest/archive/log refusal, and exact reproduction of the published evidence.
- `tests/test_atlas_formal_source_bridge.py:37` copies literal frozen control bytes into historical test roots while running the current validator. I directly compared both new literal fixtures with their base git objects: they are byte-identical. I also directly verified that both changed live control files match the new manifest digests. The manifest diff changes only those two existing hashes; mathematical files and non-file metadata are unchanged.
- I directly compared the historical receipt, execution archive and published evidence module with the base git objects: all are unchanged. These checks corroborate `identity-verification.json`; no historical receipt rewrite or attempt insertion is present.
- `tests/test_required_formal_check.py:102` adds boundary tests for stale, absent and malformed receipt attempts while retaining the positive non-default attempt 2 and existing assertions. I compared the isolated mutation binder with current source: its sole difference removes the new attempt comparison. `mutation.log` shows all three regression methods killed that mutation with 11 assertion failures and no errors. The failure is specifically that the unsafe binder no longer raises, rather than an unrelated setup/import failure.
- The additional positive-fixture change in `tests/test_required_formal_binding_control.py:36` supplies attempt 2 already required by its environment; it preserves the existing assertions and does not change production logic.

## Verification limits

The recorded normal and optimized focused suites pass, including the repaired seven-test binding-control suite. The recorded source-only gate passes at manifest SHA-256 `49360aa6588167f2e4cd49e90ba0e8a6dfce6c47a11f1f09ce64def51e0e8590`. These results were inspected rather than rerun during review.

The initial full discovery reported 1551 tests, one error, and 24 skips; the error was the now-corrected positive receipt fixture. The final complete rerun was pending when this review was requested and is not certified by this report. Real Lean execution was skipped/unavailable in this environment; no local kernel PASS is claimed. Before declaring integration verified, the controller must record the completed final discovery result and observe the required hosted checks at the actual integration source.
