# Numeral boundaries in Lean declaration metadata

The former metadata lexer treated a leading numeral and following letters as one identifier. The exact source `def n := 0theorem hidden (h : False) : True := False.elim h` therefore produced no declaration metadata; a hash-bound manifest with `vacuity={}` reported `STATIC_PASS`. The hazard scanner's canonical numeral boundary did not protect this separate metadata stream.

The metadata lexer now imports the unchanged canonical `_NUMERAL` spelling recognizer, separates numeral tokens before identifiers, and retains each numeral's full text and source location. It performs no numerical evaluation. Existing identifier suffix digits, primes, qualified names, quoted identifiers, strings and characters keep their behavior. Binary, octal, hexadecimal, decimal-fraction and exponent spellings are covered. Hexadecimal maximal munch is preserved: an `e` directly after a hexadecimal numeral can belong to that numeral rather than start `example`.

Evidence in `evidence/declaration-numeric/` is finite and source-bound:

- The original seven-control RED receipt records 34 assertion failures and no errors. The first post-fix seven-control run caught an incorrect test-column expectation; the retained source-line calculation corrects it from 22/28 to 20/26. The original RED and initial post-fix failure are retained.
- 52 affected methods (45 existing Lean integration methods plus seven new methods) passed in normal and optimized modes. Three new production-source guard mutations were killed by real body assertions in both modes, with zero errors.
- The first outer campaign driver compared ordered test identities to a sorted list and refused the already successful receipts. The corrected caller reused those completed runs. No 52-method replay was performed for that orchestration error; the original driver and correction are retained.
- Independent review found that the appended class followed the direct `unittest.main()` guard. The unchanged guard was moved to the final line. Class/function ASTs were unchanged, so the import/discovery receipts remain applicable. A new direct-entry seven-control run passed, resolving that previously untested registration path.
- All 72 prior auditor method ASTs, all 15 grader methods, all grader function/class ASTs and the 57 existing mutation definitions remain unchanged. The 42 unaffected predecessor methods retain applicable evidence. The current 94-method/60-definition inventory is a composition, not a fresh full G0 run.

The three new mutation cases restore generic numeral swallowing, discard numeral metadata tokens, or overshoot a numeral boundary. Earlier 57 mutation definitions and their historical source-bound 19/38 evidence remain preserved. `current-composed-admission.json` states the exact current and predecessor source hashes, scopes and limits. `registration-move-applicability.json` binds the final test registration change to the earlier completed campaign. This package does not rerun Lean, project mathematics, CI or an actual repository G1.

Run new metadata methods when a relevant change warrants verification, with a fresh output path:

```sh
python3 -B -S mutation_test.py --internal-test-runner --test-root tests --test test_auditor.NumericDeclarationRules --output /absolute/new-evidence/numeric-controls.json
python3 -O -B -S mutation_test.py --internal-test-runner --test-root tests --test test_auditor.NumericDeclarationRules --output /absolute/new-evidence/numeric-controls-optimized.json
```

The existing public full G0 command includes the seven new methods and three new mutants. Its receipt is required only when that full scope is newly warranted; it was not run for this repair. Recorded historical campaign drivers are evidence of those executions, not independent production certifiers. Canonical governance/tool registration remains the final controller's responsibility.

The final controller must update vacuity classifications only for source declarations newly exposed by these boundaries. A prior clean result for the exact adversarial layout is superseded. The tool still has its documented bounded syntax contract, not arbitrary Lean parser completeness, semantic nonvacuity, mathematical closure or inventory authentication.
