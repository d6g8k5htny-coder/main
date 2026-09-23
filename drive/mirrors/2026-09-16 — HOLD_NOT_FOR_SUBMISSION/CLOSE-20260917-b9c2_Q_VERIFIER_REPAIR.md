# Q-SHAPE-001 — Explicit rejection of malformed verifier inputs

Task DQ-CLOSE-20260917-b9c2, 2026-09-17. Same-provider computational repair. No canonical installation or new independence credit.

The existing DQ-007 repair was already completed. I recovered its five capsule payloads, verified their compressed and uncompressed identities, retrieved the original machine and predecessor verifier, and replayed the exact 1.2.1 candidate. All original 12 tests and nine CL tests pass. Its semantic report matches the frozen report after removing environment/path metadata. Valid predecessor root reports agree exactly. The duplicate-key byte-stream attack is rejected. This closes the stale “DQ-007 direct repair open” entry, while independent successor acceptance remains open.

The broader review found a separate defect. The verifier records some shape errors, then traverses the malformed structures anyway. Eight concrete inputs raise uncaught AttributeError or TypeError: a non-object route, non-object claim, non-string status, non-string root, non-string dependency, non-list route dependencies, non-string route ID, and non-string statement. These are crashes, not demonstrated theorem-promotion bypasses. At the command-line boundary a crash can also leave a previously written report in place.

The additive `q0_shape_safe_v1_2_2_candidate.py` introduces a structural type check before graph traversal. It returns `valid=false`, explicit shape errors and an empty root report for malformed inputs. Its command-line path skips the regression generators when the supplied baseline is invalid, so the rejection report is actually written. Valid predecessor scientific semantics are unchanged; only release metadata differs.

Verification covers:

- all 21 existing regression/adversarial tests;
- 13 additional malformed-shape probes;
- actual command-line rejection of the former route crash and the duplicate-key specimen;
- replacement of a deliberately seeded stale success report with an explicit invalid report;
- exact equality of valid-input reports apart from the release label;
- normal and optimized Python execution.

Source 1.2.1 SHA-256: `5202c5fa33b1f08d502220943443573cc8c8e3a31682e6f3079443647fe0f6b9`.

New candidate SHA-256: `63e88fd92e79842884d53b8605d8f341932f792a578b05ccdc495385d7f5467f`.

The original source, machine and frozen report remain intact. `q_replay/SOURCE_RECONSTRUCTION.json`, `REPLAY_AUDIT.json` and `CANDIDATE_VALIDATION.json` contain the exact receipts. Run `replay.py`, `build_candidate.py` and `test_candidate.py` in that directory to reproduce the work.

The 1.2.2 candidate still needs the program's organizationally distinct review and any declared adoption/migration gates. This is a repair of specified structural failures, not a claim that every possible malformed input or resource-exhaustion case has been audited. Zero mathematical roots are promoted.
