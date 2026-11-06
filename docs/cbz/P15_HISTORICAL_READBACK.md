# P15 historical artifact readback

This isolated tool checks the downloaded artifact from historical run `38096827443` against independently recorded ZIP, receipt, manifest, checked-commit and run identities. It hashes every archive member and verifies every receipt-bound log, every manifest-bound source file, the complete 59-target receipt inventory and all 28 P15 axiom lists. It preserves the original ZIP and literal elaborated-type log. This is byte/identity consistency for a historical run; it executes no Lean and establishes no informal alignment or current-baseline admission.

Tool: `tools/p15_historical_artifact.py --archive PATH --manifest HISTORICAL/formal/manifest.json --source-root HISTORICAL --output NEW.json`. Its caller must check out main commit `3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c` cleanly and obtain original artifact `11686340530`. The workflow performs these steps.

Self-verification: `python3.11 -B -S tools/p15_historical_mutations.py --output NEW_DIRECTORY` runs 16 ZIP/source and harness controls, seven named intended-failure source mutants, and one mutation of the harness in both normal and optimized modes. A test setup failure, exception or unrelated error is not a mutant kill.

## Observed evidence, 29 October 2026

At candidate head `e37e1328cacc13c52798ebbd3d41c903ccd7576c`, [engineering review 5705411166](https://github.com/d6g8k5htny-coder/main/pull/711#pullrequestreview-5705411166) records both 16-test baselines passing, all seven source mutants and the harness mutant killed for their intended reasons in each mode, and an independent original-artifact readback PASS. These are the actual review executor's observations; this documentation repair did not repeat them.

The [hosted closeout](https://github.com/d6g8k5htny-coder/main/pull/711#issuecomment-6510315628) records [run 40959663683](https://github.com/d6g8k5htny-coder/main/actions/runs/40959663683), attempt 1, and [job 123748813092](https://github.com/d6g8k5htny-coder/main/actions/runs/40959663683/job/123748813092) completing successfully. The tested checkout was synthetic merge `7c185ad202c58358e3699e112ba80ce155fc387d`, joining base `1e2be44c69e08e47472041ac329e1b27f7dec793` with the candidate head above. The job checked the separate historical source `3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c`; it did not rerun that source's Lean build.

The hosted verifier reported `cbz.p15-historical-byte-readback.v1` PASS: 59 targets, 28 P15 declarations, 23 receipt-bound logs, 49 original archive members and 24 manifest/source files. Its retained source identities are:

- Original ZIP SHA-256: `7a72c7a3405976c15437e14c9d7201997378704c16003db7bce45444b592ee45`.
- Historical receipt SHA-256: `115533a9ba6a5e7dc264376f25821bbc09168c6bdf546424f99cfa37f09485c5`.
- Historical manifest SHA-256: `d039ef64a841ac382a9ee7fbea9576f8691322a5dc87cf01eab127084a4394e1`.

The new [readback artifact 12938170916](https://github.com/d6g8k5htny-coder/main/actions/runs/40959663683/artifacts/12938170916), named `cbz-p15-historical-readback-40959663683-1`, is 87,336 bytes with 26 files. The closeout records an independent downloaded-byte SHA-256 equal to the provider digest: `b2ed6cac354cf3e3627fdab1a16eab72b596a3fac26d6d3d50a52dc96c2bf526`. This new readback artifact and the original historical artifact are different objects.

The later [current-Main composition review](https://github.com/d6g8k5htny-coder/main/pull/711#pullrequestreview-5706457697) records a disjoint 50-path delta against Main `e6f2653841053004d213b3647a7e14e799f503b2`, reconstructed overlay tree `c0c675f05418867b1358213b0c4c5437ca2f6344`, fresh 16/16 controls in each mode, and intended kills for all seven source mutants plus the harness mutant in each mode. That overlay is a reviewed tree, not a hosted-run identity.

## Composition and scope

These observations remain bound to their recorded source and checkout. The historical hosted success is not a result for a later documentation repair, current-Main composition or protected-branch merge. Obtain applicable fresh hosted checks on the actual final composition and preserve its head/base/tested-checkout identities. Do not relabel the historical Lean receipt as a fresh Lean execution or infer theorem, real/logarithm claim, informal-alignment or CBZ-0 closure.

Maintenance owner: none; self-contained. Scope: CBZ criteria 3, 6, 7 and 9. Record adopted-tool usage and bounded verification in the canonical tooling section of GOVERNANCE.md (or TOOLING.md), with the actual final composition evidence.
