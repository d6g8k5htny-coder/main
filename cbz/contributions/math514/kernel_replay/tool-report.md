# Exact-module kernel replay tooling report

Complete scoped tool at `/workspace/cbz-0/tools/kernel_replay/`. It imports the installed Lean 4.34.1 `LeanChecker.replayFromImports` and runs one exact module per fresh direct `lean -j1` process, with explicit virtual-memory and wall-time bounds. No target pre-import, prefix expansion, ignored flags or concurrent module scheduling exists. All available exported/server/private data are handled by the official routine.

G0: **27 applicable tests pass**: 25 after the last production change plus 2 newly added tests; unchanged completed verification is reused under `/workspace/AGENTS.md`. `evidence/g0-applicable-tests.json` records that composition. **16/16 mutations killed**, threshold100%, zero survivors/errors/skips in the final matrix. The initial11/15 run with4survivors is preserved, including the exact test/fixture repairs it motivated.

Real fixtures: public/private/data/inductive Good PASS with3oleanparts; public and genuinely private self-referential malformed theorem bodies REJECTED with unknown-constant kernel errors. The exported-only and target-preloading mutants are killed. An unapproved axiom correctly replays, while a separate explicit fixture axiom-policy check rejects it. **Project G2 remains independent and unassessed.**

Independent review identified a main-olean symlink mismatch. Before the fix, a real malformed private-body fixture falsely passed because bound destination siblings differed from lexical siblings selected upstream. The original receipt is retained as `evidence/initial-real-symlink-replay.json` and explicitly INVALIDATED as G4 evidence. The fixed wrapper rejects symlink/noncanonical components in roots, selected modules and sibling parts; focused wrapper/real regressions and a killed mutant cover the fix. A project-local checker shadow regression is also fixed by installed-sysroot precedence.

Resource facts: initial2GiB virtual-address cap failed before replay (retained). Tiny direct Lean diagnostic reserved6222420KiB virtual with613748KiB RSS; actual fixtures passed with explicit8192MiB virtual cap. This does not show every project module fits. Root685's `leanchecker ResearchFormalCoreR1` exit137 OOM at16GiB remains non-green. Parent's unsupported upstream `--help` diagnostic (~3GiB then own-PID SIGTERM) remains a diagnostic failure, not G4 evidence. No project Math core replay was performed here.

Usage and exact commands: `README.md`; full specification: `SPEC.md`; registry/ledger/quarantine payload: `REGISTRY.json`; absolute source-to-`cbz/contributions/math514/kernel_replay/*` mapping: `PUBLICATION.json`. Mapping excludes temporary directories, caches and generated fixture binaries. The tool does not prove G3 source-to-bytecode provenance, undeclared dependency closure, caller inventory completeness, G2 compliance or full CBZ.

Production bindings:

- `kernel_replay.py`: `416b688929daff9f473ce3e0abd399d8be2cde46c12956d3e5d5d9937d914293`
- `ExactReplay.lean`: `24e55f5e99c83a46f9f908bbc90f22292a94305294e287d9b2eccbf24e8e6b66`

No mathematical source, Git, remote or canonical surface was edited by this task. Existing applicable M67 and target-build evidence was reused without a new proof replay. The canonical writer can now publish the mapped contribution and integrate its registry payload before separately diagnosing actual project G4.
