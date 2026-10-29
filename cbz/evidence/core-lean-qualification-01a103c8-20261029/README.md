# Pinned Core Lean controls and Main formal package qualification

Owner-delegated engineering evidence, OpenAI/Codex chat 01a103c8. Scientific status authority: false. Layer A is unchanged. This packet adds evidence only; it is not an empirical-harness implementation, new research theorem, canonical governance edit, or final CBZ-0 admission.

## Exact runtime and sources

The official Lean 4.34.1 Darwin arm64 release archive is 797,037,816 bytes, SHA-256 `9019fcd34e93fddf0ecf9d40d28f05a86ba563ac3b701a9c5116ce1c96379360`. The observed compiler identifies commit `5045d0056413266e57c625dcd7c365b10e377c52`, and its executable SHA-256 is `1b370cfcbf44e80d1b004ab1b1ab9a4c73951f9f7c242140bcff9bc577576554`. Archive and compiler remain outside the repository in the local offline-capable runtime directory. No global toolchain configuration changed.

Primary release: https://github.com/leanprover/lean4/releases/tag/v4.34.1 . Core Int semantics: https://lean-lang.org/doc/reference/latest/Basic-Types/Integers/ . Actual pinned compilation below controls the observed version-specific arithmetic claims.

## Thirteen finite semantic controls

`core/CoreArithmeticControls.lean.txt` is the exact 1,725-byte compiled source (SHA-256 `123b4c0db32f3b0ac9b1faf5e6392143e431fd9c2ddcda55cf37966fb5aa5fb7`), saved with a `.txt` suffix so it does not enter research Lean discovery. Both original execution and independent rerun exited zero, emitted exactly thirteen named `does not depend on any axioms` reports, and produced byte-identical stdout. Neither used `sorry`, `axiom` or `native_decide` in these controls; proofs use `decide`.

Controls distinguish default Core Int Euclidean division/remainder from Python floor division for negative divisors; include totalized zero division and remainder; distinguish Nat truncating subtraction from Int subtraction; and retain truth-preserving inequality mutation as TRUE alongside the known FALSE inequality. They do not grant semantics to unsupported overloaded instances or user notation. In particular, 58*9=522 is below both 529 and the nonstrict bound 529, while the corresponding strict 484 inequality is false.

Reproduce after making a private working directory, copying `CoreArithmeticControls.lean.txt` to `CoreArithmeticControls.lean` in it, and setting CBZ_LEAN to the pinned compiler's absolute path:

```sh
cd "$CBZ_CONTROL_WORK"
"$CBZ_LEAN" -DwarningAsError=true -o CoreArithmeticControls.olean CoreArithmeticControls.lean
```

The working directory is a required precondition. The original raw result recorded argv but omitted cwd. The independent reviewer first replayed from a default cwd and encountered Lean's input-root containment failure before arithmetic checking. That failed precondition attempt is retained under `core/independent-root-precondition-attempt/`; it does not count as mathematical rejection. The successful independent result explicitly records cwd at the original source directory, exact argv, before/after metadata and digests, and raw output. Original artifacts were not rewritten. Compiled `.olean` binaries remain local; their digests are recorded. The independent reviewer also compared all 17,711 installed regular runtime files against the rehashed official archive, with zero missing files or mismatches. The 3,650,729-byte detailed comparison remains in local evidence (SHA-256 `a9fdc1c7fc2a4f13465baf60f7bcc88275274236a69ea985202023a9cb518821`); it is not included in this portable packet. The first failed precondition attempt retained no exact compiler exit code: only its wrapper exit 1 is known, as explained by the independent review.

These are fixed auditor golden controls, not a randomized campaign, a candidate empirical-fuzzer result, or proof closure for a project bound.

## Main formal package actual execution

The checked source commit is `9a5d313bc3a2fc880044498b8e48288d53437ad4`, parented on PR712 `15e8f689d00ca03e28e9bc5d97b6aba3f19ec900`. Its formal source is inherited unchanged from Main `e6f2653841053004d213b3647a7e14e799f503b2`. The receipt belongs to checked commit 9a5d313b; it is not relabeled as a run on a different commit or hosted workflow. No source hash refresh was used.

At 2026-10-29 23:50:33–23:50:45 UTC, with the exact runtime bin directory prepended to the process PATH, this command ran from the isolated repository root:

```sh
/usr/bin/python3 -B tools/formal_gate_check.py --run-lean
```

Observed exit 0; fresh `lake build`, `leanchecker`, transitive axiom audit, compiled inventory and elaborated-type capture all succeeded. All 31 registered public theorem targets have empty axiom sets. Compiled inventory reports 45 declarations. All ten negative controls were rejected by the intended Lean diagnostic or axiom gate: seven false mathematical statements by Lean, plus sorry/custom-imported/native controls by the axiom gate. Each negative transcript was checked with the gate's exact output, exit and empty-stderr contract. All 25 bound source/control/alignment files retained their before/after hashes; tracked Git remained clean.

`formal/receipt.json` is the unchanged original receipt (SHA-256 `1487e7b3f0a0348fe8ccc13a7e2db63cb99bbf13a0e65e4b2d2e46d934ca6794`). Every original receipt log was independently rehashed before copying and is included byte-for-byte under `formal/logs/`. Native execution has null hosted repository/run/attempt identity. No trusted hosted receipt or independent informal/formal alignment acceptance is inferred.

This bounded package result does not validate the new drift crawler, the empirical candidate, mathematical generalizations absent from the registered targets, cross-repository semantic equivalence, the full G0–G11 matrix, CLOSED admission, or a final simultaneous CBZ-0 run. PR712 graph/pin/registry/freshness failures from the separate qualification remain unresolved at 15e8f689.

## Custody and integration

`FILE_MANIFEST_V1.json` gives bytes and SHA-256 for every other file in this packet. Raw outputs and records are retained, including the independent failed precondition attempt. Canonical B-003 integration should reference this evidence at its own exact source applicability; it should not promote partial coverage or install a stale success. Production tooling and shared canonical files are untouched by this contribution.
