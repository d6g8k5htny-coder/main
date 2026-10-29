# Pinned Lean corroboration of PR712 C1 and C3

This is bounded engineering evidence for the previously reported lexical counterexamples. No production tool, research source, worktree source, canonical record, external payload, or project status was changed.

## Source and runtime

The checkout was `00e5c50d070c246fda4560d7420ad34244201487` before and after execution. Its `tools/cbz_drift/{lean_decls.py,crawl.py}` are inherited from PR712 `15e8f689d00ca03e28e9bc5d97b6aba3f19ec900`; both retained byte hashes. The loaded lexical parser module hash is in the result.

All four files were compiled with Lean 4.34.1, arm64-apple-darwin24.6.0, commit `5045d0056413266e57c625dcd7c365b10e377c52`, Release. Executable SHA-256: `1b370cfcbf44e80d1b004ab1b1ab9a4c73951f9f7c242140bcff9bc577576554`.

## Actual observations

At 2026-10-29 23:56:55–23:56:57 UTC, all four complete source files compiled with `-DwarningAsError=true`, exit 0, empty stderr, an output `.olean`, and a `does not depend on any axioms` report. There were no failed attempts in this bounded run.

| Pair | Actual pinned Lean output | Actual lexical comparison |
|---|---|---|
| C1Nat / C1Int | `KernelDrift.t : ∀ (x : Nat), @Eq Nat x x` versus `KernelDrift.t : ∀ (x : Int), @Eq Int x x` | Both return exactly one declaration `KernelDrift.t`, interface `: x = x` |
| C3One / C3Two | `#reduce KernelDrift.f` returns `1` versus `2` | Both return exactly one declaration `KernelDrift.f`, interface `: Nat := by` |

C1 changes only the in-scope variable's type. The header looks identical, but the elaborated quantified statement differs. C3 changes only the numeral following an indented `set_option maxRecDepth 100 in` within the definition's tactic block. Both definitions are valid, have the same type, and reduce to different natural values; truncating their value at the command token misses this difference. No extra differing theorem names were appended to manufacture the result. Checks/reduction/axiom commands introduce no additional literal declaration in these inputs.

## Reproduce

Use a private source directory as the working directory, with the exact published bytes saved under the original `.lean` names, and point CBZ_LEAN at the pinned compiler. For each of C1Nat, C1Int, C3One and C3Two:

```sh
cd "$CBZ_CONTROL_WORK"
"$CBZ_LEAN" -DwarningAsError=true -o C1Nat.olean C1Nat.lean
```

Use the corresponding basename for the other three files. To reproduce the lexical side, load `tools/cbz_drift/lean_decls.py` from exact PR712 head 15e8f689 (or the byte-identical inherited file in 00e5c50d), and invoke `declarations()` on each entire source text. Exact cwd, absolute argv, runtime version, source/output/olean digests and full parser return values are recorded in `ACTUAL_RESULT_V1.json`. If publishing these fixtures, retain `.lean.txt` suffixes in evidence and instruct the reader to save identical bytes as `.lean` before compiling; keep compiled binaries local.

## Limits

These executions corroborate two existing specific lexical misses. They do not establish a complete Lean semantic extractor, AST coverage, whole-federation drift qualification, source-to-numerical translation, empirical fuzzing coverage, merge/admission enforcement, project theorem CLOSED status or any final CBZ-0/G0–G11 result. The C1 source-context omission is a gap relative to elaborated-statement synchronization; the tool's documented non-elaboration boundary must remain explicit. C3 additionally contradicts its intended inclusion of definition values. No broad G7 result is inferred from these controls.
