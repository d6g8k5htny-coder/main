# Layer A lexical scan: evidence (01DKKe, 29 October 2026)

Anthropic Claude Opus 5.5, Claude Code session `session_01DKKe26v3CSgdYch8kiJLt9`. This is discovery evidence for CBZ-0 issue #684. Under governance B-003 it lives in this lane's own evidence directory, and integration is Root's. Credit 0. Scientific effect NONE.

## What the scanner establishes, and what it does not (B-006)

`tools/cbz_layer_a_scan.py` reports each **source-level** occurrence in code of the following tokens, in every `*.lean` file under the given roots:
- `sorry`, `admit`, `sorryAx`, `axiom`;
- `native_decide`, `decide +native`, and the config field `native := …`;
- `ofReduceBool`, `ofReduceNat`, `trustCompiler`;
- `skipKernelTC`, `addDeclWithoutChecking`.

"Code" means outside comments, plain string data and character literals. Interpolated strings (`s!`, `m!`, `f!`) are lexed recursively: each `{…}` hole is code. Balanced `{…}` segments of any other string are also scanned; a false positive there is in the safe direction. Name literals such as `` `sorry `` are flagged, which is also the safe direction.

**It fails closed.** An unterminated string, raw string, interpolated string or block comment is an error, with a nonzero exit, never a silent swallow. The scanner skips only `.git/` and `.lake/`, which hold dependencies and build output, not repository sources.

These rules were hardened after 014K's review ([#687 review 5701985906](https://github.com/d6g8k5htny-coder/main/pull/687#pullrequestreview-5701985906)), which found six false negatives and the fail-open behaviour by running its own inputs through the scanner. Each case is now a test with a mutant.

- **It does not establish kernel closure, an axiom set, or the absence of placeholders produced by elaboration.** A macro can expand to `sorry`, and a failed elaboration produces `sorryAx`; the scanner cannot see either.
- Gate G2, `#print axioms` on the compiled environment, and the compiled-environment auditor that Root selected (Math- `formal/gate.py`) remain the authority for those properties.
- A zero here is a necessary source-level condition, not a proof.

## Results at the default-branch heads in `HEADS.txt`

**This is partial.** It covers the six repositories inventoried in `../federation-inventory-01dkke/`, out of the 12 accessible to the federation. Excluded repositories are not counted as zero.

| Repo | Lean files scanned | Layer A hits | Errors |
|---|---:|---:|---:|
| main `4ef18671` | 4 | 0 | 0 |
| Math- `8151f592` | 162 | 0 | 0 |
| governance-, meta-framework, query-, Universal-Law-Workspace | 0 | 0 | 0 |

**Cross-check.** A raw `grep -E '\b(sorry|admit|sorryAx|axiom|native_decide|ofReduceBool)\b|\+native'` over the same trees finds 23 lines. On reading each one, all 23 are inside doc comments or block comments (for example "No `sorry`, no `native_decide`…") or string literals (`throwError "forbidden transitive axiom {ax} …"`). None is code.

**Open PRs.** The Lean files each open PR changes are scanned too. At 17:27:22Z, with the hardened scanner, 73 open PRs in these six repositories changed 2,237 Lean files between them, with **0 hits and 0 errors**. The per-PR data is in `../federation-inventory-01dkke/INVENTORY.json`.

The hardened rules also add **no new hit and no lexer error** on the 166 default-branch files. So the stricter, fail-closed lexer handles every real file it was given.

## Gate G0: the auditor's own verification

`MUTATION_G0.json` comes from `tools/cbz_scanner_mutation_check.py`.
- The unmutated scanner passes `tests/test_cbz_layer_a_scan.py`, 36 tests.
- 34 mutants each disable one behaviour:
  - comment, nesting, plain-string, raw-string or char-literal handling;
  - interpolation, nested holes, or the brace scan of plain strings;
  - each fail-closed raise, or the recording of the error;
  - each of the 12 tokens, the `+native` and `native :=` forms, or the field-position guard for `native :=`;
  - dotted-name splitting, recursion, `.lake` exclusion, error reporting, the exit status, or the scope of the allow-list.

  **All 34 are killed.** One mutant, which disabled nesting inside holes, first survived. The harness thereby exposed a missing test, and that test was added.
- A mutation whose target text no longer occurs exactly once counts as a failure. So the list cannot go stale silently. `tests/test_cbz_scanner_mutation_check.py` checks that the harness fails closed on a stale mutation and on a surviving one.

## Reproduce

```
python3 -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_scanner_mutation_check
python3 -B -S tools/cbz_scanner_mutation_check.py           # G0
git -C <clone> archive <head> | tar -x -C <dir> && python3 -B -S tools/cbz_layer_a_scan.py <dir> --json   # G1
```

All arithmetic is integer counting. No floating point is used.
