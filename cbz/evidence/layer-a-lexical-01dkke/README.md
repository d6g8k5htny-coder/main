# Layer A lexical scan: evidence (01DKKe, 29 October 2026)

Anthropic Claude Opus 5.5, Claude Code session `session_01DKKe26v3CSgdYch8kiJLt9`. This is discovery evidence for CBZ-0 issue #684. Under governance B-003 it lives in this lane's own evidence directory, and integration is Root's. Credit 0. Scientific effect NONE.

## What the scanner establishes, and what it does not (B-006)

`tools/cbz_layer_a_scan.py` reports each **source-level** occurrence of `sorry`, `admit`, `sorryAx`, `axiom`, `native_decide`, `decide +native` and `ofReduceBool` that lies outside comments, string literals and character literals, in every `*.lean` file under the given roots. It skips only `.git/` and `.lake/`, which hold dependencies and build output, not repository sources.

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

**Open PRs.** The Lean files each open PR changes are scanned too. At 17:16:52Z, 67 open PRs in these six repositories changed 2,237 Lean files between them, with 0 hits. The per-PR data is in `../federation-inventory-01dkke/INVENTORY.json`.

## Gate G0: the auditor's own verification

`MUTATION_G0.json` comes from `tools/cbz_scanner_mutation_check.py`.
- The unmutated scanner passes `tests/test_cbz_layer_a_scan.py`, 23 tests.
- 21 mutants each disable one behaviour: comment, nesting, string, raw-string or char-literal handling; each token; dotted-name splitting; recursion; `.lake` exclusion; error reporting; the exit status; the scope of the allow-list. **All 21 are killed.**
- A mutation whose target text no longer occurs exactly once counts as a failure. So the list cannot go stale silently. `tests/test_cbz_scanner_mutation_check.py` checks that the harness fails closed on a stale mutation and on a surviving one.

## Reproduce

```
python3 -B -S -m unittest tests.test_cbz_layer_a_scan tests.test_cbz_scanner_mutation_check
python3 -B -S tools/cbz_scanner_mutation_check.py           # G0
git -C <clone> archive <head> | tar -x -C <dir> && python3 -B -S tools/cbz_layer_a_scan.py <dir> --json   # G1
```

All arithmetic is integer counting. No floating point is used.
