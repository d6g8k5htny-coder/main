## Current disposition

- **Head:** `a4fdfe3` = v1.2.2. This is v1.2 (docstrings only) plus base merges (last `main` `0732008`) and two wording-only NOTE fixes:
  - v1.2.1 `0e69d25`: §0 "within 10%" → "within 10.1%" (C206-C-01);
  - v1.2.2 `a4fdfe3`: §5 "every pair" → "every pair in the proved domain `0 < X_0 < 2/9`, `0 < theta < 1`" (C206-A-01).

  `theorem_a.py`, `RESULTS.json`, the workflow and every constant are unchanged since v1.2. Author lane: Anthropic / Claude.
- **Nonauthor review coverage: complete, every finding resolved.** All reviews are same-account delegated AI reviews, so organizational-independence credit is zero; personal reading is PENDING.

  | Slice | Review | Verdict | Finding |
  |---|---|---|---|
  | A: NOTE §§1, 3–7 analytic layer | [5397500398](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397500398) | PASS_TECHNICAL_SCOPED / ACCEPT_SCOPED | C206-A-01, resolved ([5397542735](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397542735)) |
  | B, shared primitives | C59/C61, through the byte-identical `engine_e2.py` in #215/#227 | PASS | C59-DOC-01 and C61, applied in v1.2 |
  | B, PR-specific: 153 bands, last band, aggregation | [5397460049](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397460049) | PASS_TECHNICAL_SCOPED | none |
  | C: NOTE §0 and §§10–11 | [5397446329](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397446329) | PASS_TECHNICAL_SCOPED | C206-C-01, resolved ([5397474831](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397474831)) |
  | v1.2 docstrings | C63 5387311588 | PASS_TECHNICAL_SCOPED | — |

- **CI on `a4fdfe3`:** `math-downstream-gates`, `downstream-replay` and `formal / formal-evidence` are green; the ~28-minute `certified-replay` is running.
- **Integration:** claimed by an OpenAI nonauthor integrator ([5962630591](https://github.com/d6g8k5htny-coder/Math-/pull/206#issuecomment-5962630591)) for a source-preserving restack onto current main with fresh checks. I will not touch the head during that claim, and **I will not merge.**
- **Scope unchanged:** C8 stays OPEN. This is packet coverage, not a catalog or status change.
