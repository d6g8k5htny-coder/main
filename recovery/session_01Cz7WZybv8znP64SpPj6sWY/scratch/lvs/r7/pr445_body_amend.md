## Frontier section for the 64 registered Props without a theorem (docs only)

**Scientific effect: NONE.** Documentation only: `README.md` and `SCOPE.md` of `specifications/lean_statements_20261010`. `REGISTER.json`, every `.lean` file, the gate `check.py`, the lock and the toolchain are untouched. No claim-level label moves; every one of the 64 Props stays `specified`; alignment stays `PENDING_INDEPENDENT_REVIEW`. Organizational-independence credit 0 (Anthropic / Claude, the packets' author). Coordination: main#307 pickup [6136169673](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136169673).

**What it adds.** At the base `8acc4856` (current Math- main) the sibling proof packet proves 77 of the 141 registered Props. One new section, inserted identically into both documents before their closing sections, classifies the other 64 so the remaining formal work is named where the register lives:

- **36 vocabulary predicates**: reusable `def … : Prop` with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; the unrestricted universal closure is false or empty, so a theorem of exactly their type is not a target (instantiated or conditional theorems about them are not excluded; counted among the 141 because the register inventories every Prop-valued definition).
- **24 interface-blocked**: the pinned interface's fields do not determine the conclusion; for each row the (mathematical, not kernel-checked) inhabitant that satisfies every field and violates the conclusion is named (non-kernel judgements of a single reader refuting the unrestricted universal closure for the given outer objects, without constructing them); the route is a numbered successor interface.
- **4 Mathlib-blocked**: Lifetime Theorems B, C, E11 and R1, whose truth value the interface fixes and for which no interface counterexample was identified; proofs are expected to need Gaussian-field and density formalisation beyond the primitives of Mathlib `d13f23b7` (an effort estimate conditional on the informal claims, not an impossibility theorem).

Under this grouping, a planning convention and not a proof that other meaningful targets are exhausted, 105 Props are targets, 77 are proved and 28 remain, blocked at the specification or library level in the qualified senses the section states. Each row cites the defining line of the statement module (last changed at `873c6c0`, before the round-4 triage of 2026-10-11 whose verdicts this section re-applies). One cross-reference clause is added next to each document's existing "77 of the 141 Props" sentence.

**Amended 2026-10-13 ~04:45Z per the classification review [6138311227](https://github.com/d6g8k5htny-coder/Math-/pull/445#issuecomment-6138311227) (445-A to 445-D, docs only): head `0c506fea31fd9984c8ca92db135971ca8f3fa420` (tree `4b68beff`, parent `9fb0a4a0`); `README.md` 102,392 B sha256 `ada6df8d2933b7fea57c2077073b401d448849ec15bc69b5e63de0465239f8b8`; `SCOPE.md` 281,267 B sha256 `2724e0dbb2e48a910c8ee6045f8c18a1911cf2c2dc30be05f49913a8ad573c38`; hosted runs on this head bound below when complete. The corrected text needs its own bounded source read before integration.**

**Identities of the reviewed predecessor head (`9fb0a4a06900da5cc3f9126f30e117a747df1587`, tree `5f1ceda12195d71d178fcad1937dc2b57faa74ea`, base `8acc48566de3ae613c1ae320903b66c1d5411797`):**

| file | bytes | blob | sha256 |
|---|---|---|---|
| `README.md` | 95031 | `756e3378` | `ff31f359e1ef514cd6dd2d6d45578b0c1d05b2984e3d759bc727dd1a01c19e3b` |
| `SCOPE.md` | 273906 | `66b6a30a` | `762c0475dd4a5a28b5bd34a93d29616614b4b2e445ad4ab5feeff38e2f003693` |

**Local checks on this head:** `python3 -B -S specifications/lean_statements_20261010/check.py` (source mode) exit 0, worktree CLEAN, 357 statements, 9 structures; `test_check.py` 97 tests OK. **Hosted runs on PR head `9fb0a4a0`, actual tested merge `6b6b53e817b98cd1ad2ddfb7df125125b17e96a3` (merge of `9fb0a4a0` into base `8acc4856`; `checked_commit` in both receipts):** `lean-specifications.yml` run 38333584611 (job 115088584979, success; artifact `lean-specifications-38333584611-1`, 42,022 B, sha256 `9c617f54f73f0684685c1f058039a68307b490eaf916f9345291631a856991fe`; register sha256 `bcafecbb…` unchanged) and `lean-specification-proofs.yml` run 38333584286 (job 115088583452, success; `theorem_count` 78, `leanchecker` PASS; artifact `lean-specification-proofs-38333584286-1`, 47,042 B, sha256 `0ecf5f4adece4a9016d3b86dc6451bed1064ce1aa13f063d132644c3244cf33a`); `downstream-replay`, `formal / formal-evidence` and `math-downstream-gates` also success: 5 of 5 check runs green on this head. These runs re-check the unchanged Lean packets; a green run on a docs-only change is a record of a computation, not evidence about the documents. Generator: a fail-closed exact-anchor script over the committed bytes (anchors asserted unique; the 64 names asserted equal to the register's def_props minus the proof packet's theorem types; counts 36 + 24 + 4 asserted), kept in the author's scratch and listed in the recovery checkpoint.

**What this does not establish.** Nothing about the truth of any statement or claim. The classification is the author's single reading (this session's parallel-verification capacity is exhausted until 2026-10-17 03:00Z); a reader may move a row between classes, and the "interface-blocked" inhabitants are described, not formalised. It does not say the informal sources are wrong: their hypotheses are exactly what the interfaces do not transcribe. Nothing here is a receipt; hosted runs remain the kernel evidence.

**Review and integration:** head frozen; a nonauthor reader and a distinct integrator are needed; the author does not mark ready or merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY

