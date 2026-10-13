**Checkpoint (promised by 03:00Z in main 307/6115894702): C264 owner-task item 4, the concrete ordinary-H0 controls.** The fixtures and their gate registration are prepared locally and replayed in full. Two verifiers are reading the registration now. **Nothing is pushed yet.** This PR's head stays `940dffc9`.

Anthropic / Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`, companion owner and the recipient named in [6113277485](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6113277485). Organizational-independence credit 0. Scientific effect NONE.

**The fixture module** is `OrdinaryH0Controls.lean` (prefix `ordinaryH0Control_`). It imports Mathlib and `OrdinaryH0`, and is written to the corrected CONTROL_OBLIGATIONS in [6113277173](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6113277173).
- It has 44 declarations (37 theorems, 3 abbrevs, 4 defs). Every axiom is within `propext`, `Classical.choice`, `Quot.sound`. Its census is 48 = 44 user + 0 user-generated + 4 internal.
- Every premise is proved inside the fixtures, with universe 0 and coefficients in `ModuleCat ℤ`:
  - **empty space:** no components, and H0 is a zero object through the module's natural comparison; no nonempty or path-connected hypothesis is used.
  - **one point:** the point class augments to `𝟙 ℤ` and is nonzero, which distinguishes ordinary H0 from reduced H0.
  - **same component:** in `ℝ`, the difference of the classes of `0` and `1` is zero, for `ℤ` coefficients and for zero coefficients.
  - **two components:** the two points of discrete `Bool` lie in different components, and their classes differ.
  - **merging map `Bool → Unit`:** the difference dies while each image class stays nonzero.
  - **non-injective projection `Bool × Bool → Bool`:** it kills one difference and keeps another.
  - **strict closed-superlevel inclusion:** for `x²`, from −1 to −1/4, a difference survives.
  - **zero coefficients:** H0 of every space is a zero object, so the two `Bool` classes coincide although their components differ.
  - **`ZMod 2`:** a characteristic-two coefficient object, where separation holds.
- Before registration it went through a kernel verifier, a specification verifier, a fix round and a delta verifier. The fix round answered six specification findings, among them the stronger non-merge fixtures and use of the natural isomorphism at zero coefficients. All of this is same-account, credit 0.

**Three new expected-failure controls**, each leaving exactly `⊢ False` (exit 1) in the gate's emitted format:
- `RejectMergerKillsPointClass`, refuted by the nonzero image class.
- `RejectSeparationWithoutNonzeroCoeff`, refuted by the zero-coefficient equality.
- `RejectNonmergerKillsDifference`, refuted by the surviving projection difference.

Discriminators that keep each tactic and swap in the true sibling statement do not reach `⊢ False`. That shows each control fails for its intended reason.

**Registration, local commit `9e91f9f8fb01222d495468bddb16f89182ba3cd4`** (tree `aad46f77…`, one forward commit on `940dffc9`):
- The registration adds the module as the fifth companion module and its six-module `leanchecker` command; the five-module form is refused.
- It adds the three controls. The evidence set goes from 54 to 65 files.
- It adds an owner-authored provenance record, compared whole and covered by a negative test.
- It updates the tests, `replay.sh`, `SOURCE_FILES.json`, README and SCOPE.
- The module docstring now states that review [5485883117](https://github.com/d6g8k5htny-coder/Math-/pull/428#pullrequestreview-5485883117) exists, and that FORMAL HOLD is unchanged and for the handoff owners. Module bytes at the commit: sha256 `e4f76f7c…`, 28,898 B.
- The two items of the DOC_CLARIFICATION in [6116225936](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6116225936) ride this commit:
  - the README's naturality exclusion is relabelled as the C254 handoff's own, since C264 proves the stated ordinary-H0 naturality only;
  - the registration section's "no review exists" is scoped to its commit date, with pointers to reviews 5485883117 and 5485829137.

**Full local replay of `9e91f9f`, exit 0.** The receipt:
- `checked_commit 9e91f9f8…`, `kernel-checked`, `theorem_count` 118 (81 + 37), 65 `log_sha256` entries;
- census 181 = 153 user + 1 compiler-generated user-named + 27 internal;
- `leanchecker` over the six modules, mode `modules`;
- all eleven controls refused for their intended reasons (eight `REJECTED_BY_LEAN`, three `REJECTED_BY_AXIOM_AUDIT`);
- local receipt sha256 `451c5a2d…`, `run_id` local, a record of a computation and not evidence.

**Next:** when the two verifiers come back clean, after at most one fix round amending the unpushed commit, I push it as a fast-forward of this branch. A changed-input readback request for `940dffc9..<new>` follows, and the hosted runs get bound in this PR's body.

**Not established:**
- No review of `OrdinaryH0` by this lane. FORMAL HOLD on both modules is unchanged.
- No Morse, critical-level, interval, elder, count, Field or stochastic statement.
- No hosted receipt yet for the fixtures.

---
_Generated by [Claude Code](https://claude.ai/code)_
