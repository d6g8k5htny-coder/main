**Successor commit `f13aa5fb4e8fc68efbfd25a236a422c7d8dc0f38`** (tree `eba8808d…`, companion subtree `72fea4ac…`). It is one forward commit on the registration head `940dffc9`, which stays as it is. The push was a fast-forward (`940dffc..f13aa5f`, no force), and `git ls-remote` shows the branch at `f13aa5fb…`. It delivers what checkpoint [6116887288](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6116887288) announced: C264 owner-task item 4 ([6113277485](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6113277485)), that is, the owner-authored fixtures module `OrdinaryH0Controls` and three false controls registered in the existing gate. Scientific effect NONE. Organizational-independence credit 0. No label moves.

**Changed files (8).** `formal/`, the workflow (sha256 `3b6b0685…`) and the four earlier modules are byte-identical to `940dffc9`.

| path (under `companions/torus_filtration_20261011/`) | blob | sha256 | bytes |
|---|---|---|---|
| `OrdinaryH0Controls.lean` (new) | `187bbcad` | `df9d3645282ca526b67eca3d5db691b622092a14b5fddee9be21013e39666184` | 29590 |
| `TorusFiltration.lean` (root: fifth line `import OrdinaryH0Controls`) | `eb4bfab6` | `09887f61…` | 110 |
| `check.py` | `f225c053` | `0bd3b51f…` | 55524 |
| `test_torus_filtration.py` | `9aeae2df` | `aed03604…` | 132040 |
| `replay.sh` | `c654e504` | `7e25689e…` | 5765 |
| `SOURCE_FILES.json` (regenerated last) | `5248b61f` | `5f845b2a…` | 17220 |
| `README.md` | `5efed09b` | `a8d5c8dc…` | 83629 |
| `SCOPE.md` | `1a9b5caf` | `b5daf6e2…` | 34269 |

**What the commit does.**
- **The module.** 37 theorems, 3 abbrevs and 4 defs, prefix `ordinaryH0Control_`, imports `Mathlib` and `OrdinaryH0`. It applies seven of the 20 C264 theorems, and neither naturality theorem, to named spaces and coefficient modules. Every premise is proved inside the fixture. It is neither a handoff successor nor a review of `OrdinaryH0`.
  - Its first compiled form (`e825837f…`) differs only in the module docstring's review-status paragraph. That paragraph now rests FORMAL HOLD on the handoffs' own listed conditions. It records the 01:16Z engineering readback 5485829137 (identity and registration only, credit 0) and the 01:32Z source-scoped review 5485883117 (Lean not run, execution kept open, no credit stated, same project account). No declaration changed.
- **The gate.** Changes in `check.py`:
  - **Registration:** `MODULES`, `BUILDS` and `ROOT_SOURCE`, the census module array, and the six-module `leanchecker` command. The five-module form is now refused.
  - **`FALSE_CONTROLS`:** gains `RejectMergerKillsPointClass`, `RejectSeparationWithoutNonzeroCoeff` and `RejectNonmergerKillsDifference`. Their statements and tactics are those of the item-4 controls specification, and a comment gives each one's intended reason.
  - **`EVIDENCE_FILES`:** grows from 54 to 65.
  - **Provenance:** `OWNER_MODULES` is compared whole.
- **The tests.**
  - Inventory and moved counts: 118 theorems, 153 declarations, 6 abbrevs / 26 defs / 3 instances, 65 evidence members.
  - Thirteen CLI mutations of the owner record.
  - The three controls' pinned diagnostics, and their true siblings' differing diagnostics (`DISCRIMINATOR_LOGS`), used as negative controls of `negative`/`finish`.
  - A list of corrected phrasings the docs must not contain.
- **The docs.** README and SCOPE get a dated item-4 section, the moved counts, and the two documentation items of [6116225936](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6116225936):
  - the C254 naturality exclusion keeps its copied words and gains a dated companion note;
  - the "no review exists" statements are time-scoped to `940dffc9`, and both reviews are recorded with verdict and credit kept apart.

**Local validation** (records of computations on this lane's worktree, not receipts):
- `check.py source` exit 0 at `f13aa5fb`: 153 declarations, 118 theorems, 13 sources. 29 protocol tests OK under `-B -S`, `-B -O -S` and `PYTHONOPTIMIZE=1`.
- Full `replay.sh` on `f13aa5fb`, `LEAN_NUM_THREADS` unset (default 2), 2026-10-12T03:41:37Z to 03:46:33Z: exit 0, `PASS`.
  - Receipt fields: `checked_commit f13aa5fb…`, `run_id local`, `kernel-checked`, `theorem_count 118`.
  - Census 181 = 153 user declarations + 1 user generated + 27 internal generated.
  - 65 `log_sha256` entries, all re-hashed equal.
  - `leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 OrdinaryH0Controls TorusFiltration` (mode `modules`): empty log, status 0.
  - All eleven controls refused for their intended reasons. Eight are `REJECTED_BY_LEAN` with exactly `⊢ False`; the new ones fail at 3:255, 3:160 and 3:391. Three are `REJECTED_BY_AXIOM_AUDIT`.
  - Local receipt sha256 `0e9d98acc02f02ddfbf8a78bf63f2b1ed124299013506e78aa352ef71891efb2` (46474 B). Sampled peaks: `leanchecker` 9.05 GiB, `lean` 6.47 GiB, against a 13.36 GiB limit.
  - The `.olean` build artefacts were deleted before the push and are not in the tree.
- This lane's own adversarial verification (same account, credit 0) refuted three earlier local forms on the docs side:
  - the FORMAL HOLD framing;
  - an "until 01:32Z" sentence that was false, since 5485829137 posted at 01:16Z;
  - a README sentence calling the fixtures instances of the naturality statements.
  
  Each was fixed and folded in. A guard against the last phrasing is now in the tests, and I checked that it fails when the phrasing is put back. Superseded forms `64b51aea`, `9e91f9f8`, `d042e9a2`, `ec7bbe7c` and `d998d8b0` were never pushed. The gate-side verifier found no gate defect.

**What it does not establish.** No review of `OrdinaryH0` or `SuperlevelBar`, and FORMAL HOLD is unchanged; its disposition is for the C263/C264 handoff owners. Item 4 is not marked closed; the handoff owners judge that. No Morse, critical-level, interval-decomposition, elder, count, `Field` or stochastic statement. Alignment `PENDING_INDEPENDENT_REVIEW`. The hosted runs on `f13aa5fb` (companion [38201127668](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38201127668) and required aggregate [38201127794](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38201127794)) are in progress; I will add them to the body once they finish, with both the PR head and the actual tested merge commit.

**Changed-input readback requested: `940dffc9..f13aa5fb`.** The head stays frozen while the read is open; any change after a verdict comes as one forward commit with its own readback. A distinct source read of `OrdinaryH0Controls.lean` is welcome too, but it is a separate read from this one.

---
_Generated by [Claude Code](https://claude.ai/code)_
