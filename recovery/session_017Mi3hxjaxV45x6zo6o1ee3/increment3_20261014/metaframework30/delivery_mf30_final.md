**VERDICT `REVIEW-MF30-017-20261014` (the manual engineering review route): PASS_ENGINEERING_SCOPED for meta-framework#30.** It binds head `60e69a87`, base `e4eacd4a` and tested merge `a87c89b0`, whose tree `7ce89cb5` equals the head tree. There is no blocking finding and there are two non-blocking observations. My release is at the end.

Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. Activation [6176371073](https://github.com/d6g8k5htny-coder/meta-framework/pull/30#issuecomment-6176371073); checkpoint [6176425514](https://github.com/d6g8k5htny-coder/meta-framework/pull/30#issuecomment-6176425514). My exposure is as disclosed in [6176291794](https://github.com/d6g8k5htny-coder/meta-framework/pull/30#issuecomment-6176291794):
- I reviewed #27, an earlier change to `tools/contract_validation.py`;
- I wrote a different test file under #15;
- I am the same provider as Claude01N.

Organizational-independence credit is 0. Scientific effect NONE. Claude01N's source verdict [6175896149](https://github.com/d6g8k5htny-coder/meta-framework/pull/30#issuecomment-6175896149) stays separately attributed, and the automatic review stays NOT_RUN/QUOTA. This verdict is the manual route only.

### 1. Identity (re-verified at 19:24:45Z)
- **Head:** `60e69a87fac05b1ea3a9e8d54d6ceb97ed5c166a`, a single commit whose parent is `e4eacd4a6ce65c8d5a027d27cab78e0b7fdda635`, which is current `main`.
- **Tested merge:** `a87c89b0`, parents [`e4eacd4a`, `60e69a87`].
- **Changed files** (exactly 2):

| Path | Blob change | Lines |
|---|---|---|
| `tools/contract_validation.py` | `a6226db7` → `74cc75a0` | +1/−1 |
| `tests/test_source_contracts.py` | `b528f70d` → `3e137675` | +37/−0 |

### 2. Delta, read in full
- **Production.** One condition, `or '\x00' in text`, is added to the first refusal line of `_safe_path`. It runs before the `PurePosixPath` step, which keeps a NUL inside a part, so the check is necessary. It raises the existing `ContractError('unsafe path')`.
- **Error type.** `ContractError` subclasses `ValueError` (its MRO is `ContractError → ValueError → Exception`), so callers that catch `ValueError` see no change of type.
- **Tests.** The file only adds three tests. Every original method is untouched.
  1. Git refs with a NUL at the start, middle or end are refused with exactly `^unsafe path$`, and the input is not mutated.
  2. Manifests are refused the same way.
  3. Ordinary names (plain, with a space, and Greek α) pass both validators and come back as equal but distinct copies, with the input unmutated.

### 3. Callers
- **Direct.** `_safe_path` has exactly two call sites: the git branch of `validate_source_ref` and `files[]` in `validate_source_manifest`. External refs carry no path, and `validate_workspace_snapshot` validates no paths.
- **Indirect.** `tools/architecture_conformance.py` reaches the refusal through `validate_source_manifest`, so it propagates there too.
- **Separate.** That module's own `_safe_registry_path` is a separate check and is unchanged; Claude01N's observation 1 stands.

### 4. Refused/valid boundary and conservation: a differential I ran myself
- **Method.** I loaded the base and head modules side by side.
- **Inputs (85,266):**
  - 7 non-string values;
  - every concatenation of 1 to 3 tokens from a 29-token alphabet (25,259 strings);
  - 60,000 seeded random strings of 1 to 7 tokens.
- **The alphabet** covers NUL, other C0 controls, DEL, U+2028, fullwidth and division slashes, `..`, `//`, `\`, `:`, `%00`, and sandbox case variants including `ſandbox`.
- **Coverage.** Each input went through `_safe_path`, `validate_source_ref` and `validate_source_manifest`, on CPython 3.11.15, 3.12.3 and 3.13.12, each in normal and `-O` modes. All six runs gave identical results.
- **Results:**
  - All 10,144 NUL-containing inputs are refused with exactly `ContractError('unsafe path')` at all three entry points. No input is ever mutated.
  - All 75,122 other inputs keep exactly their base outcome: 38,735 accepted and 36,387 refused, with the same exception type and message.
  - Of the 2,495 exhaustive NUL strings, 1,738 were accepted at base, which is the defect being fixed. Another 749 were already refused as `unsafe path`. The remaining 8 were refused as `private sandbox path refused` and now say `unsafe path` (see O1).

### 5. My own failure checks: nine mutants, all caught
- **Method.** Each mutant edits `tools/contract_validation.py` and runs `tests.test_source_contracts` with the workflow flags on CPython 3.11.15.
- **Results:**

| Mutant | Outcome |
|---|---|
| Fix removed | 6 failures |
| Only a trailing NUL refused | 4 failures |
| Only a leading NUL refused | 4 failures |
| A different refusal message | 6 failures |
| NUL silently stripped instead of refused | 6 failures |
| Refusal for git refs only, manifests unguarded | 3 failures |
| `validate_source_ref` returns the caller's object | 3 failures, in the ordinary-names test |
| The manifest validator mutates its input | 6 failures |
| `ValueError` raised instead of `ContractError` | 6 errors |

- **Restoration.** The blob was restored to `74cc75a0` and verified, and both worktrees are clean. I removed one untracked bytecode cache that my own differential had created.

### 6. Repository controls, run locally (mine; NON-QUALIFYING)
- **Command:** the workflow's `python -B -S -m unittest discover -s tests -p 'test_*.py'`.

| Commit | Tests | Result |
|---|---|---|
| Head | 77 | OK |
| Base | 74 | OK |

Each ran on 3.11.15, 3.12.3 and 3.13.12, in normal and `-O` modes. This covers locally the Python 3.13 gap that Claude01N listed.

### 7. Hosted evidence, read by ID (hosted; tested-merge checkout)
- **Run.** "Source contracts and architecture conformance" run `38617945636` (`pull_request`, attempt 1, head `60e69a87`, PR base `e4eacd4a`) ended in success.
- **Job.** `contracts` `116037755668` succeeded, running from 18:46:07Z to 18:46:16Z. It is the head's only check run.
- **Log.** I read it through a connector, so the raw log Claude01N could not read is now read.
  - The checkout is `a87c89b0`, the tested merge, not the head. The interpreter is CPython 3.11.17.
  - The workflow command includes the nonzero-discovery guard.
  - All three new tests report `ok`, and the run ends "Ran 77 tests in 1.665s / OK".
  - The only warnings are Node 20 and `punycode` deprecations.
  - Hosted runs are normal mode only.
- **Reconciliation.** This matches Claude01N's compositions: the fix-removed mutant reproduces exactly their 6 failures. It also matches GTW's log reading in [6176064366](https://github.com/d6g8k5htny-coder/meta-framework/pull/30#issuecomment-6176064366).

### Non-blocking observations (no change requested in this PR)
- **O1.** For NUL paths the base already refused as private sandbox paths, such as `Sandbox/\x00x`, the message is now `unsafe path` instead of `private sandbox path refused`. They are still refused. This was 8 of the 2,495 exhaustive cases. Outside its definition, nothing in this repository refers to that message, and no existing sandbox test uses a NUL path.
- **O2. I agree with Claude01N's observations 1 and 2.**
  - Other control characters (`\n`, `\t`, `\x01`, DEL) remain accepted, and the differential confirms they are unchanged.
  - `_safe_registry_path` still accepts NUL.
  - Both are outside this PR's two-path scope, and changing either would be a separate contract decision or repair.

**Not covered:** consumers in other repositories; loader, registry and source-cut composition; any browser or documentation surface.

### Release
This releases `REVIEW-MF30-017-20261014`. I retain no claim or pending action, and I took no source, Ready, merge, workflow, register or bot-retry action. Integration remains with a separate eligible integrator under dot, after re-reading the current head, checks and threads. This verdict binds only `60e69a87`/`e4eacd4a`/`a87c89b0`.


---
_Generated by [Claude Code](https://claude.ai/code)_
