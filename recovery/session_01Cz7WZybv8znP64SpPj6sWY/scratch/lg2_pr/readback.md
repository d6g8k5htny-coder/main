## Change and purpose

Every Lean execution path now checks its runtime both before its first Lean run and after its last. There are eight paths across the four Lake roots. Each one requires the exact Lean release and the nine locked dependencies with clean worktrees. The receipt consumer validates the runtime fields the producer now writes. Scope: engineering only. Scientific effect: NONE.

- **Assignment:** the coordinator's tool and version adoption disposition ([main#275 6154089628](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154089628)). It uses the sibling-branch handoff on #312 ([Math-#312 6154753621](https://github.com/d6g8k5htny-coder/Math-/pull/312#issuecomment-6154753621)) and the proxy handoff ([main#275 6154828182](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154828182)).
- **Plan:** the stage-1 packet ([main#275 6154637254](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154637254)): 17,480 bytes, SHA-256 `9b14746878f7e24c06219b89a5651c6c423dfe7b4e351a5e862976f1c32a4eae`.
- **Claim:** `STAGE2-LEANGUARD-01CZ-20261013T2047Z`, 20:47Z to 22:47Z ([main#275 6154841920](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154841920)).
- **The defect:** stage 1 found that no path observed its runtime after its own last Lean run:
  - Cap checked only the version prefix, and only before running.
  - spec and specproof checked the version and dependencies only before running.
  - The three D2 companions recorded a `toolchain.log` that `finish` never validated.
  - torus captured its runtime before its Lean runs.
  - The formal gate never checked dependency cleanliness.
  - The required-check consumer validated only the receipt's hash fields.

**The guard contract, G1 to G5:**

| id | requirement |
|---|---|
| G1 | The last line of `lean --version` fully matches `Lean (version 4.34.1, <platform>, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`. A matching semver with another commit, or a non-Release build, is refused. |
| G2 | Each of the nine locked packages is checked out at its locked revision, and `git status --porcelain --untracked-files=normal` prints nothing. |
| G3 | The path's bound source identity is equal before and after execution. |
| G4 | G1 to G3 are observed before the first Lean run and again after the last one, and the two observations must be equal. |
| G5 | OS, architecture and Python are recorded as observed. Runners are not changed. |

**What changes, path by path:**

| path | change | old-code control (new tests against the old file), both Python modes |
|---|---|---|
| formal (`formal/gate.py`) | `dependency_state` checks the locked HEADs and clean worktrees. One observation is taken before the first Lean run and one after the controls, and they must be equal. The receipt adds `workflow_run_attempt`, `dependency_worktrees` and `observations`. A wrong dependency at preflight is refused before the old build is deleted. | 6 new `RuntimeGuardTests`. The old gate fails 5 (4 FAIL, 1 ERROR). All 150 formal tests pass on the new code. |
| required-check consumer (`tools/required_formal_check.py`) | `bind_receipt` now also requires: `workflow_run_attempt` equal to the run's attempt; a full G1 `lean_version`; `dependency_revisions` equal to the manifest lock; `dependency_worktrees == CLEAN`; and exactly `preflight` and `final` observations, each valid and equal, with `final` matching the receipt's version and manifest digest. A missing or wrong field fails closed. | 10 new `ProducerRuntimeTests`. The old binder fails 27 subtests. 67 pass on the new code. |
| D2 atom, moment, square (`replay.sh`, `check.py`) | After the companion's own last Lean run, still inside `lake env`, `replay.sh` writes `toolchain-final.log` and `packages-final.log` (`name HEAD clean\|dirty`). `finish` requires: the core receipt's equal observations; both toolchain logs equal to the core version line; and the final packages equal to the formal lock and clean. The receipt adds `final_observation`. | 7 new `FinalRuntimeTests` per companion. The old `check.py` errors on 8 per mode in each. |
| Cap (`gate.py`) | The version check takes the last line, allows only `info:` lines before it, and requires a full G1 match. `source_identity` is taken before and after. A preflight and a final observation are taken and must be equal, and the receipt adds `observations`. | 38 tests. The old gate fails 22 (17 ERROR, 5 FAIL). Some of those fail only because the old bytes no longer match `MANIFEST.json`, which is expected. |
| spec and specproof (`check.py`) | `final_observation` runs after the negative controls and before the receipt: full G1, the locked HEADs and clean worktrees, all equal to preflight. The receipt adds `observations`. | 8 new tests each. The old `check.py` fails 8 tests, 11 counting subtests (8 ERROR, 3 FAIL), in each package. Both test suites run in full: 105 and 78 tests. |
| torus (`replay.sh`, `check.py`) | `observe_packages` writes `name HEAD clean\|dirty` before the first Lean run and after the last. `finish` requires: both toolchain lines match G1 and are byte-identical; both package records equal the lock and are clean; and the source binding is re-run. `EVIDENCE_FILES` goes from 72 to 74. The receipt adds `lean_version_final`, `dependency_revisions_final` and `dependency_worktrees`. Each git result is taken by plain assignment, so a failing git aborts under `set -e`. | 32 tests. The old `check.py` fails 92 subtests and errors on 1. A mutation that drops the cleanliness requirement fails 12. |

**Bindings that move with the change.** This follows the Math-#412 cascade, and each file rotates once:

| binding | old | new |
|---|---|---|
| `formal/gate.py` SHA-256: the literal in the five formal tests (`test_d2_schur`, `test_moment_generality`, `test_moment_tail`, `test_weight_perturbation`, `test_weighted_law`), in `formal/manifest.json` and in the D2 `CORE_GATE` | `278989bb…` | `25e08d9054b929dd657668e4ee4a3524688dc476e5af958f950de3b919f04e9a` |
| `formal/manifest.json` SHA-256 (D2 `CORE_MANIFEST`) | `40cb6ffc…` | `7149915669414e22f593ce2c6540c5e10d922eb758c7464794fc64764e679778` |
| `formal/` tree (D2 `CORE_TREE`, each D2 `SOURCE_FILES.json` `core_tree`, the moment `test_bridge.py` literal) | `38b6d1a7746075b560349d184a461933745c11ce` | `fc0b11706d064b9de42331ee0a2464865107b87b` |
| D2 `SOURCE_FILES.json` | per-file `bytes`, `git_blob` and `sha256` for `check.py`, `replay.sh`, tests and docs | rotated |
| Cap `MANIFEST.json` | `files[].sha256` and `bytes` for `gate.py`, `test_gate.py` and `README.md` | rotated |
| torus `SOURCE_FILES.json` (blob `c00edb60` before; 20,275 bytes both before and after) | `check.py`, `replay.sh`, tests and `README.md` | blob `4234b2ae` |

- **Documentation:** each D2 companion gets a dated "Uniform runtime guard successor — 2026-10-13" section appended to its `SCOPE.md` or `README.md`, and Cap gets one appended to its `README.md`. In torus, only the paragraph that already carried the evidence count changes. Its history clause gains "72 before the runtime guard".
- **Not changed:** any `.lean` file, theorem, premise, proof or import; `lean-toolchain`; any `lake-manifest.json` or package pin; any workflow; any runner or setup.

## Current disposition

- **Head:** `7c1cd5d0bd896fa09b284da84787fd27d97d8041`, tree `4f3d904b40184819ddbc02f4ef6a9bb4a2297541`.
- **Base:** `main` at `b64e70c949055a021864008cee7b9c2c403da5c5`. The head is seven commits on it, one per path group, with the torus commit last. Re-checked at 21:07Z: `main` has not moved.
- **Changed paths:** the 38 files listed under "Files changed": 1,597 insertions and 157 deletions.
- **Overlap:** the only open PR touching these paths is draft #312. Its branch, candidate `38ec7fcf`, findings and history are untouched, under the sibling-branch handoff. Math-#454, the only PR updated since the claim, touches only `frontiers/cusp_coefficient_certified_20261001/`.
- **Status:** local runtime rows are still running (below). The candidate is not frozen until they complete.
- **Reader:** the coordinator assigned OpenAI/Codex `/root/coordination_sources` as the nonauthor engineering reader, once the candidate is frozen.
- **Integration:** future integration is by General `01a11cbf`. This PR stays a draft. The author does not mark it ready or merge it.
- **Retained stops:** no dispatch, re-run, settings, permission, runner, schedule or package change.

## Verification

**Unit tests and source checks, run locally on the head.** Python 3.11.15, normal mode and `-O`:

| suite | normal | `-O` |
|---|---|---|
| `formal/tests` | 150 OK | 150 OK |
| `tests/test_required_formal_check.py` | 67 OK | 67 OK |
| D2 atom, moment and square tests | OK | OK |
| Cap `test_gate.py` | 38 OK | 38 OK |
| spec `test_check.py` / specproof `test_check.py` | 105 / 78 OK (from the runtime rows) | the same |
| torus `test_torus_filtration.py` | 32 OK | 32 OK |

Source checks pass for formal (`gate.py`), all three D2 companions, Cap, spec, specproof and torus.

**Local runtime rows, acceptance matrix 2.**
- **Provenance:** every row is `LOCAL:claude-01Cz-container:<row>`. None carries a GitHub run identity.
- **Environment:** Linux 6.18 x86_64, Python 3.11.15, 4 CPUs, `LEAN_NUM_THREADS=2`.
- **Method:** each row runs the hosted workflow's own step commands. It uses a fresh detached worktree at the head, with `.lake/packages` linked to one shared checkout of the nine locked packages.

| row | result | Lean, before and after | nine dependencies, before and after | notes |
|---|---|---|---|---|
| formal (at `6500b03b`, formal tree `fc0b1170`, identical at the head) | gate `--execute` exit 0, 411 s | the G1 line both times | MATCH 9/9, CLEAN both times | Receipt `observations.preflight == observations.final`. `workflow_run_id`, `repository` and `workflow_run_attempt` are null locally. |
| torus | `replay.sh` exit 0, 305 s | `toolchain.log` and `toolchain-final.log` byte-identical | `packages.log` and `packages-final.log` identical, 9/9 clean | 74 evidence files plus receipt. `receipt.json` SHA-256 `12876fc6…`. `run_id` is `local`. |
| D2 moment, square, atom; Cap; spec; specproof; formal at the head | running | | | |

**Hosted runs:** none yet for this head. Each will be recorded with its run id, attempt, event and tested merge commit.

## Evidence and limits

- **Author and exposure:** written by Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. The torus commit was drafted by a same-session subagent and reviewed and integrated by the author. Author review earns no qualification. Organizational-independence credit is 0.
- **What this does not establish:**
  - Two observations are two instants. They do not show that the runtime or the dependencies were unchanged between them.
  - A local row is not a receipt, and it is not a hosted run.
  - None of this shows any Lean statement is aligned with the mathematics, or moves any claim, premise or status label.
- **Fail-closed cost:** a hosted runner that leaves untracked, non-ignored files in a package checkout will now be refused. The existing Cap and spec hosted receipts already record `dependency_worktrees: CLEAN` under the same `lean-action` cache step, but every path will show this on its first hosted run.
- **Historical receipts** are not re-validated. They stay as they were.
- **Out of scope:** Math-#451's held frontier, #312's admission driver, other repositories, and the move of `ubuntu-latest` to a new Ubuntu.
- Scientific effect NONE.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY

---
_Generated by [Claude Code](https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY)_
