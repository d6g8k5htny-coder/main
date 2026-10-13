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
- **Base:** `main` at `b64e70c949055a021864008cee7b9c2c403da5c5`. The head is seven commits on it, one per path group, with the torus commit last.
- **Base movement:** Math- `main` has since moved to `a6116d8a2173c7b450791bd89331d1565922bd55` (Math-#454). That change touches only `frontiers/cusp_coefficient_certified_20261001/`, with no path overlap. `git merge-tree --write-tree a6116d8a 7c1cd5d0` is clean, giving tree `d5649552758565210f750c2ecb2f319ff14acfb6`. The hosted runs tested merge `a25fd4a5` on base `b64e70c9`. Checking the change on current `main` belongs to the integrator.
- **Frozen diff:** `git diff b64e70c9 7c1cd5d0` is 169,773 bytes, SHA-256 `152a1391ac9b3262c31fa5a6538540b75eef7c8e0524569c00df2b78f2f48012`.
- **Changed paths:** the 38 files listed under "Files changed": 1,597 insertions and 157 deletions.
- **Overlap:** the only open PR touching these paths is draft #312. Its branch, candidate `38ec7fcf`, findings and history are untouched, under the sibling-branch handoff. Math-#454, the only PR updated since the claim, touches only `frontiers/cusp_coefficient_certified_20261001/`.
- **Status:** FROZEN at this head. Every hosted run and every local row below is bound to it.
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
| spec `test_check.py` / specproof `test_check.py` | 105 / 78 OK | 105 / 78 OK |
| torus `test_torus_filtration.py` | 32 OK | 32 OK |

Source checks pass for formal (`gate.py`), all three D2 companions, Cap, spec, specproof and torus.

**Hosted runs at the head, acceptance matrix 2.** All are attempt 1 and event `pull_request`, and every job concluded `success`. The tested merge commit for every run is `a25fd4a551c0ba01a490ce4e87dd962c188eb690`, whose parents are base `b64e70c9` and head `7c1cd5d0`. The formal job's receipt records that commit as `checked_commit`. The runners use git 2.55.0.

| path | run | job | note |
|---|---|---|---|
| formal (reusable, called by the downstream hard gate) | [38469027987](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027987) | `formal / formal-evidence` 115542989282 | The receipt has `observations.preflight == observations.final`, `dependency_worktrees: CLEAN`, the G1 line and `workflow_run_attempt: "1"`. `tools/required_formal_check.py receipt` then bound it, with receipt SHA-256 `fabbf496…`. The same run's `downstream-replay` and `math-downstream-gates` jobs also passed. |
| Cap | [38469028018](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469028018) | `kernel-evidence` 115542988690 | |
| spec | [38469027651](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027651) | `statement-evidence` 115542998103 | |
| specproof | [38469027859](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027859) | `proof-evidence` 115542988205 | |
| torus | [38469027830](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027830) | `torus-filtration` 115542988389 | |
| D2 moment | [38469027744](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027744) | `d2-moment-bridge` 115542988523 | |
| D2 square | [38469027999](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027999) | `d2-square-schur` 115542989372 | |
| D2 atom | [38469027661](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027661) | `d2-atom-integral` 115542987815 | |
| full repository regression suite | [38469027784](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027784) | `full-suite` 115542988476 | |
| C6 numerics; remainder-rate controls (triggered by the path filters) | [38469027942](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027942); [38469027635](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38469027635) | 115542989048; 115542987853 | |

Each companion's `finish` refuses a receipt without its final observation, so each green D2, torus, Cap, spec and specproof job wrote one. The uploaded artifacts hold those receipts.

**Local runtime rows.**
- **Provenance:** every row is `LOCAL:claude-01Cz-container:<row>`. None carries a GitHub run identity.
- **Environment:** Linux 6.18 x86_64, Ubuntu 24.04.4, Python 3.11.15, 4 CPUs, `LEAN_NUM_THREADS=2`.
- **Method:** each row runs the hosted workflow's own step commands. It uses a fresh detached worktree at the head, with `.lake/packages` linked to one shared checkout of the nine locked packages.

| row | result | before and after the Lean runs | receipt SHA-256 |
|---|---|---|---|
| formal | gate `--execute` exit 0, 143 s; 150 tests OK | equal; G1 line; 9/9 CLEAN | `b6668017…` |
| torus | `replay.sh` exit 0, 305 s | `toolchain.log` = `toolchain-final.log` byte for byte; `packages.log` = `packages-final.log`, 9/9 clean | `12876fc6…` |
| Cap | 38 + 38 tests OK; source PASS (152 targets); `--execute` exit 0, 170 s | equal; G1 line; CLEAN; source identity equal | `379bb67a…` |
| spec | 105 + 105 tests OK; LeanHelpers OK; `--execute` exit 0, 135 s | equal; CLEAN | `cea7deb4…` |
| specproof | 78 + 78 tests OK; LeanHelpers OK; `--execute` exit 0, 282 s | equal; CLEAN | `f8e878ec…` |
| D2 moment | `replay.sh` exit 0, 854 s | core receipt equal; toolchain identical; 9/9 clean after the companion's last Lean run | `0c54012e…` (core `79464b17…`) |
| D2 square (re-run with git 2.55.0) | `replay.sh` exit 0, 1,325 s; 844 + 844 repository tests OK | core receipt equal; toolchain identical; 9/9 clean after the companion's last Lean run | `0a98cc61…` (core `da196b2b…`) |
| D2 atom (re-run with git 2.55.0) | `replay.sh` exit 0, 1,357 s; 844 + 844 repository tests OK | core receipt equal; toolchain identical; 9/9 clean after the companion's last Lean run | `308337cc…` (core `0fcb366b…`) |

**Local environment finding, not a candidate defect.**
- **The first local D2 square and D2 atom rows failed before any Lean run,** at the replays' full `tests/` step: 844 run, 166 failures and 3 errors each.
- **The cause:** all 169 sit in `tests/test_proof_slice_check.py` and `tests/test_retrofit_record_check.py`. Their tools call `git --no-lazy-fetch`, which this container's git 2.43.0 lacks.
- **Why it is not this PR's:** neither the tests nor those tools are in this diff. The hosted full suite passes them on git 2.55.0.
- **The re-run:** I built git 2.55.0 from `https://github.com/git/git`, tag `v2.55.0`: tag object `5ce91c05`, commit `e9019fcafe0040228b8631c30f97ae1adb61bcdc`.
  - The build used `make NO_CURL=1 NO_EXPAT=1 NO_GETTEXT=1 NO_TCLTK=1 NO_PERL=1 NO_PYTHON=1 NO_OPENSSL=1` with gcc 13.3.0.
  - The executable is 19,761,728 bytes, SHA-256 `fc6ff15882c6e285c1319869a09e4bc5d6e72a1eeaba3099ef4907a5efad74c6`, installed into scratch.
  - I put it first on `PATH` for those two rows only. Each row header records `git version 2.55.0 at <path>`.
  - The re-runs above are those rows. The failed first attempts, run with `/usr/bin/git` 2.43.0, stay in their logs as negative evidence. A second D2 atom re-run, queued behind D2 square, was stopped as a duplicate of the parallel one; its partial log is kept.
  - This changes nothing in the repository, on any runner or in any package pin.

## Evidence and limits

- **Author and exposure:** written by Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. The torus commit was drafted by a same-session subagent and reviewed and integrated by the author. Author review earns no qualification. Organizational-independence credit is 0.
- **What this does not establish:**
  - Two observations are two instants. They do not show that the runtime or the dependencies were unchanged between them.
  - A local row is not a receipt, and it is not a hosted run.
  - None of this shows any Lean statement is aligned with the mathematics, or moves any claim, premise or status label.
- **Fail-closed cost:** a runner that leaves untracked, non-ignored files in a package checkout is now refused. On these hosted runs every path observed CLEAN, under the unchanged `lean-action` cache step.
- **Historical receipts** are not re-validated. They stay as they were.
- **Out of scope:** Math-#451's held frontier, #312's admission driver, other repositories, and the move of `ubuntu-latest` to a new Ubuntu.
- Scientific effect NONE.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
