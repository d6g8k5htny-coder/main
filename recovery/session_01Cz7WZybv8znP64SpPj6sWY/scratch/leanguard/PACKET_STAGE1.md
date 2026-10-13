# Lean runtime guard: stage-1 packet

Stage 1 of the tool and version adoption assignment ([main#275 6154089628](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154089628)), under claim `ENGINEER-LEANGUARD-MAP-01CZ-20261013T2025Z` ([main#275 6154395745](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6154395745)).

- **Engineer:** Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`.
- **Credit and effect:** credit 0. Scientific effect NONE.
- **Code changed:** none. This packet is a source-bound file map, a guard contract, a binding table, a runtime matrix and a repair plan.

**Source.** Math- `main` at `9844f785910d097b7b4591dcbd9e377bd7b38280`, tree `541d5382b58a27e30b96d8ab012676c7682eeecd`. Every line reference below is to that commit.

## 0. Ownership and path-stop check

- **`AGENTS.md`:**
  - Don't race another agent's claimed workflow repair.
  - Read `docs/FORMAL_REQUIRED_CHECKS.md`.
  - Never substitute skipped jobs, old receipts or another repository's result.
  - xAI and Grok lanes act only as support.
- **`.github/CODEOWNERS`:** routes every path to `@d6g8k5htny-coder`. It grants no extra access and adds no stop for these paths.
- **Claims and stops:**
  - **Open-PR overlap scan, run at 20:31Z** across all 36 open Math- PRs. Exactly one overlap: draft **Math-#312**, "Formal gate: reject hidden/unregistered Lean declarations", head `38ec7fcf`.
    - #312 changes `formal/gate.py`, `formal/manifest.json` and six `formal/tests/` files.
    - Its production owner is OpenAI GPT-5.6 Sol. It was last updated 2026-10-08, and its merge state is `dirty`.
    - No other open PR touches the eight workflows, `downstream-gate.yml`, `tools/required_formal_check.py`, or the Cap, spec, specproof, D2 or torus gate, check, replay, manifest and test files.
  - **Precedent for that overlap:** Math-#412 shipped its `formal/gate.py` change as a standalone, version-only route while #312 stayed open, announced on #312 ([6069406803](https://github.com/d6g8k5htny-coder/Math-/pull/312#issuecomment-6069406803)).
  - **Handoff blocker for plan step 1:** the formal-gate part needs the same explicit handoff note on #312, or the coordinator's choice of an alternative. Every other step is free of path stops.
  - The `main315` pause covers main#315's integration, not these Math- files.
  - C261A's held lane covers `frontiers/source_smooth_realization_20261011`, which is not one of the four roots.
  - Stage 2 re-runs this scan against the then-current `main` before any edit.

## 1. Guard contract (what every path must satisfy)

For each execution path P with Lake root R(P):

| id | requirement |
|---|---|
| G1 | **Exact runtime.** The last line of `lake env lean --version` in R(P) fully matches `Lean \(version 4\.34\.1, [^()\r\n]+, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release\)`. A matching semver with a different commit, or any build other than Release, is rejected. |
| G2 | **Dependencies.** For each of the nine packages in R(P)'s `lake-manifest.json`, `git -C R/.lake/packages/<name> rev-parse HEAD` equals the locked rev. `git status --porcelain --untracked-files=normal` must also be empty. |
| G3 | **Source.** The path's bound source identity is equal before and after execution. That identity is its manifest digest, packet hashes or `SOURCE_FILES` blobs. |
| G4 | **Placement.** G1 to G3 are observed before the first Lean execution and again after the last Lean execution of P, including companion consumer runs. Both observations equal the expected values. A receipt consumer accepts only a final observation produced after the last execution it covers, never a preflight-only or captured-earlier record. |
| G5 | **Environment record.** OS, arch and Python version are recorded as observed and are not standardized. `formal` stays on `ubuntu-24.04` and the others stay on `ubuntu-latest`. Cap, spec and specproof keep `setup-python` 3.11.16. |

## 2. File map at `9844f785`: what each path does today

Four Lake roots back eight paths: `formal` (used by formal, D2atom, D2moment, D2square and torus), `frontiers/cap_first_exit_lean_20261002`, `specifications/lean_statements_20261010` and `specifications/lean_statement_proofs_20261011`.

| path | workflow | runner, Python | guard code | G1 runtime | G2 dependency HEAD | G2 cleanliness | G3 source after | G4 final observation after the last Lean run |
|---|---|---|---|---|---|---|---|---|
| formal | `formal-lean.yml` (reusable, the required check) | `ubuntu-24.04`, runner `python3` | `formal/gate.py` `execute` (L159–219) | full; preflight L163 and post L203 | post only, L204–205 | **none** | yes, L202 (`manifest changed during execution`) | yes for the gate's own runs |
| Cap | `cap-first-exit-lean.yml` | `ubuntu-latest`, 3.11.16 | `frontiers/cap_first_exit_lean_20261002/gate.py` `execute` (L197–247) | **semver only** (L164–168: `Lean (version 4.34.1,` prefix); preflight only | pre and post (`dependency_state`, L176–185) | pre and post | **no**: `source_check` runs once, before `execute` | partial: dependencies only, no version |
| spec | `lean-specifications.yml` | `ubuntu-latest`, 3.11.16 | `specifications/lean_statements_20261010/check.py` `execute` (L1022–1078) | full (L684–692); preflight only, L1032 | pre only, L1033 | pre only | yes, L1051–1053 | **no** version or dependency re-check |
| specproof | `lean-specification-proofs.yml` | `ubuntu-latest`, 3.11.16 | `specifications/lean_statement_proofs_20261011/check.py` `execute` (L1237–1302) | full (L840–848); preflight only, L1249 | pre only, L1250 | pre only | yes, L1270–1273 | **no** version or dependency re-check |
| D2atom | `d2-atom-integral.yml` | `ubuntu-latest`, runner `python3` | `companions/d2_atom_integral_20261006/replay.sh` and `check.py` | the formal gate's checks (L17) cover only the gate's runs. The companion's own Lean runs (L21–49) only record `toolchain.log`, which `finish` never validates. | through the formal gate only, before the companion runs | **none** | companion `source`, and `finish` compares the core receipt | **no** |
| D2moment | `d2-moment-bridge.yml` | `ubuntu-latest`, runner `python3` | `companions/d2_moment_bridge_20261006/replay.sh` and `check.py` | as D2atom (gate at L14, companion runs L18–42) | as D2atom | **none** | as D2atom | **no** |
| D2square | `d2-square-schur.yml` | `ubuntu-latest`, runner `python3` | `companions/d2_square_schur_20261006/replay.sh` and `check.py` | as D2atom (gate at L17, companion runs L21–45) | as D2atom | **none** | as D2atom | **no** |
| torus | `torus-filtration.yml` | `ubuntu-latest`, runner `python3` | `companions/torus_filtration_20261011/replay.sh` and `check.py` | full regex (`check.py` L62, L678–681), applied in `finish` to `toolchain.log`. That log is captured before the Lean runs (`replay.sh` L37), and there is no formal gate. | `packages.log` is captured before the Lean runs (L39–42) and checked in `finish` (L822) | **none** | `source` at start; `finish` checks evidence membership | **no** |

## 3. Producer/consumer binding table (what moves together)

| change | what must rotate in the same commit | precedent |
|---|---|---|
| anything under `formal/` (`gate.py`, tests, `manifest.json`) | `formal/manifest.json` file hashes; the `CORE_TREE`, `CORE_MANIFEST` and `CORE_GATE` constants in each of the three D2 `check.py` files; each D2 `SOURCE_FILES.json` (`core_tree`, and the `check.py` git blob); moment's `test_bridge.py` L52 literal `38b6d1a7…`; the formal tests that carry those literals | Math-#412 (`de604b84`): 18 files, including five formal test literals and `+159` controls in `formal/tests/test_gate.py` |
| D2 `replay.sh` or `check.py`, or its workflow | that companion's `SOURCE_FILES.json` git-blob entries. Each D2 `SOURCE_FILES.json` pins its own workflow, `check.py`, `replay.sh` and tests. | #412 |
| torus `replay.sh`, `check.py` or workflow | torus `SOURCE_FILES.json` git-blob entries. If a new evidence file is added: `EVIDENCE_FILES` membership, the `len(EVIDENCE_FILES) != 72` guard (`check.py` L364–365) and the tests. Also completed-log hashes and replay/check identities, with `O_EXCL`/`O_NOFOLLOW` receipt creation preserved. | — |
| Cap `gate.py` or `test_gate.py` | `MANIFEST.json` `files[].sha256` for both. `MANIFEST.json` pins all nine package files. | — |
| spec or specproof `check.py` or `test_check.py` | only their own package's identity. Specproof pins the statement packet's modules (`statement_module_sha256`), not the statement `check.py`. Both checkers require their workflow to be tracked and clean (spec L424–428, specproof L639–643) but do not pin its bytes. | — |
| `tools/required_formal_check.py` (the required-check consumer) | the formal receipt producer's field names, if fields are added. It sits outside `formal/`, so it does not move the D2 core bindings. It currently validates only `checked_commit`, `repository`, `workflow_run_id`, `formalization_status`, `scientific_effect`, `manifest_sha256` and log hashes (L50–73). It does **not** validate `lean_version`, `dependency_revisions` or a run attempt. | — |

## 4. Runtime matrix

The rows below come from the latest successful hosted run of each workflow, read from its job log on 2026-10-13. No newer completed run of any outcome exists for these eight workflow files. **L\*** is this exact string: `Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`. All 17 Lean version lines found are byte-identical to it.

| path | run / attempt, job | event, branch, head | runner image | arch | Python | Lean version as printed, and where | nine dependency HEADs | dependency cleanliness |
|---|---|---|---|---|---|---|---|---|
| formal | 38444399262 / 1, job 115458892756 (`formal / formal-evidence`, called by `downstream-gate.yml`) | push, `main`, `9844f785` | `ubuntu-24.04`, image 20261004.327.1, Ubuntu 24.04.5 | NOT_OBSERVED (the Lean target is x86_64) | NOT_OBSERVED (runner `python3`) | L\*: once at the lean-action install, before the builds; once in the receipt `lean_version`, after the controls | MATCH 9/9 (receipt and Lake checkout lines) | NOT_OBSERVED |
| Cap | 38177115116 / 1, job 114581722633 | pull_request, `codex/formal-retention-90d-20261011-c266`, `f3c1a3cd`, merge `4b9e372` (no `main` runs; PR and dispatch triggers only) | `ubuntu-24.04`, image 20261004.327.1 | x64 (setup-python archive) | CPython 3.11.16 | L\* only at the lean-action install. The gate itself prints no version, and its stdout summary has no version or dependency fields. | MATCH 9/9 (Lake checkout lines only) | NOT_OBSERVED in the log. The receipt's `dependency_worktrees` is written to the artifact, not printed. |
| spec | 38350897827 / 1, job 115142809977 | pull_request, `claude/lean-statements-round7-side24b-20261013`, `f6ba2e6a`, merge `ce3b69d3` (no `main` runs) | `ubuntu-24.04`, image 20261004.327.1 | x64 | CPython 3.11.16 | L\* only at the lean-action install. The execute step prints `version: exit 0`, and its stdout summary omits `lean_version`, which `receipt.json` carries (L1070). | MATCH 9/9 (source-check receipt and Lake lines) | `"worktree": "CLEAN"` is the repository packet. `dependency_worktrees` is in `receipt.json` but not printed. |
| specproof | 38350897766 / 1, job 115142809810 | as spec | `ubuntu-24.04`, image 20261004.327.1 | x64 | CPython 3.11.16 | as spec (summary omits `lean_version`; `receipt.json` L1290 carries it) | MATCH 9/9 | as spec |
| D2atom | 38119182355 / 1, job 114410185240 | push, `main`, `2bbce4b6` | `ubuntu-24.04`, image 20261004.327.1 | NOT_OBSERVED | NOT_OBSERVED | L\* three times: lean-action install; core receipt `lean_version` after the core gate; and the bare `toolchain.log` line, written **before** the companion's Lean runs and never validated | MATCH 9/9 (core receipt) | NOT_OBSERVED |
| D2moment | 38119182482 / 1, job 114410185914 | as D2atom | as D2atom | NOT_OBSERVED | NOT_OBSERVED | as D2atom | MATCH 9/9 | NOT_OBSERVED |
| D2square | 38119182300 / 1, job 114410185070 | as D2atom | as D2atom | NOT_OBSERVED | NOT_OBSERVED | as D2atom | MATCH 9/9 | NOT_OBSERVED |
| torus | 38271813510 / 1, job 114887642848 | push, `main`, `a0323b94` | NOT_OBSERVED: the first 3,890 of 8,890 log lines are beyond this session's tool limit. The jobs API label is `ubuntu-latest`. | NOT_OBSERVED | NOT_OBSERVED | L\* in the final receipt `lean_version`, which `finish` copies from the pre-run `toolchain.log`. The install line is not readable. | MATCH 9/9 (receipt) | NOT_OBSERVED |

**What the matrix shows and does not show:**
- Every observed runtime agrees with the pins today.
- The gaps in section 2 are about what the guards would **catch**, not about a drift that has already happened.
- No log prints dependency cleanliness. No log prints a version observation taken after the last Lean execution, except formal's receipt for its own gate runs.
- Arch is printed only indirectly.
- Cap, spec and specproof have no `main` runs, so their baselines are PR runs.

## 5. Repair plan: a minimal uniform guard, in one frozen candidate

This follows the #412 shape and makes no theorem, premise, proof, import, toolchain or package-pin change.

1. **formal** (`formal/gate.py`):
   - Add G2 cleanliness and a preflight G2 HEAD check, using a `dependency_state` equivalent to Cap's, both before the first Lean run and after the last.
   - Record `dependency_worktrees` in the receipt.
   - Add new controls in `formal/tests/test_gate.py`: same semver with the wrong commit, a non-Release build, a dirty dependency, dependency HEAD drift between pre and post, and source drift.
   - Then rotate the full #412 cascade: the manifest, five test literals, D2 `CORE_*`, D2 `SOURCE_FILES` and the `test_bridge` literal.
2. **required-check consumer** (`tools/required_formal_check.py`):
   - Validate the receipt's `lean_version` against the G1 pattern.
   - Validate `dependency_revisions` equal to `formal/manifest.json`'s lock, and the new `dependency_worktrees == CLEAN`.
   - Validate a `workflow_run_attempt` field equal to `GITHUB_RUN_ATTEMPT`, which the producer would add.
   - A missing or mismatched new field fails closed. Legacy receipts stay historical and are not re-validated.
3. **Cap** (`gate.py`):
   - Replace the semver prefix check with the full G1 pattern.
   - Add the post-execution G1 check and a post-execution `source_check` equality check, the G3 after-check.
   - Add `test_gate.py` controls: same semver with the wrong commit, a non-Release build, and source drift.
   - Rotate `MANIFEST.json`'s two hashes.
4. **spec and specproof** (`check.py`):
   - Add the post-execution G1 check and `dependency_state` (G2) after the negative controls and before the receipt.
   - Add `test_check.py` controls for drift between pre and post.
5. **D2, all three** (`replay.sh` and `check.py`):
   - After the last Lean run inside the `lake env` block, write `toolchain-final.log` and a `packages-final.log` holding HEAD plus porcelain for the nine packages.
   - `finish` validates the pre `toolchain.log` and both final logs against G1 and the formal lock.
   - Rotate `SOURCE_FILES`. This merges with step 1's rotation, so each file rotates once.
6. **torus** (`replay.sh` and `check.py`):
   - Add the same final observation after the last control loop, with the cleanliness check added to the package record.
   - `finish` validates the pre and final observations.
   - `EVIDENCE_FILES` goes from 72 to 74, with the count literal and tests updated together.
   - Rotate `SOURCE_FILES`.
7. **Acceptance matrix 1, negative controls.** Every new check gets a unit test that fails on the old code and passes on the new one, in normal and `-O` Python. The full list: same semver with the wrong commit, a non-Release build, dependency HEAD drift or a dirty tree during execution, source drift, and a consumer accepting a preflight-only or earlier-captured observation when the runtime changed later. These run through each tool's own CLI, not mirror-only tests.
8. **Runtime records, acceptance matrix 2.**
   - The candidate's hosted runs record baseline and candidate rows for every affected path.
   - Local runs use an explicit `LOCAL:` provenance namespace and never a GitHub run identity.
   - A missing row is `NOT_OBSERVED`.

**Not in scope:** other repositories and unmerged branches, which get an enumerated inventory only; Math-#451's held frontier; the `ubuntu-latest` move to Ubuntu 26; any runner or package upgrade.

## 6. Stage-2 preconditions

- **Fresh path-stop and claim re-scan:** run against the then-current Math- `main`. If `main` moved, rebase the map.
- **One frozen candidate:** a draft PR on its own branch, with head, tree, base and changed-path list.
- **Then:** one nonauthor engineering read by OpenAI `/root/coordination_sources`, current required checks, and integration by General `01a11cbf` after #391 closes. #391 is now closed, merged at `9eea35f3`.
- **Not done in any stage:** no dispatch, re-run, settings, permission, runner or schedule change.

## 7. What this packet does not establish

- It is a source reading plus log reading. It is not a runtime qualification, a receipt or a proof of anything.
- The runtime matrix shows what hosted logs printed. Unprinted fields are `NOT_OBSERVED`, not `PASS`.
- The repair plan is a proposal. Its cost, and the size of the formal cascade, need the reader's and integrator's judgement.
