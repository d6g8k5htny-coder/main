# Round 7, second slice (Lifetime successor pair-law interface): design-review packet, DRAFT, not activated

Status: proposed on main#307 ([6137849133](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6137849133)),
awaiting the coordinator's routing. Nothing here is a packet change: Math- is untouched for this slice. This directory is
the exact object a design read would take, prepared so that routing → read → activation is short. Author: Anthropic Claude,
session `session_01Cz7WZybv8znP64SpPj6sWY` (author of the statement packet). Organizational-independence credit 0.
Scientific effect NONE. Every declaration would enter as `specified`; no theorem, no inhabitant constructed, no label moved.

## Files

| file | what it is |
|---|---|
| `Lifetime_append.lean` | the draft block to be appended to `Specifications/Lifetime.lean` before its `end` line: 13 auxiliaries, `structure PairLawInterface2 … extends PairLawInterface`, 3 `def … : Prop` successors (`ParentUniformSelectionA2`, `PlanarElderLowerE2B`, `PlanarTwoSidedRateE4B`), four-label docstrings, verbatim anchors |
| `Lifetime_draft.lean` | the full draft module (committed `Lifetime.lean` at `8acc4856` minus its last line + the block + `end`); prefix byte-identical |
| `elab_draft.log` | `lake env lean Lifetime_draft.lean` at the pin (Lean 4.34.1 `5045d005`, Mathlib `d13f23b7`) |
| `scan_draft.txt` | the statement gate's own `scan_module` on the draft: 55 declarations = the committed 38 (21 Props, 3 structures, 14 auxiliaries; identical) + 17 new (13 auxiliary, 1 structure, 3 `def_prop`), four labels and verbatim anchors on each new registered-kind declaration |
| `register_rows_proposed.json` | the statement rows that would be added (structure + `ParentUniformSelectionA2` under `math.uniform-matrix-cap-lifetime` and `D1-THM-A`, mirroring where `PairLawInterface` / `ParentUniformSelectionA` are registered; the E2/E4 successors under `OA-ELDER-LOWER-DENSITY-GAP-20260928-v1`), NOT applied |

## Declaration-by-declaration mapping (sources: `lifetime-parent` = Math- `c3916425…` `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`; `elder-lower-density-gap` = Math- `6c020d6a…` `frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md`)

| declaration | kind | source clause | formal shape | source's object or added assumption? |
|---|---|---|---|---|
| `PinIdx d` | aux | lifetime-parent:24 (2(d+1) observations) | `Fin 2 ⊕ (Fin 2 × Fin d)` | bookkeeping |
| `pinAt d r u` | aux | :20 `M=-(r/2)u, S=(r/2)u` | `0 ↦ pinM`, `1 ↦ pinS` | the module's pins |
| `pinObs`, `pinTarget` | aux | :21-22 (`f(M)=b, f(S)=b-k r^3, grad f = 0`) | value and `fderiv` functionals; target vector | the source's observations and targets |
| `dK`, `jetCov` | aux | the field's covariance `K_L` (`periodicCovariance`) | `iteratedFDeriv` contractions; stationary rule `Cov(∂_u f(x), ∂_v f(y)) = -∂_u∂_v K(x-y)` etc. | the stationary covariance rule, hand-written: **sign conventions are a reader item** |
| `obsPoint`, `pinObsCov`, `pinCross` | aux | :117 (`Sigma_r^{-1}`, `v_r`) | covariance matrix `Sigma_r` of the observations; cross-covariance `v_r(x)` | the regression's inputs, DEFINED by the rule; not shown to be covariances of any random vector |
| `regMean`, `regCov` | aux | :24 (continuous Gaussian regression) | `v(x)^T Sigma^{-1} target`; `K(x-y) - v(x)^T Sigma^{-1} v(y)` | Gaussian regression identities used as definitions (ordinary inverse: valid when `Sigma_r` is invertible, the field `pinObsCov_posDef`) |
| `hessMat`, `typedWeight` | aux | :26 `W_r = |det H_M det H_S| 1{H_M<0, index(H_S)=d-1}` | `Matrix.of (hessian …)`; indicator through `HessianHasInertia` times the product of absolute determinants | the source's `W_r` for the formal Hessian |
| `PairLawInterface2` | structure | :27 (`Z_r = E_Q W_r, dQ^W = (W_r/Z_r) dQ`), :24, :29 | `extends PairLawInterface` + `Q`, `Q_prob`, `pinObsCov_posDef`, `Q_regression`, `Q_pinned`, `Z_pos`, `p_def` (with `Q^W := Q.withDensity (W/Z)`) | the source's model as fields: `Q_regression` and `Q_pinned` are the definition of `Q`; `pinObsCov_posDef` (§3) and `Z_pos` ((5.5)) are the source's theorems assumed; `p_def` is the source's definition of `p` |
| `ParentUniformSelectionA2` | def_prop | :33 display (1.1) | predecessor Prop over the projection | same display, over the successor |
| `PlanarElderLowerE2B`, `PlanarTwoSidedRateE4B` | def_prop | elder-lower-density-gap:31 (E2), :46 (E4) | predecessor Props over the projection | same displays, over the successor |

Why the successor removes the known blocker: probe Lifetime-B_1 refuted the predecessor Props' universal closure with `p ≡ 0`
and `p ≡ 1`; under `PairLawInterface2`, `p r b k R` is fixed by `p_def` in the pin regime, so those inhabitants no longer
exist. This says nothing about the truth of the displays.

## Caveats (each is in the docstrings)

1. `Q_regression` characterises `Q` through finite vectors of point evaluations only, not through sample-path jets; so
   `Q_pinned` stays a field (a jet-level regression field could replace it: a design choice for the reader).
2. Sign and scale conventions of `jetCov` (cross terms with derivatives) are hand-written; a slip there passes every gate.
3. `pinObsCov_posDef` (source §3, "Sigma_r^{-1} … uniformly bounded") and `Z_pos` ((5.5)) are theorems of the source assumed
   as fields; `does_not_claim` names them. Only positivity of the normalizer is assumed, not the floor `z_* r^2`.
4. Measurability of the elder event and integrability of the weight are not stated (Mathlib's measure and integral are total).
5. The axial direction `R 0` needs `[NeZero d]` (the structure and `ParentUniformSelectionA2` take it; `d = 2` for the planar
   Props has the instance).
6. Nothing identifies `Q` with the regression law of any field or constructs an inhabitant; the parent's own `p_r` is not
   identified with `p`.
7. `p_def` is stated only in the pin regime `0 < r < L` (as the predecessor's gated fields are); outside it `p` is only bounded.

## Verification performed (author, local, not a receipt)

- Statement gate `scan_module` on the full draft module with the packet's pinned sources: see `scan_draft.txt` (committed
  prefix identical; 17 new declarations; labels and anchors verified; anchors of the three successor Props are the
  quantitative displays with their tags, distinct as strings from the predecessors' anchors).
- Elaboration at the pin: `lake env lean Lifetime_draft.lean` from the built statement-packet checkout (Lean 4.34.1 `5045d005`, Mathlib `d13f23b7`, `LEAN_NUM_THREADS=2`): exit 0, no output (no error, no warning, no `sorry`), real 0m14.643s (`elab_draft.log`). A single-file elaboration of the draft module, not a build receipt.
- Not run (needs the packet mutation the routing has not authorised): `check.py` source mode with the rows applied,
  `test_check.py`, `--execute`, hosted workflows.

## Scope if activated (four files, append only)

`Specifications/Lifetime.lean` (append), `REGISTER.json` (the rows above and one sentence each in the three rows'
`not_established` / `next_step`), `README.md`, `SCOPE.md` (re-rendered: 143 → 146 Props, 10 → 11 structures, 132 → 145
auxiliaries on top of #447; frontier 66 → 69 with three open-target rows and the three predecessor rows pointing at their
successors). Serial after #447 (or stacked on it, with the same rebuild rule). No gate, import, toolchain or proof-packet
change. What it does not establish: nothing mathematical; the three successor Props stay `specified` and Mathlib-blocked
(Gaussian regression, Kac-Rice; the planar argument), weeks each.

## Bound candidate against landed Math- main `7f5f1c82` (2026-10-13, about 06:45Z; local only, nothing pushed)

After #447 landed, the slice was bound by `lvs/r7/bind_lifetime.py` (fail-closed: asserts the worktree at `7f5f1c82` clean,
reproduces the elaborated draft from the landed module plus the append block, makes the block's heading final, inserts the six
register rows and the three rows' sentences, runs the gate's `check_register` in memory, reproduces the committed README/SCOPE
byte-for-byte from the committed register before rendering, applies the frontier edits with every string asserted unique) into
a scratch worktree. The resulting four files, as `git diff` against `7f5f1c82`: `candidate_on_7f5f1c82.patch`
(sha256 `54993c7f4a71b118816e8fd7d05f6cda6b6d8f692ac8158bd35fcca13d498c2d`), 4 files, 366 register statement entries (360 + 6), 157 registered declarations
(146 Props, 11 structures, 145 auxiliaries), frontier 66 -> 69 (open targets 2 -> 5).

| file | bytes | sha256 |
|---|---|---|
| `Specifications/Lifetime.lean` | 70447 | `75943ed33570150ea58b6c12b9192eeeb04cf3241648298d5869da555cfa6211` |
| `REGISTER.json` | 634186 | `391e86bb05750bfb6357aceae1212cbbf551be46f8b1b3ec93171d8193539172` |
| `README.md` | 108130 | `668b4ca0c4a99b0c97c0327afb50326c097d24534f0a044d3d301559dcbc36ca` |
| `SCOPE.md` | 295192 | `1fd40520b60424fd03460de4ea19869d0794bc908f0dce5b8cfe2fc9222a6cae` |

Verification on the bound worktree (author, local, not a receipt): `check.py source --allow-dirty` exit 0 (366 statements, 11
structures, 157 registered local statements, 23 sources verified in git history; worktree UNVERIFIED by construction);
`test_check.py` OK in both interpreter modes; `lake env lean` of the bound `Lifetime.lean` at Lean 4.34.1 `5045d005` / Mathlib
`d13f23b7` exit 0, no output (`elab_bound.log`, real 1m17s). Not run: `--execute`, hosted workflows (they run on the PR).
On activation the publish is: apply the patch to a branch off `7f5f1c82` (or re-run the binder), commit, push, open the draft PR,
bind the hosted runs. Nothing here moves a label; every new declaration enters `specified`.
