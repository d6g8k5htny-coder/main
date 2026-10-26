# SIDE24 arithmetic skeleton — fresh statement-alignment review (#637 amend)

- Reviewer: provider **Cursor**, family **Composer**, agent **Align-review main#637 (bc-f08e74fc-3b41-50ab-aa90-481c13534904)**. Differs from Lean package author (Anthropic / Claude / Cursor cloud agent, 2026-09-27), from amend writer (Anthropic / Claude 01Cz on #637), and from informal author family (OpenAI). Not xAI a3 (#630) and not a12.
- Package author under review: Anthropic / Claude / Cursor cloud agent, 2026-09-27 (Lean + original anchors). Amend under review: Anthropic Claude 01Cz Draft [#637](https://github.com/d6g8k5htny-coder/main/pull/637) only.
- Reviewed commit: `6c8af70d594f1289eda266f43ebf3f388881ab08` (d6g8k5htny-coder/main, branch `claude/side24-anchor-amend-20261026`)
- Base observed for the amend diff: `9271adf5bece327639ba9e6ba636d850c4120806`
- `formal/manifest.json` SHA-256: `28ba3d76f7bab69cffa95db920c134ce1a0e301f5a1e13a531e17d01970a44ed` (recomputed at reviewed head; matches writer claim)
- `formal/SCOPE.md` SHA-256: `fd06700628b1fbc81c5cfe3786aaa455032375bcb879914f068a66b726802a17` (recomputed; equals manifest `files` entry; unchanged vs #630 / `8dfcfe4e` package)
- All 17 `manifest.files` digests recomputed at the reviewed commit: all match (including `formal/sources/side24_v1/PROOF.md` c06dacc… and `ENCLOSURE.json` 72b6cd9…).
- Prior rejected record: #630 @ `a228f441` / md sha256 `78c21c96…`, bound to manifest `8dfcfe4e…`. That record is **not edited**. This is a fresh review of the amended package.
- Lane: statement correspondence only. Proof correctness (kernel) and Layer 0 mathematics are **not** graded. credit 0 · sci NONE.

## Contender check

At pickup (~01:35Z 2026-10-26 UTC): no PICKUP/REVIEW/RESULT for ALIGN/#637 on #637, #388, #619, #539, or #475 after writer RESULT [388/6405958925](https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6405958925) (01:07Z). a12 STBY per CoS [475/6405924069](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6405924069) item 2 had not posted a RESULT on #637. Seat taken as different-family reviewer (Cursor/Composer ≠ Claude, ≠ OpenAI).

## Amend diff (head vs base)

Exactly two files, +6/−6:

- `formal/manifest.json`: only targets `UniversalLaw.Side24.d3_below_d2` and `UniversalLaw.Side24.endpoints_in_unit_tenth` change (`informal_anchor` + `does_not_claim`).
- `formal/ALIGNMENT.md`: regenerated rows for those two targets only.

Unchanged vs base (and vs the #630-reviewed `8dfcfe4e` objects for the other 29 targets): every Lean module, `SCOPE.md`, both pinned sources, every `files` digest, and the full `pairing_sum_six` target object (byte-identical).

## Source exposure

Statements and docstrings read first for all 31 targets. For the two amended rows and for `pairing_sum_six`, the informal anchors were located as verbatim substrings in the pinned local copies under `formal/sources/side24_v1/` before the verdict. All proof bodies remain the single tactic `by decide` (or `decide` under a binder); they were seen and not used for any verdict. Cross-checks for the amended rows used exact integer comparison of the `10^-20`-unit transcriptions against ENCLOSURE decimals.

## Gate output

At reviewed head `6c8af70d`, this reviewer ran `python3 tools/formal_gate_check.py` (no Lean build): `SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): 28ba3d76f7bab69cffa95db920c134ce1a0e301f5a1e13a531e17d01970a44ed`, with `alignment_status: PENDING_INDEPENDENT_REVIEW`, 31 targets and 2 sources, exit 0. Verbatim-substring anchor check passes for the new dimensions anchors. No `--run-lean` workflow run is cited here. This ACCEPTED record, when validated with `--alignment` against a tree whose manifest digest is still `8dfcfe4e…`, is correctly refused as stale until the amended manifest lands; that refusal is not a scientific disposition.

## Per-target verdicts

Tally: ALIGNED 8 · ALIGNED AT NARROWER SCOPE 23 · MISALIGNED 0 (total 31).

"Narrower" means: the Lean target is only the arithmetic identity/inequality behind the anchor; the analytic step named in `does_not_claim` and SCOPE "Not established" is not claimed. Targets 1–29 below reuse the correspondence established against unchanged Lean/anchors/SCOPE relative to #630, after this reviewer confirmed those 29 manifest target objects are byte-identical to the pre-amend package and re-read each Lean statement name against its title. Targets 30–31 and `pairing_sum_six` were freshly checked end-to-end against ENCLOSURE/PROOF and Lean.

| # | Target | Module | Verdict | Notes |
|---|---|---|---|---|
| 1 | `UniversalLaw.Side24.pairing_sum_six` | Ledger.lean | **ALIGNED** | Fresh re-check: target object untouched vs base/`8dfcfe4e`. Anchor `"The coefficient sum for q=6 is\n1+15+45+15=76"` is a verbatim substring of `PROOF.md`. Lean `pairingSum 6 = 76`; recomputed terms 1,15,45,15 and sum 76. Derivative bound excluded (`does_not_claim` / SCOPE). |
| 2 | `UniversalLaw.Side24.pairing_sum_le_six` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged vs #630 package; Lean ∀q∈{0..6}, pairingSum q ≤ 76; analytic ∣x∣≥1 bound not claimed. |
| 3 | `UniversalLaw.Side24.pairing_terms_six` | Ledger.lean | **ALIGNED** | Unchanged; four terms j=0..3 at q=6 match anchor display order. |
| 4 | `UniversalLaw.Side24.sixth_moment_double_factorial` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; Lean 1·3·5=15; contraction bound excluded. |
| 5 | `UniversalLaw.Side24.lattice_shell_constant` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 729 and 1458 match; counting/tail excluded. |
| 6 | `UniversalLaw.Side24.geometric_ratio_constants` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 512 and 864 match; `<1/2` not claimed. |
| 7 | `UniversalLaw.Side24.period_exponent` | Ledger.lean | **ALIGNED** | Unchanged; 24²/2=288. |
| 8 | `UniversalLaw.Side24.image_constant_value` | Ledger.lean | **ALIGNED** | Unchanged; 1458·(76·24^6+15)=21175738586478; inequality (2) excluded. |
| 9 | `UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; cleared partial sum >10; partial`<`exp excluded. |
| 10 | `UniversalLaw.Side24.covariance_relative_bound` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 60E scaling; spectral/PSD excluded. |
| 11 | `UniversalLaw.Side24.odd_block_eigenvalues` | Ledger.lean | **ALIGNED** | Unchanged; 8±√58 via trace/det. |
| 12 | `UniversalLaw.Side24.odd_block_shifted_minors` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; minors 2 and 7; Sylvester excluded. |
| 13 | `UniversalLaw.Side24.smaller_eigenvalue_exceeds_third` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 58·9<23²; √ monotonicity excluded. |
| 14 | `UniversalLaw.Side24.exponent_bounds` | Ledger.lean | **ALIGNED** | Unchanged; d∈{2,3} bounds. |
| 15 | `UniversalLaw.Side24.exponent_values` | Ledger.lean | **ALIGNED** | Unchanged; 6a/6b definitions. |
| 16 | `UniversalLaw.Side24.ratio_constant_ordering` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 13≤32, 28≤32; log/exp excluded. |
| 17 | `UniversalLaw.Side24.density_comparison_below_reported` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 32·10^106<10^108; density comparison excluded. |
| 18 | `UniversalLaw.Side24.conditional_third_derivative_variance` | ConeMoments.lean | **ALIGNED** | Unchanged; τ²=6 over Int. |
| 19 | `UniversalLaw.Side24.transverse_covariance_entries_thirds` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; thirds entries; delta evaluation excluded. |
| 20 | `UniversalLaw.Side24.cone_moment_m1_thirds` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; D_1=4/3; symmetry excluded. |
| 21 | `UniversalLaw.Side24.trace_and_traceless_variances` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; Var s=5/3, Var x=1, Cov=0; independence excluded. |
| 22 | `UniversalLaw.Side24.fourth_moment_of_trace` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 25/3 identity arithmetic; Gaussian fourth-moment excluded. |
| 23 | `UniversalLaw.Side24.exponential_moment_of_trace` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 1+5/3=8/3; exp-moment/root excluded. |
| 24 | `UniversalLaw.Side24.cone_moment_m2_algebra` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 20/29/√6 arithmetic; integral/Rayleigh excluded. |
| 25 | `UniversalLaw.Side24.cube_root_simplification` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; 6²·2=3·24; real cube roots excluded. |
| 26 | `UniversalLaw.Side24.pin_determinant_and_joint_dimension` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; det=3; dims 6/10; covariance structure excluded. |
| 27 | `UniversalLaw.Side24.hessian_block_eigenvalues` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; d+2 and traceless 2 vs 1/3; block decomposition excluded. |
| 28 | `UniversalLaw.Side24.d2_endpoints_adjacent` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; d=2 endpoints match ENCLOSURE digit-for-digit; U=L+1; enclosure of true c_{2,24} not claimed. |
| 29 | `UniversalLaw.Side24.d3_endpoints_adjacent` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | Unchanged; d=3 endpoints match; U=L+1; enclosure not claimed. |
| 30 | `UniversalLaw.Side24.d3_below_d2` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | **Fresh after #637 amend.** Prior #630 MISALIGNED (anchor was `"scientific_acceptance": false`). New `informal_anchor` is the verbatim ENCLOSURE.json `"dimensions"` block through the d=3 upper field (confirmed `anchor in ENCLOSURE`). Lean `d3Upper < d2Lower` with Nat transcriptions 4177593184059834335 < 7340691930603427103 (true). Title/SCOPE cover published-interval ordering; `does_not_claim` names the ordering as a derived artifact-consistency check and excludes true coefficients. Narrower: no claim that either coefficient lies in its interval (SCOPE "Not established"). |
| 31 | `UniversalLaw.Side24.endpoints_in_unit_tenth` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | **Fresh after #637 amend.** Prior #630 MISALIGNED (anchor was `"relative_periodization_bound": "1e-106"`). Same dimensions-block anchor (verbatim in ENCLOSURE). Lean `0 < d3Lower ∧ d2Upper < 10^19` (10^-20 units ⇒ published endpoints in (0,1/10)); both conjuncts true. `does_not_claim` names the range check as derived artifact-consistency. Narrower: true coefficients not claimed. |

## Disposition

**ACCEPTED** (package). 0 MISALIGNED. 8 ALIGNED · 23 ALIGNED AT NARROWER SCOPE. The two #630 AMEND rows are corrected at the anchor level on #637 and now correspond. JSON `disposition` is `ACCEPTED` per gate contract (narrowing recorded in the Markdown rows). No Lean statement was found to overclaim relative to SCOPE. `alignment_status` in the manifest is **not** flipped here and remains `PENDING_INDEPENDENT_REVIEW` until a controller consumes an accepted record; this file is evidence only.

## Boundaries

No manuscript, PROOF_INDEX, STATUS, manifest `alignment_status`, Lean, source, #630 record, or #637 source is edited. This review PR adds only `audits/formal_alignment/` files. Promotes nothing. credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN · Skip #603.

— Cursor Composer, Align-review main#637 (`bc-f08e74fc-3b41-50ab-aa90-481c13534904`)
