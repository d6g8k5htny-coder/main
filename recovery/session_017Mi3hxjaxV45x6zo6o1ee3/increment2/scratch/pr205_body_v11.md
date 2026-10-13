**Object** `CL-C8-WINDOW-COEFFICIENT-D3-TORUS-20260930-v1` (v1.2). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#197 (comment 5921235693). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

Math-#197 gives the compact-window coefficient `c_{B,K}` of [LP] Theorem B in the plane, with its torus transfer, and in `d = 3` only the reference closed form; its §4 says "no torus transfer is made in `d = 3`", because under the perturbed entries the conditional mean of the transverse Hessian given `f = b, V = 0` is no longer `−b I` and the truncated cone moment has no closed form. This record supplies that transfer.

**Lemma S (window-rescaling sandwich).** The birth integral is `I_B = E[1{f ∈ B} det(A)² 1{A ≺ 0} | V = 0]`, an expectation of a nonnegative function under the centred 4-dimensional Gaussian `(f, A_11, A_12, A_22) | V = 0`. If `(1−ε) C'_ref ≤ C' ≤ (1+ε) C'_ref`, then

`(1−ε)⁴/(1+ε)² · I^ref_{B/√(1−ε)} ≤ I'_B ≤ (1+ε)⁴/(1−ε)² · I^ref_{B/√(1+ε)}`,

by the Gaussian density comparison and the degree-4 homogeneity of `det(A)²`: the birth window, the only non-homogeneous ingredient, is merely rescaled. #197's closed form (4.1) evaluates `I^ref` at the interval endpoints `b_±/√(1 ± ε)`. This is SIDE24 §4's scaling argument extended from the full window to a birth window.

**The certified ε.** The exact reference Schur complement `C'_ref` (rationals) has smallest eigenvalue `2 − √(8/3) = 0.367`; `C'_ref ≥ I/3` is checked exactly by Sylvester's criterion. So `ε = 3‖C' − C'_ref‖_F` over the entry box of the `d = 3` image bound `E^(3)_L = 1404(76L⁶ + 15)e^{−L²/2}` (shells `(2j+1)³ − (2j−1)³ ≤ 26j²`, `|n|² ≤ 3j²`), with `C'` by interval linear algebra. `p_G(0)`, `p_V(0)` and `τ²` are interval-evaluated over the same box. `ε(L ≥ 10) ≤ 5.972e-10`; at `L = 24` it is arithmetic-limited.

## Values

Displayed intervals are rounded outward (v1.1) and contain the certified intervals of `RESULTS.json`.

| window | reference `c^(3)_{B,K}` | torus, every `L ≥ 10`, every frame |
|---|---|---|
| `B = [0,1]`, `K = [1/2,2]` (C8 band) | `0.0009743823660242504624710565…` | `[0.00097438236231621369, 0.00097438236973228725]` |
| `B = R`, `K = [1/2,2]` | `0.0028145169435725674316…` | `[0.00281451693155790684, 0.00281451695558722807]` |
| `B = R`, `K = (0,∞)` | `0.0417759318405983433429…` (= `c_{3,ref}`) | `[0.04177593168364896978, 0.04177593199754771747]` |

Relative half-width at most `4.3e-9` at `L ≥ 10` on all twelve windows (NOTE §5). At `L = 24` every enclosure agrees with the reference to about 55 digits, and the full window lies inside SIDE24's `c_{3,24}` interval.

## Checks

Fifteen rules: `REFERENCE_EXACT` (the exact `C'_ref`, `A | f = b ~ −bI + Q`, `τ² = 6`), `LAMBDA_FLOOR`, `D2_EXACT` (full window `= 29/6 − √6`), `M3_FLOAT`, `IMAGE_BOUND`, `EPSILON`, `SANDWICH_EXACT_SCALING` (for `C' = (1+η)C'_ref` the exact value `(1+η)² I^ref_{B/√(1+η)}` lies inside its own sandwich, `η ∈ {±1/50, ±1/200}`, five windows including `[3, 7/2]` and `[−7/2, −3]` where the rescaling dominates), `FLOAT_CONTROL` (a non-scalar perturbation that tilts the conditional mean; float quadrature inside the sandwich; the same code reproduces `I^ref` to `2e-9` relative), `FACTORIZATION`, `SIDE24_CONSISTENT`, `NESTING`, `WIDTHS`, `SANDWICH_ORDERED`, `LIBRARY_EXACT`, `PINNED`. Eleven mutants exit 1 in both modes (`image-shells`, `lambda-min`, `sandwich-swap`, `window-scale`, `d3-cross`, `schur-sign`, `sphere`, `tau-cross`, `tail-dropped`, `no-owen`, `gamma-shape`). About ten seconds per run.

## Files

`frontiers/c8_window_coefficient_d3_torus_20260930/{NOTE.md, window_coefficient_d3.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-d3-torus.yml`. Pins on `main 3e0a91b` (workflow-checked): [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0`. Companion (not on main): Math-#197 at `63e123f` (`window_coefficient.py` `43b4f51b` reused verbatim for the interval toolkit and Owen primitives; `NOTE.md` `56d99b48` §4; `RESULTS.json` `95443edf`).

## What this does not do

Not a review of [LP], SIDE24 or Math-#197. Not `d ≥ 4` window coefficients here (the `d = 4` case is taken up separately under the claim in comment 5922418823). Not `C`, `r_*` or `z_*`. C8 stays OPEN. **Windows outside [LP] Theorem B's hypothesis** (compact `B`, `0 < k_− ≤ k_+ < ∞`): for the tabulated `K = (0,∞)` and the unbounded `B` (`[0,∞)`, `(−∞,0]`, `R`), the values are the coefficient integrals (1.1) on those sets only; Theorem B's asymptotic statement is not extended to them (the full window is [LP] (15.2)'s `c_{3,L}`). The replay workflow follows the repository's per-packet convention; the owner's decision on such workflows (open on Math-#202/#204) applies here too.

## Current disposition

- **Head / owner:** `HEAD_PLACEHOLDER` (v1.2) on `claude/c8-window-d3-torus-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** Lemma S and its use; the `d = 3` image bound; the certified `ε`; torus enclosures of `c^(3)_{B,K}` for every `L ≥ 10` and every frame, and for `L = 24`. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes** (all OpenAI / Codex, personal owner reading pending, zero organizational-independence credit):
  - Codex code review at `b875a46` (no findings).
  - **C40 nonauthor read, Slices A and B: PASS_TECHNICAL, no actionable findings** (review 5373331587 at `b875a46`).
  - **C41 nonauthor read, Slice C: PASS_TECHNICAL** for the numerical assembly, full-precision `RESULTS.json`, the cone evaluator and the consumed primitive interface; **AMEND** on reporting only, finding C41-205-C-01 (review 5373681438 at `b875a46`).
  - **C41 read of the imported #197 toolkit and reference primitives: PASS_TECHNICAL, no finding** (Math-#197 review 5373681556).
- **Finding C41-205-C-01 (fixed in v1.1, `01d09e9`; awaiting the reviewer's bounded check):** the eight §5 upper endpoints and the §0 C8 interval were truncated downward, so the displayed intervals did not contain the certified ones. Every displayed interval is now rounded outward, matching the reviewer's eight corrected uppers digit for digit, and an exact containment audit against `RESULTS.json` passes for every bracketed interval in the NOTE. Two further reporting repairs went in alongside: the `τ²` and `p_G p_V` lines are now explicit outward intervals (the old `± 1.3e-13` line did not contain the certified upper endpoint), and the `ε` values are rounded-up bounds. The float-control accuracy is stated as relative. The script, `RESULTS.json`, the rules and the mutants are byte-unchanged (NOTE §10, `SOURCE_FILES.json` `revisions`).
- **Revision v1.2 (`HEAD_PLACEHOLDER`, NOTE only; awaiting a bounded check).** NOTE §0 said `k_− ≥ 0`. It now states [LP] Theorem B's hypothesis `0 < k_− ≤ k_+ < ∞`, and a new §7 bullet classes windows outside it (the tabulated `K = (0,∞)`, the unbounded `B`) as coefficient integrals only, as this PR body already did. This is the defect Codex found on Math-#213 (thread 4151593177), repaired in the whole series. The script, `RESULTS.json`, the rules and the mutants are byte-unchanged (NOTE §10, `SOURCE_FILES.json` `revisions`).
- **Unresolved finding IDs:** none open on the author side. C41-205-C-01 is confirmed fixed (C42). The v1.2 wording awaits a bounded check.
- **Validation:** 15/15 rules in `-B -S` and `-B -O -S`, byte-identical output, identical to the unchanged `RESULTS.json`; 11/11 mutants rejected in both modes; workflow replayed locally at `01d09e9`; hosted run on this head pending.
- **Next action:** a bounded nonauthor check of the v1.2 NOTE §0/§7 wording. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
