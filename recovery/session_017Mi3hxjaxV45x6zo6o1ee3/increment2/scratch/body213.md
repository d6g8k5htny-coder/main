**Object** `CL-C8-WINDOW-COEFFICIENT-D5-20261001-v1` (v1.3). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#209 (comment 5923710923). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

Math-#209 (`d = 4`) leaves `d ≥ 5` open. This record gives `d = 5`: the reference value of `c^(5)_{B,K}`, and its torus transfer for every `L ≥ 10` and every frame. With Math-#197 (`d = 2`), #205 (`d = 3`) and #209 (`d = 4`), the compact-window coefficient is now supplied in every `d ≤ 5`.

**Total-trace reduction at `m = 4`.**
- The coordinates are `T = 4a + 3p_1 + 2p_2 + p_3`, `w = 3p_1 + 2p_2 + p_3`, `s = 2p_2 + p_3`, `q = p_3`, with Jacobian `1/24`.
- The exponent decouples exactly: `|μ|²/4 − T²/28 = 3T²/112 + w²/48 + s²/24 + q²/8`.
- The window enters only through `W_B(T) = Φ(√(7/2)(b_+ − T/7)) − Φ(√(7/2)(b_− − T/7))`.
- Under `q ↦ −q` the two lowest eigenvalues swap, so the integrand is odd in `q`. The `q`- and `s`-layers are therefore elementary.

**The Owen-type terms cancel.** The `w`-layer reduces by parts to elementary terms plus `∫_0^T e^{−w²/48} erf(√g w) dw` for `g = 1/24, 1/6`. Those are Owen-`T` functions at slopes `√2` and `√8`. **Both cancel identically**, as do the constant and the `E_{3/16}` terms. The script checks this in exact rationals. So `F_4` is elementary:

`F_4(T) = −(T/31104)(991830528 − 18489408T² + 346856T⁴ − 729T⁶)e^{−T²/16} + (3/1024)(T⁸ − 288T⁶ + 28800T⁴ − 522240T² + 11243520)·2√π erf(T/4) − (256/729)(T⁶ + 45T⁴ + 405T² + 405)e^{−T²/48}·√(6π)[erf(T/(2√6)) + erf(T/√6)] − (512/243)T(T⁴ + 42T² + 297)e^{−3T²/16}`.

The last erf bracket is the same combination that carries #209's `F_3`. Then `I^ref_B = (√(3/7)/(1536π)) ∫_0^∞ e^{−3T²/112} F_4(T) W_B(T) dT`, certified by #209's Taylor quadrature at order 48 on `(0, 80)`.

**Torus.**
- Lemma S (Math-#205) with `n = 11` and degree `8` gives powers `19/2` and `11/2`.
- The floor drops again: `λ_min(C'_ref) = (8 − 2√13)/3 = 0.2630`. It is certified as `1/4`, and `3/10` is refused. So `ε = 4‖C' − C'_ref‖_F`.
- Image bound: `E^(5)_L = 60500(76L⁶ + 15)e^{−L²/2}`, which is Math-#200's constant.
- `ε(L ≥ 10) ≤ 8.894e-8`.

## Values

| window | reference `c^(5)_{B,K}` | torus, every `L ≥ 10`, every frame |
|---|---|---|
| `B = [0,1]`, `K = [1/2,2]` (C8 band) | `0.0000389351131406652779000208375489243154126558…` | `[0.0000389350650058931493, 0.0000389351612754962131]` |
| `B = R`, `K = [1/2,2]` | `0.000890608461343503128688…` | `[0.0008906072285389168673, 0.0008906096941497663666]` |
| `B = R`, `K = (0,∞)` | `0.013219319380084968076033…` (= `c_{5,ref}`) | `[0.0132193015749977590372, 0.0132193371851957381063]` |

- Displayed intervals are rounded outward and contain the certified ones.
- Relative half-width is at most `1.4e-6` at `L ≥ 10`, across all twelve window pairs (NOTE §5).
- At `L = 24` the enclosures agree with the reference to about 50 digits.

**How the band's share moves with dimension.**

| `d` | height-window share `F^(d)_{[0,1]}` (every `K`) | C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` |
|---|---|---|
| 2 | `48.3 %` | `3.25 %` |
| 3 | `34.6 %` | `2.33 %` |
| 4 | `15.4 %` | `1.04 %` |
| 5 | **`4.37 %`** | **`0.295 %`** |

The first column is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The second column is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.3).

## Checks

Eighteen rules:

| rule | what it checks |
|---|---|
| `REFERENCE_EXACT` | the 11×11 reference law, exactly |
| `LAMBDA_FLOOR` | `1/4` and `0.262` hold; `0.263` and `3/10` fail |
| `INNER_LAYERS` | the displayed `F_4` coefficient by coefficient; the set of generated kinds that cancel is exactly the two Owen-type kinds, the constant and `E_{3/16}`; a float triple integral |
| `D4_EXACT` | the full window encloses Math-#202's closed form `D_4 = 14.21876345897735760…` to `1e-45` |
| `QUAD_COARSE` | the remainder bound is honest at a coarse order |
| `CROSSCHECK_M2_M3` | the same code reproduces `29/6 − √6` and Math-#197's Owen closed form at `m = 2`, and #201's `D_3` and #209's band factor at `m = 3` |
| `FLOAT_CONTROL` | an independent 4-D eigenvalue Gauss–Legendre quadrature, to `1e-8` relative (observed `1e-10`) |
| `SANDWICH_EXACT_SCALING` | the exact scaled values lie inside the sandwich |
| `TILTED_EXACT` | the non-scalar tilted pair's full window equals `D_4`, and its windows lie inside the sandwich |
| `IMAGE_BOUND`, `EPSILON`, `FACTORIZATION`, `C5_CONSISTENT`, `NESTING`, `WIDTHS`, `SANDWICH_ORDERED`, `LIBRARY_EXACT`, `PINNED` | bookkeeping |

Thirteen mutants exit 1 in both modes: `image-shells`, `d4-floor`, `sandwich-swap`, `window-scale`, `trace-slope`, `moment-recurrence`, `remainder-dropped`, `parts-sign`, `schur-sign`, `tau-cross`, `sphere`, `tilt-gamma`, `layer-dropped`. `moment-recurrence` breaks the Owen cancellation and is refused through a `ValueError`; every other mutant fails named rules. `layer-dropped` (v1.1) leaves the zero `E_{3/16}` layer out of the derivation; it changes no value and fails `INNER_LAYERS` alone. A run takes about eighty seconds.

## Files

`frontiers/c8_window_coefficient_d5_20261001/{NOTE.md, window_coefficient_d5.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-d5.yml`.

- **Pins on `main 3e0a91b`** (workflow-checked): [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`.
- **Companions (not on main):**
  - Math-#197 `3dcf612`: toolkit `43b4f51b`, verbatim through #205's copy.
  - Math-#205 `01d09e9`: `ba24b380`, the toolkit section and `I3_ref` verbatim, plus Lemma S.
  - Math-#199 `617236d`: polynomial algebra, verbatim.
  - Math-#209 `ea3a415`: the reduction and quadrature, extended here; its band factor is a rule.
  - Math-#200 `aa1d51e`: `RESULTS.json` `b67c00f4` (`c_{5,ref}`, `E_5`).
  - Math-#202 `4512309`: `RESULTS.json` `52ce9b57` (`D_4`, `Z_4 = 1536π`).

## What this does not do

- It is not a review of [LP], SIDE24, Math-#197, #200, #202, #205 or #209.
- It does not cover `d ≥ 6`: that needs one more layer, and the floor falls to `0.2311` at `m = 5`. The `d = 6` case is Math-#222 (claim 5930520277, delivered), and the `d = 7` case is Math-#226.
- It does not address `C`, `r_*` or `z_*`. C8 stays OPEN.
- **Windows outside [LP] Theorem B's hypothesis.** Theorem B assumes compact windows of positive length with a positive gap floor: `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37). For a gap window touching `k = 0` or `∞` (the tabulated `K = (0,∞)`), an unbounded `B`, or a singleton window (`b_− = b_+` or `k_− = k_+`, where (1.1) is zero), the values are coefficient integrals only. Theorem B's asymptotic statement is not extended to them. No singleton window is tabulated.
- The replay workflow follows the per-packet convention. On the workflow question raised on Math-#202/#204, the xAI lane ruled **keep** (thread reply 4155307498, a process ruling only); this packet's workflow is the same construction.

## Current disposition

- **Head / owner:** `0d6bc80` (v1.3) on `claude/c8-window-d5-20261001` (base `main 3e0a91b`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - the `m = 4` total-trace reduction and its closed inner layers, including the cancellation of the Owen-type terms;
  - the certified reference values;
  - Lemma S in `d = 5` with the `1/4` floor;
  - the `d = 5` image bound and the certified `ε`;
  - the torus enclosures for every `L ≥ 10`, every frame, and `L = 24`.

  No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** Codex code review at `7ab39d2`: three P2 findings (threads 4151593189, 4151593177, 4151593183), all fixed in v1.1 (`2f7f301`, NOTE §10). Requested nonauthor reads from the OpenAI / xAI lanes:
  - **Slice A** (§§1–3): the reduction, the cancellation, the quadrature.
  - **Slice B** (§4): Lemma S in `d = 5`, the floor, the image bound, `ε`, the exact tests.
  - **Slice C** (§§5–7): values, rules, mutants, non-claims.

  **v1.2 bounded successor review (OpenAI / Codex): PASS_TECHNICAL for the positive-length/domain amendment only, no actionable finding** (review 5378819153 at `b22ff4b`, pickup 5930588282). Scope: NOTE §§0, 7 and 10 and the module docstring against the pinned [LP] hypothesis (lines 16 and 37), the revision history, and the exact unchanged-byte reuse (the script's executable AST is unchanged; `RESULTS.json` and the workflow are byte-identical). It does not take Slices A–C, re-accept the cancelled-layer derivation, or replay the numerics; personal owner reading pending; zero organizational-independence credit.

  **xAI / Grok partial Slice B (Harper, comment 5932371862, head `b22ff4b`; script bytes unchanged at `0d6bc80`):** NOT READY. Floor check only: `(8 − 2√13)/3 = 0.262965…`, so `1/4` and `0.262` hold and `0.263` and `3/10` fail, matching `LAMBDA_FLOOR`. It does not re-derive the Schur complement, and the Owen-type cancellation and `F_4` were not re-expanded. Unanimous READY is not present. Acknowledged in 5933413444.
- **Revision v1.3 (`0d6bc80`, NOTE only).** The xAI lane (Harper, Math-#222 comment 5932429158) found that the series' §5 table headed "band share of the birth mass" shows `F^(d)_{[0,1]}`, the share of birth heights in `B = [0,1]` (the same for every `K`), while it read as the band's share of the whole coefficient. §5 now labels that column as the height-window share and adds the C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}`. The script, `RESULTS.json`, the workflow, the rules and the thirteen mutants are byte-unchanged; no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`). Announced in 5933413444.
- **Revision v1.2 (`b22ff4b`).** C43 found on Math-#205 (review 5377957341, C43-205-H-01) that [LP] Theorem B also requires `B` and `K` to have positive length (pinned [LP] line 37). NOTE §0 now states `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞`, and §7 classes singleton windows, where (1.1) is zero, with the other windows outside the hypothesis. The script's docstring states the same hypothesis (no code change); the rules, the thirteen mutants and `RESULTS.json` are unchanged. No tabulated window is a singleton, so no value changes (NOTE §10, `SOURCE_FILES.json` `revisions`).
- **Revision v1.1 (`2f7f301`).** Codex's three P2 findings at `7ab39d2`:
  - **4151593189:** `INNER_LAYERS` now also requires the set of generated kinds that cancel to equal exactly the two Owen-type kinds, the constant and `E_{3/16}`; the new mutant `layer-dropped` shows that the check bites.
  - **4151593177:** NOTE §0 states [LP] Theorem B's hypothesis `0 < k_− ≤ k_+ < ∞`, and §7 classes windows outside it as coefficient integrals only.
  - **4151593183:** the script's docstring now describes the `d = 5` calculation.

  Values, the eighteen rules and the twelve earlier mutants are unchanged; `RESULTS.json` is byte-identical (NOTE §10, `SOURCE_FILES.json` `revisions`). The same §0/§7 wording is repaired in Math-#205 and #209.
- **Unresolved finding IDs:** none. The three Codex threads are answered (replies 4151775451, 4151775688, 4151775939) and resolved.
- **Validation:** 18/18 rules in `-B -S` and `-B -O -S`, byte-identical output; 13/13 mutants rejected in both modes; workflow replayed locally at v1.2; at v1.3 the manifest and pins verify and the containment audit passes 11/11 NOTE intervals and 16 prefixes; hosted run 4/4 green at `0d6bc80` (`certify-and-reject-mutants` 36875155827 in 20 minutes, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates` 36875156758), and at `b22ff4b` before it.
- **Next action:** nonauthor reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. I will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
