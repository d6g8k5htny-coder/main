**Object** `CL-C8-WINDOW-COEFFICIENT-PLANAR-20260930-v1` (v1.2). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#195 (comment 5915784573). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

The compact-window coefficient `c_{B,K}` of [LP] Theorem B ((11.3) with the §15 factorization) in `d = 2`. The parent lists it among its unsupplied constants, and Math-#195 (C8, `z_*`) explicitly leaves it aside.

**Window hypothesis (v1.2).** [LP] Theorem B assumes compact windows of positive length with a positive gap floor: `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37). For windows outside it the values are coefficient integrals (1.1) only, and Theorem B's asymptotic statement is not extended to them. These are gap windows touching `k = 0` or `∞` (the tabulated `[1/2,∞)`, `[1,∞)` and `(0,∞)`), unbounded `B` (`[0,∞)`, `(−∞,0]`, `R`), and singleton windows, where (1.1) is zero; none of the last is tabulated.

**Theorem W (reference kernel, exact).** `c_{B,K} = c_{2,ref} F_B G_K` with `c_{2,ref} = 2 Γ(7/6) (3/2)^{1/3} / (3 √3 π^{3/2})` (SIDE24 (1) at `d = 2`), where:

- `G_K = P(7/6, 12 k_+²) − P(7/6, 12 k_-²)`, the regularized lower incomplete gamma, from the gap density `k^{4/3} e^{−12k²}` (mode `k = √2/6`);
- `F_B = Ψ(b_+ √(3/2)) − Ψ(b_- √(3/2))` with `Ψ(x) = Φ(x) − 2 T(x, 1/√3) − ½ x φ(x) Φ(x/√3) − (√3/4π) e^{−2x²/3}` (`T` is Owen's function). This comes from the contact law `A | f=b, V=0 ~ N(−b, 2)` ([LP] (5.4)) and the Owen primitives `∫ tⁿ φ(t) Φ(at) dt`, `n ≤ 4`;
- symmetric window: `F_{[−β,β]} = erf(β√3/2) − β√3 e^{−3β²/4}/(4√π)`; half line: `F_{[0,∞)} = 2/3 + √3/(4π)`.

**Torus kernel.** Every 3-jet covariance entry of `K_L`, in any frame, lies within `E_L = 128 (76 L⁶ + 15) e^{−L²/2}` of its reference value; this is the `d = 2` form of SIDE24 §2. The same formulas, evaluated over those entry intervals, give one enclosure for every `L ≥ 10` and every frame, and a sharper one at `L = 24`.

**`d = 3` reference.** The closed form is `m_{3,b} = (b³ + b) φ(b) + (b⁴ + 2b² + 7) Φ(b) − 4√2 e^{−b²/4} Φ(b/√2)`. Its Gaussian mean is SIDE24's `D_2 = 29/6 − √6` (a rule). The corresponding window factor is also given. No torus transfer is made here, for the reason stated in the NOTE; Math-#205 supplies it.

**Exhaustion.** Two-sided elementary bounds on `1 − G_K` (`k_-^{7/3}` small-gap loss, `e^{−12k_+²}` large-gap loss) and Gaussian bounds on the birth tails, checked against the certified values.

## Certified values (C8 band of Math-#195: `B = [0,1]`, `K = [1/2, 2]`)

Displayed intervals are rounded outward (v1.1) and contain the certified intervals of `RESULTS.json`.

| quantity | value |
|---|---|
| `F_B` | `0.48264345641927413020434204645556992729…` |
| `G_K` | `0.06737173342564166734884441847440277343…` |
| `c_{B,K}` reference | `0.0023869380211529483091019117…` (`3.25 %` of `c_{2,ref}`) |
| `c_{B,K}` torus, every `L ≥ 10`, every frame | `[0.00238693802093000833, 0.00238693802137588829]` |
| `c_{B,K}` torus, `L = 24` | `0.0023869380211529483091019117…` |
| `c_{2,ref}` | `0.07340691930603427103013596295777…` (inside SIDE24's `c_{2,24}` interval) |

Five windows and both factors are tabulated separately in `NOTE.md` §6 and `RESULTS.json`. Enclosure widths are below `1e-50` for the reference and `L = 24`; the relative half-width for `L ≥ 10` is `2.7e-11`.

## Method and controls

- **Arithmetic.** `decimal` interval arithmetic at 60 digits with directed rounding. `π` by Machin; fractional powers by exact integer roots; `erf` and `γ(7/6, ·)` by positive series with geometric tails; Owen's `T` and `arctan` by alternating series bracketed by consecutive partial sums; `Γ(7/6)` as `γ(7/6, 120)` plus a two-sided tail.
- **Fourteen rules:**
  - independent float controls, including the direct double integral for `m_{3,b}`;
  - exact identities: `D_1`, `D_2`, half line, symmetric closed form;
  - closed form against the pipeline on every window;
  - SIDE24 consistency;
  - image-bound value;
  - monotonicity;
  - exhaustion bounds;
  - truncation nesting;
  - `LIBRARY_EXACT`;
  - pinned digits.
- **Ten mutants** exit 1 in both interpreter modes.
- **Hosted workflow:** replays the manifest, the pins, both modes byte for byte, and the mutants, and checks the SIDE24 interval quoted in the script against `ENCLOSURE.json`.

**Method note (also affects Math-#190).**
- **Defect in the first draft:** interval endpoints were negated with the Decimal unary minus, which rounds to the thread's default 28-digit context. `exp(−x)` at a 60-digit `x` was then wrong beyond the 28th digit, while every other rule passed.
- **Repair:** exact `copy_negate`, explicit contexts, and rule `LIBRARY_EXACT`, which compares the module's `exp`, `sqrt` and negation with exact rational brackets.
- **Math-#190:** it uses the same interval class, but never applies the operation to a long endpoint, so its 167 intervals are byte-identical under the repaired class. Math-#190 v1.3 carries the repair, the rule, and a mutant reinstating the old negation.

## Files

`frontiers/c8_window_coefficient_planar_20260930/{NOTE.md, window_coefficient.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-planar.yml`.

- **Pins on `main 3e0a91b`, workflow-checked:** [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0`.
- **Read:** SIDE24 remainder `0bcf0a6f`, C8 catalog entry `02455f05`.
- **Companion:** Math-#195 `NOTE.md` `a8b1f71d` at `16da75f`.

## Current disposition

- **Head / owner:** `ebc93cd` (v1.2) on `claude/c8-window-coefficient-planar-20260930` (base `main 3e0a91b`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - exact factorization and closed forms of `c_{B,K}` in `d = 2` (reference kernel);
  - certified enclosures on the declared windows;
  - torus transfer for every `L ≥ 10` and `L = 24`;
  - the `d = 3` reference closed form;
  - exhaustion bounds.

  No theorem of [LP] is proved or reviewed.
- **Completed review scopes.** All are OpenAI / Codex reads; personal owner reading is pending; zero organizational-independence credit.
  - **C41: PASS_TECHNICAL, no finding** (review 5373681556 at `63e123f`). Scope: the exact interval and transcendental primitives, the G0–G4 integrals, the gap normalization, and the `d = 3` reference-integral slice consumed by Math-#205. Blob `43b4f51b`, unchanged in v1.1 and v1.2. This is not acceptance of the whole PR.
  - **C42 bounded follow-up: PASS_TECHNICAL, no actionable finding** (review 5374044799 at `3dcf612`). Scope: the v1.1 reporting amendment.
    - Only `NOTE.md` and `SOURCE_FILES.json` changed. The script, `RESULTS.json` and the workflow are at unchanged blobs.
    - All six original truncated occurrences are rejected, and all 8 amended enclosures are accepted at minimal outward rounding.
    - All 40 `...` prefixes pass. That is 39 direct `RESULTS.json` prefixes plus the derived `P(7/6, 3) = 0.9326...`, which goes through the exact complement `1 − G_{[1/2,∞)}`. My earlier count of 39 omitted that derived one.
  - **v1.2 bounded successor review: PASS_TECHNICAL for the positive-length/domain amendment only, no actionable finding** (review 5378817469 at `ebc93cd`, pickup 5930584275). Scope: NOTE §§0, 8 and 10 against the pinned [LP] hypothesis (lines 16 and 37), the revision history, and the exact unchanged-byte reuse of the script, `RESULTS.json` and the workflow. It does not take the remaining analytic slices or a numerical replay.
- **Revision v1.2 (`ebc93cd`, NOTE only).**
  - NOTE §0 said only `0 < k_-`. It now states [LP] Theorem B's hypothesis: compact windows of positive length with a positive gap floor, `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37).
  - §8 classes the windows outside it as coefficient integrals only.
  - These are the defects found by Codex on Math-#213 (thread 4151593177) and by C43 on Math-#205 (C43-205-H-01), repaired in every packet of the series.
  - `window_coefficient.py` (`43b4f51b`), `RESULTS.json`, the rules and the mutants are unchanged (NOTE §10, `SOURCE_FILES.json` `revisions`). No value changes.
- **Still requested:** nonauthor reads from the OpenAI / xAI lanes:
  - Slice A: §2 conditional laws, Owen primitives, Theorem W;
  - Slice B: §3 image bound and interval transfer;
  - Slice C: §4 `d = 3` closed form, §5 bounds, §§6–8 scope and reporting (the v1.2 window hypothesis now has its bounded read).
- **Unresolved finding IDs:** none.
- **Validation:** 14/14 rules in `-B -S` and `-B -O -S`, byte-identical output, identical to the unchanged `RESULTS.json`; 10/10 mutants rejected in both modes; workflow replayed locally at v1.2; containment audit 8/8 NOTE intervals and 39 prefixes; hosted run 4/4 green at `ebc93cd` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`).
- **Next action:** the remaining slice reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
