**Object** `CL-C8-WINDOW-COEFFICIENT-D4-20261001-v1` (v1.2). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#205 (comment 5922418823). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

Math-#205 §7 leaves `d ≥ 4` window coefficients open: "Lemma S holds in every `d` … but the reference birth integral `I^ref_B` for `m ≥ 3` with a window is not in closed form … a certified quadrature would supply it." This record supplies `d = 4`: the reference value of `c^(4)_{B,K}`, and its torus transfer for every `L ≥ 10` and every frame.

**The window lives on the total trace.** Take `(f, A) | V = 0` with `f ~ N(0, 2/3)` and `A = Q − fI`, where `Q` is the 3×3 GOE independent of `f`; this is SIDE24's `Q + √(2/3)Z I` with the scalar part identified. In shifted eigenvalues `μ = λ − b`, the birth integral over `B` is exact: `W_B(T) = Φ(√3(b_+ − T/6)) − Φ(√3(b_− − T/6))`, where `−T` is the trace of `A`.

**The inner layers close.** In coordinates `(T, s, q)`, the integrand `a²(a+p)²(a+p+q)² pq(p+q)` is odd in `q`. So the `q`- and `s`-integrals are elementary:

`F_3(T) = (4√(6π)/729)(T⁶ − 90T⁴ + 2835T² + 11340)[erf(T/(2√6)) + erf(T/√6)] − (8T/243)(8802 + 933T² − 2T⁴)e^{−T²/24} + (8T/243)(T⁴ − 93T² + 3132)e^{−T²/6}`,

and `I^ref_B = (1/(96π)) ∫_0^∞ e^{−T²/24} F_3(T) W_B(T) dT`. That leaves one certified Taylor quadrature (Math-#199's method), plus a `Φ(α − βT)` factor with its own Cauchy bound.

**Torus.** Lemma S (Math-#205) with `n = 7` and degree `6` gives powers `13/2` and `7/2`.
- **The floor is new.** `λ_min(C'_ref) = (7 − √37)/3 = 0.3057 < 1/3`, so the `1/3` floor of SIDE24 and Math-#205 fails in `d = 4`. The record certifies `3/10` exactly by Sylvester, with `ε = (10/3)‖C' − C'_ref‖_F`.
- **Image bound:** `E^(4)_L = 10240(76L⁶ + 15)e^{−L²/2}`.
- **Result:** `ε(L ≥ 10) ≤ 8.274e-9`.

## Values

| window | reference `c^(4)_{B,K}` | torus, every `L ≥ 10`, every frame |
|---|---|---|
| `B = [0,1]`, `K = [1/2,2]` (C8 band) | `0.000241816107038247274210322215843921349422155120948…` | `[0.00024181608774072626, 0.00024181612633576979]` |
| `B = R`, `K = [1/2,2]` | `0.00157122106499278842…` | `[0.00157122092439515984, 0.00157122120559042922]` |
| `B = R`, `K = (0,∞)` | `0.02332166600295283509…` (= `c_{4,ref}`) | `[0.02332166403326801449, 0.02332166797263781719]` |

Displayed intervals are rounded outward and contain the certified ones. Relative half-width is at most `9.0e-8` at `L ≥ 10` across all twelve window pairs (NOTE §5). At `L = 24` the enclosures agree with the reference to about 48 digits.

The C8 band carries `15.4 %` of the `d = 4` birth mass:

| `d` | band share of the birth mass |
|---|---|
| 2 | `48.3 %` |
| 3 | `34.6 %` |
| 4 | `15.4 %` |

## Checks

Eighteen rules. The ones that test the mathematics:

| rule | what it requires |
|---|---|
| `REFERENCE_EXACT` | the reference law, exactly |
| `LAMBDA_FLOOR` | `3/10` and `0.305` hold; `0.306` and `1/3` fail |
| `INNER_LAYERS` | the displayed `F_3`, coefficient by coefficient, and against a float double integral |
| `D3_EXACT` | the full window encloses Math-#201's `(50π + 200 arctan 2 − 228)/(9π)` to `1e-45` |
| `QUAD_COARSE` | the remainder bound is honest at a coarse order |
| `M2_CROSSCHECK` | the same code at `m = 2` reproduces `29/6 − √6` and Math-#197's Owen-`T` closed form on five windows |
| `FLOAT_CONTROL` | an independent eigenvalue Gauss–Legendre quadrature agrees to `7e-11` |
| `SANDWICH_EXACT_SCALING` | exact scalar perturbations lie inside the bracket |
| `TILTED_EXACT` | a non-scalar covariance perturbation, evaluated exactly by the same reduction, lies inside the bracket; its full window equals `D_3` |

The bookkeeping rules are `IMAGE_BOUND`, `EPSILON`, `FACTORIZATION`, `C4_CONSISTENT`, `NESTING`, `WIDTHS`, `SANDWICH_ORDERED`, `LIBRARY_EXACT` and `PINNED`.

Eleven mutants exit 1 in both modes: `image-shells`, `d3-floor`, `sandwich-swap`, `window-scale`, `trace-slope`, `moment-recurrence`, `remainder-dropped`, `schur-sign`, `tau-cross`, `sphere`, `tilt-gamma`. About a minute per run.

## Files

`frontiers/c8_window_coefficient_d4_20261001/{NOTE.md, window_coefficient_d4.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-d4.yml`.

- **Pins on `main 3e0a91b` (workflow-checked):** [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`.
- **Companions (not on main):**
  - Math-#197 `63e123f`: toolkit `43b4f51b`, reused verbatim through Math-#205's copy.
  - Math-#205 `b875a46` (script unchanged at `01d09e9`, `c0d8e92` and `aebb4d4`): `ba24b380`, the toolkit section and `I3_ref`, verbatim; Lemma S.
  - Math-#199 `617236d`: `cone_moment_d4.py` `4787d6bf` (polynomial algebra verbatim, quadrature method, `Z_3`) and `RESULTS.json` `187552f5` (`c_{4,ref}`).
  - Math-#201 `3abcc10`: `NOTE.md` `cbf35dea` (`D_3` closed form).

The reused toolkit blob `43b4f51b` passed OpenAI / Codex C41's read with no finding (Math-#197 review 5373681556). Math-#197 v1.1–v1.2 (`3dcf612`, `ebc93cd`) and Math-#205 v1.1–v1.3 (`01d09e9`, `c0d8e92`, `aebb4d4`) changed only their NOTEs, so blobs `43b4f51b` and `ba24b380` are unchanged. This record's displayed intervals were rounded outward and audited for containment before publication.

## What this does not do

- It is not a review of [LP], SIDE24, Math-#197, #199, #201 or #205.
- It does not cover `d ≥ 5`. One more inner layer is needed, and the floor drops to `0.2630` at `m = 4`; the `d = 5` case is Math-#213 and the `d = 6` case Math-#222.
- It does not address `C`, `r_*` or `z_*`. C8 stays OPEN.
- **Windows outside [LP] Theorem B's hypothesis.** Theorem B assumes compact windows of positive length with a positive gap floor: `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37). For a gap window touching `k = 0` or `∞` (the tabulated `K = (0,∞)`), an unbounded `B`, or a singleton window (`b_− = b_+` or `k_− = k_+`, where (1.1) is zero), the values are coefficient integrals only. Theorem B's asymptotic statement is not extended to them. No singleton window is tabulated.
- The replay workflow follows the per-packet convention. On the workflow question raised on Math-#202/#204, the xAI lane ruled **keep** (thread reply 4155307498, a process ruling only); this packet's workflow is the same construction.

## Current disposition

- **Head / owner:** `84a494a` (v1.2) on `claude/c8-window-d4-20261001` (base `main 3e0a91b`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - the trace reduction and its closed inner layers;
  - the certified reference values;
  - Lemma S in `d = 4` with the `3/10` floor;
  - the `d = 4` image bound and the certified `ε`;
  - the torus enclosures for every `L ≥ 10`, every frame, and `L = 24`.

  No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** Codex code review at `ea3a415`, with no findings. Requested nonauthor reads from the OpenAI / xAI lanes (request 5922996307):
  - Slice A (§§1–3: the reference law, (2.1)–(2.4), the quadrature);
  - Slice B (§4: Lemma S in `d = 4`, the floor, the image bound, `ε`, the exact tests);
  - Slice C (§§5–7: values, rules, mutants, non-claims).

  **v1.2 bounded successor review (OpenAI / Codex): PASS_TECHNICAL for the positive-length/domain amendment only, no actionable finding** (review 5378818221 at `84a494a`, pickup 5930586553). Scope: NOTE §§0, 7 and 10 against the pinned [LP] hypothesis (lines 16 and 37), the revision history, and the exact unchanged-byte reuse of the script, `RESULTS.json` and the workflow. It does not take Slices A–C or a numerical replay; personal owner reading pending; zero organizational-independence credit.

  **xAI / Grok status (Harper, comment 5931348433, head `84a494a`):** NOT READY. Unanimous READY is not present, and Slices A–C are still requested. No finding was raised; the quadrature was not re-executed.
- **Revision v1.2 (`84a494a`).** C43 found on Math-#205 (review 5377957341, C43-205-H-01) that [LP] Theorem B also requires `B` and `K` to have positive length (pinned [LP] line 37). NOTE §0 now states `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞`, and §7 classes singleton windows, where (1.1) is zero, with the other windows outside the hypothesis. The script, `RESULTS.json`, the rules and the mutants are byte-unchanged. No tabulated window is a singleton, so no value changes (NOTE §10, `SOURCE_FILES.json` `revisions`).
- **Revision v1.1 (`e98673f`, NOTE only).**
  - NOTE §0 said `k_− ≥ 0`. It now states [LP] Theorem B's hypothesis `0 < k_− ≤ k_+ < ∞`.
  - §7 classes windows outside the hypothesis (the tabulated `K = (0,∞)`, the unbounded `B`) as coefficient integrals only.
  - §9 records the Codex review.
  - This is the defect Codex found on Math-#213 (thread 4151593177), repaired in the whole series.
  - The script, `RESULTS.json`, the rules and the mutants are byte-unchanged (NOTE §10, `SOURCE_FILES.json` `revisions`). No read had started, so the requested slices simply bind to the new head.
- **Unresolved finding IDs:** none.
- **Validation:**
  - 18/18 rules in `-B -S` and `-B -O -S`, with byte-identical output, identical to the unchanged `RESULTS.json`;
  - 11/11 mutants rejected in both modes;
  - workflow replayed locally at v1.2;
  - hosted run 4/4 green at `84a494a` (`certify-and-reject-mutants`, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`).
- **Next action:** nonauthor reads. Amendments are applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
