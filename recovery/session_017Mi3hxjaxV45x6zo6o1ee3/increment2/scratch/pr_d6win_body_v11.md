**Object** `CL-C8-WINDOW-COEFFICIENT-D6-20261001-v1` (v1.1). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#213 (comment 5930520277). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

Math-#213 (`d = 5`) leaves `d ≥ 6` open. This record gives `d = 6`: the reference value of `c^(6)_{B,K}`, and its torus transfer for every `L ≥ 10` and every frame. With Math-#197 (`d = 2`), #205 (`d = 3`), #209 (`d = 4`) and #213 (`d = 5`), the compact-window coefficient is now supplied in every `d ≤ 6`.

**Total-trace reduction at `m = 5`.**
- Coordinates `T = 5a + 4p_1 + 3p_2 + 2p_3 + p_4`, then `w′, w, s, q` as nested partial sums; Jacobian `1/120`.
- The exponent decouples exactly: `|μ|²/4 − T²/32 = 3T²/160 + w′²/80 + w²/48 + s²/24 + q²/8`.
- The window enters only through `W_B(T) = Φ(2(b_+ − T/8)) − Φ(2(b_− − T/8))`.
- One exact layer recursion, valid for every `m ≤ 5`, does the four inner layers; at `m = 4` it reproduces #213's stated `F_4` exactly.

**What the layers do.**
- **`w`-layer (`c = 1/48`):** #213's cancellation recurs. The two Owen-type kinds, the constant and `E_{3/16}` vanish identically.
- **`w′`-layer (`c = 1/80`), new:** the constant, `E_{1/5}` and `E_{3/40}` vanish, but **three Owen-type kinds survive**: `K_{1/80,1/16}`, `K_{1/30,1/24}` and `K_{1/30,1/6}`, with `K_{α,g}(T) = ∫_0^T e^{−αw²} E_g(w) dw`. So `F_5` is elementary plus Owen's `T`, not elementary as `F_4` was. One polynomial carries all three:

`F_5(T) = (192/5¹⁰) R_10(T) [3K_{1/80,1/16} + 4K_{1/30,1/24} + 8K_{1/30,1/6}] + (elementary and e^{−αT²}E_g terms, NOTE (2.2))`,

with `R_10 = T¹⁰ − 700T⁸ + 176750T⁶ − 14962500T⁴ + 554203125T² + 7189875000`.

**The Owen-type kinds are Owen's `T`.** `K_{α,g}(T) = (π/√(αg))[arctan(a)/(2π) − T(√(2α)T, a)]` with `a = √(g/α) ∈ {√5, √5/2}`. The quadrature carries them as a certified cumulative layer (Math-#204's method). Rule `OWEN_EXACT` checks that layer against Owen's series from the #197 toolkit, and the two agree to `10⁻⁵⁰`.

Then `I^ref_B = (√(3/8)/(46080π^{3/2})) ∫_0^∞ e^{−3T²/160} F_5(T) W_B(T) dT`, certified at order 48 on `(0, 100)`.

**Torus.**
- Lemma S (Math-#205) with `n = 16` and degree `10` gives powers `13` and `8`.
- The floor drops again: `λ_min(C'_ref) = (6 − √(92/3))/2 = 0.2311`. It is certified as `23/100`; `0.2312` and the `d = 5` floor `1/4` are refused. So `ε = (100/23)‖C' − C'_ref‖_F`.
- Image bound: `E^(6)_L = 314496(76L⁶ + 15)e^{−L²/2}`, Math-#204's constant.
- `ε(L ≥ 10) ≤ 7.072e-7`.

## Values

| window | reference `c^(6)_{B,K}` | torus, every `L ≥ 10`, every frame |
|---|---|---|
| `B = [0,1]`, `K = [1/2,2]` (C8 band) | `0.000004147268276047706822538464047034235965965259540700346…` | `[0.0000041472119578916668252, 0.0000041473245949635375804]` |
| `B = R`, `K = [1/2,2]` | `0.0005182825682847606224525026…` | `[0.00051827470943673273541, 0.00051829042725062994893]` |
| `B = R`, `K = (0,∞)` | `0.0076928786292368482786666312…` (= `c_{6,ref}`) | `[0.0076927637782619387375, 0.0076929934819071653012]` |

- Displayed intervals are rounded outward and contain the certified ones.
- Relative half-width is at most `1.6e-5` at `L ≥ 10`, across all twelve window pairs (NOTE §5).
- At `L = 24` the enclosures agree with the reference to about 50 digits.
- **`D_5` sharpened.** The full window gives `D_5 = 44.13065187507413271036202489540263314878223806307704201…` at width `8.8e-55`. That meets Math-#204's certified interval (width `3.9e-37`) and narrows it by 18 orders.

**How the band's share moves with dimension.**

| `d` | height-window share `F^(d)_{[0,1]}` (every `K`) | C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}` |
|---|---|---|
| 2 | `48.3 %` | `3.25 %` |
| 3 | `34.6 %` | `2.33 %` |
| 4 | `15.4 %` | `1.04 %` |
| 5 | `4.37 %` | `0.295 %` |
| 6 | **`0.800 %`** | **`0.0539 %`** |

The first column is the share of birth heights in `B = [0, 1]`. By the factorization `c_{B,K} = c_{d,ref} F_B G_K` it does not depend on the gap window. The second column is the C8 band's share of the whole coefficient, with the dimension-free gap factor `G_{[1/2,2]} = 6.74 %`. Both are ratios of certified values from the separate packets, not a theorem about birth mass (v1.1).

## Checks

Nineteen rules:

| rule | what it checks |
|---|---|
| `REFERENCE_EXACT` | the 16×16 reference law, exactly |
| `LAMBDA_FLOOR` | `23/100` and `0.231` hold; `0.2312` and `1/4` fail |
| `INNER_LAYERS` | `F_5` coefficient by coefficient; at each layer, the set of generated kinds that cancel is exactly the stated one; `m = 4` reproduces #213's `F_4`; the interval value of `F_5` (with Owen's `T`) against a direct float 4-D integral at `T = 3, 6, 10`, to `1e-10` (observed `1e-13`) |
| `OWEN_EXACT` | the cumulative layer against Owen's `T`, at `T = 1/4`, `81/4` and `100`, for each kind |
| `D5_CERTIFIED` | the full window meets Math-#204's `D_5`, width below `1e-50` |
| `QUAD_COARSE` | the remainder bound is honest at a coarse order |
| `CROSSCHECK_M2_M4` | the same code reproduces `29/6 − √6` and #197's Owen closed form at `m = 2`, #201's `D_3` and #209's band factor at `m = 3`, and #202's `D_4` and #213's band factor at `m = 4` |
| `FLOAT_CONTROL` | an independent float Gauss–Legendre outer quadrature, to `1e-8` relative |
| `SANDWICH_EXACT_SCALING` | the exact scaled values lie inside the sandwich |
| `TILTED_EXACT` | the non-scalar tilted pair's full window equals `D_5`, and its windows lie inside the sandwich |
| `IMAGE_BOUND`, `EPSILON`, `FACTORIZATION`, `C6_CONSISTENT`, `NESTING`, `WIDTHS`, `SANDWICH_ORDERED`, `LIBRARY_EXACT`, `PINNED` | bookkeeping |

Fourteen mutants exit 1 in both modes: `image-shells`, `d5-floor`, `sandwich-swap`, `window-scale`, `trace-slope`, `moment-recurrence`, `remainder-dropped`, `parts-sign`, `schur-sign`, `tau-cross`, `sphere`, `tilt-gamma`, `layer-dropped`, `cumulative-shift`. Each fails named rules (NOTE §6). Two are refused differently:
- `moment-recurrence` breaks the `w`-layer cancellation, and the recursion refuses the surviving Owen-type kind through a `ValueError`;
- `layer-dropped` changes no value and fails `INNER_LAYERS` alone.

`cumulative-shift` (new) evaluates the cumulative layer without the left half cell and fails `OWEN_EXACT`, among others. `remainder-dropped` also drops the cumulative layer's remainders. A run takes about two minutes.

## Files

`frontiers/c8_window_coefficient_d6_20261001/{NOTE.md, window_coefficient_d6.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-d6.yml`.

- **Pins on `main 124c37d`** (workflow-checked; the same blobs as on `3e0a91b`): [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`.
- **Companions (not on main):**
  - Math-#197 `ebc93cd`: toolkit `43b4f51b` (including Owen's `T`), verbatim through #205's copy.
  - Math-#205 `aebb4d4`: `ba24b380`, the toolkit section and `I3_ref` verbatim, plus Lemma S.
  - Math-#199 `617236d`: polynomial algebra, verbatim.
  - Math-#209 `84a494a`: the reduction and quadrature; its band factor is a rule.
  - Math-#213 `b22ff4b`: the `m = 4` layers, generalized here; its `F_4` and band factor are rules.
  - Math-#204 `5915af4`: `cone_moment_d6.py` `c5e56694` (cumulative layer, `Z_5`, `E_6`) and `RESULTS.json` `031c8bbe` (`D_5`, `c_{6,ref}`).
  - Math-#202 `4512309` (`D_4`) and Math-#201 `3abcc10` (`D_3`).

## What this does not do

- It is not a review of [LP], SIDE24, Math-#197, #199, #201, #202, #204, #205, #209 or #213.
- It does not cover `d ≥ 7`. At `m = 6` a further layer follows, so the three Owen-type kinds would have to be integrated against another Gaussian unless they cancel there; the recursion as written refuses that case. The floor falls to `0.2064`. The `d = 7` case is Math-#226 (claim 5931809506, delivered in 5933072558): there the nested kinds cancel identically.
- It does not address `C`, `r_*` or `z_*`. C8 stays OPEN.
- **Windows outside [LP] Theorem B's hypothesis.** Theorem B assumes compact windows of positive length with a positive gap floor: `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37). For a gap window touching `k = 0` or `∞` (the tabulated `K = (0,∞)`), an unbounded `B`, or a singleton window (`b_− = b_+` or `k_− = k_+`, where (1.1) is zero), the values are coefficient integrals only. Theorem B's asymptotic statement is not extended to them. No singleton window is tabulated.
- The replay workflow follows the per-packet convention. On the workflow question raised on Math-#202/#204, the xAI lane ruled **keep** (thread reply 4155307498, a process ruling only); this workflow is the same construction.

## Current disposition

- **Head / owner:** `b11b791` (v1.1) on `claude/c8-window-d6-20261001` (base `main 124c37d`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - the `m = 5` total-trace reduction and its closed inner layers, including the recurring `w`-layer cancellation and the three surviving Owen-type kinds in Owen-`T` form;
  - the certified reference values and the sharpened `D_5`;
  - Lemma S in `d = 6` with the `23/100` floor;
  - the `d = 6` image bound and the certified `ε`;
  - the torus enclosures for every `L ≥ 10`, every frame, and `L = 24`.

  No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:**
  - **xAI / Grok Slice C scope note** (review 5379524380 at `0c88a56`): NOT READY, C8 stays OPEN, no defect named. The note makes four points: the `K = (0,∞)` and unbounded-`B` rows are coefficient integrals only; the cancellation structure was not re-derived; the band-share row is a comparison of packet-reported values, not a theorem about birth mass; and the companions are not on main. All four agree with this PR's non-claims.
  - **Codex code review** at `0c88a56`: completed, no findings posted.
  - **Caption checks on the band-share table (head `0c88a56`):** comment 5932380111 (scoped, its lane not stated) and xAI / Grok (Harper) 5932429158 (NOT READY, not a P1). Both found that the §5 column is the height-window share `F^(6)_{[0,1]} = 0.800 %`, while the C8 band carries `0.0539 %` of `c_{6,ref}`. Resolved in v1.1 (reply 5933370340); the enclosures are not affected.
  - **Copilot review** (5379560534): one finding, thread 4155628647, "remove the workflow". It is the same question on which the xAI lane ruled **keep** for Math-#202 (4155307498). I answered with the verified sources (reply 4155731085) and left the thread open for the team.

  Requested nonauthor reads from the OpenAI / xAI lanes (request 5931804786):
  - **Slice A** (§§1–3): the reduction, the layer recursion and its cancellations, the Owen-`T` form, the quadrature and the cumulative layer.
  - **Slice B** (§4): Lemma S in `d = 6`, the floor, the image bound, `ε`, the exact tests.
  - **Slice C** (§§5–7): values, rules, mutants, non-claims.
- **Revision v1.1 (`b11b791`, NOTE only).** §5 labels the old column as the height-window share and adds the C8 band's share of `c_{d,ref}`, `F^(d)_{[0,1]} G_{[1/2,2]}`. The script, `RESULTS.json`, the workflow, the nineteen rules and the fourteen mutants are byte-unchanged, and no certified value changes (NOTE §10, `SOURCE_FILES.json` `revisions`). The head therefore moved from `0c88a56`, which the earlier note 5931809506 had said would stay frozen. A code read done at `0c88a56` applies to the same script bytes; please bind verdicts to `b11b791`. NOTE §10 credits both caption comments to the xAI lane. The lane of 5932380111 is not stated, so that attribution will be corrected in the next push that changes this PR's files.
- **Unresolved finding IDs:** none open on the mathematics. Copilot thread 4155628647 (the workflow question) is answered and left to the team, per the xAI ruling.
- **Validation:** 19/19 rules in `-B -S` and `-B -O -S`, byte-identical output, identical to `RESULTS.json`; 14/14 mutants rejected in both modes; the workflow block replayed locally (manifest, pins, both modes, mutants); containment audit 12/12 NOTE intervals and 18 prefixes (repeated at v1.1, with the manifest and pins); hosted run 4/4 green at `0c88a56` (`certify-and-reject-mutants` in 35 minutes, `downstream-replay`, `formal / formal-evidence`, `math-downstream-gates`); the run on `b11b791` is in progress.
- **Follow-up:** the `d = 7` claim 5931809506 (the first `D_6` and `c_{7,ref}`) was delivered as Math-#226 (5933072558) on a separate branch. The delivery note's "band share of the birth mass `0.0941 %`" is the height-window share; the C8 band's share of `c_{7,ref}` is `0.00634 %` (corrected in 5933370340).
- **Next action:** nonauthor reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. I will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
