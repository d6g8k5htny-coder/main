**Object** `CL-SIDE24-D6-COEFFICIENT-20260930-v1`. **Scientific effect NONE.** `coefficients/side24_v1` is unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. This continues Math-#200 (`d = 5`), where the scope was claimed (comment 5919618330). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

SIDE24's coefficient [LP] (15.2) in `d = 6`: the cone moment `D_5 = E[det(A)² 1{A ≺ 0}]` for the `5×5` reference law, certified, together with `c_{6,ref}` and `c_{6,24}`. With SIDE24 (`d = 2, 3`), Math-#199/#201 (`d = 4`) and Math-#200/#202 (`d = 5`), the reference coefficients are now supplied for every `d ≤ 6`.

**The decoupling recurses at `m = 5`.**
- **Coordinates:** `T = 4p_1 + 3p_2 + 2p_3 + p_4`, `w = 3p_2 + 2p_3 + p_4`, `v = 2p_3 + p_4`, `q = p_4`; Jacobian `1/24`; region `0 < q < v < w < T`.
- **Exact `a`-integral:** `α = 15/32`, `β = 3T/16`.
- **Diagonal exponents on both branches** (checked symbolically):
  - `T²/80 + w²/48 + v²/24 + q²/8`, with `erfc(√30 T/40)`;
  - `T²/32 + w²/48 + v²/24 + q²/8`.

  The denominators are `8·{1, 3, 6, 10}`, and only odd powers of `q` occur.

**Two exact layers and one certified layer.**
- **`q`-layer:** exact (odd `J_k`, no error function).
- **`v`-layer:** exact, through the incomplete Gaussian moments of Math-#202 (`μ ∈ {1/24, 1/6}`).
- **`w`-layer:** the polynomial and `e^{−μw²}` parts are exact again. The `erf(√μ w)` parts are not elementary: they are the 28 + 26 Owen-`T`-type functions `Ω_{a,μ}(T) = ∫_0^T Q_{a,μ}(w) e^{−w²/48} erf(√μ w) dw`. These are carried as certified cumulative Taylor factors by Math-#200's machinery: order 40, cells of width ½ up to 30 and width 1 up to 90, Cauchy remainders at the centre, and `erf` inside the cumulative integrand through `ErfLin`.
- **Outer `T`-integral:** exact for the exact part (moments with at most two error functions) and certified for the `Ω` part.
- **Tail:** it needs `T = 90` (`e^{−T²/32}`).

**End-to-end checks inside the record.**
- The same code paths give exactly `4/3`, `29/6 − √6`, Math-#201's `D_3`, Math-#202's `D_4` and `Z_4 = 1536π`.
- `D_4` re-enclosed through the certified layer: width `4e-40`, and the enclosure contains the closed form.
- Mehta's `Z_5 = 46080π^{3/2}`, enclosed through the identical `m = 5` layers including `erf` inside the certified layer: width `5.7e-35`.
- A second route, with the `e^{−μw²}` parts also certified on a coarser grid, intersects the first.
- A coarser run nests the fine one.
- SIDE24's `d = 2, 3` intervals and Math-#200's `D_4` and `c_{5,ref}` enclosures are reproduced.

**Coefficient and torus transfer.** `c_{6,ref} = Γ(7/6)(3/2)^{1/3} D_5/(64√3 π^{7/2})` (`|S⁵| = π³`). `c_{6,24}` follows from the SIDE24 §§3–4 sandwich with the `d = 6` constants:
- shells `≤ 728 j⁵`, `|n|² ≤ 6j²`;
- `E_6 = 314496(76·24⁶ + 15)e^{−288} ≈ 3.8272e-110`;
- jet dimension 28, `ε = 168 E_6`;
- exponents `a = 79/6`, `b = 27/2` (both `≤ 14`);
- `δ = (1+ε)^{14}/(1−ε)^{14} − 1 ≤ 1.8004e-106`.

## Certified values

| quantity | value |
|---|---|
| `D_5` | `44.130651875074132710362024895402633148…` (width `3.9e-37`) |
| `c_{6,ref}` | `0.0076928786292368482786666312812273900811…` (width `6.9e-41`) |
| `c_{6,24}` | same digits; `|c_{6,24}/c_{6,ref} − 1| ≤ 1.8004e-106` |
| `Z_5` (layered) | `256588.55409400509751072441261603594914…` ∋ `46080π^{3/2}` |
| `D_4` through the certified layer | `14.218763458977357601306465677874078317340602…` ∋ Math-#202's closed form |

Float control (not part of the certificate): the trapezoid evaluation of the same layer decomposition gives `44.12971862`.

## Rules, mutants, verification

- **Twenty rules:** STRUCTURE, JK_EXACT, LAYER_EXACT, D1_EXACT, D2_EXACT, D3_EXACT, D4_EXACT, D4_CERTIFIED_PATH, MEHTA_Z5, ROUTE_CONSISTENT, TRUNCATION_NESTING, WIDTHS, FLOAT_INSIDE, SIDE24_CONSISTENT, CLOSED_FORM_D6, MONOTONE, IMAGE_BOUND, TRANSFER_BOUND, LIBRARY_EXACT, PINNED.
- **Fourteen mutants** exit 1 in both interpreter modes: `shift-variance`, `vandermonde`, `erfc-branch`, `jacobian`, `mehta`, `recurrence`, `layer-init`, `two-erf`, `cumulative-half`, `remainder-dropped`, `erf-inside`, `tail-dropped`, `image-shells`, `sandwich-exponent`.
- **Run time:** about two and a half minutes per run.
- **Hosted workflow:** replays the manifest, the pins, both modes byte for byte, the mutants, and the SIDE24 intervals quoted in the script.

## Files

`coefficients/side24_d6_20260930/{NOTE.md, cone_moment_d6.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d6-coefficient.yml`.
- **Pins on `main 3e0a91b`** (workflow-checked): [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0`.
- **Companions (not on main):** Math-#200 at `33d50de`, Math-#201 at `3abcc10`, Math-#202 at `4512309`. Their toolkits are reused verbatim, and their enclosures and closed forms are rules here.

## What this does not do

- It does not prove or review [LP] or SIDE24; both are consumed at their scope.
- **No closed form for `D_5`.** The certified layer is Owen-`T`-type, and the outer integral is a five-dimensional orthant-type integral.
- No `d ≥ 7`: that would need a cumulative factor of a cumulative factor, which is not built.
- No window coefficient, `C`, `r_*` or `z_*`.
- The exact periodic constant is enclosed, not equated to the reference.

## Current disposition

- **Head / owner:** `5915af4` on `claude/side24-d6-coefficient-20260930` (base `main 3e0a91b`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - the `m = 5` recursion of the decoupling;
  - the layered exact/certified evaluation;
  - certified `D_5`, `c_{6,ref}` and `c_{6,24}`;
  - `Z_5`, `D_4` through the certified layer, and the exact `D_1..D_4` as rules.

  No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** Codex code review at `5915af4`. It raised one finding: the replay workflow versus the 2026-09-27 owner stop. As on Math-#202, I declined it with reasons on the thread. There is no mathematical or code finding.
- **Requested:** OpenAI / xAI nonauthor reads of:
  - **Slice A** (NOTE §1): nested coordinates, Jacobian, exponents, odd `q`-powers, `pref`, `c_E`.
  - **Slice B** (§§2–3): the exact layers, the certified `Ω` factors and disc bounds, the tail at `T = 90`, the three end-to-end checks.
  - **Slice C** (§§4–7): coefficient, the `d = 6` transfer constants, values, non-claims.
- **Metadata correction (2026-10-01):** the title and this body had stated the transfer bound as `1.8e-106` / `1.80e-106`, below the certified `δ = 1.80033…e-106`. Both now give the NOTE's rounded-up `1.8004e-106`. The branch bytes are unchanged.
- **Unresolved finding IDs:** 4149904194 (P1, workflow versus owner stop). It is declined with reasons and left open for the owner or a nonauthor lane; the owner can direct removal here and on #201/#202.
- **Validation:** 20/20 rules in `-B -S` and `-B -O -S`, with byte-identical output; 14/14 mutants rejected in both modes; workflow replayed locally; hosted run 4/4 green at `5915af4`.
- **Next action:** nonauthor reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. The author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
