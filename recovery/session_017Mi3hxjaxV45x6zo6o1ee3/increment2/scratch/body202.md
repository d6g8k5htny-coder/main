**Object** `CL-SIDE24-D4-CLOSED-FORM-20260930-v1` (v1.1 at `4512309`). **Scientific effect NONE.** `coefficients/side24_v1` unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#200 (comment 5918942608). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## Statement

**Proposition.** For SIDE24's reference law in `d = 5`, the cone moment of [LP] (15.1) is

```
D_4 = 6695/54 − (405/32)√6 − (1/π) [ (1375/72)√21 + (6695/27) arctan√(3/7) + (405/16)√6 arctan(1/√14) ]
    = 14.21876345897735760130646567787407831734060231608202158418…
```

**Corollary.** `c_{5,ref} = Γ(7/6)(3/2)^{1/3} D_4/(12√3 π^{7/2}) = 0.01321931938008496807603348751464139652105289012569990545…`; `c_{5,24}` as in Math-#200 (`|c_{5,24}/c_{5,ref} − 1| ≤ 1.8554e-107`). With SIDE24 (`D_1 = 4/3`, `D_2 = 29/6 − √6`) and Math-#201 (`D_3`), the reference coefficients `c_{d,ref}` are now in closed form for every `d ≤ 5`.

Math-#200's certified enclosure of `D_4` (independent method: order-40 Taylor quadrature with certified cumulative integrals) contains the interval evaluation of the closed form **at its centre** (`1.94e-42` from either endpoint of a `3.9e-42`-wide enclosure; 43 common digits). Mehta's `Z_4 = 1536π` is reproduced exactly by the same triangle machinery.

## How

Math-#200 reduces `D_4` to `pref[c_E ∫e^{−T²/48} erfc(κT) Σ_a T^a L_a^E + ∫e^{−T²/21} Σ_a T^a L_a^R]`, `L_a(T) = ∫_0^T e^{−v²/24} Σ_k e_{a,k}(v) J_k(v) dv`, `κ = √21/28`. **Only odd `k` occur** in both branches (`{1,3,5,7,9}`), so `J_k = B_k + C_k(v)e^{−v²/8}` and the cumulative integrals are incomplete Gaussian moments `G_n(μ,T) = ∫_0^T v^n e^{−μv²}dv`, `μ ∈ {1/24, 1/6}`: polynomials, `e^{−μT²}` polynomials and `erf(√μT)` terms (`G_0 = ½√(π/μ) erf(√μT)`, `G_1 = (1 − e^{−μT²})/(2μ)`, `G_n = [(n−1)G_{n−2} − T^{n−1}e^{−μT²}]/(2μ)`). The outer integrand then carries at most two error functions, `erfc(κT)` and `erf(√μT)`, whose moments are elementary: `∫_0^∞ e^{−λs²} erf(us) erf(vs) ds = arctan(uv/√(λ(λ+u²+v²)))/√(πλ)` (proved in NOTE L3) plus the by-parts recurrence of Math-#201. The script does everything in exact arithmetic on the `Q`-span of `√r · π^{k/2} · arctan-atom`; the output is an identity. The raw output has atoms `arctan(√14/4)` and `arctan(√14/7)` with opposite coefficients, which combine to `arctan(1/√14)`.

## Rules, mutants, verification

Sixteen rules: `STRUCTURE` (constants, odd `k`, atoms), `GAUSS_MOMENTS`, `JK_EXACT`, `CUMULATIVE_EXACT` (`G_n(μ,0) = 0`, `∂_T G_n = T^n e^{−μT²}` exactly in the T-function space), `D1_EXACT`, `D2_EXACT`, `D3_EXACT` (Math-#201's form), `MEHTA_Z4_EXACT`, `D4_FORM` (exact equality with pinned coefficients), `BASE_INTEGRALS_FLOAT`, `MOMENTS_FLOAT` (controls), `D4_INSIDE_CERTIFIED`, `SIMPLIFIED_FORM`, `C5_CLOSED` (60-digit interval evaluations inside #200's enclosures), `LIBRARY_EXACT`, `PINNED`. Fourteen mutants exit 1 in both modes (`shift-variance`, `vandermonde`, `erfc-sign`, `jacobian`, `mehta`, `by-parts`, `one-erf`, `two-erf`, `jk-init`, `gauss-half`, `boundary-term`, `atom-pi`, `cumulative-init`, `cumulative-recurrence`). About two seconds per run; hosted workflow replays manifest, pins, both modes byte for byte, the stated closed-form strings for `D_4` and `Z_4`, and the mutants.

## Files

`coefficients/side24_d5_closed_form_20260930/{NOTE.md, closed_form_d4.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d5-closed-form.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04` (workflow-checked). Companions (not on main): Math-#200 at `33d50de` (`RESULTS.json` blob `b67c00f4`, `NOTE.md` blob `ab3fe17d`), Math-#201 at `3abcc10` (`closed_form_d3.py` blob `a0c91692`, reused verbatim).

## What this does not do

Not a proof or review of [LP] or SIDE24 (consumed at their scope). **No closed form for `D_5` (`d = 6`) is claimed or expected by this method:** the `m = 5` reduction needs a second cumulative layer, and a cumulative integral of `e^{−μ'v²}·poly·erf(√μ v)` is not elementary in its upper limit (Owen-`T`-type), so the outer integrand would carry a non-elementary factor. Math-#200 is not edited (its §7 "no closed form for `D_4`" line is superseded by this record if accepted). C8 stays OPEN.

## Revisions

- **v1.1 (`4512309`, Codex review 5371784947 on `60e56ef`).** Thread 4149168080 (P2) taken: the `mehta` mutant now targets `Z_4` (`Z_5` for `Z_4`); it had altered `Z_3` and was rejected only through `D3_EXACT`. Thread 4149168070 (P1, replay workflow vs the 2026-09-27 owner stop) not taken, reasons on the thread (the stop's own resumption clause, the owner's later directives in `AGENTS.md` and `OP-WORKFLOW-20260930.md`, the workflow's path-scoped pull-request-only trigger, and the repository's practice of about thirty such workflows merged since the stop); the owner can direct removal. No certified value changed.

## Current disposition

- **Head / owner:** `4512309` on `claude/side24-d4-closed-form-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** the Proposition and Corollary above, proved by exact integration of the Math-#200 reduction; exact `m = 1, 2, 3` and `Z_4` regressions; the closed form at the centre of the independent certified enclosure. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** Codex code review at `60e56ef` (two findings, one taken, one declined with reasons). Requested: OpenAI / xAI nonauthor reads of Slice A (NOTE §1: reduction data, odd `k`, constants; L1), Slice B (§2 L2–L4 and §3: the T-function layer, the two-erf base integral, the by-parts recurrences, atoms and cancellations, the `Z_4` identity), Slice C (§§4–6: rules, mutants, containment, the coefficient table, the `D_5` non-claim).
- **Unresolved finding IDs:** 4149168070 (P1, workflow vs owner stop; declined with reasons, open for the owner or a nonauthor lane).
- **Validation:** 16/16 rules in `-B -S` and `-B -O -S`, byte-identical output; 14/14 mutants rejected in both modes; workflow replayed locally at `4512309`; hosted run on `60e56ef` green (`certify-and-reject-mutants`), run on `4512309` pending.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
