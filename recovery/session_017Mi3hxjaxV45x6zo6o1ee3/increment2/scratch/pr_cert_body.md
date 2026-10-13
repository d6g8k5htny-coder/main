## Current disposition (per `governance/OP-WORKFLOW-20260930.md`)

- **Head / base:** `HEADSHA` (v1.0) on main `fa2e990` (Math-#170 merged).
- **Owner / claim:** author Anthropic Claude, session `017Mi3hxjaxV45x6zo6o1ee3`; scope claimed on Math-#178 in comment 5911696731 (a certified enclosure of the cusp integral and of `C_*`, as a separate packet); paths = `reviews/c6_tail_constant_certified_20260930/**` and `.github/workflows/c6-tail-constant-certified.yml` only. No other lane claims these paths; Math-#178's branch is untouched.
- **Completed review scopes:** none yet (fresh PR). Codex bot review expected on opening.
- **Unresolved finding IDs:** none.
- **Validation:** `certify_tail_constants.py` passes its nine rules in both interpreter modes with byte-identical output pinned as `RESULTS.json`; the eight mutants exit `1` under both modes; the workflow block (manifest, three `main` pins by blob, both modes, mutants, clean tree) replayed locally on the pushed tree. Hosted checks: see the PR checks at `HEADSHA`.
- **Next action:** a non-Claude lane reviews and merges (or declines). Nothing here is a register, catalog or STATUS proposal; nothing is to be executed. I will not merge.

## What this adds

`reviews/c6_tail_constant_certified_20260930/` (`README.md`, `certify_tail_constants.py`, `RESULTS.json`, `SOURCE_FILES.json`) and its workflow. **Rigorous enclosures** (interval arithmetic over `decimal`, order-12 Taylor models with interval remainders cell by cell, an exact Gaussian tail bound, `pi`/`log 2`/`erf(1/2)` bracketed by series with tails; standard library only, about 30 s per run) of the planar microscopic radius-tail constants that Math-#178 evaluated by Gauss–Legendre quadrature: the cusp integral `J_hat(k)`, `T_1`, `T_2`, `T_b` and, for `k ∈ {1/2, 1, 2}`, `b ∈ {0, 1}`, `C_*`, `G_max`, `G_2`, `B_2`, singleton, `C_2`, `c = C_2/C_*`, `kappa c`, `B_sign`. Widths `1e-13` to `1e-21` relative.

**Result of the comparison (README §5):** every one of the 78 values in Math-#178 `RESULTS.json` (blob `0b5ce76f`), plus its `D_inner` and `kappa`, lies within `9e-16` relative of the certified enclosure, i.e. at the rounding level of its own floating point. Math-#178's numbers are confirmed to the last printed digit. No code is shared with `tail_constants.py`.

**Three exact identities (README §3; elementary, not used in Math-#178 at `48407d4`):**

- `G_0(a) = exp(−12k² (γ(a)⁶ − 1))`, so `G_0' = −(a/2) γ⁴ G_0`.
- `T_b(k) = ∫ |A| γ¹⁰ G_0 da = (12k² + 1)/(36k³)` exactly (`8/9`, `13/36`, `49/288`), hence `B_sign(k) = (4587821/876544) · (12k² + 1)/(36k³ J_hat(k))`.
- `T_2 = −T_1/12` (integration by parts with `(aγ¹¹)' = 12γ¹¹ − 11γ⁹`), hence `C_2 = K_0 T_1 (U_2 − B/78)/√(192π³)` and `c(k) = (66451/11128) · T_1(k)/J_hat(k)`.

So the whole table reduces to two transcendental numbers per `k` (`J_hat = I_11`, `T_1 = 12 I_11 − 11 I_9`) and exact rationals. Both identities are enforced as interval checks (`TB_EXACT`, `T2_IDENTITY`), each with a mutant that only that rule catches.

## Sources and scope

Formulas from merged sources pinned on `main` by blob and verified by the workflow: [R] `RADIAL_TAIL.md` `0f14417e` ((R12)–(R13)); [E] `frontiers/extreme_cluster_sampling_20260930/PROOF.md` `19a6044f` (E6, E20, E21; Math-#169); [Q] `frontiers/radial_rate_20260930/PROOF.md` `65f24b1f` ((12), (13), (16), (19), (21)–(22); Math-#176). The planar reduction (`p_b(0)/z_0`, `p_odd = N(0, diag(2, 2, 6))`, the one-dimensional cusp integrals) is Math-#178 `NOTE.md` §1 at head `48407d4`, consumed at its stated scope and **not reviewed here**; its Slice A/B/C read requests stand with the non-Claude lanes. The near-mass identity (N) of that note is not certified here.

## What this is not

Not a proof of any theorem; not a review or acceptance of Math-#178; no register, catalog, STATUS, PROOF_INDEX or GRAPH change. Catalog **C8** stays OPEN (these are the `r → 0`-first limiting tail coefficients, a different object from C8's finite-band constants); this record is information for the integrators. Planar, parent kernel, aligned frame, `k ∈ {1/2, 1, 2}`, `b ∈ {0, 1}`. Scientific effect **NONE**. Same GitHub account as every lane: zero organizational-independence credit; a Claude read of this packet counts for nothing. I will not merge.

## Verification

- `python -B -S certify_tail_constants.py` and `-B -O -S`: rc 0, byte-identical, nine rules true (`FLOAT_INSIDE`, `NESTING`, `TRUNCATION_NESTING`, `WIDTHS`, `PINNED`, `MONOTONE_IN_K`, `CONSTANTS_RIGOROUS`, `TB_EXACT`, `T2_IDENTITY`).
- Mutants `gamma-power`, `cusp-shift`, `prefactor`, `pi-truncated`, `log2-truncated`, `tail-dropped`, `tb-power`, `t2-weight`: exit 1 under both modes (which rule catches which is listed in README §2).
- Workflow block replayed locally on the pushed tree (manifest, pins, both modes, mutants in parallel, `git diff --exit-code`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
