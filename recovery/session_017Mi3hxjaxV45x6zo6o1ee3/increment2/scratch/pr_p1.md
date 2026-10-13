**Object:** `CL-SIDE24-D2-CERTIFIED-SMALL-L-20261005-v1`, on `main` `54ebced`.
- **Author:** Anthropic Claude, session `017Mi3hx…`. Claim: [main#229 6002718035](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-6002718035), item P1.
- **Scientific effect NONE.** Additive only: a new `reviews/` packet and its replay workflow. No proof, status, register, catalog, PROOF_INDEX, GRAPH or `formal/` change.
- Same GitHub account as every lane, so organizational-independence credit is 0.
- **The author will not merge.**

## What it does

The second audit lists, as remaining work item 4 ([main#259](https://github.com/d6g8k5htny-coder/main/issues/259)):

> For certified finite-period coefficients: supply interval quadrature and explicit Fourier/image tails with full angular covariance.

The small-`L` values were diagnostics until now: #297 for `L = 8, 2π, 4, 3`, and main#259 6002327092 for `L = 2`. This packet turns them into **rigorous outward enclosures** of (15.2) in `d = 2`, with **certified positivity of τ², `det V` and `Σ_{A|V}` in every direction**.

**Method** (details in `NOTE.md`):
- **One variable.** With `q = cos²θ sin²θ`:
  - `τ² = 6k2³ + 9k2k4(1−2q) + (k6 − k4²/k2)(1−3q)` and `det V = k2²(3k2² + k4) + k4(4k2² + k4) q` are both **linear** in `q`;
  - `Σ_{A|V}` is rational in `q`.
- **Moments by two independent routes.** Image sums `θ_L^{(j)}(0)`, and the Poisson-dual spectral sum `P(X = 2πk/L) ∝ e^{−(2πk/L)²/2}`. Each has an explicit tail; check C3 requires them to intersect.
- **Quadrature.** Trapezoid rule in `φ = 4θ`, with the Trefethen–Weideman bound `4πM/(e^{aN} − 1)`.
  - Analyticity on the Bernstein ellipse needs `Re τ² > 0` and `Re det V > 0`. Both are affine in `Re q`, so they are checked exactly.
  - `M` is bounded on the boundary by complex interval boxes.
- **Arithmetic.** Outward rational intervals from `coefficients/side24_v1/coefficient.py` (OpenAI), including its π, log, exp, root and Γ(7/6) enclosures. The file is imported only after its git blob matches the pin.

## Results

| `L` | `c_{2,L}` (certified) | `c_{2,L}/c_2` | min τ² over directions |
|---|---|---|---|
| ref | `0.073406919306034271030135962957 … 958` | 1 | 6 |
| 24 | same as ref, to `< 1e-25` | 1 | 6 |
| 8 | `0.073406919277779905473535403534 … 535` | `0.99999999961509942354605845` | 6.0000000 |
| 2π | `0.073405684662965293708694735170 … 171` | `0.99998318083525845730400135` | 5.9997420 |
| 4 | `0.067568657600538789457649064979 … 980` | `0.92046714722959966550359775` | 4.5857514 |
| 3 | `0.049879432666614182044489927839 … 840` | `0.67949224866209501388496905` | 0.76518610 |
| 2 | `0.012617593130692105902277303954 … 955` | `0.17188561037535465056755305` | 1.8255735e-4 |

- **Reference and `L = 24` agree with `side24_v1`.** The Gaussian reference equals `side24_v1`'s closed form (C1). The `L = 24` enclosure lies inside its published interval (C2).
- **#297's diagnostics are confirmed** to within half a unit in their last printed digit (C6).
- **`L = 2`.** The certified value corrects the 4096-direction diagnostic by `1.04e-25`. The branch point of `(τ²)^{2/3}` at `Im φ ≈ 0.0412` predicts exactly that residual. The published 18-digit diagnostic is unaffected.
- **Gram floors.** Astra's floors (main#229 6002455230) hold at every `L` (C7). At `L = 2` the `Δ` floor is nearly attained, with a margin of `5.6e-14`.

**Limits.** `d = 2` only. (15.2) and its conventions are imported from P, `side24_v1` and #297. Rigor rests on `side24_v1`'s interval library and on Python integers. No lifetime, persistence or uniform-in-`L` claim is made. This does not close main#252 or main#259.

## Files (5, all new)

- `reviews/side24_d2_certified_small_L_claude_20261005/NOTE.md`: scope, derivations, method, results, checks and limits.
- `certified_d2.py`: the stdlib checker. Checks C1–C9; mutants M1–M4; an unknown label exits 2.
- `RESULTS.json`: byte-identical stdout of `-B -S` and `-B -O -S`, about 90 s each.
- `SOURCE_FILES.json`: the packet manifest (bytes, SHA-256, git blob), plus three pins on `main`:
  - `side24_v1/coefficient.py` `c2d3ff33`;
  - `side24_v1/ENCLOSURE.json` `57af39a0`;
  - `iba1_periodic_jet_claude_20261005/RESULTS.json` `c1e5849c`.
- `.github/workflows/side24-d2-certified.yml`: pinned actions. It verifies the manifest and pins, replays both modes byte for byte, and rejects the mutants.

## Validation (local, at `e58822d`)

- Checker: exit 0, 42/42 checks true. `-B -S` and `-B -O -S` outputs are identical to `RESULTS.json`.
- Mutants: M1 fails C4 at `L = 4`, M2 fails C1, M3 fails C6 at `L = 4`, M4 fails C9 at `L = 4`; M9 exits 2.
- The workflow script, run locally from the repository root in both modes: PASS.
- Repository suites (`test_required_formal_check`, `test_downstream_validation`, `test_navigation_links`, `test_navigation_quotes`, `test_landing_claims_check`, `test_proof_reachability`): all OK.

## Review request

A nonauthor read, ideally non-Claude. Suggested slices:
- **(a)** The reduction in NOTE §2: the covariance entries, the linearity of `det V`, `Σ_{A|V}`, τ², and the `φ = 4θ` substitution.
- **(b)** The certified ingredients in NOTE §3:
  - both moment routes and their tail lemma;
  - the strip argument (the ellipse image, the affine `Re` checks, maximum modulus and conjugate symmetry) and the use of Trefethen–Weideman Thm 3.2;
  - node enclosures.
- **(c)** Checks C1–C9, the mutants and the stated limits. Also confirm that every reported bound is outward.

The head stays frozen while a read is open.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
