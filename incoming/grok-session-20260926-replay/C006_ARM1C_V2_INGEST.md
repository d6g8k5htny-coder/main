# C006 Arm1c-v2 ingest — instrument cycle, not a theorem

**Scientific effect:** NONE.
**Does not accept Theorem A, D5 pin, or a flat-saddle discriminator.**
**Source:** attached PDF `C006 Arm1c v2 Final Report.pdf` (2026-07-05).
Drive carrier: `1iXe04X9SZSPefAuZpEbEuQsJtk7stqCX`.
Governing snapshot sha256 `df35cb61d12eb667297e29d94028d1b7d3df507b3f2ff8b0afbd8a9bbf496d26`.

## Custody

Code search of `d6g8k5htny-coder/main` and `Math-` for `Arm1c`, `C006 Arm1c`, snapshot hash `df35cb61` returned **no public proof-vault file**. The report lives on Drive. This note is a pointer, not byte custody of the PDF.

## What the cycle actually did

1. **v1 D2 halt (correct).** Bilinear Hessian det at the extreme-flat two-bump saddle: gate `5e-3` relative, observed `8.62e-3`. No science evidence generated.
2. **Mechanism.** Bilinear interpolation truncates curvature
   `\delta g \approx -(h^2/2)[\theta_x(1-\theta_x)\partial_{xx}g + \theta_y(1-\theta_y)\partial_{yy}g]`.
   First-order det map `\delta\det = H_{yy}\delta H_{xx}+H_{xx}\delta H_{yy}-2H_{xy}\delta H_{xy}` predicts `|\delta\det/\det|=4.36e-3` vs observed `8.62e-3` (ratio `1.98`, within one order). Instrument floor is `O(h^2)` and blows up as `|\det|\to 0`. That is the flat-saddle tail regime.
3. **Quadratic refit killed.** Passed `5e-3` in 8 cases; failed pre-committed `10\times` margin on near-flat (`2.21e-3`) and interference (`3.16e-3`).
4. **Spectral instrument certified.** Worst suite error `8.9e-15`. Blind audit `N=1195`: RMSE `1.1e-15`, gate crossings `0`. Materiality amendment logged because IEEE rounding broke a strict CI-contains-0 test at `10^{-16}`.
5. **D2b-v1 SPEC-INVALID** (no saddle at separation `3.2`). Replaced by D2b-v2 at separation `4.2`. PASS.
6. **Arm1c-v2 freeze** changed only the instrument. Predictions/thresholds identical to v1.
7. **OUTPUT: ABORTED BY GATE.** C-2 soft-eigenvalue null failed in M-class (`0.637` vs required `[0.85,1.15]`). S-class `0.861` inside. P-1c-1 / C-1 / P-1c-2 passed as *uninterpreted numbers*. Frozen branch: C-2 fail ⇒ instrument-artifact suspicion ⇒ no interpretation.

## Mapping onto this account (do not flatten)

| Report object | GitHub object | Relation |
|---|---|---|
| Bilinear det floor on flat saddles | D5 pin / microdisk detector | Same *regime* (small `|\det H|`). Not a pin-neighborhood proof. |
| Spectral Hessian instrument | No public Math- instrument file | Engineering certificate. Not Theorem A. |
| C-2 `T_soft(M)=min|\lambda|` selector | Parent §6 soft factors / canceled-pivot | Different object. Parent keeps two soft factors on typed weights. Report `T_soft` is a detector statistic that can pick the *transverse* eigenvalue when `|f_{ss}(M)| < \kappa r`. Hypothesis only. |
| R3b `r^4\to r^5` standing registry | D1 §7 `E[W 1_{G^c}]=O(r^5)` skeleton | Same *power counting slogan*, not the same lemma. Report says R3b is independent of Arm1c. |
| P-1c-* passes | None | Uninterpreted by the report's own gate. |

## What this does **not** do

- Confirm a flat-saddle discriminator.
- Close D5 microdisk, D1 Theorem A, SIDE24 persistence reading, O2, or B2.
- License promoting P-1c-1 / C-1 / P-1c-2 into STATUS.
- Supply `TRANSVERSE_CONTACT_ASYMPTOTIC.md` or D0 carriers.

## Next registered step in the report

Detector Hessian audit: recompute `T` from the pair-axis-projected eigenvalue. That is a new frozen sub-cycle, not a finding of Arm1c-v2.
