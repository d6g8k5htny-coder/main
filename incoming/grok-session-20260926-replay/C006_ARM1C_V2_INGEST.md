# C006 Arm1c-v2 ingest — instrument loop, not a theorem

Scientific effect: NONE. Does not accept Theorem A, D5 pin, R3b, or any persistence discriminator.

Source: user-attached PDF `C006 Arm1c v2 Final Report.pdf` (82,688 bytes), dated 2026-07-05.
Drive twin: `1iXe04X9SZSPefAuZpEbEuQsJtk7stqCX`.
Governing snapshot sha256 `df35cb61d12eb667297e29d94028d1b7d3df507b3f2ff8b0afbd8a9bbf496d26`.

GitHub search across `d6g8k5htny-coder/{Math-,main}` and org-wide for `Arm1c`, `C006_Arm1c`, `T_soft`, `R3b`, `ABORTED BY GATE`: **zero hits**. This cycle is **ABSENT from public GitHub**. Do not invent a proof body for it.

## Exact output of the frozen run

    OUTPUT: ABORTED BY GATE

because C-2 failed in the M-class: T_soft S = 0.861 (inside [0.85, 1.15]), M = 0.637 (outside). Frozen branch table: C-2 fail ⇒ instrument-artifact suspicion ⇒ no science interpretation.

Recorded, uninterpreted numbers:

    P-1c-1 ratio 0.643, CI [0.464, 0.784]     PASS as a number
    C-1     +0.292, CI [0.141, 0.429]         PASS as a number
    P-1c-2  weighted 2.212 (unweighted 1.271) PASS as a number

Holm p-values are logged. They do not license a discriminator theorem.

## What the report itself says is proved

1. Arm1c-v1 D2 halt is explained: bilinear interpolation truncates pure curvature, δdet/det predicted 4.36e-3 vs observed 8.62e-3 (ratio 1.98). O(h²) floor, unbounded as |det|→0. Fatal on the flat-saddle tail. Halt stays on record (hash 0529b88c…).
2. 9-point quadratic refit killed at the pre-committed 10× margin on the two cases that matter (near-flat 2.21e-3, interference 3.16e-3).
3. Spectral / trigonometric-interpolant Hessian: suite worst 8.9e-15, ~10¹¹× margin; blind N=1195 audit RMSE 1.1e-15, zero gate crossings. D2b-v1 was SPEC-INVALID (no saddle). D2b-v2 at separation 4.2 PASS. Original D2 field with spectral instrument: 1.80e-6, position error 0.
4. Governance killed its own constructions four times this loop.

## What it explicitly does not prove

- Flat-saddle discriminator (blocked by C-2).
- r⁵ inner-zone upper bound (independent lemma; Arm1c does not touch it).
- O2 (annulus γ=p−1) and B2 (adjacency).
- Reconciliation of unweighted slope 1.27 here vs 1.44 in A1.

## Mapping onto the public D1/D5 ledger (no status transfer)

The C-2 audit *hypothesis* (not a finding): at a maximum both eigenvalues are negative, soft = min|λ|. In the R3b flat-saddle tail, |f_ss(M)| can fall below κr, so the *detector selector* picks the transverse eigenvalue instead of the fold axis and depresses T_soft(M) only in the window class.

That is a **finite-grid selector convention**. It is not parent G_r, not the canceled-pivot bound (6.1), and not the A7 depth-failure integral. Those objects live on Math- default and remain AMEND for their own reasons. Do not import Arm1c-v2 PASS/FAIL into Theorem A or D5 pin.

Registered next step in the PDF: recompute T from the pair-axis-projected eigenvalue under a new frozen sub-cycle. That work is not in GitHub.

Standing registry named by the PDF and unchanged: K1–K4, R3 (exp-small inner claim), R3b (r⁴→r⁵ correction). Those names are also **not** on public Math- default as proof files.
