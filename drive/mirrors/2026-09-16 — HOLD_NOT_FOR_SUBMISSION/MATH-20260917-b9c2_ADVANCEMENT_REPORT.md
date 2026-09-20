# Mathematical advancement report — 17 September 2026

Task: DQ-MATH-20260917-b9c2. Status: additive, author-side results and repairs. The overarching research problems remain open.

This session recovered the current Drive routing and source-bound work, advanced the H5 and RN-UNIF analytic interfaces, and proved an additional conditional combinatorial covering result. It also found a falsifying example for an archived H5 Taylor enclosure and repaired an RN adaptive-cover bookkeeping defect. The accompanying proofs, code, source extracts, audits, and receipts make the advances reviewable and reproducible.

## Results by research front

| Front | New result | Scope still open |
|---|---|---|
| H5 small-separation jets | Exact-torus cancellation-free c2 polynomial enclosure, generic stencil construction, and uniform normalized pair-covariance enclosure on [0,0.05] | Full specified 24-jet interface, complete conditional/operator pipeline, and propagation of the source defect |
| RN-UNIF | Exact whitened Gaussian derivatives through order four; rational interval Schur propagation; a square-root-safe modulus; accepted-leaf aggregation repair | Certified field-wide input jets, all other factors in κ_far, complete cover and Piece-2 integral |
| Finite-family covering / prize line | Core-palette sufficient criterion extending the inspected P12-F condition, with a strict nonvacuous example | Existence of useful cores in general; unrestricted conjecture; active P13 weighted-gate work |
| Other routed research | Read-only current-state and routing review, including P0.2 review obligations | No new result or status promotion claimed for these fronts |

This was a bounded review of the controlling sources and relevant branches, not an assertion that every mathematical file in Drive was read or that all open problems were solved. The older forward plan's prize pointer lagged the Phase12/13 material; the newer material governed the narrow extension. Existing frozen artifacts and theorem labels were preserved.

## 1. H5: a stronger enclosure and a confirmed source defect

For the exact side-24 torus kernel K1, write a_(2j) for its even spectral moments and c2(r)=(1−K1(r))/r², extended continuously at zero. The new proof gives, uniformly for every r≥0,

    −a12 r^10 / 12! ≤ c2(r) − P8(r) ≤ 0,

    P8(r)=a2/2−a4 r²/24+a6 r⁴/720−a8 r⁶/40320+a10 r⁸/3628800.

On [0,0.05], the residual is below 2.120×10^−18. Both numerator and normalization tails are enclosed. The exact torus moments are retained: even the approximately 9.65×10^−123 difference between a2 and 1 is resolved. The first prototype band's enclosing range width is approximately 3.90564×10^−5, about 659 times tighter than its former slab hull. That range width and the much smaller polynomial residual measure different quantities.

The normalized pair construction encloses all 78 distinct entries of W G12 Wᵀ, with W=diag(T,I6), over [0,0.05]. Exact rational cancellation removes every negative Laurent power. A rational preconditioner B proves

    ||I−BA||∞ < 0.003125196,
    ||A^−1−B||∞ < 0.028214935,
    λ_min(A) > 0.1107638,

for the six-dimensional normalized pin block A, including r=0. This establishes this block's uniform conditioning; it does not prove every downstream H5 estimate.

The archived `Lattice.lambda_q` instead sums signed odd powers of frequencies. Symmetry cancels these terms, producing an invalid near-zero derivative majorant. An actual `KernelSeries.tm_factor` example has endpoint error about 7.285×10^−21 while its claimed remainder is only about 1.686×10^−119. A separate image-space/Hermite computation confirms the failure. Absolute moments give a valid replacement bound. This refutes the archived primitive enclosure, not the mathematical theorem itself; every dependent certificate needs its own impact assessment and replay.

The additive one-line source repair and actual-consumer checks are in `h5/H5_LAMBDA_REPAIR.md` and `h5/repair/`. The repair passes in both ordinary and optimized Python. Read also `h5/H5_ANALYTIC_ADVANCE.md`, `h5/H5_NORMALIZED_PAIR_BAND.md`, and the `h5_audit` records.

## 2. RN-UNIF: complete Gaussian calculus and a route through zero

After one fixed whitening, let p=N(m,S), q=N(0,I), A=I−S, and 0<S<2I. Completing the Gaussian integral gives

    log(1+χ²(p||q)) = −½ log det(I−A²) + mᵀ(I+A)^−1m.

The implementation supplies the ordered, noncommuting matrix derivatives through order four, complete exponential and square-root composition, and rigorous rational norm envelopes conditional on valid uniform input bounds. It also propagates the actual Schur identities A=F D^−1 Fᵀ and m=F D^−1z with all inverse and mean derivatives included.

There is a real zero-set obstruction: with S=I and m=t, √χ²=√(e^(t²)−1) behaves like |t|. Thus an unconditional fourth derivative of √χ² does not exist. The separate Hilbert-space argument uses

    √χ²(p||q) = ||p/q−1||_(L²(q))

to prove a finite Lipschitz bound through these zeros. It gives an exact Gaussian score-norm expression and an Arb-evaluated scalar majorant, conditional on uniform covariance/mean and first-derivative bounds. The parent cover's adequacy is not inferred from these formulas.

The adaptive engine also retained rejected parent-cell bounds in its final maximum. The additive patch records only accepted leaves. A regression using the actual control flow has 2,307 accepted leaves after one subdivision: the original rejects the valid refined cover; the repaired code accepts it. A genuinely overbudget terminal leaf still fails. These are mock-bound control-flow tests, not an RN field certificate.

Read `rn_math/RN_WHITENED_JET_THEOREM.md`, `rn_modulus/SQRT_CHI2_MODULUS.md`, `rn_engine/README.md`, and `rn_audit`.

## 3. Combinatorics: a conditional core-palette extension

For a finite decreasing family with nonnegative prices c_i≤min(1,−log(1−q_i)), choose a properly t-colored core of its nonsingleton minimal-forbidden hypergraph. Reserve these t colors and give the residual vertices L=K−t≥2 disjoint colors. Crossing edges then cannot be monochromatic.

On each component of the **induced** residual hypergraph, let r_j be its minimum edge size and π_j the probability that none of its vertices is selected. If

    L^(r_j−1) π_j ≥ 1

for every residual component, the proved cover cost is at most min(1,−log μ_q(A)). The argument retains necessary zero-price generators and rederives the exact-cell probability estimate rather than assuming acceptance of P12-F.

A family of all r-subsets except one, on K(r−1)+2 vertices, gives a strict extension of the old sufficient condition: the old condition fails, the core condition holds, the K-obstruction is nonempty, and the probability bound is below the trivial cap. This is a conditional elementary result, with no asserted novelty or solution of the unrestricted conjecture.

Read `prize/CORE_PALETTE_001_PROOF.md` and `prize_audit`.

## Verification and evidence limits

The package contains analytic proofs, directed interval or exact-rational bounds where claimed, implementation comparisons, explicit negative controls, and scoped reviews. Numerical point probes are labeled as such. RN validation includes 123 author checks, 139 separate implementation comparisons, 35 analytic/quadrature checks, and 23 modulus checks. These counts are checks, not independent proofs or organizationally independent reviews. H5 includes an alternative image-space calculation as well as spectral arithmetic.

All reviewers in this session belong to the same OpenAI line. Shared reconnaissance and source exposure are disclosed. No organizational-independence gate, frozen-package membership, external release, or parent theorem status is promoted.

## Next mathematical obligations

1. **H5:** trace use of the false odd-moment bound, apply the reviewed repair to explicitly identified successors, and certify the complete 24-jet/conditional/operator chain using the normalized band construction.
2. **RN-UNIF:** feed the new calculus or modulus with certified cell-wide exact-field bounds, then bound every remaining κ_far factor, execute complete accepted-leaf coverage and Piece-2 integration, and discharge the separate uniform-in-r requirements.
3. **Covering theorem:** obtain the required distinct-provider review of the exact new proof and determine when useful cores exist. Preserve separation from the currently active P13 weighted-gate scope.

The evidence bundle's SHA-256 manifest binds the exact delivered files. Source identifiers and hashes are included in the manuscripts and intake records; fresh candidate outputs supplement the original sources.
