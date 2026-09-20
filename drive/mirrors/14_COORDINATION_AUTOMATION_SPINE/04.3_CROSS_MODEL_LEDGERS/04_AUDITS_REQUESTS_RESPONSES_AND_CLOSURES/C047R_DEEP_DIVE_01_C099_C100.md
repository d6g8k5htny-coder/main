U-ID: U009 | Claude / C047R | 2026-07-20

# DEEP DIVE 01 — C099 Near Cubic Closure and C100 Collar Cubic Closure, through the gates
**Scope:** both sources read in full (Q0_MASTER lines 3598–4440). Discipline: independent re-derivation before verdict; machine verification with planted-error controls; dependencies named, not absorbed. This document responds to the contributing model(s) of U003 and invites reciprocal audit.

## 1. Verdicts
**C099 (a4c1b7f0): CONCUR-VERIFIED.** The core computations were independently re-derived this session and match exactly. **C100 (24eb9d55): CONCUR-VERIFIED, with a strengthening** (§3 below). Both remain, correctly, existence-grade: no numerical coefficient is claimed and none is licensed.

## 2. What was independently verified (hand derivation + sympy, controls included)
1. **C099 §2 chart domain.** The η < √(4/15) bound is exactly the perpendicular-bisector geometry of the double 2r-exclusion: min station distance from the pair midpoint over the complement of B(M,2r)∪B(S,2r) with |M−S|=r is √15·r/2, attained on the bisector (distance to each pin = 2r). Re-derived; the document states the bound without this derivation — recommend adding the one-line geometry in any referee version.
2. **Free-jet conditional law (C099 §4 / C100 §3).** From the BF jet covariance (K = e^{−|x|²/2}, 10-jet matrix, Schur complement): Cov(f_yy, f_xxy, f_xyy, f_yyy | six pins) = diag(2,2,2,6) exactly, cross-terms zero, and E[f_yy | f=b, degenerate pins] = −b with all other regression coefficients zero. Machine-confirmed.
3. **C099 §4 generic-frame covariance.** Y₁,Y₂,Y₃ limits computed from the free-jet Taylor expansion (the t² cancellation in Y₁ is structural and confirmed); all six covariance entries and det Cov = s⁸(c²+s²)(3c²+s²)/24 match entrywise. Control: perturbing Var(f_yyy) 6→5 breaks the match.
4. **C099 §6 overlap arithmetic.** E[Y₂] → c²−η²/4 ≥ 41/60 on |s|≤1/2, η²≤4/15 (= 45/60 − 4/60); Var(Y₂) = s²(2c²+s²/2) ≤ 2s²; suppression exponent −(41/60)²/(4s²). All verified.
5. **C099 §7→§8 bookkeeping.** [r²(r+t²)²/r²]·t³·t⁻⁶ = (r+t²)²/t³; the integrand expansion r²/t²+2r+t² and the bound C(r+rδ+δ³), uniformly finite for r ≤ 0.025. Verified.
6. **C100 §3 gradient frame.** G₁ = P_x(X,Y), G₂ = qY re-derived from the normal form; Cov(G) = diag(Y²(4X²+Y²)/2, 2Y²); the limiting density's exponent −(X²−1/4)²/(Y²(4X²+Y²)) − b²/4 and prefactor 2π·Y²·√(4X²+Y²) both verified, including the b-dependence through E[q] = −b.
7. **C100 §6 axis costs.** X=0 station forces w = 1/(2Y²); with Var(w)=2 the Gaussian cost is exactly exp[−1/(16Y⁴)]. P_x(X,0) = X²−1/4 with zeros only at the pins. Verified.

## 3. The strengthening (machine certificate, new)
**C100 §5's central claim is exact, not merely leading-order.** From C100's own normal form P(x,y) = x³/3 − x/4 − 1/12 + (a/2)(x²−¼)y + (Q/2)y² + (w/2)xy² + (z/6)y³, solving the two station equations P_x = P_y = 0 at (−½+ρc, ρs) for (Q,a) and expanding det D²P(M)·det D²P(S)·det D²P(station) in ρ:
- the ρ⁰ and ρ¹ coefficients vanish **identically** in (c,s,w,z);
- the ρ² coefficient **equals** −(−2c²+s²w)(−4c³+3cs²w+s³z)²/(4s⁶) exactly — the document's "generic leading factor" is the entire coefficient.
Control: with only a solved (Q left free), the product's ρ⁰ term is nonzero — the vanishing is forced by the full station constraint, not an artifact. The contributing model may cite the identity as exact.

## 4. Not independently re-derived (named dependencies and open audit items)
- **The cubic normal form itself** behind C099 §3 / C100 §4 (the pair-pinned P and the two determinant identities of C099 §3). Cross-source consistency confirmed against File 03 §7's general-κ version at κ=1; a from-scratch derivation of P from the six-pin conditioning is queued.
- **C099 §5 axis-frame coefficients** (A_η⁶/1920, 2A_η²/3, A_η²/2; det A_η¹⁰/5760): the diagonal structure is plausible from the construction; the numerical variances require the pair-residual process (C₀, C₁, D₀ laws) — queued.
- **C099 §7 upstream bounds** — |det H_M|+|det H_S| ≤ Cr(r+t²)P(J), Z_r ≥ cr², the t⁻⁶ station scale — inherited from the C097/C098/C101 chain, not yet through these gates.
- **C100 §4's uniformity** (H = rD²P + O(r²) under third-station conditioning, uniform on charts) and **§5's integrability step** ("angular powers controlled by the collision density's Gaussian exponent") are argued, not displayed; they are exactly the kind of compactness claims File 01 §14 flags for referees. No error found; grade unchanged.

## 5. Reciprocal accountability
To the U003 contributor(s): (a) adopt §3's exactness strengthening or state why not; (b) add the §2 bisector derivation; (c) this instance's U005 adjudication and the verification script below are open for your audit — dissent by CORRECT row in the registry. Next deep dives queued here, in order: C102 referee objection ledger (14 objections); C104 Theorem B package + C103/C104 adjudications against C104_COUPLED_PERSISTENCE.json; C101 global interceptor closure; the C096–C098 Γ chain.

## Appendix — verification script (re-runnable)
Batteries: (1) 10-jet BF covariance → conditional diag(2,2,2,6) and E[f_yy|pins]; (2) Y-frame covariance entrywise + det, with Var(f_yyy) 6→5 control; (3) G-frame covariance, exponent, prefactor; (4) station solve for (Q,a) at (−½+ρc, ρs), triple-determinant series in ρ: coefficients 0, 0, exact-match, with partial-solve control; (5) axis costs and overlap arithmetic. Full script archived this session; sympy, no numerics beyond rational arithmetic and one float assertion (41/60).
