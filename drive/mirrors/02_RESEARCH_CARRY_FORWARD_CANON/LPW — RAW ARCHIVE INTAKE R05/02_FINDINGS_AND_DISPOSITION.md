# LPW-R05-FINDINGS — exact repair targets, not blanket withdrawal

## F1 — wrong headline decimal and missing end-to-end check (AMEND REQUIRED)

The exact incoming constant denominator is B3_rat=3790446482793, while earlier prose shortened it to 3.79e12. Because the shortened number is SMALLER, it cannot replace an upper denominator bound in a probability lower bound.

The exact coefficient supplied by the package is

    260/(3790446482793 * 2^40 * 10^21)
    = 13/208381999114677270242918400000000000000000000
    = 6.2385427029355877929430410845189...e-44.

Therefore `6.239e-44` is too large for this chain; `6.238e-44` is downward-safe. The repeated `6.239277637492383...e-44` is the value obtained from the smaller rounded 3.79e12 denominator.

The author-side prior arithmetic confirmations repeated the rounded-denominator value. Those confirmations are expressly corrected here. The defect affects the displayed licensed lower floor, not a demonstration that the true pairing-defect probability violates 6.239e-44.

The received certifier prints/checks the weaker 10^-44 power floor, not the promoted 6.239e-44 headline. Its falsifier also contains a vacuous `... or True` guard. Require an exact rational assertion of every exported numerical lower bound against the same authoritative fraction. The new interval companion rejects `--claim 6.239e-44` with nonzero exit in both modes.

Radius: K_rat=9432 and r0=1/2414592 remain consistent; 256*K_rat*r0=1 and 16/1024+8*K_rat*r0=3/64<1/16.

## F2 — false Gaussian fourth-moment identity (LOAD-BEARING PROOF AMENDMENT)

The report §3(5) and `lpw_constant.py` claim, for independent standard normal xi,eta,

    E(|xi|+|eta|)^4 = 12+16/pi.

The correct value is **12+32/pi**. The two mixed cubic-linear contributions each equal 16/pi; both are needed. Thus the report's particular majorant proof understates its fourth moment. Replaying the same code cannot repair that mathematical error.

A constructive new repair is supplied in 03_RAYLEIGH_REPAIR_AND_INTERVAL_CERTIFICATE.md: use rho=sqrt(xi^2+eta^2) as the uniform real-mode amplitude. Then E rho^4=8 <=12+16/pi and E rho=sqrt(pi/2) <=2sqrt(2/pi). The old numerical budgets become valid upper bounds under the NEW majorant. This is an explicit proof amendment, not a relabeling of the false identity as true.

A new directed-interval implementation recomputes the chain with full infinite tails and the changed majorant; it gives the same integer B3 and K ceilings and the corrected 6.238e-44 floor on the same radius. Kimi must inspect and acknowledge this new lemma/implementation before the improved certificate receives an unqualified repaired review status. Original files stay frozen. Qualitative LPW and QC-RETURN03 do not consume this false fourth-moment identity and are not retracted.

## F3 — unconditional versus conditional covariance (DOCUMENTATION AMENDMENT)

The lead constant review describes the raw Cov(J) as diagonal. For
J=(f_yy,f_xxy/2,f_xyy/2,f_yyy/6),

    Cov(J_A,J_D3)=a2*a4/12 != 0.

The conditional Schur complement Cov(J|U0) is diagonal in the moment-dependent formulas; the unconditional matrix is not. The original CODE does include the nonzero entry in its Gershgorin bound, and the new interval program does too. The approximately 3.0 spectral upper bound survives. Correct the explanation without claiming this reporting defect invalidates a correctly computed bound.

## F4 — W8's final run is not a complete PASS

The delivered twin v3 transcripts are byte-identical and end with:

    FAIL-CLOSED TRIGGER: used core spacing violates kill condition at floor margin

The earlier refined-rung ratio 1.8846416 passes. The report separately claims the intermediate difference S-E_cert approximately 0.40371620975 conditional on H-B3 and at the r->0 limit object. It expressly treats the later excursion/spacing failure as a separate mechanism; do not infer that the intermediate conditional claim is false merely because that mechanism failed.

Equally, do not label the complete executable certificate ALL PASS: its own completion lines were never reached. Record stage results separately. Explain which gates are necessary for each claimed subresult. Any reclassification of the kill gate as optional must be explicit and reviewed, not removal of a failing assertion after seeing output.

S and E displayed with finite digits do not certify a strict exact lower decimal 0.403716209750; supply directed lower/upper values and a downward-rounded result. The reported “positive r-uniform constant for the r->0 limit object” conflates scopes. A limit-object inequality still needs a separate quantitative transfer for any finite-r assertion. H-B3 remains unproved; a passing rung ratio does not prove a cellwise derivative supremum.

## F5 — certificate arithmetic versus numerical-enclosure method

The incoming source uses ordinary 60-dps mpf arithmetic plus a blanket ROUND=1e-45. It is not a directed-interval engine. A rigorously established global rounding budget could in principle validate such a method, but the explanation that all quantities are <=1e6 is inconsistent with the final approximately 3.8e12 B3 computation. The present review does not infer a wrong number from this discrepancy; it requests a complete propagated error bound or actual directed enclosures.

The supplied R05 program uses mpmath.iv for finite arithmetic and transcendental evaluations, carries exact binary-rational endpoint receipts, and includes analytic infinite-sum tail bounds. This resolves the finite-arithmetic concern for its own new result; it is not a retrospective alteration of the received implementation or an independently machine-checked formal proof.

## F6 — domain wording

The stronger-radius interval (0,1/2414592] already contains (0,10^-28]. The fallback does not extend the tiny-r domain; it is a different conservative bound and proof route on a subinterval. Preserve its independent value without a false interval comparison.

## Operational effect

Retain: received qualitative APPROVE, R3/S1 confirmations, operator acceptance, received fallback analytic PASS, QC reproduction, corrected RB evidence, existing 3D ratification.

Update: raw current package custody is now received and manifest-verified; original-body rule needs one additive correction; improved constant package AMEND REQUIRED at F1/F2/F5; new corrected author-side certificate candidate supplied; W8 staged conditional progress plus final FAIL-CLOSED recorded.

Do not infer: failure of qualitative LPW, a two-sided 2D theorem, a limiting coefficient, human peer review, any P0.1 Boolean or RP transition, closure of H-B3/WP/exit/Bonferroni, or full Class-3 eligibility from owner permission alone.
