# Paper 1 note — short-lifetime asymptotics

**Scientific effect: NONE.** This file is a reading note for a future paper. It does not update STATUS.md, claims/LANDING_CLAIMS.json, or any acceptance register. A merge of this note is not mathematical acceptance.

**Title:** Short-lifetime asymptotics for H0 persistence of smooth Gaussian fields.

Not "Universal Law."

## Conditional statement

Let f be the variance-one periodized Bargmann-Fock field on the flat torus of side 24, in dimension d in {2,3}. Under the compact-birth and compact positive-gap hypotheses of D1, and under Hypothesis P below, the first-moment density of finite nonessential superlevel H0 bars is proposed as

nu_{d,24}(ell) = c_{d,24} ell^{-1/3} (1+o(1)) as ell -> 0+.

The coefficient c_{d,24} is the Kac-Rice / cone integral in Math-/coefficients/side24_v1. Copy the enclosure from ENCLOSURE.json or PROOF.md at the commit cited when the paper is written. Do not retype a decimal from memory. SIDE24 acceptance is arithmetic. It is not acceptance of the parent lifetime theorem and not acceptance of pairing.

The o(1) is the D2 remainder relative to the leading term, at existential scope. No numerical remainder cutoff is claimed. No 24-jet certificate is claimed. No other covariance, and no H_k for k>0, is claimed.

## Hypothesis P — not proved

> P (not proved). For this model and these marks, the intensity of typed max-saddle contacts of fold-gap less than ell and the intensity of elder-paired finite nonessential H0 bars of lifetime less than ell share the same leading coefficient. Equivalently, after ell ~ r^3, the integrated probability that a typed contact is not the elder pair is o(1) relative to the leading intensity.

D1 Theorem A is a pointwise bound 0 <= 1-p_r <= C r^3 on compact marks. It is not this integrated statement.

The shrinking witness-collision node remains OPEN_ACTIVE. Local D4/D5 estimates do not close global pairing.

frontiers/local_elder_geometry_20260930/PROOF.md is author-side / HOLD. It may be cited as a candidate. It is not a lemma of this note.

## Scope box (not enlarged)

Inputs that may be used, at their existing scopes: D1 A/B/C existential, D2 Theorem R, D3 arithmetic only, SARD-G as genericity for a fixed planar law, C6 planar factorials as tools.

Not claimed: global pairing, a numerical constant C, an R^d limit, H1/H2, a cosmology detection, Lean of Kac-Rice.

## Section map

1. The hole. The Gaussian kinematic formula gives Euler characteristics. Bobrowski-Borman gives the Euler integral of persistence. Chazal-Divol gives existence of a density. Feldbrugge-van de Weygaert-Pranav gives numerical fits. Missing: a near-diagonal H0 intensity with pairing.
2. Model. Periodized Bargmann-Fock, Morse input from the SARD-G successor, elder rule as the topological convention (Curry, arXiv:1706.06059).
3. Local fold. ell ~ r^3, Jacobian ledger. Source: D1 parent packet.
4. Hypothesis P, and what D4 / D5 / C6 already give at the two ends.
5. Intensity. D1 B/C plus D2, conditional on P.
6. Coefficient. SIDE24, labeled arithmetic.
7. Numerics. Not yet. A histogram that disagrees is a result.
8. Open. Witness-collision, higher d, cluster law, infinite-volume interchange.

## Outreach draft — do not send from this repository

A later human email may ask whether the note is (i) a persistence statement, (ii) a candidate-contact statement, or (iii) not yet either. No email is sent by this file.
