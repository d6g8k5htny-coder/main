# KIMI INDEPENDENT REVIEW VERDICT — LPW-CAND-20260912 (local-path candidate, existential cubic lower bound)

Reviewer: KIMI line (lead coordinator + four independent interface reviewers RA/RB/RC/RD).
Date: 2026-09-13. Target: 02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md v1.0, body sha256
cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5 (matches the package MANIFEST).
Review packet: 03_INDEPENDENT_REVIEW_PACKET.md (six interfaces, separate dispositions required).
Exposure disclosure: the full candidate and packet were read before review — ordinary hostile review,
not blind reconstruction. Review lineages: five (lead + RA + RB + RC + RD), each with its own
independent scripts; no author-script lineage consumed.

## VERDICT: APPROVE — all six interfaces PASS WITHIN SCOPE at the exact file version.

The candidate's theorem LPW is endorsed: **there exist c > 0 and r₀ > 0 such that
1 − q(r, 6/5) ≥ c·r³ for all 0 < r ≤ r₀**, under the exact normalized periodized side-24
two-dimensional Bargmann–Fock field and the exact six-pin typed maximum–saddle pair-Palm law
(pins f(M_r) = 6/5, f(S_r) = 6/5 − r³/6, ∇f = 0 at M_r = (−r/2, 0), S_r = (r/2, 0); weight
W_r = |det H_M det H_S|·1{H_M ≺ 0, det H_S < 0}; q(r, b) = P_r{D_f(M_r) = S_r}, no adjacency
conditioning, η_r untouched).

## Interface dispositions (separate, per the packet's requirement)

| interface | reviewer | verdict | file (sha256) |
|---|---|---|---|
| R1 law and topology | RA | PASS WITHIN SCOPE | RA_R1_R6.md 3b7cda65… |
| R2 six-/ten-coordinate Gaussian limit | RB | PASS WITHIN SCOPE | RB_R2.md 07ceb63c… |
| R3 conditional C⁴ norm control | RC | PASS WITHIN SCOPE (2 non-blocking amendments) | RC_R3.md 29dcbc23… |
| R4 deterministic lift | RD | PASS WITHIN SCOPE | RD_R4_R5.md 09aaac1d… |
| R5 normalizer and final powers | RD | PASS WITHIN SCOPE | RD_R4_R5.md 09aaac1d… |
| R6 scope and status | RA | PASS WITHIN SCOPE | RA_R1_R6.md 3b7cda65… |
| lead's independent evidence pack | lead | consistent with all six | LEAD_EVIDENCE_PACK.md |

## What was independently verified (each by at least two independent lineages)

1. **The exact witness algebra**: F*'s pinned maximum/saddle (det 6, −14), the path height
   R(x) = (2x+1)²(12x²+8x−17)/240, the exact identity R + 99/1280 = (4x+5)²(48x²−40x+1)/3840,
   min R = −99/1280 > −1/6, endpoint 9/16 > 0, clearance 103/3840 > 0 under ε = 1/16,
   Hessian interval bounds (81/16, −1673/128), weight product ≥ 65.
2. **The topological implication** (R1): a strict local maximum at b with a path to a point above b
   whose minimum exceeds f(S) cannot die at S — proved tie-free; all four adversarial escapes fail;
   the needed assumptions (a.s. Morse + distinct critical values, measurability) are already the
   estimand's definitional package; NO Morse–Smale is consumed.
3. **The Hermite normalization** (R2/lead): V row-major with det −1; the displayed T_r with
   det −r^{−5}, invertible for all r > 0; u_r = (b − r³/12, −r²/4, 0, 0, 0, 1/3); U_r → U_0 in
   Gaussian L² with covariance convergence (exact through cubics; trapezoidal cancellations exact).
4. **The ten-jet nondegeneracy** (R2): the spectral-support argument is sound, and numerically the
   exact field's ten-jet covariance at 0 has certified λ_min = 0.126553449667289 (eigen-residual
   6.9e-50, spectral tail ≤ 1.2e-185); the r = 0 Schur complement is diag(2, 1/2, 1/2, 1/6) at
   machine precision; rungs r = 0.5…0.025 converge to the endpoint (λ_min → 0.12656); conditional
   means → (−b, 0, 0, 0).
5. **The density lower bound** (R2): the argument chain (uniform PD Schur + bounded
   determinants/inverse/means on the compact parameter set [0, r_G] × K_J) is sound; evidence at
   r = 0.025: density of J at the thin-box center 1.2832e-3, minimum over 16 corners 1.2781e-3.
6. **The conditional C⁴ moment** (R3): all six attack links held — the regression version is
   everywhere-defined (no version trap), the representers' C⁴ norms are uniform (Cauchy–Schwarz +
   covariance convergence, no hidden r-blowup through the Hermite frame), the residual's moments are
   finite and v-independent, Γ_r^{-1} is bounded on [0, r_G], the Markov 1/2 bound is pointwise in j,
   and the jet-box integration is measure-theoretically clean (no independence asserted, no
   subtraction-order error).
7. **The Taylor lift** (RD/lead): all six 1D pin identities exact (1/24, 1/48, 5/16, 1/128,
   23/384, and the y-equation Kr/48); the 2D remainder (9/2)Kr; monomial norms 1,2,1,4,2,12;
   total 2089/384 < 8 (reproduced term-by-term); basis norms 4,4,2,6; tolerance 3/64 < 1/16;
   Hessian scaling r per matrix, r² per determinant, r⁴ per pair.
8. **The normalizer and power ledger** (RD): Z_r ≤ 4B_3·r² = C_Z r² with correct polarity;
   Z_r > 0 grounded by the constructed event (no circularity); E_r ⊂ K_J for r ≤ 1;
   volume 32δ⁴r; numerator 1040mδ⁴r⁵; quotient c = 1040mδ⁴/C_Z = 260mδ⁴/B_3; exponent
   1 + 4 − 2 = 3.
9. **The estimand** (R1/lead): the candidate's q is exactly the program's p_r (W10 §1 firewall;
   manuscript Definition 2.8's "window" phrasing resolves to the exact pin f(S) = b − r³/6; the
   corrected pin basis, the raw six pins, and T_r span the same σ-algebra; identical weight, types,
   RN structure; no adjacency conditioning).
10. **Scope discipline** (R6): all constants structurally r-independent; no sampled
    finite-difference modulus; no planar-moment substitution; no silent promotion (GP-LB-STAT-004
    unchanged; the accepted 2D lower campaign stays OPEN); conclusion strictly existential (no
    liminf coefficient, no numerical c or r₀, no dimension-3 or adjacency variant; η_r retained).

## Amendments and scoped residues (none blocking)

- A1 (RC): the residual field is nonstationary, so its C⁴-moment bound should be stated explicitly
  via the triangle inequality through the representers (substance present; presentation gap).
- A2 (RC): the M₄ index convention (|α| = 4 vs |α| ≤ 4) should be stated once (immaterial to §8).
- S1 (RA, non-blocking): equality with GP-DER-118-v1.10 §0 (candidate line 48) is
  CANNOT-VERIFY-locally — that body is absent from the local corpus (Drive ID
  1qc6ep2S4PIoPMEWDsdDQI1dr0LOwJiK9; hashes c1d6e559…/9b7901e1… as reported by the package).
  The estimand of record is local and matches independently of that body.

## Consequences for the program (stated exactly, no promotion)

1. The existential cubic lower bound stands endorsed at review grade by an independent route that
   consumes NONE of the open quantitative obligations (WP, Λ/H-B3, exit, γ-LOC, Bonferroni,
   far-field, SARD-G, η_r). This is the program's first endorsed lower theorem for the 2D problem
   at any scope: 1 − q(r, 6/5) ≥ c·r³ for all sufficiently small r, some c > 0.
2. The candidate does NOT establish: the 0.9144-class constant, a liminf coefficient, a numerical
   c or r₀, a full lifetime distribution, a two-sided law at explicit constants, or any dimension-3
   or adjacent-conditioned variant. The K3 assembly (K3-THM-001, conditional liminf at
   measured/mixed tier with hypotheses H1–H12) is a SEPARATE, logically independent statement at
   its own tiers; the candidate neither uses nor repairs its premises. The author-side state view's
   "K3-THM-001 REFUTED AS WRITTEN / NONCONTROLLING" attaches to the v1.1-lineage defects already
   quarantined by the K3 campaign; the K3 conditional assembly stands on its own receipts.
3. Follow-up available (not performed here): the constant expression c = 1040mδ⁴/C_Z is in
   principle evaluable (RB's endpoint density evidence m ≈ 1.28e-3; B_3 via the regression moments)
   — a certified evaluation would give a small but explicit numerical c and r₀.
4. The ratified upper chain (AO48-OPR-045) is untouched; combining it with this endorsed lower
   bound gives the two-sided Θ(r³) at the existential level, subject to the upper chain's own
   grades and to operator adjudication of any status change.
5. **Status firewall (exact):** this review endorses the candidate's mathematics at review grade.
   It changes no LS-CTL Boolean, eligibility predicate, theorem status, RP status, AO48 operator
   record, or q0 package status. Any status change requires separate operator adjudication. P0.1
   remains HOLD; the accepted 2D lower campaign's quantitative obligations remain OPEN at their
   own grades.

## Falsifier (what would kill this endorsement)

A counterexample at any interface: a valid elder-rule interpretation in which M dies at S despite
the path; a failure of the ten-jet covariance's positive-definiteness on the exact field; a
conditional C⁴ moment that blows up uniformly over the compact box; a sign/typing error in the
Taylor lift; a normalizer that is not O(r²); or a proof that the program's q differs from the
candidate's under the records of record. Any one retracts the corresponding interface's PASS and,
if load-bearing, this endorsement.

---

SHA-256 of this report body (text after this line is excluded):
d399e28378d177f88d08032c81f1573f35c9af2b600b9b775ebfe40f555cbbad
