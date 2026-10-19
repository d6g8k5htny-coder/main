# C8 findings, NG smoothing-admission PROOF.md (322195ff...), part (a): the p >= 1 (Hp) steps
(Exposure: REVIEW.md §§1-10, 13, 14 had been read for the p = 0 slice, including its Hp paragraphs in §§4 and 6.
 These findings are therefore NOT pre-comparison; the §12 findings are, see OWN_FINDINGS_NG_S12_before_REVIEW.md.)

Statement (§1): oriented X, d >= 2, any fixed field K, p = 0..d-1; (5) and (6) with 1{A_u has exactly p positive eigenvalues};
leading coefficient and limiting marks independent of K; no C_p = C_(d-1-p).

Checked, no defect:
- §4: (10) holds with any nonnegative Borel whole-field mark; (11) uses HF §7's selector sigma_p (=1 iff F is Morse/distinct and the
  ordered pair is exactly one finite Hp interval), so each finite interval is counted once and essential intervals are excluded;
  for p > 0 the birth point M is a degree-p critical point of g = -F, not a maximum; (12) is the Tonelli density.
- §7, index bookkeeping: j = d-p-1; birth index_F(M) = j+1, death index_F(S) = j (HF (3)); g-degrees p and p+1 (HF (4)).
  (17) det H_i / r = alpha_i det A_i - r beta_i^T adj(A_i) beta_i (sympy); alpha_M -> -6k, alpha_S -> +6k from the pins and F_sss = 12k
  (axial profile f'(s) = 6k(s^2 - r^2/4) up to the C4 remainder, f(M) - f(S) = k r^3); the diag(r^-1/2, I) congruence plus Sylvester
  gives the indices, so V_(p,r) -> 36 k^2 (det A_0)^2 1{index A_0 = j}, i.e. the cone with p positive transverse eigenvalues.
- HF hypotheses vs NG's contact data, in Fermi coordinates (s,t) of the fixed geodesic through x along u (birth to death):
  pins M_r = (-r/2,0), S_r = (r/2,0) with zero gradients and heights b, b - k r^3 come from the target v_r of (13);
  F_0(x) = b, grad F_0(x) = 0, F_0,ss = F_0,st = 0 (V_u = 0; coordinate = covariant Hessian at a critical point), F_0,sss = 12k
  (geodesic third derivative); A_0 nonsingular with j negative eigenvalues (§6, a.s., any inertia); every other critical point Morse
  with distinct values different from b and x the only degenerate one (§6); global C4 convergence on the same realization (§5).
  HF's conclusion is the Hp interval (M_r, S_r) with p = d-j-1, so sigma_p = 1 eventually on V_(p,0) > 0.
- HF §§4-7 as imported:
  §4 ridge t = h_r(s) by the implicit function theorem; phi_r''' -> 12k > 0, so phi_r' is strictly convex with exactly the two pinned
     roots; exact LDL splitting F_r = phi_r(s) - |w_-|^2 + |w_+|^2 (no concavity for mixed inertia); old Morse branches continue and a
     gradient floor excludes other roots; no critical value between F_r(S_r) and F_r(M_r).
  §5 ascent field: unstable manifold of S_r = axial + w_+ (dim p+1), stable manifold of M_r = axial + w_- (dim d-p); transverse along
     the unique axial orbit; annulus height cost Delta_* > k r^3 excludes any other connection.
  §6 classical relative incidence (one transverse attaching/belt point gives +/-1) and the C4-to-smooth bridge by aligned critical
     values and the collar flow (13), which preserves the finite persistence module.
  §7 Borel selector by local constancy on a countable cover of the separable C4 good locus; finite intervals only in p = 0..d-1.
- §§8-9 for general p: the same all-mark envelope and far bound (sigma_p and the typed indicator are at most 1).
- §10: (27) 144 = 4 * 36; t = 12k gives 144 * 12^(-7/3) = 12^(-1/3) in (28); u -> -u leaves A_u and its inertia cone unchanged and
  gives (6); the leading term carries no K because the incidence is a unit in every field; F -> -F would change A to -A and is not
  used (shifted-Gaussian cone example 1.925 vs 0.075).

Not checked: HF §§8-11 (its stationary torus law), the Main331 D1 documentary hold, Milnor's handle theorems themselves (taken as
the stated classical import), GAD's own proof. No numerical test for p >= 1.
Verdict for part (a): PASS_SCOPED, no must-fix.
