# C8 own findings, NG smoothing-admission PROOF.md (322195ff...), slice H0 (p=0), recorded BEFORE reading REVIEW.md
(Only the README and the first ~25 lines of REVIEW.md, i.e. its verdict and reviewer identity, had been read.)

Checked, no defect (p=0 path):
- §1 statement for p=0, d>=1; coefficient (6) re-derived independently from the pair Rice scaling:
  r^(d-1) [polar] * r^-(d+3) [observation density] * r^3 [height] * r^2 [determinants] -> r^1 dr at fixed k; ell=|T| r^3/12;
  gives 12^(-1/3)|T|^(4/3)(det A)^2 and the 1/2 from the k>0 (T>0) restriction under u -> -u.
- §2 imports: for p=0 the HF topology is not used; SELECT uses GAD §8's deterministic elder enclosure, whose hypotheses
  (C4 convergence, exact pins, single cubic contact, remote Morse branches with distinct heights) are supplied for each fixed w by §§5-6.
- §3 (8) convolution density; (9) Bayes-weighted literal kernel = correct conditional law of F=G+W given LF=v; positivity.
- §4 Morse/distinct heights a.s. by fixed-w Sard then Fubini; two-point identity (10); (11)-(12) for p=0 via GAD's selector.
- §5 U_r rows; |dU/dO| = 12 r^-(d+3) (sympy d=1,2,3); targets v_r; (15)-(16).
- §6 conditional off-contact genericity at each fixed w; contact isolation via ridge derivative 6k s^2.
- §7 (17) block-determinant identity (sympy); alpha_M -> -6k, alpha_S -> 6k; inertia index A_0 = d-1 (negative definite) for H0; (18)-(20).
- §8 envelope with polynomial W moments: split N<=eps B / N>eps B; (22); integrable iff M>4/3 (exponent 1/3-M).
- §9 FAR (23); midpoint-map Jacobian det 1 and J even (sympy for d=3 blocks); (25) = own derivation (r^-2 dr = (1/3) ell^-1/3 k^-2/3 dk); (26) factor 4.
- §10 (27) factor 144; (28) 144*12^(-7/3) = 12^(-1/3); u -> -u averaging to (6); N(m,1) cone-asymmetry example (sympy: 1.925 vs 0.075).
- §11 cumulants: E[(X^4-3)^3]=9504, E[(X^2-1)^3]=8 (sympy); no exponential moment; constant-W consistency; W=-G exclusion.

End-to-end numerical test (new; d=1, p=0, circle V=2pi, F=G+eps(H^2-1), independent stationary Gaussian series G (scale 5) and H (scale 4), N=25 modes):
- formula (6) evaluated through the literal mixture kernel (condition on W first): kappa(eps=1)=1.1172 +- 0.0008,
  predicted E N((0,t]) ~ 4.599 t^(2/3); covariance-matched Gaussian would give 4.161 t^(2/3) (10.5% lower).
- Gaussian control eps=0: predicted 5.320 t^(2/3).
- simulation results: PENDING at time of writing (appended below before REVIEW.md is read).

Not checked: p>=1 steps through HF; §12 (bounded-cluster, Poisson); HF and GAD proofs themselves beyond the interfaces used.
Own verdict (pre-comparison, before simulation): PASS_SCOPED for p=0, no must-fix.
