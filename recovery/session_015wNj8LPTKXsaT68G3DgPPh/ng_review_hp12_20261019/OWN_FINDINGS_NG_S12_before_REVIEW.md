# C8 own findings, NG smoothing-admission PROOF.md (322195ff...), §12, recorded BEFORE reading REVIEW.md §§11-12
(REVIEW.md §§1-10, 13, 14 had been read for the p = 0 slice; §§11-12 not read.)

Checked, no defect:
- Degree set: for oriented X, p = 0..d-1 (no finite top-degree superlevel bar: H_d of {F>=s} appears only at the global minimum and is essential);
  H0-only case uses N_t^all = N_(0,t).
- (30) E[N_(Tt)^all 1{N_(Tt)^all >= 2}] = o(t^(2/3)):
  * the mark 1{N>=2} is a bounded Borel whole-field mark, so the once-counted identity (11)-(12) applies and
    sum over bars of 1{ell<=Tt} 1{N>=2} = N 1{N>=2};
  * near band: after ell = t a, the integrand / t^(2/3) is dominated by C a^(-1/3) k^(-2/3) H_M(b,k) (bounded mark times the §8
    all-mark envelope), integrable over 0<a<=T, b in R, k>0, and the near-band indicator k >= t a / r0^3 increases to 1;
  * pointwise limit 0 at each fixed (a,b,k,x,u) and w: r = (t a/k)^(1/3) -> 0, the pinned bar has lifetime t a <= T t, the two new
    critical points are its endpoints (unit incidence; each critical point is an endpoint of at most one bar), every other bar joins two
    old branches whose limiting heights are finitely many and pairwise distinct, so their lifetimes stay above half the minimum gap > T t;
    on V_(p,0) = 0 the determinant weight tends to 0;
  * far part: E N^far_(Tt) = integral_0^(Tt) nu_far = O(t) by (23).
- (31): the a-marginal of eta_p is 4 a^(-1/3) integral k^(-2/3) Z_(p,0) = V C_p a^(-1/3), using C_p = (4/V) integral k^(-2/3) Z_(p,0)
  from (25)-(26); total mass on 0<a<=T is (3/2) V C_p T^(2/3), matching E N_p((0,t])/V ~ (3/2) C_p t^(2/3); finite on complete windows
  by (22); no atom at a = 0.
- Laplace step: for f >= 0 bounded, h = 1 - e^(-f): 0 <= <h,Xi> - (1 - e^(-<f,Xi>)) <= N 1{N>=2} (equality for N <= 1;
  1 - prod e^(-f_i) <= sum (1 - e^(-f_i))), so E[1 - e^(-<f,Xi_t>)] = t^(2/3) sum_p integral h d eta_p + o(t^(2/3)) by (30)-(31);
  omitted far bars have expected number O(t) on a <= T.
- (32): (1 - E[1 - e^(-<f,Xi>)])^n with n t_n^(2/3) -> lambda gives exp(-lambda sum_p integral h d eta_p); Laplace functionals of
  C_c^+ tests on the locally compact separable mark space characterize convergence in law to the Poisson random measure with intensity
  lambda sum_p delta_p (x) eta_p; degree restrictions of the limit are independent Poisson; within one field they need not be.
- The marks (p, a, b, k, x, u) are jointly Borel via the unique short geodesic for near bars; far bars may be omitted or cemetery-marked.

Numerical check (new, d = 1, p = 0, same model as the slice-H0 Monte Carlo): per-realisation N_t to test (30), i.e.
E[N_t 1{N_t>=2}] / E N_t -> 0 as t -> 0 at eps = 0, 1, 3 (J = 8192): PENDING at time of writing.

Not checked here: the Hp parts (separate part (a) of the pickup); REVIEW.md §§11-12.
Own verdict for §12 (pre-comparison): PASS_SCOPED, no must-fix.
