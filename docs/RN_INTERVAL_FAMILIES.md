# RN affine Gaussian interval families — candidate v1

Object: RN-INTERVAL-FAMILIES-20260920-v1. Author: OpenAI/Codex task
01a0bbb5-2fcb-77f0-b78b-4d220ddd7ab2. Author-side exact implementation and
ordinary mathematical argument; unreviewed scientific candidate. Organizational
independence credit is zero. No claim, gate, grade or prize status changes.

This extends the fixed-input implementation in `research/rn/gaussian_moments.py`
using the existing exact rational `research.interval.Interval` arithmetic.
It does not replace RN5's Arb spatial engine. The [RN5 source](https://drive.google.com/file/d/1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5/view)
is 13,725 bytes, SHA-256
`ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`.
The earlier fixed-law candidate remains byte-for-byte intact.

## 1. Precise family and quantifiers

Let `A_i`, `B_i` and `S_ij` be closed rational intervals, for three coordinates
ordered `(xx, yy, xy)`. Symmetric intervals `S_ij=S_ji` denote a *single shared*
uncertain scalar for each unordered pair. For each fixed choice
`a in A`, `b in B`, and symmetric positive-semidefinite `Sigma in S`, consider
`X(t) ~ N(a+b*t, Sigma)` over the full closed rational interval `t in T`.
The covariance is constant in `t`; singular Gaussian laws are allowed.
Write `D=X_1 X_2-X_3^2`. The result bounds `E[D^k]` for `k=2,4`, uniformly over
all these choices and all marks. No joint independence of the coordinates,
Hessian blocks, or uncertain coefficients is assumed.

The quantifier is over PSD members, not arbitrary interval matrices. Two
explicit result states are distinguished below. An enclosing parameter box
may include nonphysical members and dependencies absent from this box; ignoring
such dependencies can widen an enclosure, but cannot remove an actual member.
The user must separately prove that an actual field's parameters lie in it.

## 2. Coefficient inclusion by induction

For a fixed member let `M_alpha(t)=E[product_i X_i(t)^alpha_i]` and `M_0=1`.
For any `i` with `alpha_i>0`, put `beta=alpha-e_i`. Gaussian integration by
parts gives

`M_alpha = (a_i+b_i*t) M_beta + sum_j beta_j Sigma_ij M_(beta-e_j)`.

Terms with `beta_j=0` are omitted. This also holds for singular PSD laws by
approximation with `Sigma+epsilon I` and continuity of finite Gaussian moments.
Induction on total degree proves `deg M_alpha <= |alpha|`.

Replace scalar parameters by their enclosing intervals, retain the polynomial
in the mark, and perform each coefficient addition and convolution with exact
interval operations. Inductively each true coefficient lies in its computed
interval: sums and products preserve inclusion, regardless of repeated
parameter dependencies. Memoization reuses the same containing polynomial.
Then expand

`E[D^k] = sum_(j=0)^k (-1)^j binom(k,j) M_(k-j,k-j,2j)`.

Signed interval multiplication is essential. Degree is at most `2k<=8`.
Repeated uncertain parameters can cause overestimation; grouping equal mark
powers does not reproduce RN5's separate spatial-monomial cancellation work.
With singleton parameter intervals, every operation is exact, trailing exact
zero coefficients are removed, and this recurrence equals the preceding
fixed-law implementation coefficient by coefficient.

## 3. A complete mark-interval enclosure

For coefficient intervals `C_j`, write `t=l+w*u`, `u in [0,1]`. The converted
power coefficient enclosure is

`A_j = sum_(r=j)^n C_r binom(r,j) l^(r-j) w^j`.

The Bernstein coefficient enclosure is

`B_k = sum_(j=0)^k A_j binom(k,j)/binom(n,j)`.

All factors are exact rationals, including possibly negative values of the
left endpoint raised to nonnegative integer powers. For any fixed polynomial
member, its Bernstein coefficients belong to these intervals. The Bernstein
basis `binom(n,k)u^k(1-u)^(n-k)` is nonnegative and sums to one. Therefore
`[min_k lower(B_k), max_k upper(B_k)]` encloses the polynomial on the *entire*
mark interval. A zero-width interval is included without division by width.

At each dyadic subdivision both closed children are evaluated and their hull
is returned. Their union is the parent. Repeating this argument proves full
coverage for every admitted depth, zero through twelve. No endpoint sampling
or one-child shortcut suffices. A second valid upper bound is
`upper(C_0)+sum_(j>=1) max(abs(lower(C_j)),abs(upper(C_j)))*H^j`,
where `H=max(abs(l),abs(l+w))`. The smaller of the two upper bounds remains
valid. The implementation refuses a negative even-moment upper bound.
Singleton polynomial intervals route through the existing exact Bernstein
routine, recovering the same cap at every depth.

## 4. Covariance feasibility is separate

For each row define
`delta_i = lower(S_ii)-sum_(j!=i) max(abs(lower(S_ij)),abs(upper(S_ij)))`.
If every `delta_i>=0`, then for every symmetric member and real vector `x`,

`x' Sigma x >= sum_i lower(S_ii)*x_i^2 - sum_(i<j) 2*m_ij*abs(x_i*x_j)`
`>= sum_i delta_i*x_i^2 >= 0`.

Here `2|x_i*x_j|<=x_i^2+x_j^2`. This sufficient diagonal-dominance test proves
**ALL_BOX_MEMBERS** PSD, including semidefinite boundaries. It is not necessary.
An exact singleton instead uses the existing exact PSD test and may pass even
when diagonal dominance fails.

If the sufficient box test fails, the result is **PSD_MEMBERS_ONLY**. A midpoint
that passes the exact PSD test supplies one feasible witness and establishes
nonemptiness only. If it fails, nonemptiness remains unestablished; the code
neither declares the family empty nor promotes all its members to Gaussian
laws. A wholly negative diagonal or an invalid singleton is refused.

For example the three-by-three box with diagonal one and all off-diagonal
intervals `[-1,1]` has the identity as a PSD midpoint. Its member with all
three off-diagonal entries `-1` has quadratic form `-3` on `(1,1,1)`, so the
whole box is not PSD. Conversely, the singleton outer product of `(1,2,3)`
is PSD despite negative diagonal-dominance margins.

## 5. Binding and verification

`engine/operations/rn_family_applicability.py` binds source ID/hash, dimension,
coordinate order, normalization, conditioned-law identity, mark domain and
candidate tier. The mean and covariance must name the same conditioned law.
All interval endpoints enter the family fingerprint. These consistency checks
cannot authenticate a caller's claimed field identity or normalization.

Deterministic tests compare enclosures with a separate expansion in independent
standard normals after correlated affine transformation. They exercise both
moments, uncertain means and slopes, covariance entries, singular inputs,
negative marks, exact singleton recovery, and all subdivision children.
There is no held-out or general utility claim.

Two transparent controls are especially useful. For centered independent
`X,Y~N(0,1)` and `Z~N(0,s)`, `s in [0,1]`, direct expansion gives
`E[D^2]=1+3s^2` and `E[D^4]=9+18s^2+105s^4`. The family caps are exactly `4`
and `132`; midpoint-only values are `7/4` and `321/16`. For centered unit-variance
`X,Y` with correlation `rho in [1/2,3/4]` and `Z=0`, the second moment is
`1+2rho^2`; dropping covariance yields the invalid cap `1`.

Remaining RN obligations include verified field-input enclosures, consistent
conditioning and separate marginal whitening, required positive interval
Cholesky pivots, complete spatial annulus coverage, normalization provenance,
weighted Palm composition and scoped technical review. RN5's 65 point laws
and 10 spatial boxes do not become a complete cover through this generic code.

## References and reconnaissance

The elementary inclusion proof above is supplied for this implementation.
Established background is [Rump, Verification methods, Acta Numerica 2010](https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf)
for interval inclusion and dependency, and [Mamis, multivariate normal moments](https://arxiv.org/html/2202.00189v6)
for Gaussian moment identities. Bernstein basis conversion and partition of
unity are proved explicitly here and in the prior fixed-law candidate.
The bounded three-query reconnaissance was frozen and raw-readback verified
before implementation as Drive `1D4vUx7dJzvFThN78k3fcagyDv46YQHeB`, SHA-256
`0af8f01306ec8efe76ab458271094c0a24913c7909bee00c8fae4691b906a6a0`.
No priority, novelty or completeness of literature search is asserted.
