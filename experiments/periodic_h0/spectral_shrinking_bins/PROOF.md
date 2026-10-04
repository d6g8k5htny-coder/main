# Shrinking lifetime bins for coupled spectral samples

Object: C132-SPECTRAL-SHRINKING-BINS-v1. Author-side conditional analytic
theorem; source-bound nonauthor review pending. Authors: OpenAI/Codex root
with the analytical outline and source triage of `next_math_triage`.
Organizational-independence credit: 0. No performed human review is claimed.

This note extends the exact-sample shrinking-bin implication to supplied
grid values coupled to a specified Gaussian Fourier field. It closes the
composition of spectral error, spatial interpolation and exceptional expected
counts under explicit input interfaces. It does not prove those imported
interfaces anew, change their scientific status, or certify the historical
PCG64/NumPy experiment. All frozen input files remain unchanged.

## 1. Observable, imported hypotheses and coupling

Fix the square torus of side `L=24`, dimension two, ordinary superlevel H0
persistence. Count every finite positive-lifetime bar with multiplicity;
exclude the single essential H0 class and zero-length intervals. A finite bar
has lifetime equal to maximum birth height minus its elder merge height.
No longest finite bar is discarded. The expected density is per unit area,
so the full-torus expected measure is `L^2 nu(ell) d ell`.

The following are consumed hypotheses, not conclusions of this note:

**D (continuum density).** For the field F below, the expected measure of
finite ordinary continuum H0 bars has a density on `(0,ell_*]`, and

```text
|nu(ell) - c ell^(-1/3)| <= C_nu,
c > 0, C_nu < infinity, ell_* > 0.
```

Here `c=c_(2,24)` is the coefficient in the pinned M4--M5 interface. The
coefficient's interpretation uses that interface's parent chain. No new
blanket pairing assumption or independent verification of that chain is
claimed. M4 does not evaluate `C_nu` or `ell_*`.

**C (continuum count).** F is almost surely Morse with distinct critical
values, and its total real critical count satisfies

```text
K_C := (E N_crit(F)^2)^(1/2) < infinity.
```

The unconditioned total-critical-count clause of Fourier C6 Theorem F is
the cited source for this moment. Its exact source disposition is retained.
The additional window count under the distinct pinned determinant-tilted
Palm C6 law is not this input. Since every finite ordinary H0 bar is born
at a maximum, every continuum bin count is bounded by `N_crit(F)`.

**A (deterministic approximation).** Use the vertex-cubical superlevel H0
filtration in the pinned approximation note: vertices have their supplied
values, edges and squares enter at the minimum of their vertex values.
Its adaptive PL representative has the same H0 barcode. For any finite real
values `s_v` on the periodic n by n grid, `h=L/n`, that note gives an actual
matching with

```text
d_B(D(F), Q(s)) <= h^2 H(F)/4 + max_v |s_v-F(v)|,
H(F) = sup_x ||Hess F(x)||_op.
```

The operator norm is Euclidean. The triangulation is the barcode-preserving
adaptive one, not an arbitrary fixed diagonal convention. This bound allows
noisy samples and is pointwise, so sample-dependent diagonals are permitted.
The exact mathematical cubical barcode has at most `n^2-1` finite H0 bars.
No assertion about a floating-point library's output is substituted for A.

Use the exact full-field representation of `GAUSSIAN_TAIL.md`. Put

```text
omega = 2 pi/L, beta = pi^2/L^2, alpha = 2 beta,
S_a = sum_(j in Z) exp(-a j^2),
S_(a,m) = sum_(|j|<=m) exp(-a j^2),
R = {k=(k1,k2) in Z^2 : k1>0 or (k1=0 and k2>0)}.
```

On one probability space take independent standard real Gaussian variables
`A_0,A_k,B_k`, `k in R`, and define

```text
F(x) = A_0/S_alpha
       + sqrt(2)/S_alpha sum_(k in R) exp(-beta |k|^2)
           [A_k cos(omega k.x) + B_k sin(omega k.x)].
```

This is the variance-one normalized periodized Bargmann--Fock field at
L=24. Let `F_m` retain the DC coefficient and `||k||_infinity<=m`, with the
same full denominator. For integer `M>=1` also define

```text
G_m^M = (S_alpha/S_(alpha,M)) F_m,
G_m^infinity = F_m.
```

All comparisons use these same Gaussian coordinates. Equal PRNG seed labels
at different cutoffs do not establish this coupling.

For every integer `n>=3`, let `m_n` be a deterministic positive integer and
`M_n` a deterministic positive integer or infinity. Let the measurable
supplied grid values `s_(n,v)` be finite real numbers.
Assume a Borel event `A_n` and deterministic numbers `eta_n>=0`, `0<=q_n<=1`
such that

```text
P(A_n^c) <= q_n,
max_v |s_(n,v) - G_(m_n)^(M_n)(v)| <= eta_n on A_n.       (1)
```

The supplied values can depend on the entire field and other randomness.
There is no independence assumption about A_n. A random computed error
certificate fits (1) by including both its validity and the condition that
its reported error is at most `eta_n` in A_n. A numerical barcode error,
if relevant, requires its own separate matching guarantee and count control;
it is not silently part of a nodal certificate.

Write `N_F(I)` and `N_(Q_n)(I)` for continuum and exact supplied-grid barcode
counts. Fix `0<lambda<mu<infinity`, and take deterministic `tau_n>0` tending
to zero. The bins are half-open:

```text
I_n = [lambda tau_n, mu tau_n).
```

## 2. Explicit analytic error constants

Define the finite, strictly positive lattice constants

```text
D_beta = sum_(j in Z) j^2 exp(-beta j^2),
K_0 = [1 + (S_beta^2-1)/sqrt(2)] / S_alpha,
K_H = sqrt(2) omega^2 S_beta D_beta / S_alpha,
T_m = (S_beta^2 - S_(beta,m)^2)/(sqrt(2) S_alpha).
```

The difference defining T_m denotes an exact positive tail sum; it is not a
floating-point subtraction. Let

```text
zeta_M = (S_alpha-S_(alpha,M))/S_(alpha,M),
zeta_infinity = 0.
```

For `M>=1`, the theta remainder in C40 yields

```text
0 < zeta_M <= 2 exp(-alpha(M+1)^2)
                  / [S_(alpha,M)(1-exp(-alpha(2M+3)))].      (2)
```

For each n set

```text
p_n = 2 ceil(4 log n),
B_n = h_n^2 K_H/4 + T_(m_n) + zeta_(M_n) K_0,
d_n = 2 [e sqrt(p_n) B_n + eta_n],
b_n = min{1, n^(-8)+q_n}.                                 (3)
```

Log denotes the natural logarithm and e its base. `p_n` is an even moment
order, not a probability. `B_n>0` because `h_n>0` and `K_H>0`.

**Theorem (finite error bound and asymptotic consequence).** Under D, C, A
and (1), for every n satisfying

```text
d_n <= lambda tau_n/2,
mu tau_n+d_n <= ell_*,                                   (4)
```

one has

```text
E |N_(Q_n)(I_n)-N_F(I_n)|
 <= 4 L^2 d_n [c (lambda tau_n-d_n)^(-1/3) + C_nu]
    + (n^2-1) b_n + K_C sqrt(b_n).                        (5)
```

In particular, if `d_n=o(tau_n)` and

```text
n^2 b_n + sqrt(b_n) = o(tau_n^(2/3)),                     (6)
```

then

```text
E |N_(Q_n)(I_n)-N_F(I_n)| / E N_F(I_n) --> 0,
E N_(Q_n)(I_n) / E N_F(I_n) --> 1.                        (7)
```

The denominator is strictly positive eventually. An explicit sufficient
family for (7) is

```text
M_n >= m_n (with infinity allowed),
[h_n^2+exp(-beta m_n^2)] sqrt(log n) + eta_n = o(tau_n),
n^2 q_n + sqrt(q_n) = o(tau_n^(2/3)).                     (8)
```

The optional restriction `2m_n<n` is eventually satisfied by the example
below and prevents frequency collisions for the usual spectral grid layout.
It is not needed for inequality (5) about supplied samples, and on its own
proves no Gaussian law or numerical accuracy for an FFT implementation.

## 3. Gaussian moments and full-field error

Put `R_k=(A_k^2+B_k^2)^(1/2)`. For an even integer `p=2q>=2`, polar Gaussian
integration and the one-dimensional Gaussian moment formula give

```text
E R_k^(2q) = 2^q q! <= (2q)^q,
E |A_0|^(2q) = (2q-1)!! <= (2q)^q.
```

Consequently both Lp norms are at most `sqrt(p)`. These inequalities include
all constants needed when p grows with n; a fixed-order moment assertion
with an unspecified dependence on p would not suffice.

The absolute coefficient majorants have finite expected sums after any
fixed derivative order, because a polynomial times Gaussian lattice decay
is summable. Tonelli and intersection over the countably many derivative
orders give almost sure absolute uniform derivative convergence. One may
first use Minkowski on finite sums, then pass to nonnegative majorants by
monotone convergence. The following estimates therefore hold for each
even p, including the selected order p_n.

Pointwise `|A_k cos(theta)+B_k sin(theta)|<=R_k`. Moreover the Hessian term
has operator norm at most `omega^2 |k|^2 R_k`. Half-lattice symmetry gives

```text
sum_(k in R) exp(-beta |k|^2) = (S_beta^2-1)/2,
sum_(k in R) |k|^2 exp(-beta |k|^2) = S_beta D_beta.
```

Hence, uniformly in m,

```text
|| ||F_m||_infinity ||_p <= K_0 sqrt(p),
|| H(F) ||_p <= K_H sqrt(p),
|| ||F-F_m||_infinity ||_p <= T_m sqrt(p).                 (9)
```

The last identity for the coefficient sum follows by removing the finite
square from `S_beta^2`; each nonzero mode has its partner -k. In particular
there is no missing factor of two or spherical counting replacement.

The full-field error for the finite denominator satisfies pathwise

```text
||F-G_m^M||_infinity
 <= ||F-F_m||_infinity + zeta_M ||F_m||_infinity.
```

Thus its Lp norm is at most `(T_m+zeta_M K_0)sqrt(p)`. Define

```text
X_n = h_n^2 H(F)/4 + ||F-G_(m_n)^(M_n)||_infinity.
```

By (9) and Minkowski, `||X_n||_(p_n)<=B_n sqrt(p_n)`. Markov gives the single
Gaussian approximation event

```text
P{X_n > e sqrt(p_n) B_n} <= exp(-p_n) <= n^(-8).           (10)
```

This handles the full field's Hessian and spectral error jointly. It does
not require a separate Hessian estimate for the growing truncations or
independence between retained and omitted modes after conditioning.

Let `G_n=A_n intersect {X_n<=e sqrt(p_n)B_n}`. By the union bound,
`P(G_n^c)<=b_n`. On G_n, (1), the triangle inequality at the grid vertices,
and A show that the continuum diagram and Q_n admit a matching of cost at
most `d_n/2`. In particular matched finite lifetimes differ by at most d_n.

## 4. Sharp bins and expected exceptional counts

For diagrams P,Q with an actual epsilon-matching, set `d=2epsilon`. If
`I=[a,b)` with `a>d`, the elementary matching injection gives

```text
|N_Q(I)-N_P(I)| <= N_P([a-d,a+d) union [b-d,b+d)).          (11)
```

Indeed, a member of either diagram in I has lifetime greater than d and
cannot be matched to the diagonal. If exactly one member of a matched pair
lies in I, its P-lifetime is in one of the two displayed strips. The strips'
lower endpoints are included and upper endpoints excluded: those choices
also cover equality at a bin boundary. Injectivity counts each discordant
pair at most once. This is the reviewed exact-sample sharp-bin lemma,
restated to make the composition explicit.

Apply (11) on G_n with the deterministic d_n. Under (4) the strips lie in
`(0,ell_*]`, their union has length at most `4d_n`, and every lifetime in
them is at least `lambda tau_n-d_n`. Hypothesis D implies

```text
E[ |N_(Q_n)(I_n)-N_F(I_n)| 1_(G_n) ]
 <= 4 L^2 d_n [c (lambda tau_n-d_n)^(-1/3)+C_nu].          (12)
```

We bounded the restricted right side of (11) by the unrestricted expected
strip count. No independence of G_n and the diagram is used. Expected
endpoint atoms vanish in this range because the expected measure has a
density; the deterministic argument itself does not discard equality cases.

On the complement, the vertex-count bound and hypothesis C give

```text
E[N_(Q_n)(I_n) 1_(G_n^c)] <= (n^2-1) b_n,
E[N_F(I_n) 1_(G_n^c)]
 <= E[N_crit(F) 1_(G_n^c)] <= K_C sqrt(b_n).              (13)
```

The second step is Cauchy--Schwarz for the unconditioned full-field count.
It applies to an arbitrary Borel bad event, including correlated sample,
normalization, Hessian and spectral failures. It neither multiplies
probabilities of independent witnesses nor substitutes a finite-polynomial
critical-count cap for the infinite field. Summing (12) and (13) proves (5).

## 5. Limit comparison and usable families of assumptions

By direct integration of D,

```text
E N_F(I_n)
 = L^2 (3c/2) (mu^(2/3)-lambda^(2/3)) tau_n^(2/3)
   + O(tau_n).                                         (14)
```

The implicit bound in the last term is at most
`L^2 C_nu (mu-lambda) tau_n`. The leading constant is positive, so the
denominator is eventually positive and comparable to `tau_n^(2/3)`.
If `d_n=o(tau_n)`, (4) holds eventually. Dividing the two good-event terms
of (5) by this scale gives `O(d_n/tau_n)` and
`O(d_n/tau_n^(2/3))`, both tending to zero. Condition (6) handles the two
bad-event terms. The second limit in (7) follows from
`|E N_(Q_n)-E N_F|<=E|N_(Q_n)-N_F|`. No exchange of a derivative and a limit
is used.

For completeness, C40's positive theta remainder gives, for m>=1,

```text
T_m <= [2 sqrt(2) S_beta / (S_alpha (1-exp(-5beta)))]
         exp(-beta(m+1)^2).                             (15)
```

To see this, factor `S_beta^2-S_(beta,m)^2`, bound its second factor by
`2S_beta`, and use
`S_beta-S_(beta,m)<=2 exp(-beta(m+1)^2)/(1-exp(-beta(2m+3)))`.
Equation (2), `S_(alpha,M)>=1`, and `M>=m` similarly imply
`zeta_M=O(exp(-alpha(m+1)^2))`. All constants are fixed functions of L,
not of m,M,n or the bins. Also `p_n=O(log n)`. The first limit condition in
(8) therefore implies `d_n=o(tau_n)`.

Eventually `tau_n>=h_n^2 sqrt(log n)`. Since `h_n=L/n`,

```text
n^(-4) / tau_n^(2/3)
 <= L^(-4/3) n^(-8/3) (log n)^(-1/3) --> 0,
n^(-6) / tau_n^(2/3) --> 0.
```

Finally

```text
n^2 b_n <= n^(-6)+n^2 q_n,
sqrt(b_n) <= n^(-4)+sqrt(q_n).
```

The last condition of (8) proves (6). In particular `q_n=O(n^(-8))` is
sufficient under that first condition.

A nonempty ideal asymptotic family is obtained by choosing fixed
`0<gamma<2`, `epsilon>0`, and

```text
tau_n = n^(-gamma),
m_n = ceil(sqrt((gamma+epsilon) log n / beta)),
M_n >= m_n,
eta_n = o(n^(-gamma)), q_n = O(n^(-8)).                   (16)
```

Then `2m_n<n` eventually. The interpolation ratio is
`L^2 n^(gamma-2)sqrt(log n)`, and the spectral ratio is at most a constant
times `n^(-epsilon)sqrt(log n)`; both vanish. This proves feasibility of
the mathematical assumptions. It does not construct a certified sampler
with the stated eta_n,q_n, or evaluate a finite grid at which the conditions
first hold.

All limits here keep L, dimension, lambda, mu and the continuum law fixed.
Only the grid, cutoffs, supplied-sample accuracy and lifetime scale vary.
There is no infinite-volume limit or varying covariance hidden in (7).

## 6. Negative controls and excluded inferences

These are analytic checks of why the hypotheses cannot be dropped, not
numerical experiments purporting to prove the theorem.

1. **Bottleneck distance alone does not control a sharp-bin count.** Put any
   number J of equal-length bars just below a bin's lower boundary and move
   their endpoints slightly so their lifetimes lie just above it. Matching
   cost tends to zero while the count discrepancy remains J. Equation (11)
   records exactly that boundary mass; D is what controls its expectation.
   At the diagonal threshold, a single bar of lifetime d has distance d/2
   from the empty diagram but count one in `[d,2d)`. Thus `a>d` is strict.

2. **Small failure probability alone does not control expected count.** An
   abstract nonnegative count equal to `n^8` on an event of probability
   `n^(-8)` and zero elsewhere has expectation one. This is not a Gaussian
   counterexample; it refutes an inference from probability alone. The
   finite grid cap and full-field K_C in (13) are essential count inputs.

3. **A fixed normalization error survives the asymptotic limit.** For any
   fixed a>0, multiplying the entire continuum field by a multiplies every
   lifetime by a. By D and (14), the count ratio for `aF` versus F in I_n
   tends to `a^(-2/3)`, not to one unless a=1. Hence a small but fixed
   normalization mismatch is not removable merely because it is small.
   Fixed `M=64` has `zeta_64>0`; (8) does not apply with that denominator.

4. **Fixed finite precision is not the family (16).** The frozen C41/C42
   clipping, bit precision and supported-grid restrictions do not establish
   eta_n tending to zero with the required rate or q_n satisfying (8).
   This theorem requires those premises for the actual supplied values.
   Independence is not a replacement for a bound on their errors.

5. **A Nyquist check is not a numerical certificate.** `2m_n<n` distinguishes
   the retained frequencies on the grid, but says nothing about PRNG law,
   coefficient generation, FFT roundoff or computed persistence endpoints.
   Uncertain intervals crossing bin boundaries cannot be classified by
   rounded midpoints. Q_n in this theorem is the exact mathematical barcode.

6. **A weighted-Palm count cannot be substituted for C.** The measure in D
   and the expectation in (13) are unconditioned full-field quantities.
   A pinned, determinant-tilted window-count moment is a different random
   variable under a different law. Only the explicit unconditioned total
   count interface listed in SOURCES.json is consumed here.

## 7. Verification boundary

The new result is (5)--(8), including the joint exceptional-count estimate
and the finite-denominator term. Its proof is analytic; file hashes and
repository checks bind its statement but do not prove it. The fixed source
identities, author exposure and exact technical review belong to this object.
The review of the earlier exact-sample lemma is not a review of this extension.

No evaluated value of K_C, C_nu or ell_*, usable finite-grid threshold,
confidence interval, historical pilot confirmation, parent theorem closure,
all-angle weighted-Palm identification, higher-dimensional result, Lean
formalization, human review or scientific-status promotion is asserted.
