# Expected-count loss for the finite Gaussian truncation

[Constructive sampler](GAUSSIAN_COUPLING.md) ·
[Exact word-to-polynomial bridge](WORD_POLYNOMIAL.md) ·
[Infinite-field tail and its separate obligations](GAUSSIAN_TAIL.md)

For a fixed Fourier cutoff, the number of finite positive ordinary H0 bars
has a deterministic bound that is independent of coefficient size. This
turns the sampler's clipping-failure probability into an explicit additive
error for expected bin counts of the **ideal finite Gaussian polynomial**.
The comparison uses the same cutoff on both sides. It uses no infinite-field
tail event, and supplies no expected-count error for the ideal infinite field.

## 1. A uniform finite-polynomial count bound

Let K>=1 be an integer. In angular coordinates on the two-dimensional torus,
write any real trigonometric polynomial with square cutoff K as

```text
P(theta) = sum_{|k1|,|k2|<=K} c_k exp(i k·theta),
c_{-k} = conjugate(c_k).
C_K = 16 K^2.
```

Coefficients can vanish or be arbitrary real/complex values subject to this
reality relation. There is no Gaussian, rationality, Morse or distinct-value
premise in the conclusion. Fix one homology coefficient field, and let N_P
count only finite diagram points of strictly positive lifetime in ordinary
superlevel H0, equivalently sublevel H0 of -P. Then

```text
N_P((0,infinity)) <= C_K.
```

The connected torus has one essential class, which is separate. No longest
finite bar is removed. The bound is deliberately conservative; no sharper
mixed-volume or critical-point formula is claimed.

**Morse polynomials and algebraic critical points.** Put z=exp(i theta1),
w=exp(i theta2), and let L(z,w)=sum c_k z^k1 w^k2 be the Laurent polynomial.
Clear the Laurent denominators in its angular derivatives by setting

```text
A(z,w) = z^K w^K sum_k k1 c_k z^k1 w^k2,
B(z,w) = z^K w^K sum_k k2 c_k z^k1 w^k2.
```

These are ordinary complex polynomials, each of degree at most 2K in each
variable and total degree at most 4K. A torus critical point is a common
zero of A and B. At such a point, differentiating the nonzero multiplier
z^K w^K adds only terms multiplied by the vanishing derivatives. The
angular-to-complex coordinate change has invertible diagonal derivative
diag(i z,i w). Thus a nonzero determinant of the real angular Hessian of P
implies a nonzero complex Jacobian determinant of (A,B). Every Morse
critical point therefore gives a distinct nonsingular isolated complex
intersection.

The equations can share components away from these points; simply assuming
they are relatively prime would be unjustified. If g=gcd(A,B), write
A=g A0 and B=g B0. At a point where g=0, the two derivative rows are
A0 times grad(g) and B0 times grad(g), so their determinant vanishes.
Consequently none of the nonsingular intersections lies on g=0. They are
intersections of the coprime residual polynomials A0 and B0. If either
residual is a nonzero constant there are none; otherwise homogenizing each
to its actual degree gives projective curves without a common component.
Removing repeated factors does not change their zero sets or increase
their degrees. Their affine intersections therefore number at most
deg(A0) deg(B0) <= 16 K^2, by
[Milne, *Algebraic Geometry*, version 6.10, Theorem 6.37, printed pages 152--153](https://www.jmilne.org/math/CourseNotes/AG.pdf).
Each distinct intersection contributes at least one to the intersection
sum. If either original derivative is identically zero, P has no
nonsingular torus critical point; this case cannot occur for a Morse P.

A Morse function on the compact torus has finitely many critical points.
Its ordinary H0 filtration has at most one birth per local maximum in
superlevel coordinates, hence at most one per critical point. This follows
from the local Morse cell-attachment theorem applied to -P: a zero-cell
can create one component; cells of positive dimension cannot create a
component. Between critical levels, the normalized gradient flow gives
the relevant deformation. The one-critical-point attachment theorem and
its proof are given in
[Weber, arXiv:1410.0995v3, page 3](https://arxiv.org/pdf/1410.0995v3),
reproving Milnor's Theorem 3.2. When several critical points have the same
value, their disjoint local neighborhoods give one attachment for each;
the birth upper bound remains the number of those critical points. One
birth survives as the essential class, but no subtraction is needed for
the stated conservative bound.

**Morse approximation at the same cutoff.** For an arbitrary P, consider

```text
P_t(theta) = P(theta) + t1 cos(theta1) + t2 sin(theta1)
                        + t3 cos(theta2) + t4 sin(theta2).
G(theta,t) = grad_theta P_t(theta).
```

These perturbations stay within cutoff K. The parameter derivative of G is

```text
[-sin(theta1), cos(theta1), 0,            0           ]
[0,            0,           -sin(theta2), cos(theta2)].
```

It is surjective at every point. Thus Z=G^{-1}(0) is a smooth
four-dimensional manifold. At a point of Z, tangent vectors (v,u) satisfy
H v + D_t G u=0, where H is the angular Hessian of P_t. The derivative of
the projection Z -> R^4 is surjective exactly when H is surjective:
invertible H solves for every u, while singular H cannot solve for all u
because D_t G is surjective. Therefore a regular value of this projection
gives a Morse P_t. By
[Sard's theorem, original paper, page 883](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/sard.pdf),
the critical values have measure zero; the manifold statement follows by
countably many coordinate charts. Regular values occur arbitrarily close
to zero. Also ||P_t-P||_infinity <= sum_j |t_j|. This proves uniform
approximation by Morse polynomials at the same K, including when P is
constant or has a curve of critical points.

**Passing the count bound to every P.** The torus is a finite polyhedron,
so its continuous-function persistence is q-tame. The specific q-tameness
and matching results are
[Chazal--de Silva--Glisse--Oudot, arXiv:1207.3674v3, Theorems 2.22 and 4.21](https://arxiv.org/pdf/1207.3674v3),
already used in [the approximation proof](APPROXIMATION.md). Uniform
function error gives shifted sublevel inclusions and the resulting matching.
If D(P) had C_K+1 finite positive bars, select that many and let ell>0
be their smallest lifetime. Choose a same-cutoff Morse P_t with
||P-P_t||_infinity < ell/2. None of the selected bars can match the
diagonal, since its diagonal distance is half its lifetime. They must
match distinct finite bars of P_t, contradicting its bound C_K. This
argument also rules out infinitely many positive bars. Degeneracy and
repeated critical values do not create an exception.

## 2. Counts are legitimate random variables

For fixed K, coefficient convergence implies uniform convergence of the
corresponding polynomials. Stability therefore makes the diagram map
continuous in bottleneck distance. For each r>0, the count of finite bars
with lifetime strictly greater than r is lower semicontinuous: any finite
set of counted bars remains counted under a sufficiently small matching.
For a>0, choose positive r_m increasing to a. Then

```text
N_P([a,infinity)) = lim_m N_P((r_m,infinity)).
N_P([a,b)) = N_P([a,infinity)) - N_P([b,infinity)).
```

The first equality uses the finite bound C_K; no infinite difference is
taken. Thus every half-open bin count with 0<a<b is Borel measurable and
bounded by C_K. This reasoning covers degenerate coefficient values as
well as Morse ones. It uses q-tameness and the preceding proof, and does
not silently assume a Morse distribution. One could also express the
torus using two pairs of sine/cosine coordinates satisfying unit-circle
polynomial equations; no additional semialgebraic triviality theorem is
needed for the measurability argument here. Counts from the finite word
space are measurable as well, including any explicitly defined outcome
for a failed computation.

## 3. The clipping loss for the matched finite Gaussian ensemble

Fix 1<=K<=64, and use exactly the probability space and input law from
[the sampler proof](GAUSSIAN_COUPLING.md): (2K+1)^2 independent U_i uniform
on (0,1), Z_i=Phi^{-1}(U_i), and J_i=floor(2^128 U_i). Let

```text
G = F64,K,
P = the dyadic polynomial constructed from these words at cutoff K,
C = {all |Z_i| <= 8},
rho = the same-cutoff coefficient_error_upper,
p = the same-cutoff clipping_failure_upper.
```

G uses the finite denominator S_alpha,64 and the retained ideal Gaussian
coordinates; it is not the infinite-normalized F_K or the infinite field F.
The already proved sampler contract gives

```text
Pr(C^c) <= p,             ||G-P||_infinity <= rho on C.
```

Both G and P have at most C_K finite positive bars for every outcome,
including outcomes outside C. No independence between C and any count is
assumed. For 0<a<b put delta=2 rho and interpret a reversed or empty
half-open interval as empty. Define

```text
L = N_P([a+delta,b-delta)),
U = N_P([a-delta,b+delta))  when a > delta.
```

The endpoint-safe matching argument from [EXACT_H0.md](EXACT_H0.md) gives
L <= N_G([a,b)) on C, and N_G([a,b)) <= U on C when a>delta.
Since 0<=L,N_G([a,b)),U<=C_K, splitting expectations over C and C^c gives

```text
max(0, E L - C_K p) <= E N_G([a,b)),
E N_G([a,b)) <= min(C_K, E U + C_K p)   when a > 2 rho.
```

For example,
E L = E[L 1_C]+E[L 1_(C^c)] <= E N_G([a,b))+C_K p.
The upper inequality follows by the same split, bounding the target on
C^c by C_K. These are expectation inequalities under the entire declared
input law, not after selecting words or conditioning on C.

If a<=2 rho, the universal upper bound is C_K. At equality, a target bar
of lifetime 2 rho may match the diagonal, so an expanded-bin count alone
is not an upper bound. The lower inequality still holds and its lower
observable is zero when its contracted interval is empty. Half-open
endpoints remain as written even when P's discrete coefficient law creates
atoms at exact lifetimes. Sharp-bin expectations are not asserted to differ
by C_K p: the expanded and contracted bins are necessary because successful
matches can move bars across a bin boundary.

The loss C_K p is evaluated without observing any random fields. It does
not evaluate E L or E U. If a computation later supplies exact or certified
expectations (or bounds on them) under the declared word law, they can be
inserted into this inequality. An arithmetic fixture or an empirical mean
is not such an expectation or confidence bound.

## 4. Using a grid with a word-dependent error

Suppose the word-to-polynomial bridge supplies for each word list J an
exact grid diagram Q_J and a certified nonnegative error e(J) satisfying
d_B(D(P_J),Q_J)<=e(J). The error can depend on the words. For a fixed bin
[a,b), put d(J)=2(e(J)+rho) and define the bounded grid observables

```text
L_grid(J) = N_QJ([a+d(J), b-d(J))),
U_grid(J) = min(C_K, N_QJ([a-d(J), b+d(J)))) if a > d(J),
            C_K                              otherwise.
```

An empty contracted interval contributes zero. On C, the triangle
inequality and the same matching argument give
L_grid <= N_G([a,b)) <= U_grid. Crucially L_grid<=C_K also holds
outside C: the deterministic P_J-to-Q_J matching alone gives

```text
L_grid(J) <= N_PJ([a+2 rho,b-2 rho)) <= C_K.
```

The first inequality uses the existing grid error e(J); if that intermediate
interval is empty, L_grid is zero. Thus the grid's possible n^2-1 bars do
not force an n^2 clipping-loss factor. Both grid observables are bounded by
C_K, and the expectation split proves

```text
max(0, E L_grid - C_K p) <= E N_G([a,b))
                         <= min(C_K, E U_grid + C_K p).
```

There is no need for a uniform upper envelope on e(J). However, the strict
upper-resolution test a>d(J) must be applied to each outcome. If a grid
computation fails or has no certificate, defining L_grid=0 and U_grid=C_K
for that outcome preserves the inequality. Such outcomes must remain in
the ensemble; discarding them would change the law. A supplied error is a
mathematical premise, not something established merely by passing rational
values to a counting routine. Rejecting inconsistent supplied counts or
errors is different from silently inventing a successful certificate.

## 5. What this closes and what it leaves open

The bound closes the exceptional-event count loss for the constructive
clipping coupling to the **same-cutoff ideal finite Gaussian polynomial**.
The independent uniform-word input model remains an explicit premise. No
physical entropy source, deterministic seed, or historical PCG64/NumPy
coefficients acquire that model from this argument. In particular the
historical coupling error remains unknown, and the deterministic cutoff-one
fixture is not an observation of any random ensemble.

The degree cap does not apply to the ideal infinite Fourier field. Adding
the spectral-tail error and its failure probability to rho and p would not
justify replacing the exceptional infinite-field count by C_K. A bound for
the infinite-field expected count on that exceptional event still requires
its own argument, such as an appropriate integrable count moment and an
explicit probability inequality. Smoothness or almost-sure finiteness
alone does not provide that quantitative bound. This note proves no
lifetime asymptotic, applicable numerical remainder, held-out confirmation,
or independent-human review condition.

The mathematical proof supplies the uniform count bound and the expectation
transfer. Exact arithmetic evaluates the same-cutoff budgets. Tests can
check type contracts, integer factors, strict endpoint conditions, cap
handling and semantic replay; they do not prove Sard's theorem, Bézout's
theorem, the matching theorem, or the law of a word source. Authorship,
nonauthor mathematical review, formal evidence and scientific acceptance
remain separate.
