# Constructing a finite-word Gaussian coupling

[Evaluated bounds](results/gaussian_coupling1/RESULTS.md) ·
[Ideal-field tail](GAUSSIAN_TAIL.md) · [Remaining inputs](CONFIRMATION_READINESS.md)

This note gives a constructive coupling for a **new, specified sampler**.
It supplies the low-mode premise left conditional in the ideal-field tail
calculation, under an explicit independent uniform-input model. It does not
identify the stored PCG64 polynomial with that sampler. The historical
coefficient error and full-field barcode error remain unknown.

## 1. Probability space and precise result

Use the side24 planar field, normalization, representatives R_K and independent
Gaussian tail modes defined in [the tail proof](GAUSSIAN_TAIL.md). For each of
the N=(2K+1)² real retained coordinates, take an independent U_i uniform on
(0,1), independent of the omitted modes, and set

```
Z_i = Phi^-1(U_i),   J_i = floor(2^128 U_i),
u_j = (j+1/2)/2^128, q_j = clip(Phi^-1(u_j),[-8,8]).
```

Phi is the standard real normal CDF and phi its density. Thus Z_i are independent
standard Gaussians and J_i are independent uniform128-bit words. Conversely,
given words with exactly this joint law, adding independent uniform residuals
within their bins constructs this coupling. This is a mathematical input law;
neither a deterministic seed nor this program proves it for an actual bit source.

The implemented dyadic value z_j satisfies `|z_j-q_j|<=2^-64` and `|z_j|<=8`
for every possible word. Section3 proves this uniformly, including termination.
On the event C={all |Z_i|<=8}, projection in probability onto
[Phi(-8),Phi(8)] contracts distance from U_i. The inverse CDF on that interval
is 1/phi(8)-Lipschitz by the mean-value theorem. Since |U_i-u_Ji|<=2^-129,

```
|Z_i-z_Ji| <= dZ := 2^-129/phi(8) + 2^-64.
```

The Gaussian integral bound
`integral_8^infinity phi(x) dx <= integral_8^infinity (x/8)phi(x) dx = phi(8)/8`
and the union bound give

```
Pr(C^c) <= p_clip := min(1, 2 N phi(8)/8).
```

The count N includes the DC coordinate and both real coordinates of every
retained pair. This is an unconditional probability over the declared input
model. It cannot be carried over after conditioning on an arbitrary fixed word
list: an extreme bin can force clipping failure. Only the tail event remains
independent of a retained-word observation. No fixed fixture receives a
Gaussian probability certificate.

## 2. The actual polynomial and coefficient error

Let S=S_alpha,64 and beta=pi²/24² as in the tail proof. In real cosine/sine
coordinates the ideal weights are

```
w0=1/S,   wk=sqrt(2) exp(-beta |k|²)/S.
```

Rational enclosures `[l_i,h_i]` follow from the unchanged exact exponential,
pi, square-root and theta routines. Set the implemented weight a_i to the
nearest dyadic128 number to `(l_i+h_i)/2` (ties upward). Its error is bounded
by `e_i=max(|a_i-l_i|,|a_i-h_i|)`. Define

```
W = h0 + 2 sum_{k in R_K} hk,
Ew = e0 + 2 sum_{k in R_K} ek,
Bp = 8 (|a0| + 2 sum_{k in R_K} |ak|),
P(x)=a0 z_J0 + sum_{k in R_K} ak [z_Jk,A cos(theta_k) + z_Jk,B sin(theta_k)].
```

All coefficients of P are exact dyadic rationals. The word order is DC,
then A,B for each pair in increasing x=0..K and y=-K..K, retaining x>0 or y>0.
The returned `real_basis_modes` are cosine/sine coefficients, not the complex
coefficient schema of the historical extractor. Conversion to that schema
would be `complex=(cosine-i*sine)/2` and requires its own explicit adapter.

On C, split `w_i Z_i-a_i z_Ji = w_i(Z_i-z_Ji)+(w_i-a_i)z_Ji`.
The triangle inequality gives the uniform spatial bound

```
||F64,K-P||_infinity <= rho := dZ W + 8 Ew,   ||P||_infinity <= Bp.
```

The multiplicity2 is essential because each pair has two real coordinates.
These bounds hold for every word realization on C, not just tested coordinates.
The coefficient budget is below1.47e-18 at K24 and K32. Quantile rounding is
already negligible relative to the spectral-tail budget, so further precision
here is not the current approximation bottleneck.

Using the unchanged tail event E_K,8 and normalization loss delta from the
prior calculation, the union bound and its pathwise composition give

```
Pr( ||F-P||_infinity <= tau_K,8 + rho + delta Bp )
    >= 1 - p_clip - eta_K,8.
```

This statement is about the prospective ensemble. If its *newly constructed*
polynomial later has a certified barcode Q with finite error epsilon, the
same uniform-stability argument gives
`d_B(D(F),Q)<=epsilon+tau+rho+delta Bp` on those events. Such a barcode is not
computed here. Expected bin counts on the exceptional event still require
separate count-moment bounds; a high-probability supremum bound is insufficient.

## 3. Certified inverse-normal evaluation

For 0<=x<=8 define `a_n=x^(2n+1)/(2n+1)!!`. The positive entire series S(x)
satisfies S'(x)-xS(x)=1 and S(0)=0, by termwise differentiation of its
absolutely convergent power series. Therefore

```
I(x) := integral_0^x exp(-t²/2) dt = exp(-x²/2) sum_{n>=0} a_n.
Phi(x)=1/2+I(x)/sqrt(2pi),    Phi(-x)=1-Phi(x).
```

The program sums n=0..191. After the last term every successive ratio is at
most r=x²/385<=64/385<1. Thus the omitted sum is at most `a_191*r/(1-r)`.
Each term recurrence is rounded down/up to denominator2^256; replacing the
last term by its upper endpoint preserves this positive remainder bound.
The exponential and reciprocal square-root use their respective lower/upper
endpoints. Reflection is exact. Intersecting with [0,1] remains a valid CDF
enclosure. There is no floating-point transcendental evaluation in this path.

Here is the uniform precision argument, not a sample-based termination claim.
Put u=2^-256. If D_n bounds the width of a directed term enclosure for any
x in [0,8], then

```
D0=2u,   Dn=64 D_(n-1)/(2n+1)+2u,  n=1..191.
D=sum Dn.
```

The directed positive-series upper sum plus its remainder, evaluated at8,
majorizes every smaller argument and is below2^50. Write R for that upper
remainder. `precision_contract()` evaluates these exact rational bounds.

For the exponential at x²/2<=32, the existing routine uses at most8 squarings.
Its initial consecutive Taylor polynomials differ by
`y^49/49! <= (1/8)^49/49! < 2^-279 < u`. Consequently their outward dyadic
enclosure has width at most2u (an integer number of grid steps). All endpoints
lie in [0,1]. Each squaring increases width by at most a factor2 plus2u.
After8 squarings width is at most1022u<2^-246; denote the latter by de.

Let dl,dh enclose 1/sqrt(2pi) and m be the positive lower bound for phi(8).
The unnormalized integral interval width is at most
`G=de*2^50+D+R`. Since I(x)<=8, its upper endpoint is at most8+G. Hence every
CDF enclosure has width at most

```
w = (dh-dl)(8+G) + dl G.
```

Exact evaluation verifies `w<2^-123` and `2w/m<2^-65`. The certificate records
the rational witnesses. This includes rounding in every positive-series term.

For a midpoint probability u_j, bisection starts with [-8,8], which brackets
the clipped inverse. If u_j lies strictly to one side of a midpoint CDF
interval, the corresponding half brackets it. If u_j lies in that interval,
the mean-value bound brackets the clipped inverse within
`mid +/- max(u_j-CDF_lower,CDF_upper-u_j)/m`; intersect with the old bracket.
This argument still holds when u_j is outside [Phi(-8),Phi(8)], because
clipping can only shorten the distance to the appropriate boundary.

The ambiguous branch has width at most2w/m<2^-65. Ordinary bisection takes
69 halvings to reach width2^-65 from width16. The loop allows80 iterations,
including the terminal check. Rounding the bracket midpoint to dyadic64
adds at most2^-65; the bracket contributes at most2^-66, so total error is
at most3*2^-66<2^-64. Its rounded value remains in [-8,8]. Definite outer-tail
probabilities return the exact clipping endpoint. If a required bound ever
fails, the implementation raises an error; there is no numerical fallback.

## 4. Executed evidence and boundaries

The certificate evaluates the uniform K24/K32 budgets and replays the exact
unchanged tail source. It constructs one cutoff1 polynomial from nine fixed,
evenly spaced word values including both extreme words. That fixture tests
clipping and coefficient ordering. It is deliberately deterministic and is
not represented as an independent Gaussian sample, new barcode, or held-out
experiment. A future experiment needs its own word-source premise, input
receipt, converted polynomial, nodal and barcode verification.

Tests compare the positive CDF expansion to a separate alternating integral,
check quantile residuals and clipping, include every real mode in probability
and error sums, reject invalid cached input types, and exercise semantic
negative controls. Receipt verification binds exact consumed bytes, recomputes
all claims and checks the displayed report. Rebinding a false result's hashes
cannot make it pass replay.

This closes the stated constructive sampler coupling under its input model.
It does not certify PCG64/NumPy or a physical entropy source, change the old
polynomial, supply a Gaussian barcode observation, identify an elder-pairing
event, prove a lifetime asymptotic or its numerical remainder, establish a
weighted-Palm loss estimate, or supply independent human review. No novelty
or priority claim is made.
