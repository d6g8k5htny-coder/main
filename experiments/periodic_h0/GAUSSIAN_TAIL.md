# From a finite polynomial to an ideal Gaussian field

[Evaluated bounds](results/gaussian_tail1/RESULTS.md) ·
[Exact finite barcode](EXACT_H0.md) · [Remaining inputs](CONFIRMATION_READINESS.md)

This note supplies a uniform probabilistic bound for the omitted modes of a
specified ideal Gaussian field. It also states the additional coupling needed
to apply that bound to a certified finite polynomial. The stored PCG64/NumPy
coefficients have no such coupling certificate. Their total infinite-field
barcode error therefore remains unknown. No new random samples are used.

## 1. Exact ensemble and normalization

Set L=24, beta=pi²/L², alpha=2 beta, and

```
S_c,M = sum_{j=-M}^M exp(-c j²),    S_c = sum_{j in Z} exp(-c j²).
R = {(x,y) in Z² : x>0 or (x=0 and y>0)}.
R_K = {k in R : ||k||_infinity <= K}.
```

On one probability space let A_0 and every A_k,B_k, k in R, be independent
standard real Gaussian variables. Define

```
F(x) = A_0/S_alpha
       + sqrt(2)/S_alpha sum_{k in R} exp(-beta |k|²)
           [A_k cos(2 pi k·x/L) + B_k sin(2 pi k·x/L)].
```

For each derivative order the expected sum of absolute coefficient majorants
is finite, since polynomially weighted Gaussian-decay lattice sums converge.
Tonelli's theorem gives almost-sure absolute uniform convergence of each
derivative series. The intersection over integer derivative orders still has
probability one; F has a smooth version. Finite sums and their L² limits are
centered Gaussian and have covariance

```
E F(x)F(y) = sum_{k in Z²} exp(-alpha |k|²) exp(2 pi i k·(x-y)/L)
            / S_alpha².
```

This has variance one. The Fourier coefficient of
`sum_{n in Z²} exp(-|h+Ln|²/2)` is
`(2 pi/L²) exp(-2 pi²|k|²/L²)`: integrate over a period, unfold to R²,
and use the Gaussian Fourier integral. Dividing the periodized covariance by
its value at zero cancels the prefactor and gives the expression above.
This is the normalized periodized Bargmann–Fock covariance. The underlying
Fourier/Poisson identity is also given in
[DLMF §1.8(iv)](https://dlmf.nist.gov/1.8.iv); no external tail theorem is used.

Let F_K retain the DC term and modes R_K with this infinite normalization.
Let F64,K use the **same ideal Gaussian variables**, but replace S_alpha by
S_alpha,64. Thus

```
F_K = gamma F64,K,    gamma = S_alpha,64/S_alpha in (0,1).
```

The mathematical comparison ensemble follows the convention in
`experiment.py`: its denominator D64 is S_alpha,64², and its complex
coefficient is `exp(-beta |k|²)(A_k-iB_k)/(sqrt(2) S_alpha,64)`.
This identification of formulas does not certify the law or numerical
accuracy of the implementation. Changing the generator's maximum cutoff
also changes how it assigns random draws to modes; equal seed labels at
different maximum cutoffs do not establish a common coefficient bank.

## 2. A simultaneous tail event

There are exactly 4m representatives in the shell `||k||_infinity=m`.
For each pair define R_k=sqrt(A_k²+B_k²). Integrating the two-dimensional
Gaussian density in polar coordinates gives

```
Pr(R_k > v) = exp(-v²/2),    v >= 0.
```

Fix integer K>=1 and t>0. On the event E_K,t, require simultaneously

```
R_k <= t+j,    ||k||_infinity=K+1+j,    j=0,1,2,... .
```

The countable union bound, and `(t+j)²/2 >= t²/2+t j`, imply

```
Pr(E_K,t^c)
 <= sum_{j>=0} 4(K+1+j) exp(-(t+j)²/2)
 <= eta_K,t
 := 4 exp(-t²/2) [(K+1)/(1-exp(-t)) + exp(-t)/(1-exp(-t))²].
```

The permissible probability upper bound is min(1,eta_K,t). This bound does
not use independence between different tail modes; independence of tail
variables from retained variables will matter for conditional statements.

## 3. Uniform norm on that event

The inequality `|A cos(theta)+B sin(theta)|<=sqrt(A²+B²)` holds at every x.
The exact shell identity is

```
sum_{k in R, ||k||_infinity=m} exp(-beta |k|²)
 = exp(-beta m²) [S_beta,m + S_beta,m-1].
```

For example, subtracting full square sums gives
`S_beta,m²-S_beta,m-1² = 2 exp(-beta m²)(S_beta,m+S_beta,m-1)`;
the half-plane contains one from each equal ± pair. Therefore on E_K,t,

```
||F-F_K||_infinity <= tau_K,t
 := sqrt(2)/S_alpha sum_{m=K+1}^infinity
      (t+m-K-1) exp(-beta m²) [S_beta,m+S_beta,m-1].
```

This is a uniform bound on the whole continuous torus, not a grid-point
variance calculation. It holds with probability at least 1-min(1,eta_K,t).

## 4. Infinite remainders, including modes beyond64

For c>0 and M>=0, successive ratios of exp(-c j²), j>=M+1, are at most
`q_c=exp(-c(2M+3))`. Hence the **two-sided** theta remainder satisfies

```
0 < S_c-S_c,M <= 2 exp(-c(M+1)²)/(1-q_c) = T_c,M.
```

For M>=K the part of the amplitude sum above M is bounded by

```
U_K,t,M = 2 S_beta exp(-beta(M+1)²)
            [(t+M-K)/(1-q_beta) + q_beta/(1-q_beta)²].
```

Indeed m=M+1+j gives weight t+M-K+j, and the exponential ratio is at most
q_beta^j. Use S_beta,m+S_beta,m-1<=2S_beta and sum the geometric series and
its first moment. The implementation sums shells through M=64 and includes
this positive infinite remainder. It uses upper bounds for numerator terms
and a lower bound for S_alpha. Nothing sets an omitted sum to zero.

The normalization mismatch has the separate deterministic bound

```
1-gamma = (S_alpha-S_alpha,64)/S_alpha
        <= T_alpha,64/S_alpha,64 = delta_norm.
```

Computing the numerator by a remainder formula avoids subtracting nearly
equal interval enclosures. For this model the evaluated delta_norm is less
than 3×10^-64. It is a relative multiplier, not itself a sample error.

## 5. The additional coupling premise

Let P be a specified real finite polynomial with cutoff K<=64 and certified
norm upper bound B_P. Suppose a comparison on the same probability space
supplies the pathwise premise

```
||F64,K-P||_infinity <= rho.
```

On E_K,t, the triangle inequality and F_K=gamma F64,K give

```
||F-P||_infinity
 <= tau_K,t + gamma rho + (1-gamma)||P||_infinity
 <= tau_K,t + rho + delta_norm B_P.
```

A sufficient coefficient-level premise for rho is

```
|A_0/S_alpha,64 - c_0|
 + 2 sum_{k in R_K} |exp(-beta |k|²)(A_k-i B_k)/(sqrt(2) S_alpha,64)-z_k|
 <= rho,
```

where `P=c_0+2 Re sum z_k exp(2 pi i k·x/L)`. This is an explicit
obligation, not an assertion that rounded pseudorandom coefficients meet it.

If an event C establishing this premise is measurable with respect to the
retained Gaussian variables and Pr(C)>0, independence gives
`Pr(E_K,t | C)=Pr(E_K,t)>=1-min(1,eta_K,t)`. If C can depend on tail variables,
that conditional conclusion is unavailable. A separate bound Pr(C^c)<=p
would instead give an unconditional success bound at least 1-p-eta_K,t by
the union bound. We supply neither C nor p for the stored polynomial.
Conditioning a fixed polynomial on rho=0 is generally conditioning on a
probability-zero event and is not justified here.

## 6. Consequence for a genuinely coupled barcode

Assume an exact finite barcode Q already satisfies
`d_B(D(P),Q)<=epsilon_finite`. Uniform function error gives shifted sublevel
inclusions. The q-tame matching theorem and continuous finite-polyhedron
scope used in [APPROXIMATION.md](APPROXIMATION.md) apply to both smooth
functions on the torus. On the stated coupling and tail events,

```
d_B(D(F),Q) <= epsilon_finite + tau_K,t + rho + delta_norm B_P.
```

The endpoint-safe half-open bin sandwich from that note applies with this
total error; a clean upper count still requires a>2 epsilon_total. One
essential H0 class is separate and no longest finite bar is discarded.
This is a samplewise implication. A high-probability diagram bound alone
does not control an expected bar count on the exceptional event; an
appropriate count-moment/tail argument would be an additional input.

The C39 object is one K24 rounded polynomial and a certified 1024² dyadic
barcode. Its finite error is available and B_P<16, but rho is not. The new
certificate records both rho and the full-field diagram bound as null.
The K32 examples illustrate a possible different ideal truncation; they
are not a certificate for an uncomputed K32 sample or the historical pilot.

## 7. Arithmetic and replay boundary

`finite_certificate.py` supplies the existing rational Machin bounds for pi
and integer square-root bounds. In `gaussian_tail.py`, exp(-x) is enclosed
for rational 0<=x<=4096 by reducing x=2^s y with y<=1/8. The alternating
Taylor polynomial P49(y) is a lower bound and P48(y) an upper bound.
Terms decrease in magnitude. Rounding the lower endpoint down and the
upper up to denominator 2^256 preserves enclosure. Repeated squaring,
with outward rounding after each square, encloses exp(-x). Zero is exact.
All endpoints remain nonnegative. Decay bounds use the lower pi endpoint
for upper exponentials and the upper pi endpoint for lower exponentials.
Every subsequent operation uses rational endpoint monotonicity.

The certificate includes the exact rational shell sums, positive infinite
remainders, theta enclosures and probability bounds. Decimal display values
are rounded upward. Tests include direct independent alternating sums,
half-lattice shell identities, multiplicity checks, positive omitted-tail
checks, invalid-input/cached-type controls, and rejection of invented
coupling or scientific-promotion fields. The wrapper binds the unchanged
C39 certificate, original receipt and consumed sources before adding the
new result. Exact replay checks the displayed report as well as its hashes.

The result closes this explicit ideal-field uniform-tail calculation.
It does not certify a random-number generator, the historical FFT arrays,
a persistence intensity coefficient, an asymptotic remainder window,
weighted-Palm event identification, higher homology or dimensions, or
independent human review. No novelty or priority claim is made.
