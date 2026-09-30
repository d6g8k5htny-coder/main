# Exact bounds for the retained finite Fourier polynomials

[Experiment](README.md) · [Approximation argument](APPROXIMATION.md) ·
[Eight-field certificate](results/certificate8/RESULTS.md) ·
[Confirmation requirements](CONFIRMATION_READINESS.md)

This calculation turns the finite-polynomial **derivative majorants** into
evaluated exact rational certificates. It does not yet certify the numerical
samples used in the persistence study. The missing nodal error remains visible
as `null`, alongside a `null` diagram-error field. Neither means zero.

## The exact mathematical input

`COEFFICIENTS.json` defines eight real trigonometric polynomials on the torus
of side 24. Each has 1,200 nonzero-frequency *positions*, some of which may in
principle have zero coefficients, and a real DC coefficient. For

\[
 R_{24}=\{(k_x,k_y):0\le k_x\le24,\ -24\le k_y\le24,
                  \ k_x>0\text{ or }k_y>0\},
\]

the polynomial is

\[
 F_z(x)=c_0+2\sum_{k\in R_{24}}
       \Re\!\left(z_k e^{2\pi i k\cdot x/24}\right).
\]

Every real and imaginary part is a canonical rational string with a power-of-two
denominator. These rational values, interpreted exactly, define the object being
bounded. Conjugate coefficients at `-k` are implicit. The input validator checks
the complete ordered half-plane, rejects duplicates, rejects noninteger and
Boolean indices, requires a real DC term, and rejects noncanonical, nondyadic,
nonfinite and malformed rational data. Duplicate JSON keys are rejected at every
depth, including in the certificate and execution receipt.

The seed numbers 34000–34007 are reconstruction labels. They do not certify a
Gaussian distribution, randomness, or independence. Rounded Gaussian draws,
rounded exponential weights, and a finite mode set are different mathematical
inputs from exact Gaussian draws, exact exponential weights, and the intended
infinite field.

## How the coefficients were obtained

The extraction command invokes the existing `experiment.field_grid` with its
existing mode bank, cutoff24, default factors, and a 64×64 coefficient array.
During that call alone, the `ifft2` boundary is intercepted: the exact array
passed by the actual generator is copied, and a zero complex array is returned
instead of evaluating an FFT. The extractor verifies the forward normalization,
shape, finiteness, conjugate symmetry, real DC term, and absence of occupied
positions outside the declared modes. Since 64 is greater than twice the cutoff,
these retained frequencies do not collide in the array.

This is **coefficient extraction, not a field evaluation**. No FFT output from
the interception is used as a sample or persistence input. The actual generator,
extractor, exact arithmetic module, dependency file and existing refinement
configuration are bound by filename, byte count and SHA-256 in `RUN.json`.

The prior refinement did not archive its coefficient arrays. This delivery
reconstructs them from the same bound source, configuration and seed labels; it
does not invent a receipt for the past execution. The stored rational coefficient
file now provides a fixed exact object which can be consumed without NumPy,
GUDHI, the random generator or floating-point transcendental functions.

## Exact arithmetic proof

The core checker uses Python integers and `fractions.Fraction`. All subsequent
bounds are obtained without floating-point arithmetic.

### Square roots

For a nonnegative rational `q=p/d`, let `S=2^128` and

\[
 m=\left\lfloor\sqrt{\left\lfloor pS^2/d\right\rfloor}\right\rfloor.
\]

Integer square root gives `m²d <= pS² < (m+1)²d`. Thus `m/S` is a lower
bound for `sqrt(q)`, and `(m+1)/S` is an upper bound. If `m²d=pS²`, both
endpoints are `m/S`. Each amplitude

\[
 \rho_k=2\sqrt{(\Re z_k)^2+(\Im z_k)^2}
\]

therefore has a certified rational upper bound `rho_k_plus` with error at most
`2^-127`. This error allowance is included term by term; it is not omitted
because it is numerically small.

### Pi, including the branch of the angle identity

Use Machin's identity

\[
 \pi=16\arctan(1/5)-4\arctan(1/239).
\]

For completeness, put `alpha=atan(1/5)` and `beta=atan(1/239)`.
The double-angle formula gives `tan(2 alpha)=5/12` and
`tan(4 alpha)=120/119`; the subtraction formula then gives
`tan(4 alpha-beta)=1`. Also

\[
 0<4(1/5-1/(3\cdot5^3))-1/239
    <4\alpha-\beta<4/5<\pi/2.
\]

The lower estimate follows from the alternating arctangent series, and the
upper estimate from `atan(t)<t`. For the last inequality it is enough that
`pi/2=2 atan(1)>1`, by integrating `1/(1+t²)>1/2` on `(0,1)`.
Tangent is injective on `(0,pi/2)`, so `4 alpha-beta=pi/4` with the stated
branch, proving the identity.

The checker sums 64 terms for `atan(1/5)` and 24 terms for `atan(1/239)`.
Alternating-series remainders bound each by its next omitted term, with the
correct sign. The subtraction is enclosed as
`[16 alpha_lower-4 beta_upper,16 alpha_upper-4 beta_lower]`.
Finally its lower endpoint is rounded down and its upper endpoint up to the
dyadic lattice of spacing `2^-128`. The resulting rational interval encloses
pi; the displayed decimal constants are not the source of this enclosure.

### Derivative and interpolation majorants

Let `p_plus` be the upper pi endpoint and put `w=(2 p_plus/24)^2`.
Termwise differentiation and the triangle inequality give

\[
\begin{aligned}
 M_{xx}&=w\sum_k\rho_k^+ k_x^2,&
 M_{yy}&=w\sum_k\rho_k^+ k_y^2,\\
 M_{xy}&=w\sum_k\rho_k^+ |k_xk_y|,&
 H&=w\sum_k\rho_k^+(k_x^2+k_y^2).
\end{aligned}
\]

These are upper bounds for the three component sup norms and the Hessian
operator sup norm of the exact polynomial `F_z`. The Hessian of a single
mode is a scalar of magnitude at most its amplitude times the frequency outer
product; the operator norm of that outer product is the squared frequency norm.
No grid search or unproved maximum-location assumption is used.

For `h=24/n`, the [interpolation argument](APPROXIMATION.md) supplies

\[
 B_n=\min\left\{\frac{h^2H}{4},\;
       \frac{h^2(M_{xx}+M_{yy})}{8}+\frac{h^2M_{xy}}4\right\}.
\]

All four grid budgets (128, 256, 512 and 1024) are computed exactly. For these
Fourier triangle majorants the component expression is no larger, because
`2|k_x k_y| <= k_x²+k_y²`. It improves the prior operator-only diagnostic.
The report rounds each nonnegative bound **upward** to 15 decimal places.
Exact rational values remain in `CERTIFICATE.json`.

## What is established, and what still needs an input

For the frozen eight polynomials, the 1024² spatial budgets range from
`0.007186147859908` to `0.007929637977099`, using upward-rounded endpoints.
These are certified *upper bounds*, not measurements of the actual interpolation
error and not bounds for the historical FFT alone.

If supplied samples `s_v` later satisfy a certified
`max_v |s_v-F_z(v)| <= eta`, the deterministic approximation theorem gives
`epsilon <= eta+B_n` for the declared cubical and PL filtrations. This delivery
does not supply `eta`. It also does not certify persistence software or the
rounding of reported barcode endpoints. Consequently it supplies no certified
bin counts, no continuum-field error and no coefficient-confirmation window.

A subsequent useful step is a controlled interval evaluation of this stored
finite polynomial, or a separately justified FFT error analysis, compared with
the actual samples. Such a calculation must use these exact coefficient bytes
and include its own numerical error. Uniform control of the infinite spectral
tail, the correction to the intended field law, and a numerical asymptotic
remainder and cutoff are additional, separate obligations.

## Reproduction and negative controls

From the repository root, exact replay needs only Python's standard library:

```sh
python -B -S experiments/periodic_h0/finite_certificate.py --verify experiments/periodic_h0/results/certificate8
```

Regeneration, in the pinned numerical environment, writes a new output directory:

```sh
python -B experiments/periodic_h0/run_certificate.py --output /tmp/new-finite-certificate
```

The directory verifier recomputes the entire rational certificate, checks its
binding to the coefficient object, checks exact output and source bytes,
reconciles the current configuration, and regenerates the readable report.
These identity checks do not authenticate an external party or retroactively
prove that an unobserved historical calculation occurred.

Tests cover hundreds of nonsquare and exact-square enclosures; a single-mode
field with explicit derivative constants; actual generator input capture;
malformed and duplicated modes; noncanonical rationals; duplicate JSON keys;
altered source/output hashes; rebound but inconsistent certificates; report
drift; and the attempted promotion of `eta` from unknown to zero. Both ordinary
and optimized Python execution retain the validation checks.
