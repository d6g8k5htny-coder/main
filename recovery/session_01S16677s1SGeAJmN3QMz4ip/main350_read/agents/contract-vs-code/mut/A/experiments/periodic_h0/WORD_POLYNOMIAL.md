# From explicit words to a certified finite-polynomial barcode

[Sampler and coupling premise](GAUSSIAN_COUPLING.md) ·
[Exact nodal evaluator](NODAL_CERTIFICATE.md) ·
[Hessian enclosure](HESSIAN_GRID.md) · [Exact H₀](EXACT_H0.md)

This bridge takes the exact real coefficients constructed by
`gaussian_coupling.polynomial` from an explicit ordered word list. It converts
them to complex Fourier coefficients without rounding, evaluates a new exact
dyadic grid, and bounds its barcode error against that same smooth finite
polynomial. The implementation is [word_polynomial.py](word_polynomial.py).

The fixture below is deterministic. It is not a Gaussian
sample, a held-out experiment, or a replacement for the historical PCG64
polynomial. The sampler's independent uniform-word law remains a premise of
its ensemble theorem. A selected fixed word list does not inherit that law or
its unconditional probability statement.

## Exact object and Fourier conventions

Set the torus side to 24 and let

```text
R_K = {(kx,ky): 0 <= kx <= K, -K <= ky <= K,
                 kx > 0 or (kx = 0 and ky > 0)}.
theta_k(x) = 2 pi k·x/24.
P(x) = d + sum_{k in R_K} [C_k cos(theta_k(x)) + S_k sin(theta_k(x))].
```

Here `d`, `C_k` and `S_k` are the exact dyadic `Fraction` values returned by
the sampler, not floating approximations to those values. The word order is
DC first, then the cosine and sine coordinates for each frequency in increasing
`kx`, then increasing `ky`. There are `2K(K+1)` pairs and `(2K+1)^2` words.
The bridge requires the complete ordered mode set, strict integer frequencies
and cutoff, and real coefficients with denominator at most `2^192`.

Define

```text
z_0 = d,                 z_k = (C_k - i S_k)/2.
P(x) = z_0 + 2 Re sum_{k in R_K} z_k exp(+i theta_k(x)).
```

This identity follows by multiplying `C_k-iS_k` by `cos(theta)+i sin(theta)`
and taking its real part. Both the minus sign and the division by two are
necessary. The DC term is real and is neither halved nor duplicated. The
negative-frequency coefficient is the conjugate of the positive-frequency
coefficient. The sampler's real weight
`sqrt(2) exp(-beta |k|^2)/S_alpha,64` therefore becomes
`exp(-beta |k|^2)/(sqrt(2) S_alpha,64)` in the complex convention, with no
further normalization change.

At row-major vertex `a*n+b`, the physical point is `(24a/n,24b/n)` and the
phase is `2 pi (kx*a+ky*b)/n`. The existing inverse DFT uses the positive
exponent and no normalization in either axis. In particular, no factor `n^2`
is inserted or removed. Both grids must be supported powers of two, at most
1024 and strictly greater than `2K`; the sample grid is at least four.

## Exact lifting into the existing evaluator

The real coefficients can require denominator `2^192`; after conversion,
a complex component can require `2^193`. The unchanged nodal evaluator uses
`S=2^96` and requires every input coefficient to be exactly representable at
that scale. Converting these inputs to float64 or silently rounding them to
that scale would define a different polynomial.

Instead, let `r` be the largest denominator exponent among the DC and all
converted complex components, and set

```text
lambda = 2^max(0,r-96),       P_lift = lambda P.
```

Every component of `P_lift` multiplied by `S` is an integer. This is the
smallest nonnegative power-of-two lift with that property; `lambda <= 2^97`
for the accepted inputs. The bridge stores canonical rational strings for
the lifted complex record and performs no coefficient rounding. Its legacy
`seed=0` field is solely a compatibility record label, with no RNG meaning.

Suppose `nodal_core.evaluate` returns complex integer centers `a_v`, its
stage records, and an error `E_lift` bounding each component of every exact
node of `P_lift`. Define the supplied real samples and their error by

```text
sample_scale = lambda S,
s_v = Re(a_v)/sample_scale,
eta = E_lift/lambda.
```

Dividing the existing component inequality by positive `lambda` proves
`max_v |P(v)-s_v| <= eta`. Since `P` is real, the same inequality bounds
`|Im(a_v)|/sample_scale`; the implementation also checks this consequence.
There is no added coefficient-conversion error. A scale of `2^193` describes
the exact sample denominator, not 193-bit transcendental accuracy: the bound
still includes the evaluator's existing 96-bit twiddle errors.

The sample digest is the SHA-256 of the canonical JSON array of decimal real
integer strings in row-major order, including the canonical trailing newline.
The full array need not be stored when replay regenerates it, its stage bounds
and its digest. A digest identifies an array; the regenerated arithmetic and
the nodal proof supply its error bound.

## Hessian, stability and exact bin transfer

Apply `hessian_grid_core.bound_record` to `P_lift` on a derivative grid of
size `m`. This consumes all derivative-grid nodes and the existing fourth
derivative interpolation covers. Let `c_lift` denote its returned
`spatial_coefficient`. For sample grid size `n`, set

```text
B = (24/n)^2 c_lift/lambda,       epsilon = eta+B.
```

Every actual derivative of `P_lift` is `lambda` times the corresponding
derivative of `P`, so division by positive `lambda` transports the certified
upper bound to `P`. This argument does not require the rounded upper bounds
returned by the program to be exactly homogeneous. The complete lifted
Hessian result remains available for replay in its original units.

The [existing approximation argument](APPROXIMATION.md) constructs a
continuous PL representative of the periodic vertex cubical filtration with
uniform error at most `eta+B` from the smooth finite polynomial. Its cited
persistence stability theorem supplies an epsilon-matching to `D(P)`.
No Morse assumption, distinct-critical-value assumption or Gaussian premise
is added by this step.

The bridge computes `exact_h0.compute(Re(a_v),n)` with the four periodic
axis neighbors and runs `verify_by_connectivity` on that same integer array.
The latter checks plateau births, essential elder, complete finite-bar labels
and both sides of every proposed death level. This is algorithmic independent
verification, not a claim of organizationally independent review. Endpoint
integers remain unchanged; a pair `(b,d)` has exact lifetime
`(b-d)/sample_scale`. One essential class is kept separate, zero pairs are
omitted, and no longest finite interval is removed. The computed endpoint
arithmetic error is zero relative to these supplied samples.

For the exact sample diagram `Q`, put `delta=2 epsilon`. The existing
[bin-transfer proof](EXACT_H0.md) gives, for `0<a<b`,

```text
N_Q([a+delta,b-delta)) <= N_P([a,b)),
N_P([a,b)) <= N_Q([a-delta,b+delta))  when a > delta.
```

The lower count is zero when the contracted interval is empty. The upper
bound is absent when `a <= delta`, including equality. All endpoints, scales,
errors and comparisons remain exact integers or rational numbers. Essential
classes and diagonal points are not counted. These are realization counts
for the specified finite polynomial, not ensemble expectations.

## Deterministic fixture and certified result

The new fixture has cutoff one and nine ordered words

```text
word_i = floor(j_i (2^128-1)/17),
(j_0,...,j_8) = (1,13,4,16,7,11,2,15,9).
sample grid n=128; derivative grid m=32.
```

It is distinct from the evenly spaced clipping fixture in the earlier
coupling certificate. It is deliberately selected arithmetic input. The
frozen clipping fixture remains useful for checking the conversion, but its
existence does not certify this new grid or barcode.

The [produced certificate](results/word_polynomial1/CERTIFICATE.json) regenerates
this fixture from the words, with `lambda=2^97` and sample scale `2^193`.
Its [readable result](results/word_polynomial1/RESULTS.md) reports three
positive finite grid bars and the exact rational diagram bound. Decimal error
bounds in that report are rounded upward, rather than probe approximations.

Both the lower and upper counts for `[1/125,2/125)=[0.008,0.016)` equal one.
The smooth finite polynomial therefore has **exactly one finite bar in that
bin**, and zero in each of the other seven displayed bins spanning
`[1/1000,32/125)`. This does not exclude bars outside those bins. The two
smaller grid bars are below `2 epsilon` and are not certified smooth-field
detections. No conclusion about all sufficiently short smooth-field bars
follows from this bin result.

The [execution receipt](results/word_polynomial1/RUN.json) binds the sources.
Run the certificate verifier to regenerate and compare every value, rather
than treating the prose or file hashes alone as a numerical proof.

## Frozen dependencies and verification boundary

The construction uses the unchanged sampler documented in
[GAUSSIAN_COUPLING.md](GAUSSIAN_COUPLING.md), with the frozen upstream objects

```text
results/gaussian_coupling1/CERTIFICATE.json
1ab5e225ec3e0a570910afc11026adf38ce39d01a4a8a16caa9ad487c95e1e32
results/gaussian_coupling1/RUN.json
dbc5b27e6ea7e147133b48194889599a7fe1471bdb218c0513cb1ccb96a77d61
```

The consumed mathematical interfaces are the unchanged
[nodal core](nodal_core.py), [Hessian core](hessian_grid_core.py),
[integer H₀ algorithm](exact_h0.py), and their linked derivations above.
The new receipt must bind their actual source bytes, the sampler inputs and
the new bridge. Its verifier must regenerate the polynomial, exact lift,
centers, Hessian bound, barcode and bin counts, and check the rendered report.
Rebinding the hashes of a false result does not replace this semantic replay.

Focused controls cover the sine sign, factor two, DC and transform
normalization; complete mode and word ordering; the 193-bit conversion case;
minimal lift; sample and error units; malformed types and aliased grids; and
altered outputs with rebound receipt hashes. The existing independent
connectivity checker remains necessary for each produced barcode.

This bridge supplies a deterministic finite-polynomial certificate. It does
not prove any actual word source IID, attach an unconditional Gaussian
probability to the fixture, apply K24/K32 tail budgets to K1, promote the
historical PCG64 polynomial, or establish an infinite-field observation,
expected-count bound, lifetime asymptotic or numerical remainder. The
historical coupling error and this fixture's infinite-field diagram bound
remain unknown in the output. Author work, substantive review, formal
evidence and scientific acceptance retain their separate meanings.
