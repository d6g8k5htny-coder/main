# Bounding the numerical values at every grid point

[Experiment](README.md) · [One-field result](results/nodal1/RESULTS.md) ·
[Spatial certificate](FINITE_CERTIFICATE.md) · [Approximation argument](APPROXIMATION.md)

The stored sample snapshot for seed label 34000 on a 128×128 grid now has an
exact, reproducible nodal-error bound. This supplies the missing numerical input
for this **new snapshot of one finite polynomial**. It does not supply a receipt
for the earlier pilot or refinement, whose sample arrays were not retained.

## The object and the new measurement

The unchanged [C36 coefficient file](results/certificate8/COEFFICIENTS.json)
defines the exact dyadic Fourier polynomial F. Its coefficients already include
the original rounding of the random draws and spectral weights. They are the
mathematical input here, not an ideal Gaussian law.

The producer places the selected record and its conjugates in a complex128
array. It verifies that each stored rational coefficient is exactly representable
as a float64 component before calling NumPy `ifft2(norm='forward')`. This inverse
transform has positive exponential sign and no normalization factor. At node
(i,j), the intended exact value is F(24i/n,24j/n). The producer stores the real
part of every output as a canonical hexadecimal float64 string in row-major
x-then-y order. No sample or unfavorable node is omitted.

The certificate concerns these real supplied values. It does not need to infer
an internal error bound for NumPy: an independent evaluator encloses the exact
polynomial at every node and compares every stored value against that enclosure.
A different FFT on a different platform may yield different sample bytes; those
bytes need their own certificate and new execution receipt.

## Exact fixed-point inverse transform

Let S=2^96. Each complex center is represented by two Python integers divided by
S. The retained coefficients are exactly representable at this scale. The core
rejects an input that is not; it does not silently discard coefficient bits.
There is no frequency collision because n exceeds twice the retained cutoff.

Bit reversal followed by the usual radix-two inverse butterflies computes the
positive-exponent discrete Fourier sum. Apply the one-dimensional transform
along rows and then columns. There is no division by n or n². The two-dimensional
sum equals the finite polynomial at the declared grid coordinates, by grouping
the terms by their two integer frequencies.

Every multiplication forms its entire integer complex-product numerator first,
then rounds each component once to the nearest integer (a fixed tie convention
is used). Its error per component is at most 1/(2S). Additions and subtractions
of the integer centers are exact; Python integers do not overflow.

### Twiddle enclosures

Reduce each required root of unity by exact quadrant/octant symmetries to an
angle of absolute value at most pi/4. Use the existing rational Machin enclosure
of pi, with a rational midpoint for the evaluation angle. Sine through degree31
and cosine through degree30 have Taylor remainder bounded by

```
(pi_upper/4)^32 / 32!
```

on that interval; the sine's extra omitted term is smaller there. The error
from the midpoint angle is at most `(pi_upper-pi_lower)/8`, because sine and
cosine are 1-Lipschitz and the reduced angle is at most pi/4. The core verifies
with rational arithmetic that the combined error is at most 1/(2S). Rounding
the rational Taylor centers to scale S adds at most 1/(2S). Thus every real
and imaginary twiddle coordinate has error at most 1/S. Exact symmetries preserve
that bound. No floating-point trigonometric value is used in this proof.

### Error carried through every stage

Suppose every exact input component differs from its current integer center
by at most E. Let K be the largest sum of absolute real and imaginary integer
components among all current nodes. For the exact butterfly `a ± w b`, use
`|Re w|+|Im w| <= sqrt(2) < 3/2`. The propagated input error is at most
`E+(3/2)E`. The twiddle error times the *computed center* of b is at most
`K/S²` per component. The final product rounding adds at most `1/(2S)`.
Consequently both butterfly outputs have component error at most

```
E_next = (5/2) E + K/S² + 1/(2S).
```

This decomposition uses the exact twiddle on the input error and the approximate
input center on the twiddle error, so it omits no error-product term. Start at
E=0 for exactly representable coefficients. The checker records K and E at all
`2 log2(n)` stages, including the transition from rows to columns, using exact
rational arithmetic throughout.

Let z_v/S be a final real center, E_final the final bound, and s_v the exact
dyadic value represented by a stored float64 string. Compute

```
D = max_v |s_v - z_v/S|,
eta = D + E_final.
```

Then `max_v |s_v-F(v)| <= eta`. This is a maximum over the entire finite grid,
not a probabilistic statement or a spot check. The certificate also hashes the
entire reconstructed complex-center array, so a replay checks both components
and their order even though the claimed supplied samples are real.

## What combining this with the spatial bound establishes

The unchanged spatial calculation supplies the exact B for the same polynomial
and grid. The [deterministic argument](APPROXIMATION.md) therefore gives
`||F-p_h||_infinity <= eta+B` for the specified continuous PL representatives.
Its stated stability input gives the same upper bound for the distance between
the **mathematical** H0 diagrams of F and the declared supplied-sample filtrations,
including the vertex cubical convention through its minimum-vertex triangulation.

The certificate names this `abstract_diagram_bound`. The separate
`computed_barcode_error` stays null. No GUDHI execution or rounded barcode endpoint
is certified here. No observed bin count is promoted. The spatial term on this
coarse grid remains far too large for the shortest experimental bins, regardless
of how small the nodal term is.

The infinite spectral tail, relation to the intended Gaussian law, numerical
asymptotic remainder and cutoff, held-out test, and required independent human
review remain separate inputs. Historical C36 null values keep their meaning;
the new certificate does not alter that frozen delivery.

## Replay, generation and custody

Only the standard library is needed to replay the stored certificate:

```sh
python -B -S experiments/periodic_h0/nodal_certificate.py --verify experiments/periodic_h0/results/nodal1
```

With the pinned numerical dependencies installed, record a fresh snapshot in a
new directory:

```sh
python -B experiments/periodic_h0/run_nodal_certificate.py --output /tmp/new-nodal-certificate
```

The receipt binds both output objects, the exact coefficient bytes and the
consumed core, producer, verifier, derivative checker, approximation argument
and dependency file. Its schema and scope are checked; its complete bytes are
bound by a sidecar. The report is regenerated from the exact certificate.
A self-consistent replacement of both receipt and sidecar is a different object,
not proof of an earlier execution. Compare immutable commit and delivery hashes
for external identity. Environment strings are observations, not authentication.

Tests and the nonauthor technical review cover sign and normalization, both axes,
conjugation, nonrepresentable coefficients, irrational roots, every sample
including the final node, malformed values/layouts, altered sources or scope,
and understated error bounds. These are scoped verification checks, not a
formalization of persistence or evidence of organizational independence.
