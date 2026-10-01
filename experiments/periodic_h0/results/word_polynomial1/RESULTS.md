# From explicit words to a certified smooth-field barcode

This deterministic example connects the new finite-word sampler to
exact Fourier-grid evaluation and periodic H₀ persistence. It uses
nine declared words, cutoff 1, a 128×128 sample grid and a 32×32
Hessian grid. No random source is used.

The exact grid has **3 positive finite bars** and one essential class.
A separate breadth-first connectivity calculation verifies the barcode.
Its endpoints are exact for the computed dyadic samples.

The diagram bound against the original smooth finite polynomial is at most **0.000375408633135674692468**.
The bounds below account for both nodal arithmetic and spatial interpolation.

| Lifetime bin | Exact grid count | Smooth polynomial lower count | Smooth polynomial upper count |
|---|---:|---:|---:|
| [1/1000, 1/500) | 0 | 0 | 0 |
| [1/500, 1/250) | 0 | 0 | 0 |
| [1/250, 1/125) | 0 | 0 | 0 |
| [1/125, 2/125) | 1 | 1 | 1 |
| [2/125, 4/125) | 0 | 0 | 0 |
| [4/125, 8/125) | 0 | 0 | 0 |
| [8/125, 16/125) | 0 | 0 | 0 |
| [16/125, 32/125) | 0 | 0 | 0 |

Thus the smooth finite polynomial has **exactly one finite bar in
[1/125,2/125)** and none in the other seven displayed bins. This does
not exclude bars below 1/1000 or above 32/125. The two smaller positive
grid bars are below twice the diagram error and are not certified
smooth-field detections. Counts here are for this one polynomial,
not expected intensities or evidence for a Gaussian lifetime law.

The real Fourier coefficients are converted without rounding. An
exact positive power of two lifts them into the existing fixed-point
evaluator; sample units and all error bounds are divided back by
that same factor. The 96-bit twiddle accuracy is unchanged. The
`seed:0` field in the compatibility record is only record ID 0.

[Derivation and scope](../../WORD_POLYNOMIAL.md) ·
[Exact inputs, endpoints and bounds](CERTIFICATE.json) ·
[Source receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/word_polynomial_certificate.py --verify experiments/periodic_h0/results/word_polynomial1
```

One deterministic cutoff 1 side 24 finite-word polynomial: exact coefficient conversion, certified grid/Hessian error and finite H0 lifetime-bin counts. No IID source law, Gaussian observation, infinite-field diagram bound, historical coupling or lifetime-law confirmation certified.
