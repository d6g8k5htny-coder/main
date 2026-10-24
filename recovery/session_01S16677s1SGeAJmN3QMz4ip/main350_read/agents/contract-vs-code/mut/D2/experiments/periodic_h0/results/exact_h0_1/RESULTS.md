# Exact barcode and certified finite-field counts

One frozen finite polynomial (seed label 34000), on the C38 **1024² exact dyadic grid**.

The integer-only sweep gives **69 positive finite H₀ bars** and one essential class.
A separate plateau and breadth-first connectivity check verifies the birth and death witnesses.
Zero-lifetime pairs are omitted; the longest finite interval is retained.

The barcode endpoints have **zero arithmetic error** relative to these exact samples. The diagram bound
against the frozen smooth finite polynomial is at most **0.001061264136756651120320**
(the unchanged C38 interpolation/stability bound).

| Lifetime bin | Exact grid count | Smooth finite-field lower bound | Smooth finite-field upper bound |
|---|---:|---:|---:|
| [1/1000, 1/500) | 0 | 0 | not bounded by this argument |
| [1/500, 1/250) | 0 | 0 | not bounded by this argument |
| [1/250, 1/125) | 0 | 0 | 1 |
| [1/125, 2/125) | 1 | 0 | 1 |
| [2/125, 4/125) | 1 | 0 | 2 |
| [4/125, 8/125) | 5 | 4 | 6 |
| [8/125, 16/125) | 3 | 3 | 3 |
| [16/125, 32/125) | 5 | 5 | 5 |

Bounds use contracted and expanded bins with exact rational endpoints.
They are counts for one deterministic smooth finite field, not expected intensities,
a Gaussian-law fit, a tail enclosure, a remainder estimate or a held-out test.

[Algorithm and proof](../../EXACT_H0.md) · [Exact endpoints and counts](CERTIFICATE.json)
· [Source and execution receipt](RUN.json) · [C38 spatial bound](../hessian1/RESULTS.md)

```sh
python -B -S experiments/periodic_h0/exact_barcode_certificate.py --verify experiments/periodic_h0/results/exact_h0_1
```

Exact periodic vertex-cubical H0 barcode of the C38 dyadic grid and deterministic bin bounds for its one frozen finite polynomial. No historical FFT, ideal Gaussian ensemble, infinite-field, lifetime-law or held-out confirmation certificate.
