# A smaller certified spatial error

Frozen finite field seed **34000**; Hessian grid **256²**;
deterministically defined dyadic sample grid **1024²**.

| Quantity | Certified upward endpoint |
|---|---:|
| Global Hessian operator norm | 7.727889607387098644 |
| Earlier coefficient-triangle spatial bound | 0.007496405530103797242139 |
| New spatial bound B | 0.001061264136756651119821 |
| Nodal error eta | 0.000000000000000000000499362592 |
| Abstract filtration bound eta+B | 0.001061264136756651120320 |

Every sample is the exact real integer evaluator center divided by 2^96.
Replay regenerates all samples and checks their ordered digest; the large
array is not stored. These are new mathematical samples, not the historical
NumPy FFT arrays, and no persistence software or computed barcode is certified.

The following conditions concern the proved abstract-filtration bin sandwich.
They do not provide counts, confidence intervals or an asymptotic window.

| Lifetime bin | Clean upper condition a > 2 epsilon | Nonempty contracted bin |
|---|---|---|
| [1/1000, 1/500) | no | no |
| [1/500, 1/250) | no | no |
| [1/250, 1/125) | yes | no |
| [1/125, 2/125) | yes | yes |
| [2/125, 4/125) | yes | yes |
| [4/125, 8/125) | yes | yes |
| [8/125, 16/125) | yes | yes |
| [16/125, 32/125) | yes | yes |

Exact finite rounded polynomial and the abstract filtrations of deterministically regenerated dyadic center samples only. No historical FFT, computed barcode, Gaussian-law, infinite-field or lifetime-law certificate.

[Derivation](../../HESSIAN_GRID.md) ·
[Exact certificate](CERTIFICATE.json) · [Execution/source receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/hessian_grid_certificate.py --verify experiments/periodic_h0/results/hessian1
```
