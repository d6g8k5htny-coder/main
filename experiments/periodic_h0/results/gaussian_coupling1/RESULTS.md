# A constructive Gaussian coefficient coupling

A finite-word sampler now has an explicit uniform error budget **under
the declared independent uniform-input model**. It constructs a new
dyadic polynomial; it does not certify the law of any physical or
pseudorandom bit source. The historical polynomial remains separate.

| Cutoff | Low-mode error at most | Clipping failure at most | Full-field error at most | Combined failure at most |
|---|---|---|---|---|
| 24 | 0.000000000000000001465715649537 | 0.000000000003032625717894 | 0.001308340741324927550808 | 0.000000000004299484255617 |
| 32 | 0.000000000000000001465732013400 | 0.000000000005336461331986 | 0.000000386931068362268551 | 0.000000000007008709160237 |

The full-field column includes the threshold8 infinite spectral tail and
normalization error. It compares the ideal smooth field with the new
polynomial, before any grid or barcode computation. Add a verified
finite-polynomial barcode error to use the persistence bound.

The only executed construction is a nine-word deterministic arithmetic
fixture at cutoff1. No new Gaussian barcode or held-out confirmation
sample was generated. The table is a uniform sampler theorem for
cutoffs24 and32, not an observation of those ensembles.

[Proof and input assumptions](../../GAUSSIAN_COUPLING.md) ·
[Exact certificate](CERTIFICATE.json) · [Source receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/gaussian_coupling_certificate.py --verify experiments/periodic_h0/results/gaussian_coupling1
```

Prospective side24 finite-bit inverse-normal coupling under explicit ideal IID uniform-word inputs. No RNG law, historical Gaussian sample, barcode experiment, lifetime law, or expected-count confirmation certified.
