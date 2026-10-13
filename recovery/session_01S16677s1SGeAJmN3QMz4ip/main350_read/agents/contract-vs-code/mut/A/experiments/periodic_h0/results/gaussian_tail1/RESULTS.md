# A quantified omitted-mode bound

For the **ideal normalized Gaussian field on the side-24 square torus**,
the following uniform tail bounds hold outside the stated probability budget.
Every mode outside the cutoff is included, including modes beyond64.

| Square cutoff K | Rayleigh threshold t | Uniform tail at most | Failure probability at most |
|---|---|---|---|
| 24 | 6 | 0.001006390726275545712889 | 0.000001526934246675641069 |
| 24 | 8 | 0.001308340741324926085092 | 0.000000000001266858537724 |
| 32 | 6 | 0.000000295299183735264832 | 0.000002015504643440881766 |
| 32 | 8 | 0.000000386931068360802819 | 0.000000000001672247828251 |

The table concerns the ideal field and its coupled ideal low modes.
K32 is a possible future truncation, not the stored K24 polynomial.
The infinite versus mode64 normalization loss is below **3 × 10⁻⁶⁴**.

To connect a certified finite polynomial P to this field, additionally prove
`||F64,K - P||∞ <= rho` for the same low-mode Gaussian coefficients.
Then the existing finite barcode bound composes with the tail as

`epsilon_total <= epsilon_finite + tau + rho + normalization_loss * ||P||∞`.

The retained seed34000 polynomial has a certified norm upper bound of
15.810045844661777130197921. Its coefficient-coupling error
**rho is still unknown**, so no full-field error or Gaussian probability is
attached to its barcode. A missing premise remains null, not zero.

[Proof and exact conventions](../../GAUSSIAN_TAIL.md) ·
[Exact certificate](CERTIFICATE.json) · [Source receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/gaussian_tail_certificate.py --verify experiments/periodic_h0/results/gaussian_tail1
```

Planar side24 ideal Gaussian omitted-mode probability bound and conditional finite-polynomial coupling contract. Historical Gaussian sampling, coefficient coupling, lifetime-law and held-out confirmation remain uncertified.
