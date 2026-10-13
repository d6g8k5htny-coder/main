# Exact finite-polynomial derivative bounds

These numbers enclose the derivative majorants of explicitly stored
rounded finite Fourier polynomials. They do **not** certify historical FFT
samples, the ideal Gaussian sampling law, the infinite field, or bar counts.
Every displayed upper bound is rounded upward to 15 decimal places.

| Seed | Hessian operator majorant H | Component interpolation budget B at 1024² |
|---|---:|---:|
| 34000 | 67.460271466058527 | 0.007496405530104 |
| 34001 | 67.035480784400099 | 0.007483490927181 |
| 34002 | 64.195163697286401 | 0.007186147859908 |
| 34003 | 65.742948051625869 | 0.007360423112138 |
| 34004 | 66.796647806381118 | 0.007521684810746 |
| 34005 | 65.970068591698600 | 0.007451983427933 |
| 34006 | 70.804259327680146 | 0.007929637977099 |
| 34007 | 65.855456109235311 | 0.007349520395714 |

The componentwise interpolation bound is no larger than the operator bound
for these inputs. `B` is only the spatial part of `epsilon = eta + B`:
the nodal error `eta` is not supplied and is stored as null. Consequently
the diagram-error field is also null. Setting either value to zero would
change the claim and fails the exact replay.

With *hypothetical exact nodal samples*, the largest certified spatial
budget at 1024² is at most 0.007929637977099. The clean bin upper
bound would then require `a > 2 B`. This conditional observation does not
certify any historical bin or select an asymptotic confirmation window.

[Derivation and limitations](../../FINITE_CERTIFICATE.md) ·
[Exact coefficients](COEFFICIENTS.json) · [Certificate](CERTIFICATE.json) ·
[Execution and source identities](RUN.json)

Replay from the repository root without numerical dependencies:

```sh
python -B -S experiments/periodic_h0/finite_certificate.py --verify experiments/periodic_h0/results/certificate8
```
