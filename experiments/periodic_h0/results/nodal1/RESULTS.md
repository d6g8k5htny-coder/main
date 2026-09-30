# A certified nodal error for one finite field

Seed label **34000**, grid **128×128**.

This new snapshot records every real float64 FFT sample and compares it
with an independent integer evaluator of the frozen exact finite polynomial.
The replay uses only Python integers, rational arithmetic and exact parsing
of the stored float64 hexadecimal values.

| Bound | Certified upper endpoint |
|---|---:|
| Maximum nodal error eta | 0.000000000000000994777712 |
| Spatial interpolation budget B | 0.479769953926643024 |
| Abstract filtration diagram bound eta+B | 0.479769953926644019 |

Decimal endpoints are rounded upward; exact rational values and all FFT
stage error bounds are in [CERTIFICATE.json](CERTIFICATE.json). The last
row applies the declared deterministic interpolation/stability argument
to the mathematical filtration of these supplied samples. No persistence
software or rounded barcode endpoint is certified by this calculation.

This does not certify historical pilot/refinement samples, the infinite
Gaussian field, its sampling law, a spectral tail or a lifetime asymptotic.
The large spatial budget remains the limiting input on this coarse grid.

[Derivation](../../NODAL_CERTIFICATE.md) · [Samples](SAMPLES.json) ·
[Execution and source identity](RUN.json)

```sh
python -B -S experiments/periodic_h0/nodal_certificate.py --verify experiments/periodic_h0/results/nodal1
```
