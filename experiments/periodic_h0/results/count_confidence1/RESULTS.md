# What sampling error would cost

The finite-cutoff bar cap gives simultaneous mean-confidence intervals
under a fixed plan and the declared independent uniform-word input law.
For eight bins and 40,000 draws, a radius of 1% of the cap has the
following exact upper bound on family failure probability. These are
**prospective calculations; no ensemble has been sampled.**

| Cutoff | Cap | Count radius | Failure bound | Draws for radius one at the same bound |
|---|---:|---:|---:|---:|
| 1 | 16 | 4/25 | 0.005367402046440190 | 1024 |
| 24 | 9216 | 2304/25 | 0.005367402046440190 | 339738624 |
| 32 | 16384 | 4096/25 | 0.005367402046440190 | 1073741824 |

The intervals are
`[max(0, mean(L)-r-C*p), min(C, mean(U)+r+C*p)]`, with `C=16*K^2`.
Every draw remains in the denominator. An unresolved computation supplies
`[0,C]`. Grid refinement may depend on the whole sample because concentration
is applied to fixed latent polynomial counts and rows only bound those counts.
Bins, radius and sample count must be fixed before inspecting the sample.

The large sample sizes at larger cutoffs expose the cost of this worst-case
bound. They are not a recommendation to launch an ensemble. The gap between
lower and upper grid means and the inherited clipping correction add uncertainty.

Exact abstract finite-law checks detect omitted cap scaling, omitted bin
multiplicity, dependent rows and discarding unresolved outcomes. They are
neither polynomial data nor evidence of an IID physical source.

The arithmetic checks do not authenticate input-law or row-certificate premises.
No observed ensemble mean, infinite-field inference or lifetime-law confirmation
is certified. The existing deterministic word fixture remains deterministic.

[Proof and assumptions](../../COUNT_CONFIDENCE.md) · [Certificate](CERTIFICATE.json) · [Receipt](RUN.json)

```sh
python -B -S experiments/periodic_h0/count_confidence_certificate.py --verify experiments/periodic_h0/results/count_confidence1
```

Fixed-sample simultaneous confidence transfer for finite-cutoff expected counts under the stated IID word law and pointwise row bounds. Exact prospective plans and abstract finite-law controls, not observed Gaussian data, authenticated sampling, infinite-field inference or lifetime-law confirmation.
