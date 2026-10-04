# From spectral samples to shrinking persistence bins

This conditional result explains when increasingly accurate spectral samples
recover the expected number of short H0 persistence bars in a smooth Gaussian
field. It extends the earlier exact-sample argument to include spectral tails,
field normalization and certified errors in the supplied grid values.

For a fixed torus of side 24 and lifetime bins
`[lambda tau_n, mu tau_n)`, the conclusion is

```text
E |sampled count - continuum count| / E continuum count -> 0.
```

A sufficient accuracy schedule is

```text
(grid spacing^2 + spectral tail scale) sqrt(log n)
    + certified sample error = o(tau_n),
```

together with an explicit bound on the probability that the sample certificate
fails. The proof controls the expected bar counts on those failure events;
a small failure probability by itself would not suffice.

[Read the statement and proof](PROOF.md) · [Exact source identities](SOURCES.json)

The result assumes the stated continuum-density and unconditioned
critical-count interfaces. It supplies an asymptotic implication, with an
explicit error formula but unevaluated continuum constants. It does not
certify the stored PCG64/NumPy experiment, floating-point persistence
software, confidence intervals or a usable finite-grid threshold. Fixed
precision and a fixed normalization error cannot be ignored as the lifetime
scale tends to zero.

This is an author-side candidate awaiting its own nonauthor technical review.
The review of the earlier exact-sample lemma does not cover this extension.
The prior approximation and Gaussian-tail sources are unchanged.
