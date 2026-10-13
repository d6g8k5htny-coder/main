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

A sufficient schedule from proof (8) is

```text
M_n >= m_n (with the exact full denominator, M_n = infinity, allowed),
[h_n^2 + exp(-beta m_n^2)] sqrt(log n) + eta_n = o(tau_n),
n^2 q_n + sqrt(q_n) = o(tau_n^(2/3)).
```

Here `h_n=24/n` is the grid spacing, `m_n` is the spectral cutoff,
`M_n` is the normalization cutoff, `beta=pi^2/24^2`, `eta_n` bounds the
supplied-value error on the certificate event, and `q_n` bounds its failure
probability. A fixed finite `M_n` does not satisfy this schedule as `m_n`
grows: its normalization error must be retained. The proof controls the
expected bar counts on those failure events;
a small failure probability by itself would not suffice.

[Read the statement and proof](PROOF.md) · [Exact source identities](SOURCES.json)

The result assumes the stated continuum-density and unconditioned
critical-count interfaces. It supplies an asymptotic implication, with an
explicit error formula but unevaluated continuum constants. It does not
certify the stored PCG64/NumPy experiment, floating-point persistence
software, confidence intervals or a usable finite-grid threshold. Fixed
precision and a fixed normalization error cannot be ignored as the lifetime
scale tends to zero.

The conditional implication has its own source-bound technical reviews:
[OpenAI/Codex](https://github.com/d6g8k5htny-coder/main/pull/264#pullrequestreview-5404446570)
and [Anthropic/Claude](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5977511080).
Both report `PASS_TECHNICAL_SCOPED` for the original proof and retain its
imported hypotheses; organizational-independence credit is zero. These reviews
do not certify an actual sampler or promote the imported sources' scientific
status. The review of the earlier exact-sample lemma is a separate record.

The [original reading cut](https://github.com/d6g8k5htny-coder/main/tree/ad7ea9156138911f44b91bcc0e9497a4ba53f181/experiments/periodic_h0/spectral_shrinking_bins)
retains the pre-review wording. This reader-summary correction incorporates
proof (8)'s normalization condition and links the delivered reviews;
`PROOF.md` and `SOURCES.json` remain unchanged.
