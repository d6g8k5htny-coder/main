# Mean-adaptive confidence for certified finite-field counts

[Fixed-sample hypotheses and retained rows](COUNT_CONFIDENCE.md) ·
[Finite-cutoff expectation transfer](FINITE_COUNT_LOSS.md) ·
[Exact arithmetic](mean_adaptive_confidence.py)

The existing Hoeffding bound uses the largest possible variance associated
with the cap `C=16 K^2`. A Chernoff bound that retains the mean can give
substantially smaller intervals for sparse counts. This note supplies that
alternative with exact outward arithmetic. The old theorem, interface and
frozen results are unchanged.

This is a conditional finite-sample theorem and arithmetic implementation.
It generates no fields and evaluates no observed Gaussian ensemble. It
does not authenticate an input law, a sampling plan or the grid evidence
behind supplied count rows.

## 1. The unchanged model and the fixed plan

Fix `1<=K<=64`, `C=16 K^2`, and the sampler's same-cutoff budgets `rho,p`.
The target remains `G=F64,K` with the finite normalization `S_alpha,64`.
Use the exact word polynomial, uniform-word probability space and latent
observables `X_j^-`, `X_j^+` from [COUNT_CONFIDENCE.md](COUNT_CONFIDENCE.md).
In particular,

```text
0 <= X_j^- <= X_j^+ <= C,
mu_j^- = E X_j^-,                 mu_j^+ = E X_j^+,
max(0,mu_j^- - C p) <= nu_j <= min(C,mu_j^+ + C p),
nu_j = E N_G([a_j,b_j)).
```

Before observing the words, fix a positive integer field count `n`, the
complete family of `m>=1` positive half-open lifetime bins, and a rational
`beta>0`. Assume all `n` word blocks have the specified IID law, including
independence of the words within each block. The bins may overlap; their
counts and the two signs need not be independent.

For every planned field and bin, retain an actual pointwise bracket

```text
0 <= L_ij <= X_j^-(J_i) <= X_j^+(J_i) <= U_ij <= C.
```

The earlier word-polynomial and grid arguments can supply those brackets.
An unresolved computation contributes `[0,C]` and remains in the denominator
`n`. Adaptive grid work may depend on all the observed words, provided the
final brackets remain valid. It does not change these fixed latent random
variables. No independence or identical-distribution claim about the
computed brackets is required.

`n`, the bin family and `beta` are fixed-plan premises. A numerical API
cannot infer whether they were chosen prospectively. Optional stopping,
postselected bins or choosing beta after observing favorable counts are
not covered. A missing row does not establish that an unrecorded draw
occurred, and plausible count pairs do not establish their grid bounds.

## 2. Bounded-variable Chernoff inequality

Let `Y=X/C` take values in `[0,1]` and put `q=E Y`. For every real `t`,
convexity gives

```text
exp(tY) <= 1-Y+Y exp(t),
E exp(tY) <= 1-q+q exp(t).
```

For independent copies and `s>q`, exponential Markov with `t>0` yields

```text
Pr(Ybar >= s) <= exp(-n t s) (1-q+q exp(t))^n.
```

When `0<q<s<1`, the minimizing value has
`exp(t)=s(1-q)/(q(1-s))`. Substitution proves

```text
Pr(Ybar >= s) <= exp(-n kl(s||q)),
kl(s||q) = s log(s/q) + (1-s) log((1-s)/(1-q)).
```

Taking `t<0` proves `Pr(Ybar<=s)<=exp(-n kl(s||q))` for `s<q`.
The cases `s=0,1` follow by the corresponding limit. This argument uses
only boundedness and independence, not a Bernoulli distribution or a
variance estimate. It is the classical bounded-variable Chernoff--Hoeffding
argument; see [Hoeffding, 1963](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830).
The inequalities needed here are derived above rather than imported as an
unverified software convention.

Use `0 log(0)=0`. For `q=0` or `q=1`, divergence is infinite unless `s=q`,
where it is zero. An actual bounded variable of mean zero or one equals
that endpoint almost surely, so the confidence claim at these means is
immediate.

## 3. Inversion, monotonicity and simultaneous transfer

For `s in [0,1]`, define the closed ideal interval

```text
I_beta(s) = {q in [0,1]: n kl(s||q) <= beta}
          = [ell_beta(s), u_beta(s)].
```

It contains `s`. At interior `s`, its two endpoints are the unique roots
of `n kl(s||q)=beta` on `(0,s)` and `(s,1)`. Indeed,

```text
partial_q kl(s||q) = (q-s)/(q(1-q)),
partial_s kl(s||q) = log(s(1-q)/(q(1-s))).
```

The first derivative is negative below `s` and positive above it, with
divergence tending to infinity at the corresponding outer endpoint.
The implicit root derivative `-partial_s/partial_q` is positive on
both root branches. Their continuous endpoint extensions are

```text
ell_beta(0)=0,                  u_beta(0)=1-exp(-beta/n),
ell_beta(1)=exp(-beta/n),        u_beta(1)=1.
```

Thus both ideal endpoints are nondecreasing in `s` on the closed interval.

For a fixed true mean `q`, the event `ell_beta(Ybar)>q` requires
`Ybar>q` and `n kl(Ybar||q)>beta`. On `[q,1]` that divergence is increasing.
If the event is possible, the Chernoff bound at the unique threshold where
the divergence equals `beta/n` bounds its probability by `exp(-beta)`.
If there is no such threshold, the event is empty. The lower-tail argument
similarly gives

```text
Pr(ell_beta(Ybar)>q) <= exp(-beta),
Pr(u_beta(Ybar)<q)   <= exp(-beta).
```

Strict exclusion, rather than exclusion at equality, handles atoms at an
endpoint. Continuity of the divergence gives the same conclusion when a
threshold is approached by a limit.

For bin `j`, put `s_Lj=Lbar_j/C` and `s_Uj=Ubar_j/C`. The pointwise row
brackets, followed by monotonicity, give

```text
ell_beta(s_Lj) <= ell_beta(Xbar_j^-/C),
u_beta(s_Uj)   >= u_beta(Xbar_j^+/C).
```

Apply the first tail to each fixed latent `X_j^-` and the second to each
fixed latent `X_j^+`. A union bound over `2m` events proves, with probability
at least `1-R`, simultaneously for every bin,

```text
R = min(1, 2m exp(-beta)),

max(0, C ell_beta(s_Lj)-C p) <= nu_j
                            <= min(C, C u_beta(s_Uj)+C p).
```

This is why processing dependent on the complete sample is permitted: the
random variables entering concentration are the unchanged latent counts.
The clipping correction remains `C p`, an expectation loss, not `n C p`.
The interval covers an unconditional mean, not the next realized count.

The implementation returns `ell_out(s)<=ell_beta(s)` and
`u_out(s)>=u_beta(s)`. Replacing ideal endpoints by these outward values
therefore preserves the theorem. Numerical root brackets at different
precision settings need not nest; the proof requires their outward
containment, not monotonic rounding by a particular computer routine.

## 4. Exact arithmetic and conservative termination

All statistical real-valued inputs must be actual `fractions.Fraction`
objects. Integers such as `n,m,K,bits` must have strict integer type, so
booleans are rejected. Floats, infinities and NaNs are not silently
converted to rational hypotheses.

`log_bounds(v,bits)` accepts positive rational `v` and `1<=bits<=256`.
For `v>=1`, write `v=2^k r` with integer `k>=0` and `1<=r<2`. For
`z=(r-1)/(r+1)`, the exact integral of the geometric series gives

```text
log(r) = 2 sum_{j>=0} z^(2j+1)/(2j+1),       0<=z<1/3.
```

After `M` terms, the remainder is nonnegative and at most

```text
2 z^(2M+1) / ((2M+1)(1-z^2)).
```

The same formula at `z=1/3` encloses `log(2)`. Each constituent enclosure
is refined until its width is at most `2^(-bits)/(k+1)`, so the combined
width is at most `2^(-bits)`. The geometric decay guarantees termination.
For `0<v<1`, negate and reverse the enclosure of `log(1/v)`. All sums,
powers, reductions and error comparisons are rational; no floating log
is used.

`kl_bounds(s,q,bits)` combines those log enclosures with nonnegative
weights `s,1-s`. Its width is at most `2^(-bits)`. Zero-weight terms are
omitted, and nonnegativity of divergence permits clamping the lower bound
at zero. This function requires `0<q<1`; the confidence routine handles
the infinite-divergence endpoints separately.

`confidence_bounds(s,n,beta,bits=48)` starts the lower-root bracket at
`[0,s]` and the upper-root bracket at `[s,1]`. Each receives at most `bits`
bisections. At a midpoint, it computes a KL enclosure using
`min(256,bits+16+n.bit_length())` log bits. It updates a side only when
the corresponding comparison of `n*KL` with `beta` is certified.
If the enclosure straddles beta, it stops that root search. Returning
the outer bracket endpoint is conservative even if the midpoint is
exactly a root. This rule guarantees finite work and cannot convert an
unresolved comparison into a precise claim.

If every comparison is decided, each root's bracket width is at most
`2^(-bits)`. Otherwise the bracket can be wider; `bits` is not an
unconditional accuracy promise. Increasing arithmetic precision can
improve usefulness but is unnecessary for validity.

For `s=0,1`, the routine reuses `gaussian_tail.exp_neg_bounds(beta/n)`.
The lower exponential endpoint supplies the outward confidence endpoint.
It also uses `1-exp(-x)<=x` to retain the analytic zero-count bound below
the exponential evaluator's rounding scale. When `beta/n>4096`, outside
that evaluator's domain, it returns the universal `[0,1]` interval.

`risk_upper(m,beta)` returns an exact upper bound for R. For beta above
4096 it evaluates at 4096, which remains conservative because the
exponential decreases. `mean_interval(K,p,rows,beta,n,bits=48)` checks
that all `n` rows are present, replaces `None` by `[0,C]`, computes the
two normalized rational averages, and applies the unchanged finite-count
expectation-transfer arithmetic. Domain validation establishes none of
the external statistical or grid premises.

## 5. What the sparse-count improvement means

If every certified upper row for a bin is zero, then `s_U=0` and

```text
nu <= C(1-exp(-beta/n))+C p <= C beta/n+C p.
```

For `beta=8,m=8`, the family failure upper bound is the same as in the
earlier planning example: its upward decimal rendering is
`0.005367402046440190`. At K32, `C=16384` and `n=8C=131072` imply an upper
target-mean bound of `1+C p` **if all certified upper rows are zero**.
The earlier sample-independent Hoeffding radius-one plan uses
`n=4C^2=1073741824` at that same displayed risk level.

The smaller number is a conditional arithmetic example, not a universal
sample-size recommendation, a promise of zero rows, or an observed mean.
Nonzero upper rows, expanded-bin overlap, unresolved grids and clipping
increase the reported bound. No fields were consumed to obtain it.

From the repository root, the small exact calculation is:

```sh
PYTHONPATH=experiments/periodic_h0 python -B - <<'PY'
from fractions import Fraction as Q
from mean_adaptive_confidence import confidence_bounds, risk_upper
C = 16*32**2
n = 8*C
lo, hi = confidence_bounds(Q(0), n, Q(8))
assert lo == 0 and C*hi <= 1
assert risk_upper(8, Q(8)) < Q(1,100)
print('Hypothetical zero-upper-row count-mean bound <= 1 before C*p')
PY
```

## 6. Controls and remaining obligations

The tests include an independent small-case rational-power oracle. For
integer total `T` and `s=T/(nC)`,

```text
[exp(-n kl(s||q))]^C
    = (q/s)^T ((1-q)/(1-s))^(nC-T),
```

with the limiting formulas at `s=0,1`. Integer powers of exact rationals
therefore check root orientation without using the new log routine.
Exact enumeration of the equally weighted law `{0,3,16}`, with `n=6`
and beta 4, gives exclusion probability `1/729`. This is an abstract
three-valued law, not polynomial data or the uniform-word ensemble.

Negative controls distinguish the hypotheses from arithmetic success:

- For `X=16` with probability `1/16`, otherwise zero, and `n=16`, the
  mean is one and the all-zero probability is `(15/16)^16>1/4`. Omitting
  C from the zero-count formula falsely gives upper mean `8/16=1/2`,
  with failure exceeding the claimed beta-8 budget.
- Repeating one random bit eight times violates the field independence
  premise and can miss its mean with probability one.
- For IID Bernoulli observations, stopping at the first excluded mean
  through time eight, with beta 2, has exact failure probability `35/128`.
  This exceeds `2 exp(-2)`: a fixed-sample interval is not an
  optional-stopping guarantee.
- With one uniformly chosen bin of four having count 16, all others zero,
  the selected bin always excludes its mean four at beta 1. Falsely
  reporting it as a single predeclared bin gives a risk below one;
  accounting for the complete family gives the valid vacuous bound one.

Other controls cover exact-log range reduction, reciprocal signs,
root rounding, zero and full counts, deliberately inconclusive low
precision, clipping factors, malformed inputs, unchanged denominators and
adaptive unresolved rows. Run them with

```sh
python -B -m unittest discover -s experiments/periodic_h0 -p test_mean_adaptive_confidence.py
```

The tests verify those arithmetic properties. The proof supplies the
conditional confidence statement. Neither establishes the source-law,
prospective-plan or row-certificate premises. The ideal infinite field's
exceptional-count moment, expected-count spectral-tail transfer and
quantitative lifetime-law remainder remain separate obligations. This
work adds no observed ensemble, held-out confirmation, formal kernel
certificate or organizationally independent review.
