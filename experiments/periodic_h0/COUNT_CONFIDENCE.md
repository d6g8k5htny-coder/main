# Fixed-sample confidence for finite-cutoff expected counts

[Finite-cutoff expectation transfer](FINITE_COUNT_LOSS.md) ·
[Explicit-word polynomial bridge](WORD_POLYNOMIAL.md) ·
[Sampler and its input-law premise](GAUSSIAN_COUPLING.md) ·
[Exact confidence arithmetic](count_confidence.py)

The finite-polynomial bar cap also bounds sampling error. This note gives
simultaneous confidence intervals for unconditional expected bin counts of
the same-cutoff ideal Gaussian polynomial, provided a fixed number of word
blocks have the declared IID law and every retained row has valid count
bounds. Grid errors may vary between rows. Grid refinement and the choice
to leave a computation unresolved may depend on the sample: concentration
is applied to fixed latent polynomial counts, not to the computed grid
observables.

This is a prospective mathematical statement. The arithmetic implementation
does not authenticate a word source, a sampling plan or a grid certificate.
No observed IID or Gaussian sample, evaluated ensemble mean, infinite-field
expected count or lifetime-law confirmation is supplied by this note.

## 1. Fixed observables and the sampling assumptions

Fix an integer 1<=K<=64 and write C_K=16 K^2. Use the polynomial P_J,
normalization S_alpha,64 and word order of the unchanged sampler. Let rho
and p be its same-cutoff coefficient-error and clipping-probability upper
bounds. In particular they are uniform bounds from that construction, not
estimates fitted to a sample. The target is G=F64,K, not the infinite field.

Before inspecting the sample, fix a positive integer n, a finite family of
m>=1 bins [a_j,b_j) with 0<a_j<b_j, and a common radius r>0. Overlapping bins
are allowed. The exact implementation uses rational endpoints, budgets and
radius. Assume J_1,...,J_n are independent blocks, each consisting of
(2K+1)^2 independent uniform 128-bit words. This law includes independence
both within a block and between blocks. It remains an explicit hypothesis;
neither a seed nor a list of integer words proves it. Repeated word values
are possible under this law and are not a reason to discard a draw.

Define the following fixed functions on the entire word space:

```text
X_j^-(J) = N_PJ([a_j+2 rho, b_j-2 rho)),
X_j^+(J) = N_PJ([a_j-2 rho, b_j+2 rho))  if a_j > 2 rho,
           C_K                          otherwise.
mu_j^- = E X_j^-(J),       mu_j^+ = E X_j^+(J),
nu_j   = E N_G([a_j,b_j)).
```

An empty or reversed contracted interval has count zero. Only strictly
positive finite ordinary H0 lifetimes are counted; the essential class
remains separate. The measurability and deterministic cap proved in
[FINITE_COUNT_LOSS.md](FINITE_COUNT_LOSS.md) give

```text
0 <= X_j^-(J) <= X_j^+(J) <= C_K.
max(0, mu_j^- - C_K p) <= nu_j <= min(C_K, mu_j^+ + C_K p).
```

Every expectation here is over its entire declared law. There is no
conditioning on clipping success, on a chosen word list or on successful
grid computation. The latent variables X_j^-(J_i) and X_j^+(J_i) are IID
across i for each fixed j. Different bins and the two signs can be dependent.

## 2. What a retained row must establish

For each i and j, require actual bounds

```text
0 <= L_ij <= X_j^-(J_i) <= X_j^+(J_i) <= U_ij <= C_K.
```

A separately certified grid diagram Q_i with
d_B(D(P_Ji),Q_i)<=e_i supplies such bounds. Set d_i=2(e_i+rho) and use

```text
L_ij = N_Qi([a_j+d_i,b_j-d_i)),
U_ij = min(C_K, N_Qi([a_j-d_i,b_j+d_i)))  if a_j > d_i,
       C_K                               otherwise.
```

The contracted grid count is zero for an empty interval. To see the lower
inequality for every word, apply the grid-to-polynomial matching argument
to the fixed polynomial interval [a_j+2 rho,b_j-2 rho). Its grid contraction
is exactly the interval defining L_ij. If that polynomial interval is empty,
so is the grid contraction. For the upper inequality, when a_j>d_i the
polynomial interval [a_j-2 rho,b_j+2 rho) starts strictly above 2 e_i, so its
expanded grid count bounds X_j^+. Both that count and C_K bound X_j^+;
their minimum therefore does too. In the remaining cases the cap suffices.
The strict test is necessary, including at equality, because a bar at the
diagonal threshold can disappear in a matching.

These statements are deterministic and do not require the Gaussian clipping
event. They reuse the existing bin-transfer theorem and
`finite_count_loss.grid_observables`; no probability is assigned to a fixed
word fixture by this reasoning.

If a grid calculation fails or has no valid certificate, retain the draw
with L_ij=0 and U_ij=C_K. All n draws remain in every average. The grid size,
refinement, error bound and choice to leave a computation unresolved can
depend on all the observed words or calculations, provided each final row
still satisfies the pointwise inequalities. Such choices only replace the
unknown latent sample averages by conservative bounds. They do not require
the reported grid rows to be IID.

The fixed n, bin family and radius remain part of the sampling premise.
Choosing them after seeing the sample, stopping when an interval looks
favorable, or discarding unresolved draws is not justified by this theorem.
An unresolved row is a retained draw with missing computational evidence;
it is not evidence that an unrecorded or never-generated draw occurred.

## 3. Simultaneous confidence theorem

Let Lbar_j and Ubar_j be the sums of all n respective row bounds divided
by n, and put

```text
R = min(1, 2 m exp(-2 n (r/C_K)^2)).
```

Under the assumptions above, with probability at least 1-R, simultaneously
for every j,

```text
mu_j^- >= max(0, Lbar_j-r),
mu_j^+ <= min(C_K, Ubar_j+r),

max(0, Lbar_j-r-C_K p) <= nu_j
                       <= min(C_K, Ubar_j+r+C_K p).
```

Thus the random intervals cover fixed unconditional means. They are not
probability intervals for the count of the next realized field. No
independence between bins, between the two observables, or between clipping
and counts is assumed.

**Bounded-variable concentration.** For a variable X taking values in
[0,C_K], define h(t)=log E exp(t(X-E X)). Its distribution here has finite
support, so differentiation is legitimate for every real t. Under the
exponentially tilted probability law, its second derivative is

```text
h''(t) = Var_t(X)
       = E_t[(X-C_K/2)^2] - (E_t[X]-C_K/2)^2
       <= C_K^2/4.
```

Since h(0)=h'(0)=0, the integral form of Taylor's formula gives
h(t)=t^2 integral_0^1 (1-u)h''(ut)du <= t^2 C_K^2/8. For n independent
copies, multiplication of moment-generating functions and exponential
Markov therefore give, for t>0,

```text
Pr(Xbar-E X > r) <= exp(-t n r + n t^2 C_K^2/8).
```

Taking t=4r/C_K^2 proves an upper bound exp(-2n r^2/C_K^2).
Using h(-t) gives the same bound for Pr(E X-Xbar>r).

Apply the upper tail to X_j^- and the lower tail to X_j^+. A union bound
over 2m events gives R. Outside their union,

```text
mu_j^- >= Xbar_j^- - r >= Lbar_j-r,
mu_j^+ <= Xbar_j^+ + r <= Ubar_j+r.
```

This step explains why adaptively refined or unresolved grid rows are
permitted: only the fixed latent polynomial variables enter concentration.
Insert these inequalities into the unchanged finite-cutoff expectation
transfer to obtain the claimed intervals for nu_j. The clipping correction
is C_K p, not n C_K p. It bounds an expectation difference; this argument
does not demand clipping success for every sampled coupling.

This theorem bounds mu_j^- and mu_j^+. It does not assert that an adaptively
produced grid average estimates a fixed mean E L_grid or E U_grid. Such a
claim instead needs a fixed total per-word evaluation rule defining those
random variables.

## 4. Exact arithmetic and its evidence boundary

`count_confidence.risk_upper(K,n,m,r)` evaluates a rational upper bound for
R using `gaussian_tail.exp_neg_bounds`. If x=2n(r/C_K)^2 exceeds the inherited
evaluator's domain, the function evaluates at min(x,4096). This is safe
because exp(-x)<=exp(-min(x,4096)); it can be conservative. The returned
upper exponential endpoint is multiplied by 2m and the result capped at
one. No floating logarithm, square root or exponential is needed. A desired
failure level is established only when the resulting exact rational upper
bound is no greater than that level.

`count_confidence.mean_interval(K,p,rows,r,n)` uses exactly n rows for one
bin. Each integer pair must satisfy 0<=L<=U<=C_K. A `None` row contributes
[0,C_K]. The function computes rational averages, adds the sampling radius
and passes the resulting mean bounds to the existing expectation-transfer
arithmetic. For a simultaneous claim, use the complete fixed bin family
and its actual cardinality m with the same n and r. Although the generic
cap arithmetic accepts positive integer cutoffs, the sampler-based theorem
in this note is restricted to 1<=K<=64 with its corresponding budgets.

Type checks, row counts, order, sums, exponential enclosures and interval
arithmetic are executable checks. They do not establish that a supplied
pair brackets the latent polynomial counts. That requires the actual
word-to-polynomial and grid evidence, or another valid argument. Likewise,
the code does not establish independence, uniformity, prospective choice
of the sampling plan, or that p belongs to the stated sampler and cutoff.
Those are external premises, not properties inferred from plausible
numbers or a successful arithmetic replay.

An illustrative row list must therefore remain labeled as deterministic
arithmetic input. It supplies no observed ensemble mean and is not a
collection of authenticated grid certificates. Historical PCG64 fields and
the earlier selected word fixture acquire no new law or confidence claim.

## 5. Planning scale and exact negative controls

For eight bins, n=40000 and r=C_K/100 give exponent exactly 8. The exact
exponential evaluator bounds the family failure probability from above by
a rational value whose upward decimal rendering is

```text
0.005367402046440190.
```

This is below 0.01 under the theorem's premises. The corresponding absolute
count radii and the sample sizes for radius one at that same bound are:

| Cutoff K | Cap C_K | Radius C_K/100 at n=40000 | n=4 C_K^2 for radius one |
|---|---:|---:|---:|
| 1 | 16 | 4/25 = 0.16 | 1,024 |
| 24 | 9,216 | 2304/25 = 92.16 | 339,738,624 |
| 32 | 16,384 | 4096/25 = 163.84 | 1,073,741,824 |

These are prospective arithmetic examples, not consumed sample sizes or a
held-out lifetime-law design. The conservative worst-case cap makes useful
absolute precision expensive at larger cutoffs. The interval width also
includes the gap between the lower and upper grid averages and the clipping
correction; the sampling radius alone is not the total uncertainty.

The following finite-law examples check the statistical arithmetic without
claiming to be polynomial or Gaussian data:

- Let X be 0 or 16 with equal probability, use n=8 and r=4, and supply
  exact rows [X,X]. Exhaustive rational summation gives
  Pr(|Xbar-8|>4)=9/128. The correct bound is 2 exp(-1); omitting C_K^2
  would give 2 exp(-256), smaller than the actual failure probability.
- With the same law and plan, certify zero outcomes as [0,0] and retain
  positive outcomes as unresolved [0,16]. The exact probability that the
  upper confidence endpoint misses the mean 8 is 9/256. Dropping all the
  unresolved outcomes and treating the remaining zero rows as the sample
  instead misses on 255/256 of all sequences; the last sequence has no
  retained row. This detects conditioning on success and a wrong denominator.
- Replace selected exact rows by [0,16], making the selection depend on
  the whole sequence. Every resulting interval contains its exact-row
  interval. Enumeration checks the permitted adaptive widening directly.
- For four bins and n=1, choose uniformly which one bin has count 16;
  the other three have count zero. Each bin mean is 4, and the total count
  is always 16. With r=10, the selected bin has lower endpoint 6, so
  simultaneous failure has probability one. Omitting the family factor
  would give 2 exp(-25/32), whose upward decimal bound is
  0.915666723543228522, less than one. The correct family bound is capped
  at one.

These examples supplement strict-endpoint, empty-contraction and unresolved
grid controls in the existing finite-count tests. A replay must also reject
altered n, family size, means, risk or scope even when output hashes are
rebound. Tests establish these arithmetic properties; they do not certify
the IID input premise or prove a supplied grid matching.

## 6. Remaining obligations

The new result supplies a sampling-error theorem for the finite-cutoff
expectation transfer. It evaluates no random sample mean. Actual inference
still requires the specified input law, the fixed sampling plan, all
retained draws and valid row evidence. A small-support arithmetic toy is
not the full uniform-word ensemble.

The ideal infinite field has no fixed-cutoff cap C_K. Its exceptional-event
count moment, the expected-count spectral-tail step and a numerical
lifetime-law remainder remain separate. This result provides no held-out
confirmation, physical source authentication, formal proof certificate or
independent-human review. Authorship, substantive nonauthor review, formal
evidence and scientific acceptance retain their distinct meanings.
