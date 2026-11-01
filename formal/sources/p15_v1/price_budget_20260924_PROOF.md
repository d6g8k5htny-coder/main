# Restoring the transformed-price budget under explicit hypotheses

**Object:** P15-TRANSFORMED-PRICE-BUDGET-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side derivation; nonauthor review open.

[Research home](https://github.com/d6g8k5htny-coder/main) · [Math index](../../README.md) · [Prior construction](../three_fronts_20260924/P15_REALIZED_COVERS.md) · [Unchanged counterexample](../three_fronts_20260924/P15_PRICE_BOUNDARY.md)

## 1. What changes and what does not

The exact original-coordinate construction in `P15_REALIZED_COVERS.md`, SHA256
`c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`, proves a
cover theorem for prices c_v<=p_v. The separate two-coordinate counterexample,
SHA256 `498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b`,
refutes extending the SAME palette to every c_v<=phi(p_v) with no additional
hypotheses. Both sources remain unchanged. Here phi(p)=min(1,-log(1-p)), with
phi(1)=1. This note supplies additional sufficient hypotheses; it does not retract
that counterexample, accept an unrestricted prize, or claim new classical coloring theory.

Use the prior construction exactly. H is a clutter of block supports, with every
edge of size at least two; the original disjoint blocks satisfy |X_i|=a_i*d_i+1.
The downset D consists of sets using at most a_i vertices of each block and whose
occupied-block support contains no H-edge. The complete crossing witnesses are
ALL H-edge transversals. K_H(d) is the integer demand-palette optimum. The specified
cover consists of the full blocks X_i; an obstructed set contains a generator.
A generator g costs product_(v in g)c_v, the empty generator costs one, and an
empty family costs zero. No random coordinates are copied.

**Theorem (restricted transformed-price extension).** Suppose every a_i>=1,
every d_i>=2, and all independent original probabilities satisfy 0<=p_v<=1/4.
Then for EVERY set of prices 0<=c_v<=phi(p_v), at the SAME K>=K_H(d),

    covercost_c(O_K(D)) <= min(1,(16/27)*[-log mu_p(D)]).       (T1)

The uncapped full-block union has cost at most the expression inside the second
argument of this minimum. Cap with the empty generator when needed. The palette
and the original-coordinate support family are not enlarged. The constants in
this theorem are explicit; unlike the Gaussian lifetime remainder, there is no
unevaluated radius or error constant here.

## 2. Local budget proof

For 0<=p<1, elementary integration gives

    -log(1-p)=integral_0^p dt/(1-t) <= p/(1-p).               (T2)

Consider one block with capacity a>=1, demand d>=2 and n=ad+1. Its local failure
probability is q=P(number of selected original coordinates > a). If any p_v=0,
the full-block generator price is zero and the desired inequality is immediate;
the zero-price generator remains in the cover for setwise completeness.

Otherwise choose any subset S of a+1 original coordinates. The event that all
members of S are selected is a local failure, regardless of the other coordinates.
By independence, q>=product_(v in S)p_v. Using (T2), p_v<=1/4, and n>=2a+1,

    product_(v in X)c_v / product_(v in S)p_v
       <= product_(v notin S)p_v / product_(v in X)(1-p_v)
       <= (1/4)^(n-a-1)/(3/4)^n
       = (4^(a+1))/(3^n)
       <= (4/3)*(4/9)^a <=16/27.                          (T3)

Consequently the full-block generator satisfies

    price({X}) <= (16/27)q <= phi(q).                     (T4)

This proof does not require equal probabilities. No independence of different
crossing-witness events is used. The extra demand hypothesis d>=2 is load-bearing:
when a=d=1, n=2, the ratio calculation does not supply (T3), and the earlier
counterexample applies. The bound p_v<=1/4 is a sufficient range, not an asserted
sharp threshold.

A more general sufficient ratio, for any common upper bound p_*<1, is

    rho(a,d,p_*)=p_*^(ad-a)/(1-p_*)^(ad+1).                (T5)

If rho<=1 the same argument proves price({X})<=q for that block. This is only a
sufficient test; rho>1 is not a counterexample and must not be labeled infeasible.

## 3. Global budget and unchanged palette

Write q_i=P(U intersect X_i notin D_i) for the ACTUAL capacity restriction.
Disjoint original blocks make their local-good events independent. Global goodness
implies all local restrictions are good, even though crossing constraints may
create further failures. Therefore

    mu_p(D)<=product_i(1-q_i),
    sum_i q_i<=-log(product_i(1-q_i))<=-log mu_p(D).       (T6)

Sum (T4), then use (T6). The exact setwise argument of the prior realization
shows that the union of full-block generators covers O_K(D) for K>=K_H(d),
independently of prices. This proves (T1). If mu_p(D)=0 the price-one cap applies;
if mu_p(D)=1 the union price is zero by the same estimates. This is the same
local-cover/budget interface as original P15-B, source SHA256
`9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3`.

## 4. Exact original-coordinate consequence for the 816-label example

Use the prior six blocks of 409 original coordinates, a_i=1, d_i=408, and all
15 four-block forbidden supports. At p_v=1/40900, every hypothesis above holds.
Thus the SAME 816-label obstruction cover is valid for the FULL transformed range

    0<=c_v<=log(40900/40899),

not merely c_v<=1/40900. At maximal equal prices its nonzero union price is

    6*log(40900/40899)^409 <=6/(40899^409).                (T7)

An explicit rational price strictly larger than p is

    c=p+p^2/2=81801/3345620000.

It is admissible since integral_0^p 1/(1-t)dt > integral_0^p(1+t)dt for p>0.
Its cover price is exactly 6*c^409. The palette remains 816; the entire ground
set still requires 818 colors, and there are 2454 original coordinates. No large
witness enumeration or new claim about arbitrary old P15 instances is involved.

## 5. A useful exact checker beyond the simple uniform hypothesis

The supplied standard-library program computes q exactly for rational inputs via
a truncated Poisson-binomial recursion. If b_j denotes P(exactly j successes) for
j<=a, one new coordinate updates b_j to (1-p)b_j+p b_(j-1); discarded higher
counts never return. Hence q=1-sum_(j=0)^a b_j after all coordinates.

It bounds every transformed generator price by

    U=product_v min(1,p_v/(1-p_v)),

with the p_v=1 factor defined as one. It certifies a sufficient local budget if
U<=q, or if q>=2/3 (then phi(q)=1, since log3>1). The low-probability theorem
additionally verifies U<=(16/27)q. It rejects inexact inputs and explicitly limits
problem size; its arithmetic cost is O(n*a) rational operations, not a polynomial
bit-complexity or arbitrary-hypergraph optimization claim.

A non-certified output means only these sufficient tests did not prove the budget.
The program is not a full cover optimizer, a logarithmic interval decision procedure,
a test of global scientific status, or an independent proof reviewer. The earlier
two-coordinate counterexample is correctly not certified. The proof of log3>1
is 2*integral_0^(1/2)1/(1-t^2)dt>1, so the cap test is also exact.

## 6. Review and attribution

Review (T3)'s ratio exponents, the demand and probability conditions, (T6)'s
inequality direction, zero-price completeness, and the distinction between the
specified cover threshold and whole-ground colorability. Finite tests compare
the local recursion with subset enumeration and preserve the earlier counterexample.
They do not establish independent analytic acceptance.

Primary-source reconnaissance on 24 September 2026 inspected Park–Pham,
*A Proof of the Kahn–Kalai Conjecture*, arXiv:2203.17207 abstract, and
Gunby–He–Narayanan, *Down-set thresholds*, arXiv:2112.08525 abstract.
These are neighboring threshold results, not inputs to (T1) and not evidence of
novelty. The proof here uses the named in-project sources and the explicitly derived
elementary inequalities. No theorem from those abstracts is imported.
