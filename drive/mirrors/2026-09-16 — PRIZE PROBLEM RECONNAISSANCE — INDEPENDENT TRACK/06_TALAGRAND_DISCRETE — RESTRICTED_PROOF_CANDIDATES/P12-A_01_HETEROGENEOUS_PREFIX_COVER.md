# P12-A — Exact prefix cover at heterogeneous nonnegative prices

Status: self-contained author-side lemma. This is an adaptation of the pre-existing Talagrand singleton cover argument (Frankston–Kahn–Park, arXiv:2105.10905v1, Proposition 2.1), not a claim of new singleton rounding. Correctness and historical novelty are separate; external review remains absent.

Let X={1,...,n}; let a_i>=0 be real coefficients and z_i>=0 nonnegative **generator prices**, not necessarily the original selection probabilities. Sort coordinates so a_1>=...>=a_n, and write P_j=sum_{i<=j}z_i and M=sum_i a_i z_i. For a real R>0, define

\[
 n_k=\max\{j\in\{0,...,n\}:P_j\le k/R\},\quad 1\le k\le n,
 \qquad \mathcal G_R=\bigcup_{k=1}^n { [n_k]\choose k}.
\]
An impossible k-subset contributes nothing. Zero prices are allowed. Then

\[
 \boxed{\{U:a(U)>RM\}\subseteq\langle\mathcal G_R\rangle.} \tag{A1}
\]
For R>e the price cost obeys

\[
 \boxed{\operatorname{cost}_{\mathbf z}(\mathcal G_R)
 \le\sum_{k=1}^n e_k(z_1,...,z_{n_k})
 \le\sum_{k=1}^n\frac{(k/R)^k}{k!}
 <\frac{e}{R-e}.} \tag{A2}
\]
Repeated generator descriptions can be removed; that only lowers cost. For rational R, coefficients and prices, every prefix and elementary-symmetric cost is computable with exact rational arithmetic. The proof does not require evaluating e.

## Proof of coverage

Let U={u_1<...<u_m} avoid the cover. For each j<=m, u_j>n_j, hence P_{u_j}>j/R. On the price-mass interval [P_{i-1},P_i), define f(t)=a_i; zero-length intervals disappear. The function is nonincreasing and has integral M. For almost every t< P_{u_j}, f(t)>=a_{u_j}. In particular this holds on [(j-1)/R,j/R), because P_{u_j}>j/R. These m intervals are disjoint. Therefore

\[
 M\ge\frac1R\sum_{j=1}^m a_{u_j}=a(U)/R.
\]
Thus a(U)>RM forces coverage. This proves (A1), including prices zero and M=0; empty U cannot violate its nonnegative threshold.

## Proof of cost

The kth prefix has total price <=k/R. Expansion of its total price to the kth power counts every distinct k-product k! times and adds nonnegative repeated-index terms. Thus e_k<=P_{n_k}^k/k!. The elementary factorial bound k!>=(k/e)^k gives the geometric-series upper bound. Truncation and e<R yield (A2). The strict bound for finite n also follows from a finite geometric sum.

With R=24, using e<3,

\[
 \operatorname{cost}<\frac17. \tag{A3}
\]
A symbolic generator representation is enough; polynomial-size expansion is NOT promised. This is an integral generator cover, not merely a fractional cover or a small probability estimate.

## Important boundaries

The ordering is by a_i, not by price and not ascending. The cutoff uses cumulative price, not j times a common probability. The threshold is **strictly greater than** RM. A certificate may intentionally overcover. In particular its cost need not be bounded by the original event probability; P12-E gives a counterexample to that inference.
