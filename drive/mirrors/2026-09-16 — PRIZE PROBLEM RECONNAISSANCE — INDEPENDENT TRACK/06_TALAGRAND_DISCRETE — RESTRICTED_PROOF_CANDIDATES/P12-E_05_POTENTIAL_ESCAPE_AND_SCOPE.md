# P12-E — An exact limitation of this certificate, not of the conjecture

Status: elementary author-side counterexample. It refutes an attempted reuse of the constructed certificate, not the existence of a different compatible cover.

Take n=193, every coefficient a_i=3/5, every original selection probability p=10^(-8), and let D={U:sum a_i<1}. Its good sets contain at most one element. Thus D^(192) consists exactly of the full193-element set, and is NONEMPTY.

Let z_i=p_i (valid prices bounded by phi). Then Q=P(Bin(193,p)>=2)<=binom(193,2)p^2<2*10^(-12). In contrast, the fixed-R=24 prefix construction of P12-A has n_1=n because np<1/24. Its generator family contains EVERY singleton. After removing redundant supersets its cost is

\[
 np=193*10^{-8}.
\]
Since phi(Q)<=2Q<=2*binom(193,2)p^2,

\[
 np>phi(Q).
\]
The ratio to this upper bound on phi(Q) is exactly 1/((n-1)p), greater than520000. All comparisons are rational. This prefix family is nevertheless a perfectly valid inexpensive cover of D^(192).

There is a much better cover: the single full-set generator has cost p^193. Thus the example does not refute a universal compatible-potential theorem for weighted gates, even on this instance. It only proves that the certificate actually constructed in this pass cannot automatically be fed back into the prior induction.

This is the same kind of distinction that must be maintained throughout: a valid constant-smallness cover is not necessarily a failure-proportional/hazard-compatible cover.

## Why coefficient rounding/copying does not close the gap

Replacing a weighted coordinate by many independent equal-weight copies changes the law. For one coordinate copied twice into an AND, the actual reused-variable output has probability p, while the fake independent output has probability p^2. Cloning may preserve a deterministic weighted sum on a diagonal, but it does not supply independent child supports or a valid product generator-cost transfer.

## Current frontier after this pass

1. A self-contained no-dilution constant cover for an arbitrary finite positive-budget root, including unequal prices.
2. A common-palette extension when its children already have certified hazard-compatible covers.
3. No all-depth weighted-gate compatibility theorem.
4. No rank-independent theorem for arbitrary downsets.
5. No q0 scientific changes, no external review, no original prize closure.

The elementary prefix theorem has known ancestry. The value of this pass is a checked interface, explicit general constants, implementation and a guard against overstating composability, not a declaration of historical originality.
