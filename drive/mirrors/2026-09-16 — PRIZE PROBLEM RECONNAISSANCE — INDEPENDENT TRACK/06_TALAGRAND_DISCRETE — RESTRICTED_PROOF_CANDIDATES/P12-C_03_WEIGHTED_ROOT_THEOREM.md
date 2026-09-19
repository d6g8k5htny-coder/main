# P12-C — Arbitrary positive-budget gates at a terminal root

Status: complete self-contained author-side argument, external review0, novelty UNESTABLISHED. The covering mechanism is credited to Talagrand's prior singleton rounding argument; the heterogeneous-price formulation and composition constants require their own priority comparison. This is NOT a universal hazard-compatible primitive theorem.

Let X be finite. Let d>=1 and let a_{ji}>=0 be arbitrary finite real coefficients. Define the decreasing family

\[
 \mathcal A=\{U\subseteq X:\ \sum_{i\in U}a_{ji}<1\text{ for all }1\le j\le d\}.
\]
Any positive capacity can be normalized to one. Let X_p have independent, possibly unequal coordinate probabilities p_i in [0,1]. Define phi(t)=min(1,-log(1-t)), phi(1)=1. Let nonnegative prices c_i satisfy c_i<=phi(p_i).

**Theorem.** If mu_p(A)>=3/4, then for every integer K>=192d there is an explicit generator cover G of A^(K) satisfying

\[
 \boxed{\sum_{g\in G}\prod_{i\in g}c_i<\frac{31}{70}<\frac12.} \tag{C1}
\]
No bound is imposed on the coefficient ratios, number of variables, witness rank, or crossing/incidence geometry. The palette depends linearly on the number d of resource inequalities, not on their witness rank. The theorem concerns a terminal gate/root, NOT arbitrary repetitions of this gate.

## Proof

Let J={i:max_j a_{ji}>=1}. Any selected i in J alone violates the constraints. The good-event premise therefore implies product_{i in J}(1-p_i)>=3/4. Since c_i<=phi(p_i)<=-log(1-p_i),

\[
 \sum_{i\in J}c_i\le-\log\prod_{i\in J}(1-p_i)
 \le\log(4/3)<3/10. \tag{C2}
\]
No p_i=1 can occur in J under the premise. The final strict bound follows from a finite Taylor lower bound e^(3/10)>4/3.

Outside J set a_i=max_j a_{ji}<1 and M_c=sum_i a_i c_i. The elementary inequality phi(p_i)<=2p_i and P12-B give

\[
 M_c\le2\sum_i a_i p_i<4d. \tag{C3}
\]
If U avoids J but is not a union of K members of A, it is not partitionable into K scalar bins of max-coefficient load<1. Hence

\[
 a(U)>K/2\ge96d>24M_c.
\]
Apply P12-A at prices c and R=24. Its integral prefix cover has cost<1/7 and covers every such U. Together with singleton generators for J, the total is

\[
 \operatorname{cost}<3/10+1/7=31/70.
\]
This proves C1. The strict/weak thresholds used above match the original strict good capacity.

## Refinements and consequences

1. **Single arbitrary weighted constraint:** d=1 gives K=192. Arbitrarily small or large coefficients, irrational coefficients in the mathematical theorem, and arbitrary product biases are admitted.
2. **No singleton-forbidden inputs:** when J is empty, the cost is <1/7.
3. **Instance-specific effective load:** replace 192d by any K satisfying K/2>=24M_c. The same cover works at cost<1/7 outside J. This may be much smaller than the dimension bound. A verified load is required, not a heuristic estimate.
4. **Uniform probability:** at c_i=p, C1 is an ordinary p-small conclusion with NO dilution.
5. **Transformed prices:** the stronger c_i<=phi(p_i) version is what permits P12-D's disjoint subtree application. A claim merely at c_i=p_i would not suffice there.
6. **Unrestricted original conjecture:** d is not universally bounded for arbitrary downsets. Writing one inequality per minimal forbidden set can make d enormous and does not eliminate this dependence.

## Prior-work boundary

Talagrand's positive-linear-weight rounding is pre-existing: Frankston–Kahn–Park, arXiv:2105.10905v1, Proposition2.1 and its sorted-prefix proof. Our proof uses a heterogeneous price-mass version of that argument plus elementary packing and moments. We do not assert that constant-smallness for positive linear constraints was previously unknown. The exact explicit no-dilution/transformed-price formulation and the terminal composition interface are the statements being documented, with novelty pending.
