# PR-AP-005 — Explicit infinite witnesses and the literature boundary

2026-09-16. Complete author-side derivations and exact numerical certificates; no external review or novelty certification. No prize problem is solved.

## 1. A simple strict improvement on the base-11 benchmark

Let C be the nonnegative integers with all base-11 digits in D11=(0,1,2,4,5,7). Let E=(0,1,2,4,5,7,8,9,14,17) and G={0,1,2,4}. Define

    T_mix = union_(d in G) {d+11(e+22n): e in E, n in C}
            union union_(d in D11\G) {d+11n: n in C},
    A_mix=1+T_mix.

All unions are disjoint at their stated digit level. Every local alphabet is modular 4-AP-free, including the repeated-residue cases for composite radix 22. PR-AP-003's infinite-descent argument therefore proves that A_mix is integer 4-AP-free.

If S(a)=sum_(n in C)1/(n+a), its exact reciprocal sum is

    sum_(d in G) (1/(11*22)) sum_(e in E) S((e+(d+1)/11)/22)
    + sum_(d in D11\G) (1/11) S((d+1)/11).

All shifts lie in (0,1]. Apply the inherited rational moment recurrence with degree 40 and its complete remainder separately to every term. The result is

    4422264769900097669/10^18 <= H(A_mix)
                                <= 4422264769900097670/10^18.

Relative to H(1+C), the strict gain lies in

    [517237501865143,517237501865144]/10^18.

This is a proof about the infinite set, not a finite truncation. The positivity of the improvement is established by nonoverlapping rational intervals. The code also checks a finite prefix as a secondary diagnostic, not as the progression-free proof.

## 2. An explicit near optimizer for the entire radix-2-to-30 class

The file results/EXPLICIT_POLICY.json specifies a mesh of 1,000,000, depth 64, and the action policy of each finite-horizon lower Bellman iterate. There are 441 run-length encoded action regions across all 64 layers. Its root state is j=1,000,000. For positive a=n+1, test n through the stored radix policy; after accepting digit d in base b, replace the state by ceil((d*m+j)/b) and the integer by its quotient. After the prescribed depth, use the canonical base-11 alphabet forever.

The integer recurrence underestimates the sum of this actual infinite set: rounded-up shifts decrease each subtree sum, immediate rewards are floored, and the continuation beyond the finite horizon only adds nonnegative mass. Consequently its sum is >=4.422891010185. The full-class upper certificate is <=4.422891978614. This explicit policy is within 10^-6 of the class optimum.

No decimal approximation to the exact switching threshold is used to define membership. Membership of each integer terminates and is exact.

## 3. Important priority correction

Alexander Walker's paper first posted in 2022, arXiv:2203.06045, supplies a stronger stationary base-55 construction. Its digit alphabet is

    (0,1,2,4,5,9,10,11,14,16,17,18,21,24,30,37,39,41,42,45,47).

Our generic rational summation companion reproduces its sum in

    [4439753369254540648,4439753369254540649]/10^18.

Therefore neither new radix-11/22 construction is a record lower bound on the unrestricted four-progression-free harmonic supremum. The proper result is the complete optimization of a stated adaptive class, not an improvement on the best example in that paper. The base-22 alphabet is also in Walker's table and is NOT newly discovered here.

Walker's Theorem 2.1 also already supplies the substance of the digital approximation/reformulation used in Phase01. That earlier derivation must be credited as a reconstruction of known work, not a new route that removes the conjecture's difficulty. This correction leaves its mathematical identity intact.

The priority of our adaptive optimization and threshold results is unestablished. A source search is not a novelty certificate.
