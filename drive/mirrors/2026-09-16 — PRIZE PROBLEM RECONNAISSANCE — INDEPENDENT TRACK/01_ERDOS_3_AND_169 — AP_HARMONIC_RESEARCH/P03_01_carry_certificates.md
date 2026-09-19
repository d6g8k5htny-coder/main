# PR-AP-008 — Exact carry certificates for progression-free digital sets

2026-09-16. Complete author-side derivation plus exhaustive finite implementations. No external review. Automata-based arithmetic decision procedures are established methodology; historical novelty of the particular state bound has not been established. The proof below must not be read as a claim of a new unrestricted harmonic-sum record.

## 1. What changed from Phase02

The strong local modular rule of Phase02 is sufficient for AP-freeness but not necessary for a stationary digit set. This phase tests the actual INTEGER language, including carries. Fix b>=2, 0 in D subset {0,...,b-1}, and K=K(D,b), the nonnegative integers all of whose base-b digits lie in D.

A nontrivial k-AP is a list x_0,...,x_(k-1) with constant nonzero difference. Negative differences can be reversed. The second-difference equations are

    x_i - 2x_(i+1) + x_(i+2)=0,   0<=i<=k-3.

## 2. Second-difference carry machine

Read the k integers' digit columns e=(e_0,...,e_(k-1)) from least to most significant. Keep c in {-1,0,1}^(k-2) and a flag f indicating that some column has not been constant. Start at (0,false). A transition exists if

    c'_i=(c_i+e_i-2e_(i+1)+e_(i+2))/b

is integral for each i, with every e_j in D. Set the flag to f or [the column is nonconstant]. Given c,e_0,e_1, all later e_j are forced modulo b; hence each state requires at most |D|^2 candidate transitions.

The carry range is invariant: the numerator is between -(2b-1) and 2b-1. An integral quotient in that range is in {-1,0,1}.

After L columns, telescoping gives

    x_i-2x_(i+1)+x_(i+2) = b^L c_(i,L)-c_(i,0).

Thus a path from (0,false) to (0,true) gives exactly a nontrivial integer AP. Conversely, any integer AP has such a path after zero-padding all expansions to the same length. The flag distinguishes nonzero difference from the always-present constant progression.

**Theorem.** K is k-free if and only if (0,true) is unreachable.

**Short-witness consequence.** If a violating AP exists, one exists with at most 3^(k-2) digit columns and hence every term < b^(3^(k-2)). Before the flag becomes true the only reachable state is (0,false). Afterward there are at most 3^(k-2) carry states. A shortest successful path has no repeated state, giving at most that many edges. In particular, for k=4 at most NINE columns suffice, independent of b.

This is a proved finite reduction for stationary digit languages. It is not a finite-cutoff heuristic for arbitrary integer sets.

For a free language, a certificate lists the whole reachable-state set, excluding (0,true). A verifier checks every permitted outgoing transition remains in that set. For a nonfree language, the certificate contains the reconstructed explicit integer progression.

## 3. Independent positive-difference frame

A second implementation writes x_j=a+j*Delta and reads the digits of a and Delta. Carries satisfy c_j in {0,...,j} for j=1,...,k-1; the emitted digit is (a_digit+j*Delta_digit+c_j) mod b, and the next carry is the integer quotient. The start/end carry vector is zero, and a separate flag demands Delta>0.

This implementation uses a different frame, state space, transition generator, and (in C++) descending subset traversal. Agreement with the second-difference code is meaningful same-author cross-implementation evidence, not a new independent provider or human reviewer.

## 4. A specific published example is incorrect

Walker, arXiv:2203.06045v2, Remark 1.3, states that K({0,2,5},7) is 3-free. The fixed-v2 PDF and HTML were both checked; the PDF page was rendered for inspection.

The explicit counterexample is

    2   = (002)_7,
    133 = (250)_7,
    264 = (525)_7.

Every digit is 0,2,or5, and 133-2=264-133=131. Equivalently 2+264=2*133. The carry states are

    0 --(2,0,5)--> 1 --(0,5,2)--> -1 --(0,2,5)--> 0.

This refutes that illustrative sentence only. It does NOT refute Walker's modular sufficient condition, his base-11/base-55 constructions, or his approximation theorem. No contact with the author was made and no correction is represented as acknowledged externally.

Valid replacements showing the intended phenomenon include:

- b=11,D={0,3,8},k=3. The modular AP (3,0,8) exists, but the only reachable true-flag carry is +1; no complete integer AP exists.
- b=11,D={0,2,4,9},k=4. The modular AP (4,2,0,9) exists. Its true-flag reachable carries are (0,1) and (1,0), never (0,0).

Both have explicit reachability certificates in results/examples.json.

## 5. A new compositional trap

Stationary integer-language safety does NOT substitute for Phase02's strong local modular rule in an arbitrary adaptive tree.

Both K({0,2},4) and K({0,1},4) are 3-free. But use {0,2} at the units position and {0,1} at every higher position. The resulting tree contains

    0,2,4,6,

a nontrivial 4-AP. The translated positive set contains 1,3,5,7.

Therefore individually safe actions cannot be mixed using the old local Bellman proof without a NEW joint language certificate. The obstruction is arithmetic carries across interfaces, not numerical error. This is a blocked proof route, not a counterexample to Phase02, which had stronger hypotheses.

## 6. Finite-state generalization

For a complete LSD-first DFA with m states recognizing a zero-padding-invariant set, keep the four automaton states, two carries, and the nontriviality flag simultaneously. The resulting product machine has at most

    2 * 3^2 * m^4 = 18m^4

states. The transition reads a digit column in the four copies. A bad terminal has all four accepting states, zero carries, and a true flag. Exact reachability proves four-AP-freeness for ALL integer lengths, including carry-dependent languages. The code checks the zero-padding convention on every reachable DFA state before making that conclusion.

## 7. Complete stationary search at bases 2..30

The two exact C++ enumerators agree on:

    total integer-4-free alphabets containing zero: 2,569,644;
    those violating the stronger modular rule:       224,069;
    strong modular family:                        2,345,575.

Subset enumeration is complete by heredity: if a digit set already contains an integer progression, every superset contains it. Pruning only nonfree partial sets therefore discards no free alphabet.

The diagnostic reciprocal-score computation in the first implementation uses long double and a truncated series; it is NOT a certificate of the harmonic winner over the larger class. The exact counts/AP decisions do not depend on those scores. In particular, no unrestricted harmonic record or all-prime optimum is inferred.

## 8. Scope

The automaton decision algorithm is exact for its specified language. It does not decide Erdos #3 for arbitrary subsets of N. The adaptive mixing counterexample mandates a joint certificate rather than permitting silent reuse of the previous recurrence. The scan closes an AP-language classification, not the unrestricted optimization problem.
