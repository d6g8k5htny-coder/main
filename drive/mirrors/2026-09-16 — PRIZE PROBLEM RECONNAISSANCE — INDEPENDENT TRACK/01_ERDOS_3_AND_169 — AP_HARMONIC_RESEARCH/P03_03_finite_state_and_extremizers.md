# PR-AP-010 — Finite-state sparsity and the structure of weighted extremizers

Date: 2026-09-16. Complete author-side proofs, consuming classical van der Waerden and (only in Section 5) Bloom–Sisask. No external review. Novelty is unestablished; automatic-set sparsity and Behrend replacement are standard ingredients. Do not advertise an independent proof of either external theorem.

## 1. Automaton convention

Fix base b>=2 and a complete deterministic finite automaton with s>=1 states reading digits LEAST significant first. Acceptance is invariant under appending zero digits; hence the automaton recognizes a well-defined subset A of the nonnegative integers. When optimizing reciprocal sums, delete zero. A conventional most-significant-first automatic set can be translated to this convention by reversal/determinization and zero-padding normalization, with its own possibly larger state count. The numerical bounds here refer to the actual normalized complete automaton, not an unconverted presentation.

A state is live if some finite continuation is accepted. Dead states have no accepted continuation and cannot lead back to a live state.

## 2. Every reachable live state must reach a dead state

Assume A contains no nontrivial k-AP for some fixed k>=3. Suppose a reachable state q has no dead descendant. Take a fixed word u reaching q. Choose L such that b^L-1>=W(k,s), where W(k,s) is the classical finite van der Waerden number.

Color each integer x in [1,b^L-1] by the state reached from q after its exactly L base-b digits (zero-padded). There is a monochromatic increasing k-AP x_j of positive difference. All these words end at the same state q', which is live by assumption. Choose one continuation word v taking q' to an accepting state.

The words u, digits_L(x_j), v are accepted. Their integer values are

    value(u) + b^len(u)*x_j + b^(len(u)+L)*value(v).

This is an increasing k-AP of positive integers, contradiction. Thus every reachable live state has a path to a dead state. A shortest path has at most s edges because its states before first arrival at a dead state do not repeat.

This argument is combinatorial; it does not estimate any van der Waerden number or assume positive natural density.

## 3. Quantitative polynomial sparsity

Extend each short dead path to length s by zeros. Therefore, from every reachable live state, at least one of the b^s words of length s ends dead.

For L=ms+r, 0<=r<s, at most

    b^r (b^s-1)^m

length-L words can end live, hence at most that many are accepted. Define

    alpha=log(b^s-1)/(s log b)<1.

Then b^r(b^s-1)^m <=b^s b^(alpha L). Choosing L=floor(log_b N)+1, using unique padded representations of [0,N], gives

    |A intersect [0,N]| <=b^(s+1) N^alpha, N>=1.       (1)

If a rational exponent is desired, log(1-x)<=-x and log b<=b give

    alpha <= beta:=1-1/(s*b^(s+1))<1.                 (2)

The graph certificate `dfa_sparsity_certificate` checks each short escape word. The implication from AP-freeness to existence of those words is the written van der Waerden argument, not a numerical experiment.

### Automatic-set case of the divergent-reciprocal assertion

If an automatic positive-integer set A has divergent reciprocal sum, then it contains arithmetic progressions of every finite length. Otherwise it would be k-free for some k>=3, and (1) would make its reciprocal sum converge. This is a complete restricted-class consequence using classical van der Waerden, not a new solution of the full Erdős conjecture and not claimed historically novel. In particular, an unrestricted counterexample, if it exists, cannot be automatic in any fixed base.

## 4. No fixed finite-state AP-free set maximizes harmonic mass

For k>=3, (1) and PR-AP-009 imply:

**Corollary.** Every base-b automatic k-free set A has a finite k-free competitor F with H(F)>H(A). Thus no fixed finite automaton can describe a global harmonic maximizer, if one exists.

The same holds for weighted sums H_sigma whenever sigma>alpha; in particular for every sigma>=1. Here sigma is the weight exponent, while s above is the state count.

This is a limitation on finite-state WITNESS families, not on finite proofs, computer-assisted proofs, or sequences of automata with increasing state count. Every finite set is itself automatic; increasing the allowed complexity can approximate arbitrary finite witnesses.

## 5. Two unconditional existence consequences

### Summable weights sigma>1

For any k>=3, let X_k be the family of k-free subsets of the positive integers, identified with a closed subset of {0,1}^N. Closedness follows because every forbidden AP is finite. Product compactness applies.

For sigma>1, the functional H_sigma(A)=sum_(n>=1)1_A(n)n^(-sigma) is continuous on X_k, because its uniformly bounded tail is at most sum_(n>N)n^(-sigma)->0. Hence it attains a maximum.

No finite set is a maximizer: one can add a sufficiently large integer avoiding the finitely many AP completions of that finite set. Moreover PR-AP-009 rules out every count bound A(N)=O(N^alpha) with alpha<1. Therefore every maximizer is infinite, is nonautomatic in every fixed base, and has

    limsup_(N->infinity) log |A intersect [1,N]|/log N =1.

The limsup claim means absence of a uniform power-saving count bound. It does not assert positive density or an exact density asymptotic.

### Harmonic three-term problem sigma=1, k=3

Consume the established Bloom–Sisask theorem: there are absolute C,c>0 such that r_3(N)<=C*N/(log N)^(1+c) for large N. Each block A intersect [2^j,2^(j+1)) of a 3-free set has reciprocal mass O(j^(-1-c)), uniformly in A. Summing j>=J gives a uniform tail tending to zero.

Thus H_1 is also continuous on X_3 and attains its finite maximum. Combining with PR-AP-009, every such harmonic maximizer is nonautomatic and has upper counting dimension one. This is a COROLLARY using Bloom–Sisask, not a solution of the remaining k>=4 problem and not a new proof of the 2020 result. The value of the three-term maximum is not determined here.

For k>=4, PR-AP-012 now proves the CONDITIONAL statement that finiteness of the harmonic supremum implies a uniform tail theorem and actual attainment. That does not establish finiteness, so there is no unconditional existence assertion for those cases. Whenever a finite harmonic maximizer exists, the tail-replacement and automaton arguments here apply to it.

## 6. The critical-exponent picture — known inputs, explicit assembly

For 0<sigma<1, classical Behrend sphere blocks have H_sigma(B_t)>=N_t^(1-sigma)/(2t3^sigma2^(4t))->infinity, so the supremum of H_sigma over finite k-free sets is infinite for every k>=3.

For sigma>1 it is finite and attained, as above. At sigma=1, k=3 is controlled using Bloom–Sisask; the corresponding k>=4 harmonic question remains open here. The phase distinction uses standard constructions and convergence theory. It is not claimed as a novel resolution of any original prize problem.


## 7. An explicit positive gap for bounded automaton complexity

Fix B>=2 and s>=1. Restrict to k-free sets recognized by normalized complete automata with base 2<=b<=B and at most s states. The bounds above hold uniformly with

    C=B^(s+1), e=1/(s*B^(s+1)), beta=1-e.

Every such set has tail T(N)<=C/e*N^(-e). Set the explicit integer

    t=16*C/e=16*s*B^(2*s+2), N=2^(2*t*(t+1)).

The sphere replacement has new mass at least L=1/(6*t*2^(4*t)). The ratio L divided by the old tail cap is at least

    e/(6*C*t) * 2^(2*e*t*(t+1)-4*t).

Because e*t=16*C and C>=1, the exponent is >=28*t. Also 12*C*t/e=3*t^2/4, and 2^(28*t)>3*t^2/4 for every positive integer t (for example use 2^t>=t and 28t>2t). Thus the ratio exceeds2. Consequently every set in the bounded-complexity class admits a k-free finite replacement with gain at least

    Delta(B,s)=1/(12*t*2^(4*t))>0.                  (3)

The unrestricted harmonic supremum therefore exceeds the bounded-complexity supremum by at least Delta(B,s). This remains a statement in the extended reals if the unrestricted supremum is infinite. There are finitely many automata in the specified class, so its own finite supremum is attained.

Equation (3) is intentionally extremely weak. It is not a practical harmonic-record construction. It gives a mathematically explicit reason that arbitrarily accurate approximation to a finite unrestricted optimum must escape every fixed base/state budget. The number t and exponent 4t may be stored symbolically: the package does not enumerate the giant replacement or expand arbitrarily huge denominators.
