# PR-AP-011 — Exact infinite reciprocal-sum enclosure for finite-state languages

Date 2026-09-16. Author-side proof and implementation. This extends the campaign's exact summation tool to arbitrary fixed-base finite-state digit languages. Summing digit-pattern-restricted reciprocal series is established prior work; novelty is not claimed. AP-freeness must be supplied separately by PR-AP-008 or another proof.

## 1. Set and zero-padding semantics

Use a complete LSD-first base-b DFA, with acceptance invariant under appending zero at every reachable state. Let T_q be the set of nonnegative integers accepted from state q and epsilon_q=1_{0 in T_q}. For 0<a<=1 write

    S_q(a)=epsilon_q/a+B_q(a),
    B_q(a)=sum_(n>0,n in T_q)1/(n+a).

Zero padding implies epsilon_q=epsilon_delta(q,0). Disjoint root residue classes therefore give the exact identity

    B_q(a)=h_q(a)+(1/b)sum_(d=0)^(b-1) B_delta(q,d)((d+a)/b),
    h_q(a)=sum_(d=1)^(b-1) epsilon_delta(q,d)/(d+a).       (1)

A state need not accept zero. Omitting epsilon from h_q would incorrectly count one-digit integers. The d=0 pole cancels only because zero-padding invariance is checked.

## 2. A rational supersolution includes the ENTIRE infinite tail

Restrict to reachable live states (states with some accepting continuation), and omit dead states, whose B is zero. Let

    P_qr = number_of_digits_d_with_delta(q,d)=r / b.

Assume every reachable live state reaches a dead state. With s states, P^s has row sums at most 1-b^(-s), so the Neumann series converges. In particular

    w=(I-P)^(-1)1

exists and has rational entries at least one. Let H=sum_(d=1)^(b-1)1/d. Then h_q(a)<=H and

    h_q(a)+(P(Hw))_q <=H+H(w_q-1)=H w_q.

Thus zero is a lower bound and the constant vector H w is an upper bound for all B_q on [0,1]. Finite-digit expansions converge monotonically under this bound. The s-step contraction identifies the bounded solution of (1) with the actual infinite sums, not an unspecified formal solution.

Every fixed-state AP-free language has the required escape property by PR-AP-010. The summation algorithm itself requires the graph property, not AP-freeness; it is also valid for other transient digit languages.

## 3. Integer grid enclosure, with no omitted interpolation error

For mesh M, integer scale Q, and a=j/M, child shifts are (d+j/M)/b. Since every B_q is nonincreasing,

    B_q(ceil((dM+j)/b)/M)
      <= B_q((d+j/M)/b)
      <= B_q(floor((dM+j)/b)/M).

Initialize lo=0 and hi_q=ceil(Q H w_q). At each update, sum individually floored/ceiled exact reward terms Q*M/(dM+j) with epsilon=1, and lower/upper child values at the directed neighboring grid points, dividing their sum by b with downward/upward integer rounding.

Induction gives lo_q[j]/Q<=B_q(j/M)<=hi_q[j]/Q at EVERY iteration. The radius of the enclosing interval, rather than an empirical convergence criterion, controls what may be stated. Explicit integer overflow guards precede vectorized int64 arithmetic.

At j=0 this bounds the reciprocal sum over positive members of T_initial. At j=M, add epsilon_initial to bound the sum of 1/(n+1) over n in T_initial. These are different functionals and are labeled separately in the output.

## 4. Verification boundaries

The production computation uses exact rational Gaussian elimination for w and exact integer grid updates. Small finite-set languages are checked against exact Fraction sums. Stationary languages are cross-checked against Phase02's different centered-moment certificate. AP-free examples additionally receive the full carry/DFA reachability check.

No sampled finite list is passed off as an infinite sum. Conversely this finite-state summation engine cannot alone settle an unbounded-state or arbitrary-set harmonic supremum.
