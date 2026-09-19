# PR-AP-004: A single-switch policy and an exact continuous-state enclosure

Date: 2026-09-16. Complete author-side analytic derivation with executable finite certificates; external review pending. Uses PR-AP-003's exact class and two canonical actions.

## 1. Upper and lower envelopes for the WHOLE continuum

Let m be a positive mesh integer, Q an integer scale, and j=0,...,m. U is the regular optimal value of PR-AP-003, not the full function 1/a+U. Initially

    lo_0[j]=0, hi_0[j]=9Q.

For b in {11,22}, d in its canonical alphabet, and a=j/m, let

    i_floor = floor((dm+j)/b),
    i_ceil  = ceil((dm+j)/b).

Since U is decreasing,

    U(i_ceil/m) <= U((d+a)/b) <= U(i_floor/m).

Thus the fact that a child does not lie on the mesh causes no omitted error: it is enclosed by the two exact neighboring grid states.

The immediate reward h_b(a)=sum_(d>0) m/(dm+j) is enclosed by individually flooring/ceiling each scaled term. Given an existing enclosure at every grid point, form

    L_b[j]=h_floor_b[j]+floor(sum_d lo[i_ceil]/b),
    H_b[j]=h_ceil_b[j]+ceil(sum_d hi[i_floor]/b),
    lo_new[j]=max_b L_b[j],
    hi_new[j]=max_b H_b[j].

Induction proves lo[j]/Q <= U(j/m) <= hi[j]/Q after EVERY iteration. This is not an interpolation guess and does not require stopping only after numerical convergence. Sixty-four iterations with m=10^6,Q=10^12 give the recorded value enclosure. All operations fit signed 64-bit integers under explicit range guards. In particular Qm<=10^18, every accumulator is below 10^15, and ceil numerators stay below 2^63.

A separately written Python Fraction implementation reproduces the same updates on small meshes. The production outputs are checked again in normal and optimized Python and from a fresh extracted package.

## 2. Explicit near-optimal INTEGER SET, not just existence of a supremum

At each iteration the lower-envelope calculation records the maximizing base at each grid state (ties use base 11). These choices are stored losslessly by run-length encoding in EXPLICIT_POLICY.json. There are 64 horizon layers and 441 runs in the supplied policy.

Starting at integer n-1, depth 64, and grid index j=m, select the recorded action for that depth and j. Divide by its base and reject any unallowed digit d. Update

    j <- ceil((dm+j)/b), depth <- depth-1.

At depth zero continue with the canonical base-11 digit rule. If the integer quotient becomes zero earlier, it is accepted: every subsequent zero digit is permitted.

This defines a fully explicit infinite set. The grid state is part of the actual construction, not a claim to compute the unknown exact optimal policy. The lower dynamic program underestimates its value. To see this, suppose the actual shift is a<=j/m. Immediate rewards increase when the shift decreases; the updated upper-rounded grid state bounds the actual child shift from above. Induction proves that the stored lower value is achieved or exceeded by the constructed finite-horizon prefix, and adding the canonical infinite tail only adds mass.

Consequently this explicit set is 4-AP-free and has reciprocal sum at least

    4.422891010185.

It is within less than 10^(-6) of the optimum over every allowed radix 2 through 30.

## 3. A single switching threshold describes an exact optimizer

Define

    G(a)=L22 U(a)-L11 U(a).

We prove G is strictly decreasing and changes sign once. The optimal action is therefore base 22 when a<tau and base 11 when a>tau; either is optimal at equality. tau is defined by G(tau)=0, not asserted to have a closed form.

The nontrivial point is that U can have kinks. We do NOT assume it is continuously differentiable.

For a full canonical-action tree T define

    Q_T(a)=sum_(n in T,n>0)1/(n+a)^2.

Every Q_T is decreasing and 3-Lipschitz because 2 sum_(n>=1)n^(-3)<=3. Also 0<=Q_T<=2. Let q_min=inf_T Q_T and q_max=sup_T Q_T. They inherit monotonicity and Lipschitz continuity. Their dynamic recursions use respectively min and max of

    s_b(a)+(1/b^2)sum_d q((d+a)/b),
    s_b(a)=sum_(d>0)1/(d+a)^2.

Starting from lower zero and upper two, neighboring-grid evaluation as in Section 1 gives rigorous lower envelopes for q_min and upper envelopes for q_max. The squared denominator here is b^2; replacing it by b is a test mutation.

For every fixed tree, B_T(a+h)-B_T(a)=-integral_a^(a+h)Q_T(t)dt. Taking suprema and the uniform min/max bounds shows

    -q_max(a) <= U'(a) <= -q_min(a)

where the Lipschitz function U is differentiable, hence almost everywhere.

Since D22 contains D11 and adds E={8,9,14,17}, almost everywhere

    G'(a) = -sum_(d in E)1/(d+a)^2
             +(1/22^2)sum_(d in D22)U'((d+a)/22)
             -(1/11^2)sum_(d in D11)U'((d+a)/11).

On each FULL mesh cell [j/m,(j+1)/m], an upper bound is obtained by evaluating the negative terms at the right endpoint (using q_min lower bounds) and the positive terms at the left endpoint (using q_max upper bounds), with neighboring-grid monotonicity in the child arguments. This explicitly covers the intervals between sample points.

With m=10000,Q=10^10 and eight iterations, exact integer arithmetic gives

    G'(a) <= -42799603/10^10 < -1/250

almost everywhere on [0,1]. The exact reported numerator is verified by the script and receipt. G is Lipschitz, thus absolutely continuous, so integrating this a.e. bound proves strict decrease despite any kinks.

The fine U enclosure gives G(4539/10000)>0 and G(4546/10000)<0. Therefore there is exactly one threshold satisfying

    0.4539 < tau < 0.4546.

This proves the threshold characterization of the exact optimum. It does NOT claim that tau is rational or that a finite decimal rule implements the exact optimum. The explicitly computable near-optimal set in Section 2 avoids that issue.

## 4. What the computer does and does not establish

The finite code certifies rational enclosures and algebraic/combinatorial inputs. The proofs of AP-freeness, the tree recursion, the continuous-state envelope induction, a.e. derivative control, and the infinite-horizon passage are written above and in PR-AP-003. They remain subject to independent mathematical review.

A mesh alone would NOT be a uniform proof. Here monotonicity encloses every off-mesh child, and the derivative certificate encloses every full parent cell. The full range of a, all tree depths, and all permissible alphabet choices are addressed separately.
