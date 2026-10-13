# From rare pairing failure to expected counts: exact interface and sharp limits

**Object:** RN-COUNT-INTERFACE-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side mathematical derivation, not an RN numerical certificate.

## 1. What is, and is not, supplied

Campaign task RN-04 asks whether a pairing-failure probability bound can supply the unresolved RN expected-critical-count estimate. The answer depends on an extra count-weighted estimate. This note proves the exact implication, its limitations, and the corresponding full-pin Kac-Rice expression. It does not invent an absent RN driver, claim a 24-jet enclosure, or relabel a probability theorem as an expected-count theorem.

At the observed main hardening source `docs/math_status/STATUS_RN_UNIF.md`, Git blob `ccd542fb174e5a242ee343fad213aa40b35b733d`, the original lane remains OPEN with separate ENV-RESCOV, ALLCELL-FDZ-Q4, LOGQ-TAIL, SYM-Fw-jet and CH-LIFT obligations. Those source-specific predicates are not modified by this new logical interface.

## 2. General probability-to-count estimates

Let E_r be an event, N_r>=0 an integer-valued random variable, and assume P(E_r)<=C r^3. A count of interest may be supported on E_r; otherwise the results here apply to N_r 1_Er, not its entire expectation.

Markov gives P(N_r>=1)<=E N_r. It cannot be reversed. For every p>1, Holder gives

    E[N_r 1_Er] <= (E N_r^p)^(1/p) P(E_r)^(1-1/p).       (N1)

If E N_r^p<=A r^(-beta), then

    E[N_r 1_Er] <= A^(1/p) C^(1-1/p)
                     r^([3(p-1)-beta]/p).               (N2)

In particular a uniformly bounded p-th moment supplies r^(3-3/p), not r^3. The powers in (N2) refer to the p-th MOMENT itself, not its L^p norm; conflating those conventions changes beta's contribution.

For a probability q=P(E_r)>0 there is an exact identity

    E[N_r 1_Er]=q E[N_r | E_r].                          (N3)

Thus a uniform conditional mean bound E[N_r|E_r]<=A, plus P(E_r)<=C r^3, is enough for the desired cubic rate. When additionally q>=c r^3, a cubic expectation bound is equivalent to a bounded conditional mean. Without this lower comparison, bounded conditional mean is sufficient but not necessary.

Another exact route is to establish the weighted quantity E[N_r 1_Er] directly. Merely knowing that all derivative suprema have all moments is not a count-moment estimate without a separate theorem tying the number of zeros to those quantities.

## 3. Sharp counterexamples: all finite moments still do not preserve the exponent

These are abstract probability counterexamples to an inference. They are NOT asserted to be realizations of the Gaussian field.

On one space take U uniform on [0,1]. For 0<r<=exp(-1), let

    E_r={U<=r^3},
    N_r=ceil(log(1/r)) 1_Er.

Then P(E_r)=r^3 and N_r is supported on E_r, but

    E N_r/r^3=ceil(log(1/r)) -> infinity.                 (N4)

Nevertheless for EACH finite p>=1,

    sup_r E N_r^p <= sup_(t>=1) e^(-3t)(t+1)^p < infinity.

The last supremum is finite because its logarithmic derivative is -3+p/(t+1); beyond its possible finite maximum it decreases exponentially. The constants may depend on p. Hence even a statement giving every finite moment, with no controlled dependence on the order, does not establish the cubic expectation rate.

For sharpness at one specified p, take N_r=ceil(r^(-3/p))1_Er. Its p-th moment is bounded by 2^p for r<=1, while E N_r is of order r^(3-3/p). Integer rounding does not remove the exponent loss.

There is a useful exponential-tail version. If P(N_r>t)<=A exp(-t/B), A>=1, then layer cake gives, for q=P(E_r)>0,

    E[N_r 1_Er] <= integral_0^infinity min(q,Ae^(-t/B))dt
                  = B q[1+log(A/q)].                    (N5)

A cubic event bound therefore gives a cubic rate with a logarithmic loss for sufficiently small r. The example (N4) even satisfies a uniform exponential tail with suitable fixed A,B, so that loss cannot be removed from the tail assumption alone. It does not become a cubic estimate merely by optimizing a large Holder exponent.

## 4. Apply the interface to the actual pinned weighted law

Use exactly the parent main issue63 Gaussian regression law Q conditioned on O_r=v, with

    W_r=|det H_M det H_S| 1{M maximum,S index-(d-1) saddle},
    Z_r=E_Q W_r,             dQ^W=(W_r/Z_r)dQ.

The parent source SHA is `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`. For a Borel region B not containing M,S and a Borel selector 0<=chi(x,f)<=1, define

    N_B(f)=sum_(x in B: grad f(x)=0) chi(x,f).

For instance chi may select a critical-point type, a height window, or another exact witness property. If it is not 0/1 this is a weighted count; all formulas still apply. Start on compact B separated from both pins. Residual finite-jet rank gives a nonsingular conditional gradient law there, and Gaussian regression gives locally finite conditional moments. Apply the marked Kac-Rice identity under Q, retaining the ENDPOINT weight as a field mark. The result is

    E_Q^W N_B
      = 1/Z_r integral_B p_(grad f(x)|O_r=v)(0)
           E[W_r |det H_x| chi(x,f)
                          | O_r=v, grad f(x)=0] dx.      (N6)

For E_r insert 1_Er INSIDE the conditional expectation. It must not be factored out. The result is E_Q^W[N_B 1_Er]. Endpoint Hessians, the witness Hessian and E_r can be strongly dependent under these pins.

The Borel-mark passage is the parent Section9 convention: establish equality on bounded continuous cylinder marks on a compact domain and extend the resulting finite measures, then truncate W_r if needed and use monotone convergence. For regions approaching pins, exhaust by compacts separated from pins. Equation (N6) then holds as an equality of nonnegative extended quantities, possibly +infinity. Uniform finiteness or a rate cannot be concluded simply from this identity.

With a lower normalizer bound Z_r>=z_*r^2, an actual enclosure

    integral_B p_(grad f(x)|O_r=v)(0)
       E[W_r |det H_x| chi(x,f) | all pins] dx <= A r^5  (N7)

would imply E_Q^W N_B<=(A/z_*)r^3. If the event indicator is included, the same statement controls the event-weighted count instead. The r^5 numerator and its three determinant factors are explicit. A cell decomposition must cover the WHOLE declared B and all stated parameter faces before its cell enclosures sum to (N7).

The cap probability argument controls E_Q[W_r 1_Gc], not the integral in (N7). That is the missing mathematical distinction. An original RN spatial parameterization may carry additional change-of-variable factors; they must be kept when translating (N7) to that parameterization rather than copied from the probability ledger.

## 5. Where support on the rare event is actually valid

The exact deterministic marked-cylinder source (Drive `1BnPods7Lf-ECdD34noQihZcEfcqpy7R5`, SHA `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc`) proves that its full cylinder contains precisely M,S on G_r. Therefore an additional-critical count in a region contained in THAT cylinder, excluding the pins, is zero on G_r. For that count, N=N1_Gc and (N1)-(N7) apply to its entire expectation.

For a remote region outside that cylinder, this support statement is not supplied. A remote critical point can exist while the prescribed maximum/saddle pair is correctly paired. Even before a moment estimate, a proposed RN transfer must identify its count region and prove the required support inclusion, or work with N1_Gc only.

This location distinction is independent of the parent's unconditional fixed-off-diagonal O(1) height-density bound. That bound is not a shrinking-annulus triple-count enclosure.

## 6. Completion test for this interface

A genuine cubic expected-count completion must provide at least one of:

1. the source-specific full integrated numerator (N7), with all chart/Jacobian and conditioning controls; or
2. a uniform conditional count mean on the actual rare event and the required support relation.

A finite p-th moment gives the precise weaker rate (N2). An exponential tail gives (N5). Neither is silently rounded up to exponent3. A collection of finite-radius samples does not prove any all-small-r version of these claims.

The supplied program checks exact Bernoulli examples, rational rate exponents and the correct direction of the count/event inequalities. It does not evaluate the Gaussian triple integral or discharge ENV-RESCOV, the 24-jet roster or any historical missing carrier.

External context inspected: Armentano-Azais-Leon, arXiv:2304.07424v3 Theorem7.1 for expected integrals and Section8.1 for critical points. Armentano et al., arXiv:1909.10243v2, abstract, identifies separate differentiability, derivative-moment and density hypotheses for scalar level-set moment results. Only its abstract was inspected; no uniform vector-gradient count-moment theorem is imported from it. Holder, layer cake and the counterexamples are derived directly above.
