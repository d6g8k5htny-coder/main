# Weighted-Palm transfer: retained analytic tool, not a claimed WP closure

**Status:** elementary conditional lemma, derived here; independent review open. This appendix is not needed by the local-path candidate. It records a safe fallback for the original W10 quantitative route.

Let Q_theta be a reference law and let w_theta>=0 with 0<Z_theta=E_Q w_theta<infinity. Define P_theta by dP_theta=w_theta dQ_theta/Z_theta. For any event B_theta and p>1,

    P_theta(B_theta)
    <= ||w_theta||_(Lp(Q_theta)) / Z_theta
       * Q_theta(B_theta)^(1-1/p).

Proof: apply Holder to w_theta times 1_(B_theta), then divide by Z_theta. All quantities refer to the same theta and the same conditional law. For a nonnegative loss X_theta, the corresponding statement is

    E_P X_theta <= ||w_theta||_p/Z_theta * ||X_theta||_(p/(p-1)).

An event probability bound cannot be substituted for the norm of an unbounded saddle count.

If normalized weights have uniformly bounded Lp norm and sup_theta Q_theta(B_theta)->0 in the small-r regime, the weighted probabilities also vanish uniformly. A rate Q(B)<=C r^alpha becomes P(B)<=M C^(1-1/p) r^(alpha(1-1/p)). The exponent may weaken, but remains positive. This can suffice for an existential composition without preserving the old O(r^3) remainder.

A still weaker sufficient assumption is uniform integrability of w_theta/Z_theta. For any cutoff T,

    P_theta(B_theta)
    <= T Q_theta(B_theta)
       + E_Q[(w_theta/Z_theta) 1_{w_theta/Z_theta>T}].

First make the second term uniformly small by choosing T, then the first by taking r small. No bounded pointwise density ratio is required.

## Exact obstruction to an unsupported transfer

Let Q be uniform on [0,1], B_r=[0,r], and w_r=(1/r)1_(B_r). Then E_Q w_r=1 and Q(B_r)=r->0, but P_r(B_r)=1 for every r. The normalized weights are not uniformly integrable. Thus vanishing under an unweighted law does not imply vanishing under its weighted version without additional control.

## What the old route still needs

For the exact arch-dependent triple-Palm law, establish the normalized-weight Lp bound or uniform integrability over *every consumed location, mark and r*, and prove the reference-law loss on that same family. A fixed station or representative height is not that family. If height is integrated over a window, its conditional mixture must be integrated exactly or enclosed; evaluation at a clipped mean is not an exact replacement without a separate argument.

The local-path route avoids this additional transfer altogether. It still uses, rather than ignores, the original pair-Palm weighting in its numerator and denominator.
