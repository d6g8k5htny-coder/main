# Correction to the cumulative transfer statement (D2)

**Author:** OpenAI / ChatGPT. **Scientific effect:** NONE. Separate reviewer confirmation requested.

This additive correction replaces only the cumulative paragraph and its proof in Section D of NOTE.md at source ad35e46d15c2815c36746442808a1626a9724e8a (SHA256530dd3efaa965c850ea9e6575f42d3952c3efe6a89b2b5355b6afb4285c9d37e). The original source stays unchanged for provenance. Sections B/C and the density theorem D1 are not changed.

Nonauthor review PR32, commit189628119f65d845e270759e1b2b1e84ddb4cfe3, correctly found that saying only 'monotonicity and the same comparison/envelope assumptions' could drop the necessary normalization h/(kappa*r^m)->1. Two-sided comparisons do not determine the sharp constant. The correction explicitly retains that ratio hypothesis while dropping only derivative assumptions.

## Corrected cumulative theorem

Let (Z,lambda) be sigma-finite, alpha>-1, m>0, beta=(alpha+1)/m, and r0>0. Let all integrands be jointly measurable. Assume, for lambda-almost every z:

1. kappa(z)>0; 0<=B(r,z)<=G(z); B(r,z)->B0(z) as r->0.
2. The weighted negative moment integral G(z)*kappa(z)^(-beta) dlambda is finite.
3. h(r,z) is continuous and strictly increasing for 0<r<r0, with h(0+,z)=0 and

       h(r,z)/(kappa(z)*r^m) -> 1.

4. Uniformly in r,z, h(r,z)>=c0*kappa(z)*r^m for a fixed c0>0.

Neither a derivative limit nor a lower derivative bound is required for this cumulative theorem. Define

    N(ell)=integral_Z integral_0^r0 r^alpha B(r,z) 1{h(r,z)<=ell} dr dlambda.

Then

    lim_(ell->0) N(ell)/ell^beta
       = 1/(alpha+1) * integral_Z B0(z)*kappa(z)^(-beta) dlambda.

If the displayed coefficient is positive, this limit is a positive asymptotic equivalence. If it is zero, only the corresponding o(ell^beta) conclusion is asserted.

## Proof

At each fixed z and sufficiently small ell, let R=R(ell,z) be the inverse of h. The explicitly retained ratio hypothesis gives

    R^(alpha+1) / (ell/kappa(z))^beta -> 1.

Also,

    R^(-alpha-1) integral_0^R r^alpha B(r,z) dr
        -> B0(z)/(alpha+1),

by convergence of B at zero and alpha>-1. Therefore the normalized inner integral converges to B0(z)*kappa(z)^(-beta)/(alpha+1).

For domination, the lower comparison implies h(r,z)<=ell only if r<=(ell/(c0*kappa(z)))^(1/m). Thus, also when the inverse would exceed r0,

    ell^(-beta) integral_0^r0 r^alpha B(r,z)1{h<=ell}dr
       <= c0^(-beta) G(z)*kappa(z)^(-beta)/(alpha+1).

This is integrable by assumption2. Dominated convergence proves the result.

## Why the added precision matters

For h(r)=2*kappa*r^m with constant amplitude1, the actual coefficient is multiplied by2^(-beta). This example satisfies suitable fixed two-sided comparison bounds but violates h/(kappa*r^m)->1. For alpha=1,m=3,kappa=1 and R=1/2, ell=1/4 and N=1/8: the actual cube is1/512, whereas the uncorrected candidate constant (ell^(2/3)/2)^3 is1/128. The ratio hypothesis excludes precisely this mismatch.

This corrects an underspecified theorem statement; it is not an excuse to infer a density asymptotic from a cumulative asymptotic. D1 still requires its separate derivative controls. The Gaussian contact, thin-belt, whole-annulus, and global persistence statuses are untouched.
