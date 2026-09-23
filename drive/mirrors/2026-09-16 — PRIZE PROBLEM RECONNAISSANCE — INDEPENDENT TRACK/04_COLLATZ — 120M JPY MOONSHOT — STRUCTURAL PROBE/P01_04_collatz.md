# PR-COL-001 — Two precise obstructions to restricted descent certificates

Date: 2026-09-16. Complete elementary author-side proofs; known-type obstruction, no novelty or solution claim. We use the shortcut Collatz map T(n)=n/2 for even n and T(n)=(3n+1)/2 for odd n. This has the same positive-integer convergence question as the usual unshortened map.

## 1. Exact long increasing runs

For integers K,u>=1 and n=2^K u-1,

T^j(n)=3^j 2^(K-j)u-1, 0<=j<=K.

For j<K the value is odd, so the identity follows by induction. Moreover

T^K(n)/n > (3/2)^K.

These are arbitrarily long increasing initial segments, NOT a demonstrated divergent orbit. The quantifiers are 'for every K there exists n', not 'there exists n for every K'.

## 2. No globally bounded correction to logarithmic descent

There is no function V(n)=alpha log n+g(n), alpha>0, with g globally bounded, such that V(T(n))<=V(n) for every n outside a finite exceptional set.

Proof. Choose K with alpha*K*log(3/2)>sup g-inf g. Choose u sufficiently large that the entire increasing K-step segment starts beyond the exceptional set. Nonincrease at every step would imply V(T^K(n))<=V(n), whereas the exact expansion implies the reverse strict inequality. QED.

## 3. Finite residue / bounded-horizon obstruction

Fix m>=0,K>=1, and any g depending only on n modulo 2^m. No certificate of the form V(n)=alpha log n+g(n), alpha>0, can demand strict descent within a universally bounded horizon tau(n) in {1,...,K} for all sufficiently large n.

Take n=2^(m+K)u-1. For every 1<=j<=K, T^j(n)>n and T^j(n) is -1 modulo 2^m. The correction g is unchanged, so V strictly increases at every possible chosen horizon. QED.

## Exact scope of the obstruction

Unbounded corrections, horizons depending unboundedly on n, different well-founded orders, and richer nonperiodic certificates are not excluded. These no-go results only prevent wasted effort on the precisely stated restricted families. Tao's almost-all theorem does not supply the missing every-orbit implication; neither do these identities.
