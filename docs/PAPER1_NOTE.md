# Short-lifetime H0 persistence: statement and measurement guide

[Read the manuscript](research-translation/20260930/MANUSCRIPT.md) · [Follow the proof chain](research-translation/20260930/PERSISTENCE_BRIDGE.md) · [Reproduce the experiment](research-translation/20260930/EXPERIMENT.md) · [Inspect exact sources](research-translation/20260930/SOURCES.json)

This is a short reading companion to the existing planar manuscript. It explains the quantity being counted and how to compare it with data. It does not introduce another theorem or replace the manuscript's source-bound proof chain. Scientific effect: NONE.

## One field and one observable

The model is the centered, variance-one periodized Bargmann-Fock field on the two-dimensional flat torus of side 24. Its covariance is

```text
K_24(z) = sum[n in Z^2] exp(-|z+24n|^2/2)
          / sum[n in Z^2] exp(-|24n|^2/2).
```

Decrease the level through the ordinary superlevel filtration. A finite H0 bar born at a maximum M and killed at its merging saddle S has lifetime f(M)-f(S)>0. Exclude the essential class born at the global maximum; do not discard the longest finite bar. For Borel sets A contained in (0,infinity), define

```text
mu(A) = 24^(-2) E[# finite ordinary superlevel H0 bars with lifetime in A],
nu(ell) = dmu/dell.
```

The area divisor appears exactly once. This expected intensity includes all birth heights and spatial separations. It is not normalized by the random number of bars in a realization. The source chain supplies the density; it is not assumed merely from a plotted histogram.

## What the existing manuscript states

At its recorded planar consumption scope, manuscript sections 2-6 assemble D1's repaired global elder-selector and lifetime results, D2's bounded remainder, and the SIDE24 coefficient expression:

```text
nu(ell) = c_(2,24) ell^(-1/3) + O(1),       ell -> 0+,
mu((0,t]) = (3/2)c_(2,24) t^(2/3) + O(t),  t -> 0+.
```

The torus side and dimension are fixed before the lifetime limit. The remainder constant and valid lifetime cutoff are existential, not evaluated error bars. The SIDE24 enclosure is an arithmetic result for the coefficient expression; its interpretation as a persistence coefficient consumes the repaired parent chain. See the [manuscript's exact statement](research-translation/20260930/MANUSCRIPT.md#2-the-assembled-planar-statement) and [source manifest](research-translation/20260930/SOURCES.json), whose mathematical source cut is Math- `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1`.

The parent must be read with its congruence erratum, Section 9 replacement v1.1, wording W1 and embedding restriction `r < L/(4 sqrt(2))`. The [statement crosswalk](research-translation/20260930/CROSSWALK.md) and [completion audit](research-translation/20260930/COMPLETION_AUDIT.md) explain the consumed interfaces and remaining assembly limits. This guide records their existing scope; it does not perform a new acceptance review of those proofs.

A compact birth/gap population is a different measure. Restrict its count and use its compact coefficient c_(B,K) together. Do not attach unrestricted c_(2,24) to that population. The enclosure of c_(3,24) likewise does not provide a dimension-three persistence proof for this planar exposition.

## Counts, density and bin width

Under the leading density law nu(ell)=c ell^(-1/3)(1+o(1)), integration gives

```text
mu((0,t]) = (3c/2) t^(2/3) (1+o(1)),
mu((ell,2ell]) = (3c/2)(2^(2/3)-1) ell^(2/3) (1+o(1)).
```

Raw counts in proportional-width bins therefore also have exponent +2/3. For m independent field realizations and a bin [a,b), 0<a<b,

```text
mean count per unit area = aggregate bin count / (m * 24^2),
bin-average density     = aggregate bin count / (m * 24^2 * (b-a)),
leading bin mass        = (3c/2)(b^(2/3)-a^(2/3)).
```

Compare bin mass with the integrated prediction, or divide both by bin width. A proportional-width bin average has an integrated shape factor; do not equate its prefactor with c evaluated at an arbitrary bin center. Independent realizations, rather than bars within one realization, are the replication units.

The leading asymptotic alone gives the displayed little-o cumulative law. The stronger cumulative error O(t), or bin-mass error bounded by C(b-a), uses a density bound `|nu(ell)-c ell^(-1/3)| <= C` throughout (0,t] or [a,b), respectively. Without evaluated C and a valid cutoff, it is not a numerical accuracy guarantee. The [experiment protocol](research-translation/20260930/EXPERIMENT.md) specifies these normalizations and the discretization controls.

## Which questions remain separate

The [research plan](research-translation/20260930/README.md) and [completion audit](research-translation/20260930/COMPLETION_AUDIT.md) distinguish the existing D1/D2 leading-law chain from the regional shrinking multiple-witness problem. The latter remains a separate obligation; it is neither an extra blanket premise of this manuscript nor discharged by its leading law. A global factorial-moment bound alone does not supply the missing regional collision estimate.

Actual local partners, additional witnesses, rejected candidates and replacement bars remain different populations. The current [refined-selector work](https://github.com/d6g8k5htny-coder/main/issues/229) has its own finite-radius, mark, weighting and confinement requirements. None of those live candidates is silently imported into this source cut.

For comparison with observations, the next requirements are evaluated remainder constants, a justified lifetime window, and certified continuum-to-computation error bounds. Finite spectral truncation and a grid change the object being measured; refinement or a good fit alone does not establish a continuum theorem. Other dimensions, covariances, higher homology and joint infinite-volume/lifetime limits require separate arguments. The [literature comparison](research-translation/20260930/LITERATURE.md) states the different observables of related work without claiming exhaustive novelty clearance.

The manuscript is still a draft exposition with imported proofs and stated assembly limits. Source identities, software checks, AI technical reviews and this companion do not constitute independent human review, journal acceptance or mathematical completion of the entire program.
