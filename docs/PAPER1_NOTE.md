# Short-lifetime asymptotics for H0 persistence of smooth Gaussian fields

Scientific effect: NONE. This note does not close a claim, does not accept a theorem, and does not change STATUS.

## Observable

`nu_{2,24}(ell)` is the per-unit-volume intensity of finite nonessential H0 bars of lifetime less than `ell`, divided by `24^2`. It is not the raw expected count on the torus. A comparison that omits the volume divisor is off by `24^2`.

The persistence statement below is for dimension 2 only. The arithmetic enclosure of `c_{3,24}` is not a three-dimensional persistence theorem. A d=3 statement needs its own analytic chain and is not given here.

## Theorem (conditional, d=2)

Let f be the variance-one periodized Bargmann-Fock field on the flat torus of side 24 in dimension 2. Two populations must not be mixed.

- Compact birth and compact positive-gap hypotheses use the compact coefficient `c_(B,K)`, not the unrestricted `c_{2,24}`.
- The unrestricted coefficient `c_{2,24}` is the Kac-Rice / cone integral in `Math-/coefficients/side24_v1`, pinned by `docs/research-translation/20260930/SOURCES.json`. That enclosure is arithmetic. It is not, by itself, a persistence theorem, and it is not automatically the coefficient of the compact-restricted population.

Under the population actually proved, and under the shrinking-witness gap below,

nu = c * ell^{-1/3} (1+o(1)) as ell -> 0+,

where c is the coefficient of that same population. The o(1) is the D2 remainder relative to the leading term, at the scope STATUS already gives. Other constants in D1 are existential. This is not a numerical remainder, not a 24-jet certificate, not a claim for other covariances, and not a claim for H_k with k > 0.

## What stays open

D1 Theorem C, at the scope STATUS accepts, is an existential candidate/elder statement. D2 is a bounded remainder. Neither is a discharge of the shrinking witness-collision node. Hypothesis P, if it is still the right name, means only this narrower gap: after ell ~ r^3, the probability that a typed contact is not the elder pair is o(1) uniformly in the shrinking window. Do not treat this note as reopening a discharged obligation, and do not treat D1 C as closing that window.

## Scope box

Inputs: D1 A/B/C at the scope STATUS already records, D2 Theorem R, D3 arithmetic only, SARD-G as genericity for a fixed planar law, C6 planar factorials as tools.

Not claimed: a numerical C, an R^d limit, H1/H2, a cosmology detection, Lean of Kac-Rice, a d=3 persistence theorem, or attachment of `c_{2,24}` to a compact-restricted population.

## Literature, qualified

Chazal-Divol is a density result for other filtrations (fixed-size point samples; Brownian sublevel sets). It is not the existence input for this superlevel filtration. Feldbrugge et al. (Feldbrugge, van Engelen, van de Weygaert, Pranav, and Vegter) is a numerical fit, not a theorem. Bobrowski-Borman is the Euler integral of persistence, not the near-diagonal H0 intensity.

## What this note is for

A paper can be written from the d=2 box above. It cannot be cited as a theorem, and it cannot quote `c_{2,24}` for a population the enclosure was not computed for.
