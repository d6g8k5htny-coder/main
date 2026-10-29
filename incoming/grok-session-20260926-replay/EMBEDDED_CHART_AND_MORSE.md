# Embedded 2r chart and Morse import — parent vs cap

**Object:** GROK-HEAVY-EMBED-MORSE-20260926-v1
**Scientific effect:** NONE. Does not accept Theorem A.
**Sources:** parent SHA256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` at Math- `5ed3b455`; cap blob `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1`.

## What the cap file actually assumes

Verbatim:

> Let d>=2 and use coordinates (x,y) in R x R^(d-1) on an **embedded product cylinder**
> D=[-2r,2r] x closed_ball(0,2r), r>0.

The cap theorem is Euclidean. It does not mention the torus. It does not prove embedding. Morse / distinct critical values appear only in the interpretation sentence: on a compact manifold, for a global Morse extension with distinct critical values, the ordinary superlevel elder partner of M is S.

## What parent §1 actually writes

On X = R^d/(L Z^d):

> For sufficiently small r>0, use the embedded local cylinder centered at the origin with pins M=-(r/2)u, S=(r/2)u.

No displayed inequality `r < L/c`. Theorem A exists `r_* = r_*(d,L,B,K)`. That is the whole embedding sentence.

A sufficient Euclidean condition, **not claimed as parent text**: D sits in the ball of radius `2r√2` about the origin, and the torus exponential chart is injective on radius `L/2`, so `2r√2 < L/2` i.e. `r < L/(4√2)` embeds D. Parent absorbs this into `r_*`.

## What parent §8 actually writes

At each fixed r and each Q:

- overdetermined maps `(grad f(x), H_f(x)v)` (range 2d, param 2d-1),
  `(grad f(x), grad f(y), f(x)-f(y))` (range 2d+1, param 2d),
  `(grad f(x), f(x)-b)` (range d+1, param d);
- pin-residual covariance PD from §2;
- mesh + density bound ⇒ no zeros;
- pinned Hessians have full symmetric density ⇒ `det ≠ 0` a.s.;
- pinned heights differ because `k,r>0`;
- countable exhaustion ⇒ Q a.s. Morse with distinct critical values;
- transfer to Q^W by absolute continuity and `0<Z_r<∞`.

Elder event: `d_f(M)` is the maximin connection level; `{d_f(M)>h}` is a countable union of polygonal-path tests; `{d_f(M)=f(S)}` is Borel. On the Morse distinct-value locus this is ordinary elder death at S.

On `G_r` the cap supplies a connection through S above b and controls every boundary exit, so the **global** ordinary elder partner is S. Cap also gives exactly one ascending branch to M; it does not identify that branch with the transverse ridge.

Parent then says equations (7.8) plus this locus prove Theorem A. That last sentence is the A7 slogan, not a new estimate.

## Slogan vs source

| Slogan | Source |
|---|---|
| “embedded 2r product chart on the torus” | Cap assumes Euclidean embedding. Parent says “sufficiently small r” / `r_*`. Explicit `r < L/(4√2)` is a reader comparison, not a displayed parent line. |
| “G_r ⇒ elder pairing” | Cap + Morse + distinct values + Borel `d_f(M)=f(S)` + no wrap. All of those are named. The torus embedding inequality is the only missing displayed line. |
| “Morse-Smale” | Not used. §8 is Morse + distinct values + maximin elder, not stable/unstable manifold transversality. |

## Verdict

ACCEPT the cap Euclidean theorem as a pathwise implication of its estimates on an already-embedded cylinder.
ACCEPT §8 Morse/distinct-values/Borel structure as written (prior D1-A review already took §8–§15; this note does not re-open it).
AMEND the slogan that parent §1 displays an embedding radius. It exists `r_*`. That is enough for an existential Theorem A and not enough for a numerical `r_*`.
