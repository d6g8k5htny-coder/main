# Explicit analytic guards for the fixed cusp population

This successor source supplies an explicit member of the smooth-cutoff class
consumed by the fixed population experiment. It proves the derivative and
annular guard premises that its frozen numerical receipt records as
`REVIEW_REQUIRED`. It introduces no new control value, random field draw,
population run, fitted coefficient, or claim of Gaussian universality.

Actual author: OpenAI/Codex /root, source exposed. Separate mathematical review
is required before this source is accepted. Organizational independence and
blind credit are zero. This document is a mathematical source, with complete
field-evaluator execution, independent global H0-selector execution, Lean
kernel verification, higher homology, spatial limits and scientific promotion
still `NOT_RUN`.

The consumed sources remain unchanged:

- `PROOF.md`: 782 lines, SHA-256
  `9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326`.
- `CONTROLLED_FALSIFICATION.md`: 270 lines, SHA-256
  `a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e`.
- The completed numerical receipt:
  `population-numerical-increment.json`, 2,206,767 bytes, SHA-256
  `cbaf37ef3b8002057dac7f81db72b95c3bf03015fc39dd34d7d2745f4ef4aa94`,
  private Drive file `1WvOSM_i6ECXQe3f06wO4DzjkhI0lz01U`.

Those historical bytes and their scope statuses are not rewritten.
In particular, the explicit radial argument below was not present in the
frozen cutoff-class description. This is new evidence for a concrete
realization of that class, rather than a claim of prior field execution.

## 1. Parameters, coordinates and the actual field

Keep exactly the declared parameters

```
d=2, L=1, R=1/8, r1=1/32, A=2^-16,
u0=3*A^2=3/2^32, v0=2*A^3=1/2^47,
a*=r1/4=1/128, delta=r1^4/64=2^-26.
```

Let `y` be the torus representative in `[-1/2,1/2]^2), and put

```
s=sqrt(2/3)*sin(pi*y1),
z=sqrt(2/9)*sin(pi*y2),
r=s^2+z^2, b=4/9.
```

In the chart `|yi|<1/4`, these are smooth coordinates with nonsingular
Jacobian, and the seed is exactly `h=b-r`. Since `R^2=1/64<1/9=w2`,
the superlevel `{h>b-R^2}` is precisely this chart's ball `D_R`, as
proved in the consumed source.

Define the functions below on all their indicated real arguments and set

```
F_(u,v)(y)=b-s^2*c(r)-z^2+chi(s,z)*(u*s^2/2+v*s),
                 |u|<=u0, |v|<=v0.                 (G1)
```

The definitions give `c=1` and `chi=0` for `r>=R^2/2`, so (G1)
equals the seed on a neighborhood of the exterior of `D_R`.
Although signed sine coordinates are represented with a seam, near any
torus seam (G1) equals the smooth periodic seed. Its values and every
derivative therefore match there. The formula defines a global C-infinity
field, jointly smooth in the two controls. Translating it preserves all
critical heights and H0 barcodes; no translation draw is needed for this
analytic assertion.

All gradient norms used in the guards below are Euclidean norms in
`(s,z)` coordinates. Nonsingularity of the chart transfers critical-point
exclusion to physical coordinates; no physical-gradient bound is asserted.

## 2. The flat step and its derivative

For a real argument define

```
q(t)=exp(-1/t) if t>0, and q(t)=0 otherwise;
psi(t)=q(1-t)/(q(t)+q(1-t)).
```

The denominator is positive on the whole real line: at least one of
`t,1-t` is at least `1/2`. The zero extension of q is C-infinity and
flat at zero. Indeed, by differentiating on the positive half-line, every
derivative is `exp(-1/t)` times a polynomial in `1/t`. For each
integer m, put `x=1/t` and choose an integer N>m. The exponential series
gives `exp(x)>=x^N/N!`, so
`x^m*exp(-x)<=N!*x^(m-N)->0`. The same estimate with one additional
power proves the difference-quotient matching at zero at every derivative
order. Induction therefore verifies the smooth zero extension and all its
zero jets.

It follows that psi is C-infinity, equals one for `t<=0`, equals zero
for `t>=1`, and is flat at both transition endpoints.
On `0<t<1` its derivative is

```
psi'(t)=-[q'(1-t)*q(t)+q(1-t)*q'(t)]
                     /[q(t)+q(1-t)]^2.              (G2)
```

Thus psi is nonincreasing and lies in `[0,1]`.
For `t>0`,

```
q'(t)=t^-2*exp(-1/t),
q''(t)=(1-2*t)*t^-4*exp(-1/t).
```

Consequently `0<=q'<=M=4*exp(-2)`, with its maximum at `t=1/2`.
Since q is nondecreasing and one argument is at least `1/2`,
`q(t)+q(1-t)>=exp(-2)`. The numerator in (G2) is at most
`M*[q(t)+q(1-t)]`. Hence, including the flat endpoints,

```
abs(psi')<=M/[q(t)+q(1-t)]<=4.                       (G3)
```

This proves the frozen weaker bound `abs(psi')<=8` without a numerical
estimate of an exponential.

## 3. Explicit radial cutoff and the four perturbation norms

Make the previously admitted cutoff class concrete by choosing

```
chi(s,z)=psi((s^2+z^2-r1^2/4)/(3*r1^2/4)).           (G4)
```

This is C-infinity, equals one on `|(s,z)|<=r1/2`, and equals zero on
`|(s,z)|>=r1`, with all jets matching across both boundaries.
On its transition region the inner radial argument has gradient norm

```
8*|(s,z)|/(3*r1^2)<=8/(3*r1).
```

Combining this with (G3) proves

```
||grad chi||<=32/(3*r1)<64/r1.                       (G5)
```

Zero extension from the local chart is smooth because the support lies
strictly inside `D_R`. Write B0,D0 for the sup norms of
`chi*s^2/2, chi*s`, and B1,D1 for their gradient sup norms.
On the support `|s|<=r1` and `0<=chi<=1`, so

```
B0<=r1^2/2,                         D0<=r1,
B1<=(64/r1)*(r1^2/2)+r1=33*r1,      D1<=(64/r1)*r1+1=65. (G6)
```

Here the gradient inequalities are the product rule and triangle
inequality, not assumptions about machine arithmetic.

## 4. Monotone surgery and the annular gradient

Use exactly the proposed surgery profile

```
c(r)=1-(1-r/4)*psi((r-r1^2)/(R^2/2-r1^2)), r>=0.    (G7)
```

Its transition width is positive since `r1=R/4`.
It equals `r/4` for `r<=r1^2` and equals one for
`r>=R^2/2`, with all derivative jets matching at both endpoints.
During the transition `0<=r<=R^2/2<1`; the formula is a convex
combination of `r/4` and one. Thus `0<=c<=1`. With w denoting the
positive transition width, differentiation gives

```
c'(r)=psi((r-r1^2)/w)/4
       -(1-r/4)*psi'((r-r1^2)/w)/w >=0.
```

For `r>0`, c(r)>0. Thus the unperturbed chart field
`F0=b-s^2*c(r)-z^2` has derivatives

```
(F0)_s=-2*s*[c(r)+s^2*c'(r)],
(F0)_z=-2*z*[1+s^2*c'(r)].
```

Both vanish together only at the origin. Outside the chart the seed
critical points remain unchanged. In the entire closed annulus
`rho=|(s,z)| in [r1/2,r1]`, (G7) gives the exact core polynomial,
including matching at its outer endpoint:

```
F0=b-s^4/4-(1+s^2/4)*z^2,
abs((F0)_s)=|s|*(s^2+z^2/2)>=|s|*rho^2/2,
abs((F0)_z)=2*|z|*(1+s^2/4)>=|z|*rho^2/2.
```

The final inequality uses `rho<=r1<1). Taking the Euclidean norm
therefore yields

```
||grad F0||>=rho^3/2>=r1^3/16>r1^3/64.              (G8)
```

Thus the frozen conservative witness `g*>=r1^3/64` is now proved for
this explicit field, in the same coordinate norm used by (G6).

## 5. The unchanged rectangle satisfies the uniform guards

Substituting the fixed controls into (G6) gives exactly

```
B1*u0+D1*v0 <= 101441/2^47 < 2^-22 <= g*/2,
B0*u0+D0*v0 <= 1537/2^52 < 2^-28 = delta/4.          (G9)
```

For the first strict comparison, `101441<2^25`; for the second,
`1537<2^24`. The root-containment guards are also strict:

```
u0=3/2^32 < a*^2/4=2^-16,
v0=2^-47 < a*^3/4=2^-23.
```

The wedge coverage is exact:
`(2/(3*sqrt(3)))*u0^(3/2)=2*A^3=v0`.
These checks use the existing parameter rectangle, without shrinking it.

The perturbation gradient is uniformly less than half the unperturbed
annular gradient, so it introduces no transition critical point.
Outside its support there is no perturbation. The core transverse
critical equation forces `z=0`, leaving exactly
`-s^3+u*s+v=0`. The consumed proof's root-containment argument places
every real cubic root within `|s|<a*<r1/2`.

There is also a uniform height barrier. For `rho>=r1/2` in the chart,
monotonicity and the core formula give `c(r)>=r1^2/16`. Since
`r1^2/16<1`,

```
b-F0=s^2*c(r)+z^2 >= (r1^2/16)*rho^2 >= delta.
```

Outside `D_R` the seed loss is at least `R^2>delta`.
The height estimate in (G9) consequently gives
`F_(u,v)<=b-3*delta/4` outside the core.
At a new critical point, the same elementary estimate consumed in
PROOF.md Section 2 gives `|P_(u,v)(s)|<delta/4`; hence every new
critical height is above `b-delta/4`. The two bands are separated
uniformly over the full rectangle.

The old saddle heights are `2/9,-2/9`, and the old minimum is
`-4/9`. They remain unchanged and lie below the isolated high-level
merger. The consumed transverse-fiber contraction then establishes the
actual component pairing for this explicit realization. This is an
analytic transfer through proved uniform guards; it does not assert an
executed independent whole-field selector.

## 6. Scope of closure and the next falsification control

This source discharges the stated cutoff smoothness, derivative,
perturbation-norm and annular-gradient premises for (G1), supplying a
concrete admissible realization of the fixed model. The original direct
and CDF population calculations do not depend on which admissible
cutoff realizes the isolated core; their original source identities and
historical review statuses remain intact.

Execution of a full periodic evaluator and an independent H0 selector
remains `NOT_RUN`. An eventual selector must derive connectivity from
certified axial level-set incidence and the proved global barrier,
before comparing its lifetimes with the local gap formula. A finite
regular grid does not resolve arbitrarily microscopic components.

It must also distinguish actual topology from the probabilistic
null-set convention. At `v=0,u>0`, the core maxima
`s=+-sqrt(u)` both have height `b+u^2/4`, the middle saddle has
height b, and the global barrier still isolates their merger.
The ordinary H0 barcode therefore contains one finite bar of lifetime
`u^2/4`, as well as the essential class. Tied component identity may
require a label convention, but that finite lifetime is unambiguous.
The frozen population law's convention excludes the null line and
projects its count to zero. That law projection must be applied
separately after recording the actual tied barcode.

No Gaussian universality, independent organizational vote, new random
benchmark, marked spatial limit, higher-homology or Lean acceptance is
provided here. The full landmark goal remains active.
