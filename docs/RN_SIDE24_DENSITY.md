# Fixed-point SIDE24 density and height-window factor

This author-produced component completes the density/window factor for the
existing SIDE24 point calculation at \(y=(1,1)\), \(r=1/20\), \(b=6/5\).
It uses the entire height interval and imports the exact pinned H3 floor as a
separate premise. It re-proves no H3 assertion, supplies no spatial cover or
spatial integral, proves no all-small-r extension, changes no scientific
status, and earns zero organizational independence credit.

## Statement and composition

Use the field, coordinate order and six fixed pin values from
[RN_SIDE24.md](RN_SIDE24.md). Write \(Q\) for the law conditioned on those six
pins, \(Y=(f(y),f_x(y),f_y(y))\), \(g=\nabla f(y)\),
\(\ell=1/48000\), and \(J=[b-\ell,b]\). The new calculation bounds

\[
D=p_{g\mid Q}(0)\,
  \Pr\{f(y)\in J\mid Q,g=0\}.
\]

All conditional laws here are the Gaussian disintegrations given by positive
covariance pivots. Conditioning on gradient zero is not an event of positive
probability. The computed upper for D is approximately
\(4.00147595754\cdot10^{-6}\); the result stores an exact rational upper.

Let \(T(v)\) be the conditional typed determinant-product expectation given
the six pins and \((f(y),g)=(v,0)\). The preceding moment certificate bounds
\(T(v)\leq U\) for every \(v\in J\). Thus

\[
\frac{1}{Z_r}\int_J p_{Y\mid Q}(v,0,0)T(v)\,dv
\leq \frac{D_{\rm upper}U}{Z_{\rm lo}},
\qquad Z_r\geq Z_{\rm lo}>0.
\]

The floor \(Z_{\rm lo}=0.0077592917375327855\) is imported, as described
below. Even using the slightly rounded former bound \(U=0.007322236\), the
right side is below the exact rational \(3776086/10^{12}\). This is one
fixed-point RN integrand upper conditional on the imported H3 premise. It is
not an annulus bound, a bound after spatial integration, or a closure of the RN
obligation. The checker composes the actual exact rational caps, not these
approximate displayed values.

## Six-pin jet law

`side24.nine_pin_blocks` supplies the coherent raw covariance under the exact
normalized torus kernel. The existing point-law document proves the kernel
tail and joint derivative Gram arguments. The new code takes its first six
coordinates as the fixed pins and its next three as Y, without conditioning
on a height mark yet.

For the raw block covariance
\(\left(\begin{smallmatrix}A&C^T\\C&B\end{smallmatrix}\right)\)
and fixed six-pin vector c, the law of Y under Q has

\[
m=CA^{-1}c,\qquad S=B-CA^{-1}C^T.
\]

Certified interval LDL solves enclose these quantities. The code requires
positive lower endpoints for all six pin pivots and all three jet pivots. A
failed sufficient pivot check is inconclusive, not evidence that the actual
field covariance is singular.

Partition \(m=(m_f,m_g)\) and
\(S=\left(\begin{smallmatrix}s_f&c_g\\c_g^T&G\end{smallmatrix}\right)\).
Then

\[
p_{g\mid Q}(0)=
\frac{\exp(-q/2)}{2\pi\sqrt{\det G}},\qquad
q=m_g^TG^{-1}m_g.
\]

If \(G=L\operatorname{diag}(d_1,d_2)L^T\), solve
\(w=L^{-1}m_g\) and evaluate \(q=w_1^2/d_1+w_2^2/d_2\).
This square sum preserves nonnegativity in interval arithmetic. Both pivots
must have positive lower endpoints. Also \(\det G=d_1d_2\), so no floating
determinant or matrix inverse enters the density formula.

The height conditional on gradient zero is Gaussian with

\[
\mu=m_f-c_gG^{-1}m_g,\qquad
\sigma^2=s_f-c_gG^{-1}c_g^T.
\]

The subtraction and the height/gradient covariance are essential. At this
point the enclosed conditional height mean is approximately 1.22864492785 and
the variance is approximately 0.113522346312. The calculation refuses unless
the variance lower endpoint is positive.

## Complete window bound

Suppose \(\mu\in[\mu_-,\mu_+]\), \(\sigma^2\in[v_-,v_+]\) with
\(v_->0\), and the window is \([a,b]\). Set

\[
\delta=\max(0,\mu_- - b,a-\mu_+).
\]

For every admitted mean and every x in the entire window,
\(|x-\mu|\geq\delta\). Consequently

\[
\Pr\{X\in[a,b]\}
\leq\min\left(1,(b-a)\sup_{v\in[v_-,v_+]}
\frac{\exp(-\delta^2/(2v))}{\sqrt{2\pi v}}\right).
\]

The code evaluates the expression for the full variance interval, taking its
upper endpoint. Repeated appearances of v may widen the enclosure; they do
not assume independent variances or weaken containment. Multiplication by the
nonnegative gradient-density upper gives the asserted density/window upper.
At y=(1,1), the entire mean interval lies above b, so the closest point is the
upper endpoint b. A wrong farthest-endpoint choice can underestimate the mass;
the tests include that negative control.

This bound uses neither numerical quadrature nor a CDF difference. A separate
test evaluates the existing certified CDF primitives and proves with exact
rational comparisons that the length-times-sup cap exceeds the enclosed
probability and is within three parts per million of its lower endpoint at
this point. That is a tightness comparison for this one calculation, not the
reason the whole-window inequality is valid.

## H3 custody and imported premise

The source is the member
`intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md`
inside the same frozen RN5 archive pinned by `side24.source_binding`:

- Archive SHA-256: `28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e`.
- H3 member: 7003 bytes, SHA-256
  `6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa`.
- Exact imported floor: `0.0077592917375327855` at \(r=1/20\), \(b=6/5\).

The member's displayed rung interval is parsed, required to occur exactly once,
and compared as an exact Fraction with the pinned floor. Its source describes
the normalized side-24 field and the equivalent transformed six-pin law.
RN5's `round5/near_moments.py` lines 90–110 supplies the density/window
factorization and its division by this same floor. The new code reads those
frozen bytes as data and executes none of them.

The returned H3 status is `IMPORTED_SOURCE_PREMISE_NOT_REPROVED`. Byte matching
authenticates the dependency identity, not its mathematical proof or canonical
acceptance. This component does not replay H3's interval integral, infer an
all-r floor, or relabel its source status. `h3_reproved`, `field_certified`,
`spatial_cover_certified`, and `scientific_status_changed` stay false.

## API and validation

`research.rn.side24_density.density_window(point=(Fraction(1), Fraction(1)),
bits=192)` returns:

- the six-pin values/order, jet mean/covariance and positive pivots;
- gradient mean/covariance, positive pivots, Mahalanobis interval and density;
- conditional height mean/variance, full window, nearest-distance lower bound,
  density upper and probability upper;
- the exact `density_mass_upper` and separate `imported_h3_floor` metadata;
- source identity and explicit scope flags.

The reusable `gradient_density` and `height_window_cap` helpers permit exact
semantic checks with simple Gaussian laws. Float/bool data, invalid dimensions,
nonpositive or unresolved covariance/variance, reversed windows, unsupported
precision and oversized rational inputs are refused. Calculations use the
existing certified interval exponential, square root and pi, with outward
dyadic rounding and bounded input sizes. No additional dependency is needed.

```bash
python -m pytest -q -p no:cacheprovider tests/test_rn_side24_density.py
```

Tests compare correlated gradient densities to closed-form Gaussian laws,
challenge exponent signs and nearest-window endpoints, verify uncertain
mean/variance member inclusion through a separate CDF formula, check source
coordinate/height-window binding, and reject altered H3 identity/floor data.
These tests share interval primitives with production. They earn zero
organizational independence credit and do not replace review of the argument.
