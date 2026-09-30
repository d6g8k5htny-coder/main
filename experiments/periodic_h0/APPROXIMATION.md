# Deterministic comparison with a finite Fourier realization

[Experiment](README.md) · [Original protocol](../../docs/research-translation/20260930/EXPERIMENT.md) · [Fixed pilot](results/pilot32/RESULTS.md)

This note derives deterministic spatial approximation and lifetime-bin bounds.
They apply to a specified finite Fourier realization, or more generally a
specified twice continuously differentiable function on the square torus. They
do not establish a uniform spectral-tail enclosure, a lifetime asymptotic
window, a remainder constant, or acceptance of the continuum coefficient.
Ordinary floating-point evaluations of the bounds are diagnostics until their
inputs and arithmetic have rigorous enclosures.

The distinction between a function approximation and its persistence diagram
matters here. The direct cubical step function has a first-order uniform error
bound. Nevertheless, the planar vertex cubical filtration has a continuous PL
representative with a second-order error bound. Consequently, a first-order
limitation for the cubical H0 barcode does **not** follow from the step-function
bound. The proof below constructs that representative explicitly.

## Setting and stability input

Let `T = (R/LZ)^2`, let `n >= 3`, and put `h = L/n`. Use the periodic grid
`(ih,jh)`, with no duplicated endpoint at `L`. Let `f` be real and `C^2` on
`T`. Suppose supplied vertex values `s_v` satisfy

```text
max_v |s_v - f(v)| <= eta.
```

Exact sampling means `eta = 0`. All statements use homology over one fixed
field, for example the pilot's `F_2`. Define the cubical superlevel complex

```text
C_t = union of closed grid cells Q for which min_{v vertex of Q} s_v >= t.
```

Vertices and edges are themselves cells in this definition. This is the
vertex lower-star construction applied to `-s`, with the parameter reversed;
its H0 connectivity is given by axis edges. It is the convention used by
`PeriodicCubicalComplex(vertices=-values, periodic_dimensions=[True,True])`,
not the constructor from top-dimensional cells. The pinned
[GUDHI 3.13.0 user manual](https://gudhi.inria.fr/python/3.13.0/cubical_complex_user.html)
describes extension from vertices by lower-star filtration; the
[periodic reference](https://gudhi.inria.fr/python/3.13.0/periodic_cubical_complex_ref.html)
distinguishes the two input conventions and their pairing accessors.

Let `D(f)` denote the ordinary H0 diagram of the sublevel filtration of `-f`,
equivalently the ordinary superlevel diagram with signs reversed. The
following are the external persistence results used here, in the exact
[Chazal–de Silva–Glisse–Oudot, arXiv:1207.3674v3](https://arxiv.org/pdf/1207.3674v3)
version:

- Theorem 2.22: a continuous function on a finite polyhedron has q-tame
  sublevel persistent homology. Here q-tame means that every structure map
  between strictly separated parameters has finite rank.
- Theorem 4.21: two q-tame modules that are epsilon-interleaved admit an
  epsilon-matching of their undecorated diagrams. Theorem 4.11 also states
  the equality of interleaving and bottleneck distances.

The torus is a finite polyhedron, so continuity suffices for this application;
no unverified Morse or distinct-critical-value assumption is needed. A finite
cubical or simplicial filtration is q-tame as well. Uniformly epsilon-close
functions have mutually shifted sublevel inclusions, which directly give an
epsilon-interleaving. Thus the cited matching theorem applies. The classical
[Cohen-Steiner–Edelsbrunner–Harer stability theorem, SoCG 2005, Section 3.1](https://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Stability.pdf)
gives the same sup-norm-to-bottleneck inequality for continuous tame functions
on a triangulable space; its continuity hypothesis alone is not a justification
for applying that version to the discontinuous cubical step function.

Diagrams below count finite points of strictly positive lifetime with
multiplicity. The torus has one essential ordinary H0 class, which is kept
separate. Finite-distance matching cannot pair an essential interval with a
finite one or the diagonal. No longest finite interval is removed.

## A direct bound for the cubical step function

For `x` in the relative interior of a grid cell `Q_x`, define

```text
c_h(x) = min_{v vertex of Q_x} s_v.
```

Use the unique cell whose relative interior contains `x`; in particular, use
an edge's own endpoints on an edge, and the supplied value at a vertex. Then
`{c_h >= t} = C_t` exactly. The boundary convention is necessary for this
identity.

If `M_x >= ||f_x||_infinity`, `M_y >= ||f_y||_infinity`, and
`G >= sup_x ||gradient f(x)||_2`, then

```text
||f-c_h||_infinity <= eta + min{h(M_x+M_y), sqrt(2) h G} = E_step.
```

Indeed, in a lift of each cell to the plane, every vertex is at most `h` away
in each coordinate and at most `sqrt(2) h` away in Euclidean distance. Apply
the fundamental theorem of calculus to `f(x)-f(v)`, add the nodal error, and
then take the minimum of the vertex values. The same estimates cover edges
and vertices. The set inclusions

```text
{f >= t+E_step} subset C_t subset {f >= t-E_step}
```

give an interleaving after parameter reversal, so the algebraic stability
result yields `d_B(D(f),D(C)) <= E_step` even though `c_h` is discontinuous.

This particular surrogate need not approximate quadratically. For
`f(x,y)=sin(2*pi*x/L)`, the first square to the right of `x=0` has exact
vertex minimum zero for sufficiently fine grids. At interior points with
`x` tending to `h`, the error tends to `sin(2*pi*h/L)`, which is of order
`h`. This example concerns the step function, not a lower bound on the
persistence-diagram error.

## PL interpolation on either diagonal orientation

Triangulate every square by one of its two diagonals. The choices may be
fixed in advance or may depend on the vertex values; shared square edges
are unchanged, so either choice in each square gives a conforming periodic
triangulation. Let `p_h` be the continuous affine interpolant of `s_v` on
its triangles.

Take finite certified bounds

```text
M_xx >= ||f_xx||_infinity,
M_yy >= ||f_yy||_infinity,
M_xy >= ||f_xy||_infinity,
H >= sup_x ||Hessian f(x)||_operator.
```

Then, for every such choice of diagonals,

```text
B_h = min{ h^2 H/4,
           h^2 (M_xx+M_yy)/8 + h^2 M_xy/4 },
||f-p_h||_infinity <= eta+B_h.
```

Here the operator norm is the Euclidean matrix operator norm. In particular,
`d_B(D(f),D(p_h)) <= eta+B_h`.

**Proof of the operator bound.** First use exact nodal values on the chosen
triangulation and write `x = sum_i lambda_i v_i` in a triangle. Taylor
expansion at `x`, with integral remainder and `sum_i lambda_i(v_i-x)=0`,
gives

```text
|sum_i lambda_i f(v_i)-f(x)|
  <= (H/2) sum_i lambda_i |v_i-x|^2.
```

In local coordinates on the containing square, each vertex coordinate is
either `0` or `h`. Therefore, for either diagonal and either triangle,

```text
sum_i lambda_i |v_i-x|^2
  = x_1(h-x_1)+x_2(h-x_2) <= h^2/2.
```

This proves the `h^2 H/4` bound. Interpolation of the nodal errors is a convex
combination, so its absolute value is at most `eta`, also when the chosen
triangulation depends on the supplied noisy values.

**Proof of the componentwise bound.** Let `b_h` be the bilinear interpolant
of the exact values on a square. Successive one-dimensional interpolation,
whose sup-norm operator has norm one, and the one-dimensional second
derivative error bound give

```text
||f-b_h||_infinity <= h^2(M_xx+M_yy)/8.
```

Label the corners `A=f(0,0)`, `B=f(h,0)`, `C=f(0,h)`, `D=f(h,h)` and put
`Delta=A-B-C+D`. The fundamental theorem of calculus twice gives

```text
Delta = integral_0^h integral_0^h f_xy(u,v) dv du,
|Delta| <= h^2 M_xy.
```

For diagonal `AD`, with normalized coordinates `u,v` in `[0,1]`, the
difference between the affine and bilinear interpolants is
`v(1-u) Delta` if `v<=u` and `u(1-v) Delta` if `u<=v`. Its absolute value
is at most `|Delta|/4`. Reflecting one coordinate gives the same estimate
for the other diagonal. Add the bilinear error and the nodal error. This
proves the asserted componentwise bound.

**Why the simplicial computation represents the PL function.** At threshold
`t`, let `K_t` contain precisely those simplices whose vertices all have
value at least `t`. Its inclusion into `{p_h>=t}` induces an isomorphism
on homology. To see this directly, in each simplex normalize the barycentric
coordinates belonging to vertices with value at least `t`, and set the other
coordinates to zero. Their total weight is positive at every point of
`{p_h>=t}`. The resulting point lies in the high-vertex face, and the straight
homotopy remains in the simplex's superlevel set because `p_h` is affine.
These maps agree on shared faces and fix `K_t`, giving a deformation
retraction. The inclusions `K_t -> {p_h>=t}` commute as thresholds change,
so their homology isomorphisms identify the persistence modules; the
retractions themselves need not commute across thresholds.

Thus a fixed-diagonal PL H0 calculation uses axis edges **and that declared
diagonal orientation**, with an edge entering at the smaller original
endpoint value. It is a different grid observable from axis-edge cubical H0.
For example, in one square with `A=D=1`, `B=C=0`, the axis graph has two
components above zero, while adding diagonal `AD` connects the two high
vertices at level one. Equality of those two filtrations must not be assumed.

## A quadratic bound for the cubical filtration itself

For each square choose a vertex attaining the **minimum supplied value**
among its four corners, breaking ties by any deterministic rule, and draw
the diagonal from that vertex to its opposite corner. Let `p_h^C` be the
continuous PL interpolant on this triangulation.

The upper-star simplicial subcomplex of this triangulation and `C_t` have
the same underlying subset of the torus at every threshold `t`. On each
square, both triangles and the added diagonal contain the chosen minimum
vertex. They therefore enter at the square minimum, exactly when the full
cubical square enters. Before then, only the original high vertices and
axis edges are present. This verifies the equality cell by cell, including
ties and periodic seams.

The preceding deformation-retraction argument consequently identifies
`D(C)` with `D(p_h^C)`. Apply the PL interpolation estimate on this particular
triangulation to obtain

```text
d_B(D(f),D(C)) <= eta+B_h.
```

This is a proof device for the already declared vertex cubical filtration;
it does not require changing the experiment to an adaptive triangulation.
It also does not identify cubical H0 with the PL filtration on a fixed
diagonal. Both approximations can instead be compared separately to `f`.
Together with the direct step bound, a permissible cubical endpoint-error
bound is `min{E_step, eta+B_h}`. The argument is deterministic and supplies
no claim that a particular pilot grid already resolves its shortest bins.

## Derivative majorants from the retained Fourier coefficients

Write a specified real finite Fourier field as

```text
f_K(x) = c_0 + sum_{k in R_K} r_k cos(omega_k dot x - phi_k),
omega_k = (2*pi/L) k,
r_k >= 0,
```

where `R_K` has one representative from each retained pair `{k,-k}`. No
probabilistic assumption is needed for the following pathwise bounds:

```text
M_x  = sum_k r_k |omega_k,x|,
M_y  = sum_k r_k |omega_k,y|,
G    = sum_k r_k |omega_k|,
M_xx = sum_k r_k omega_k,x^2,
M_yy = sum_k r_k omega_k,y^2,
M_xy = sum_k r_k |omega_k,x omega_k,y|,
H    = sum_k r_k |omega_k|^2.
```

Termwise differentiation and the triangle inequality prove the component
bounds. Each Hessian summand is a scalar of absolute value at most `r_k`
times `omega_k omega_k^T`, whose operator norm is `|omega_k|^2`, proving
the bound for `H`. These are convenient, sometimes conservative, finite
sums. A smaller certified derivative bound can replace them.

For the model declared by `field_grid`, put

```text
D_64 = (sum_{j=-64}^{64} exp(-2*pi^2*j^2/L^2))^2,
w_k  = exp(-decay*2*pi^2*|k|^2/L^2) / D_64,
r_k  = sqrt(2*variance_factor*w_k) * sqrt(A_k^2+B_k^2).
```

The actual experiment uses `decay=variance_factor=1`; the other values are
model controls. The zero mode has no effect on any derivative or unrestricted
finite lifetime. Equivalently, if the specified field is defined by stored
complex Fourier coefficients `z_k`, use `r_k=2|z_k|` and record that
definition. A field with rounded stored coefficients and a field with exact
exponential weights are different mathematical inputs; their difference must
be included if comparing them.

The requirement `n>2K` in the FFT generator prevents its retained modes
from colliding in the coefficient array. The interpolation theorem itself
only requires actual samples with the stated nodal error. A claim that an
FFT array contains those samples still needs the generator convention and
its numerical error bound.

For two nested finite mode sets with identical retained coefficients and
normalization, their pathwise uniform difference is bounded by the sum of
`r_k` over the omitted finite set. Derivative differences admit the same
frequency-weighted sums. This does not enclose the infinite Gaussian field:
the pilot's omitted spectral variance and derivative-variance sums are sums
of deterministic squared weights, not realized amplitude sums. They neither
bound a realization's sup norm nor enclose modes beyond 64. A bound against
the intended infinite field additionally needs a justified uniform tail
bound and any correction for its different normalization. If such a bound
is `tau`, it can be added to the spatial error by the triangle inequality;
this note does not supply `tau`.

## Endpoint-safe lifetime-bin inequalities

Let `P` be the target diagram and `Q` an approximation, with an actual
epsilon-matching, and put `delta=2*epsilon`. This follows from the stated
interleavings and the cited algebraic stability theorem. For a set `J` of
positive lifetimes, let `N_P(J)` and `N_Q(J)` count finite positive diagram
points with multiplicity. They do not count essential intervals or the
diagonal. An expression `[u,v)` with `u>=v` denotes the empty set here.

A matched pair of finite intervals has lifetime difference at most `delta`,
because each of its two endpoints moves by at most `epsilon`. A finite
interval matched to the diagonal has lifetime at most `delta`; equality
is possible. For `I=[a,b)`, `0<a<b`, these facts imply

```text
N_Q([a+delta,b-delta)) <= N_P([a,b)).
```

If **`a>delta`**, they also imply the clean upper bound

```text
N_P([a,b)) <= N_Q([a-delta,b+delta)).
```

**Proof.** Every bar in the first, contracted interval has lifetime greater
than `delta`, so it is matched to a finite target bar. Its target lifetime
is at least `a` and strictly less than `b`. Distinct bars have distinct
partners, proving the lower inequality. If `a>delta`, every target bar in
`[a,b)` also has a finite partner; that partner's lifetime is at least
`a-delta` and strictly less than `b+delta`. Injectivity proves the upper
inequality. Inclusion of the lower endpoint and exclusion of the upper
endpoint in every displayed bin are deliberate: these statements remain
valid for exact equalities at bin boundaries.

For `a<=delta`, a universally valid upper bound is instead

```text
N_P([a,b))
  <= N_Q([a-delta,b+delta) intersect (0,infinity))
     + N_P([a,b) intersect (0,delta]).
```

The added term bounds target bars in the bin that may be unmatched; it is
not generally known from the approximate diagram. Thus this inequality can
be uninformative. The strict threshold in the clean formula is necessary:
a diagram consisting of a single bar of lifetime exactly `delta>0` is at
bottleneck distance `delta/2` from the empty diagram, yet its count in
`[delta,2*delta)` is one. No finite approximate bar witnesses that count.
The same statements hold with `P` and `Q` exchanged.

Even above the diagonal threshold, bottleneck distance alone does not give
a small error in a sharp histogram bin. Arbitrarily many matched points
can lie just on opposite sides of a bin boundary after an arbitrarily small
endpoint perturbation. The expanded/contracted counts, or a separate bound
on mass near the boundaries, are essential. An empty contracted interval
provides only the lower bound zero.

For this experiment, divide the inequalities by `L^2`. For `m` fields with
possibly different certified errors `epsilon_i`, sum the individual
inequalities and divide by `m*L^2`; use their individual
`delta_i=2*epsilon_i`. A common envelope `epsilon=max_i epsilon_i` permits
common expanded and contracted bins, with the clean upper bound when
`a>2*epsilon`. These statements bound the empirical mean of the exact
realization counts. They are not confidence intervals for an ensemble
expectation and do not convert dependent bars into independent replicates.

## What an evaluated certificate would still require

A rigorous application to retained numerical output must specify the exact
finite field and enclose its derivative majorants, nodal error, and arithmetic
used in the bounds. For endpoint-sensitive bin tests, use exact arithmetic
for represented values or outward enclosures; an interval overlapping a bin
boundary cannot be assigned a certified side by its rounded midpoint.
Any persistence implementation or subsequent arithmetic error requires its
own justification. Agreement between two float64 programs and agreement
between coupled grids are useful controls, but do not supply these
enclosures.

The current pilot therefore retains its original disposition: inconclusive
at the tested resolutions. The estimates above make one spatial comparison
precise. They do not settle Fourier truncation, sampling uncertainty, the
finite-lifetime validity range of the asymptotic formula, or its remainder.
Their derivation uses standard stability and elementary interpolation; no
novelty or change in scientific acceptance is asserted.
