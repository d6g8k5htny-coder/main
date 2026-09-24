# Source-bound SIDE24 point conditioning

This author-produced candidate connects the generic interval moment certificates
to the normalized SIDE24 Gaussian field at one spatial point. It changes no
scientific status or obligation, supplies no spatial cover or all-small-r result,
and earns zero organizational independence credit. The density/window factor
and H3 normalizer are not included. `field_certified` remains false: this is a
bounded point calculation, not the complete RN field certificate.

Prep order is this note, then [RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md),
then [RN_SIDE24_CELL.md](RN_SIDE24_CELL.md). The ABSENT / REFUSED / OPEN walls
for this step are under [Prep path and honesty walls](#prep-path-and-honesty-walls).

## Precise statement

Use the centered real Gaussian field with covariance

\[
K(x)=k(x_1)k(x_2),\qquad
k(s)=\frac{\sum_{j\in\mathbb Z}e^{-(s+24j)^2/2}}
                 {\sum_{j\in\mathbb Z}e^{-(24j)^2/2}}.
\]

Set \(r=1/20\), \(b=6/5\), \(\ell=r^3/6=1/48000\),
\(M=(-1/40,0)\), \(S=(1/40,0)\), and \(y=(1,1)\). The nine
conditioning coordinates, in order, are

\[
G=(f(M),f_x(M),f_y(M),f(S),f_x(S),f_y(S),f(y),f_x(y),f_y(y)).
\]

Condition on
\(G=(b,0,0,b-\ell,0,0,b-\ell/2+t,0,0)\) for every
\(t\in[-1/96000,1/96000]\). This means the continuous Gaussian
conditional law defined by the nonsingular covariance of \(G\), including the
endpoints; it is not a conditioning event assigned positive probability.
Each Hessian has the coordinate order \((xx,yy,xy)\), in original field units,
and determinant \(h_{xx}h_{yy}-h_{xy}^2\).

The reconstructed candidate gives uniform conditional moment caps for degrees
four at M, four at S, and two at y. Their approximate display values are
\(4.79675171\cdot10^{-5}\), \(3.88013635\cdot10^{-5}\), and
\(1.24276873\), respectively; the candidate stores the exact rational caps.
Conditional Hölder, with exponents 4, 4, and 2, then gives the exact rational
bound

\[
\mathbb E\left[
  |\det H_M\det H_S\det H_y|\,1_E\mid G
\right]
\leq U_M^{1/4}U_S^{1/4}U_y^{1/2}
\leq \frac{7322236}{10^9}=0.007322236
\]

for every mark in that interval and any intersection \(E\) of Hessian type
indicators. Dropping the indicators only increases this nonnegative expectation.
No independence among the three Hessians is assumed. This is the conditional
determinant factor only; multiplying by unproved external factors is outside
this statement.

## Source custody

The mathematical reading copy is
`drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md`.
Its Drive ID is `1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5`, and its SHA-256 is
`ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`.
The sibling archive `RN5_REPAIR_AND_ERRATUM_BUNDLE.zip` has SHA-256
`28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e`.
The adapter hashes these bytes and the following members before deriving a law:

| Member | SHA-256 | Relevant source |
| --- | --- | --- |
| `closure_round2/rn_field.py` | `d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8` | Normalized kernel, derivative sign and coordinate construction |
| `round5/near_moments.py` | `03c6e35c426eee8ed91139b93f0ada4e96c7293e5afb4176677850596fbfca6e` | Nine pins, affine mark, Hessian order and moment formula |
| `round5/verify_round5.py` | `614daf280f5037757d261281b73ea66edb350da4619066ee56ea5a4216b7737e` | Direct 18-coordinate assembly and conditioning/target partition |

These files are read as data. No archive member or other frozen executable is
imported or run. Hash agreement establishes byte identity; the mathematical
association between those bytes and this adapter remains a reviewable author
derivation. The adapter does not use the different, truncated `C_KERN` constant
from older carriers.

## Kernel enclosure argument

For probabilists' Hermite polynomials,
\(D^n e^{-u^2/2}=(-1)^n\mathrm{He}_n(u)e^{-u^2/2}\).
The coefficient recurrence is \(\mathrm{He}_{n+1}=u\mathrm{He}_n-n\mathrm{He}_{n-1}\),
starting from 1 and u. The adapter uses orders zero through four, the full range
needed for covariance between values, first derivatives, and Hessians.

One common exact period translation reduces a displacement interval of width
at most 12 to an interval inside \([-18,18]\). This operation preserves the
entire interval. Let its maximum absolute endpoint be \(B\leq18\).
The images \(j=-1,0,1\) are evaluated using certified rational interval
polynomials and the existing certified interval exponential. The remaining
images are not discarded.

Write \(C_n\) for the sum of the absolute coefficients of \(\mathrm{He}_n\).
For \(|u|\geq1\), \(|\mathrm{He}_n(u)|\leq C_n|u|^n\). Thus the pair of
images indexed by \(\pm j\), for \(j\geq2\), is bounded in absolute value by

\[
a_j=2C_n(24j+B)^n e^{-(24j-B)^2/2}.
\]

The consecutive ratio is

\[
\frac{a_{j+1}}{a_j}=
\left(\frac{24(j+1)+B}{24j+B}\right)^n
e^{-((24(j+1)-B)^2-(24j-B)^2)/2}.
\]

Both factors are nonincreasing in j, so their value \(q\) at j=2 bounds
every subsequent ratio. The code explicitly proves \(q<1\) using the upper
endpoint of its enclosure. Hence the complete omitted tail is at most
\(a_2/(1-q)\). Every derivative numerator gets the symmetric interval with
that radius. The denominator is the positive interval

\[
Z\in1+2e^{-288}+[0,T_0(0)],
\]

where \(T_0(0)\) is the same complete tail bound at B=0. Division is allowed
only after its positive lower endpoint is established. At exact multiples of
24, \(k=1\) follows from identical numerator and denominator; odd derivatives
vanish by evenness. No other derivative is replaced by a planar value.

All arithmetic uses `research.interval` and exact rational endpoints. Outward
dyadic rounding limits expression growth while preserving containment.
Precision controls tightness, not validity. Input rationals above 4096 bits,
unsupported orders, overly wide intervals and unsupported precision requests
are refused. The default point calculation uses 192-bit rounding; diagnostic
tests use 512 and 1024 bits to resolve very small torus corrections.

## Joint covariance and conditioning argument

For \(d=p-q\), the covariance convention is

\[
\operatorname{Cov}(D^\alpha f(p),D^\beta f(q))
=(-1)^{|\beta|}D^{\alpha+\beta}K(d).
\]

The sign comes from differentiating the second spatial argument. The code
assembles all 18 coordinates directly and preserves this sign. The first nine
are G; the next nine are the Hessians at M, S and y in that order.

The joint PSD premise comes from the field, not from treating every independent
entry in an interval box as a coherent covariance matrix. In one dimension,
the periodized Gaussian has strictly positive Fourier weights proportional to
\(e^{-(2\pi m/24)^2/2}\). Their product gives the two-dimensional weights.
Gaussian decay makes all differentiated sums absolutely convergent. For any
finite linear combination of derivative evaluations, its covariance quadratic
form is a sum of these positive weights times squared absolute Fourier linear
forms. It is therefore nonnegative. Pairing conjugate frequencies gives the
same real Gaussian covariance. Normalization divides by positive Z on each
axis and preserves this argument.

Partition that coherent covariance as
\(\left(\begin{smallmatrix}A&C^T\\C&B\end{smallmatrix}\right)\).
`research.rn.conditioning` encloses an LDL factorization of A, requires every
pivot's lower endpoint to be positive, and solves the corresponding systems.
With zero prior means, the exact conditional law is

\[
\mu(t)=CA^{-1}(g_0+t g_1),\qquad
\Sigma=B-CA^{-1}C^T.
\]

Covariance is constant in the mark and the mean is affine, so interval
intercept/slope arrays enclose the full mark family without sampling. The
adapter also proves positive pivots for each 3-by-3 marginal Hessian covariance
box. It retains the joint covariance and gain for inspection; it imposes no
unnecessary positive-definiteness condition on the whole Hessian block.
A failed pivot check is an inconclusive sufficient calculation, not a proved
counterexample to the Gaussian law.

The generic [moment certificate replay](RN_CERTIFICATES.md) then verifies the
polynomial coefficients and complete Bernstein mark partition. Its producer
and replay recurrence differ, but they share exact interval primitives. The
source adapter is reconstructed by the report checker, so a shared adapter bug
could survive reconstruction. Explicit polynomial, normalization, sign and
omitted-image negative controls challenge that shared trust; they do not supply
organizational independence or substitute for review of the written argument.

## Reproduction and API

`point_laws(point=(Fraction(1), Fraction(1)), bits=192)` returns the source
binding, raw 18-coordinate covariance, nine pin arrays and coordinate order,
conditioning gain, joint affine mean/covariance, positive pivot intervals, and
three `FamilyLaw` objects under `laws['M']`, `laws['S']`, and `laws['y']`.
All three identify the same joint conditioning law and full mark domain.
`nine_pin_blocks`, `kernel_derivative`, and `covariance` expose the lower-level
construction. Other admitted rational points are calculation inputs, not an
assertion of spatial uniformity; colliding or unresolved pins are refused.

```bash
python tools/rn_side24_check.py
python -m pytest -q -p no:cacheprovider tests/test_rn_side24.py
```

The report checker compares the committed candidate with a fresh source-bound
reconstruction and replays its three portable moment witnesses. Its simple
rational Hölder bound is rounded upward and checked against the exact caps.
The source test suite checks all derivative orders, period/evenness identities,
covariance signs, complete pin/domain metadata, conditioning and marginal
pivots, source corruption and resource refusals. At 512 bits it excludes both
the planar derivative variance and an omitted normalizer. At 1024 bits it
detects removing the omitted positive image correction at displacement 12.
Ordinary 100-decimal-place agreement would not detect the first-shell effect
on the gradient variance at zero, which is about \(9.65\cdot10^{-123}\).

## Prep path and honesty walls

Read the local SIDE24 notes in this order. Each later note consumes the
earlier one and does not enlarge its scope. The lane pointer is
[RESEARCH_MAP.md](RESEARCH_MAP.md) §3. The matching commands are in
[RESEARCH_EXECUTION.md](RESEARCH_EXECUTION.md).

| Step | Note | What it prepares | What it does not supply |
| --- | --- | --- | --- |
| 1 | this note | one-point conditional determinant factor at \(y=(1,1)\) | density/window factor; H3 normalizer; spatial cover |
| 2 | [RN_SIDE24_DENSITY.md](RN_SIDE24_DENSITY.md) | density/window factor at the same point, divided by an imported H3 floor | spatial integral; H3 reproof; annulus bound |
| 3 | [RN_SIDE24_CELL.md](RN_SIDE24_CELL.md) | one declared rectangle, using steps 1–2 | complete annulus cover; remote-budget assembly; all-small-r theorem |

**ABSENT** from this point note: the density/window factor, the H3 normalizer,
any spatial cover, and any all-small-r result. Those objects are not invented
here. `field_certified` stays false.

**REFUSED**, as already stated above: unsupported derivative orders, overly
wide intervals, unsupported precision requests, input rationals above 4096
bits, and colliding or unresolved pins. A failed pivot check stays an
inconclusive sufficient calculation, not a proved counterexample.

**OPEN:** this note changes no scientific status and no obligation.
`D3-LEMMA-RN-UNIF` stays OPEN. `OBL-H5-JETMOD` stays OPEN. `lemma_closed`,
`certified_C_H`, and `prizes_solved` stay false. Replaying
`tools/rn_side24_check.py`, and a green CI run, are engineering checks. They
are not discharge. Organizational independence credit stays zero.
