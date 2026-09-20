# Twelve mathematical candidates: results and application gaps

The immutable delivery is
`research/campaigns/q0_twelve_20260920_v1.zip`, SHA-256
`7a36b7dbaa45b80df9e75a2c063c141440a329c1fb66b85ba828a7fe6517dc8f`.
Each project retains its `PROOF.md`, exact `result.json`, source identities,
checker, negative controls and `RUN.json`. This overview distinguishes actual
field estimates from conditional lemmas and synthetic demonstrations. It does
not change any canonical status, restore Theorem B, close q0, or compose the
two-dimensional upper/lower and three-dimensional lifetime tracks.

The field estimates below use the normalized SIDE24 covariance
`K(x,y)=k(x)k(y)`, where
`k(s)=sum_j exp(-(s+24j)^2/2)/sum_j exp(-(24j)^2/2)`.
All lattice sums include certified omitted tails. Torus derivative moments
are not silently replaced by planar Gaussian moments. For the H3 results,
`M=(-r/2,0)`, `S=(r/2,0)`, `b=6/5`, and the six observations are
`(fM,fxM,fyM,fS,fxS,fyS)=(b,0,0,b-r^3/6,0,0)`.

1. **H3 conditioning and small-radius limit — `h3_uniform`.**
   The transformed pin vector
   `V=(fM,fxM,fyM,(fxS-fxM)/r,(fyS-fyM)/r,
   (fS-fM)/r^3-(fxS+fxM)/(2r^2))`
   extends in Gaussian L2 to zero. Its covariance satisfies
   `Cov(V)>=81/6400 I` throughout `0<=r<=1/20` on this fixed axis.
   For the typed normalizer
   `Z(r)=E[|det H_M det H_S| 1{M maximum,S saddle}|pins]`,
   `Z(r)/r^2` tends to `E[Q^2 1{Q<0}]`, with
   `Q~N(-(6/5)m2,m4-m2^2)`, `m2=-k''(0)`, `m4=k''''(0)`.
   The limit lies in the exact decimal interval
   `[3.23097853528700494809,3.23097853528700494810]`.
   Thus `Z(r)>=3r^2` holds for some sufficiently small radius, but that
   radius is **not numerically certified**. No positive unscaled constant
   floor can persist to zero. Neither the raw pins nor the limiting soft
   Hessians inherit the displayed covariance floor; all-angle control remains
   separate.

2. **H3 ceiling — `h3_ceiling`.**
   At exactly `r=1/20`, the same law gives
   `Z(r)/r^2<=3.230674239349`, hence `Z(r)<=0.0080766855983725`.
   These are upward rational displays. Gaussian regression and truncated
   normal moments evaluate both factors of Cauchy–Schwarz on the common
   necessary half-space `fyy(M)<0`; no determinant independence is assumed.
   A separate exact witness shows that a historical binary64 tail conversion
   can replace a positive Gaussian tail by zero. This refutes that enclosure
   step, not its final printed ceiling. An all-radius ceiling still needs a
   uniform conditional-law enclosure with outward tail arithmetic.

3. **LPW amplitude and field norms — `lpw_amplitude`.**
   Independent standard cosine/sine coefficients give Rayleigh amplitude
   `rho=sqrt(xi^2+eta^2)`, with `E rho^4=8` and
   `E rho=sqrt(pi/2)`. The old absolute-sum moment is instead
   `E(|xi|+|eta|)^4=12+32/pi`. Full-lattice shell bounds and the weights
   `max(1,|k1|,|k2|)^p` give
   `E||f||C3^4<4727655047` and `E||f||C4<536.605087`.
   Here `||f||Cp=max_{|alpha|<=p} sup|D^alpha f|`.
   These are unconditional, spatially uniform sufficient budgets, not actual
   moment values or conditional norm bounds. A sum-of-derivatives norm needs
   an explicit conversion.

4. **Two source-defined jets — `jet_definition`.**
   The unconditioned covariance of
   `((fxS-fxM)/r,(fyS-fyM)/r)` is diagonal and has smallest eigenvalue
   greater than `1599/1600` on `0<=r<=1/20`, with the quotient interpreted
   by its L2 extension at zero. Integral differences retain normalized image
   corrections without division by an interval containing zero. The full
   historical 24-jet list, powers, conditional covariance and cover map remain
   insufficiently defined. Conditioning does not preserve an unconditional
   covariance lower bound.

5. **LPW conditional constant — `lpw_headline`.**
   Whole-box Gaussian density and conditional regression bounds yield the
   frozen candidate coefficient
   `c=13/468574870385996070912000 > 2.77e-23`, for
   `0<r<=1/356352`. The claimed inequality is `1-q>=c r^3` under the
   candidate's stated LPW coordinate, topology/weight and Palm-event
   interfaces. The proof uses a density floor `1/1000`, conditional Markov
   bound and the exact budget `16 delta+8Kr0=3/64<1/16`, with
   `delta=1/1024`, `K=1392`. It derives conditional norms by Gaussian
   regression rather than substituting unconditional amplitude estimates.
   The later source audit supports the deterministic interfaces at the
   **elder-pairing** scope described below; it does not silently enlarge the
   frozen result's scope. For the separate historical fraction,
   `6.238e-44` is downward-safe and `6.239e-44` is not.

6. **Persistence multiplicity — `tb_multiplicity`.**
   Under a finite, strictly ordered binary H0 event interface, directed elder
   pairs are in bijection with finite ordinary bars. Essential classes and
   loop attachments are excluded; repeated lifetimes retain multiplicity.
   A separate boundary-matrix rank calculation checks the whole persistence
   rank invariant. Conditional Borel chart hypotheses give a measurable
   selected point process dominated by the all-typed process. The actual
   Gaussian field's event reduction, good locus and measurable atlas remain
   to be established. The finite graph showing a selected nonadjacent pair
   is not a realization theorem for this field.

7. **Contact-to-lifetime transport — `tb_contact`.**
   For an ordered maximum–saddle pair with `0<r<r_c<12`, the spatial
   Jacobian is `r`, the height Jacobian is `r^3/6`, the corrected six-pin
   density contributes `r^-5`, and the Hessian product contributes `r^2`.
   The last three factors have product `1/6`; the spatial `r dr` remains
   in the reference measure. With `ell=kappa*r^3/6`,
   `r dr=(6^(2/3)/3) kappa^(-2/3) ell^(-1/3) d ell`, retaining
   `kappa>6ell/r_c^3`. Full directed angle requires no half factor, and
   ordinary torus area contributes `576` once under stationarity.
   These exact identities require the declared Kac–Rice/Palm representation;
   they do not supply persistence identification, a contact limit or selection
   convergence.

8. **Full-mark tails — `tb_tails`.**
   Supplied uniform pin covariance floor/ceiling, cross-covariance and residual
   trace bounds imply a Gaussian quartic majorant proportional to
   `kappa^(-2/3)[1+(|b|+kappa)^4] exp(-c(b^2+kappa^2))`.
   Explicit integrated errors cover small kappa, large kappa and large birth
   height. The numerical examples use hypothetical constants. Actual coherent
   joint Gaussian bounds for every consumed radius and angle, and the selected
   contact limit, are still missing. Fixed-axis results or a finite grid do
   not supply those hypotheses.

9. **Weighted WP inequality — `wp_event`.**
   Under the base conditional Gaussian law Q, weight W and normalizer
   `Z=E_Q W`, Cauchy–Schwarz requires both square roots:
   `sqrt(E_Q[W^2 D^2|G]) sqrt(Q(event|G))`.
   Conditioning on height first permits the linear window factor
   `B_y Q(X in I|G)`, where
   `B_y^2>=ess sup_{x in I} E_Q[W^2 D^2|G,X=x]`.
   The quartic/Bernstein numerical example has `W=1`; it does not certify
   the source's weighted moment. A cubic integrated loss requires uniform
   spatial control of `p_G B_y/(Z sigma sqrt(2pi))`, including pin
   neighborhoods and shrinking conditional variance. The actual event
   inclusion determines whether an absolute loss or a conditional fraction
   is needed.

10. **Lambda derivative replacement — `lambda_uniform`.**
    Exact smooth counterexamples show that finitely sampled jets and grid
    stability alone cannot bound an off-grid third derivative, and that a
    center-gradient/Hessian fallback needs its Taylor remainder. Valid
    replacements use genuine ingredient C3 bounds, or a genuine D4 bound
    giving `sup||D3 f||<=sqrt(2)(Q+B4 h)` on a grid with its complete stencil
    halo. A positive typed box also gives a direct positive integral floor.
    Actual Lambda ingredient bounds, variance and negative-trace/type margins,
    clipping interfaces and finite-radius transfer remain missing. The
    synthetic Gaussian exercise is not an actual Lambda certificate, and the
    generic counterexamples do not prove actual H-B3 false.

11. **Gaussian tube — `gaussian_tube`.**
    For the continuous Gaussian components `(E,h Ex,h Ey)` on a deterministic
    square of side D, suppose their standard deviations are at most v and
    their canonical metrics at most L times max-norm distance. With mean
    mismatch at most a, infinite dyadic chaining gives threshold
    `a+(v+LD)sqrt(2(u+4))+5LD` and failure probability at most `2exp(-u)`.
    Escape also needs a perturbation ball wholly inside the designated
    third-saddle success event. Actual geometry, clearance scaling, exceptional
    events and weighted-Palm transfer remain open inputs. The all-small-r
    numerical family is hypothetical. The exact counterexample to source D2
    corrects a reversed probability implication; fixed far-end slack alone
    does not establish a shrinking failure rate.

12. **Count and eta budgets — `bonferroni_eta`.**
    For an integer count N under one law with `{N>=1}` contained in true
    failure, `E N>=m>0` and `E[N(N-1)]<=S` imply
    `P(failure)>=(2km-S)/(k(k+1))`, optimized by `k=floor(S/m)+1`.
    A positive cubic first moment and an `O(r^3)` factorial moment can
    therefore retain a positive cubic coefficient; `o(r^3)` preserves the
    full first coefficient by this bound. Actual counts, event inclusion and
    weighted moment/loss estimates are not supplied. The exact identity
    `p=a q_adj+eta` requires an eta debit for the proxy `1-a q_adj`, but
    not for a direct inclusion into true elder failure or an adjacent-failure
    mass already contained in it.

The LPW event boundary is essential. The archive's `LPW_INTERFACE_AUDIT.md`
checks the original six-pin Fourier profiles, Taylor lift and deterministic
path argument separately from project 5's frozen custody statement. The path
reaches a point above the maximum's birth height while staying above the
proposed saddle level, so that maximum joins an older superlevel component
before that level. Under a measurable, well-defined elder-persistence
interpretation, this excludes the saddle as its death partner. The audit
retains a path clearance `103/3840` and typed pair weight at least `65r^4`.
The selected-gradient-branch convergence event in GP-DER-118 is a different
adjacency event. Equal six pins, determinant weight and Palm law do not make
the events equal. Applying the LPW coefficient to that event, an
adjacency-conditioned q, or a differently defined canonical q0 needs an exact
event identity and any required eta estimate. Gaussian persistence genericity
is not established merely by this path computation.

Coordinate adapters must also be explicit. Existing RN Hessian triples use
`(xx,yy,xy)`; the WP quadratic-form helper uses `(xx,xy,yy)`. Contact transport
uses raw order `(fM,fS,ftM,fsM,ftS,fsS)` in a rotating frame, whereas RN pins
are grouped by point. The H3 vector V is different from the symmetric contact
vector and the LPW Hermite vector, whose endpoint is
`(f,fx,fy,fxx/2,fxy,fxxx/6)`. Covariance eigenfloors depend on these exact
scales and transformations. Sharing a limiting jet does not identify two
finite-radius matrices.

The nearest substantive targets, ranked by the missing mathematics, are:

1. Quantify the H3 conditional-law transport and typed-boundary error to turn
   the positive limit of `Z(r)/r^2` into an explicit radius interval; retain
   the fixed-axis restriction until angle dependence is bounded.
2. Build the actual weighted conditional moment envelope for the WP
   conditioning-first inequality, together with spatial bounds near the pins.
   Identify the failure event before choosing its rate budget.
3. Enclose the actual Lambda ingredients on one positive-area typed box,
   proving variance, density and sign margins. This is a smaller positivity
   target than an unsupported global derivative grid.
4. Establish coherent all-angle covariance bounds and selected-contact
   convergence for the tail theorem, alongside the actual measurable H0
   event reduction. Neither component substitutes for the other.
5. Construct an actual third-saddle success tube and a quantitative clearance
   bound before applying Gaussian chaining. Covariance estimates alone do
   not establish the required dynamic event.

Using the repository's Python 3.11 runtime, the portable replay command is:

```sh
python tools/twelve_project_check.py
```

Replay must preserve the archive, check its manifest and source/dependency identities,
and write working output outside the immutable payload. Consult the embedded
proofs and run recipes for each result's precise scope. A dependency mismatch
requires an explicit reconciliation, not silent replacement of expected hashes.
The wrapper supplements the original 24-entry dependency list with the RN5
repair note and the original `docs/OPEN_PROBLEMS.md` snapshot, both already
pinned by individual checker bindings. These inputs remain byte-exact;
current progress is documented here. A minimal relocated checkout must replay
all twelve reports without host-path normalization. The archived author tests
retain host-specific defaults and run separately when requested; portable CI
uses the exact candidate replays and the wrapper's own controls. Archived
subtests are not silently added to the master JUnit count.

`SOURCE_BOUNDARY.md` records that a short inert excerpt of a quarantined/history
Lambda script was exposed during intake before its identity was checked. That
script is excluded from the delivery and all mathematical evidence; the
incident is retained, not erased by the corrected source reader. Native DOCX
export hashes, derived-reading hashes and marked frozen-body hashes remain
distinct. This delivery and its reviews are source-exposed, same-provider work
with zero organizational independence credit. Passing replay checks supports
the recorded computations; it is neither formal proof-assistant validation
nor external acceptance. `original_prize_closed` remains false.
