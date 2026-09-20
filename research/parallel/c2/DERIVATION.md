# One SIDE24 c₂ band: exact source binding and cancellation-free enclosure

Author-side candidate, 2026-09-20. Target `PARALLEL-H5-C2-BAND-20260920-v1`;
R17 claim `4fa1fee4-fcb4-4a53-b337-4c88ab80302c`. Independent review pending;
organizational independence credit 0. No scientific status is changed.

The new result is a whole-band enclosure for **one** source-defined scalar:

\[
  0.499687630167653 \le c_2(r) \le 0.499843785542625,
  \qquad 7071/200000\le r\le 1/20.
\]

The two displayed decimals are exact rationals rounded outward from the
machine result. The enclosure width is strictly below
`78093/500000000 = 0.000156186`. Exact rational endpoints and every intermediate
tail/normalizer are in `candidate.json`. This is an arithmetic
enclosure with a written author proof; its existence is not independent
acceptance or discharge of `OBL-H5-JETMOD`.

## 1. Exactly what is bound to source

`H5_PROMOTE.md` §3(ii) defines
`c₂(r) = (Var f(M) − Cov(f(M),f(S)))/r²`.
Thus the raw jet is `J = Var f(M) − Cov(f(M),f(S))` and its source power is
`p_J = 2`. That document's table supplies the literal values `0.035355` and
`0.05`. We choose their exact rational interval as this candidate's input.
We do not equate `0.035355` to `0.05/√2`, nor claim these two literals give
the authoritative full program band partition.

Source custody:

| Source | Exact identity |
|---|---|
| H5 archive | Drive `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`; SHA `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b` |
| `K3_SIDE24_LB/UPPER2D/H5_closure/H5_PROMOTE.md` | 6,936 bytes; SHA `7258c84a7720704e577f341252aa809888735d9f8ace1674429e004da6c6172a` |
| RN5 archive | Drive `1g5UP_KYlvPmx7LFwjGk6qK5QZLnL1hLj`; SHA `28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e` |
| `closure_round2/rn_field.py` | 16,701 bytes; SHA `d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8` |

The archive hashes and member hashes were checked before copying the two
members under `sources/`. No source was executed. Exclusion metadata was read
before extraction; neither member matches a `Q-R17-H5` exclusion. The
historical H5 numerical certification assertions are not inputs here. In
particular, no excluded signed odd-frequency or missing-denominator-tail
certificate is consumed or restored.

RN5 `rn_field.py` lines 17,21–23 define `R=1/20`, derivative indices
`[(0,0),(1,0),(0,1)]`, and the raw observation order at `(-R/2,R/2)`.
Lines 66–85 give the normalized periodized Gaussian kernel, retaining its
infinite image tail. Lines 117–124 give the covariance derivative sign and
separable x/y factors. The checker verifies exact source hashes and uses AST
comparison to bind the derivative order and pin order without executing code.

**New mathematical extension:** the source implementation is fixed at
`R=1/20`. This candidate defines the same normalized field and replaces the
separation parameter by every positive `r` in the chosen band. This is a new
author derivation, not a claim that RN5 previously certified variable r.

## 2. Six observation coordinates and actual displacement map

Let `M=(-r/2,0)` and `S=(r/2,0)`. The six observations are

`[f(M), f_x(M), f_y(M), f(S), f_x(S), f_y(S)]`.

Each covariance displacement is `(c_i-c_j)r` in the x coordinate and zero
in y, with `c_i,c_j ∈ {-1/2,1/2}`. Consequently its x coefficient is exactly
`-1`, `0`, or `1`. For the negative coefficient, the band box is `[-b,-a]`,
not `[-a,-b]`. All 36 ordered covariance entries, summed derivative orders,
and second-argument signs are explicitly emitted by `geometry()`.

For derivative multi-indices α,β the covariance is

\[
 \operatorname{Cov}(D^\alpha f(p),D^\beta f(q))
 =(-1)^{|\beta|} k^{(\alpha_1+\beta_1)}(p_1-q_1)
                    k^{(\alpha_2+\beta_2)}(p_2-q_2).
\]

This binds geometry and the six observations only. These six observations
are not asserted to enumerate the full 24-jet obligation.

## 3. Exact two-dimensional normalization reduces this scalar to one axis

Write `φ(u)=exp(-u²/2)` and

\[
 S(s)=\sum_{j\in\mathbb Z}\phi(s+24j),\quad Z=S(0),\quad k(s)=S(s)/Z.
\]

The complete two-dimensional periodized Gaussian sum factors, absolutely,
as `S(x)S(y)`; its complete normalizer is `Z²`. Thus
`K(x,y)=k(x)k(y)`, with `k(0)=1` **exactly** because numerator and denominator
are the same infinite sum. Evenness gives `k(-r)=k(r)`. Therefore

\[
  J(r)=1-k(r),\qquad c_2(r)=\frac{S(0)-S(r)}{Zr^2}.
\]

The y-axis sum cancels against its identical full normalizer. This is not a
discarded two-axis tail. A truncated denominator would break this identity.

## 4. Stable central term over the continuum

Let `g(r)=(1-exp(-r²/2))/r²`. Its integral representation is

\[
 g(r)=\frac12\int_0^1e^{-tr^2/2}\,dt.
\]

For positive r its integrand decreases in r, hence `g(b) ≤ g(r) ≤ g(a)`
throughout any `0<a≤r≤b≤1/20`. No fitted modulus or grid inference is used.

With `x=r²/2≤1/800`,

\[
 g(r)=\tfrac12\sum_{n\ge0}\frac{(-1)^n x^n}{(n+1)!}.
\]

The absolute ratio of successive terms is `x/(n+2)<1`; the terms tend to
zero. Consecutive partial sums therefore bracket the exact value. The
implementation uses 96 terms and the next term as the exact alternating
remainder interval, all in `Fraction`. There is no subtraction of two
near-unit exponential enclosures and no division of independent full-band
values by an independent `r²` interval. This avoids the dependency inflation
present in the older slab prototype. No comparison is made to that
prototype's certification status or to a source-approved modulus.

## 5. Uniform image correction, including every omitted image

Set `H(s)=Σ_{j≠0}φ(s+24j)`. The image series and its first two derivatives
converge uniformly on `[0,1/20]`: a polynomial times a Gaussian is dominated
by the geometric majorants below. Termwise differentiation is thus valid.
Symmetry gives `H'(0)=0`. Taylor's formula with integral remainder gives

\[
 \frac{H(0)-H(r)}{r^2}
   =-\int_0^1(1-t)H''(tr)\,dt.
\]

For all `j≠0` and `0≤s≤b≤1/20`, `|s+24j|≥24-b>1`, so
`φ''(s+24j)=((s+24j)²−1)φ(s+24j)>0`. The two terms `j=±1` are
evaluated by exact interval polynomial arithmetic and certified `exp` on
the **whole** interval `s∈[0,b]`.

For the rest, the repository's `image_tail(2,b)` is used with its existing
proved majorant. To spell out the bound consumed here, if `j≥2`, then
`|s±24j|∈[24j-b,24j+b]` and

\[
 |\phi''(s\pm24j)|
 \le 2(24j+b)^2e^{-(24j-b)^2/2}.
\]

The factor 2 is the sum of absolute coefficients of `He₂=u²−1`; it is
valid for `|u|≥1`. Let

\[
 t_2=2(48+b)^2e^{-(48-b)^2/2},\qquad
 q=\left(\frac{72+b}{48+b}\right)^2
 e^{-((72-b)^2-(48-b)^2)/2}.
\]

The ratio of successive polynomial factors decreases with j, and the
Gaussian ratio decreases with j. Thus every consecutive ratio is at most
q. The checker establishes `q<1` in exact interval arithmetic. Both signs
and every `j≥2` are covered by `T₂=2t₂/(1-q)`; there is no second truncation.

If `[h_lo,h_hi]` is the first-pair enclosure plus `[0,T₂]`, then
`H''(tr)∈[h_lo,h_hi]` uniformly. Integrating the nonnegative weight whose
integral is exactly 1/2 gives

\[
  E(r):=\int_0^1(1-t)H''(tr)dt\in[h_{lo}/2,h_{hi}/2].
\]

The final candidate interval is

\[
 c_2([a,b])\subseteq
   \frac{[g(b)_{lo},g(a)_{hi}]-[h_{lo}/2,h_{hi}/2]}{[Z_{lo},Z_{hi}]}.
\]

All interval operations are outward. A final `round_out(2048)` only widens
the result and limits endpoint serialization size.

## 6. Complete denominator and why dropping its tail is detectable

`Z=1+2exp(-288)+Σ_{|j|≥2}exp(-(24j)²/2)`.
The remaining sum is positive. Its lower bound includes the explicit pair
`2exp(-1152)`; its upper bound is `image_tail(0,0)` (the same geometric
argument with power zero). Both endpoints are retained, and positivity of
the denominator is checked before division. The finite part uses 610
decimal precision only to make the tiny missing-tail negative control
resolvable; exact rational enclosure arithmetic supplies validity.

The omitted pair is around scale `exp(-1152)`, so a default machine double
could silently erase it. The exact control proves
`Z_finite.lo + 2exp(-1152).lo > Z_finite.hi`. Thus omitting the denominator
tail fails containment even at side 24. This is a mathematical witness,
not just a check for a field name in JSON.

## 7. Verification and obstructions still open

Run from the repository root using Python 3.11:

```sh
python -B research/parallel/c2/c2_band.py
python -B -O research/parallel/c2/c2_band.py
python -B -m pytest -q -p no:cacheprovider tests/test_c2_band.py
```

The portable integration derives the checkout from the module path and verifies
`candidate.json` by default. `DEPENDENCIES.json` retains the 14 mathematical
repository import identities, including package initializers; source binding
rejects changed sizes or hashes. The report also records the four direct
mathematical modules. CLI input parsing additionally uses the shared SIDE24
report loader to refuse duplicate keys, noninteger numbers and oversized data.
Canonical-byte comparison distinguishes bool from int. Output creation is
exclusive, and the original exact candidate data and analytic argument are
unchanged. No source code in `sources/` is imported or executed.

Nineteen tests check source identities, exact geometry/signs, continuum
enclosure arithmetic, rational series against a separately composed
certified exponential, supplementary 17-point interval containment,
subband tightening, normal/-O replay, and the scope fields. Negative controls
include numerator-image omission, wrong image sign, missing denominator
tail, hundredfold weakening of each omitted tail, loss of the
second-argument derivative sign, wrong normalization power, source hash
corruption, unsafe inputs, and an inward-mutated stored report rejected
through the actual CLI in both normal and optimized modes.

The initial test draft mistakenly expected a 400-digit exponential enclosure
to fit inside a much tighter 96-term series enclosure at four radii. That
test assertion was false; the enclosures intersected and no candidate bound
failed. The comparison now deliberately uses a 32-term outer series bracket,
which the separately composed exponential enclosure demonstrably lies inside.
The delivered 96-term candidate computation was unchanged.

Missing exact inputs are named rather than inferred:

1. The reviewed full 24-entry `(J,p_J)` map is not recovered by the two
   permitted sources used here. Only the explicit `c₂` definition is bound;
   the lower-campaign `jets.py` is not substituted for the H5 jet set.
2. The source-approved full partition `[r_{k+1},r_k]` and a claimed modulus
   for this exact normalized scalar are not bound. A chosen adjacent literal
   rung interval is narrower scope. The modulus verdict is
   **INSUFFICIENT_DATA**, even though this one band has an enclosure.
3. The conditioned G12 band, its positive inverses/Schur complements, the
   cover map F, other 23 jets, and their uniform tails/powers remain separate.
4. No H5 excluded certificate is restored, no H3 floor is replayed, no RN
   spatial area integral is supplied, and no all-small-r 2D theorem is closed.

Tests are internal verification. No organizational independence, formal proof
assistant verification, novelty, or permission to publish publicly is claimed.
