# LPW parent-Gaussian amplitude companion — candidate v1

This source-bound, author-side companion makes a narrow part of the R05
repair executable in exact arithmetic. For independent parent variables
`xi,eta ~ N(0,1)`, it reconstructs

`E rho^4 = 8`, where `rho=sqrt(xi^2+eta^2)`,

`E (|xi|+|eta|)^4 = 12+32/pi`.

The historical expression `12+16/pi` is therefore not the fourth moment of
`|xi|+|eta|`. It remains a valid upper budget for the **different** Rayleigh
amplitude `rho`. The source already derives this repair; the new work is a
small exact algebra/custody check and adversarial test suite, not a novelty
claim or an independent mathematical review.

No source artifact, headline, scientific status, review gate or register is
modified by the checker. The qualitative LPW and fallback results retain
their own scopes. No 2D/3D composition or original prize closure follows.

## Exact custody and assumptions

The checker reads these two existing local files as bytes and verifies their
complete digests and sizes. It executes neither source body.

| Source | Drive identity | Bytes | SHA-256 |
|---|---|---:|---|
| R05 analytic repair | [1ngaO6hzeIXdCYWMwKl8JYPPZTeryGFTx](https://drive.google.com/file/d/1ngaO6hzeIXdCYWMwKl8JYPPZTeryGFTx/view) | 9,628 | `840c75a7825c67b8d99a536394beb18976475fe9bd0e474ed5f80c375004ec81` |
| Existing directed-interval companion | [1cdE15_2dd0VLHJnDGcFQ5RDQJhTkXfr-](https://drive.google.com/file/d/1cdE15_2dd0VLHJnDGcFQ5RDQJhTkXfr-/view) | 6,843 | `b37150f0eb79ff5d11b9e8b60f80afd94c677f8150aa701573ce24b337f24069` |

Both are mirrored under
`drive/mirrors/02_RESEARCH_CARRY_FORWARD_CANON/LPW — RAW ARCHIVE INTAKE R05/`.
The second is `checks/interval_repair.py`. These bytes also occur in the
R05 archive, Drive `1gE3ZNbNRcpH2K0uHVJPiofJ3F-hkYURV`. The existing companion
has its own recorded 39 checks and normal/optimized receipts. The new checker
does not rerun, replace, extend the authority of, or count those 39 checks as
its own. Custody alone does not verify the source's analytic claims.

The parent Gaussian pair is independent, centered and unit variance. Its
moments are **unconditional parent-law moments**; conditioning generally
changes this law. R05's conditional norm estimates use a separate regression
argument. They cannot be obtained by declaring the conditioned amplitudes
independent standard normals.

R05 uses a full-lattice real Fourier representation with normalized masses,
the max-convention `C^p` norm, and Gaussian-decay summability. Opposite modes
are allowed as separate Gaussian coordinates in that representation. This
companion records that convention; it does not switch to a half-lattice with
unchanged weights or authenticate the representation as a particular field.
The infinite-sum Tonelli/Minkowski justification remains an analytic import.

## Derivations checked

For a standard normal `Z`, the half-normal density and integration by parts
give `m0=1`, `m1=sqrt(2/pi)` and `mn=(n-1)m(n-2)` for `n>=2`. Thus

| Order | `E |Z|^n` |
|---:|---|
| 0 | 1 |
| 1 | `sqrt(2/pi)` |
| 2 | 1 |
| 3 | `2 sqrt(2/pi)` |
| 4 | 3 |

Independence and the binomial expansion of `(a+b)^4` give the five exact
terms `3`, `16/pi`, `6`, `16/pi`, `3`. Their sum is `12+32/pi`. The code stores
these as rational pairs `(a,b)` denoting `a+b/pi`, without approximating pi.
Expansion of `(xi^2+eta^2)^2` gives `3+2*1*1+3=8`.

The pointwise inequalities do not require Gaussianity. For
`a=|xi|>=0`, `b=|eta|>=0`, the checker expands the polynomial identities

`(a+b)^2-(a^2+b^2)=2ab>=0`,

`2(a^2+b^2)-(a+b)^2=(a-b)^2>=0`.

Nonnegativity permits taking square roots, proving
`rho <= |xi|+|eta| <= sqrt(2) rho`. The constants are attained, respectively,
when one coordinate vanishes and when their absolute values agree.

For the Fourier-phase bound, it also checks the polynomial identity

`(x^2+y^2)(c^2+s^2)-(xc+ys)^2=(xs-yc)^2`.

When `c^2+s^2=1`, this gives `|xc+ys|<=sqrt(x^2+y^2)`. Differentiation changes
the phase and contributes the frequency powers stated in R05; passing from
finite sums to the full field still requires the imported analytic argument.

The radial parent density is `r exp(-r^2/2)` for `r>=0`; integrating its tail
gives `E rho=sqrt(pi/2)`. The old first budget is `2sqrt(2/pi)`. The checker
compares their squares, `pi/2` and `8/pi`, using a positive rational enclosure
with upper endpoint less than 4. Since both means/budgets are nonnegative,
the squared comparison proves the intended first-moment inequality.

`research.interval.pi(12)` supplies a certified rational Machin-series
enclosure. All comparison endpoints are exact fractions. The candidate
records strictly positive enclosures for the historical fourth budget minus
8, the corrected fourth moment minus the historical budget (`16/pi`), both
fourth-moment norm comparisons, and the first-budget squared difference.
These are exact enclosure statements, not floating-point evidence.

An independent semantic test uses polar coordinates: `rho^2/2` has an
exponential distribution, so `E rho^4=4*2!=8`. Uniform-angle averages give
`E(|cos theta|+|sin theta|)^4=3/2+4/pi`, yielding `12+32/pi` by a different
route. The code's ordinary mathematical argument is documented here; the
finite tests are not a proof-assistant formalization of Gaussian integration.

## Radius arithmetic and remaining imports

Conditional on R05's imported `K<=K_ceiling=9432`, the exact values
`delta=1/1024`, `R=1/100000` give

`r0=1/(256*9432)=1/2414592<R`,

`16delta+8K_ceiling*r0=3/64<1/16`, with slack `1/64`.

This verifies the radius assembly arithmetic. It does not prove that 9432
is a valid conditional bound on K. The endpoint eigenfloor `31/250`, full
spectral tails, profile modulus, shared compact set, conditional
mean/density/norm estimates and LPW topology/Taylor/weight proof remain
imported and unverified by this module. In particular it does not verify the
derivation of the headline fraction. `research/lpw/headline.py` separately
checks decimal direction against that fraction. The R05 targeted review and
actual exported-headline mutation remain separate work.

## Reproduce and falsify

```bash
python tools/lpw_amplitude_check.py
python -O tools/lpw_amplitude_check.py
python -m pytest tests/test_lpw_amplitude.py tests/test_lpw_headline.py -q
```

The candidate is `research/lpw/candidates/amplitude_certificate_v1.json`.
`--certificate PATH` checks an explicit alternative through the same CLI.
The verifier checks the pinned source bytes, assumptions, every reconstructed
proof field and exact no-authority flags. It rejects unknown versions/fields,
duplicate JSON keys, noninteger JSON numeric literals and oversized input.
Payloads contain no instructions or executable code. All checker decisions use
explicit exceptions rather than `assert`, and normal/optimized output is
tested for byte equality. The actual override path is tested with the wrong
`16/pi` moment, so negative controls cannot silently recheck the good default.

The producer and checker share the finite algebra routines and interval
library. This is a small deterministic replay checker, not independent code
or organizational review. Tests also alter source bytes, the parent-law
assumptions, norm/phase identities, moment terms, pi enclosure, K, radius,
geometry budget and review/status fields. Every such mutation is refused.
