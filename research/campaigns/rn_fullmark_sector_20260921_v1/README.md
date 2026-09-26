# Certified full-height RN sector expansion

Sixteen new polar cells extend the existing fixed-radius wedge to
`rho in [1/10,3/25]`, signed turns `[-1/128,1/128]`. The complete height
interval remains `[57599/48000,6/5]`. The whole sector has **352/21 times
the old area**, approximately **16.76 times**.

Every new cell rigorously establishes `q >= 103` and a positive actual
three-jet covariance determinant. The full-adjugate energy argument succeeds
on all 16 cells; the simpler constant projection alone establishes that
threshold on none of them. This is a substantive stronger inequality,
not a conclusion drawn from unsuccessful sampling.

The new cells each have integrand upper bound below `7/10000000`.
Together with the existing wedge, the whole sector satisfies

`I(y) < 1/1000000`,

`integral_W I(y) dy < 7/100000000000 = 7e-11`.

The exact aggregate upper bound is recorded in the result. Its approximate
value is `6.732393130703628e-11`; that decimal is for orientation only.
The old wedge's integral upper bound is added exactly once.

## Evidence

- Fresh normal and optimized Python runs both certified all 16 cells.
  Their 841,420-byte `result.json` files are byte-identical, SHA-256
  `337afa45b99141444833daefb19dcbc2612d8296df35f8fea92aa31697c219ba`.
- 59 exact control tests pass in each mode: 19 energy, 18 geometry and
  22 aggregate/source-order controls.
- All 36 declared repository dependencies and the numerical implementation
  identities were unchanged through both runs. Current source admissibility
  was checked before source-dependent computation and again during execution.
- Source-exposed technical reviews found no remaining mathematical or
  composition defect. Independence credit is zero.
- Failed probes and an interrupted pre-correction development run are retained
  as diagnostics. They confer no spatial certificate.

See `PROOF.md`, `projection/PROOF.md`, `geometry/PROOF.md`, `REVIEW.md` and
`VALIDATION.json`. `MANIFEST.json` is the delivery file allowlist and byte map.
Hashes support reproducibility; they are not a replacement for the proof.

## Replay

Use Python 3.11 and a repository checkout matching `DEPENDENCIES.json`.
The recorded source checkout was
`dca27f77eaad7453b95c53933069955669ad8b65` of
`d6g8k5htny-coder/main`. The checker refuses changed pinned dependencies,
unlisted imported repository code, altered prior evidence and existing output
directories. Place the delivery outside the repository and select fresh
external output directories whose parents already exist.

```sh
python3 -B check.py --repo /path/to/research-main --output /fresh/sector-normal
python3 -B -O check.py --repo /path/to/research-main --output /fresh/sector-optimized
cmp /fresh/sector-normal/result.json /fresh/sector-optimized/result.json
```

The defaults are the recorded 16-cell sector and target 103. Exit 0 means
a complete scoped cover; exit 2 means retained inconclusive cells and no total.
A valid weaker energy bound may still yield a valid, weaker integrand cap:
inspect the recorded bounds and totals rather than interpreting exit 0 as a
particular threshold or global theorem closure.

The control tests run independently:

```sh
python3 -B test_check.py
python3 -B geometry/test_geometry.py
python3 -B projection/test_projection.py --repo /path/to/research-main
```

Repeat with `-O` to check optimized Python. The serialized headline can be
checked with exact fractions:

```python
import json
from fractions import Fraction as F
from pathlib import Path
r = json.loads(Path('results/normal/result.json').read_text())
if r['totals']['pending_count'] != 0:
    raise ValueError('Incomplete cover')
s = F(22,7) * sum((F(a['area_pi_coefficient']) * F(a['integrand_upper'])
                   for a in r['attempts']), F(0)) + F(33,2560000000000)
if s != F(r['totals']['whole_sector_integral_upper']) or not s < F(7,10**11):
    raise ValueError('Aggregate headline mismatch')
```

This last calculation checks arithmetic on a recorded certificate; it does
not replace fresh source admission or the complete numerical replay.

## Scope and remaining work

The normalized SIDE24 law, `r=1/20`, `b=6/5`, horizontal pin axis, pin values,
full mark interval and imported H3 floor are unchanged. The H3 and six-pin
energy proofs are imported existing premises, not reproved here. The new
energy error terms remain in matching coordinates and keep all remainders.

This does not close the full annulus, all radii, all pin orientations,
weighted-Palm/event identification, independent acceptance or q0. No
scientific status is promoted. The wider 32-cell geometry has unit tests,
but no numerical cover is claimed for it in this delivery. Integration and
publication belong to the sole repository writer; the scratch work changes
no shared checkout or frozen source archive.
