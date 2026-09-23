# `research/lpw/` — the LPW headline constant, and which decimals may stand in it

Scope: `docs/OPEN_PROBLEMS.md` **section C, "LPW constant repair"**, and the
`LPW_CONSTANT as delivered` / `R05 Rayleigh quantitative repair` rows of the
`lpw_fold_dispositions` register tab (`registers/json/lpw_fold_dispositions.json`,
sheet 41 of the GP-REG-032-v1.2 export).

Standard library only, Python 3.11, `fractions.Fraction` throughout. **No float,
no `decimal`, no `mpmath`, no `numpy`** — every number here is an exact rational
and every verdict is an exact rational comparison, so no output of this lane
needs, or carries, a NON-CERTIFYING float label. It also means nothing here is a
certificate of anything but arithmetic.

| file | what it is |
|---|---|
| `headline.py` | the exact fraction from its named factors; correctly-rounded significant-digit values with the direction each moved; the admissibility rule; `report()` |
| `../../tests/test_lpw_headline.py` | end-to-end headline tests, including three negative controls |

Run `python3 research/lpw/headline.py` for the full comparison table.

## What the sources say

Quoted from the register rows, verbatim where quoted:

* `LPW_CONSTANT as delivered` — "Exact fraction
  `260/(3790446482793*2^40*10^21)=6.23854270293559...e-44`; reported
  `6.239e-44` exceeds that chain. Radius `1/2414592`." Operator disposition:
  "**AUTHORIZED; AMEND REQUIRED on actual evidence — no new owner vote**".
  Formal promotion: "**Do not admit the delivered `6.239e-44` certificate as
  unqualified; qualitative LPW unchanged**". Evidence: "Kimi PASS preserved as
  received. R05 finds wrong headline decimal, false `E(|xi|+|eta|)^4=12+16/pi`
  identity, and incomplete blanket-rounding explanation." Custody: the constant
  manifest is 9/9 payload hashes matched, and "the original certifier and
  falsifier replay both modes exactly; **they do not catch these
  mathematical/report defects**". Next decisive action: "Review R05
  Rayleigh-amplitude repair and directed intervals; **freeze corrected exact
  fraction or `6.238e-44` with end-to-end headline tests**."
* `R05 Rayleigh quantitative repair` — "New proposed repaired bound:
  `1-q>=6.238e-44 r^3` on `0<r<=1/2414592`, under unchanged LPW law." Operator
  disposition: "**RESEARCH AUTHORIZED; TARGETED REVIEW REQUIRED**". Formal
  promotion: "**AUTHOR-SIDE REPAIR — does not inherit Kimi approval
  automatically**". Custody: "39 interval checks, normal/-O identical;
  oversized `6.239e-44` headline rejected both modes."
* `LPW qualitative` — "2D side-24 exact six-pin typed pair-Palm elder pairing;
  exists `c,r0>0`: `1-q>=c r^3`", **ACCEPTED AT REVIEW SCOPE**.

`docs/RESEARCH_MAP.md` §5 adds that PKG-04 ships the LPW review bundle, that
**Consequence 4 (the two-sided claim) is WITHDRAWN**, and that the LPW-CONSTANT
v4 brick is on **HOLD, titled NOT READY**.

## The finding this lane makes permanent and machine-checked

Reproduced here independently from the named factors, in exact rationals:

```
exact              = 260 / (3790446482793 * 2^40 * 10^21)
                   = 13 / 208381999114677270242918400000000000000000000
                   = 6.238542702935587792943041084518910355631...e-44   (truncated display)
delivered 6.239e-44  is ABOVE exact,  relative excess  +7.330e-05
proposed  6.238e-44  is BELOW exact,  relative deficit -8.699e-05
6.239e-44 is also the correctly-rounded 4-significant-figure value of exact.
```

The exact fraction's decimal expansion **does not terminate**: its reduced
denominator carries the B3 ceiling `3790446482793`, a prime factor other than 2
and 5. So no finite decimal equals it, and every decimal headline is a rounding.

**DIRECTION DECIDES ADMISSIBILITY.** A rounded decimal is admissible only when
it is rounded **away from the asserted inequality**:

| headline asserted as | it asserts | admissible iff | `6.239e-44` | `6.238e-44` |
|---|---|---|---|---|
| **upper** bound | `quantity <= D` | `D >= exact` | ADMISSIBLE | **NOT ADMISSIBLE** |
| **lower** bound | `quantity >= D` | `D <= exact` | **NOT ADMISSIBLE** | ADMISSIBLE |
| **equality** | `quantity == D` | `D == exact` | NOT ADMISSIBLE | NOT ADMISSIBLE |

Two consequences worth stating plainly, because both are easy to get backwards:

1. "`6.239e-44` is the correct 4-significant-figure rounding" and "`6.239e-44`
   exceeds the exact chain" are **both true at once**, and correct rounding
   settles nothing by itself. Blanket rounding to the nearest decimal is exactly
   the move the register calls an "incomplete blanket-rounding explanation".
2. A decimal on the wrong side of the exact value asserts something **stronger
   than the exact chain supports**. Under an equality headline nothing rounded
   is admissible at all.

Which of the three directions the LPW headline actually asserts is **transcribed
from the sources by the caller, never decided in this lane**. `headline.py`
supplies the rule and the exact comparisons and takes `direction` as an
argument. The register's own words for the direction it is operating in are
that `6.239e-44` "exceeds that chain" and is the "oversized" headline "rejected
both modes", and that the R05 proposal is `1-q >= 6.238e-44 r^3`.

## Negative controls

`tests/test_lpw_headline.py` carries three, each run against a deliberately
broken copy of `headline.py` to confirm it fires:

| control | fails when |
|---|---|
| `test_negative_control_comparison_flip` | the `upper`/`lower` comparison is flipped (`>=` ↔ `<=`) — all four bound verdicts must disagree with the flipped rule |
| `test_negative_control_perturbed_factors` | any named factor moves: numerator ±1, B3 ceiling ±1, `2^39`/`2^41`, `10^20`/`10^22`. The perturbation list is written from the register's values, not imported, so editing the module's constants makes its own value land on a perturbation |
| `test_negative_control_wrong_side_rejected` | `6.238e-44` is ever admitted as an upper bound, or `6.239e-44` as a lower bound |

## What remains OPEN

Nothing in this lane moves any of these. They stand exactly as
`docs/OPEN_PROBLEMS.md` §C and the register write them:

* **The R05 review Kimi owes.** "Kimi review the amplitude lemma, full
  tails/profile modulus, conditional bounds, exact fraction and actual headline
  mutation." All five are outstanding. The R05 repair is author-side and "does
  not inherit Kimi approval automatically"; the earlier Kimi PASS is preserved
  as received and does not transfer to the repair.
* **The R05 Rayleigh amplitude lemma** (`rho = sqrt(xi^2+eta^2)`, `E rho^4 = 8`
  below the old `12 + 16/pi` budget). Not examined here.
* **The full tails / profile modulus.** Not examined here.
* **The conditional bounds.** Not examined here.
* **The actual headline mutation.** Not performed here; no headline is mutated
  by this lane.
* **The false identity.** R05 finds `E(|xi|+|eta|)^4 = 12 + 16/pi` false. This
  lane neither confirms nor repairs that finding.
* **The freeze itself.** §C asks that either the corrected exact fraction or
  `6.238e-44` be frozen, with end-to-end headline tests. The tests exist now;
  **the freeze is an operator act and has not been made.** This lane does not
  choose between the two options and does not recommend one.
* **The delivered certificate** remains "not to be admitted unqualified". The
  constant manifest's 9/9 hash match and the certifier/falsifier replays do not
  catch the report defects, and nothing here changes that.
* `OBL-H5-ZBAND` hi side still needs the **band** version of the LPW bracket
  (§A3), untouched here.

## What this lane does NOT establish

* It **does not verify the derivation** behind `260/(3790446482793*2^40*10^21)`.
  The fraction and every factor in it are transcribed from the register. If the
  derivation is wrong, every comparison here is a correct comparison against a
  wrong number.
* It **closes, discharges, reduces, promotes, admits and reclassifies nothing**.
  `admissible_as` returns a verdict about a decimal's direction of rounding.
  That is not a review, not an approval, not a certificate and not a promotion.
* Admissibility as computed here is **direction-soundness only**, never
  tightness: a grossly weak decimal on the sound side passes.
* **Qualitative LPW is unchanged by any of this**, as the register says. The
  qualitative statement `exists c, r0 > 0 : 1-q >= c r^3` remains ACCEPTED AT
  REVIEW SCOPE, with its own custody, and this lane neither strengthens nor
  weakens it. The QC-RETURN03 fallback (`1-q >= 10^-1235 r^3` for
  `0 < r <= 10^-28`) is preserved separately; "new defects in the improved
  constant do not consume or refute this proof".
* It relates the 2D lower track to the 2D upper track in no way, and to the 3D
  lifetime track in no way. **No composition across tracks**; Consequence 4
  stays WITHDRAWN and the 2D two-sided Θ stays not admitted.
* It bears on **no prize problem**. None is solved.
* Green tests are not a mathematical review.

It settles exactly one question: **which rounded decimals are admissible in
which direction.**
