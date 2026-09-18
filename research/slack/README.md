# `research/slack` — the bound-slack registry

## What this is for

This program's bounds are recorded honestly but their *looseness* was recorded
only in prose. The headline case, from `CL-RNU-001` as quoted verbatim in
`LANE_RN_UNIF.md` §2:

> Engine `d3_rn_unif.py` located (Kimi mid-build); certifier never invoked.
> Root cause: `chi2_grad_bound` ~1.57e14 vs true |∇χ²| ~1.563e-5 (**~1e19
> slack**). E-RNU-1: `mean_grad_exact` missing chain-rule terms (fix validated,
> not patched into frozen engine). Piece 2 driver unwritten.

A bound nineteen orders of magnitude loose cannot close anything, and until this
module nothing in the repository noticed. `registry.py` turns "that bound is
loose" into a tracked record: the bound's name, the carrier and the source
document that state it, the claimed value, the best known true or attained
value, the provenance of each, the slack ratio computed **exactly**, and an
order-of-magnitude classification.

Run it:

```bash
python3 -c 'import research.slack as S; S.report()'
```

---

## THE HONESTY RULE

**A large slack ratio is a statement about a BOUND'S UTILITY. It is never a
statement about the bound's CORRECTNESS.**

A bound loose by `1e19` is still, as far as this module knows, a perfectly true
bound. `chi2_grad_bound` returning `1.57e14` where the true gradient magnitude
is `1.563e-5` is, on that evidence alone, a *correct* upper bound that happens
to be useless. Nothing here may be read as saying otherwise, and nothing it
prints is allowed to imply it.

The rule is enforced in code, not merely asserted:

1. `SlackRecord.known_unsound` is a separate field, defaulting to `False`, and
   it is computed from data disjoint from every slack quantity.
2. `known_unsound=True` is **not settable from a slack ratio**, however large.
   It requires a `SoundnessWitness` — a pair of certified enclosures that
   *exhibits* the failure — and `SoundnessWitness.violates()` **recomputes** the
   violation by exact interval comparison. A witness that does not actually
   exhibit a violation is refused at construction. So a merely loose bound
   cannot be marked unsound: its own numbers, offered as a witness, prove the
   opposite of a violation.
3. `SlackRecord.severity()` never reads `known_unsound`. Flipping one may not
   move the other, and `tests/test_slack.py` pins that.
4. The severity names are utility words — `TIGHT`, `ROUTINE`, `LOOSE`,
   `SEVERE`, `UNUSABLE_AS_STATED`. Not one is a correctness word, and
   `forbidden_words_in_report()` pins the absence of correctness vocabulary
   from the generated table.
5. The report prints the rule in its banner on every single run, because a
   table of enormous numbers read without it is exactly the misreading this
   module exists to prevent.

**The converse half matters as much, and the registry has a worked example of
it.** `cone_slope_margin` (`GP-DATA-106`, audited by `CL-AUD-202-v1.0`) is the
**tightest** record here — its slack ratio differs from 1 by about `7e-18` —
and it is one of the two records **known to be unsound**. Sort the registry by
slack and the loosest bound is not known to be wrong while the tightest one is.
Tightness is not soundness; looseness is not unsoundness.

Where a record *is* marked unsound, the unsoundness was established and
published by a **source** — RN5 2026-09-17 for `envelope_v`, CL-AUD-202-v1.0 for
`cone_slope_margin`. This module re-verifies their published witness arithmetic
and nothing more. **It has discovered no unsoundness of its own and claims
none.**

---

## The severity ladder, and why these boundaries

The boundaries are anchored to quantities this program's own machinery uses, so
that a band means something operational.

| band | ratio | anchor |
|---|---|---|
| `TIGHT` | `[1, 2)` | where the published certified enclosures live — the H3 rung floor's `+92.1%` margin, RN3's `1.273×` far-count spread |
| `ROUTINE` | `[2, 10)` | inside the fudge the frozen engine already grants itself: `SAFE_H = 3`, `SAFE_T = 9` in the cell sup-bound at line 2163, "absorbing cell-scale variation" |
| `LOOSE` | `[10, 1e3)` | recoverable by refinement in principle — the first-order term shrinks linearly in the cell half-width |
| `SEVERE` | `[1e3, 1e6)` | at or past the refinement budget the program has: the polar cover `d ∈ [5,17]` bisects to a fail-closed floor `hw < 4e-4`, i.e. ≤ 15,000 radial cells, ~`1e6` with θ-halving |
| `UNUSABLE_AS_STATED` | `[1e6, ∞)` | no feasible subdivision of this program's own covers recovers it |
| `VIOLATION_WITNESSED` | `[0, 1)` | not a looseness class; reachable only with a re-verified `SoundnessWitness` |

Severity is classified from the ratio's **lower** endpoint — the least slack
consistent with the recorded data — so the classification never overstates how
bad a bound is. `severity_is_sharp()` says whether both endpoints land in the
same band.

---

## What is registered, and what the ratios mean

Seven records, sorted worst-utility-first by `report()`:

| # | bound | slack ratio | band | unsound? |
|---|---|---|---|---|
| 1 | `chi2_grad_bound` | `~1.00e19` | `UNUSABLE_AS_STATED` | no |
| 2 | `T4_kap(5) = C_comp · T4_form(5)` (candidate) | `~3.9e5` | `SEVERE` | no |
| 3 | `τ` entrywise envelope `‖Δ‖_F²/λ_min` | `~5.3e3` | `SEVERE` | no |
| 4 | `c_Z·r²` floor for `Z_r` at `r = 1/20` | `≥1.921` | `TIGHT` | no |
| 5 | `I_far` upper (RN3 far zone) | `≤1.273` | `TIGHT` | no |
| 6 | `cone_slope_margin` (GP-DATA-106) | `1 − 7.04e-18` | `VIOLATION_WITNESSED` | **yes (CL-AUD-202)** |
| 7 | `envelope_v` (d3_perc.py) | `~0.302` | `VIOLATION_WITNESSED` | **yes (RN5)** |

A ratio is only as good as the two numbers it divides. `attained_kind` records
what the attained value is relative to the true value, and `ratio_kind()`
propagates that into a statement about what the ratio enclosure *means* — an
enclosure of the true slack, an upper bound on it, a lower bound on it, or
`INDICATIVE_ONLY`. That is the whole difference between "the bound is 1.27×
loose" and "the bound is *at most* 1.27× loose, and may be exactly tight".

Only record 5 has both sides certified *and* a two-sided relation. Records 1, 2
and 3 are `INDICATIVE_ONLY`: their inputs are quoted digits, mpmath floats, a
candidate constant and a finite difference. **An exactly-computed ratio between
two uncertified inputs is an exact ratio of uncertified inputs.**

---

## Arithmetic discipline

* Values are `Fraction`, exact decimal/rational strings, `Decimal` or
  `Interval`. **A bare `float` in a value field raises** — this module's own
  refusal, over and above `research.interval.to_fraction`'s, with its own
  message.
* `to_fraction`'s documented hole applies here too: `Decimal(0.1)` is the exact
  binary double, not `1/10`. A `Decimal` carries no provenance so this cannot be
  detected; pass strings.
* Quoted values such as `~1.57e14` are stored as the **rounding interval of the
  quoted digits** — `[1.565e14, 1.575e14]`, an honest enclosure of *what the
  document reported*, which is not an enclosure of the mathematical quantity.
  Truncated expansions ending in `...` use `truncated_interval`, which is
  one-sided.
* Exactly one value is stored as a binary double's exact rational: record 6's
  `claimed`. CL-AUD-202 prints the digit string `0.0086443674942901349`, but
  read as an exact decimal that overshoots by `+7.3506e-20`, not the
  `+6.083e-20` the audit states; read as the IEEE-754 double whose shortest
  repr it is, it overshoots by exactly `6.083098693184418e-20`, reproducing the
  audit's own figure. Both readings overshoot, so the verdict is unaffected;
  both are pinned in `tests/test_slack.py` rather than one being quietly
  picked.
* `order_of_magnitude()` and `sci()` are exact integer computations. No
  logarithm and no float enters any path.

---

## What was searched

`SEARCH_LOG` in `registry.py` carries this verbatim and `report()` prints it.
Searched over `docs/`, `registers/json/` (42 exported tabs), `claims/`,
`research/`, `reviews/`, `engine/` and the Drive index (`tools/drive_index.py
find`, 4,456 items), for: *slack, vs true, versus true, orders of magnitude,
loose, conservative, overestimate, overbound, overshoot, too loose, not tight,
fail-close(s), collapsed, margin, factor of N*.

Documents stating **both** a claimed bound and a measured or true value for the
same quantity in the same breath are rare in this corpus. These seven are all
that were found.

**Is the headline case the only well-documented one?** No, but it is the only
one of its size. `chi2_grad_bound`'s `~1e19` is fourteen orders of magnitude
clear of the next-largest slack in the registry, and it is the only record in
the `UNUSABLE_AS_STATED` band. The other six are well documented; they are not
comparably bad.

Six cases were examined and **deliberately not registered**, each with its
reason recorded in `SEARCH_LOG` — including RN3 §9's `2.34195 r³` corrected
versus `17.67237 r³` wrong-power (factor ~7.5), which is a corrected diagnostic
against a defective one rather than a bound against a true value, rides the
`envelope_v` defect already registered as record 7, and would have been the
registry's only `ROUTINE`-band entry. Padding a band is not a reason.

**Coverage gap, stated rather than filled:** no record lands in `ROUTINE`
`[2, 10)` or `LOOSE` `[10, 1e3)`. The registry is bimodal — two records under
`2×`, two violations, three at `5e3` or worse. Whether that is the corpus or the
search is not established here.

Not searched, on the standing rules: `02_LEGACY_Q0_ARCHIVE` (zero evidentiary
authority) and the folder `99_DO_NOT_OPEN`, which was neither opened nor listed.

---

## WHAT THIS MODULE DOES NOT ESTABLISH

* It **closes, discharges, reduces, promotes, reclassifies and repairs
  nothing.** `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side),
  `OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` (chart side) and **both** pieces
  of `D3-LEMMA-RN-UNIF` stay exactly as OPEN as `docs/OPEN_PROBLEMS.md` records
  them. Receipts still carry `lemma_closed: false`. **Measuring a defect is not
  fixing it.**
* It **refutes no bound.** The two `known_unsound` records were established by
  their sources; this module re-verifies published witness arithmetic and has
  discovered no unsoundness of its own.
* It **repairs no bound and patches no engine.** The frozen engine under
  `engine/rn_engine/frozen/` is untouched and byte-identical to the archive.
  Recording that `chi2_grad_bound` is `1e19` loose does not make it tighter,
  and does not address the separate and more serious matter that it is fed
  `mean_grad_exact`, which `docs/ENGINE_RECOVERY.md` §3.6 reports is missing
  chain-rule terms.
* It **certifies nothing about the quantities it compares.** Every value here
  is quoted from another document. Where those are mpmath floats, finite
  differences, candidate constants or quoted digit strings, the ratio inherits
  exactly that status and is labelled `NON-CERTIFYING`. High precision is not a
  certified enclosure; a Monte Carlo estimate, a fitted exponent, a dense
  sampling, a display, a probe cell, a candidate constant, a smoke test, a
  session CLOSE and a registration are none of them certificates either — the
  sources say so about themselves.
* It **never composes** the 2D upper track, the 2D lower track and the 3D
  lifetime track, and relates them in no way.
* **Original prize problems solved: 0.** This module bears on none of them.
* Passing tests show the arithmetic does what the docstrings say. **A green
  build is not a mathematical review** of any bound's derivation.

## Tests

`tests/test_slack.py`: exact slack-ratio arithmetic on the `chi2_grad_bound`
case reproducing the source's own `~1e19`; every severity-ladder boundary from
both sides; the exact reproduction of CL-AUD-202's `+6.083e-20` from the double
and of `+7.3506e-20` from the digit string; and negative controls — a record
conflating looseness with unsoundness, a witness that fails to recompute to a
violation, a proved violation recorded as sound, a `float` in a value field, and
an unsourced record must each be refused. Every control was run against a
deliberately broken copy of `registry.py` and confirmed to fail there; each
names its mutation in its docstring.
