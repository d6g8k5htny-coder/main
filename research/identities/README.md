# `research/identities/` — exact settlement of asserted moment identities

One module, `gaussian_moments.py`, and one test file,
`tests/test_identities.py`. Python 3.11, **standard library only**
(`fractions`, `math`, `dataclasses`, `itertools`, `typing`). Every verdict is
decided in exact arithmetic. `math.gamma` appears only as a labelled
**NON-CERTIFYING** oracle and decides nothing.

---

## 1. The refutation

`docs/CONTRIBUTION_PLAN.md` §2 names a false identity carried in the corpus:

```
E(|ξ| + |η|)^4  =  12 + 16/π            xi, eta iid N(0, 1)      CLAIMED — FALSE
```

It is asserted verbatim in `registers/json/lpw_fold_dispositions.json`
(LPW_CONSTANT disposition, R05 finding), and echoed in `claims/graph.json`
line 194 and `docs/RESEARCH_MAP.md` line 220. The register itself already
labels it false; **this module supplies the exact derivation, not the
discovery.**

### Derived here, from the Gamma formula, not from any reported number

For `X ~ N(0,1)`, `E|X|^k = 2^(k/2) Γ((k+1)/2) / √π`. Splitting on parity with
`Γ(m+½) = (2m)!/(4^m m!)·√π` and `Γ(m+1) = m!`:

| | even `k = 2m` | odd `k = 2m+1` |
|---|---|---|
| value | `(2m)!/(2^m m!) = (2m−1)!!` | `2^m · m! · s` |
| `s`-power | 0 | 1 |

with `s := √(2/π)`, so `s² = 2/π`. First few: `1, s, 1, 2s, 3, 8s, 15`.

Binomial expansion plus independence gives **five** terms:

| exponents | coefficient | value |
|---|---:|---|
| `(0,4)` | 1 | `3` |
| `(1,3)` | 4 | `8s² = 16/π` |
| `(2,2)` | 6 | `6` |
| `(3,1)` | 4 | `8s² = 16/π` |
| `(4,0)` | 1 | `3` |

```
E(|ξ| + |η|)^4 = 12 + 16 s^2 = 12 + 32/π = 22.18591635788…   (decimal NON-CERTIFYING)
```

| | exact, in the `{1, 1/π}` basis |
|---|---|
| claimed | `(a, b) = (12, 16)` |
| true | `(a, b) = (12, 32)` |
| **discrepancy** | `(0, 16)`, i.e. exactly **`16/π`** |
| direction | the claim **understates** the truth (proved exactly via `223/71 < π < 22/7`) |

**Named structural cause: one dropped cross term.** The two cross terms
`4·E|ξ|³·E|η|` and `4·E|ξ|·E|η|³` are equal and each is worth exactly `16/π`.
Keeping one instead of both yields `12 + 16/π`. This is a dropped-term defect —
not rounding, not a variance convention, not a normalisation.

### Representation

Values live in `Q[s]`, `s = √(2/π)`, as `{power_of_s: Fraction}` (`Sym`). Since
`s` is transcendental the representation is faithful and `==` is exact equality
of real numbers. `Sym.as_a_plus_b_over_pi()` recovers the `{1, 1/π}` basis and
**refuses** — raises — on any odd power of `s`, because odd `k` produces `s¹`,
which the two-element basis does not span. Refusing is deliberate: silently
projecting an `s¹` term onto `{1, 1/π}` is the class of error this module
exists to catch.

---

## 2. The corpus audit — what was searched

All searches were run from `/home/user/main` on 2026-09-18 against the repo as
checked out on branch `claude/drive-audit-github-migration-rrglpp`.

| # | Command / pattern | Scope | Result |
|---|---|---|---|
| 1 | `python3 tools/drive_index.py find "12 + 16"` / `"16/pi"` / `"32/pi"` / `"sqrt(2/pi)"` / `"2/pi"` | Drive inventory | **no hits.** `drive_index find` matches item titles and paths; Drive **bodies are not in this repository**, so this tool cannot see an identity written inside a document. This is a real limit on the audit's reach, not a clean negative. |
| 2 | `python3 tools/drive_index.py find moment` | Drive inventory | title-level hits only: `LEMMA_P02-LM-002 — Determinant-Weight Second Moment`, `LCR-CAP-011 … Exact Determinant-Weight r4 Moment Interface`, `LCR-CAP-016 … Gaussian Density and Moment Interface`, `GP-DER-071 … limiting fourth-moment constant`, `AO48-DER-031 … uniform moments`, `GP-AUD-228 … Moment Audit`, `three_hessian_conditional_moment_envelope.md`. Titles carry no identity to settle. |
| 3 | `grep -rniE '12 *\+ *16\|16 */ *(pi\|π)\|12 *\+ *32\|32 */ *(pi\|π)'` | whole repo | 8 hits, all the same identity: the two register rows, their CSV twins, the source export, `claims/graph.json`, `docs/RESEARCH_MAP.md`, `docs/CONTRIBUTION_PLAN.md`. **No second `a + b/π` identity exists in the repository.** |
| 4 | `grep -rniE 'E *rho\|rho\^4\|chi.?squared\|Rayleigh'` | `registers/json`, `docs`, `claims` | found `E rho^4 = 8` — settled below. |
| 5 | `grep -rnoE 'E *[\(\[][^,;"]{0,60}'` | `registers/json` | 24 distinct expectation expressions; triaged below. |
| 6 | `grep -rnoE 'E\\?\[[A-Za-z_][^]]{0,25}\\?\][^\|]{0,90}'` | `registers/source/GP-REG-032_v1.2_export_2026-09-17.md` | 14 distinct; same triage. |
| 7 | `grep -rnoiE '.{60}(Gamma\(\|Γ\()'` | `registers/`, `docs`, `claims` | one hit: `c_infinity = 2^(2/3) 3^(5/6) Γ(1/6)/(54 π^(3/2))`. |

`02_LEGACY_Q0_ARCHIVE` was not used as evidence, and the `99_DO_NOT_OPEN`
vault was not opened or referenced.

---

## 3. What the audit found

### 3.1 Settled exactly

| Key | Identity | Source | Verdict |
|---|---|---|---|
| `LPW-R05-ABS4` | `E(\|ξ\|+\|η\|)^4 = 12 + 16/π` | `lpw_fold_dispositions.json` | **REFUTED**, discrepancy exactly `16/π` |
| `LPW-R05-RHO4` | `E ρ^4 = 8`, `ρ = √(ξ²+η²)` | `lpw_fold_dispositions.json`, R05 Rayleigh row | **CONFIRMED_EXACT** |
| `GP-CLS-017-B-Z0` | `z₀(b) = E[Q²1{Q<0}] = (b²+2)Φ(b/√2) + √2·b·φ(b/√2)` for `Q ~ N(−b, 2)` | `closure_log.json`, row `GP-CLS-017-B` (EC-006) | **CONFIRMED_EXACT** |

**`E ρ⁴ = 8`.** `E[(ξ²+η²)²] = E ξ⁴ + 2 E ξ² E η² + E η⁴ = 3 + 2 + 3 = 8`. Fully
rational, no `π`.

Two consequences worth stating plainly, because both are easy to over-read:

* The same register row says *"E rho^4 = 8 is below old 12 + 16/pi budget."*
  That comparison is **true under the false value** (8 < 17.09…) and **remains
  true under the corrected value** (8 < 22.19…). The comparison survives the
  correction. **Surviving is not discharging.** The row's disposition, the
  author-side status of the R05 repair and its targeted-review requirement are
  unchanged by anything here.
* The exact pointwise bracket `ρ⁴ ≤ (|ξ|+|η|)⁴ ≤ 4ρ⁴` gives
  `8 ≤ E(|ξ|+|η|)⁴ ≤ 32`. **The false value 17.09… also satisfies it**, so that
  bracket could never have detected the defect. Pinned as a negative control
  (`test_NEGATIVE_CONTROL_rayleigh_bracket_does_not_detect_the_false_value`) so
  the weakness of the available check is on the record.

**`z₀(b)`.** Settled by exact comparison of coefficients in the basis
`{Φ(b/√2), φ(b/√2)}` over the ring `Q[√2]` — `Φ` and `φ` are not rational, but
their *coefficients* are. Substituting `Q = −b + √2 Z` and using the exact
recursion `T_j = (j−1)T_{j−2} − c^{j−1}φ(c)` gives
`b²T₀ − 2√2·b·T₁ + 2T₂`, whose `φ` coefficient is `2√2·b − 2c = 2√2·b − √2·b =
√2·b`. That matches the register exactly.

This is the **same shape** as the refuted identity: two contributions to one
basis coefficient must both be carried. Here the corpus carried both; in the
`12 + 16/π` case it carried one. A negative control fires if the `φ`
coefficient is doubled to `2√2·b` — the exact analogue of the dropped cross
term.

### 3.2 Found, derived, but deliberately **not** settled

| Key | Item | Verdict |
|---|---|---|
| `GP-DER-071-Z4` | *"planar closed form 30.5469700802916329… at b = 6/5"*, described elsewhere as a "truncated-normal fourth-moment formula" (`transition_log.json`, TR-P02-003) | **NOT_SETTLED_IDENTIFICATION_INFERRED** |

The same recursion gives, exactly,
`E[Q⁴1{Q<0}] = (b⁴ + 12b² + 12)·Φ(b/√2) + √2·b·(b²+10)·φ(b/√2)`, whose float
evaluation at `b = 6/5` is `30.54697008029163`, agreeing with the register
decimal to every digit double precision resolves.

**That is not a settlement, and the verdict says so.** Two reasons, both
recorded in the module:

1. The register states **only a decimal**. No closed form is written down, so
   there is no exact object to compare against. The identification of the
   register's constant with `E[Q⁴1{Q<0}]` under `Q ~ N(−b,2)` is **my
   inference** from the row's own wording plus numeric agreement.
2. Agreement of floating-point decimals is **NON-CERTIFYING**. It is evidence
   against a transcription slip and nothing more.

Settling it would need the `GP-DER-071` body, which is not in this repository.
`TR-P02-003`'s recorded status — *CANDIDATE — V2 SUBSTANTIALLY ADVANCED /
OUTSIDE-LINEAGE REVIEW OPEN* — stands exactly as written.

### 3.3 Found but not settleable — the honest negative result

`FOUND_NOT_SETTLED` in the module records six items of roughly the right shape
that this audit **could not** settle, each with its reason:

| Expression | Why not settled |
|---|---|
| `c_∞ = 2^(2/3) 3^(5/6) Γ(1/6)/(54 π^(3/2))` (`dispatch_queue.json`) | A **definition** of a constant, not an asserted equality between two independently computable quantities. Nothing to refute; `Γ(1/6)` is not exactly representable in rational arithmetic. |
| `E[Δ²] = 14.0881601505150626…< 15` (source export L203) | A decimal with no closed form and no stated law for `Δ` in the row. |
| `σ₅² = 945`, `L₆² = 11340` (TR-P02-003) | Attached to a conditioned fifth-derivative tail of a specific periodized field; reproducing them needs that field's covariance, which lives in `GP-DER-046/071`, not here. (`945 = 9!!` is suggestive of a Gaussian even moment. Suggestive is not settled.) |
| `E[J\|P] = (−b,0,0,0)`, `Cov = diag(2,2,2,6)`; nine-jet `(−b,0,0,0,−3b,0,0,0,3b)` | A specification of a conditional law, not a closed-form moment identity. Its first coordinate is consistent with the `Q ~ N(−b,2)` above — a coherence observation only. |
| `E[R^8] < ∞`, `E[W_r²] ≤ 16 M8 r⁴`, `E[W_r] ≥ c_Z r²`, `E[W_r²] ≤ C_W r⁴` | Inequalities with unspecified constants, not identities. No exact comparison exists. |
| `E[Q_L² 1{Q_L<0}] ∈ (0,∞)` (exact-torus, `GP-CLS-P01G-20260730`) | A positivity/finiteness statement about the **exact-torus** object, which the corpus explicitly distinguishes from the planar `Q ~ N(−b,2)`. No closed form asserted. |

**The honest negative result, stated plainly.** Beyond `E ρ⁴ = 8` and `z₀(b)`,
the audit found **no further asserted closed-form moment identity of the
`a + b/π` shape anywhere in this repository**, and **no second false one**.
Search #3 returned exactly one identity of that shape and it is the one already
named. That negative is bounded by search #1's limit: Drive *bodies* are not in
this repository, so a sibling written inside a Drive document would not be
visible to any search run here. The audit is complete over the repository and
incomplete over the Drive, and those are different statements.

---

## 4. Negative controls, and proof that they fire

`tests/test_identities.py` carries 39 tests. Seven are negative controls. Each
was run against a **deliberately broken copy** of `gaussian_moments.py` in a
scratch directory to prove it actually fires:

| Break applied to a scratch copy | Controls that fired | Suite result |
|---|---|---|
| drop the `(1,3)` cross term from the expansion | `dropping_one_cross_term_reproduces_the_false_claim`, `discrepancy_is_not_32_over_pi`, `sign_flip`, `swapped_basis_coefficients` | **16 failed**, 23 passed |
| swap the `{1, 1/π}` basis coefficients in the view | `swapped_basis_coefficients_are_rejected`, `rayleigh_bracket_does_not_detect` | **5 failed**, 34 passed |
| state the discrepancy as `32/π` | `discrepancy_is_not_32_over_pi`, `sign_flip` | **10 failed**, 29 passed |
| flip the sign of the discrepancy | `sign_flip_of_the_discrepancy_is_rejected` | **10 failed**, 29 passed |
| drop the `−c^{j−1}φ` term from the truncated-moment recursion | `truncated_second_moment_phi_coefficient_is_not_doubled` | **3 failed**, 36 passed |
| make the odd absolute moments rational (`s`-power 0) | `odd_moments_must_not_be_rational` | **25 failed**, 14 passed |
| make the `{1, 1/π}` view permissive (swallow `s¹`) | `basis_view_must_not_swallow_an_s1_term` | **3 failed**, 36 passed |

Two further structural controls run against the unbroken module:
`test_no_sibling_record_promotes_anything` pins the verdict vocabulary to
`{REFUTED, CONFIRMED_EXACT, NOT_SETTLED_IDENTIFICATION_INFERRED}` and fails on
any of `CLOSED`, `DISCHARGED`, `PROMOTED`, `CERTIFIED`, `RESOLVED`;
`test_no_verdict_depends_on_a_float` fails if any record's exact fields become
`float`.

---

## 5. What this does **NOT** establish

* **Refuting the identity does not invalidate any derivation that cited it.**
  It flags **every consumer for re-check**. The consumers are **not enumerated
  here** and must be enumerated separately. What the exact discrepancy does
  give a re-check is a **direction**: the claim understates the truth
  (`12 + 16/π < 12 + 32/π`). So a consumer that used the value as an **upper**
  budget for `E(|ξ|+|η|)^4` used a bound the true value violates, and that step
  is unsound; a consumer that used it as a **lower** bound used a true, merely
  non-sharp inequality, and that step survives on its own terms. **Which
  direction any given consumer needs is not decided here, and no consumer is
  classified here.**
* **No status label is promoted, closed, discharged or reclassified.** The
  `LPW_CONSTANT` disposition stands at `AMEND REQUIRED`; the R05 Rayleigh
  repair stands as author-side with targeted review required and does not
  inherit the prior external PASS; `GP-CLS-017-B` and `TR-P02-003` stand as
  their registers write them; all five D1 validity premises stand as
  `docs/OPEN_PROBLEMS.md` records them. Every label above is **transcribed**.
* **Confirming a sibling identity certifies nothing about the object that used
  it.** A true lemma inside an unreviewed derivation leaves the derivation
  unreviewed. `E ρ⁴ = 8` being correct says nothing about the `6.238e−44` /
  `6.239e−44` headline chain, the Rayleigh amplitude lemma, the tails/profile
  modulus, or any conditional bound in the R05 repair.
* **No decimal here is certified to any digit.** `22.18591635788…`,
  `3.230978535287005` and `30.5469700802916329` are all **NON-CERTIFYING**
  displays. Certifying them would need enclosures for `Φ` and `φ`, which this
  module does not supply. `math.gamma` is a float oracle; a float computation
  is not a certified bound.
* **This is not a certified bound, a Monte Carlo estimate, a fitted exponent, a
  smoke test, a session CLOSE or a registration.** It is exact arithmetic on
  finitely many rational coefficients, and that is all.
* **Nothing here relates the 2D upper track, the 2D lower track and the 3D
  lifetime track,** and nothing here composes any of them.
* **No prize problem is touched. None is solved.**
* **Green tests are not a mathematical review.** The derivations are written
  out in the module docstrings so a human can read and challenge them.

---

## 6. Running it

```bash
python3 research/identities/gaussian_moments.py   # prints the refutation + expansion
python3 -m pytest -q tests/test_identities.py     # 39 tests
```
