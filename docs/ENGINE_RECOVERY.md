# Recovering the frozen RN engine `d3_rn_unif.py`

Closes a caveat the sources record as open. Promotes nothing.

`LANE_RN_UNIF.md` (Drive `1dK4ZimCC8o9670K-9ZJS9Jtquid1CAA6`) caveat 5 says:

> Frozen engine `d3_rn_unif.py` is not in the RN_UNIF Drive folder; scripts
> import/monkey-patch it from tree — not downloaded here.

That is why `rnu_ds3.py` — the carrier `docs/OPEN_PROBLEMS.md` A5 names as
preferred — has been un-runnable: its first real statement is
`import d3_rn_unif as R`, and the module it imports was not in the repository.
This document records the recovery of that module and of everything it pins,
and reports the engine's structure for the other A5 lanes.

---

## 0. What this does NOT establish

* It closes, discharges, reduces, promotes and reclassifies **nothing**.
  `OBL-H5-JETMOD`, `OBL-H5-ZBAND`, `OBL-H5-REMOTE-THRESHOLD`,
  `OBL-D1-PROMOTE` and **both** pieces of `D3-LEMMA-RN-UNIF` are exactly as
  open after this work as before it. `D3-LEMMA-RN-UNIF` Piece 1 is OPEN and
  Piece 2 is OPEN — the annulus Riemann-sum driver is still unwritten. The
  receipts still carry `lemma_closed: false`.
* It does not patch the frozen engine. Nothing in `engine/rn_engine/frozen/`
  has been edited; a frozen body is never edited in place. The defects
  described in §3 are *reported*, not repaired.
* It does not run the engine and does not invoke its certifier. No number
  printed by that engine is quoted here as a result, and no bound is claimed
  from it.
* **Every computation inside the recovered engine is `mpmath` binary floating
  point at `mp.dps = 100` (set once, `d3_rn_unif.py` line 41, never changed).
  That is NON-CERTIFYING.** There is no `fractions.Fraction`, no
  `decimal.Decimal` and no interval arithmetic anywhere in the recovered
  bodies. High precision is not a certified enclosure. The word "exact" as the
  engine uses it means "the exact differentiation formula", evaluated in
  floating point — not exact arithmetic.
* The verification in §2 is about **bytes**. A byte-exact recovery is
  provenance. It is not a proof, not a bound, and not a review.
* It relates the 2D upper track, the 2D lower track and the 3D lifetime track
  in no way and composes none of them.
* Original prize problems solved: **0**.

---

## 1. Step 1 — what actually happened on download

**Answer: (b) on the raw bytes, (a) after a stated, digest-verified
reconstruction.** Both halves matter, so both are reported.

### 1.1 The fields, confirmed against `drive/source_map/Archive_Members.csv`

Every field below was re-read from the CSV in this repository, not taken on
trust from the task description.

| field | value |
|---|---|
| member path | `intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py` |
| payload SHA-256 | `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930` |
| bytes | 103,166 |
| reading copy | Drive doc `1zaumNVU8r1FKXp9DzNN7Y0Id3DRQBaQsI3QlOzFVd1A` |
| member occurrences | **6** rows, across **5** distinct carrier Drive ids |
| digest agreement | all 6 rows carry the same payload digest and byte count |
| scope holds | `[]` |
| context | `RESEARCH_SOURCE_CHECK_STATUS` |

The five carriers are `RN5_REPAIR_AND_ERRATUM_BUNDLE.zip`
(`1g5UP_KYlvPmx7LFwjGk6qK5QZLnL1hLj`),
`RN3-20260917-b9c2_v1.0_PROOF_CODE_AND_VERIFICATION.zip`
(`1z_zNdgxtbs7XWxLu56HN8_QbVXxdmMMq`, which carries the member twice — once
directly and once nested inside `round3/intake/CLOSE_prior.zip!`),
`CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip`
(`1WZfhLuZBEzvdUUkQw7v5JubmAiI1W4gt`),
`MATH-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip`
(`1TJr1awRjB0D9niq1Jyi6XGnJKuvW9Wmx`) and
`09152026OKComputer_Project_Gap_Closure.zip`
(`1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`). The engine bytes are unambiguous.

### 1.2 The raw download does **not** hash to the payload

Downloading Drive doc `1zaumNVU…` and base64-decoding gives:

| | raw reading copy | archived payload |
|---|---|---|
| bytes | **220,170** | 103,166 |
| sha256 | `cecb3c55ce9369ae9929165d9cd2d14dc2bec8f82bfa7dd9561a2d6830d934f0` | `85d7725f…0930` |

So the literal answer to Step 1 is **(b)**, and the reason is more interesting
than "encoding drift": the reading copy is **not a copy of that file at all**.
It is a Google Doc titled `READING RESEARCH_SOURCE_CHECK_STATUS 129` whose
`text/plain` export is an **ACCESS READING VOLUME** bundling **fourteen
unrelated sources**, each wrapped in a banner:

```
BEGIN SOURCE 85d7725f…0930 PART 1/1
Name: intake/rn_source/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py
Original bytes: 103166
Source SHA-256: 85d7725f…0930
Display encoding: UTF-8. Display line endings normalized; raw source identity remains above.
CONTENT START 85d7725f…0930-part1
…
CONTENT END 85d7725f…0930-part1
END SOURCE
```

The volume announces its own lossiness in that banner. The other thirteen
members are other people's files; storing this blob "as `d3_rn_unif.py`" would
have been simply wrong.

### 1.3 The de-transform, and why the payload comes back exactly

Slicing out the one delimited block still does not match: it is 103,276 bytes,
sha256 `6c14478f…0197` — 110 bytes long. The difference is entirely blank
lines. In the extracted block every maximal run of blank lines has **even**
length (107 runs of 2, one run of 4, no odd runs), which is what Google Docs
does to a source newline: it renders one as two. Halving every run and dropping
the volume's own trailing newline gives

* **103,166 bytes**,
* sha256 **`85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`** —
  an exact match with the source map, with the byte count, and with all five
  carriers.

The full rule, implemented in `engine/rn_engine/reconstruct.py`, is: strip the
UTF-8 BOM, rewrite CRLF as LF, slice the `CONTENT START` / `CONTENT END` block
for the wanted digest, halve every maximal blank run, join with LF, no trailing
newline. `extract()` returns **only** on a SHA-256 match; on a mismatch it
raises and says *DO NOT store these bytes as the payload*.

### 1.4 The reading copy is still a reading aid, and here is the proof

The transform above is a *hypothesis about how the bytes were mangled*; the
digest is the only thing that makes it trustworthy. Run against all fourteen
members of the same volume, it reconstructs **11 of 14** exactly and **fails on
3**:

| member | declared | reconstructed | why |
|---|---|---|---|
| `C097_COLLAR_GAMMA_BUDGET.md` | 2,593 | 2,605 | backslash escapes rewritten as text: the source's `\boxed{` appears in the export as `\x08oxed{` and `\frac` as `\x0crac` (2 occurrences each, +2 bytes apiece) |
| `LS-DER-026_native_text_export.txt` | 15,281 | 14,749 | 532 bytes short; content lost, not merely reflowed |
| `SIDE24_v5_2026-08-02/00_READ_ME_FIRST_V5.pdf` | 68,535 | 2,985 | a PDF has no meaningful `text/plain` export |

So the honest statement of the finding is:

> A Drive reading copy is a *reading volume*, not a payload. Its display
> transform is lossy in general — for LaTeX-bearing Markdown it destroys
> content irrecoverably — and it happens to be losslessly invertible for
> `d3_rn_unif.py`. That is a fact about this payload, established by its
> SHA-256, and it transfers to no other member.

### 1.5 `rnu_ds3.py` and the six pinned dependencies

`rnu_ds3.py` is a **native** Drive file (`14FIaDPHfbTTGQWzxZi8xKAzlKub5Iud8`,
`text/x-python`, 9,704 bytes), so its download decodes straight to the payload:
sha256 `bd3074fd900fc80bebdd430e34ae953f0a78685fff672758e7b927cc1e6d9421`,
matching `Files.csv`, `Payloads.csv` and `drive/inventory.jsonl`. It was **not**
already bound by `engine/carriers/` when this pass ran (no `MANIFEST.json`
existed; the `blobs/` directory held only
`63ef800a51969903__GP-DATA-214-v1.0_r_uniform_exact_endpoint_normalizer.py` and
`3693655f45d03d32__rnu_meanfix.py`).

Recovering the engine alone does **not** make it runnable. Lines 59–70 of
`d3_rn_unif.py` hash-pin six dependencies and `ck(...)`-fail-close on any
mismatch; lines 74–81 then re-open the sixth of them, `H3_RUNG_FLOOR.md`, to
parse `Z_lo` out of it and fail-close unless it reads
`7.7592917375327855e-3`. All six resolve
unambiguously in the source map (one payload digest each, across 4–5 carriers),
and all six were recovered the same way and verified against the engine's own
pins:

| pinned as | recovered blob | bytes | sha256 |
|---|---|---|---|
| `d3_perc.py` | `frozen/…/D3_percolation/d3_perc.py` | 40,337 | `5bc09241…dfa4` |
| `d3_amend.py` | `frozen/…/D3_percolation/d3_amend.py` | 21,619 | `12175217…b5b6` |
| `../H2_foundations/cov_exact.py` | `frozen/…/H2_foundations/cov_exact.py` | 21,961 | `f08c1c5f…e783` |
| `../H2_foundations/pin_transform.py` | `frozen/…/H2_foundations/pin_transform.py` | 17,623 | `c6988ac7…fd41` |
| `../H2_foundations/reg_lemmas.py` | `frozen/…/H2_foundations/reg_lemmas.py` | 10,582 | `e7005a5a…3450` |
| `../H3_closure/H3_RUNG_FLOOR.md` | `frozen/…/H3_closure/H3_RUNG_FLOOR.md` | 7,003 | `6347275d…0dfa` |

**Scope hold carried forward.** `d3_perc.py` carries `Q-RN5-MOMENT-001` in
`drive/source_map/Payloads.csv`: *"`envelope_v`; `window_cap_env` and consumers
relying on the claimed polarity-safe upper bound. Other functions are outside
this finding."* Recovering the file does not lift that hold, and
`research/rn/moment_envelope.py` is the exact-rational statement of why.

Eight payloads are now under `engine/rn_engine/frozen/`, laid out at their
archived paths. Running the engine additionally requires `mpmath`, which this
repository does not guarantee — and running it is out of scope here anyway.

---

## 2. What was verified, and the negative controls

`engine/rn_engine/verify_recovery.py` (standard library only) runs four checks:

* **A — blob integrity.** Every file named in `engine/rn_engine/BINDING.json`
  hashes to the recorded digest and byte count.
* **B — source-map agreement.** Each digest appears in
  `drive/source_map/Archive_Members.csv` or `Files.csv` with the same byte
  count, and every occurrence of that payload in the source map agrees on the
  digest (so an ambiguous member name cannot pass as "recovered").
* **C — engine self-pin.** The `_PINS` block is parsed back **out of the
  recovered `d3_rn_unif.py`** and its six digests are compared with the six
  recovered dependency blobs. This anchor is independent of the source map: it
  is the engine's own fail-closed contract.
* **D — negative controls** (`--selftest`), on a synthetic reading volume built
  in-process so they run offline.

```
$ python3 engine/rn_engine/verify_recovery.py --selftest
ok    A blob integrity
ok    B source-map agreement
ok    C engine self-pin
ok    D negative controls

8 carriers bound, 0 problems
```

**Each control was proved to fire, by running it against a deliberately broken
copy of the tree** (copies made in a scratch directory; nothing in the
repository was damaged):

| break | result |
|---|---|
| one byte inside the frozen engine flipped — `_LT = 24` → `_LT = 25`, i.e. the torus period itself | **A fails**: `RNENG-01: sha256 38910c31… on disk, 85d7725f… recorded` |
| self-consistent forgery: `cov_exact.py` mutated **and** `BINDING.json` rewritten so the ledger agrees | A passes (as it must), **B fails** (`digest dec235c0… not in the source map`) and **C fails** (`engine pins … = f08c1c5f… but the bound blob is dec235c0…`) |
| the blank-run halving removed from `reconstruct.py` (`// 2` dropped) | **D fails** on CONTROL-0 (the honest volume no longer reconstructs), CONTROL-2 and CONTROL-3b |
| the SHA-256 guard inside `extract()` disabled | **D fails** on CONTROL-1 (altered content accepted) |
| one digit of one recorded digest changed (CONTROL-5) / recorded byte count changed (CONTROL-6) / a dependency digest set to contradict the engine's pins (CONTROL-7) | each is detected by A, A and C respectively |

The second row is the one worth keeping: a forgery that rewrites *both* the
artifact and its ledger still dies, because the source map and the engine's own
hash pins are two anchors this directory does not own.

**One documented boundary, recorded rather than hidden.** Halving an odd blank
run rounds down, so appending a single trailing blank line to a volume does not
change the reconstruction (CONTROL-3b pins this behaviour so a future change to
the rule cannot pass unnoticed). The rule is therefore not injective: it can
*fail* to invert a volume, which is what happened to 3 of the 14 members in
§1.4. It can never invent a payload, because `extract()` returns only on a
digest match.

`engine/rn_engine/` is the only place these controls live. The engineering rules
for this pass put tests under `tests/`, but `tests/` is not a path this pass
owns, so the controls ship inside the owned directory as a runnable script.
**Follow-up:** whoever owns `tests/` can add a two-line wrapper calling
`verify_recovery.main(["--selftest"])` to bring it into CI.

---

## 3. Step 2 — the engine's structure, with line references

All line numbers refer to
`engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py`
(2,241 lines, sha256 `85d7725f…0930`). Quotations are verbatim.

### 3.1 `kplane(ex, ey, dx, dy)` — the plane kernel

```python
 97: C_KERN = mpf(1)   # the rung kernel is the unit separable Gaussian e^{-d^2/2}
143: def kern(d2):
144:     return C_KERN * mpmath.exp(-d2 / 2)
146: def kplane(a1, a2, dx, dy):
147:     d2 = dx * dx + dy * dy
148:     return ((-1) ** (a1 + a2)) * he(a1, dx) * he(a2, dy) * kern(d2)
```

The base kernel is the **unit separable Gaussian** `K(x) = exp(-|x|²/2)` with
`C_KERN = 1`. The closed form is given by the code itself:

> `kplane(a1, a2, dx, dy) = (-1)^(a1+a2) · He_{a1}(dx) · He_{a2}(dy) · exp(-(dx²+dy²)/2)`

`He` is the probabilists' Hermite polynomial, computed by the three-term
recurrence `he(n, x)` at lines 100–106 (`h0, h1 = h1, x*h1 - (k-1)*h0`). This
is exactly `∂^{(a1,a2)} K`, since differentiating a unit Gaussian produces
Hermite polynomials with that sign.

**`ex`, `ey` are the two components of the derivative multi-index** — the total
number of `∂x` and `∂y` derivatives applied to `K`. In `kdcov` they arrive as
`a[0] + b[0]` and `a[1] + b[1]`, the sum of the two field jets' multi-indices;
in `rnu_ds3._kd_multi` they arrive as `a[0] + b[0] + Σ e[0]`, adding the extra
differentiations in `y` that the DS3 lift needs. Line 152:

```python
152: def kdcov(a, b, dx, dy):
153:     """cov(d^a f(p), d^b f(q)) with x = p - q = (dx,dy): (-1)^|b| d^{a+b}K(x),
154:     plane kernel + first-image shell; residual tail bounded by tail_bound."""
```

A companion `he_abs(n, t)` is the all-plus-coefficient majorant of `|He_n|`
(`_HE_ABS` at 108–121, `_he_abs_poly` at 123–138, `he_abs` at 140–141), used
only inside the envelopes.

### 3.2 `_IMG`, `_LT`, and the truncated image sum

```python
 98: _LT = 24
150: _IMG = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)]
155:     s = kplane(a[0] + b[0], a[1] + b[1], dx, dy)
156:     for (i, j) in _IMG:
157:         s += kplane(a[0] + b[0], a[1] + b[1], dx + _LT * i, dy + _LT * j)
```

* **`_LT = 24`: yes, the side-24 torus period.**
* `_IMG` is the **first image shell only** — the 8 points of
  `{-1,0,1}² \ {(0,0)}`. Counting the untranslated term, **9 plane-kernel
  evaluations are summed per covariance entry**, and the same 9 appear in
  `kdcov_d` (338–342), in `form_hess_fn` (352–376), in `kdcov_dd`, and in
  `rnu_ds3._kd_multi`.
* **Yes, the image sum is truncated**, at `|n|∞ = 1`.

**Where is the truncation error accounted for?** This is the heart of
`OBL-H5-JETMOD`'s "lattice-tail constants re-certified uniformly in the band",
so it gets a precise answer rather than a summary.

```python
160: def tail_bound(order):
161:     """|sum_{|n|>=2} d^order K(x + 24n)| for |x| <= 17: images at separation
162:     >= 31: bounded by shell sums of the plane kernel."""
163:     tot = mpf(0)
164:     for m in range(2, 8):
165:         rho = m * _LT - 17
166:         tot += 8 * m * he_abs(order, mpf(rho)) * kern(mpf(rho) ** 2)
167:     return tot
...
181: ck(_mx < mpf('1e-25'), "kernel closed-form validation gap %s" % mpmath.nstr(_mx, 3))
182: for _o in range(5, 11):
183:     ck(tail_bound(_o) < mpf('1e-60'), "image tail too large at order %d" % _o)
```

So the truncation is **not unaccounted for — but it is not carried either**.
Five things must be said exactly:

1. **The tail is never added to the value.** `kdcov` returns the 9-point sum.
   `tail_bound` is a separate *side assertion* at lines 182–183 that the
   discarded remainder is below `1e-60`. No interval is widened, no enclosure
   propagates; the residual is argued to be negligible, not enclosed.
2. **The bound is conditioned on a point separation.** `|x| <= 17` is hard-coded
   into `rho = m * _LT - 17` at line 165. It is a bound for points of the far
   zone, not for `r` ranging over a band — *precisely* the defect
   `OBL-H5-JETMOD` names ("LAT's tail bound currently certifies at point
   separations"). A band version would have to take the worst `|x|` over the
   band; the constant `17` would have to become a band-dependent quantity, and
   nothing in this engine does that.
3. **The tail bound is itself truncated, with no remainder.** `for m in
   range(2, 8)` sums shells `m = 2 … 7` and silently drops every `m ≥ 8`. The
   dropped terms are minute (`rho = 8·24 − 17 = 175`, so `kern(rho²) ≈
   e^{-15312}`), but the code contains no term for them, so the printed
   "bound" is formally a partial sum.
4. **Only orders 5–10 are checked, and only as a scalar order.** The check
   `for _o in range(5, 11)` covers the derivative orders `kdcov` is called at
   in this engine (`|a|+|b| ≤ 4`, plus up to 3 more through `kdcov_dd` and the
   DS3 lift). But the true tail term is
   `|He_ex(dx') · He_ey(dy')| · exp(-(dx'² + dy'²)/2)` in **two** axes, while
   `tail_bound` evaluates a **one**-axis majorant `he_abs(order, rho) ·
   kern(rho²)`. The docstring asserts the reduction; the body does not prove
   it. That is an argument a band certification would have to write out.
5. **It is all `mpmath` floating point.** `tail_bound` is NON-CERTIFYING like
   everything else here.

Only in the moment-series envelope is an image allowance actually **added into**
a returned quantity — `env_form`, lines 417–421:

```python
    # image allowance: image separations >= 24 - d - R/2 (>= 11.97 on the torus)
    img = mpmath.fsum(abs(c) for (p, a, c) in FORMS[k])
    rimg = 24 - d - R / 2
    img_env = (img * 8 * he_abs(6 + qord, rimg) * kern(mpf(rimg) ** 2))
    return tot + rem + img_env
```

Note this allowance also covers the **first** shell only (the factor `8`), and
that the parenthetical "`>= 11.97 on the torus`" holds at `d ≈ 12`; at the
outer edge of the polar cover, `d = 17`, the same expression gives
`rimg = 6.975`. Whether the constant was meant to be read at `d ≈ 12` only is
not stated in the body.

**Since 2026-09-22 that question has a number attached to it.**
`research/rn/env_form_reference.py` assembles this same three-part shape in
exact rationals with certified enclosures, so the two parts can be weighed
against each other directly. They move in opposite directions, and that much is
a property of the shape rather than of any moment table: the moment series
carries `exp(-d²/2)` against a polynomial in `d` and decays, while the image
allowance sits at `rimg = 24 − d − R/2` and therefore *grows* as `d` grows. A
crossover exists for every moment table; only its location depends on the
table.

On the module's own **reference** moments at `qord = 2` — which are not the
program's `MOMS[k]`, and the location is not transferable to them — the
measured ratio of image allowance to moment series is `1.0e−68` at `d = 5`,
`0.77` at `d = 12`, and `5.5e+48` at `d = 17`, crossing between
`d = 12.0115565` and `12.0115566`. The total bottoms out near `d = 12.0111` and
by `d = 23` is about `5.3e+24` times that minimum; past `d = 24 − R/2 = 23.975`
the image separation is negative, and the reference module refuses that input
rather than return the number `he_abs`'s `abs(t)` would otherwise produce.

Two things follow, and only two. First, "take `d` larger to get a smaller
bound" stops working at a computable place, so the best bound of this shape is
the one at its minimum rather than the one at the largest `d` in range.
Second, on this reference data the crossover falls **inside** the lane's own T4
region `d ∈ [5, 17]` — negligible where the push evaluated, the whole bound at
the top. The near-coincidence between the crossover at `12.0116` and the `d`
at which the parenthetical's `rimg ≥ 11.97` stops holding (`d ≈ 12.005`) is
noted and **not** concluded from: the crossover moves with the moment table and
the parenthetical does not, so the two agreeing here is a fact about the
reference data and not a derivation. Nothing above is a statement about the
program's envelope, and no status turns on it.

**What a certified interval version needs**, stated without overclaiming: a
proved remainder for `Σ_{|n| ≥ 2}` that is (i) added into the enclosure rather
than asserted negligible, (ii) uniform over an `r`-band rather than pinned to
`|x| ≤ 17`, (iii) complete in the shell index `m` rather than truncated at 7,
and (iv) two-axis rather than reduced to a single `he_abs` without proof.
`research/interval/` supplies the arithmetic in which such a remainder could be
carried. It does not supply the remainder, and nothing here discharges
`OBL-H5-JETMOD`.

### 3.3 `kdcov` / `kdcov_d` — covariance entries and their derivatives

* `kdcov(a, b, dx, dy)` (152–158) is the periodized covariance entry
  `cov(∂^a f(p), ∂^b f(q))` at separation `x = p − q`, equal to
  `(−1)^{|b|} ∂^{a+b} K(x)` summed over the 9 images, with the sign
  `((-1) ** (b[0] + b[1]))` applied at line 158.
* `kplane_d(a1, a2, dx, dy, e)` (332–336) differentiates once more in slot `e`
  by *raising the multi-index*: `e == 0` → `kplane(a1 + 1, a2, …)`, else
  `kplane(a1, a2 + 1, …)`. Exact, not finite difference.
* `kdcov_d(a, b, dx, dy, e)` (338–342) is the same 9-image sum built from
  `kplane_d`, with the same `(−1)^{|b|}`.
* `kdcov_dd` (1271–1276) goes to second order the same way; `form_eval` /
  `form_grad` / `form_hess_fn` (326–376) contract these against the residual
  forms `FORMS[k]`, and `form_thrd` (1257–1270) gives the third-derivative
  Frobenius norm.
* `kc(d1, d2, p1, p2)` (276–277) is the entry accessor the station uses.

Every one of these truncates the image sum at the first shell.

### 3.4 `kappa_far` — what is actually computed

There is **no function named `kappa_far`**. Three things carry the name:

* **`kappa_pieces_fast(fs, v)` (658–747)** — the scalar assembly of the pieces
  from the fast station, with `Z_LO` denominators, ending at line 723 with
  `kap_far = Rpg * Rwm * ((1 + kap_pair) * (1 + kap_y) + kap_cross) - 1`.
* **`kappa_far_ds(y, v)` (1144–1244)** — the same assembly rebuilt in DS/DM
  arithmetic so value, gradient and Hessian come out together, ending at 1234
  with the identical expression in `DS`. This is the function `rnu_ds3.py`
  lifts to third order. Its five factors are: `Rpg` (the gradient-density
  ratio against `pgrad0 = 1/(2π a2)`), `Rwm` (the mark-window ratio of two
  `Φ` differences), `kap_pair` (`hL2 · sqrt(chi2) / Z_LO`, line 1196),
  `kap_y` (the `W2`/Bures piece, via a Denman–Beavers matrix square root) and
  `kap_cross` (`sqrt(3) · EW4 · Eq2^{1/4} · sqrt(tau) / (Z_LO · m_sad)`).
  Immediately after the definition, line 1246 evaluates it at `(5, 0)` and
  lines 1251–1253 fail-close unless it reproduces the frozen v2 value `0.677284905`
  to `1e-6` — which sits just under `KAP_BUDGET = mpf('0.68')` (line 2144).
* **`kappa_far_point(y, d)` (788–825)** — despite the docstring "certified
  (kap_far, grad-bound, hess-bound) at point `y`", this function is **not
  finished**. Line 818 contains `rv = mp.matrix([v - (fs['mean_v'](0)[6] -
  fs['mean_v'](0)[6]), 0, 0])  # placeholder, refined below`, whose inner
  expression is identically zero, and the function returns raw ingredients
  (`fs, kp, tau0, gtau, htau, Bnorm, dE0, dE1, dE2`), not the triple its
  docstring promises. The working per-point bound is `kp1_point` (1722–1837).

`kappa_far_ds` is also the entry point `rnu_ds3.py` exercises: after
`install()` rebinds `R.DS`, `R.DM.inv`, `R.DM.det_ds`, `R.bures_trace_ds`,
`R.de1`, `R.de2`, `R.d_entry` and `R.dconst`, `R.kappa_far_ds(y, v)` returns a
`DS3` carrying `t_xxx, t_xxy, t_xyy, t_yyy` as well.

### 3.5 `chi2_grad_bound` — where the nineteen orders of magnitude come from

`chi2_grad_bound(fs, kp, y, d)` is at **line 1683**. Its docstring claims
"tight certified bound of `|grad chi2|` at `y`: exact solve-products and exact
directional derivatives throughout (no norm products)". The lane memo
(CL-RNU-001) records that it returns **~1.57e14** where the true `|∇χ²|` is
**~1.563e-5** — a **~1e19 slack**.

The whole of the function's inexactness is these four lines:

```python
1709:        glogdet = abs(qtr(Stinv, dS))
1710:        gdetM = abs(qtr(At, dS))
1716:        gcq = abs(2 * (w_c.T * dc)[0] - (w_c.T * dMt * w_c)[0])
1717:        gkk = abs(2 * (w_m.T * sdmu_v)[0] - (w_m.T * dS * w_m)[0])
1718:        glq = glogdet + mpf('0.5') * gdetM + gcq + gkk
1719:        g2 += glq ** 2
1720:    return (1 + kp['chi2']) * mpmath.sqrt(g2)
```

**The single sentence.** `χ² = exp(log q) − 1` (lines 1187–1189), where `log q`
is a **sum of four signed terms** — `− log det S_pair`, `− ½ log det M`, `+ q₁`
and `− kk`; each of `glogdet`, `gdetM`, `gcq`, `gkk` is the *exact* derivative
of one of those four terms, but line 1718 replaces the signed sum of the four
by **the sum of their absolute values**, and the nineteen orders of magnitude
are exactly the cancellation that the four `abs()` calls throw away.

Why the individual terms are astronomically large while their signed sum is
not: the pair-Hessian conditional covariance is near-singular. The engine's own
header says so, lines 3–8:

```
# Piece 1 (RIGIDITY-DECOUPLING, foundations): … The pair-Hessian
# conditional covariance Spair is rigid (lambda_min = 2.6e-10 at the rung):
# entrywise envelopes overbound tau by x5.3e3 (||Delta||_F^2/lambda_min =
# 1.881 vs true tau = 3.53e-4) because the pins EXPLAIN the pair block's
# rigid directions.
```

Every one of the four terms is `dS` conjugated by `S_pair⁻¹`, whose norm is
`1/λ_min ≈ 3.8e9`. The `gdetM` term carries `S_pair⁻¹` **twice** — line 1700 is

```python
1700:    A = Spinv * Minv * Spinv
```

— so it scales like `1/λ_min² ≈ 1.5e19`, the same order as the reported slack.
But `log q` is a **log-likelihood ratio**: the near-null directions of `S_pair`
appear in its numerator and its denominator alike — that is what
"the pins EXPLAIN the pair block's rigid directions" means — so they cancel in
the signed sum, leaving `|∇χ²| ≈ 1.563e-5`. The `x5.3e3` overbound the header
reports for `tau` is the same mechanism at first order; here it is applied at
gradient level with two inverse factors instead of one.

**Flagged honestly.** I did not run the engine, so I did not *measure* the four
terms. The identification is structural: it comes from the code at lines
1700 and 1709–1720, from the engine's own stated `λ_min = 2.6e-10`, and from
the arithmetic coincidence that `λ_min⁻² ≈ 1.5e19` sits at the reported slack
ratio. A reader who wants it settled should print the four terms separately at
`(5, 0)` — that is a one-line diagnostic, and it is *not* done here because it
would mean running the engine.

**And it gets worse downstream, conditionally.** The bound is consumed at line
1796:

```python
1796:    gkp = kp['kap_pair'] * gchi2 / (2 * kp['chi2']) if kp['chi2'] > 0 else mpf(0)
```

If `chi2` is small — which is what a near-unity likelihood ratio in the far
zone means, though I did not evaluate it — this divides the already-loose
`gchi2` by `2·chi2`, amplifying it again before it reaches `gB`, `g0` and the
cell sup-bound at line 2163
(`sup = kap0 + g0*hw + SAFE_H*h0*hw²/2 + SAFE_T*t0*hw³/6`). A `g0` of that size
cannot close a cell against `KAP_BUDGET = 0.68` at any `hw` the bisection is
allowed to reach (`hw < 4e-4` raises `FAIL-CLOSED`, line 2170). That is the
mechanical reason the lane's certifier "never invoked" — and see §3.8.

**Also note:** `chi2_grad_bound` calls `mean_grad_exact` at line 1704. The next
section shows that function is missing terms. A "certified bound" fed a
defective derivative is not merely loose — its certification is **void**, not
just slack. Fixing the `abs()` cancellation without fixing §3.6 would produce a
tight number that still is not a bound.

### 3.6 `mean_grad_exact` — the missing chain-rule terms, identified exactly

CL-RNU-001 records: *"E-RNU-1: `mean_grad_exact` missing chain-rule terms (fix
validated, not patched into frozen engine)."* The code shows which terms.

The conditional mean is built in `station_fast_full` (615–655) as

```
mu(v) = TC·W6  +  TY6 · YY6⁻¹ · rv,
   TY6 = TY − TC·G6inv·YCᵀ,   YY6 = YY − YC·G6inv·YCᵀ,   rv = (v,0,0) − YC·W6
```

so the product rule needs `dTY6 = dTY − dTC·G6inv·YCᵀ − TC·G6inv·dYCᵀ`.

`mean_grad_exact` (1641–1681) builds its `dTY6` surrogate at lines 1653–1658:

```python
1654:            if pi == 'Y':
1655:                continue
1656:            for j, (pj, dj) in enumerate(YJET):
1657:                TY6g[e][i, j] = -kdcov_d(_GI[di], _GI[dj],
1658:                                         PTS[pi][0] - y[0], PTS[pi][1] - y[1], e)
```

That is **`dTY` alone**. Both residualization terms are absent, and the
`pi == 'Y'` rows are additionally left identically zero — yet for those rows
`TY6[i,j]` still moves with `y` through `TC` and `YC`, even though the raw
`TY[i,j] = kc(di, dj, y, y)` is constant. `dYY6` and `drv` *are* differentiated
correctly at lines 1676 and 1678, so the omission is specific to `TY6`.

**The engine already contains the right formula, 200 lines later.** In
`cov9_grad_exact`, line 1882:

```python
1882:        dTY6 = dTY[e] - (dTC[e] * G6inv * YC.T + TC * G6inv * dYC[e].T)
```

and `ty6_derivs` (1300–1329) likewise carries the `− TCG[i,:]·dYC` term, at
lines 1320–1321. So `mean_grad_exact` is not using an approximation the engine
lacks — it is using an incomplete copy of a rule the engine writes out
correctly in two other places.

**Independent corroboration, flagged as such.** While this pass was running, a
concurrent workflow (which owns `engine/carriers/`, not this directory) placed
`blobs/3693655f45d03d32__rnu_meanfix.py` in the tree. That artifact is
presumably the "fix validated, not patched" object CL-RNU-001 names. Read-only,
its `mean_grad_fixed` computes
`dTY6 = dTY - dTC * G6inv * YC.T - TC * G6inv * dYC.T` and its docstring says it
*"includes the `TC G6inv dYC^T` term of `dTY6` for all rows and the
`dTC G6inv YC^T` term for Y-target rows (both missing in the engine's
`mean_grad_exact`)"* — the same two terms identified above, found independently
from the frozen source. That file was mid-write by another workflow; it is
cited as corroboration, not as authority, and it has not been verified here.

**Do not patch the frozen body.** The repair belongs in a sibling module that
rebinds the symbol, exactly as `rnu_ds3.py` already does for eight other
symbols. `engine/rn_engine/frozen/` stays byte-identical to the archive.

**A second, separate observation — my own reading, unverified.**
`ty6_derivs` skips `pi == 'Y'` rows at line 1314 (`if pi == 'Y': continue`),
leaving `dTY6[e][i, j]` zero for `i ∈ {6,7,8}`; `tau_bounds_full` then reads
exactly those rows at lines 1397–1400 (`dm_i = [[dTY6[e][6 + i, :] …`). I did
not run the engine and do not assert this is wrong — the Y-row residual term
may be zero for a reason the code does not state. It is recorded here as a
question for whoever patches the engine, not as a finding.

### 3.7 `env_form`, `env_tau`, `T4_form`

* **`env_form(k, gamma, d, qord)` (381–421)** — "certified bound of
  `|d^q F_{R_k,gamma}(y)|` for `|y| >= d`, `|q| = qord`". Three additive
  parts: a termwise exact-moment series over `MOMS[k]` scaled by `kern(d·d)`;
  a Taylor remainder at order 9 (the moments are carried to order 8), using
  `he_abs` majorants along the segment at radius `rho = d − R/2`; and the
  first-shell image allowance quoted in §3.2. This is the *only* place a
  lattice-tail allowance is added into a returned value. It is `mpmath`
  floating point, so "certified" here means "derived by an envelope argument
  and evaluated in floating point", not enclosed.
* **`env_tau(d)` (2120–2141)** — the crude theta-free envelope of `tau` over
  `{|y| ≥ d}`, assembled from `env_form(k, ·, d, 0)` with a reduction
  `redmax`, a per-`k` floor `lam = LAM0[k] - redmax`, and a cross allowance
  through `env_TY6(d)`. The fail-close the lane memo records is line 2131:

  ```python
  2131:        ck(lam > LAM0[k] / 2, "envelope lambda floor collapsed k=%d d=%s"
  2132:           % (k, mpmath.nstr(d, 4)))
  ```

  So `env_tau(5)` fail-closing (λ₀ floor collapsed) is this `ck`, fired because
  at `d = 5` the reduction `redmax` eats more than half of `LAM0[k]`. The scope
  hold stands: **do not use `env_tau` on the zone boundary.** The identical
  guard exists on the net path at line 550.
  Worth recording: **`env_tau` is defined but never called** anywhere in the
  frozen body. The header at line 16 describes an envelope zone `d >= d_env`,
  but no numeric `d_env` exists in the file and no envelope-zone driver is
  wired; the polar cover in `run_certification` runs the pointwise path over
  the whole of `d ∈ [5, 17]`.
* **`T4_form` is not in this file.** Neither `T4_form`, nor `T4_kap`, nor
  `C_comp` appears anywhere in the 2,241 lines. The EXECUTE freeze list's
  "valid T4 from `env_form`, not measured-scale" therefore refers to machinery
  that lives elsewhere — the sources put it in `rnu_t4.py`, which
  `docs/OPEN_PROBLEMS.md` §E lists under **`CANNOT_VERIFY`**. Anyone planning
  work against the EXECUTE freeze list should know that recovering this engine
  does **not** recover `T4_form`; the fourth-order envelope machinery is a
  separate, still-unverified carrier. What this engine supplies toward it is
  `env_form` and the DS/DM arithmetic that `rnu_ds3.py` lifts to third order.

  Two corrections to that sentence, both made 2026-09-22. **Until then it read
  "`env_form` (orders 0–2 in `qord` as used)".** "As used" reads as "as
  exercised", and it is not: `env_form` is itself unreachable from this
  module's top-level execution, and so are all three of its in-file call sites
  — `env_tau` and `env_small` (lines 478–483) and `bounds_point` (556–557),
  the last reached only from `kappa_far_point`, which has no call site either.
  Orders 0–2 are the orders appearing in code no run of this body reaches. And
  the only code in this repository that *does* call `env_form` calls it at
  **orders 0 through 4**: the carrier
  `engine/carriers/blobs/7b7cc46ba5605250__rnu_t4_push.py` does
  `import d3_rn_unif as R` and evaluates `R.env_form(k, g, d, q)` over
  `for q in range(5)`.

  **The never-called list in this document is not complete, and §3.8's heading
  promises that it is.** Computed with `ast` over the body's own module-level
  entry points, **22 of its 79 top-level definitions are unreachable**:
  `CertStats`, `Hn_diag`, `_Refine`, `bounds_point`, `certify_cell`,
  `dE_crude`, `d_entry`, `dm_vec_col`, `env_TY6`, `env_form`, `env_m`,
  `env_small`, `env_tau`, `form_thrd`, `kap_fn_p1`, `kappa_far_point`,
  `kp_msad`, `point_pieces`, `qblock`, `run_certification`, `spair_dir`,
  `wick4_grad`. This document named three of them; `certify_cell`, whose
  signature §3.8 quotes at line 2158, is among the nineteen it did not.
  `tests/test_frozen_reachability.py` recomputes the set and fails if it
  changes, so a future transcription cannot wire one in quietly.

  "Unreachable" here means *from this module's own top-level statements*, which
  is not the same as never called: the T4 push carrier above calls `env_form`,
  and `engine/rn_engine/frozen/RN_UNIF_2026-09-16/rnu_ds3.py` rebinds
  `R.d_entry` in its `install()`. An unreachable definition is not a defect —
  this body is a snapshot of work in progress and says so — and none of this
  moves any status. The frozen body is not edited; it is read.

### 3.8 The certifier, and a flat statement of what is not called

```python
2144: KAP_BUDGET = mpf('0.68')
2145: CELL_MARGIN = mpf('0.0002')
2146: SAFE_H = mpf(3)     # absorbs cell-scale variation of the crude hess scales
2147: SAFE_T = mpf(9)     # absorbs cell-scale variation of the crude third scales
2158: def certify_cell(y, d0, hw, budget, depth, stats, kap_fn):
2163:     sup = kap0 + g0 * hw + SAFE_H * h0 * hw ** 2 / 2 + SAFE_T * t0 * hw ** 3 / 6
2184: def run_certification(kap_fn, budget, tag):
2186:     {y in T^2 : |y| >= 5} (polar cover to d = 17 >= 12*sqrt(2))
2227: def kap_fn_p1(y, d0):
```

`run_certification` builds a polar cover from `d = 5` to `d = 17` in steps of
`0.5` with 96 base angles and bisects into four on refinement, fail-closing at
`hw < 4e-4`. `SAFE_H = 3` and `SAFE_T = 9` are fudge factors "absorbing
cell-scale variation", not certified constants.

**`run_certification` is defined and never called. `kap_fn_p1` is defined and
never called.** The file's last executable statement is the probe loop at lines
2231–2241, which evaluates `kp1_point` at four representative points and
asserts `ck(_k0 < KAP_BUDGET, "probe point already over budget")`. That is the
literal form of CL-RNU-001's "certifier never invoked": the driver call is
simply absent from the frozen body.

### 3.9 The arithmetic

* `mp.dps = 100`, set once at **line 41** and never changed anywhere in the
  file. There is no second precision setting, no precision sweep, and no
  residual-based precision certificate.
* **No part is exact-rational.** `fractions`, `decimal` and interval arithmetic
  do not occur. Everything — kernel, station, Schur complements, Wick moments,
  Denman–Beavers square roots, envelopes, tail bounds, the certifier — is
  `mpmath` binary floating point.
* `rnu_ds3.py` is the same: `DS3` stores ten `mpf` slots and validates against
  the engine's `DS` to `~1e-80` and against central finite differences with
  `h = 1e-12`. A finite-difference agreement is a consistency check, not a
  bound.
* The engine's fail-closes use absolute thresholds against floating-point
  residuals (`1e-25` kernel gap, `1e-60` image tail, `1e-80` parity and
  inversion residuals, `1e-90` Denman–Beavers convergence). These are
  smoke-test guards. **None of them is a certified enclosure**, and the
  program's own standing rule applies: a high-precision float computation is
  not a certified bound.

---

## 4. Step 3 — the consequence, stated flatly

Recovering this engine:

* **does not** close `D3-LEMMA-RN-UNIF` Piece 1;
* **does not** close Piece 2 — the annulus Riemann-sum driver is still
  unwritten, and `docs/OPEN_PROBLEMS.md` A5's instruction to schedule it
  explicitly rather than hide it under a T4 push is untouched;
* **does not** discharge `OBL-H5-JETMOD`, `OBL-H5-ZBAND`,
  `OBL-H5-REMOTE-THRESHOLD` or `OBL-D1-PROMOTE`;
* **does not** patch the frozen engine. The defects in §3.5 and §3.6 are
  reported. `engine/rn_engine/frozen/` is byte-identical to the archive and
  must stay that way;
* **does not** produce or endorse any number. The certifier was not invoked and
  no bound is claimed from it;
* **does not** make anything certified. The engine is `mpmath` floating point
  end to end.

What it does: **it makes a carrier readable, and — the digests having matched —
re-runnable**, subject to two conditions the caveat did not mention. Running
`rnu_ds3.py` needs (i) all six hash-pinned dependencies present at their pinned
digests, which is why they were recovered too, and (ii) `mpmath`, which this
repository does not guarantee. With those, `import d3_rn_unif` resolves and the
`install()` monkey-patch has something to patch.

That is all it does.

---

## 5. Follow-ups for other owners

1. **Merge into the carrier manifest.** `engine/rn_engine/BINDING.json` holds
   eight records in the same shape as `engine/carriers/MANIFEST.json`
   (`carrier_id`, `title`, `drive_id`, `drive_path`, `bytes`, `sha256`,
   `blob_stored`, `blob_path`, `lane`, `computes`, `arithmetic`, `certifying`,
   `certifying_note`, `authority_tier`, `source_status`, plus `recovery` and
   `scope_holds`). They were **not** written into that manifest because another
   workflow owns it and a concurrent write would corrupt it. **Merging these
   records into `engine/carriers/MANIFEST.json` is a follow-up step for whoever
   owns that file.** At the time of writing, `MANIFEST.json` did not exist and
   `engine/carriers/blobs/` held two unrelated blobs, so nothing was
   duplicated. `drive_id` is `null` for the seven zip-member payloads — they
   have no standalone Drive file id — and the Drive objects actually used are
   under `recovery.reading_copy_doc_id` and
   `recovery.archive_carrier_drive_ids`.
2. **CI wrapper.** Add a `tests/` wrapper calling
   `verify_recovery.main(["--selftest"])` (see §2).
3. **Recovering other payloads.** `reconstruct.py` generalises to any ACCESS
   READING VOLUME, but read §1.4 first: it inverted 11 of 14 members of the one
   volume it was measured on. Only a digest match makes an output usable.
4. **The three unrecovered members of volume `…129`** (§1.4) are candidates for
   the `ENCODED_BLOCK_FAILURE` / `READ_FAILED` accessibility exception list in
   `docs/OPEN_PROBLEMS.md` §E; that list is not this pass's to edit.
5. **`rnu_t4.py`** remains `CANNOT_VERIFY` and `T4_form` is not in this engine
   (§3.7). Work against the EXECUTE freeze list should not assume otherwise.

---

## 6. Files

| path | what |
|---|---|
| `engine/rn_engine/BINDING.json` | the eight carrier records |
| `engine/rn_engine/reconstruct.py` | the reading-volume de-transform; returns only on a SHA-256 match |
| `engine/rn_engine/verify_recovery.py` | checks A–D; `--selftest` runs the negative controls |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py` | the frozen engine, 103,166 bytes, `85d7725f…0930` |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_perc.py` | pinned; carries scope hold `Q-RN5-MOMENT-001` |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_amend.py` | pinned |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H2_foundations/cov_exact.py` | pinned |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H2_foundations/pin_transform.py` | pinned |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H2_foundations/reg_lemmas.py` | pinned |
| `engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md` | pinned; source of `Z_lo = 7.7592917375327855e-3` |
| `engine/rn_engine/frozen/RN_UNIF_2026-09-16/rnu_ds3.py` | the preferred A5 carrier, 9,704 bytes, `bd3074fd…9421` |

Nothing under `frozen/` may be edited.
