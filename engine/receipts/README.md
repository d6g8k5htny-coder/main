# `engine/receipts/` — the run record

One JSON file per execution, at `<lane>/<receipt_id>.json`. A receipt says
which lane was run, which module and entry point were called with which
arguments, at which commit, when, for how long, what numbers came out with what
provenance, and — mandatorily — what the run **does not establish**.

> **A receipt is not evidence.** It is not a proof, not a certificate, not a
> review verdict, not a status and not a promotion. A green run is a run.
> `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side), `OBL-H5-REMOTE-THRESHOLD`,
> `OBL-D1-PROMOTE` and **both** Pieces of `D3-LEMMA-RN-UNIF` are OPEN. No
> receipt in this directory changes that, and none can: the schema's `verdict`
> field admits exactly one value, `NO-MATHEMATICAL-VERDICT`, and its
> `status_effect` field admits exactly one, which says nothing changed. The
> only thing that can change a mathematical status is an operator decision
> under [`governance/`](../../governance/).

```bash
python3 engine/run.py --list            # lanes, and which have a runnable entry
python3 engine/run.py --lane A1         # run one lane, write one receipt
python3 engine/run.py --all             # run every registered lane
python3 engine/run.py --lane A5 --dry-run   # resolve and print; write nothing
python3 tools/receipts_check.py         # validate every receipt; non-zero on failure
```

## Append-only

Receipts are never rewritten and never deleted. This follows the pattern the
repository already enforces for `registers/json/work_events.json`
(`tests/test_registers.py`), and here it is structural in three independent
ways:

| | mechanism | what it catches |
|---|---|---|
| 1 | the writer creates files with `open(path, "x")` only — no truncating open, no `os.remove`, no `os.replace` anywhere in `engine/receipt.py` | a second write to an existing id |
| 2 | each receipt carries the SHA-256 of its own canonical JSON body | a later edit, with no reference copy needed |
| 3 | `tools/receipts_check.py` also compares against `git HEAD` | a deletion, or an edit that recomputed the hash too |

The writer additionally refuses any destination outside this directory and
refuses outright to write under `engine/lanes/`, `claims/`, `registers/` or
`drive/`. `engine/run.py` imports that writer and no other, which is how
"a run records, it does not decide" is enforced rather than promised.

## Provenance of every number

Three kinds, and nothing else is accepted:

| provenance | what it is | `certifying` |
|---|---|---|
| `certified_interval` | a two-sided enclosure from `research/interval/`, exact rational endpoints, containment unconditional | true |
| `exact_rational` | a `fractions.Fraction` computed in exact arithmetic | true |
| `float_noncertifying` | a binary float, an `mpmath` value, a Monte Carlo estimate, a fitted exponent, a dense sampling, a display, a probe | **false** |

`runtime_seconds` sits outside this scheme on purpose: it is a wall-clock
measurement of the machine that ran, NON-CERTIFYING, and a bound on nothing
mathematical. It is stored as fixed three-decimal text so that no float ever
enters the hashed body.

A `float_noncertifying` result **is not a bound**. High precision is not
certification — the sources say so themselves, repeatedly — and the writer
forces the label `NON-CERTIFYING` into the note of every such result so it
cannot travel without it.

## What the registered runs are, and are not

`engine/run.py` currently registers two lanes. Both drive **reference**
machinery, and both say so in their own `does_not_establish`:

- **A1** runs the published-point arithmetic on the three `I_hi/r^3` point
  certifications (exact rationals plus certified enclosures) and one
  interval-`r` band enclosure with a proved uniform tail bound — for a
  **reference Gaussian kernel** over a **reference band** with a
  **placeholder** displacement map. The program's own `kplane`, its 24 jets
  with their powers `p_J`, and the actual band endpoints `r_k` are none of them
  bound in this repository, so no jet of this program is enclosed. It does not
  discharge `OBL-H5-JETMOD`.
- **A5** runs the adaptive cover of the RN5 annulus `0.1 ≤ |y| ≤ 5` on a
  **reference integrand** — not `kappa_far`, not the corrected RN5 envelope —
  and reproduces the RN5 moment-envelope counterexample in exact rationals. It
  closes neither Piece of `D3-LEMMA-RN-UNIF`, certifies no cell of the
  program's cover, and does not reassemble the remote budget. The RN5 annulus
  and the T4 polar cover `d ∈ [5, 17]` are different regions and are not
  merged.

A lane with no registered entry point is reported by `--list` as having none.
That is a statement about code present in this repository, like a lane's
`repo_state`. It is not a mathematical status and no verdict may be read from
it.
