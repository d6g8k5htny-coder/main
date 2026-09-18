# Bound carriers — content-addressed, hash-verified, unreviewed

The Drive holds 206 hash-stamped runnable Python carriers. This directory binds
a curated 25 of them so the git side can *hold the exact bytes* the program's
own computations were run from.

> **Binding is not review. Binding is not replay. Binding is not endorsement.**
> A record here says: these bytes were fetched from this Drive object, their
> SHA-256 matched `drive/inventory.jsonl`, and this is what the file does when
> read. It says nothing about whether the computation is correct, whether its
> result holds, or whether any premise moved. **No carrier in this manifest is
> certifying**; every record carries `certifying: false`.

## What is here

```
MANIFEST.json     one record per bound carrier (25 records, 20 with stored bytes)
blobs/            the stored bytes, at <sha256[:16]>__<sanitised name>
README.md         this file
```

Verified by [`tools/carriers_verify.py`](../../tools/carriers_verify.py),
exercised by [`tests/test_carriers.py`](../../tests/test_carriers.py).

## Content addressing

Every stored file lives at

```
engine/carriers/blobs/<first 16 hex of sha256>__<sanitised Drive file name>
```

so the path *is* the content address: you can read the expected digest off the
file name, and the verifier checks that `sha256(bytes)` reproduces the full
digest in the manifest, that the byte count matches, and that both still agree
with the row in `drive/inventory.jsonl`. Change one bit of one blob and the
verifier exits non-zero. Edit a digest in the manifest to match a doctored blob
and the inventory cross-check exits non-zero instead. The name prefix is a
convenience; the digest comparison is the check.

Retrieval itself was gated the same way: each carrier was downloaded, decoded,
and its digest compared with the inventory **before** the bytes were written.
A mismatch would have been recorded with `not_stored_reason: HASH_MISMATCH` and
no blob. **No mismatch occurred** for any of the 20 stored carriers.

## The manifest fields that matter

`arithmetic` and `certifying` were determined by **reading each downloaded
carrier**, not by guessing from its title:

| `arithmetic` | meaning | bound here |
|---|---|---|
| `exact_rational` | `fractions.Fraction` throughout | 3 |
| `mpmath_float` | imports mpmath, sets `mp.dps` — high precision, still floating point | 6 |
| `float` | `float` / numpy / scipy, including Monte Carlo and fitted exponents | 4 |
| `mixed` | exact and inexact in the same chain (e.g. `Decimal` + `Fraction`, sympy + mpmath) | 8 |
| `unknown` | not read, or the arithmetic is the callee's | 4 |

`certifying` is `false` on all 25 records. High precision is not a certified
enclosure; an interval computation carried out in floating point is not a
certified enclosure; a Monte Carlo estimate, a fitted exponent, a smoke test, a
byte-identity replay and a passing mutation suite are none of them certified
bounds. Where a carrier *prints* a status line of its own — `h5_zband_consume.py`
prints `OBL-H5-ZBAND DISCHARGED at consumption grade`, `lpw_constant.py` prints
`CERTIFIED` lines — the manifest transcribes what the carrier prints and records
the source's own countervailing status (`docs/OPEN_PROBLEMS.md` A3 has that ZBAND
consumption as **PROPOSED**, authority none; section C says the delivered
`6.239e−44` certificate must not be admitted unqualified). Binding a carrier
does not adopt its output.

`lane` names the section of [`docs/OPEN_PROBLEMS.md`](../../docs/OPEN_PROBLEMS.md)
the carrier serves, and the verifier rejects any lane that is not a real section
heading of that file. Five carriers serve work that document has no section for
(P0.1 adjacency positivity, P0.2 adjacency-to-one cubic rate, Theorem B, and a
quarantined failed P0.1 certificate); they
carry the explicit sentinel `NOT_IN_OPEN_PROBLEMS` rather than being filed under
a section that does not cover them. `lane_named_by_source` is `true` only for the
two carriers `docs/OPEN_PROBLEMS.md` names by file name (`rnu_ds3.py` and its
7.2 KB SUPERSEDED duplicate); for every other record `lane_basis` states, in
words, why that lane was chosen and that the document does not name the file.

## Two indexes, because two kinds of thing are bound

`MANIFEST.json` records **Drive files**: each record is validated against its own
row in `drive/inventory.jsonl` — title, path, byte count, digest and access
status — and its blob under `blobs/` is named by its digest.

`../rn_engine/BINDING.json` records **archive members**: the eight files of the
frozen RN engine tree recovered from inside ZIP carriers, which have no
inventory row of their own. They are validated instead against
`drive/source_map/Archive_Members.csv` (payload digest and byte count) by
`engine/rn_engine/verify_recovery.py --selftest`, pinned by
`tests/test_rn_engine_recovery.py`, and stored byte-exact under
`../rn_engine/frozen/` rather than duplicated into `blobs/`. They were
deliberately **not** merged into this manifest: forcing a ZIP member into the
Drive-file record shape would misdescribe it, and `tools/carriers_verify.py`
would rightly refuse the result. `tools/lanes_check.py` accepts a lane input
from either index, and says which one it resolved against.

## What is bound

| lane | carriers |
|---|---|
| A1 `OBL-H5-JETMOD` / chart side | `jets.py`, `GP-DATA-214…r_uniform_exact_endpoint_normalizer.py`, its byte-exact replay harness |
| A2 rung ladder | `d1_falsify_v4.py` (recorded, not stored) |
| A3 `OBL-H5-ZBAND` | `h5_zband_consume.py` |
| A5 `D3-LEMMA-RN-UNIF` | `rnu_ds3.py`, `rnu_t4_push.py`, `rnu_chi2_white_v2.py`, `rnu_meanfix.py`, and the SUPERSEDED scalar duplicate (recorded, not stored) |
| B matching 2D upper | `C101_WINDOW_SADDLE_FRAME.py`, `C099_CUBIC_TYPE_NOGO.py`, `C095_SIX_PIN_COVARIANCE_FACTORIZATION.py`, `q0_c091_gate_reduction.py`, `q0_c091_contract_checker.py` |
| C LPW constant repair | `lpw_constant.py`, `interval_repair.py`, `audit_received_claims.py`, `check_r04.py` |
| D review queue | `q0_verify.py` (recorded, not stored) |
| `NOT_IN_OPEN_PROBLEMS` | `CL-DATA-052 sixpin_ninejet_independent.py`, `CL-DATA-051_p01_jetbox.py`, `CL-DATA-056_prcp_law.py`, `C104_COUPLED_PERSISTENCE.py` (recorded, not stored), `FAILED — LS-DATA-009…` (recorded, not stored) |

## Why five carriers have no stored bytes

Every record whose `blob_stored` is `false` carries a `not_stored_reason` from a
fixed vocabulary, and the verifier rejects a missing or unknown reason.

* **`SIZE`** — `q0_verify.py` is 957 KB. Recorded, deliberately not carried.
* **`QUARANTINE_EXCLUSION`** — `rnu_ds3_scalar_SUPERSEDED.py` is the Drive object
  named by `quarantine/EXCLUSIONS.json` key `Q-R17-RN-OLD` (class `SUPERSEDED`).
  It is bound as a record and marked `authority_tier: superseded`, and its bytes
  are **not** pulled into the verified content set. `tools/quarantine_check.py`
  continues to pass.
* **`QUARANTINE_PATH`** — the `FAILED — LS-DATA-009…normalizer certificate`
  sits under `99_QUARANTINED_FAILED_CERTIFICATES` and its Drive title begins
  `FAILED`; `authority_tier: quarantined`.
* **`RECONSTRUCTION_INCOMPLETE`** — `d1_falsify_v4.py` and
  `C104_COUPLED_PERSISTENCE.py` were downloaded and read (so `computes` and
  `arithmetic` are filled from the actual text), but a byte-exact local copy was
  not achieved in this session, so the SHA-256 check never passed and **no blob
  was stored**. This is a limitation of the porting run, not a finding about the
  carriers.

## Available but deliberately unbound

`drive/inventory.jsonl` lists 19 runnable Python carriers under
`02_LEGACY_Q0_ARCHIVE — INSPIRATION ONLY / REVERIFY FROM FIRST PRINCIPLES`
(`c020 instr.py`, `c021 instr.py`, `c019 flow4.py`, `annulus v2.py`, …). They have
**zero evidentiary authority** and are not bound here; if any is ever wanted it
belongs under `legacy/`, never under `engine/` or `research/`. The verifier fails
if a `02_LEGACY_Q0_ARCHIVE` path ever appears in this manifest, and a negative
control in `tests/test_carriers.py` proves that check fires. The
`99_DO_NOT_OPEN` vault was never opened.

`drive/inventory.jsonl` holds 179 runnable Python rows in the active lanes
(168 distinct payload digests). After this binding, 154 rows / 144 distinct
digests remain unbound. This is a curated subset, not a mirror.

## What this directory does NOT establish

* It does **not** run, replay, review, reproduce or endorse any carrier. Nothing
  here was executed; several carriers import `mpmath`, `numpy`, `scipy`, `sympy`
  or `numba`, which this repository does not require.
* It does **not** promote, close, discharge or reclassify any claim, premise or
  obligation. `D3-LEMMA-RN-UNIF` is not closed. `OBL-H5-ZBAND`, `OBL-H5-JETMOD`,
  `OBL-H5-REMOTE-THRESHOLD`, `PERC-DECAY`, `OBL-B1-BRANCH` and the chart side of
  `OBL-D1-PROMOTE` are exactly as `docs/OPEN_PROBLEMS.md` leaves them.
* It does **not** certify any bound. `certifying` is `false` everywhere, and the
  verifier rejects `certifying: true` on anything but exact rational arithmetic.
* A matching SHA-256 establishes only that the bytes here are the bytes the
  inventory recorded on 2026-09-17. It says nothing about whether those bytes
  compute what their docstring says.
* `lane` is the binder's reading of which open problem a carrier serves. It is a
  navigation aid, not a source status label, and `lane_named_by_source` marks the
  two cases where the sources name the file themselves.
* The 2D upper track and the 3D lifetime track are not composed anywhere here.
* No original prize problem is solved.
