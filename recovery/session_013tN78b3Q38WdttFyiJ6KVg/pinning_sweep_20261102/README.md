# Pinning sweep: source files that adopted quarantine records name but do not bind (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. These are
mechanical byte pins, proposed. Nothing is adopted and no record is edited.

## Why

Many adopted rows in staging `32ab7d90` quarantine a claim at claim level: their location is a REGISTER anchor or a
graph node, and their text says the source is "not pinned here". Earlier I found two such cases by hand, Q-1043 and
Q-1025, which became tranche-2 amendments QP-C54-T2-031 and -034. This sweep finds every case mechanically.

## Rule (`pin_sweep.py`)

For each adopted row whose text says "not pinned", take every repository path the text names (under `frontiers/`,
`reviews/`, `experiments/` and similar roots) that is absent from the row's location and `source_binding(s)`.

- A path preceded by "NOT in" is skipped as a negated mention. Six such mentions are skipped, in Q-1143 to Q-1145
  (`formal/manifest.json`).
- Each remaining path is pinned at the first commit named in the row's own text at which the path exists, or else at
  the catalog's source cut.
- Recorded for each pin: blob, bytes, SHA-256, and whether the bytes are identical at the current tip (Math- `3b11bff6`,
  main `d2b0216e`).

## Result (`pins.json`)

**23 pins across 14 records. All 23 are byte-identical at the current tip.**

| Records | Repository and commit | Paths |
|---|---|---|
| Q-1025, Q-1043 | Math- `8151f592` | `frontiers/rn_mesoscopic_20260925/PROOF.md`, `frontiers/intermediate_window_20260928/PROOF.md`. These are the same blobs as tranche-2 amendments T2-034 and T2-031, so the two agree. |
| Q-1032, Q-1041 | Math- `8151f592` | `reviews/d5_local_collar_20260928/PUNCTURED_PIN_PROOF.md`, `COLLAR_PROOF.md`. Q-1022's own `next_step` asks for exactly these two pins. |
| Q-1160 to Q-1168 | main `f6deeba7`, the commit the rows name | The 9 `experiments/periodic_h0/*_certificate.py` (or `exact_h0.py`) programs and their `results/*/CERTIFICATE.json` |
| Q-1169 | main `81acc060`, the cut | `experiments/universality/fold_controls.py` |

## Limits

- A pin binds bytes to a record that already quarantines the claim. It does not say the bytes prove the claim.
- The rule finds paths by pattern. A row that names its source without one of these roots, or names a directory rather
  than a file, is not swept.
- Whether a pinned file is the row's source rather than a related file is judged only by the text naming it. All
  23 were read for that, and Q-1169's title names `fold_controls.py` as its subject.

```sh
python3 -I -B pin_sweep.py STAGING_CATALOG MATH_CLONE MAIN_CLONE OUT.json
```
