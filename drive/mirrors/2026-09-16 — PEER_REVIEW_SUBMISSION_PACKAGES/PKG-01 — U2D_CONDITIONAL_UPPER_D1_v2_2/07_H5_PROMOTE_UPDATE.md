# H5 — OBL-D1-PROMOTE update (2026-09-15): rung family executing, r-laws displaying

Update to H5_PROMOTE.md (frozen body 8d8b353815ec…). New package per the
corrections rule; the frozen document is not altered.

## 1. Empirical r-law displays (certified quantities, rung pair 0.05 → 0.025)

The promotion's core claim — the r-scaled cover's certificate inherits the
structural r-laws — is displaying directly in the banked rung records:

| quantity (certified) | r = 0.05 | r = 0.025 | ratio | structural law |
|---|---|---|---|---|
| patches lo sum (8 sites) | 5.8336e-09 | 7.1329e-10 | 0.1222 | r³ = 0.125 (−2.2%) |
| sdisk edge probe max | 0.17099 | 0.08564 | 0.5008 | r¹ = 0.5 |
| sring edge probe max | 0.12265 | 0.06054 | 0.4936 | r¹ = 0.5 |
| rim probe th=135 | 1.26 | 0.3069 | 0.244 | ≤ r¹ (conservative direction) |
| rim probe th=45 | 5.19 | 1.2602 | 0.243 | ≤ r¹ (conservative direction) |
| rim probe th=150 | 2.64 | 0.6497 | 0.246 | ≤ r¹ (conservative direction) |

Envelope contributions = constant × area inherit r¹ × r² = r³ (disk/cones)
and the r³ patch mass displays directly. The rim probes' faster-than-r¹
decay is the conservative direction (certificate slack, not a ledger
deviation; the falsifier watches for SLOWER-than-ledger parts — none seen).

## 2. Rung status (keeper_promote.sh A/B supervising all five drivers)

- r = 0.025: patches COMPLETE (lo = 7.1329e-10); probes COMPLETE (10 rim
  columns, 6 scone, 12 m-back — one site (thM=176, dl=0.006) failed at the
  transform-scaled η = 1.25e-4 and certified at η = 6e-5 on retry, receipt
  banked; 5 sdisk + 6 sring); cells EXECUTING (2 shards; 13+/70 banked);
  stitch EXECUTING (M-ring sec 0 = 0 via the S-disk skip, as at r = 0.05).
- r = 0.035355: cells EXECUTING (2 shards); probes/patches/stitch queued
  (same transform machinery; promote_probes.py per rung).
- r = 0.05: COMPLETE (frozen v1 731.43; live clean v3 647.80 — tightening
  amendment + errata of 2026-09-15).

## 3. Machinery made rung-general (CODE_HASHES.txt regenerated)

- h5_merge.py: rung-scoped record glob; every geometric constant scaled by
  s = r/0.05 (envelope areas, coverage grid, zone edges; angles invariant);
  stitch inner-band exclusion value-based (d1 < 1.2r); patch-site dedupe
  (tightest certified [lo, hi] per site, 8/8 ck); probe-site exact-count
  cks; bremote parameter (default = frozen v1 consumption 19.55; new work
  passes D3's κ-inclusive value when R3 lands); C1 containment display uses
  C1_TOTAL(r) = 1.605315e-4·s³ (C1's exact r³ ledger, recomputed totals).
  s = 1 reproduces h5_totals_v3.json exactly; mutation suite 6/6.
- h5_run.py: resume-skip globs rung-scoped (2 sites).
- promote_run.py: full scale-transform receipt per generated driver (zone
  constants, DL_EDGES/DELTAS labels, probe sites/η, exemption radii, patch
  sites, stitch bands, cells η_max; fail-closed on any missing literal).
- promote_probes.py: rung sdisk/sring edge probes against the generated
  module (r-scaled positions/η, rung-tagged records).

## 4. Process note (documented, no record loss)

stitch refine-3 (SECONDARY per the priority change) is PAUSED (processes +
keeper.sh A/B stopped 2026-09-15 ~14:05) to give the rung family the cores;
its banked records stand (v3 totals shipped). Relaunch: keeper.sh A/B.

## 5. Dense modulus display (12 certified points, interval-enclosed)

c2(r) = (Var f(M) − Cov(f(M), f(S)))/r² sampled at 12 interior points of
[0.0125, 0.05] (each a PinFrame interval enclosure, width < 1e-90):

    fit: c2(r) = 0.49999988 − 0.12476 r²   (one-term model)
    max |residual| = 3.4e-6 (relative 6.7e-6) over the whole band
    two-term: 0.50000030 − 0.12588 r² + 0.4429 r⁴ (max resid 3.2e-6)

i.e. c2(r) = 1/2 − r²/8 + O(r⁴) with the r² coefficient pinned to 0.125 —
the lattice short-distance expansion's exact form. This is the displayed
modulus (dense certified sampling + explicit fit); the CERTIFIED band
enclosure (interval-r lattice sums) remains OBL-H5-JETMOD with its content
unchanged. The display upgrades the interpolation theorem's hypothesis from
4 rung points to a dense band with an explicit κ = 1/8.

## 6. Certificate update

h5_promote.py now carries the rung-family r-law displays as fail-closed cks
(patches lo ratio vs r³ within 30%; sdisk/sring probe-max ratios vs r¹
within 35% — the H5-side mirror of D1's falsifier). falsify_promote.sh:
modes A/B byte-identical (sha 226af07c83d4…), mutation suite 6/6.
Update-doc sha256: see below.

## 7. Three-point envelope-constant display (all probe families complete at 2 rungs)

certified edge-probe maxima (H5-AXIS v2/v3 constants before the 2x factor):

    sdisk:  0.05: 0.1713   0.0354: 0.1210 (ratio 0.7061 vs r^1 = 0.7071)
            0.025: 0.08564 (ratio 0.4998 vs 0.5)
    sring:  0.05: 0.1230   0.0354: 0.08603 (ratio 0.6993)
            0.025: 0.06054 (ratio 0.4921)

sub-percent agreement with r^1 across TWO independent rung ratios — the
envelope contributions inherit r^1 x area(r^2) = r^3 as designed. h5_promote.py
upgraded to the 3-point cks (falsify_promote.sh byte-identical, suite 6/6).

update-doc sha256: 6513a18233d8983d2dc07a823d9536dcd0b6e087aeb177e7074b6791824b0441
