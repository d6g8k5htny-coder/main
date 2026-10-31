# H5 closure — TIGHTENING amendment (2026-09-15)

Amendment to the frozen H5_CLOSURE.md (body-sha256
465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301, shipped to
D1 v2.0). Per the standing rule the frozen body is NOT altered; this package
carries (i) the H5-AXIS v3 statement IN FULL (cited by D1's register but
absent from the v1 carrier — Stage-E provenance/scope finding), and (ii)
the refine-3 improved totals (cited from the version-frozen
h5_totals_v2_2026-09-15.json, sha256 f3e672a3cb4d4016a117624493a8ce766c5bf21e1a7697b840da82e896f98ecf;
version discipline per H5_TOTALS_FREEZE_2026-09-15.md). The frozen v1
interval remains the D1 v2.0 dependency; everything here is
improvement-only (downward drift of the upper end).

## 1. H5-AXIS v3 (named lemma, full statement)

**Regions.** With M = (−r/2, 0), S = (+r/2, 0), d_M/d_S the point distances
and θ_M/θ_S the unsigned polar angles about the +x axis (θ = 0 pointing
away from the segment on each side; the M–S segment itself is θ_S = 180° /
θ_M = 180°):
  - S-cone wedges: {d_S < 0.105} ∩ ({θ_S < 18°} ∪ {θ_S > 162°})   (v1, widened 12°→18° in v3)
  - M-forward wedge: {d_M < 0.10, θ_M < 12°}                       (v1)
  - M-backward wedge: {d_M < 0.10, θ_M > 168°}                     (v1)
  - S-disk: {d_S < 0.062}, ALL directions                          (v2: 0.03; v3: 0.062)

**Statement.** ρ_w(y) ≤ C_region for every chart point y in each region,
with machine-certified constants (each = 2 × max over the region's
certified edge-probe ρ_hi values, CoarseBox + coarse_fine_hi, η = 2.5e-4,
K = 8):
  - C_scone = 0.18998 (edge probes at θ_S ∈ {20°, 30°} × d_S ∈ {0.03, 0.06},
    all off-wedge; the θ_S = 12° v1-era sites are interior to the widened
    wedge and included conservatively in the max)
  - C_mfwd = 1.3254 (edge probes at θ_M ∈ {8°, 12°} × d_M ∈ {0.012, 0.03, 0.06})
  - C_mbwd = 1.6522e-05 (edge probes at θ_M ∈ {171°, 176°} × same radii)
  - C_disk = 0.34268 (edge probes on the ring d_S = 0.040 at θ_S ∈
    {30°, 60°, 90°, 120°, 150°} from the M-ward axis — values 0.170, 0.064,
    0.020, 0.039, 0.171 — plus the v3 ring at d_S = 0.065, θ_S ∈
    {30°, 60°, 90°, 120°, 150°} and d_S = 0.085 at 90° — values ≤ 0.123 —
    plus the scone d_S = 0.06 sites, all conservatively in the max)

**Flatness hypothesis (the lemma's content).** ρ_w is O(1)-flat across each
region at the probe scale: the certified edge-ring values bound the
interior. Mechanism: the pins' collinear conditioning (det Σ_gg ~ 1e-15
below the TM slack inside the wedges) makes the pgrad factor diverge as the
point approaches the M–S axis, while the windowed v-integral I is suppressed
by the same degeneracy; C1's exact values inside the wedges are ≤ 2.5e-5
(mostly boosted-zero), two to four orders below the certified constants —
the constants are honest certified envelopes, not tight values. The
hypothesis is stated per region with the probe receipts banked
(h5_results_r0.05_sdisk.jsonl, h5_results_r0.05_sring.jsonl,
h5_results_r0.05_s0p1_rimprobes.jsonl).

**Why the regions exist (obstruction record).** The Kac–Rice intensity
needs E[|det H| · 1_typed | six pins] via the 18-jet Gram; on the M–S axis
the pins are collinear and the conditional covariance Σ_gg degenerates
(machine-certified det Σ_gg.lo below the TD = 6 Taylor-model slack), so no
box straddling the axis certifies. The poison cell th15_dl0.066 (first
record hi = 6.47e7, certified but vacuous, preserved as the failed-cell
receipt in h5_results_r0.05_s1p2_cells.jsonl) sat in the off-cone near-S
band that v2's S-disk (0.03) and then v3's S-disk (0.062) absorbed; the
cell's machine-covered remainder recertified at hi = 2.17e-5
(h5_results_r0.05_refine.jsonl, min-per-cell).

**Grade.** Named-lemma (probe constants machine-certified; flatness stated
with receipts). **Production path (documented, not run):** the factored-det
expansion det Σ_gg = (d_M d_S)⁴·Q(1 + O(d)) with Q certified nonzero makes
the wedges and the disk machine-coverable (Grade-B fine path); the v3
quantifiers then retire.

**v1→v2→v3 amendment chain.** v1: cones only (12° S-wedges). v2 (poison
cell th15_dl0.066): S-disk d_S < 0.03. v3 (stitch execution): S-disk
extended to d_S < 0.062 absorbing the degenerate inner S-annulus band and
the S-disk boundary-exemption ring; S-cone wedges widened 12°→18° (the
[14°, 18°] sliver certified only vacuously); each extension with certified
edge probes banked BEFORE the envelope applied; the merger's coverage
self-test (81×73 grid) mirrors the final regions and fails closed on any
uncovered point.

## 2. Refine-3 improved totals (h5_totals_v2_2026-09-15.json)

    I_lo = 1.2328893552e-08   (partial by construction: 2 × Σ 8 patch lowers)
    I_hi = 8.0975589252e-02   (= 647.80 r³; 504.4× C1's 1.605315e-4)
    part ledger:  2R1 = 1.2217e-2 (unchanged; cells2 phase targets it)
                  2rim = 1.9492e-4        scone = 1.3161e-3
                  mfwd = 2.7759e-3        mbwd  = 3.4604e-8
                  sdisk = 4.1383e-3 (v3: C_disk·π·0.062²)
                  stitch = 5.78896e-2 (refine-3 min-per-sector: M-ring sec 0
                    collapsed to 0 via the v3 S-disk skip; M10/M2, S1/S9/S4/S6
                    refine passes banked; v1's 6.8550e-2 → 5.7890e-2)
                  remote = 2.4438e-3 (D3 bracket; see §4)
    vs frozen v1: I_hi 9.142887704e-02 → 8.0975589252e-02 (−11.4%),
                  I_lo 1.166721190e-08 → 1.2328893552e-08 (upward: more
                  patch mass banked — the lower end is partial by design).

**Re-verified cks (this amendment's merge, h5_merge.py current):** 70/70 R1
cells min-per-cell; 10 rim columns nonzero off-cone; probe buckets exact
(6 scone + 12 m-cone + 5 sdisk + 6 sring sites); 22 stitch sectors
(12 M-ring + 10 outer S-annulus; inner band excluded — superseded by the
v3 S-disk, documented); coverage self-test 81×73 grid every point ≥1
mechanism PASS (census rim 365 / r1 2409 / stitch 2476 / remote 1761 /
env 2495 / multi 3108); ck(C1_TOTAL = 1.605315e-4 ∈ [I_lo, I_hi]) PASS
(consistency display only).

## 3. Mutation-hardened merger

h5_merge.py refactor: assemble(r, mutate_index, write_totals); versioned
totals writes ('x'-mode auto-increment, never overwriting; the v1 consumed
artifact untouched per H5_TOTALS_FREEZE_2026-09-15.md); exact-count
probe-site cks added; mutation suite 6 selector classes (R1 cell / rim
column / stitch sector / scone site / sdisk site / sring site deletion) —
6/6 trip fail-closed cks.

## 4. Note on the remote part (D3 R3 correction)

New work carries the remote as the symbolic bracket B_remote·r³; the D3
lane's κ-inclusive correction lands as its own amendment and is consumed
then. This amendment's totals still consume the v1 bracket 19.55·r³
(the frozen D1 v2.0 dependency; the correction is expected to move
B_remote, direction per D3).

## 5. Receipts

- h5_totals_v2_2026-09-15.json (sha256 above), H5_TOTALS_FREEZE_2026-09-15.md
- refine-3 records: h5_results_r0.05_stitchref3.jsonl (+ stitchout*.log transcripts)
- v3 probe records: h5_results_r0.05_sring.jsonl (6 sites ≤ 0.123)
- merger + certificate hashes: CODE_HASHES.txt (regenerated at this freeze)
- falsify.py modes A/B byte-identical (3ef903bbe260…), falsify_promote.sh
  modes byte-identical (3c10935be0ca…), mutation suites 6/6 both

---

body-sha256: 7fefa17b74836092c0dd5912e028e3b09c1bf4f68c02e35390b552d4479f0561
