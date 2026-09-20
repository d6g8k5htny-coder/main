# H5 — OBL-D1-PROMOTE package (r-scaled cover + interpolation theorem)

New package per the standing rule (the frozen H5_CLOSURE.md, body
465c97c0…, is NOT touched; D1 v2.0 consumes that interval unconditionally of
this promotion work). Obligation verbatim: D1_ASSEMBLY_v2_0.md §4(d)
(frozen body 86882dca…, verified against the document's self-declared hash).

## 1. What is certified TODAY

- **Rung r = 0.05 (banked, frozen v1):** I_lo = 1.166721190e-08,
  I_hi = 9.142887704e-02 = **731.43·r³** (all fail-closed cks; live records
  now read 690.55·r³ after refine-3 min-per-sector gains — improvement only,
  shipped as the separate tightening amendment, never consumed here).
- **h5_promote.py certificate (fail-closed, modes A/B byte-identical,
  mutation suite 6/6):** r-scaled geometry table (every zone constant
  displayed as c·r^p, p checked against D1's structural prescription),
  per-rung Z windows (lo = H3's c_Z·r² theorem-grade; hi = 5.60·r²
  conservative; banked r=0.05/0.025 windows reproduced exactly), rung-0.05
  reproduction from banked records with C1 containment ck, and the
  scaled-jet convergence display (§3).
- **Rung re-execution IN FLIGHT (keeper_promote.sh A/B, flock singletons,
  3 s heartbeats):** promote_run.py generates the r-scaled driver
  h5_run_r{tag}.py by an explicit printed scale-transform receipt (s = r/0.05
  on every geometric constant; transform failure = SystemExit). Cells
  (the long pole) launched for r = 0.025 and r = 0.035355, shard 0/2 each.
  r = 0.025 Z window recomputed live = [1.0096807923e-03, 3.4978774469e-03]
  — byte-matches the crude-rung banked window (determinism display).

## 2. C_unif candidate (from current data; projections LABELED, never consumed)

- Certified anchor: **C_unif ≤ 731.43** at rung r = 0.05 (the frozen v1
  interval / r³), dominated by the coarse stitch 548.4·r³.
- PROJECTION (labeled, not consumed): the r-scaled cover re-executes the
  stitch at boxes ∝ r with r-invariant relative slack; refine-3-class passes
  at the rungs project C_unif ~ 150–400. The truth (C1 ledger): C* = 1.284.

## 3. The interpolation theorem (design + proof display; modulus named)

**Theorem (r-scaled certificate interpolation).** Let Î(r) be the H5 cover's
certified envelope executed with the r-scaled geometry (zone constants
c·r, η ∝ r, fixed K/TD/lattice). Then:

(i) *Structural form.* Every factor of Î(r) at an r-scaled station is a
structural power law (C1 §7 ledger: pgrad ∝ r⁻³, g ∝ r⁶, Z ∝ r², arch-band
volume ∝ r², needle mass scale-invariant Θ(1); rim disk ∝ r^{4.1}
subleading) times a factor depending on r only through the 12×12 pin/Hessian
Gram G12(r) (lattice covariances at separation r). The box tree in the
scaled coordinate u = y/r is r-invariant, so the certificate's relative
slack (TM remainder ∝ (η/δ)^{TD+1}) is r-invariant. Hence
    Î(r)/r³ = F(G12(r))
with F the cover's explicit interval computation.

(ii) *Modulus display (certified at the rungs, this package).* The scaled
short-distance coefficient c₂(r) = (Var f(M) − Cov(f(M),f(S)))/r² is
interval-enclosed at the rungs:
    r = 0.05:    0.49968763016765…
    r = 0.0354:  0.49984378254700…
    r = 0.025:   0.49992188313739…
    r = 0.0177:  0.49996093953443…
converging to 0.5 (the spectral second moment) with the correction HALVING
per 1/√2 step (1.56e-4, 7.81e-5, 3.91e-5 — O(r²) lattice correction, the
expected short-distance expansion C(r) = C(0) − r²/2 − c₄r⁴ − …).

(iii) *Band certification (the proof step).* For r ∈ [r_{k+1}, r_k],
G12(r) is a finite combination of lattice sums with certified tails;
evaluating those sums with r as an interval over the band yields
G12-band enclosures, hence Î(r)/r³ ≤ F(G12-band) for the whole band — a
FINITE computation per band, never a fitted exponent. C_unif =
sup over bands. The ladder's rung values must converge (falsifier: any
certified band value exceeding the claimed C_unif, or a part deviating from
C1's ledger powers).

**Named sub-obligations (exact content, not absorbed):**
- **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set (not
  just the displayed c₂) over the r-bands [r_{k+1}, r_k], with lattice-tail
  constants re-certified uniformly in the band (LAT's tail bound currently
  certifies at point separations). Content: for each jet J and band B,
  J(B)/r^{p_J} ∈ certified interval; falsifier: a band enclosure whose width
  exceeds the claimed modulus.
- **OBL-H5-ZBAND:** Z_r window bounds over bands (point-rung windows are
  banked; lo rides H3's c_Z·r² theorem-grade uniform bound; the hi side
  needs the band version of the LPW bracket). 
- **OBL-H5-REMOTE-THRESHOLD:** the remote is consumed at D3's grade
  (19.55·r³, I_ann 17.02 + I_far 2.5283) with threshold d ≥ 2r at the rungs;
  D1's prescription names an ABSOLUTE d₀. Content: certify D3's remote
  bracket at the r-scaled threshold d ≥ 2r, or extend the chart's machine
  cover to absolute d₀ per rung. Rides with D3-LEMMA-RN-UNIF (foundations).
- **Named lemmas at the rungs:** H5-RIM retired via shell_hi_fi (production
  path parked, p_sup-dominated); H5-AXIS/S-disk machine-certified via the
  factored-det expansion (production path documented). Until those land,
  the lemmas re-certify per rung in their r-scaled form (constants from
  r-scaled edge probes — the probe scripts scale by the same transform).

## 4. Rung ladder table (status)

| rung r | geometry | Z window | cover | envelope/r³ |
|---|---|---|---|---|
| 0.05 | r-scaled (v1) | banked | COMPLETE (frozen) | 731.43 certified |
| 0.035355 | r-scaled (s=0.7071) | live-recomputed | cells executing (keeper) | pending |
| 0.025 | r-scaled (s=0.5) | live = crude banked exactly | cells executing (keeper) | pending |
| 0.0177, 0.0125 | r-scaled | formula (H3 lo + 5.60 hi) | queued | pending |

Per-rung merger: h5_merge.py --r <r> once the rung's parts bank (the merger
is r-parameterized; the coverage self-test grid scales with the rung table).

## 5. Receipts

- h5_promote.py (certificate), falsify_promote.sh (modes A/B byte-identical
  sha256 3c10935be0ca18985bb906678748bf38cfb0074b34a71ebf74a0bbb85faecc4a;
  mutation suite 6/6), promote_modeA.txt ≡ promote_modeB.txt.
- promote_run.py + generated h5_run_r0p025.py / h5_run_r0p035355.py
  (scale-transform receipts embedded in the generated files' headers).
- keeper_promote.sh A/B (flock singletons; watch.log relaunch lines).
- h5_merge.py: refactored assemble(r, mutate_index) (behavior-preserving;
  clean assemble reproduces the live totals; 6 selector-mutation classes all
  trip fail-closed cks; new exact-count probe-site cks 6/12/5/6).
- Refine-3 (secondary, per lead): still running under keeper.sh A/B.

Body hash below; CODE_HASHES.txt regenerated at this freeze.

---

body-sha256: 8d8b353815ec7b0ace3215149dcd45f971dd510170d5c879b3cf5f31fcd63eb2
