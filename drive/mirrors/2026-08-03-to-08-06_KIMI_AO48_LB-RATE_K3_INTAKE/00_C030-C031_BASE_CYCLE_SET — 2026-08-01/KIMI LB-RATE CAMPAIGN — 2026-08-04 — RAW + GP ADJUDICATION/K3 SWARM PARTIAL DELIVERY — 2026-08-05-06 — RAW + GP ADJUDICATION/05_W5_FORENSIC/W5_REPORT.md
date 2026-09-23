# W5 FORENSIC REPORT — Provenance of `3e-15`/`3.6e-15` (rigidity zone) and `0.213·r³` (Lemma WP budget)
**Agent:** C030/C031 Forensic Agent (K3 swarm, workstream W5) · **Mode:** read-only investigation (deliverable file excepted)
**Bottom line up front:** Both target numbers are fully reconstructible from the raw ledgers, to more digits than are printed anywhere. The rigidity figure is `lam_sad(d=1 station, r=0.025) × π·1.5² = 3.5578e-15`; the WP total is `rigidity + lam_sad(d=2)×π(2.5²−1.5²) + ρ_sad(1.2)·ℓ·π(3²−2.5²)·2 = 3.3214e-6 = 0.212571·r³`. The underlying station table was independently re-derived by this agent from the C034 pin geometry at mp.dps=40 and matches the ledger to 12+ significant digits on the variance entries. The zone-wide kill premise ("superexponentially killed **throughout** d ≲ 1.5") entered the budget at C030 as a **station-sup over the zone carried by a single interior station (d = 1)**; that exact step is a NAMED, unproven formality in the program's own register ("sup-over-zone" / "station-density / on-grid class").

---

## 1. Provenance chain for the rigidity-zone figure (~3e-15 → 3.6e-15)

### 1.1 The printed forms (verbatim)
- **C030 CountingLemmas Package.md, line 5** (sha256 `aed6683e…c7b`):
  > "E[N_ws(B₃∖collars) | 9 pins] ≤ **0.21·r³** at r = 0.025 (= 1.28·ℓ), three-zone: rigidity zone d ≲ 1.5 contributes ~3e-15 (superexponential kill, below); transition (1.5–2.5) 1.9e-6 (ℓ-scaling verified: rung ratio 7.76 ≈ 8); far (2.5–3) via the derived unconditional ρ_sad(1.2) = 0.030449 × measured enhancement ≤ 2 × TV."
- **C031_LBRATE_Integration.md, line 29** (sha256 `e7998ef0…f32e`; identical hash for both "C031 LBRATE Integration.md" and "C031_LBRATE_Integration.md"):
  > "| — window-pass channel | E[N_ws(B₃∖collars) \| 9] ≤ **0.213·r³** (rigidity 3.6e-15 + transition 1.95e-6 + far 1.37e-6 at r = 0.025; ℓ-ratio verified 7.76/8.19 ≈ 8) | derived-structure + measured constants (Lemma WP) | C030 (freeze ae20f4e3…f4b) | KR validity; collar exclusion |"

### 1.2 The derivation "below" that the package points to
"(superexponential kill, below)" refers to the same package's **"THE DISCOVERY — the jet-cluster prediction horizon"** section (line 11):
> "Var(f(y′) | 9 pins) at r = 0.025: **0.018 / 0.55 / 0.976 at d = 1/2/3.** … Consequences: (i) the band intensities are superexponentially killed (e^{−(b−m)²/2v}) throughout d ≲ 1.5 — the assemblies strengthen; … (iii) B3's above-b-saddle rigidity piece has its mechanism identified (same kill; **sup-over-zone argument remains the named step**); …"

### 1.3 The exact mathematical step (reconstructed, exact match)
The C030 Freeze (sha256 `ae20f4e3…ef4b`, line 5) froze the intensity formula:
> "intensity λ₉ws(y′) = φ∇(0|9)·φ_f(b | 9, ∇=0)·ℓ·E[|det H|1_sad | 9, ∇=0, f=b]·(1+O(ℓ))"

The station ledger **`c030 lemmas.json`** (sha256 `5afabb60…bcf19`) stores, at rung 0.025, station d1:
`phi2 = 0.22614080268832848`, `phif = 1.96644423101905e-10`, `Es = 4.346355426639727`, `v = 0.0005554263945655655`, `lam_sad = 5.03332040886253e-16`.

Verification (this agent, exact):
- `lam_sad = phi2 · phif · ℓ · Es` with `ℓ = r³/6 = 2.6041666666666674e-06`: product = `5.033320408862532e-16` = ledger value to all 15 digits. ✔
- **Rigidity-zone contribution = lam_sad(d1) × Area(disk of radius 1.5)** = `5.03332040886253e-16 × π·1.5²` = **3.557844544420372e-15**.
  - → printed as **"3.6e-15"** (C031, correct rounding) and **"~3e-15"** (C030 package, coarse/truncated). ✔
- The zone bound uses the **d = 1 station as the representative (sup) intensity for the entire disk d ≤ 1.5**; there is no station between d = 1 and d = 2 anywhere in the ledger. The sup-over-zone step is exactly the named formality (see §5).

### 1.4 First-principles re-derivation of the station row (independent anchor)
Using the exact pin configuration of **C034 Definitions Derivatives Canon.md** (sha256 `9bc03991…05d9`, §3): pins M = (−r/2, 0) @ (b,0,0), S = (+r/2, 0) @ (b−ℓ, 0, 0), y = M + r·(−0.76, 0.24) @ (v*,0,0), BF kernel K = exp(−|u|²/2); station x = d·(cos 2.35, sin 2.35) (angle per package line 17: "stations at one angle (2.35 rad)"), r = 0.025, mpmath dps = 40:

| quantity (d = 1) | W5 recomputation | ledger (`c030 lemmas.json`) | match |
|---|---|---|---|
| v = Var(f\|9 pins, ∇=0) | 0.000555426394566 | 0.0005554263945655655 | 14 digits ✔ |
| v_val = Var(f\|9 pins) | 0.01808 | printed "0.018" (pkg line 11) | ✔ |
| m_cond = E[f\|9 pins, ∇=0] | 1.3674 (m9 = 1.5071 before ∇-conditioning) | implied 1.3672 from phif | ✔ (see note) |
| phif = φ(b; m_cond, v) | 1.87e-10 | 1.96644423101905e-10 | 5% — fully explained by v* sensitivity (dm_cond/dv* = −2.94e5; a 5.5e-10 = 2e-4·ℓ shift in v* vs the asymptotic b−0.4999290ℓ closes it) ✔ |
| phi2 = φ_∇(0\|9) | 0.22589 | 0.22614080268832848 | 0.1%, same v* sensitivity ✔ |
| Es = E[\|det H\|1_sad\|9,∇=0,f=b] | 4.3585 (4e6-draw MC on the mp conditional law) | 4.346355426639727 | 0.3%, same sensitivity + MC error ✔ |

d = 2, 3 variance matches: v_cond 0.27810031/0.97359414 vs ledger 0.2781003073319938/0.9735941430560074 (13 digits); v_val 0.5495/0.9763 vs printed 0.55/0.976.

**Conclusion:** the ledger was produced by an mp-grade Schur-computation of the frozen Kac–Rice formula at the C034 pin geometry. The producing script itself is **not in the corpus** (`c027 station.py`, sha256 `ec6ee2f4…1d3d`, is the Λ-side exact-limit coefficient machinery, a different object). The record reaches: Freeze formula → station JSON → assembly JSON → package prose. My independent recomputation confirms every link numerically.

### 1.5 Mean/variance inputs to the kill (as the record states them)
- Variance v at d=1, r=0.025: **5.5543e-4** (∇-conditioned, used in the intensity; the "DF table", package line 14: "v = 0.001/0.278/0.974 at d = 1/2/3 (r = 0.025), FD limit 1").
- The kill factor: `e^{−(b−m)²/2v}` with b = 1.2, m = m_cond ≈ 1.367 at d = 1 → exponent ≈ −25.2, prefactor 1/√(2πv) ≈ 16.93 → phif ≈ 1.97e-10.
- Note: the **un**-∇-conditioned mean at d = 1 is m9 = 1.507 **above** b (this is the rim crest sector at angle 2.35 rad); the ∇=0 conditioning shifts the value mean down by ≈ 0.14. The kill is therefore driven by the *gradient-conditioned* mean, a subtlety the prose never states.

---

## 2. Provenance chain for the WP budget `0.213·r³`

### 2.1 Component arithmetic (verified exactly)
All at r = 0.025, ℓ = 2.6041666666666674e-06, r³ = 1.5625e-05:

| zone | formula | value | printed |
|---|---|---|---|
| in-ball B_2.5r∖collars | stations near3r/near5r, lam_sad = 0.0 (float underflow; phi2 = 0.0 at r=0.025 near3r, 3.3e-212 at r=0.05) | 0.0 | — |
| rigidity d ≲ 1.5 | lam_sad(d1)·π·1.5² | 3.5578445e-15 | "~3e-15" (C030) / "3.6e-15" (C031) |
| transition 1.5–2.5 | lam_sad(d2)·π·(2.5²−1.5²) = 1.5528090728834476e-07·4π | **1.9513174e-6** | "1.9e-6" (C030) / "1.95e-6" (C031) |
| far 2.5–3 | ρ_sad(1.2)·ℓ·π·(9−6.25)·2 = 0.030449079155158258·ℓ·2.75π·2 | **1.3701102e-6** | "1.37e-6" |
| **total** | sum | **3.3214276e-6 = 0.21257137·r³** | "0.21·r³" (C030) / "0.213·r³" (C031) |

Ledger check: **`c030 assembly.json`** (sha256 `49e197e9…6990`) = `{"WP": 3.3214240768939414e-06, "C_wp_r3": 0.2125711409212122, "MB": 9.84086697460724e-07, "C_mb_halfell": 0.7557785836498359}`. My reconstruction of WP agrees to 3.6e-12 absolute (2.6e-6 relative in the far term; the last-digit arithmetic of the assembly — exact enhancement/TV handling — is not recorded; see §6.3). The printed 1.95+1.37 = 3.32e-6 gives 0.21248·r³ (the prompt's figure); the ledger's precise total is 0.2125711.

- "1.28·ℓ" check: 0.212571·r³/ℓ = 0.212571×6 = 1.2754 ≈ 1.28 ✔
- C031's "0.213" rounds the ledger 0.2125711 up (conservative); C030's "0.21" truncates.

### 2.2 Rung ratios 7.76 / 8.19 (G-F2a, d2/d3)
From the ledger's two rungs: lam_sad(0.05)/lam_sad(0.025): d2: 1.2043468073692815e-06 / 1.5528090728834476e-07 = **7.7559 ≈ 7.76** ✔; d3: 1.2386690731123047e-06 / 1.5124772801215892e-07 = **8.1897 ≈ 8.19** ✔. Committed gate (C030 Freeze line 15): ratio ∈ [6.5, 9.5] (the ℓ = r³/6 ratio 8). **d1 FAILED-AS-WRITTEN: ratio 1.3897** (ledger `"GF2a": false`), adjudicated in the package line 14 as "gate design error (mine): d1 lies inside the horizon where the ℓ-factorization does not apply; benign-to-strengthening."

### 2.3 Component origins
- **ρ_sad(1.2) = 0.030449079155158258** — from **`c030 uncond.json`** (sha256 `2f0f5802…42e39`): `Esad_b = 0.9852262912564838`, and ρ_sad = (1/2π)·Esad_b·φ(1.2) = 0.0304491 ✔ (verified). The Hessian-given-value law (mean −uI, Var(Hxx|f) = Var(Hyy|f) = 2 indep, Hxy ~ N(0,1)) is the DERIVED fixed-zone core, C030 package line 8 / Freeze line 8; the E-term from the conditional-Gaussian reduction (route a, 2e6-draw MC), two-route-verified at 9.5% against direct counting on 44 synthesized fields (route b: 0.0391 ± 0.0044 for ρ_mx; `"GF1": "PASS"`), with "one recorded normalization slip (factor √(2ΣS)·Ng, derived and fixed)" (package line 8).
- **Enhancement ≤ 2** — G-F2b FD cross-tie (package line 14): "λ₉/λ_unc = 1.95/1.91 at d = 3, within 3×; enhancement ≈ 1.9 recorded and margined at 2.0". Verified: lam_sad(d3)/(ρ_sad·ℓ) = 1.9527 (r = 0.05) and 1.9074 (r = 0.025) ✔ — the printed 1.95/1.91 are the two rungs. Lemma FD itself: C022 Observed Update.json (sha256 `9bc0647b…5cb`), "all one-point conditional quantities within 2.2% of unconditional at d ≥ 3, < 1e-4 by d = 5".
- **Zone stations** — C030 station table (§1.4), machinery inherited from the C027 coefficient-field program (C029 package line 18: "machinery exists — the C027 coefficient field"; C027 Foundation Package, sha256 `c9d1466a…a3b4`).
- **The channel itself** — C029 B3 mountain-pass reduction (C029 BasinPersistence Package, sha256 `b23c3d42…a785`, line 8: diversion forces "a **window-class saddle** — Λ-suppressed O(r³) by proven machinery — or an above-b saddle near the pinned pair (rigidity-class, NAMED)…"); C030 Freeze line 2: "Purpose: convert C029-B3's main channel and C025-R2's near term to derived grade."

---

## 3. Verbatim rigidity-zone definition (for the K3 WP agents)

**What d is.** C034 canon §5 (sha256 `9bc03991…05d9`, precedence-governing document): "**horizon** | the variance structure v(d) := Var(f(x) | 9 pins) at distance d from the cluster: 0.018 / 0.55 / 0.976 at d = 1/2/3; v = 1 − O(1e-7) at d = 5; "inside the horizon" = the rigid zone where v ≪ 1 | C030 (discovery)". Theorem A Master v3 2 Distribution.md line 35 (sha256 `70fb6484…b709`): "v(d) := Var(f(x) | 9 pins) at distance d from the **cluster centroid**". W5 numeric confirmation: d is the radial distance **from the origin (the M–S pair midpoint)** along the ray at angle **2.35 rad**; reproducing the printed v table requires exactly this geometry (§1.4).

**Zone boundaries (WP, verbatim).** C030 package line 5: "three-zone: rigidity zone **d ≲ 1.5** … transition (**1.5–2.5**) … far (**2.5–3**)", all inside B₃. The C030 Freeze (line 5) had committed only a **two-zone** plan: "in-ball B_2.5r∖collars (expect double suppression: area r² × band ℓ) and fixed zone 0.5 ≤ |y′| ≤ 3 (expect → unconditional per Lemma FD)". The three-zone split at 1.5/2.5 appears first in the C030 package itself.

**Collar exclusions (verbatim).** C034 canon §5: "**collar** | the open disk of radius 2r about a pin point (M, S, or y) | C030; carries the collar-exclusion named item". The WP statement counts on **B₃ ∖ collars**. C034 §9.4: "**KR validity + collar exclusion:** Kac–Rice applicability at the pinned law + no second critical point within a pin collar (rigidity-class); note attached (C033): M's Hessian scale is O(r), so the exclusion argument must be ℓ-vs-r quantitative."

**Grade of the zone bound.** C034 §8: "**On-grid:** exact at stations, sup-over-zone by the station-density named formality"; §9.9: "**Station-density / on-grid class:** the single formality converting station-exact values into zone sups (C027 grid; R3′ slab; C033 sweeps…)".

---

## 4. Where the zone-wide kill premise entered the WP budget (exact step)

**File/line of entry: `C030 CountingLemmas Package.md`, line 5** (the rigidity-zone term "~3e-15" in Lemma WP), licensed by **line 11, consequence (i)**: "the band intensities are superexponentially killed (e^{−(b−m)²/2v}) **throughout d ≲ 1.5** — the assemblies strengthen", and explicitly flagged as unproven at **line 17**: "the sup-over-zone argument converting the horizon kill into the closed above-b-saddle piece" (named after this cycle). Ledger-side, the entry is the multiplication `lam_sad(d=1) × π·1.5²` embodied in `c030 assembly.json`'s WP total.

The logical content of the step, as the record itself exposes it:
1. The kill e^{−(b−m)²/2v} is **verified only at stations** (d = 1, 2, 3 at one angle, 2.35 rad, plus a 1D exact probe; package line 11). The d = 1 station is the **only** station inside the rigidity zone.
2. Since v(d) and m(d) both move to weaken the kill as d → 1.5 (v: 5.6e-4 → 0.28 across d = 1→2; m9(d=1) = 1.507 is the rim crest), the d = 1 intensity is **not self-evidently a sup over the zone**; treating it as one is precisely the named "sup-over-zone" formality (C031 §6 item 5; C034 §9 item 9).
3. The gate that would have checked the d1 machinery, **G-F2a d1, FAILED-AS-WRITTEN** (rung ratio 1.39 vs committed [6.5, 9.5]) and was adjudicated "benign-to-strengthening" by the author (package line 14) — i.e., even the single station value lacks its committed scaling check; the assembly's own `c030 lemmas.json` records `"GF2a": false`.
4. The same kill premise is reused for the **above-b-saddle** channel (package line 11 (iii)), which is *not* part of the 0.213·r³ number — it remains a named channel in the AO assembly (C031 line 30).

---

## 5. Inconsistencies on the record

1. **Two variance tables coexist.** C030 package line 11 prints "Var(f(y′) | 9 pins) … 0.018 / 0.55 / 0.976" while line 14's DF table gives "v = 0.001/0.278/0.974 at d = 1/2/3 (r = 0.025)". Both are real: the first is value-only conditioning (canon definition), the second is the ∇-conditioned variance actually used in the intensities (ledger v = 5.55e-4/0.2781/0.9736). Line 11 mixes them in one sentence (its "consistent with Lemma FD at d = 3 (v = 0.974 …)" and "r-drift at d = 2 (0.248 → 0.278)" use the ∇-conditioned table; 0.248/0.278 are the ledger's two rungs). Downstream (C031 line 66, C034 §7, masters) quote only 0.018/0.55/0.976 as "the horizon".
2. **"× TV" prose not in the number.** Package line 5 says the far term is "ρ_sad(1.2) × measured enhancement ≤ 2 × TV", but the far component matching the ledger total is ρ_sad·ℓ·A·**2** with **no** TV factor (with TV ×1.022 the total would be 0.2145·r³ ≠ 0.2125711). Either TV was dropped or is considered absorbed in the 1.9→2.0 margin; the record does not say.
3. **KR-MB B₅ area factor misprint (C031 line 32).** "0.76·(ℓ/2)·(25−6.25)/(9−6.25) ≈ 2.1·(ℓ/2)": as printed the factor is 18.75/2.75 = 6.82, giving 5.18, not 2.1. The result 2.1 is consistent with factor (25−0.25)/(9−0.25) = 2.83 (→ 2.15) or 25/9 = 2.78 (→ 2.11). C030 Observed Update.json line 9 separately says "C025 R2 near term bounded **2.25**·(ℓ/2)" — a third value.
4. **Superseded intermediate assembly inside `c030 lemmas.json`.** Its `"assembly"` block (`inball 0.0, far_wp 2.309252469150711e-06, CWP_far 0.14779215802564546, kr_mb 1.598084586254326e-06, C_mb 0.613664481121661`) does **not** equal `c030 assembly.json` (WP 3.3214e-6, MB 9.8409e-7). The block is an earlier, superseded assembly; its far_wp does not decompose into any clean combination of the ledger's stations/zones that I could find (closest: ρ_sad·ℓ·2·A with A ≈ 14.56, no clean annulus) — **exact composition not recoverable from the record** (gap). Also `"GF2a": false` in the same file vs the package's "d2/d3 PASS; d1 FAILED-AS-WRITTEN" framing (the boolean conflates the per-d verdicts).
5. **Rounding drift across documents.** Rigidity: computed 3.5578e-15 → "3.6e-15" (C031) vs "~3e-15" (C030 package). Transition: 1.9513e-6 → "1.95e-6" vs "1.9e-6". Total: 0.2125711 → "0.213" vs "0.21". C031 is the accurate rendering in every case.
6. **Dates.** C022: 2026-07-09; C024, C030, C031 freezes and observed updates all 2026-07-10; C030 Freeze sha ae20f4e3… matches the hash cited inside the package (line 2) and C031 (line 29); C031 Freeze sha e165821b… matches C031_LBRATE_Integration.md line 2's self-citation — but note the Integration document's own file hash (e7998ef0…) differs from the freeze it cites (it cites the *Freeze* file, correctly). The two C031 integration filenames ("C031 LBRATE Integration.md" / "C031_LBRATE_Integration.md") are byte-identical duplicates.
7. **Downstream restatements** (02b PART II LOWER SIDE B.md lines 17–20, sha256 `b1c70570…a827`; Q0_MASTER; masters) all carry the C031 (0.213, 3.6e-15) forms verbatim; 02b line 19 restates the kill with "v(1) = 0.018" (value-only table) while the intensity used v(1) = 5.6e-4.

---

## 6. Confidence-labeled conclusions

- **[VERIFIED, exact]** 3.6e-15 = lam_sad(d=1, r=0.025) × π·1.5² with lam_sad = φ_∇(0|9)·φ_f(b|9,∇=0)·ℓ·E[|det H|1_sad|9,∇=0,f=b] = 5.03332e-16 from `c030 lemmas.json`; "~3e-15" is the same number, truncated. Independent mp re-derivation of the station row from the C034 pin geometry matches to 12–14 significant digits on v and exactly on the lam_sad product; residual 0.1–5% deviations on phi2/phif/Es are fully accounted for by the recorded v* = clip(μ_t) sensitivity (the ledger used the exact finite-r v*, not the limit 0.4999290·ℓ).
- **[VERIFIED, exact]** 0.213·r³ = 3.3214e-6 at r = 0.025: transition lam_sad(d=2)·4π = 1.9513e-6 + far ρ_sad(1.2)·ℓ·2.75π·2 = 1.3701e-6 (+ rigidity 3.56e-15), matching `c030 assembly.json` WP = 3.3214240768939414e-06 / C_wp_r3 = 0.2125711409212122. Rung ratios 7.76 (d2) / 8.19 (d3) verified from the two-rung ledger. ρ_sad verified as (1/2π)·0.9852262912564838·φ(1.2); enhancement 2 verified as the margin over the measured cross-ties 1.95/1.91.
- **[RECORD GAP, minor]** The last-digit assembly arithmetic (enhancement exactly 2.0? TV omitted? collar-area trims?) is not stored; my reconstruction of WP differs from the ledger total by 3.6e-12 (2.6e-6 relative, far below printed precision). The producing scripts for `c030 lemmas.json`/`c030 assembly.json` are absent from the corpus. The superseded `far_wp = 2.30925e-6` block is not decomposable from the record.
- **[KEY STRUCTURAL FINDING, high confidence]** The rigidity-zone number is a **single-station zone bound**: the kill is verified at d = 1 (one angle) and multiplied by the whole d ≤ 1.5 disk area. Because the kill weakens monotonically toward the zone boundary (v and |b−m| both move the wrong way; m9(d=1) = 1.507 is already above b on this ray), lam_sad(d=1) is **not a demonstrated sup over the zone** — the program's own register names exactly this step ("sup-over-zone"; "station-density / on-grid class"; plus KR validity + collar exclusion), and the d1 scaling gate G-F2a FAILED-AS-WRITTEN and was waived as "benign-to-strengthening". The 3.6e-15 figure is therefore an exact product of a station measurement and a zone area whose zone-wide validity is an open named formality, not a derived zone bound.

## Appendix — evidence files (sha256)
- aed6683edaed6d483704a63cd321fcb300e329a26bf6c143042ead58a60bfc7b  C030 CountingLemmas Package.md
- ae20f4e3a239b17a6b747eba881d18cc7cf8e51e4d6a86eaefc40d9504fdef4b  C030 Freeze.md
- 03490fe16e4e797f83792b2da758939874fd98a0b07741d62eb0e029d64af2f9  C030 Observed Update.json
- 5afabb604b35b28ce3f89a99859aa08887d7c4579d563ecd86deb4635afbcf19  c030 lemmas.json
- 2f0f5802c9577c0e6dc7db910c66b378b6fabd40ee9b543487cc6d682b642e39  c030 uncond.json
- 49e197e978d950ac356ec96866760a7d0d8337341415e501b6ac7905dcd69990  c030 assembly.json
- e165821b1479f619af8ebdb1438460edfebe88f44c189553ca86f73f8fe1afbe  C031 Freeze.md
- 5df4205f2f496f28668c7f1ea22e2a1af9cde00a72cb39a6956b20b32c995063  C031 Observed Update.json
- e7998ef0d17d951bc89978f9fe32e510019059dd0650c8f0e0ae33273f40f32e  C031_LBRATE_Integration.md (= "C031 LBRATE Integration.md", byte-identical)
- 83313c96d1610130451a7e936245a06e80d8cc84df00a20ae9e46e7aeef2a271  c031 verify.json
- b23c3d42ccc2182e9a59a4ae1b7eda8c73939f543884268a0da3e2cc6880a785  C029 BasinPersistence Package.md
- ddd8596f894c525f70b4ff91ed163abca3b831df9c24147a5d3adeb0b4da00ca  C029 Freeze.md
- c9d1466a6b06c9a5619d21e7896aab51746bdaeb8d34d1aed5bfb498ca73a3b4  C027 Foundation Package.md
- ec6ee2f49b9e4f54b07a647bd14371d84c1c0dfa78296eba82e2e222c0fc1d3d  c027 station.py
- ab27672bb6008d337321c7d99f95aae2b2efd800576a362d3c0cfb03b29ea0c2  C026 Foundation Package.md
- d1afd84b3143ed512b0fb1cafba5c2eb6eb190b28ec7f8e789cc817c08bf792a  C025 TerminalHeight Package.md
- 0df10fa670a356a2bdd528f70ea5064f6dfd164b52388dff14a5d73df180a139  C024 Corrections and Freeze.md
- 9bc0647b885931ba4dc862bedfb879f9bd12c4cd6d39f51c6e98e54c7f75d5cb  C022 Observed Update.json
- 9bc039912645a93c7d217fa578cd6d8bcdf7f1127266a440bb4f635b4af05d9d  C034 Definitions Derivatives Canon.md
- b1c70570ca5f434ec4f41742901fdcd3c37088b16a2239663c830e050d1ea827  02b PART II LOWER SIDE B.md
