# STAGE E — PROOF-FIRST INDEPENDENT REVIEW of D1_ASSEMBLY v2.0 (U2D-UPPER)

Reviewer stance: skeptical journal referee; proof-first; nothing accepted on
authority, reputation, numerical plausibility, or executable mirrors of a
different statement. Independence declaration: before beginning I had seen
ONLY the lead's mission prompt (the verbatim theorem text and the package
register). No prior draft, result, or other review was read.

DISPOSITION: **FAIL** — three independent grounds, each sufficient:

- **F1 (fatal; wrong step).** D1 v2.0 §2 "Constant bookkeeping" line
  `≤ E_w(r) + C_RN(r)·√Q_r(B1.dir) + P_r(B2) + P_r(B4)   (H4-B1, H4-JC)`:
  the joint-carrier once-payment rests on Theorem H4-JC's premise "the two
  counted populations are disjoint at each saddle" (H4_CLOSURE.md §2.3),
  which is FALSE under the frozen definitions: every α-qualifying saddle is
  an N_loop member (proved below + explicit discrete counterexample). The
  frozen carriers prove only `1 − q ≤ 2·I_hi + C_RN·√Q(dir) + P(B2) + P(B4)`.
- **F2 (insufficient step; hidden hypothesis).** Theorem (1)'s "CERTIFIED
  RUNG" divides by a normalizer floor `Z_{0.05} ≥ c_Z·(0.05)²` that no
  carrier certifies: H3's floor holds only for `0 < r ≤ r₀` with r₀
  EXISTENTIAL (no certified modulus, H3's own honesty boundary), and H5's
  own bracket is vacuous at r = 0.05 ("lo = 0, honest"). The hypothesis
  "0.05 lies inside H3's existential radius" is used but never named in
  Theorem (1); D1 §8's "unconditionally of every open item" is false.
- **F3 (citation mismatch; undercount).** D1's remote bracket
  "I_ann ≤ 17.02·r³ + I_far ≤ 2.5283·r³" misquotes the frozen D3-THM, which
  states `I_far ≤ 2.5283·r³·(1 + κ_far)`, `κ_far(5) ≤ 0.63 + ε_QMC`. The
  dropped factor (1+κ_far) ≈ 1.63 exceeds the 19.5483→19.55 round-up margin
  (+0.09%) by ~700×; the frozen D3 form gives remote ≈ 21.1·r³ > the
  19.55·r³ consumed inside I_hi. Moreover ε_QMC is Monte-Carlo evidence and
  zone uniformity is the OPEN D3-LEMMA-RN-UNIF.

______________________________________________________________________

## 1. Custody verification (all carriers, own FREEZE conventions)

Body hashes recomputed independently from the bytes, each under its own
carrier's documented extraction rule:

| carrier | rule | expected | result |
|---|---|---|---|
| D1_assembly/D1_ASSEMBLY_v2_0.md | BEGIN/END_FROZEN_BODY, blank-trimmed, one terminal LF (17988 B) | 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 | MATCH |
| B1_taxonomy/B1_TAXONOMY.md | marker segment, rstrip+LF | 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 | MATCH |
| C1_alpha_intensity/C1_ALPHA_INTENSITY.md | whole file | ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f | MATCH |
| C2_B_classes/C2_B_CLASSES.md | marker segment, raw | 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 | MATCH |
| D2_branch_control/D2_BRANCH_CONTROL.md | marker segment, rstrip+LF | 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa | MATCH |
| D3_percolation/D3_PERCOLATION.md | marker segment, raw | 8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47 | MATCH |
| H3_closure/H3_CLOSURE.md | whole file | 387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637 | MATCH |
| H4_closure/H4_CLOSURE.md | marker segment, rstrip+LF | a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 | MATCH |
| H5_closure/H5_CLOSURE.md | everything before "\n---\n\n" (separator at byte 15162, verified) | 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 | MATCH |
| H2 cov_exact.py / pin_transform.py / reg_lemmas.py | whole file | f08c1c5f… / c6988ac7… / e7005a5a… | MATCH (prefixes) |

(Note: D1 v2.0's own extraction requires stripping the LEADING blank line as
well as trailing — the documented "blank-trimmed" rule; d1_falsify.py's
"rstrip" mode does not check D1's own body, only its consumed carriers. My
independent extraction with the documented rule matches exactly.)

## 2. Items verified SOUND (with exact references)

1. **Partition (B1-COVER).** B1_TAXONOMY.md §4: on Reg ∩ TYP,
   `F_r = A ⊔ B1 ⊔ B2 ⊔ B4`, hence `1 − q = Σ P(category)` (equality; D1's ≤
   is a weakening, fine). Proof read line by line: Step 1 dichotomy τ_M > s
   (= A by T1(a); A ⊂ F_r since τ_M = s on success); Step 2 level-s
   trichotomy over S's sectors with the birth trichotomy inside MIX
   (birth(C₂) > b success; < b B2; = b excluded on Reg, B3 ⊂ N); Step 3
   below-s fates B.fin/B.ess; Step 4 converse. Exhaustive, no hidden case.
   Lemma R (P_r(Reg) = 1) is countersigned by H2's reg_lemmas.py (header:
   "OBL-B1-REG-G1, G4, G5 … addressed"; D1's register marks OBL-B1-REG
   CLOSED — consistent). P_r ≪ Q_r is trivial from P_r = W_r dQ_r/Z_r.
2. **D2-COUNT.** D2_BRANCH_CONTROL.md §1.3: N_qual ∈ {0,1} on Reg ∩ TYP and
   {N_qual = 1} = A. Proof: qualification at y ⟹ M alive at v = f(y) and the
   elder rule kills C_M at y ⟹ τ_M = v; uniqueness from "a lineage dies
   once"; converse constructs the qualifying witness from the death saddle
   (index-1 by Morse on Reg; window membership from τ_M ∈ (s,b), τ_M < b by
   B1's Lemma B). Sound. The integral form `E[N_qual] = ∫ρ_w·AO dy` consumes
   C1's Kac–Rice Step-7 machinery with AO inside the conditional
   expectation; the certificate contract (L1/L2: v-independent conditional
   covariance, affine conditional mean) is the right interface for the
   retained v-integral; the numerics are the numerics reviewer's domain.
3. **H4-RN polarity and constant arithmetic.** H4 §3.2:
   `P_r(dir) = Z_r⁻¹ E_Q[W 1_dir] ≤ (√E_Q[W²]/Z_r)·√Q(dir)` — Cauchy–Schwarz
   with correct polarity (Z_r ≥ c_Z r² lowers the denominator); the numerator
   typing-drop W² ≤ (detH_M detH_S)² is polarity-safe; the exact moment via
   Isserlis = tensor GH5⁶ (exact through per-variable degree 9 > 8) with
   1e-60 agreement is a sufficient certificate contract. Arithmetic
   recomputed: √31.23 / 1.615489267643502474 = 3.45925 ≤ 3.46 ✓; dir
   threshold √(12 ln(1/r) + 4 ln C_RN + 2 ln 2) = 6.50 at r = 0.05 ✓
   (recomputed: 6.5038).
4. **H5 totals arithmetic** (recomputed from h5_totals.json at float-exact
   Decimal): parts 2R1+2rim+scone+mfwd+mbwd+sdisk+stitch+remote sum to I_hi
   (rel diff 1e-28); remote = 19.55·r³ exactly at r = 0.05; containment
   I_lo < C1_TOTAL < I_hi ✓; frozen I_hi = 9.142887704e-02 ≥ current json
   I_hi (drift conservative) ✓; 731.43 = I_hi/r³ ✓; C_chart = 731.43 − 19.55
   = 711.88 ✓; D3 spine 17.02 + 2.5283 = 19.5483 ≤ 19.55 (round-up
   conservative as far as the DISPLAYED spine goes — see F3 for what the
   display omits); remote appears exactly once (inside I_hi; the retired
   v1.2 A.rem terms are not re-added anywhere in the v2.0 display) ✓.
5. **Z_r polarity** in every display read (H3, H4-RN, H5 line 122
   `∫_B ρ_w dA ≤ Σ area·pgrad_sup·I_hi/Z_lo`, D3 §3): Z_r sits in
   denominators and lower bounds are used for upper bounds. Consistent.
6. **3D firewall.** H3_CLOSURE.md §1/§5: the printed supplement Theorem
   G.3.1 is identified as the 3D instance and is NOT consumed; the 2D
   instance is re-derived (corrected typing det H_S < 0, corrected limit
   object c₀ = E[U²1{U<0}], certified to 60+ digits). No 3D theorem appears
   as a 2D dependency in any consumed carrier.
7. **Theorem (2)'s hypothesis naming.** The named hypotheses (H1 incl.
   OBL-B1-BRANCH(dir); OBL-D2-AO-SHARP i–iii, v; D3-LEMMA-RN-UNIF;
   OBL-D1-PROMOTE) are each genuinely used (dir/B2/B4 o(r³); remote
   uniformity; chart promotion), and the register is self-consistent
   (PERC-DECAY listed upgrade-only and indeed not consumed by the O(r³)
   form). As an EXISTENTIAL-C statement, (2) would survive F1 (constant
   doubles) — but the bookkeeping display it inherits from §2 is false (F1),
   and (1) does not survive.

## 3. F1 — The H4-JC joint-carrier disjointness is FALSE (fatal)

The step under review. D1 v2.0 §2:

    1 − q = P(A) + P(B1) + P(B2) + P(B4)
          ≤ E_w(r) + C_RN(r)·√Q_r(B1.dir) + P_r(B2) + P_r(B4)   (H4-B1, H4-JC)

and Theorem (1): "The joint carrier is paid ONCE for A+B1 (H4-JC:
N_qual + N_loop ≤ N_w a.s.; 1437/1437 executable mirror)."

The cited carrier. H4_CLOSURE.md §2.3 (Theorem H4-JC):

    On Reg ∩ TYP, the two counted populations are disjoint at each saddle:
    a loop witness has BOTH sector components in C; an α-qualifying witness
    has EXACTLY one (C2 §2.3; D2-COMPLEMENT's channel disjointness). Hence
    pointwise  N_qual + N_loop ≤ N_w  a.s. under P_r.

The frozen definitions. C2_B_CLASSES.md §2.2 (Theorem C2-B1, frozen):

    N_loop = #{ S' ∈ Crit₁(f) ∖ {S} : f(S') ∈ (s, b),
                C'⁺(f(S')) ⊆ C  and  C'⁻(f(S')) ⊆ C },
    C'±(u) = the two sector components of S' in O^u;  C := C_M(s).

D2-COUNT's qualification (level v = f(y)): exactly one sector's O^v-component
is C_M(v), the other's birth > b; and qualification ⟹ τ_M = v.

REFUTATION (deterministic, two observations).

(i) For ANY index-1 saddle y and ANY level s < f(y) =: v, both sector germs
of y lie in ONE component of O^s — the component containing y (Morse model
f = v + u² − z²: O^s near y is {u² − z² > s − v}, which is connected when
s − v < 0, since the waist at z = 0 is the full u-line).

(ii) If y qualifies, one sector's O^v-component IS C_M(v) ∋ M; by (i) the
shared O^s-component of both sectors contains C_M(v), hence equals C_M(s)
(M ∈ O^s since f(M) = b > s). Therefore BOTH sector components of y at
level v are ⊆ C_M(s): y satisfies C2's N_loop membership (y ≠ S on A ⊂ F_r;
f(y) ∈ (s,b) by the window).

Hence {N_qual = 1} ⊆ {N_loop ≥ 1}: the α-witness population is a SUBSET of
the loop population — the maximal possible overlap, the opposite of
disjoint. On A the unique qualifying saddle is double-counted:
N_qual + N_loop = N_loop + 1, which exceeds N_w whenever every window saddle
is a loop member. (Equivalently at the intensity level: Qual(y) ⊆ AO_loop(y)
conditionally, so AO(y) ≤ AO_loop(y) pointwise and
E[N_qual] + E[N_loop] ≤ 2∫ρ_w·AO_loop ≤ 2E_w — no once-payment.)

Explicit counterexample (constructed with the campaign's own PL mirror
library b1_falsifier.py, 40×40 triangulated torus, union-find elder rule):
M at value b = 1.2, elder max T at 1.3, S at s = 1.1, saddle y at v = 1.15
merging M's basin into T's basin. Union-find: D(M) = y, so A holds
(τ_M = 1.15 ∈ (s,b)). Window saddles: exactly one (y). Checks:
  – math qualification (D2-COUNT sense, level v): cluster [4]'s O^v-component
    = C_M(v), cluster [1]'s component elder (max 1.3 > b) → N_qual = 1;
  – loop membership (C2 sense, level s): both clusters' level-s components
    = C_M(s) → N_loop = 1;
  – N_w = 1. Hence N_qual + N_loop = 2 > 1 = N_w. The H4-JC display fails.

Where the carriers went wrong. C2 §2.3 claims "an A.loc (α) witness must
have EXACTLY one sector component in C" with C = C_M(s) — false at level s
(by (ii) both sectors are in C_M(s)); the "exactly one" is true only at
level v relative to C_M(v). D2-COMPLEMENT's channel disjointness (cited by
H4-JC) is the LEVEL-v statement (LOOP_y: both sectors in C_M(v)) and does
not apply to the PRICED N_loop (level-s containment). H4-JC's parenthetical
conflates the two levels.

The executable mirrors do not rescue the step — they check a different
predicate. c2_falsifier.py F-B1b and h4_falsifier.py F-H4c/F-H4d define
alpha_qualifies/my_alpha_qual as "exactly one cluster's representative lies
in C AT LEVEL key(S)" (compS from components_above(f, Sv)). By observation
(i), both clusters of any saddle share the same level-s component, so
"exactly one cluster in C at level s" is UNSATISFIABLE (inC ∈ {0, ≥2});
the mirrors' qualifying population is identically empty, F-H4d reduces to
the trivial N_loop ≤ N_w, and the "1437/1437 zero leaks" certifies nothing
about the mathematical disjointness. h4_falsifier.py lines 336–350 even
display that under the predicate "both-in-C counts as qualifying" the
disjointness gates WOULD fire — but per (ii) that "mutated" predicate is
exactly what the continuum qualification implies at level s. The mirror
thus confirms, rather than refutes, the present finding.

Consequence. The provable content of the frozen carriers is:
P(A) = E[N_qual] ≤ E_w (AO ≤ 1, D2 G0, closed) and
P(B1) ≤ E[N_loop] + P(B1.dir) ≤ E_w + P(B1.dir) (C2-B1 + AO_loop ≤ 1), so

    1 − q ≤ 2·E_w(r) + C_RN(r)·√Q_r(B1.dir) + P_r(B2) + P_r(B4),

and at the rung 1 − q ≤ 2·I_hi + … = 1.8285775408e-01 + …, NOT the displayed
single I_hi = 9.142887704e-02. Theorem (1) as stated is not implied by the
cited carriers; H4-B1's display is unproved; the "exact, no double
counting" annotation is wrong (there IS an uncorrected double count — of
the qualifying saddle, in the opposite direction: the claimed saving does
not exist). A repair would need N_loop redefined at level v (both sectors
in C_M(f(S')) — which IS disjoint from qualification by the sector
trichotomy) AND C2-B1's absorption re-proved for that smaller population
(C2-B1's proof produces a loop saddle with sectors in C_M(s); the level-v
strengthening is not in the record). No such repair exists in the package.

## 4. F2 — The rung's normalizer floor is uncertified (hidden hypothesis)

H3 §6 (the consumable closure): "there exists r_0 > 0 such that
Z_r ≥ c_Z r², 0 < r ≤ r_0 … c_Z ≥ 1.615489267643502474 (certified)". H3 §5
(honesty boundary): "r_0 is existential in the record of record (the
convergence S7 has no certified modulus) … certifying a modulus for the
expectation is an unexecuted extension — not claimed."

H5 line ~142: "Z_r interval: [max(H3 c_Z r², own lo), √(M8)] with H3
c_Z = 1.6154892… (existential r₀ flagged as a shared campaign hypothesis),
own bracket lo = m4 − √(M8·PnC) (vacuous at r = 0.05: PnC ≤ 1 → lo = 0,
honest). Measured: Z(0.05) ∈ [4.038723e-3, 1.397032e-2]". H5 line 122:
"∫_B ρ_w dA ≤ Σ area·pgrad_sup·I_hi/Z_lo".

So at r = 0.05 the certified upper I_hi divides by Z_lo = c_Z·(0.05)²
= 4.038723e-3 — exactly H3's existential-floor value instantiated at a
specific r. No carrier certifies 0.05 ≤ r₀; H5's own bracket is vacuous
("lo = 0, honest"); the only support is "Measured" numerics (labeled
non-load-bearing everywhere). The same uncertified floor underlies
C_RN(0.05) ≤ 3.46 (H4-RN's denominator) and D3's remote spine (D3 §3.3
"POLARITY: the 1/Z_r consumes the campaign-wide two-sided normalizer input"
= the same existential statement). B1's frozen setting records only
"0 < Z_r < ∞ … taken as campaign input" — positivity, not the c_Z r² floor.

Theorem (1) as displayed names no such hypothesis; D1 §8 claims the rung
statement is "theorem-grade today, unconditionally of every open item".
That is false: Theorem (1) is conditional on the unproved, unnamed
hypothesis Z_{0.05} ≥ 1.615489267643502474·(0.05)² (equivalently, that the
rung lies inside H3's existential radius). (For Theorem (2) the existential
floor is adequate — its claim is itself existential in r₀; F2 does not
touch (2).)

## 5. F3 — The remote bracket drops D3's (1+κ_far) factor (citation mismatch)

D1 v2.0 §2: "The remote bracket 19.55·r³ IS the D3 spine (I_ann ≤ 17.02·r³
+ I_far ≤ 2.5283·r³) and appears exactly once — inside I_hi."

Frozen D3-THM (D3_PERCOLATION.md §5, lines 277–278):

    I_far(r) = E_{P_r}[N_w({d ≥ 5})] ≤ (576 − 25π)·J(ℓ)·(1 + κ_far(5))
             = 2.5283·r³·(1 + κ_far),   κ_far(5) ≤ 0.63 + ε_QMC
             (certified cross bound 0.63 axis-worst + QMC RN 1.00±0.01)

The cited form (I_far ≤ 2.5283·r³) is NOT the frozen form: the frozen bound
carries the factor (1 + κ_far) with κ_far ≤ 0.63 + ε_QMC, i.e.
I_far ≤ 2.5283·1.63·r³·(1+ε) ≈ 4.12·r³. The remote total per the frozen
form is ≈ 17.02 + 4.12 = 21.14·r³ = 2.6425e-3 at r = 0.05 — larger than the
2.44375e-3 (19.55·r³) consumed inside I_hi. The "conservative round-up"
19.5483 → 19.55 (+0.09%) does not cover a dropped +63% factor. (D3's own
assembly display "P_r(A.rem) ≤ 20.9·r³" exhibits the same omission
internally: 1.284 + 17.02 + 2.5283 = 20.83, i.e. it too dropped (1+κ_far)
from its own formula.) Additionally: (a) ε_QMC ("QMC RN 1.00 ± 0.01") is
Monte-Carlo evidence, not a certified bound — a certified upper may not
contain it; (b) the uniformity of the bracket factors over the zone is the
OPEN named lemma D3-LEMMA-RN-UNIF (register row 3), so even the 21.1·r³
form is not closed today; (c) the annulus 17.02·r³ "crude spine" uses a
3-point G_cap whose interior-overshoot control is a "factor-20 margin
displayed" (evidence), and whose 1/Z_r consumes the same existential floor
(F2).

Note the direction of the error: the consumed I_hi is too SMALL to be a
certified upper on the remote zone as the frozen D3 statement stands. This
is independent of F1 (which doubles the whole E_w term) and of F2.

## 6. Secondary observations (recorded; not the verdict's basis)

- Theorem (1) rests on unproved named lemmas H5-RIM and H5-AXIS v1+v2+v3
  (flatness hypotheses; production paths "documented, not executed",
  register row 6). D1's theorem text discloses the named-lemma grade, but
  §8's "unconditionally of every open item" contradicts the register's own
  accounting: unproved named lemmas are open items.
- D1_V2_RECEIPTS.txt's drift note ("h5_totals.json currently reads
  I_hi = 9.1399568624…e-02") is stale at review time: the json now reads
  8.6318918e-02. Drift direction remains conservative (frozen consumed
  value is larger); d1_falsify.py's fail-closed drift ck still passes.
- Theorem (2) inherits the §2 bookkeeping; with F1 repaired its leading
  content becomes 2·C_unif (+ remote at the F3-corrected grade). The
  existential statement itself is repairable; the issued document's exact
  displays are not correct as written.

## 7. Disposition and load-bearing dependencies

DISPOSITION: **FAIL**.

Exact failing step: D1_ASSEMBLY_v2_0.md §2 "Constant bookkeeping (exact, no
double counting)", second display line — the inequality
`1 − q ≤ E_w(r) + C_RN(r)·√Q_r(B1.dir) + P_r(B2) + P_r(B4)` attributed to
"(H4-B1, H4-JC)" — and consequently Theorem D1 v2.0(1)'s display
`1 − q(0.05,6/5) ≤ I_hi + C_RN(0.05)·√Q_{0.05}(B1.dir) + P_{0.05}(B2) +
P_{0.05}(B4)`. The joint-carrier once-payment (H4-JC) is false under the
frozen definitions (§3: proof + explicit counterexample); the frozen
carriers yield the twice-paid bound 2·I_hi + … . Independently, Theorem
(1) uses an uncertified rung normalizer floor (§4, F2) and a remote bracket
that underquotes the frozen D3-THM by the factor (1+κ_far) ≈ 1.63 on the
far zone (§5, F3).

Theorem D1 v2.0(2) is not separately certified by this package either (its
derivation quotes the same false bookkeeping), though as an existential
statement it is repairable from the carriers conditional on the named
obligations, with leading constant 2·C_unif and the F3-corrected remote
constant.

Load-bearing dependencies for this verdict (all hash-verified by me against
the carriers' own FREEZE conventions, §1):
- D1_assembly/D1_ASSEMBLY_v2_0.md, body 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 (MATCH; marker rule with blank-trim)
- B1_taxonomy/B1_TAXONOMY.md, body 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 (MATCH)
- C2_B_classes/C2_B_CLASSES.md, body 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 (MATCH)
- D2_branch_control/D2_BRANCH_CONTROL.md, body 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa (MATCH)
- D3_percolation/D3_PERCOLATION.md, body 8e7fef6b4fb989404624bf50083c9d8624fa486eeda62e4784447f8edb3eac47 (MATCH)
- H4_closure/H4_CLOSURE.md, body a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 (MATCH)
- H5_closure/H5_CLOSURE.md, body 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301 (MATCH; separator at byte 15162 verified)
- H3_closure/H3_CLOSURE.md, whole 387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637 (MATCH)
- C1_alpha_intensity/C1_ALPHA_INTENSITY.md, whole ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f (MATCH)
- H2_foundations/{cov_exact.py f08c1c5f…, pin_transform.py c6988ac7…, reg_lemmas.py e7005a5a…} (MATCH, prefixes)
- H5_closure/h5_totals.json (arithmetic recomputed; parts/remote/containment/drift as in §2 item 4)
- Counterexample engine: B1_taxonomy/b1_falsifier.py (PL mirror library used
  to construct the §3 counterexample; deterministic, seed-free).

Prior exposure: NONE beyond the mission prompt (confirmed — I had not read
any prior draft, result, or review before beginning; the historical v1.x
assemblies were not opened and no historical authority was relied on).

---

FREEZE (REVIEW_proof.md): body = everything before the separator "\n---\n\n"; body sha256 = 922f5668d7336587ecf7e0107e07cc667c9ad00247d98f1e78141124e654d93c
