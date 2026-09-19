# STAGE E — INDEPENDENT TOPOLOGY REVIEW (U2D-UPPER, frozen package D1 v2.0)

**Reviewer charter:** the deterministic/stochastic topology layer only — the
superlevel merge-tree elder rule, the failure-event taxonomy, the event
partition, and the exact event identities the assembly consumes. Kac–Rice
intensities, normalizers, and numerics are other reviewers' lanes; interfaced
only by checking that the events certified here are the events the assembly
integrates.

**Prior exposure:** none. This reviewer had seen only the mission prompt
before beginning; no part of the package was previously known to me.

**Review date:** 2026-09-14. **Mode:** independent re-derivation of every
claimed event identity; hash-custody verification of all frozen carriers;
re-execution of the executable falsifiers; own constructed-field attacks.

---

## 0. Custody (hash verification) — ALL VERIFIED

Extraction per each file's own FREEZE block (marker-segment, LF-normalized,
each file's stated trailing-LF convention; the corpus's per-file off-by-one
LF quirks were reproduced faithfully — every frozen body hash matches under
its stated rule):

| carrier | frozen body sha256 | verdict |
|---|---|---|
| D1_assembly/D1_ASSEMBLY_v2_0.md | 86882dca5d5258e9f1bab43e52c4382d30ec9609f89cd64486b3441f5ffd62d0 (17,988 B) | MATCH |
| B1_taxonomy/B1_TAXONOMY.md | 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 (30,065 B) | MATCH |
| D2_branch_control/D2_BRANCH_CONTROL.md | 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa (28,410 B) | MATCH |
| C2_B_classes/C2_B_CLASSES.md | 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 (25,485 B) | MATCH |
| H4_closure/H4_CLOSURE.md | a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 (16,734 B) | MATCH |

Companion code/transcripts (sha256 of whole files) verified against the
FREEZE records: b1_falsifier.py 818df798…, c2_probes.py ae7d2e5c…,
c2_falsifier.py 2441b630…, d2_probes.py 62e15f5a…, d2_falsifier.py
824284be…, H2 cov_exact.py f08c1c5f…, pin_transform.py c6988ac7…,
reg_lemmas.py e7005a5a… (vs H2 MANIFEST.sha256). Cosmetic defect noted by
B2b confirmed (B1_taxonomy/FREEZE.txt whole-file line has 67 hex chars;
body hash itself is exact). No custody failure.

Re-executed (this reviewer, this machine): b1_falsifier.py (both modes,
byte-identical, digest 78ee1509… as frozen); c2_falsifier.py (PASS,
transcript byte-equal to tf_normal.txt); d2_falsifier.py (PASS, digest
877210c7… as frozen, transcript byte-equal to tf_normal.txt; it internally
re-ran d2_probes in both modes byte-identically); H2
selftest_reg_lemmas.py (PASS both modes, byte-identical, matches frozen
transcript; "SELFTEST_REG_LEMMAS: PASS (G1/G4/G5 certified on the exact
field)"); b2d_torus_attack.py (PASS, digest 48b71a51… as frozen).

---

## 1. The partition F_r = A ⊔ B1 ⊔ B2 ⊔ B4 (review target 1): PASS

Re-derived independently; no gap found.

**Exhaustiveness.** On Reg ∩ TYP: Step 1 dichotomy τ_M > s vs ≤ s; T1's
four forms of A ((a) death level, (b) path, (c) saddle, (d) component) were
re-proved — the only non-immediate directions are (b)⇒(a) (merge level
u ≥ min f∘γ > s kills the younger, M, since z's lineage is born above b)
and (d)⇒(b) (open component path-connected, compact path has clearance).
Step 2 at level s: distinct critical values ⇒ S is the unique critical
point at level s; TYP + Morse lemma ⇒ exactly two sectors; the trichotomy
{C⁺ = C⁻ = C / C ∉ {C⁺,C⁻} / exactly one} partitions ¬A; within MIX the
birth trichotomy is total on Reg (birth(C₂) = b would be a critical-value
coincidence). Step 3: survivors past s die below s (B.fin) or never
(B.ess = D_f(M) = ⊥); both failure. Step 4 converse: D_f(M) = S forces
¬A and MIX with birth(C₂) > b. The case analysis is genuinely exhaustive —
every elder-rule failure mechanism (death above s, self-attachment at S,
foreign event at S, younger merge at S) is a case, and the cases are
reached from merge-tree combinatorics alone (birth order + sector
membership), never from contractibility.

**Lemma B** (birth = sup while alive) re-proved: a point above b in
C_M(t) forces a lineage born above b sharing the component at t, hence a
merge at level ≥ t killing M — contradiction. Correct, and it is the
load-bearing wall against hidden-birth attacks (my ATK-HIDDEN carve and
B2a's ATK-6 confirm the behavior).

**Disjointness.** A vs B's definitional (B ⊂ ¬A); B1/B4/MIX a trichotomy;
MIX split by birth; refinement tags (A.loc/A.rem, B.fin/B.ess, WRAP)
overlap exactly as declared and are not additive classes. No relaxation
anywhere. My double-fire attempt (A with both sectors of S in C) confirms
precedence is definitional (B2a's ATK-7 likewise).

**Measurability.** Conn(x, y, t) Borel via rational polygonal chains
(§M1, correct); A and A.loc are countable Boolean combinations of Conn
and value events. Sector membership and births are Conn-expressible
(§M2 sketch — the "maximal rational radius ρ₀(f)" device is sketched
rather than fully expanded, but the construction is standard and I found
no obstruction); B1, B2, B4, B3, MIX, success, N all Borel. Sample space
C³(T²) with its Borel σ-algebra carries Q_r and P_r (continuous Gaussian
regression version). Acceptable at the DER grade claimed.

## 2. B3 ⊂ N, P_r(N) = 0 (review target 2): PASS (at the countersigned grade)

**B3 ⊂ N** re-proved: birth(C₂) attained at an interior local maximum
(sup over an open component is attained in the closure at a point back in
the component, which is then a local max); birth(C₂) = b with C₂ ≠ C
forces a second critical point at value b — a critical-value coincidence
⇒ N. The attainment argument was independently verified by B2c (§3) and
by my reading; no escape (non-tame attainment failures are non-Morse ⇒ N).

**Convention dependence.** Real and exhibited (B2c T-TIE: DESC-elder
pairs D(M) = S, ASC-elder pairs D(M) ≠ S; B2d A4a likewise) — so
N-residency is load-bearing, exactly as the taxonomy says. Any tie-break
convention off Reg is measure-irrelevant under P_r.

**Lemma R.** B1's own text is derivation-grade and was honestly flagged
(OBL-B1-REG). The four adversaries found the real holes: B2b/B2c proved
the R⁵ pair-map has NO density on pinned slices (rank drop 5→3, det = 0
exactly at the pins; naive tie intensity diverges |u−M|^{−5}); B2b named
G4 (pinned-value coincidences = exactly the B3 class), G1 (near-diagonal
bootstrap), G5 (ten-jet citation overreach). The H2 countersignature
(H2_foundations/reg_lemmas.py, RECEIPT_reg_lemmas.json, selftest PASS
both modes byte-identical to the frozen transcript) closes all three at
certificate grade:
- G4: pinned-pair rank **exactly 3** certified at 17 probe points;
  certified active floor **6.16287e-3 > 6.1e-3** on the d ≥ 1 grid
  (midpoint 6.16467e-3); near-pin rate **λ_min ~ C·d⁶ with C > 0
  certified** — and the transcript confirms the **d⁶-not-d⁸ correction is
  the live statement** ("R3 … (d^8 witness corrected)"; B2b's 2.4e-22 was
  below the float64 eigvalsh noise floor ~1e-16 and was uncertifiable;
  certified λ_min(d = 0.002) ≥ 5.62305e-20). The d⁶ rate is also
  consistent with B2c's det Σ ~ d^10 (eigenvalues d²·d²·d⁶).
- G1: Upair λ_min ~ δ⁶ (exponent 5.999 certified; 4.15612e-8 at δ = 0.1);
  separated-pair floor 9.9912e-6; Morse bootstrap with sympy-verified
  isolation radii.
- G5: general full-rank lemma (linear independence of derivative
  evaluations at distinct points; README proof) + certified floors
  (11-jet 7.57065e-11; near-pin 11-jet 1.314e-12; pair+pins 3.244e-10;
  14-jet 0.3776; ten-jet anchor 0.1266 marking the citation's scope).
Scope check: G4 covers precisely the B3/C-1 class (Crit ∩ f⁻¹(b) ∖ {M}
and the s-analog); the pin-isolation patch (conditioned Hessian PD, so
the pins are isolated critical points) is certified (Var(f_xxx|pins) ~
6r²; H(M)|pins λ_min ~ r⁴, 1.04115e-6 at r = 0.05). **Caveat recorded,
not a fail:** Lemma R now stands at "certified numerical floors + a
stated general lemma" grade, not a fully typeset classical proof; that is
the foundations lane's declared grade and D1 §5 carries OBL-B1-REG as
CLOSED on that basis. The topology layer's use of Lemma R (only
P_r(Reg) = 1, via P_r ≪ Q_r) is exactly what the countersignature
supports.

## 3. Torus/wrap modes (review target 3): PASS

The cover proof never uses contractibility (elder rule reads birth order;
Step 2 reads sector membership; A's path form is winding-blind). B2d's
wrap-enriched all-pairs audit (11,458 pairs, no distance filter) found
zero uncovered failures; I re-ran b2d_torus_attack.py and reproduced the
frozen digest 48b71a51…. My own wrap carve (ATK-WRAP3: preemption chain
winding through the i-seam to an elder maximum) classifies failure/A, as
predicted. Essential H₁ classes are never H₀ elder-rule partners;
D_f(M) = ⊥ is failure inside B.ess. The quotient-vs-cover trap (A4c) is
correctly identified as a lift artifact; the frozen estimand is
quotient-defined. Wrap modes generate no failure outside A ⊔ B1 ⊔ B2 ⊔ B4.

## 4. D2-COUNT (review target 4): PASS

**Theorem D2-COUNT** (D2 §1.3): on Reg ∩ TYP, N_qual ∈ {0,1} and
{N_qual = 1} = A exactly. Both inclusions re-derived:
- Qual(y) ⇒ M dies AT y: birth(C_M(v)) = b keeps M's lineage the younger
  side; MIX_y puts the two sector components distinct; birth(C₂) > b
  makes C₂ elder; the elder rule kills C_M at y, so τ_M = v and
  D_f(M) = y. "M dies once" ⇒ at most one qualifying saddle.
- A ⇒ the death saddle qualifies: τ_M ∈ (s,b) (death level < b since the
  lineage reaches M at b; > s by A); the death is a merge at S′, so the
  sectors lie in the two distinct merging components (else
  self-attachment, no death); the survivor is elder, birth > b. The one
  point the frozen proof compresses — birth(C_M(v)) = b at exactly
  t = v = τ_M although "alive" is defined as t > τ_M — is patched by the
  open-set argument: C_M(v) ⊇ C_M(t) for all t > v gives birth ≥ b, and
  any p ∈ C_M(v) with f(p) > b would connect to M in O^t for t ∈
  (v, v + path clearance), contradicting Lemma B. Correct.
- Reg/TYP content used: distinct critical values (uniqueness of the
  level-v event; no ties), Morse two-sector structure at y (TYP for S;
  Reg for y). Exact as stated.

**The 1612-window-saddle verification is logically sufficient for its
role:** the discrete mirror's `channel_of` uses the CORRECT level-v
predicates (C = comp[M] at level key(y); PRE = maxkey[C] > key(M); NA =
no cluster in C; LOOP = all clusters in C; YNG/QUAL by elder max key),
i.e. faithful mirrors of the continuous D2 definitions. F-D2a
(qual ⟺ D(M) = y per saddle), F-D2c (N_qual ∈ {0,1}, A ⟺ N_qual = 1 per
pair), F-D2b (4-channel disjoint+total per saddle) all executed
fail-closed on 427 pairs / 1612 window saddles; tally QUAL = 177 = the
176 A-pairs of the frozen ensemble + 1 synthetic (T2-QUAL) — I re-ran
d2_falsifier.py and reproduced the frozen digest 877210c7…
byte-identically. The continuous proof (above) is what carries the
theorem; the mirror is corroboration and is adequate as such.

## 5. D2-COMPLEMENT 4-channel decomposition (review target 5): PASS

PRE / NA / LOOP / YNG are pairwise disjoint and total on ¬Qual(y):
first split on birth(C_M(v)) (> b ⇔ M already dead at v, the Lemma-B
contrapositive; = b otherwise), then the sector trichotomy (neither /
both / exactly one in C_M(v)), then the birth trichotomy on C₂ (tie
N-resident). This is verbatim the taxonomy's Step 2 with (S, s) replaced
by (y, v) — valid on Reg ∩ TYP for every window saddle. All four channels
plus QUAL are realized by the mirror's synthetics; the ensemble tally
(PRE=1041 NA=288 LOOP=7 YNG=105 QUAL=177, sum 1612) is disjoint and
total. NOTE (naming hazard that enabled §6's failure): D2's LOOP channel
is a LEVEL-v condition (both sectors = C_M(v)); C2's N_loop is a LEVEL-s
condition (both sector components ⊆ C_M(s)). They are different events.

## 6. H4-JC joint carrier (review target 6): **FAIL**

**Exact lemma:** H4_CLOSURE.md §2.3, **Theorem H4-JC** (frozen body
a67d50b9…): "the two counted populations are disjoint at each saddle: a
loop witness has BOTH sector components in C; an α-qualifying witness has
EXACTLY one (C2 §2.3; D2-COMPLEMENT's channel disjointness). Hence
pointwise N_qual + N_loop ≤ N_w a.s." — consumed by D1_ASSEMBLY_v2_0.md
§2 ("The joint carrier is paid ONCE for A+B1 (H4-JC: N_qual + N_loop ≤
N_w a.s.; 1437/1437 executable mirror)") for the single-E_w display in
Theorem D1 v2.0(1) and the constant bookkeeping. Companion statement
C2_B_CLASSES.md §2.3 ("an A.loc (α) witness must have EXACTLY one sector
component in C and the other elder-foreign … mutually exclusive") is
false for the same reason.

**Exact gap (level conflation).** Under the documents' own definitions —
Qual(y) at LEVEL v ("exactly one of C_y⁺, C_y⁻ = C_M(v)", D2 §1.1) and
N_loop at LEVEL s ("both sector components of S′ at level f(S′) ⊆ C :=
C_M(s)", C2 §2.2, and the level-s containment is what Theorem C2-B1's
proof delivers) — **M's death saddle on A belongs to BOTH populations**:
at v = τ_M ∈ (s,b) its sector components are C_M(v) and the elder C₂(v),
and for every t ∈ (s,v) the two are already merged into M's component,
so at level s both are contained in C = C_M(s). "Exactly one sector
component in C" is false for the α witness: BOTH are in C (the elder one
additionally has birth > b).

**Counterexample field (continuous class, positive probability).** Any
Reg ∩ TYP field in which A holds and M's death saddle is the UNIQUE
window saddle (e.g. on a chart: maxima M of value b and M₂ of value
> b, the pair saddle S of value s, the merge saddle y of value
v ∈ (s,b) pairing M with M₂'s elder component, all other critical points
below s, distinct values; C²-stable, hence a nonempty C³-open class, hence
positive P_r-measure): N_w = 1, N_qual = 1 (D2-COUNT), N_loop = 1 (both
sector components of y ⊆ C). **N_qual + N_loop = 2 > 1 = N_w.**

**Executable confirmation inside the campaign's own frozen ensemble**
(this reviewer, using B1's frozen PL model and predicates faithful to the
continuous definitions): on ALL 175 A-pairs of the 40-field Fourier
ensemble the death saddle is an N_loop member; the smallest exhibited
violation is field `fourier_field(1000,"ens-0",ridge=True)`, pair
(M,S) = (vertex 875 = (21,35), vertex 951 = (23,31)), category A, unique
window saddle w = 915 = (22,35) with key(S) ≈ 1.12601 < key(w) ≈
1.21511 < key(M) ≈ 1.21689: qual_D2(w) = True, n_loop-member(w) = True,
N_w = 1 ⟹ N_qual + N_loop = 2 > 1 = N_w. 132/175 A-pairs violate the
inequality; 176 of the 1,435 N_loop members in the audited ensemble
qualify under D2's true predicate.

**Why the frozen mirror passed (vacuity, exact).** The mirrors
(c2_falsifier.py `alpha_qualifies`, h4_falsifier.py `my_alpha_qual`)
evaluate `inC` at LEVEL key(S) ("cluster's level-key(S) component == C")
and require `sum(inC) == 1`. At M's death saddle both clusters are in C
at level s, so the mirror's alpha predicate returns FALSE there — the
mirror's "N_qual" is 0 on every A-pair, contradicting D2-COUNT's
{N_qual = 1} = A (23/55 sampled pairs mismatch; on every A-pair the
mirror's qual set is empty where the true qual set is the death saddle).
Under the mirror's predicates disjointness is trivial (n_loop requires
ALL clusters in C at level s; its alpha requires EXACTLY ONE), so
F-B1b / F-H4c ("1437/1437, zero leaks") and F-H4d are tautological and
test nothing about the continuous claim. With the correct predicate the
leak is 176/1435.

**Repair (verified; preserves every downstream number).** The assembly's
display P_r(A) + P_r(B1) ≤ E_w + P_r(B1.dir) DOES survive via the
corrected pointwise identity

    N_qual + N_loop·1_{¬A} ≤ N_w   a.s.   (equivalently N_loop′ := N_loop − 1_A
                                           satisfies N_qual + N_loop′ ≤ N_w),

which is immediate from D2-COUNT (N_qual = 1_A, and on A the death
saddle witnesses N_w ≥ 1) and N_loop ≤ N_w: on A the left side is 1 ≤
N_w; on ¬A it is N_loop ≤ N_w. Then P(A) + P(B1) ≤ E[N_qual] +
E[N_loop·1_{¬A}] + P(B1.dir) ≤ E_w + P(B1.dir) (B1 ⊂ ¬A; AO_loop ≤ 1).
I verified the repaired identity on all 432 audited pairs of the frozen
ensemble (0 violations). So: the CONCLUSION the assembly needs is true
and one corrected-identity away, but the frozen Theorem H4-JC as stated
is false and its executable verification is vacuous. Until H4 §2.3 and
C2 §2.3 are re-issued in the corrected form (and D1's "H4-JC" citation
re-pointed), the single-E_w bookkeeping in Theorem D1 v2.0(1) cites a
false lemma; the defensible bound from the frozen text alone is the
un-sharpened 2·E_w + C_RN√Q(B1.dir) + P(B2) + P(B4).

## 7. LPW deterministic witness placement in A.loc (review target 7): PASS

B1 §7: LPW's G_r supplies (deterministically) a path r·γ_* inside rD
from M to z ∈ rD with f(z) > b and min f∘γ > s. By A's path form (T1(b))
this is A; the witness is localizable in rD, and rational approximation
of the endpoint inside the open path-connected component gives the
defining Conn_{rD}(M, q, s+1/k) ∩ {f(q) ≥ b+1/j} — hence G_r ⊂ A.loc =
A ∩ {localizable in rD} ⊂ A, disjoint from B1/B2/B4 (they live in ¬A).
The placement inference is exact and depends only on A.loc's definition.
The deterministic path theorem itself is H1's lane
(H1_v2_hardening/lpw_constant_v3.py carries the certified exact-rational
clearance 103/3840 > 0); the taxonomy layer's use of it is correct. D1
§6's "no LPW conclusion is a premise here" is honored — the upper theorem
consumes only the placement, not the lower bound.

## 8. Own attack battery (independent of the four adversaries)

Constructed fields in B1's frozen PL model (own carves): cascade death
(ATK-CASCADE2: M absorbs a 9.6-pocket at 9.55, dies at 9.5-equivalent
geometry above s → failure/A); near-simultaneous death (two merge
saddles 9.500/9.501 → failure/A, true simultaneity being N); wrap-around
preemption through the i-seam (→ failure/A); hidden elder birth reachable
only below s (→ failure/B2, B.fin content); monkey saddle at S with an
above-s bypass (→ failure/A; pure-monkey sector cases covered by B2a's
ATK-3 series: B1/B2/B4/success); near-tangency above s (B2a's ATK-4a/TIE-3
cover the exact-touch cases: regular contact ⇒ A by the perturbation
argument, critical contact at value s ≠ S ⇒ N); convention-flip ties
(B2c T-TIE: success/failure flips with the convention, N-resident).
Accumulation of saddles onto M's level: excluded analytically (Morse on
compact ⇒ finitely many critical points; complement inside N). Every
constructed failure landed in a named class; none escaped the cover. This
corroborates the four adversaries (B2a: 2,991,090 pairs, 0 uncovered;
B2b: boundary mechanisms all Q_r-null; B2c: tie fronts; B2d: 11,458
all-pairs wrap audits, 0 uncovered — re-run reproduced).

---

## Dispositions (verbatim labels)

- Target 1 (partition F_r = A ⊔ B1 ⊔ B2 ⊔ B4): **PASS**.
- Target 2 (B3 ⊂ N, P_r(N) = 0; Lemma R countersignature; d⁶ live): **PASS**
  (countersigned certificate grade; caveat recorded in §2).
- Target 3 (torus/wrap): **PASS**.
- Target 4 (D2-COUNT; 1612-saddle verification): **PASS**.
- Target 5 (D2-COMPLEMENT 4-channel): **PASS**.
- Target 6 (H4-JC joint carrier): **FAIL** — Theorem H4-JC
  (H4_CLOSURE.md §2.3, body a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3)
  and C2_B_CLASSES.md §2.3 orthogonality paragraph: the disjointness is
  false (level conflation: Qual at level v vs N_loop at level s; M's
  death saddle on A is in both populations); counterexample exhibited
  (continuous class with positive P_r-measure; concretely field ens-0,
  pair (875,951), window saddle 915: N_qual + N_loop = 2 > 1 = N_w);
  mirrors F-B1b/F-H4c/F-H4d used a mismatched level-s alpha predicate and
  are vacuous for the continuous claim (true overlap 176/1435). Repair
  stated and verified (N_qual + N_loop·1_{¬A} ≤ N_w; 432/432 pairs) — it
  preserves D1 v2.0's displays with zero numerical change once re-issued.
- Target 7 (LPW placement in A.loc): **PASS**.

## Load-bearing dependencies (paths + hashes)

- B1_taxonomy/B1_TAXONOMY.md body 7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930 (partition, Lemma B, T1, measurability, LPW placement).
- B1_taxonomy/b1_falsifier.py 818df7985412071259345455c9ff6b68cad0c0fa9d636009cf8849b550154185 (discrete model; re-run PASS).
- D2_branch_control/D2_BRANCH_CONTROL.md body 9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa + d2_falsifier.py 824284be876ce685f63e600a642ee84ad601e6fb2424a852dafe8c21178122d8 (D2-COUNT/COMPLEMENT; re-run PASS, digest 877210c7…).
- C2_B_classes/C2_B_CLASSES.md body 2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2 (C2-B1 absorption — VALID; §2.3 orthogonality — FALSE, see target 6).
- H4_closure/H4_CLOSURE.md body a67d50b9a24c82f970c60c5881eaa41cdf7d4b81cb1cf15904a3fc1b92b6b1d3 (§2.3 H4-JC — FALSE as stated; §2.2 E[N_loop] ≤ E_w — valid).
- H2_foundations/reg_lemmas.py e7005a5af73d80bc1e129ac17ed19ae3d5df5d6667b076efa311f24247043450 + RECEIPT_reg_lemmas.json 48c6494f… (Lemma R countersignature G1/G4/G5; selftest PASS both modes byte-identical).
- H2_foundations/cov_exact.py f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783, pin_transform.py c6988ac7fcfa32dcc03c1374e13fb4c26af1c12c25cb824a315e6dce8991fd41 (exact-law kernels; consumed by the above).
- B2a/B2b/B2c/B2d adversarial reports + harnesses (digests 35555d70…, 8eef9e76…, 2b8099c5…, 48b71a51…; b2d re-run reproduced).
- D1_assembly/D1_ASSEMBLY_v2_0.md body 86882dca… (consumes H4-JC at §2/§8 — citation must be re-pointed to the corrected identity).

## One-paragraph lead summary

The topology layer is sound at its core and defective at exactly one
joint: the partition F_r = A ⊔ B1 ⊔ B2 ⊔ B4, the tie class B3 ⊂ N with
its (now countersigned, d⁶-live) Lemma R null pricing, torus/wrap
coverage, D2-COUNT, D2-COMPLEMENT, and the LPW-in-A.loc placement all
survive independent re-derivation, re-execution, and constructed-field
attack — but Theorem H4-JC ("N_qual + N_loop ≤ N_w a.s.") is FALSE as
frozen: its disjointness conflates the level-v qualification predicate
with the level-s loop membership, and M's death saddle on A sits in both
populations (counterexample in the campaign's own ensemble: field ens-0,
pair (875,951), N_qual + N_loop = 2 > 1 = N_w; 176/1435 loop members
truly qualify; the frozen "1437/1437" mirror used a mismatched level-s
alpha predicate and is vacuous). The assembly's single-carrier display is
nonetheless repairable at zero numerical cost via the corrected identity
N_qual + N_loop·1_{¬A} ≤ N_w (verified 432/432 pairs); until H4 §2.3 /
C2 §2.3 are re-issued and D1's citation re-pointed, the certified rung's
"paid ONCE" step rests on a false lemma (defensible frozen-text bound:
2·E_w). Dispositions: targets 1–5, 7 PASS; target 6 FAIL (exact lemma,
exact gap, counterexample, and repair all stated above).

---

FREEZE
review: STAGE_E/REVIEW_topology.md (topology-only independent review, U2D-UPPER D1 v2.0)
reviewer-exposure: none beyond the mission prompt
reproducibility: all commands standard python3; predicates and carves inline in §6/§8;
  counterexample reproduction:
    import sys; sys.path.insert(0,'B1_taxonomy'); import b1_falsifier as B1F
    f=B1F.fourier_field(1000,"ens-0",ridge=True); dp,dl=B1F.persistence(f)
    B1F.classify_pair(f,875,951,dp,dl)            # -> ('failure','A'); dp[875]==915
    # window saddles of (875,951): [915]; qual_D2(915)=True; n_loop-member(915)=True; N_w=1
dispositions: T1 PASS; T2 PASS; T3 PASS; T4 PASS; T5 PASS; T6 FAIL (H4-JC); T7 PASS
body-sha256(everything before the separator "\n---\n\nFREEZE\n"): 915bb07a55e9f9ec3c12ec37f4c5f57b9b2207c269cfefb5671ce1445bd27f77
body-bytes: 23334
