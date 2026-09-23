BEGIN_FROZEN_BODY
# H4-JC event-level repair (STAGE E REPAIR R1): the joint carrier re-issued

**Campaign:** U2D-UPPER (2D upper theorem), Stage E repair · **Lane:** D2 (branch
control), owning the event-level re-issue of H4-JC
**Date:** 2026-09-14 · **Artifact:** H4JC-R1-20260914-v1.0
**Status:** Theorem H4-JC as frozen in H4_CLOSURE.md §2.3 (body `a67d50b9…`)
is FALSE as stated (the pointwise claim `N_qual + N_loop ≤ N_w a.s.`). The
assembly display it fed is preserved VERBATIM by the event-level repair below
(Lemma H4-JC-R1 + Theorem H4-JC-R1), proved here from the frozen carriers and
mirrored executably with the certified tallies. This artifact is the interface
change D1 v2.1 cites. Frozen carriers are untouched.

**Frozen carriers checked against (steps pinned):**
- B1 taxonomy, body `7d7ddbec6e63b4c2a2c745b20e33a3d92cfdf5cb97588312a8d54fbd1ea68930`
  (the disjoint coverage F_r = A ⊔ B1 ⊔ B2 ⊔ B4 on Reg ∩ TYP).
- C2 B-classes, body `2aba099d630dc7593b1c26bdd2ad3336c1c521e438ba5845edb0b98f2dadf3d2`
  (§2.2: the N_loop definition and the absorption B1 ⊆ {N_loop ≥ 1} ⊔ B1.dir;
  §2.3: the false orthogonality paragraph — see §4.1).
- D2 branch control, body `9e95648acbe5307d7f61f7e7ccc102577b80974b38d03fc760f7ce7766a0b8fa`
  (§1.1: the own-level Qual; §1.3: Theorem D2-COUNT).
- C1 α intensity, body `ca2c02b81a05e4cbc955b1aeab1f4120e9b8d4fda328aa191da08a49db72853f`
  (the raw envelope E[N_w] = E_w = ∫ρ_w).
- Discrete model: `B1_taxonomy/b1_falsifier.py` sha256 `818df798…` (hash-pinned,
  fail-closed on drift).

## 1. The failure, precisely

### 1.1 The two predicates at their consumed levels

**N_qual (D2 §1.1, own level).** A window saddle y (index-1, v = f(y) ∈ (s, b))
qualifies iff, at level v:

    alive : birth(C_M(v)) = b        (M's own-level component tops at M)
    MIX   : exactly one sector component of y equals C_M(v)   (EQUALITY, own level)
    elder : the other sector component C₂(y) has birth(C₂) > b.

**N_loop (C2 §2.2, containment into the level-s component).** A window saddle
S′ ≠ S is a loop witness iff, writing C′±(u) for its two sector components in
O^u at u = f(S′),

    C′⁺(u) ⊆ C  and  C′⁻(u) ⊆ C,   C := C_M(s)

— CONTAINMENT into M's LEVEL-s component; no alive/birth clause. The level-s
reading is load-bearing: C2-B1's absorption proof produces its witness S′ at
the level u where the two merging lineages C″ ≠ C_M(u) and C_M(u) are distinct
components of O^u both contained in C (containment, never equality with
C_M(u)). Any "relocalization" of N_loop to own-level equality breaks that
proof (executable: MUT-B, §3.4).

### 1.2 The counterexample class: the death-saddle double-membership

On A, let y = D_f(M) be M's death saddle, v = f(y) ∈ (s, b). Then:

- y qualifies (D2-COUNT): its two own-level sector components are C_M(v) and
  the elder C₂ — exactly one equals C_M(v), alive holds, elder holds.
- y is an N_loop member: the two own-level components C_M(v) and C₂ MERGE at
  y, so they lie in ONE component of O^t for every t < v — in particular both
  are contained in C_M(s) = C. The containment C′±(v) ⊆ C is exactly what
  C2's definition tests.

Hence on A the SAME saddle fires both counted populations, and whenever the
pair's raw window saddles are few enough (extremal case: a UNIQUE window
saddle),

    N_qual + N_loop = N_w + 1 > N_w,

refuting H4-JC's pointwise claim `N_qual + N_loop ≤ N_w a.s.` The error in
the frozen text is a LEVEL CONFLATION: H4 §2.3 (and C2 §2.3) compared "BOTH
sector components in C" (containment at level s) with "EXACTLY one sector
component in C" (the α witness's own-level EQUALITY with C_M(v)) as if they
were one predicate — they are not: at level s, BOTH of the α death saddle's
sector components are contained in C.

**Certified discrete probe** (B1's frozen PL elder-rule model; consumed probe
`/mnt/agents/output/counter_H4JC.py` and the STAGE-E attack
`UPPER2D/STAGE_E/attack_h4jc.py`, both independently reviewed; re-derived on
this artifact's own code path in h4jc_mirror.py):

- carved quiet A-field with a UNIQUE window saddle (M's death saddle):
  N_qual = N_loop = N_w = 1 for the counts, but the same saddle is in both
  populations, so N_qual + N_loop = 2 > 1 = N_w (certified);
- frozen ensemble (40 fields, 427 close pairs, 1612 window saddles):
  A-pairs = 175; death saddles firing N_loop = **175/175** (the double
  membership is generic, not exceptional); OLD-claim violations
  (N_qual + N_loop > N_w) = **132 of 175 A-pairs**;
- the reviewers' wider scan: 5638/5638 D2-qualifiers fire N_loop
  (7,541-pair scan) — consistent with the structural argument: EVERY
  A-death-saddle is an N_loop member, with no exceptions.

### 1.3 Why the frozen mirrors missed it

F-H4c/F-H4d (h4_falsifier.my_alpha_qual) and C2's F-B1b evaluate the α
predicate's inC at LEVEL key(S): `inC = [compS[vertex] == C for each cluster]`.
For a window saddle w (key(w) > key(S)), w itself lies in O^{key(S)} and all
of its upper-link clusters connect to w at level key(S): they share ONE
level-s component. Hence inC is all-True or all-False and `sum(inC) == 1` is
IMPOSSIBLE on any 2-cluster saddle: the mirrored α population is EMPTY, the
predicate is vacuous, and "1437/1437 loop saddles, none α-qualify" checked a
different, strictly-empty event — never the consumed own-level qualification.
Executable: the vacuity guard (0 firings over all 1612 two-cluster window
saddles) and the structural lemma (both clusters share exactly one level-s
component), both fail-closed in h4jc_mirror.py.

## 2. The event-level repair (proved)

**Lemma H4-JC-R1 (event-level joint carrier).** On Reg ∩ TYP, pointwise,

    N_qual·1{A} + N_loop·1{B1}  ≤  N_w .

*Proof.* The taxonomy's coverage (B1, body 7d7ddbec) gives F_r = A ⊔ B1 ⊔
B2 ⊔ B4 pairwise disjoint on Reg ∩ TYP, so 1{A} + 1{B1} ≤ 1. Both counted
populations are subsets of the raw window saddles: every qualifying saddle
and every loop witness is by definition an index-1 saddle with value in
(s, b), so N_qual ≤ N_w and N_loop ≤ N_w. Hence

    N_qual·1{A} + N_loop·1{B1} ≤ N_w·(1{A} + 1{B1}) ≤ N_w.  ∎

**Theorem H4-JC-R1 (the assembly display, preserved verbatim).**

    P_r(A) + P_r(B1) ≤ E_{P_r}[N_w] + P_r(B1.dir) = ∫_{T²} ρ_w(y) dy + P_r(B1.dir),

and with H4-RN's dir term (H4_CLOSURE.md §3: `P_r(B1.dir) ≤ C_RN(r)·√Q_r(B1.dir)`),

    P_r(A) + P_r(B1) ≤ E_w(r) + C_RN(r)·√Q_r(B1.dir),

the C_RN form consumed by D1 v2.0 §2 display (1) — UNCHANGED.

*Proof (each step pinned to a frozen carrier).*
(1) P_r(A) = E_{P_r}[N_qual] = E_{P_r}[N_qual·1{A}]: Theorem D2-COUNT (D2,
    body 9e95648a §1.3) — N_qual ∈ {0,1} on Reg ∩ TYP and {N_qual = 1} = A;
    in particular N_qual = 0 off A, so N_qual = N_qual·1{A}.
(2) P_r(B1) ≤ E_{P_r}[N_loop·1{B1}] + P_r(B1.dir): C2-B1's absorption (C2,
    body 2aba099d §2.2, proved there: B1 ⊆ {N_loop ≥ 1} ⊔ B1.dir), then

        P(B1) ≤ P(B1 ∩ {N_loop ≥ 1}) + P(B1.dir)
              ≤ E[N_loop·1{B1}] + P(B1.dir)        (1{B1∩{N_loop≥1}} ≤ N_loop·1{B1}).

(3) Sum and apply Lemma H4-JC-R1:

        P(A) + P(B1) ≤ E[N_qual·1{A} + N_loop·1{B1}] + P(B1.dir)
                     ≤ E[N_w] + P(B1.dir) = ∫ρ_w dy + P(B1.dir),

    with E[N_w] = ∫ρ_w the C1 raw envelope (body ca2c02b8) and the dir
    replacement P_r(B1.dir) ≤ C_RN(r)·√Q_r(B1.dir) exactly as in H4 §3. ∎

**What is retracted:** the pointwise claim `N_qual + N_loop ≤ N_w`, the
population-disjointness paragraph (H4 §2.3; C2 §2.3), and the F-H4c/F-H4d/
F-B1b mirrors (vacuous). Nothing else in the frozen carriers used those forms
(§4 audit). The event restriction in Lemma H4-JC-R1 is the MINIMAL repair:
relocalizing N_loop to own-level equality is not available (it excludes
C2-B1's own absorption witnesses — MUT-B, §3.4).

## 3. The replaced executable mirror

Program: `h4jc_mirror.py` (fail-closed ck → SystemExit(1); no bare asserts;
deterministic; integer/tag output only; both modes byte-identical, guarded by
`h4jc_falsifier.py`). Predicates are the CONSUMED frozen forms on an own code
path, cross-checked against the lanes' own instruments on every audited
saddle: qual_v against d2_falsifier.channel_of (hash-pinned import) and
against D(M) = y (D2's F-D2a rechecked); loop_s against an independent
whole-set containment evaluation (every vertex of each own-level sector
component checked to lie in C, vs the single-vertex refinement reading).

### 3.1 Ensemble and tallies (certified; transcripts t_normal.txt/t_opt.txt)

- B1's frozen 40-field ensemble: 427 close pairs, 1612 window saddles
  (N_w sum = 1612), all two-cluster;
- plus B1's frozen synthetics, C2's T-B1W, D2's five channel synthetics
  (T2-QUAL/PRE/NA/LOOP/YNG), and the carved R1-QUIET counterexample field:
  12 synthetic pairs, NEW lemma enforced there too (0 violations).

OLD claim (falsifier display, must NOT shrink — MUT-C):
- A-pairs = 175; death saddles firing N_loop = 175/175;
- OLD-claim violations = 132 (all on A-pairs);
- carved R1-QUIET: N_qual + N_loop = 2 > 1 = N_w, the same saddle in both
  populations.

NEW claim (Lemma H4-JC-R1): N_qual·1{A} + N_loop·1{B1} ≤ N_w on ALL 439
audited pairs — violations: 0 (fail-closed per pair).
Bookkeeping sums (ensemble): Σ N_qual·1{A} = 175 (= #A-pairs, the discrete
P(A)); Σ N_loop·1{B1} = 12; Σ N_w = 1612 — the carrier is paid ONCE at the
event level.

### 3.2 Instrument integrity checks (fail-closed, per saddle)

- qual_v ≡ D2 channel_of = 'QUAL' on all 1612 window saddles (two
  independent code paths);
- qual_v ⟺ D(M) = y on all 1612 (D2's F-D2a rechecked);
- loop_s ≡ whole-set containment on all 1612 (two independent code paths);
- N_qual ≤ 1 and A ⟺ N_qual = 1 on every pair (D2-COUNT form);
- C2-B1 absorption rechecked on every B1 pair (N_loop ≥ 1 or dir, C2's
  dir_test imported hash-pinned);
- the 2-cluster structural lemma (both clusters share ONE level-s component)
  on all 1612.

### 3.3 MUT-A: the vacuous level-s α predicate

alpha_s (the old mirrors' form: inC at level key(S)) fires 0 times over all
1612 two-cluster window saddles and misses all 175 consumed qualifiers. Any
build whose "qualification" predicate is the level-s form fails these checks
and dies.

### 3.4 MUT-B: the relocalized (own-level-equality) loop predicate

loop_eq (both own-level sector components EQUAL to C_M(f(y))) fires on 0 of
the 175 A-death-saddles (the elder side is foreign at own level) and on 0 of
T-B1W's two certified absorption witnesses {412, 413} — exactly the witnesses
C2-B1's proof constructs (lineages C″ ≠ C_M(u) contained in C). The
relocalized predicate therefore cannot serve the frozen absorption; the
event restriction is the minimal repair. Any build swapping the predicates
dies on these checks.

### 3.5 MUT-C: the old claim must fail closed on the tally

The mirror requires the OLD-claim violation count to be exactly the
certified 132 (and the A-restricted count 132/175). If any edit makes the
old claim pass on the ensemble (violations = 0), the build fires — the
failure of the old claim is itself a certified datum.

## 4. Side consequences and the downstream-consumption audit

### 4.1 E[N_loop] ≥ P(A): the consumed N_loop is not the historical loop population

Since every A-death-saddle is an N_loop member (§1.2, structural; 175/175 in
the frozen ensemble; 5638/5638 in the reviewers' wider scan),

    E_{P_r}[N_loop] ≥ P_r(A)   (≈ 0.97·r³ at r = 0.05, historical C*).

Consequences, traced through the campaign:

- **AO_loop is NOT the C021 loop fraction (1.4–3)e-4 of raw.** The
  historical fraction measured the isM-ball loop population — a strict
  subpopulation of the consumed N_loop (which contains every death saddle).
  No small-constant claim for the B1 term may run through the historical
  fraction on the consumed population. Audit of consumption: C2 §2.4 lists
  the historical fraction explicitly as "(consistency-only)"; D1 v1.2 lists
  AO_loop as "constant-only, non-blocking". No theorem-grade object consumed
  the small fraction. The correct sharp object going forward is the
  EVENT-RESTRICTED count E[N_loop·1{B1}] (the restriction removes the
  A-death-saddles: on B1 ⊂ ¬A there are none): the repair's display already
  prices B1 through it, and the branch lane's loop control
  (OBL-B1-BRANCH, D2 §2 G3) is hereby RE-POINTED to the event-restricted
  population — denoted **OBL-B1-BRANCH(loop|B1)**: control of
  P(y ∈ N_loop ∧ B1 | Palm at y), whose smallness is the genuine
  loop-fraction content. The envelope AO_loop ≤ 1 used by H4's display is
  untouched by this re-pointing.
- **C2's B1 power law survives.** The class's "O(r³) candidate" status runs
  through the raw envelope E[N_loop] ≤ E[N_w] = Θ(r³), which holds
  regardless of AO_loop's size; the absorption (Theorem C2-B1) is proved
  independently of the false §2.3 paragraph and stands.

### 4.2 Downstream-consumption audit of the false form

Grep-level trace of every consumer of "N_qual + N_loop ≤ N_w" / "population
orthogonality" across the campaign tree:

| consumer | content | verdict |
|---|---|---|
| C2_B_CLASSES.md §2.3 (body 2aba099d) | the false orthogonality paragraph + F-B1b mirror | false as stated; consumed ONLY by the joint-carrier sharpening; C2's B1 pricing (§2.2) independent and intact; repair supersedes |
| H4_CLOSURE.md §2.3 (body a67d50b9) | Theorem H4-JC + F-H4c/F-H4d mirrors | FALSE; replaced by Lemma/Theorem H4-JC-R1 here; the display it fed survives verbatim |
| H4_CLOSURE.md §2.4 power ledger | E[N_loop] ≤ E_w = Θ(r³) | unaffected (raw envelope; no use of the disjointness) |
| D1_ASSEMBLY.md line 463 gloss | "H4-JC joint carrier pays the raw window-saddle count ONCE" | the paid-once bookkeeping is saved by the event-level lemma; the named citation updates to H4JC-R1 (display and C_RN ≤ 3.47 certificate untouched) |
| D1_ASSEMBLY_v1_2_ADDENDUM.md §H4-JC (lines 78–84) | quotes H4-JC verbatim (`N_qual + N_loop ≤ N_w a.s.`) | superseded by this artifact (D1 v2.1 to cite H4JC-R1) |
| D1_ASSEMBLY_v2_0.md §2 display (1), line 89 | `1 − q ≤ E_w + C_RN·√Q(B1.dir) + P(B2) + P(B4)` | PRESERVED verbatim by Theorem H4-JC-R1 (the joint-carrier input now proved at the event level) |
| D1_ASSEMBLY_v2_0.md line 69–70 gloss | "paid ONCE (H4-JC: N_qual + N_loop ≤ N_w a.s.; 1437/1437 mirror)" | the gloss's claim form is false (mirror vacuous); the paid-once bookkeeping itself is saved by the event-level lemma; v2.1 gloss to cite this artifact |
| D2_branch_control (body 9e95648a) | — | UNAFFECTED: D2-COUNT/D2-COMPLEMENT use only the own-level qualification; D2's LOOP_y is an own-level event, a different predicate from N_loop membership; no D2 statement consumed the orthogonality (checked line by line) |
| certified constants (H5's I_hi, C_chart; H4-RN's C_RN) | — | untouched: no numerical certificate depended on the disjointness |

**Net interface change for D1 v2.1:** replace the citation "H4-JC
(N_qual + N_loop ≤ N_w)" with "H4JC-R1 (event-level joint carrier:
N_qual·1{A} + N_loop·1{B1} ≤ N_w)"; the display (1) and every certified
constant in it stand unchanged.

## 5. Certificates and falsifier

Artifacts (this directory):
- `h4jc_mirror.py` — the replaced executable mirror (§3). Transcripts
  `t_normal.txt` / `t_opt.txt` (byte-identical).
- `h4jc_falsifier.py` — regression driver: reruns the mirror under python3
  and python3 -O, requires exit 0 and byte-identical transcripts, and
  re-checks the certified headline tallies (427/1612; OLD 132/175 with
  175/175 double-firing; NEW 0 violations; vacuity and MUT-A/MUT-B/MUT-C
  guards). Transcripts `tf_normal.txt` / `tf_opt.txt` (byte-identical).
- `FREEZE.txt` — this body's hash and all artifact receipts.

Discipline: fail-closed (ck → SystemExit(1)); zero bare asserts (python -O
safe); deterministic (B1's seeded frozen ensemble; sorted iteration;
integers/tags only in the digest; no wall clock); freeze-before-consume on
all imported lanes (b1_falsifier 818df798…, c2_falsifier 2441b630…,
d2_falsifier 824284be…); receipts separate from the body.
END_FROZEN_BODY

## Freeze record (outside frozen body)

Extraction rule (corpus convention): exclude the unique marker lines
BEGIN_FROZEN_BODY / END_FROZEN_BODY, normalize line endings to LF, exactly
one terminal LF; the body's last content line directly precedes the END
marker (unambiguous).

- Frozen body bytes / SHA-256: see FREEZE.txt.
- Companions: h4jc_mirror.py, h4jc_falsifier.py, t_normal.txt, t_opt.txt,
  tf_normal.txt, tf_opt.txt (hashes in FREEZE.txt).
- Consumed counter-probes (verified by re-derivation on this artifact's own
  code path): /mnt/agents/output/counter_H4JC.py;
  UPPER2D/STAGE_E/attack_h4jc.py.
