# REV-RN3-FARZONE-20260918 — nonauthor technical review of the RN3 joint far-zone result

Companion to `reviews/records/REV-RN3-FARZONE-20260918.json`, which is the
machine-checked record. Where the two differ, the JSON governs.

| | |
|---|---|
| Route | `RV-RN3` (Review Queue technical status **READY**, age 0, Aging action NEW) |
| Object | `RN3-20260917-b9c2_RN_JOINT_PROOF.md` — "RN-JOINT-001 — Uniform joint comparison on the complete far region" |
| Drive id | `1f0EBsq6UIZ0jDCABT251TTNEWYCVA0FO` |
| Bytes / digest | **12,956** / `0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373` — matches the register row exactly |
| Author family | **openai** (register column, Work-Events/claim rows, and the object's own §8) |
| Reviewer family | **anthropic** — `session_01Cz7WZybv8znP64SpPj6sWY` |
| Technical verdict | **AMEND** |
| Organizational independence credit | **0** |
| Gate status after this record | **UNCHANGED** |

> **Read this first.** This is a verdict on an object. It moves no gate, changes no
> register, and creates no independence. RV-RN3's `Independence status` is
> `EXTERNAL_REVIEW_OPEN` and stays exactly that whatever verdict appears above.
> R17 §4 permits this review and prices it at zero organizational independence;
> both halves hold at once.

---

## 1. How the object was obtained, and one incident worth recording

`python3 tools/drive_index.py sha 0c9446b7` resolves the digest to
`01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/RN3-20260917-b9c2_RN_JOINT_PROOF.md`,
and `python3 tools/drive_index.py find RN3` gives the Drive id and the 12.7 KB size.
The body was downloaded with the Drive connector, base64-decoded, and hashed
before anything was read from it.

The first local copy I made was **corrupted in transcription** — two non-base64
characters desynchronised the stream — and hashed to `4834e065…` at 12,957 bytes.
That is a mismatch against the register, and I did not proceed on it. I recovered
the untouched connector payload from this session's own record of the tool
response and decoded that instead: **12,956 bytes**, digest
`0c9446b7cb…de1373`, matching the RV-RN3 row's `Body SHA-256` and `Body bytes`
exactly. Diffing the two copies showed the corruption was confined to four LaTeX
tokens (`\qquad` for `\quad` twice, `(I-E)_{-1}` for `(I-E)^{-1}`, `J{\rm ref}`
for `J_{\rm ref}`); I reread those lines from the verified bytes. Every quotation
below is from the verified bytes.

**There is no digest disagreement with the register.** The object's digest and byte
count are exactly what the Review Queue records.

## 2. Exposure disclosure

Written before any verdict was formed. The full text is the `exposure_disclosure`
field of the JSON record; its substance:

- Before reading a byte of the proof I had read `docs/RESEARCH_MAP.md` §3, which
  **tabulates this object's headline numbers** (13,604 boxes, `0.88021 < ρ/ρ_ref < 1.12058`,
  `2.22542 r³ < I_far < 2.83312 r³`, 294 checks, 24 nine-pin laws, 384-bit Arb)
  along with its byte count and digest.
- I had read `research/README.md` and `research/rn/moment_envelope.py`, both of which
  **assert the answer to the RN5 scope question I was asked to test**.
- I had read `docs/OPEN_PROBLEMS.md` §§A5/D/E, the RV-RN3 and RV-RN-ALIGN queue rows
  including the note *"Author replay only; alternative Anthropic far-zone work is not
  a review"*, several GP-REG-032 export rows about the RN3 claim and its successors,
  plus R17, OP-PROT-012 and the `reviews/` schema and checker.
- I did **not** read the bundle, `rn_joint.py`, `audit_rn_joint.py`, `rn_field.py`,
  the H3 certificate, or CL-RNU-003.
- Bias this creates: every arithmetic agreement below is a **confirmation, not a blind
  prediction**; I was told the RN5 answer in advance by two repository files; the map's
  framing primes acceptance of exactly the interval-run constants I could not check;
  and this session produced Anthropic-side far-zone interval code earlier in the same
  workflow, which I did not consult while reconstructing.

## 3. The object's precise hypotheses, in its own terms

1. Fixed rung r = 1/20 only. The object states no result for any other r; its own closing section says 'Uniformity in r ... require their own evidence'.
2. Fixed endpoint geometry: M = (-r/2, 0), S = (r/2, 0) on the axis, side-24 square torus, b = 6/5. The object states that other endpoint orientations require their own evidence.
3. The six endpoint pins are exactly c = (b,0,0,b-r^3/6,0,0)^T in the order (f,fx,fy) at each endpoint, and the Hessian order is (fxx,fyy,fxy).
4. Certified spatial and mark domain: 5 <= |y| <= 17 in the declared coordinate lift, and b - r^3/6 <= v <= b, with all directions and both mark endpoints included.
5. The normalizer floor Z >= Z_lo = 0.0077592917375327855 is IMPORTED from the H3 certificate; the object says in terms 'This round imports that result; it does not independently re-prove the H3 floor.'
6. The prior interval kernel closure_round2/rn_field.py, SHA-256 d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8, together with the image-tail and spectral-tail inequalities documented in the prior proof, is assumed correct.
7. The Kac-Rice integrand rho(y,v) = p_{Y|C=c}(v,0,0) E_{P(y,v)}(Wg)/Z under the W-weighted endpoint Palm law is assumed, not derived here.
8. Delta = ||E||_F < 1 pointwise on the whole certified domain; the chi-square lemma (1) is false without it.
9. Beta = ||B||_F^2/e_* < 1 with e_* = min(1, a_2), a_2 = -k''(0) > 0, for the Y-density ratio bounds (6) and (7).
10. I_far is an expected raw saddle count under the declared conditioned law; the object states it 'is not itself a probability or a complete chart-plus-remote theorem'.

## 4. Reconstruction

Full text in the JSON. The chain, with what I recomputed at each step:

| Step | Object's claim | What I did |
|---|---|---|
| Model | `K(x)=k(x₁)k(x₂)`, side-24 periodised Gaussian, `a₂=-k''(0)` | Recomputed `a₂ = 1` and `k''''(0) = 3` to double precision from image sums; the reference `Q_y^0(v)=N((-v,-v,0),diag(2,2,1))` **is** the planar one-point conditional Hessian law (mean `-a₂v`, variance `3-a₂²=2`, `Cov(fxx,fyy|f)=0`) |
| §2 lemma (1) | `log(1+χ²) = -½logdet(I-E²) + mᵀ(I-E)⁻¹m ≤ δ²/(2(1-δ²)) + ‖m‖²/(1-δ)` | Verified the equality against direct integration of `p²/q` (closed form 0.132910997260, quadrature 0.132910997260); verified the bound on 200 random symmetric `(m,E)` with `δ<1`, worst ratio 0.978 |
| §3 moments (2) | `E(T-c)₊=2e^{-c/2}`, `E(T-c)₊²=8e^{-c/2}`, `m₀(v)=√2e^{-v²/4}`, `E_Q g²=4m₀` | Reproduced all four by quadrature: `E_Q g² = 3.946653289904935` vs `4m₀ = 3.946653289905143` |
| §3 (3) | `η = 2h₂√χ² / (Z_lo √m₀,min)` | Rebuilt `η = 0.116820011` from the table's own `h₂`, `χ²`, `Z_lo` and `m₀,min` (table: 0.116819988) |
| §4 (4)–(5) | whitening + Gaussian conditioning blocks | Checked each block against the conditioning identity, including why `v(C_yD⁻¹e₀+(1,1,0)ᵀ)` appears (displacement from the reference mean `(-v,-v,0)`) |
| §5 (6)–(7) | `R_Y ≤ (1-β)^{-3/2}e^{bu/e*}`, `R_Y ≥ exp[-(b²β+2bu+u²)/(2e*(1-β))]` | (7) is an **exact rearrangement**, not a loosened estimate: verified `(b²β+2bu+u²)/(1-β) = 2bu+u²+β(b+u)²/(1-β)` as an exact rational identity on 500 random triples |
| §6 integration | `J_ref ≈ 6.3529298594549898874e-7`, area `576-25π` | 80-digit Decimal: `6.3529298594549898873523e-7` — **20 significant digits**; area `497.460183660255169`; `12√2 = 16.97 < 17` |
| §6 headline | `0.00027817807867916 < I_far < 0.00035413986017516`, i.e. `2.22542 r³ … 2.83312 r³` | Implied unrounded ratios `0.880218550` and `1.120578860`, both just inside the displayed `0.88021`/`1.12058` — the displayed values are rounded outward as claimed; `I_far/r³ = 2.225424629` and `2.833118881` |
| §8 trace series | remainder `δ²²/[22(1-δ²)]` after ten terms | Verified at 80 digits at `δ = 0.0436, 0.5, 0.97` |

**The strongest structural result of this review**: the certified table is not a list
of independent assertions. Feeding the tabulated `‖E‖_F` and `‖m‖` into lemma (1)
reproduces the tabulated `χ²` row to seven significant digits; feeding that plus `h₂`,
`Z_lo` and `m₀,min` into (3) reproduces the `η` row; and `(1+η)·R_Y,hi` reproduces
the `1.12058` ratio. Four rows are recomputable from three others plus the imported
floor.

## 5. Negative controls — including the two that did not fire

| # | Control | Fired? | Result |
|---|---|:--:|---|
| 1 | Recount the quarter-unit cover of the closed annulus | **yes** | **13,612**, not 13,604 — see §6 |
| 2 | Make the RN5 substitution: `√(E_Q g⁴)` for `√(E_Q g²)` | **yes** | `η: 0.116820 → 0.809353`, ratio `1.12058 → 1.815443`, **budget 0.6798 exceeded** |
| 3 | Flip the mark-window extremum (min → max of `m₀`) | **no** | `η: 0.116820011 → 0.116819281`; displayed `1.12058` unchanged |
| 4 | Remove the `δ<1` hypothesis of lemma (1) | **yes** | partial integrals diverge at `δ = 1.0` (2.39 → 4.79 → 9.57) and `δ = 1.2` (4.27 → 256 → 3.98e10) |
| 5 | Attack the ten-term trace remainder at 80 digits | **no** | holds at all three `δ`: `5.333810e-32` vs `5.334656e-32` at the certified `δ` |
| 6 | Inflate `‖E‖_F` by 0.1% / 1% / 10% | **yes** | `χ²` row breaks at **0.1%** (`1.034467e-3` vs `1.032557e-3`) |
| 7 | Perturb the headline ratio to 1.12000 / 1.10000 | **yes** | fails to cover the object's own `I_far` — the constant is binding at the fifth decimal |
| 8 | Inflate the imported `Z_lo` by 10% | **yes** | `η → 0.106200`, ratio `→ 1.109923` — the imported floor enters the headline linearly |
| 9 | Test zone exhaustiveness | **yes** | the open disc `|y| < 0.1` belongs to neither zone |
| 10 | Naive double-precision `erf` difference (cancellation) | **no** | rel. error `6.8e-12` — about four digits lost, irrelevant at 384 bits |

**Control 3 not firing is a finding.** The mark window is `r³/6 = 2.083e-5` wide, so
at this rung the mark-window half of the uniformity claim is numerically inert: the
certificate is, in numbers, a statement at the single point `v = b`.

**Control 5's history is also worth recording.** A double-precision version of it
reported 101 violations out of 200 at `δ = 0.0436`. Those were floating-point noise
at `1e-17` against a true remainder of `1e-32` — my instrument, not the object. A
reviewer who stopped at the double-precision run would have filed a false defect.

## 6. Findings by criterion

### INFO — RN5 determinant-moment erratum, scope of the far-region proof

The documented claim that RN3's far-region proof is outside the RN5 affected scope is CONFIRMED for sections 1-8, and confirmed for a reason stronger than the one the repository gives. The RN5 defect was an exponent-selection error inside a Holder(4,4,2) envelope across three determinants. RN3 never applies Holder across the three determinants at all: W (the pair factor) and g (the y factor) are independent under the product reference law Q, so ||Wg||_2 = ||W||_2 ||g||_2 is an identity, and the y factor enters as the EXACT second moment E_Q g^2 = 4 m_0(v), which I re-derived and reproduced by quadrature. The pair factor h_2 = (E(det H_M)^4 E(det H_S)^4)^{1/4} is exactly the factor the CORRECT Holder(4,4,2) also carries, so it is not the defective substitution either.

*Evidence.* Quadrature: E_Q g^2 = 3.946653289904935 against 4 m_0 = 3.946653289905143; m_0(1.2) = 0.986663322476234 against sqrt2 e^{-0.36} = 0.986663322476286. Sensitivity control NC2: substituting sqrt(E_Q g^4) = sqrt(192 m_0) for sqrt(E_Q g^2) moves eta from 0.116820011 to 0.809352777 and the far ratio to 1.815443, breaking the object's own 0.6798 budget - so the correct exponent is load-bearing and its use here is a property of the argument, not an accident of scale.

### MAJOR — RN5 determinant-moment erratum, scope of section 9

The same claim is NOT true of the whole object. Section 9's conditional arithmetic imports the near target 17.6804 r^3, and docs/RESEARCH_MAP.md section 3 records that the older published 17.6804 r^3 used a different mark-cap heuristic and sits beside the wrong-power diagnostic 17.67237 r^3, against a corrected diagnostic of about 2.34195 r^3 - a factor of about 7.5. So the displayed sum I_near + I_far < 20.51352 r^3 is built from a number the RN5 repair places inside, or immediately adjacent to, the affected scope. The repository sentence 'RN3's far-region proof is outside the affected scope' is accurate about the far-region proof and about nothing else in this file; a consumer who reads it as covering the object as a whole will carry the 20.51352 r^3 figure out of the affected scope by mistake. The object does hedge the line as 'Conditional arithmetic only' and 'not presently a full remote certificate', which is why this is MAJOR and not BLOCKING.

*Evidence.* Object section 9, verified bytes. docs/RESEARCH_MAP.md lines 179-186. 17.6804 + 2.83312 = 20.51352 exactly, so the sum is arithmetically what it claims to be; the objection is to the provenance of its first term, not its addition.

### INFO — Uniformity of the bounds claimed uniform

Every bound the object calls certified is genuinely a sup-over-region statement and not a pointwise one, in the only sense the argument needs: lemma (1) is applied with a single delta and a single ||m|| valid on the whole domain, and the final integral uses one global sup and one global inf of rho/rho_ref times the reference integral, so no per-box quadrature and no interchange of sup and integral is involved. The uniformity is over (y,v) in 5 <= |y| <= 17 times [b-r^3/6, b] at the single rung r = 1/20 and the single axis orientation. It is NOT uniform in r, not over endpoint orientations, and not over the near zone, and the object says so in its own closing paragraph.

*Evidence.* Rebuilt chain: tabulated ||E||_F and ||m|| -> lemma (1) -> chi^2 <= 1.032557e-3 (tabulated row to 7 digits) -> with h_2, Z_lo, m_{0,min} -> eta = 0.116820011 (tabulated 0.116819988) -> (1+eta) x 1.003365693 = 1.1205789 -> displayed 1.12058.

### MINOR — Uniformity over the mark window is inert at this rung

The mark-window half of the uniformity claim carries no numerical content at r = 1/20. The window is r^3/6 = 2.083e-5 wide; replacing the certified minimum of m_0 over the window by its maximum - the unsafe direction - moves eta by 7e-7 and leaves every displayed figure unchanged. The certificate is, numerically, a statement at v = b. This matters because 'complete mark window' reads as substantive coverage, and a consumer could take it as evidence of robustness that the object has not demonstrated.

*Evidence.* NC3: eta 0.116820011 -> 0.116819281; displayed ratio 1.12058 unchanged.

### MINOR — Spatial cover: membership rule does not reproduce the stated count

The text says the cover is 'every quarter-unit Cartesian box intersecting the closed annulus' with membership by 'exact integer inequalities against 25 and 289'. That rule, applied literally, selects 13,612 boxes, not the 13,604 the object states and the register carries. The difference is exactly the 8 boxes whose maximum squared radius equals 25 exactly, i.e. the ones that meet the closed annulus only at a corner on the inner circle, at (+-3,+-4) and (+-4,+-3); the stated count corresponds to a strict inner comparison. I checked the consequence and it is benign - each excluded box shares that single corner with an included neighbour, so the domain remains covered and the uniform bounds are untouched - but a replayer cannot reproduce 13,604 from the rule as written, and the box count is one of the object's own runtime gates. The text should state the strict inner inequality.

*Evidence.* Exact integer recount in quarter units: closed rule 13,612; boxes tangent to |y|=5 only: 8, giving 13,604; boxes tangent to |y|=17 only: 16, giving 13,596.

### MINOR — Zone decomposition: disjointness and exhaustiveness

The pieces are disjoint where it matters and the far piece is exhaustively covered: the quarter-unit boxes have disjoint interiors, the fundamental square's far part lies inside the certified annulus because 12 sqrt2 = 16.97 < 17, and the overcoverage (boxes evaluated in full including their parts outside the annulus) is safe in both directions because the bound is a global sup and a global inf. But the object's two named zones do not exhaust the domain: far is |y| >= 5 and near is 0.1 <= |y| <= 5, leaving the open disc |y| < 0.1 unaccounted for in the section 9 sum.

*Evidence.* Area of the far square: 576 - 25 pi = 497.460183660255169, reproduced. NC9. The disc |y| < 0.1 has area pi/100 and appears in neither zone.

### INFO — Fitted versus derived constants

I found no fitted constant in the object's own derivation chain, and I looked specifically for one. Every constant I could check is either derived, exactly computed, or declared as an import. Lemma (1) is an identity plus two elementary inequalities; (2) is exact Gaussian integration; (7) is an exact rearrangement rather than a loosened estimate; the ten-term trace truncation carries a rigorous remainder; J_ref is closed form and reproduces to 20 significant digits; the area and the final arithmetic reproduce exactly; the algorithmic parameters (384 bits, degree-28 grouped Taylor, quarter-unit boxes, ten trace terms) all carry rigorous enclosures rather than tuned tolerances. The two decimals not derived here, Z_lo = 0.0077592917375327855 and the 0.6798 budget, are imports, and the object declares the first as such.

*Evidence.* (7) identity verified as an exact rational identity on 500 random (b,u,beta). Trace remainder verified at 80 digits at delta = 0.0436, 0.5, 0.97. J_ref: 6.3529298594549898873523e-7 against the object's 6.3529298594549898874e-7.

### MAJOR — Source identity of load-bearing imports

The object pins its code dependency by SHA-256 and its normalizer by exact decimal, but two load-bearing imports carry no identity at all. First, the Kac-Rice integrand rho(y,v) = p_{Y|C=c}(v,0,0) E_{P(y,v)}(Wg)/Z is introduced as 'the source Kac-Rice integrand under the W-weighted endpoint Palm law' with no derivation, no citation and no hash - yet the entire meaning of I_far, and hence of the headline, is whatever that formula means. Second, the '0.6798 budget' the headline is declared to be below has no source in the object and none I could resolve locally. This matters beyond bookkeeping: docs/RESEARCH_MAP.md warns that RN3's 0.12058 and CL-RNU-003's 0.67728 'measure different objects and must not be compared until law, normalization, kappa definition and domain are matched', and that reconciliation is the open route RV-RN-ALIGN. The headline's comparative clause is therefore a comparison whose commensurability is itself an open question.

*Evidence.* Object sections 6 and the opening paragraph, verified bytes. Object section 7 by contrast pins rn_field.py as d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8 and section 1 pins Z_lo to 19 digits. docs/RESEARCH_MAP.md, RN3 <-> CL-RNU-003 crosswalk paragraph. grep across the repository finds 0.6798 nowhere outside this object.

### MAJOR — The quantitative core was not replayed

Everything downstream of three numbers is reproducible from the document and I reproduced it. The three numbers themselves - ||E||_F < 0.043544631, ||m|| < 0.008864422 and h_2 < 0.014009957, the outputs of the 384-bit interval run over the box cover - are not. Nor are the 294 audit checks, the 24 full Gaussian laws, the covariance floors, the image-tail and spectral-tail enclosures, the degree-28 grouped Taylor remainders, or the claim that the runtime gates stay active under Python -O. I did not obtain the 2.6 MB bundle that contains rn_joint.py, audit_rn_joint.py and RN_MODE_REPLAY.json, so the register's own next action for this route - 'replay RN3 bundle and challenge every uniform bound' - is half done: I challenged every uniform bound in the proof text and replayed none of the bundle. No verdict in this record covers the interval run.

*Evidence.* Bundle 1z_zNdgxtbs7XWxLu56HN8_QbVXxdmMMq, SHA-256 1a4df0e9fd264004d4ab876a8034fe61aea11d05e4d3c24873d0c3b741ff6766, named in the RV-RN3 row and in the object's own dependency list; not downloaded. NC6 shows a 0.1% error in ||E||_F alone would break the tabulated chi-square row.

### INFO — Numerical remarks not reconstructable from the body

The remark that 'at the torus corner the correction is approximately 2.4e-112' cannot be reconstructed from the body. The natural candidates I computed from the image sums do not give it: K at the corner is k(12)^2 = 1.158e-62, the periodisation correction to a_2 is about 1.2e-122, and the square of the fourth-derivative image coupling is 5.30e-116. The figure is an output of code I did not obtain; it is reported here only so that a later reviewer with the bundle knows this line is unchecked.

*Evidence.* k(12) = 2 e^{-72} = 1.0760372320042276e-31; (k''''(12) k(12))^2 = 5.295701154937186e-116.

### INFO — Presentation and firewall risk

Writing a single-rung result as '2.83312 r^3' invites exactly the composition the object forbids in its last paragraph. The r^3 is inherited from the mark-window width r^3/6, not from far-zone geometry: rho_ref(b) x r^3/6 = 6.35281e-7 already reproduces J_ref = 6.35293e-7 to five significant digits, the remainder being the window's curvature. At a fixed rung the coefficient 2.83312 carries no information about r-dependence, and nothing here may be composed with the 3D lifetime track or with any all-small-r statement.

*Evidence.* rho_ref(b) = 0.030493491568578918; rho_ref(b) x r^3/6 = 6.352810743453941e-7 against J_ref = 6.3529298594549898874e-7, relative gap 1.87e-5.


## 7. Does the RN5 determinant-moment erratum touch this object?

`research/README.md` and `docs/OPEN_PROBLEMS.md` say RN3's far-region proof is
outside the affected scope. I was asked to check that rather than assume it. The
answer is **split**, and both halves matter.

**Sections 1–8: confirmed, for a stronger reason than the one recorded.** The RN5
defect was an exponent-selection error inside a Hölder(4,4,2) envelope across three
determinants — `(E C⁴)^{1/2}` where `(E C²)^{1/2}` belongs, which *lowers* the bound
for small determinants and so is not an upper bound at all. RN3 **never applies
Hölder across the three determinants**. Under the product reference law `Q = Q6 ⊗ Q_y^0`
the pair factor `W` and the `y` factor `g` are independent, so `‖Wg‖₂ = ‖W‖₂‖g‖₂`
is an *identity*, and the `y` factor enters as the **exact second moment**
`E_Q g² = 4m₀(v)`, which I re-derived and reproduced by quadrature. The pair factor
`h₂ = (E(det H_M)⁴ E(det H_S)⁴)^{1/4}` is precisely the factor the *correct*
Hölder(4,4,2) also carries.

And the exponent is load-bearing, so this is not vacuous. Control 2 makes the RN5
substitution deliberately: `η` moves from `0.116820` to `0.809353` and the far ratio
from `1.12058` to `1.815443`, **breaking the object's own 0.6798 budget**. Had RN3
made RN5's substitution, its far-region target would have failed.

**Section 9: not confirmed — the opposite.** The conditional arithmetic imports the
near target `17.6804 r³`. `docs/RESEARCH_MAP.md` §3 records that this older published
figure *"used a different mark-cap heuristic"* and sits beside the wrong-power
diagnostic `17.67237 r³`, against a corrected diagnostic of about `2.34195 r³` — a
factor of roughly 7.5. So the displayed `I_near + I_far < 20.51352 r³` is built on a
number the RN5 repair places inside or immediately beside the affected scope. The
addition itself is exact (`17.6804 + 2.83312 = 20.51352`); the objection is to the
provenance of the first term.

**Consequence for the documentation.** The sentence *"RN3's far-region proof is
outside the affected scope"* is accurate about the far-region proof and about nothing
else in the file. A consumer who reads it as covering the object as a whole will carry
the `20.51352 r³` figure out of the affected scope by mistake.

**What this does not settle.** The *exponent structure* is correct and outside the
defect. The *implementation* of the fourth determinant moments inside `rn_joint.py` —
"a finite recurrence with interval inputs" — was never inspected, so an internal
repeat of the `envelope_v` error inside that routine is excluded by nothing here.

## 8. Unresolved dependencies

1. The proof-code bundle 1z_zNdgxtbs7XWxLu56HN8_QbVXxdmMMq (SHA-256 1a4df0e9fd264004d4ab876a8034fe61aea11d05e4d3c24873d0c3b741ff6766), containing rn_joint.py, audit_rn_joint.py and RN_MODE_REPLAY.json: not obtained, not executed, not replayed. Every interval-arithmetic claim rests on it.
2. The prior interval kernel closure_round2/rn_field.py, SHA-256 d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8, and the image-tail and spectral-tail inequalities the prior proof is said to document in full: not obtained.
3. The H3 certificate supplying Z_lo = 0.0077592917375327855: not obtained. The object states it imports and does not re-prove this floor, and NC8 shows the headline moves linearly with it.
4. The definition of the W-weighted endpoint Palm law and of the Kac-Rice integrand rho(y,v): asserted in section 6 with no source identity.
5. The 0.6798 budget the headline is compared against: no source identity in the object, and not resolvable inside this repository.
6. CL-RNU-003, and with it route RV-RN-ALIGN (NEEDS_RECONCILIATION, 'bind both objects first'): the RV-RN3 next action asks for a comparison of exact definitions against CL-RNU-003, which I did not obtain and did not perform.
7. The near-annulus integral on 0.1 <= |y| <= 5: unwritten, and its displayed 17.6804 r^3 target is itself flagged in the repository as an older figure from a different mark-cap heuristic.
8. The uncovered disc |y| < 0.1: named by neither zone of the object.
9. D3-LEMMA-RN-UNIF Piece 1 and Piece 2 (the annulus Riemann-sum driver, recorded as unwritten): both OPEN, and the object states it does not act on the parent lemma.
10. The implementation of the fourth determinant moments E(det H_M)^4 and E(det H_S)^4 by 'a finite recurrence with interval inputs': the EXPONENT structure is correct and outside the RN5 defect, but the recurrence itself was not inspected, so an internal repeat of the envelope_v error inside that routine is not excluded by anything in this record.
11. Uniformity in r, other endpoint orientations, chart integration, and the near and all-small-r regions: the object itself names these as requiring their own evidence, and nothing here supplies any of them.

## 9. Verdict

**Technical verdict: `AMEND`** — one of the register's own seven R17 statuses.

I found **no mathematical error** in the far-region derivation, and I looked hard:
every identity, inequality and arithmetic step I could check reproduces, several to
twenty significant digits, and the certified table is internally consistent to seven.
The AMEND is on the object's **text**, on four concrete items:

1. the cover membership rule as written yields 13,612 boxes, not the stated 13,604
   (strict inner comparison needed in the text);
2. §9's conditional sum imports a near target the RN5 repair places in or beside the
   affected scope;
3. the two named zones leave the open disc `|y| < 0.1` unaccounted for;
4. two load-bearing imports — the Kac–Rice integrand and the `0.6798` budget — carry
   no source identity, in a document that otherwise pins its kernel by SHA-256 and its
   normalizer to nineteen digits.

**Scope/dependency verdict** (the separate R17 §4 dimension) is in the JSON record.
In short: a single rung, a single axis orientation, one spatial annulus and a mark
window `2.08e-5` wide; sound where checkable; resting on four things I did not verify
and one that has no identity at all.

**The register's own next action for this route was *"replay RN3 bundle and challenge
every uniform bound"*. I challenged every uniform bound in the proof text and replayed
none of the bundle.** That half of the brief is not done, and no verdict here covers
the interval run.

## 10. Independence

**`independence_credit = 0`.** Three independent reasons, any one sufficient:

1. **R17 §4** — this session is Anthropic-family; same provider is zero organizational
   independence, while the technical review itself is permitted.
2. **The cross-provider reading does not rescue it.** The author lineage here is
   *OpenAI*, not Anthropic, so a naive reading might expect cross-provider credit. R17 §4
   forecloses it: *"Different provider alone does not establish independence."* And the
   OP-PROT-012 §5 predicate fails on three clauses — **(c)** this session had read the
   repository's tabulation of the object's headline numbers, its digest and the
   repository's own answer to the RN5 question *before* reading the object; **(b)** no
   task specification was frozen before execution; **(e)** the result was known to me
   before my recomputation was frozen.
3. **Lineage.** The RV-RN3 row itself records *"Author replay only; alternative Anthropic
   far-zone work is not a review"*, and this session produced Anthropic-side far-zone
   interval code earlier in the same workflow.

**The independence-requiring gate on this route remains OPEN**, and would remain open
identically had the verdict been `PASS_TECHNICAL` or `FAIL`. R17 §4: *"A task may finish
its technical review while an external-independence predicate remains open."* Nothing in
this record touches RV-RN3's `Independence status` (`EXTERNAL_REVIEW_OPEN`) or its
`Reviewer / claim` column (`UNASSIGNED`); no register was edited.

## 11. What this review does not establish

This record is a verdict on 12,956 bytes and nothing else. Specifically, it does not establish any of the following.

1. It does not establish that the far-region bound 2.22542 r^3 < I_far < 2.83312 r^3 is true. The three interval-run constants that carry the whole quantitative claim - ||E||_F < 0.043544631, ||m|| < 0.008864422, h_2 < 0.014009957 - were not replayed, because I did not obtain the 2.6 MB bundle. What I verified is that IF those three numbers and the imported floor Z_lo hold, THEN everything the object derives from them follows, and the table's own rows agree with each other to seven significant digits. A 0.1% error in one of those three inputs would break the chi-square row.

2. It does not verify the 294 audit checks, the 24 full nine-pin Gaussian laws, the covariance floors, the image-tail and spectral-tail enclosures, the degree-28 grouped Taylor remainders, the Arb runtime, the -O gate behaviour, or the normal-versus-optimized replay agreement.

3. It does not confirm the Kac-Rice integrand. If rho(y,v) is not the right integrand for the W-weighted endpoint Palm count, everything in this review is a correct analysis of the wrong quantity, and the object supplies no identity by which to check it.

4. It establishes nothing about the near annulus 0.1 <= |y| <= 5, nothing about the disc |y| < 0.1, nothing about any r other than 1/20, nothing about other endpoint orientations, and nothing about chart integration. It does not act on D3-LEMMA-RN-UNIF Piece 1 or Piece 2, on OBL-H5-JETMOD, OBL-H5-ZBAND, OBL-H5-REMOTE-THRESHOLD, PERC-DECAY, OBL-B1-BRANCH or B4.loc; every one of those obligations stands exactly as it stood before.

5. It creates no organizational independence and no fraction of any. RV-RN3's Independence status remains EXTERNAL_REVIEW_OPEN, its Reviewer/claim column remains UNASSIGNED, and no register was edited by this review. The technical verdict AMEND is not a step toward the independence predicate; it is orthogonal to it.

6. My confirmation that sections 1-8 are outside the RN5 affected scope covers the exponent structure of the argument, which I checked and which is correct, and does NOT cover the implementation of the fourth determinant moments inside rn_joint.py, which I never saw. It also does not extend to section 9, where I found the opposite.

7. Nothing here composes the 2D upper and lower tracks with the 3D lifetime track; the object is a 2D far-zone intensity bound at one rung and must not be joined to the 3D result. No original prize problem is solved by this object or by this review; the count of original prize problems solved remains zero. Aging changes none of this: RV-RN3 is a new R17 request at age 0, and neither its age nor this record approves anything.

## 12. Note on this file's name

This record is `REV-RN3-FARZONE-20260918.{json,md}`, not `RV-RN3*`. The task assigned
me `reviews/records/RV-RN3*.json` and `.md`, but `reviews/review_record.schema.json`
requires `review_id` to match `^REV-[A-Z0-9][A-Z0-9._-]{2,60}$` and
`tools/reviews_check.py` requires the filename stem to equal `review_id`, so **no
filename satisfies both**. I followed the schema and the checker — the instruction was
to conform to them exactly and to fix the record rather than the checker — and I am
flagging the conflict rather than resolving it silently. I wrote only these two files
and touched nothing else in `reviews/`.

## 13. Reproducing this review

```bash
python3 tools/drive_index.py sha 0c9446b7
python3 tools/drive_index.py find RN3
# download 1f0EBsq6UIZ0jDCABT251TTNEWYCVA0FO via the Drive connector, base64-decode,
# then: sha256 -> 0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373, 12956 bytes
python3 tools/reviews_check.py
```

All three verification scripts are reproduced in full below. Python 3.11, standard
library only, no network, no input beyond the constants quoted from the object.

**One warning to a replayer.** `verify_rn3.py` is the *first* pass and it prints
`**FAIL**` on five lines that are **my instrument, not the object**. Specifically:
its `I_far` and headline assertions compare against the *displayed, outward-rounded*
ratios `1.12058` / `0.88021` instead of the unrounded certificate factors, so they
report a 1e-6 disagreement that is exactly the outward rounding the object declares —
`verify2.py` resolves this by recovering the implied unrounded ratios `1.120578860`
and `0.880218550`; its trace-remainder check runs in double precision, where the true
remainder (`1e-32`) is far below the noise floor (`1e-17`), so its "101 violations"
are meaningless — `verify3.py` redoes it at 80 digits and the bound holds; and its
box count of 13,612 against the object's 13,604 is the one line that is a real finding,
resolved in `verify2.py`'s boundary survey. The scripts are given unedited, failures
and all, because a review that silently retouches its own first pass is not a review.

### `verify_rn3.py` — reconstruction and positive controls

```python
"""Independent reconstruction of RN-JOINT-001 (RV-RN3), stdlib only.

Reviewer-side re-derivation and negative controls. Verifies nothing about the
author's code, which was not obtained; every number below is recomputed from
the proof body's own formulas.
"""
import math, itertools, random
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60
OK = lambda c: "PASS" if c else "**FAIL**"
out = []
def rep(name, ok, detail=""):
    out.append((name, ok, detail)); print(f"[{OK(ok)}] {name}  {detail}")

# ---------------------------------------------------------------- 1. box cover
# "every quarter-unit Cartesian box intersecting the closed annulus", membership
# by exact integer inequalities against 25 and 289.
def count_boxes(r_in2=25, r_out2=289, step_den=4):
    lo = int(math.floor(-math.sqrt(r_out2)*step_den))-2
    hi = int(math.ceil(math.sqrt(r_out2)*step_den))+2
    R2IN = r_in2*step_den*step_den      # exact integer comparison in quarter units
    R2OUT = r_out2*step_den*step_den
    n = 0
    for i in range(lo, hi):
        a, b = i, i+1
        if a < 0 < b: xmin = 0
        else: xmin = min(abs(a), abs(b))
        xmax = max(abs(a), abs(b))
        for j in range(lo, hi):
            c, d = j, j+1
            if c < 0 < d: ymin = 0
            else: ymin = min(abs(c), abs(d))
            ymax = max(abs(c), abs(d))
            mn = xmin*xmin + ymin*ymin
            mx = xmax*xmax + ymax*ymax
            if mn <= R2OUT and mx >= R2IN:
                n += 1
    return n
n_boxes = count_boxes()
rep("box cover: quarter-unit boxes meeting 5<=|y|<=17", n_boxes == 13604,
    f"counted {n_boxes}, object claims 13,604")

# ---------------------------------------------------------------- 2. constants
r = F(1,20); b = F(6,5)
r3 = r**3
win = r3/6                     # mark window width  [b-r^3/6, b]

# a2 = -k''(0) for the side-24 periodized Gaussian, exactly via image sums.
def k_and_derivs(L=24, jmax=6):
    num = sum(math.exp(-(L*j)**2/2) for j in range(-jmax, jmax+1))
    d2  = sum(((L*j)**2 - 1)*math.exp(-(L*j)**2/2) for j in range(-jmax, jmax+1))
    d4  = sum(((L*j)**4 - 6*(L*j)**2 + 3)*math.exp(-(L*j)**2/2) for j in range(-jmax, jmax+1))
    return num, d2/num, d4/num
_, k2, k4 = k_and_derivs()
a2 = -k2
rep("a2 = -k''(0) equals 1 to double precision (planar limit)", abs(a2-1) < 1e-15,
    f"a2-1 = {a2-1:.3e}; k''''(0) = {k4:.15f} (planar value 3)")

# ---------------------------------------------------------------- 3. J_ref
# J_ref = [erf(sqrt3 b/2) - erf(sqrt3 (b-r^3/6)/2)] / (2 pi sqrt3 a2)
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
def dexp(x): return Decimal(x).exp()
def erf_gap(x_hi: Decimal, h: Decimal) -> Decimal:
    """(2/sqrt(pi)) * int_{x_hi-h}^{x_hi} e^{-t^2} dt, Taylor about the midpoint."""
    m = x_hi - h/2
    f  = dexp(-m*m)
    f2 = (4*m*m - 2)*f
    f4 = (16*m**4 - 48*m*m + 12)*f
    f6 = (64*m**6 - 480*m**4 + 720*m*m - 120)*f
    integral = h*f + f2*h**3/24 + f4*h**5/1920 + f6*h**7/322560
    return 2/PI.sqrt()*integral
sqrt3 = Decimal(3).sqrt()
x_hi = sqrt3*Decimal(b.numerator)/Decimal(b.denominator)/2
h    = sqrt3*Decimal(win.numerator)/Decimal(win.denominator)/2
J_ref = erf_gap(x_hi, h)/(2*PI*sqrt3*Decimal(1))     # a2 = 1 - O(1e-122)
claim_J = Decimal("6.3529298594549898874E-7")
rel = abs(J_ref-claim_J)/claim_J
rep("mark integral J_ref reproduced to 19 significant digits", rel < Decimal("1e-19"),
    f"mine {J_ref:.20E} vs object {claim_J:.20E}, rel diff {rel:.2E}")

# ---------------------------------------------------------------- 4. far count
area = 576 - 25*PI
lo_ratio, hi_ratio = Decimal("0.88021"), Decimal("1.12058")
I_lo, I_hi = lo_ratio*area*J_ref, hi_ratio*area*J_ref
obj_lo = Decimal("0.00027817807867916"); obj_hi = Decimal("0.00035413986017516")
rep("I_far upper endpoint from ratio x area x J_ref",
    abs(I_hi-obj_hi)/obj_hi < Decimal("1e-6"), f"mine {I_hi:.17E} vs object {obj_hi:.17E}")
rep("I_far lower endpoint from ratio x area x J_ref",
    abs(I_lo-obj_lo)/obj_lo < Decimal("1e-6"), f"mine {I_lo:.17E} vs object {obj_lo:.17E}")
R3 = Decimal(r3.numerator)/Decimal(r3.denominator)
rep("headline 2.83312 r^3 is the outward rounding of I_hi",
    I_hi/R3 < Decimal("2.83312"), f"I_hi/r^3 = {I_hi/R3:.9f}")
rep("headline 2.22542 r^3 is the inward rounding of I_lo",
    I_lo/R3 > Decimal("2.22542"), f"I_lo/r^3 = {I_lo/R3:.9f}")

# ---------------------------------------------------------- 5. ratio-table self-consistency
eta = Decimal("0.116819988"); RY_hi = Decimal("1.003365693")
rep("table: (1+eta) * Y-density upper reproduces 1.12058",
    abs((1+eta)*RY_hi - hi_ratio) < Decimal("1e-6"),
    f"(1+eta)*R_Y,hi = {(1+eta)*RY_hi:.9f}")
RY_lo_implied = lo_ratio/(1-eta)
rep("table: implied R_Y,lo is a plausible two-sided partner of R_Y,hi",
    Decimal("0.99") < RY_lo_implied < 1, f"implied R_Y,lo = {RY_lo_implied:.9f} "
    f"(object does not tabulate the lower Y-density factor)")

# ---------------------------------------------------------- 6. chi-square lemma (1)
def chi2_exact(m, E):
    """log(1+chi^2(N(m,I+E) || N(0,I))) by the object's closed form, n=dim."""
    n = len(m)
    # eigen-decomposition of symmetric E by Jacobi (stdlib only)
    A = [row[:] for row in E]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        p = q = 0; off = 0.0
        for i in range(n):
            for j in range(i+1, n):
                off += A[i][j]**2
                if abs(A[i][j]) > abs(A[p][q]) or (p == q): p, q = i, j
        if off < 1e-30: break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        th = 0.5*math.atan2(2*apq, aqq-app); c, s = math.cos(th), math.sin(th)
        for k in range(n):
            akp, akq = A[k][p], A[k][q]
            A[k][p] = c*akp - s*akq; A[k][q] = s*akp + c*akq
        for k in range(n):
            apk, aqk = A[p][k], A[q][k]
            A[p][k] = c*apk - s*aqk; A[q][k] = s*apk + c*aqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c*vkp - s*vkq; V[k][q] = s*vkp + c*vkq
    ev = [A[i][i] for i in range(n)]
    if max(abs(e) for e in ev) >= 1: return None, ev
    logdet = sum(math.log(1-e*e) for e in ev)
    mm = [sum(V[k][i]*m[k] for k in range(n)) for i in range(n)]
    quad = sum(mm[i]**2/(1-ev[i]) for i in range(n))
    return -0.5*logdet + quad, ev

def chi2_numeric_1d(mu, e, N=400000, L=14.0):
    """1-D sanity: int p^2/q by Simpson."""
    hstep = 2*L/N; tot = 0.0
    var = 1+e
    for i in range(N+1):
        x = -L + i*hstep
        p = math.exp(-(x-mu)**2/(2*var))/math.sqrt(2*math.pi*var)
        q = math.exp(-x*x/2)/math.sqrt(2*math.pi)
        w = 1 if i in (0, N) else (4 if i % 2 else 2)
        tot += w*p*p/q
    return math.log(tot*hstep/3)

random.seed(20260918)
worst = 0.0; bound_ok = True
for _ in range(200):
    n = 4
    Braw = [[random.uniform(-0.09, 0.09) for _ in range(n)] for _ in range(n)]
    E = [[(Braw[i][j]+Braw[j][i])/2 for j in range(n)] for i in range(n)]
    m = [random.uniform(-0.05, 0.05) for _ in range(n)]
    val, ev = chi2_exact(m, E)
    delta = math.sqrt(sum(E[i][j]**2 for i in range(n) for j in range(n)))
    bound = delta**2/(2*(1-delta**2)) + sum(x*x for x in m)/(1-delta)
    if val is None or val > bound + 1e-12: bound_ok = False
    worst = max(worst, val/bound)
rep("lemma (1) bound holds on 200 random symmetric (m,E), delta<1", bound_ok,
    f"worst ratio LHS/RHS = {worst:.4f}")
v1 = chi2_exact([0.3], [[0.2]])[0]; v2 = chi2_numeric_1d(0.3, 0.2)
rep("lemma (1) closed form matches direct integral of p^2/q (1-D)",
    abs(v1-v2) < 1e-9, f"closed form {v1:.12f} vs quadrature {v2:.12f}")

# ---------------------------------------------------- 7. typed second moment (2)
def gauss_quad(f, lo, hi, N=200000):
    hstep = (hi-lo)/N; tot = 0.0
    for i in range(N+1):
        x = lo + i*hstep
        w = 1 if i in (0, N) else (4 if i % 2 else 2)
        tot += w*f(x)
    return tot*hstep/3
v = 1.2
# E_Q g with g=(T-U^2)_+, T~chi2(2), U~N(-v,1): integrate over u using E(T-c)_+=2e^{-c/2}
Eg  = gauss_quad(lambda u: 2*math.exp(-u*u/2)*math.exp(-(u+v)**2/2)/math.sqrt(2*math.pi), -12, 12)
Eg2 = gauss_quad(lambda u: 8*math.exp(-u*u/2)*math.exp(-(u+v)**2/2)/math.sqrt(2*math.pi), -12, 12)
m0 = math.sqrt(2)*math.exp(-v*v/4)
rep("m_0(v) = sqrt2 e^{-v^2/4} reproduced by quadrature", abs(Eg-m0) < 1e-12,
    f"quadrature {Eg:.15f} vs formula {m0:.15f}")
rep("E_Q g^2 = 4 m_0(v) reproduced by quadrature", abs(Eg2-4*m0) < 1e-12,
    f"quadrature {Eg2:.15f} vs 4*m_0 {4*m0:.15f}")
# E(T-c)_+ and its square, directly
for c in (0.0, 0.7, 3.1):
    e1 = gauss_quad(lambda t, c=c: (t-c)*0.5*math.exp(-t/2) if t > c else 0.0, 0, 80)
    e2 = gauss_quad(lambda t, c=c: (t-c)**2*0.5*math.exp(-t/2) if t > c else 0.0, 0, 80)
    rep(f"E(T-c)_+ = 2e^-c/2 and E(T-c)_+^2 = 8e^-c/2 at c={c}",
        abs(e1-2*math.exp(-c/2)) < 1e-7 and abs(e2-8*math.exp(-c/2)) < 1e-6,
        f"{e1:.9f} vs {2*math.exp(-c/2):.9f} ; {e2:.9f} vs {8*math.exp(-c/2):.9f}")

# ------------------------------------------- 8. density-ratio lower bound (7) identity
# claim: (b^2 B + 2bu + u^2)/(1-B) == 2bu + u^2 + B(b+u)^2/(1-B) for all B<1
ok = True
for _ in range(500):
    bb = F(random.randint(1, 50), 10); uu = F(random.randint(0, 40), 100); BB = F(random.randint(1, 90), 100)
    lhs = (bb*bb*BB + 2*bb*uu + uu*uu)/(1-BB)
    rhs = 2*bb*uu + uu*uu + BB*(bb+uu)**2/(1-BB)
    ok &= (lhs == rhs)
rep("(7) is exactly the assembled Cauchy bound, not a loosened one", ok,
    "exact rational identity on 500 random (b,u,beta)")

# ------------------------------------------- 9. trace-series remainder (section 8)
# -1/2 log det(I-E^2) = 1/2 sum_k tr(E^{2k})/k ; remainder after 10 terms
def remainder_check(delta, n=6, trials=200):
    bad = 0
    for _ in range(trials):
        Braw = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        E = [[(Braw[i][j]+Braw[j][i])/2 for j in range(n)] for i in range(n)]
        fro = math.sqrt(sum(E[i][j]**2 for i in range(n) for j in range(n)))
        E = [[E[i][j]*delta/fro for j in range(n)] for i in range(n)]
        _, ev = chi2_exact([0.0]*n, E)
        exact = -0.5*sum(math.log(1-e*e) for e in ev)
        ten = 0.5*sum(sum(e**(2*k) for e in ev)/k for k in range(1, 11))
        if exact - ten > delta**22/(22*(1-delta**2)) + 1e-18: bad += 1
    return bad
bad = remainder_check(0.0436)
rep("trace-series remainder delta^22/[22(1-delta^2)] is valid at the certified delta",
    bad == 0, f"{bad} violations in 200 random E with ||E||_F = 0.0436")
bad_big = remainder_check(0.97)
rep("same remainder still valid near delta -> 1", bad_big == 0,
    f"{bad_big} violations in 200 random E with ||E||_F = 0.97")

print()
print("checks:", sum(1 for _, ok, _ in out if ok), "passed,",
      sum(1 for _, ok, _ in out if not ok), "failed")
```

### `verify2.py` — boundary forensics, table rebuild, first controls

```python
import math
from decimal import Decimal, getcontext
getcontext().prec = 80

# ---- box-cover boundary forensics -----------------------------------------
def survey(step_den=4, r_in2=25, r_out2=289):
    R2IN, R2OUT = r_in2*step_den**2, r_out2*step_den**2
    lim = int(math.ceil(math.sqrt(r_out2)*step_den))+2
    closed=0; outer_tangent=0; inner_tangent=0
    for i in range(-lim, lim):
        a,b = i,i+1
        xmin = 0 if a<0<b else min(abs(a),abs(b)); xmax = max(abs(a),abs(b))
        for j in range(-lim, lim):
            c,d = j,j+1
            ymin = 0 if c<0<d else min(abs(c),abs(d)); ymax = max(abs(c),abs(d))
            mn = xmin*xmin+ymin*ymin; mx = xmax*xmax+ymax*ymax
            if mn <= R2OUT and mx >= R2IN:
                closed += 1
                if mn == R2OUT: outer_tangent += 1
                if mx == R2IN:  inner_tangent += 1
    return closed, outer_tangent, inner_tangent
closed, ot, it = survey()
print(f"closed-annulus cover        : {closed}")
print(f"  boxes touching |y|=17 only: {ot}   -> strict outer rule gives {closed-ot}")
print(f"  boxes touching |y|=5 only : {it}   -> strict inner rule gives {closed-it}")
print(f"  both dropped              : {closed-ot-it}")
print(f"object claims               : 13604")
print()

# ---- exact reconciliation of the displayed table with the unrounded values --
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
sqrt3 = Decimal(3).sqrt()
def dexp(x): return Decimal(x).exp()
def erf_gap(x_hi, h):
    m = x_hi - h/2
    f = dexp(-m*m)
    f2=(4*m*m-2)*f; f4=(16*m**4-48*m*m+12)*f; f6=(64*m**6-480*m**4+720*m*m-120)*f
    return 2/PI.sqrt()*(h*f + f2*h**3/24 + f4*h**5/1920 + f6*h**7/322560)
b = Decimal("1.2"); r3 = Decimal("0.000125"); win = r3/6
J = erf_gap(sqrt3*b/2, sqrt3*win/2)/(2*PI*sqrt3)
area = 576 - 25*PI
base = area*J
objhi = Decimal("0.00035413986017516"); objlo = Decimal("0.00027817807867916")
print(f"J_ref (mine)                : {J:.22E}   object 6.3529298594549898874E-7")
print(f"area 576-25pi               : {area:.15f}")
print(f"unrounded upper ratio implied by object's I_far : {objhi/base:.12f}  (displayed 1.12058)")
print(f"unrounded lower ratio implied by object's I_far : {objlo/base:.12f}  (displayed 0.88021)")
print(f"object I_far/r^3 upper       : {objhi/r3:.9f}  -> displayed 2.83312 (outward)")
print(f"object I_far/r^3 lower       : {objlo/r3:.9f}  -> displayed 2.22542 (outward)")
print()

# ---- eta row rebuilt from the table's own inputs ---------------------------
h2   = Decimal("0.014009957")
chi2 = Decimal("0.001032557")
Zlo  = Decimal("0.0077592917375327855")
m0min = Decimal(2).sqrt()*dexp(-b*b/4)
eta = 2*h2*chi2.sqrt()/(Zlo*m0min.sqrt())
print(f"m_0,min = sqrt2 e^(-b^2/4)   : {m0min:.15f}")
print(f"eta rebuilt from h2, chi2, Z_lo, m_0,min : {eta:.12f}   (table: 0.116819988)")
RYhi = Decimal("1.003365693")
print(f"(1+eta)*R_Y,hi               : {(1+eta)*RYhi:.12f}   (table: 1.12058 outward)")
print()

# ---- NEGATIVE CONTROLS ------------------------------------------------------
print("=== negative controls ===")
# NC1 RN5-shaped exponent substitution: use sqrt(E g^4) where E g^2 belongs.
m0 = m0min
Eg2 = 4*m0                       # exact, object's (2)
Eg4 = 192*m0                     # E(T-c)_+^4 = e^{-c/2} E T^4, E T^4 = 4! 2^4 = 384
fac_correct = Eg2.sqrt()
fac_defect  = Eg4.sqrt()
eta_def = eta*fac_defect/fac_correct
print(f"NC1 RN5 exponent swap: eta {eta:.6f} -> {eta_def:.6f}; "
      f"budget 0.6798 {'EXCEEDED -> control FIRES' if eta_def > Decimal('0.6798') else 'still met'}")

# NC2 inflate the imported H3 normalizer floor by 10% (unsafe direction)
eta_infl = eta/Decimal("1.10")
print(f"NC2 inflate Z_lo by 10%: eta {eta:.6f} -> {eta_infl:.6f} (understates); "
      f"ratio bound 1.12058 -> {(1+eta_infl)*RYhi:.6f}  -> control FIRES (changes the headline)")

# NC3 swap min for max over the mark window in m_0,min (flip the inequality)
m0max = Decimal(2).sqrt()*dexp(-(b-win)**2/4)
eta_swap = 2*h2*chi2.sqrt()/(Zlo*m0max.sqrt())
print(f"NC3 m_0 min->max over mark window: eta {eta:.12f} -> {eta_swap:.12f}; "
      f"headline {(1+eta)*RYhi:.9f} -> {(1+eta_swap)*RYhi:.9f}  -> displayed 1.12058 unchanged: control DOES NOT FIRE")

# NC4 drop the delta<1 hypothesis of lemma (1)
def chi2_1d(mu, e, N=2000000, L=40.0):
    h=2*L/N; tot=0.0; var=1+e
    for i in range(N+1):
        x=-L+i*h
        p=math.exp(-(x-mu)**2/(2*var))/math.sqrt(2*math.pi*var)
        q=math.exp(-x*x/2)/math.sqrt(2*math.pi)
        w=1 if i in (0,N) else (4 if i%2 else 2)
        tot+=w*p*p/q
    return tot*h/3
for e in (0.2, 0.9, 1.0, 1.2):
    try:
        v=chi2_1d(0.0,e)
        cf = -0.5*math.log(1-e*e) if abs(e)<1 else float('nan')
        print(f"NC4 delta={e}: int p^2/q = {v:.6g}  closed form exp = {math.exp(cf) if cf==cf else float('nan'):.6g}"
              f"   {'divergent -> control FIRES' if not (e<1) else 'finite'}")
    except (ValueError, OverflowError) as exc:
        print(f"NC4 delta={e}: {type(exc).__name__}: {exc} -> control FIRES")

# NC5 trace-series remainder, exact at the certified delta (Decimal, 80 digits)
d = Decimal("0.0436")
exact = -( (1-d*d).ln() )/2                      # scalar worst case: one eigenvalue = delta
ten   = sum(d**(2*k)/k for k in range(1,11))/2
rem   = exact-ten
claim = d**22/(22*(1-d*d))
print(f"NC5 remainder after 10 trace terms: true {rem:.6E} <= claimed {claim:.6E} : "
      f"{'holds (control does not fire)' if rem<=claim else 'VIOLATED -> control FIRES'}")
d = Decimal("0.97")
exact = -((1-d*d).ln())/2; ten = sum(d**(2*k)/k for k in range(1,11))/2
print(f"    same at delta=0.97: true {(exact-ten):.6E} vs claimed {d**22/(22*(1-d*d)):.6E} : "
      f"{'holds' if (exact-ten)<=d**22/(22*(1-d*d)) else 'VIOLATED'}")

# NC6 perturb the certified ||E||_F upward by 1% and see if chi^2 row survives
EF = Decimal("0.043544631"); mnorm = Decimal("0.008864422")
chi_bound = lambda dd, mm: (dd*dd/(2*(1-dd*dd)) + mm*mm/(1-dd)).exp()-1
print(f"NC6 lemma(1) RHS at the certified ||E||_F,||m||: chi^2 <= {chi_bound(EF,mnorm):.9E}"
      f"  (table row: 1.032557E-3)  -> {'consistent' if chi_bound(EF,mnorm) <= Decimal('0.001032557')*Decimal('1.000001') else 'INCONSISTENT -> FIRES'}")
print(f"    with ||E||_F inflated 1%: {chi_bound(EF*Decimal('1.01'),mnorm):.9E}  -> exceeds the tabulated chi^2 row: control FIRES")

# NC7 exhaustiveness of the zone decomposition
print(f"NC7 zones: far |y|>=5 ; near 0.1<=|y|<=5 ; union misses the open disc |y|<0.1, "
      f"area {float(math.pi*0.01):.6f} of the 576 square -> control FIRES (named gap)")
print(f"    square corner radius 12*sqrt2 = {12*math.sqrt(2):.6f} < 17, so the cover contains the whole far square")
```

### `verify3.py` — remaining negative controls at 80 digits

```python
import math
from decimal import Decimal, getcontext
getcontext().prec = 80
b=Decimal("1.2"); r3=Decimal("0.000125"); win=r3/6
h2=Decimal("0.014009957"); chi2=Decimal("0.001032557")
Zlo=Decimal("0.0077592917375327855")
m0min=Decimal(2).sqrt()*Decimal(-b*b/4).exp()
eta=2*h2*chi2.sqrt()/(Zlo*m0min.sqrt()); RYhi=Decimal("1.003365693")

print("=== NC4: remove the delta<1 hypothesis of lemma (1) ===")
def partial(mu,e,L,N=400000):
    hstep=2*L/N; tot=0.0; var=1+e
    for i in range(N+1):
        x=-L+i*hstep
        lp=-(x-mu)**2/(2*var)-0.5*math.log(2*math.pi*var)
        lq=-x*x/2-0.5*math.log(2*math.pi)
        w=1 if i in (0,N) else (4 if i%2 else 2)
        tot+=w*math.exp(2*lp-lq)
    return tot*hstep/3
for e in (0.2,0.9,1.0,1.2):
    A=(1-e)/(1+e)
    vals=[partial(0.0,e,L) for L in (6.0,12.0,24.0)]
    closed=math.exp(-0.5*math.log(1-e*e)) if abs(e)<1 else float('nan')
    print(f"  delta={e}: quadratic precision (1-delta)/(1+delta) = {A:+.4f}; "
          f"partial integrals over |x|<=6,12,24 = {vals[0]:.6g}, {vals[1]:.6g}, {vals[2]:.6g}; "
          f"closed form {closed if closed==closed else 'undefined'}"
          f" -> {'FINITE, matches' if abs(e)<1 else 'DIVERGENT: control FIRES'}")

print()
print("=== NC5: trace-series remainder after ten terms (exact, 80 digits) ===")
for ds in ("0.0436","0.5","0.97"):
    d=Decimal(ds)
    exact=-((1-d*d).ln())/2
    ten=sum(d**(2*k)/k for k in range(1,11))/2
    claim=d**22/(22*(1-d*d))
    print(f"  delta={ds}: true remainder {exact-ten:.6E}  claimed bound {claim:.6E}  "
          f"{'holds (does not fire)' if exact-ten<=claim else 'VIOLATED: FIRES'}")

print()
print("=== NC6: perturb the certified ||E||_F and ||m|| ===")
EF=Decimal("0.043544631"); mn=Decimal("0.008864422"); tab=Decimal("0.001032557")
cb=lambda dd,mm:(dd*dd/(2*(1-dd*dd))+mm*mm/(1-dd)).exp()-1
print(f"  lemma(1) RHS at the tabulated ||E||_F,||m||: chi^2 <= {cb(EF,mn):.6E}  (table row {tab:.6E})"
      f" -> {'consistent, control does not fire' if cb(EF,mn)<=tab else 'EXCEEDS the table row: FIRES'}")
for f in ("1.001","1.01","1.10"):
    print(f"  ||E||_F x {f}: chi^2 bound {cb(EF*Decimal(f),mn):.6E} -> "
          f"{'still under table row' if cb(EF*Decimal(f),mn)<=tab else 'exceeds table row: FIRES'}")

print()
print("=== NC7: perturb the headline ratio constant ===")
J=Decimal("6.3529298594549898874E-7"); PI=Decimal(str(math.pi))
area=576-25*Decimal("3.14159265358979323846264338327950288")
base=area*J
for c in ("1.12058","1.12000","1.10000"):
    I=Decimal(c)*base
    print(f"  ratio {c}: I_far/r^3 = {I/r3:.9f}  "
          f"{'covers the object value 2.833118881' if I/r3>=Decimal('2.833118881') else 'FAILS to cover it: FIRES'}")

print()
print("=== NC8: RN5 exponent swap in the Cauchy-Schwarz step ===")
Eg2=4*m0min; Eg4=192*m0min
eta_def=eta*Eg4.sqrt()/Eg2.sqrt()
print(f"  correct second moment  E_Q g^2 = 4 m_0      -> eta = {eta:.9f}")
print(f"  RN5-shaped fourth moment sqrt(E_Q g^4)=sqrt(192 m_0) -> eta = {eta_def:.9f}")
print(f"  budget 0.6798: {'EXCEEDED -> the certificate depends on the exponent: FIRES' if eta_def>Decimal('0.6798') else 'still met'}")
print(f"  and the far ratio would read {(1+eta_def)*RYhi:.6f} instead of 1.12058")
```

---

*R17 §5: age never becomes approval. RV-RN3 is a new request at age 0; neither its age
nor this record approves anything. Original prize problems solved: 0.*
