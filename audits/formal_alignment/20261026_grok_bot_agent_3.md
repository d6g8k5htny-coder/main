# SIDE24 arithmetic skeleton — independent statement-alignment review

- Reviewer: provider **xAI**, family **Grok Bot**, agent **Grok Bot agent 3** (scientific lane; assigned by CoS on main#388 comment 6405342768).
- Author under review: Anthropic / Claude / Cursor cloud agent, 2026-09-27. All three differ.
- Reviewed commit: `3b39c42d6edcc98ae165270be346f0e31c57f81d` (d6g8k5htny-coder/main)
- `formal/manifest.json` SHA-256: `8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090` (recomputed, matches pin)
- `formal/SCOPE.md` SHA-256: `fd06700628b1fbc81c5cfe3786aaa455032375bcb879914f068a66b726802a17` (recomputed; equals manifest `files` entry and pin)
- All 17 `manifest.files` digests recomputed at the reviewed commit: all match (including `sources/side24_v1/PROOF.md` c06dacc… and `ENCLOSURE.json` 72b6cd9…).
- Lane: statement correspondence only. Proof correctness (kernel) and Layer 0 mathematics are **not** graded.

## Source exposure

Statements and docstrings read first, together with the informal anchors in `sources/side24_v1/`. All proof bodies in these modules are the single tactic `by decide`, visible in the same file; they were seen but carry no information and were not used for any verdict. Docstrings were read alongside statements (docstrings were author-written and could bias; every cross-multiplication was re-derived independently with exact rational arithmetic, e.g. image constant, Taylor partial sum ≈10.01416, pairing sums 1,1,2,4,10,26,76, shifted determinant 7/9).

## Gate output

No `SOURCE_IDENTITY_PASS` receipt or `--run-lean` workflow output for this package is committed at the reviewed commit (only prose mentions in `formal/README.md`, `formal/REVIEW_LANE.md`, docs). No committed CI receipt found (absent). This reviewer ran the existing gate locally on an export of the reviewed commit (no Lean build): `python3 tools/formal_gate_check.py --alignment <record>` printed `SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): 8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090` on a structure-diagnostic copy; this REJECTED record itself is refused with `FORMAL_GATE_FAIL: alignment not accepted`, the intended outcome per REVIEW_LANE.md. No `--run-lean` workflow run is cited.

## Per-target verdicts

Tally: ALIGNED 8 · ALIGNED AT NARROWER SCOPE 21 · MISALIGNED 2 (total 31).

"Narrower" throughout means: the Lean target is only the arithmetic identity/inequality behind the anchor; the analytic step named in `does_not_claim` and SCOPE "Not established" is not claimed. In every case that omission is named in the manifest/SCOPE; no target claims more than its anchor.

| # | Target | Module | Verdict | Notes |
|---|---|---|---|---|
| 1 | `UniversalLaw.Side24.pairing_sum_six` | Ledger.lean | **ALIGNED** | pairingSum q = Σ_{j≤⌊q/2⌋} q!/(2^j j!(q−2j)!) is exactly the displayed pairing coefficient sum (the ∣x∣-weights evaluated at ∣x∣=1); Nat division exact (counts). Value 76 matches anchor 1+15+45+15. Derivative bound itself correctly excluded (does_not_claim). |
| 2 | `UniversalLaw.Side24.pairing_sum_le_six` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Lean: ∀q∈{0..6}, pairingSum q ≤ 76 (values 1,1,2,4,10,26,76 recomputed). Anchor 'bounds the smaller orders' is the coefficient-sum dominance; narrower: the ∣x∣≥1 analytic bound 76∣x∣^6e^{-∣x∣²/2} for q<6 (needs ∣x∣^{q-2j}≤∣x∣^6) is not claimed; named in SCOPE 'Not established'. |
| 3 | `UniversalLaw.Side24.pairing_terms_six` | Ledger.lean | **ALIGNED** | Four terms j=0..3 at q=6: 720/720=1, 720/48=15, 720/16=45, 720/48=15; matches anchor display order exactly. |
| 4 | `UniversalLaw.Side24.sixth_moment_double_factorial` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Lean 1·3·5=15 (5!!). Anchor is the bound 'contraction at zero ≤15'; only the numeric value 15 is stated, the bound is excluded by does_not_claim/SCOPE. |
| 5 | `UniversalLaw.Side24.lattice_shell_constant` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 27j³ count × 27j⁶ ⇒ 729 j⁹ ⇒ 27·27=729; geometric factor 2 ⇒ 729·2=1458. Constants match; counting and tail sum excluded (recorded). |
| 6 | `UniversalLaw.Side24.geometric_ratio_constants` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 512=2^9 = worst (j+1)^9/j^9 at j=1; 864 = 288((j+1)²−j²) at j=1. Constants match anchor 512e^(−864); '<1/2' explicitly not claimed. |
| 7 | `UniversalLaw.Side24.period_exponent` | Ledger.lean | **ALIGNED** | 24²/2=288 (Nat division exact, 576/2); matches e^(−288∣n∣²) for period 24 under exp(−∣z∣²/2). |
| 8 | `UniversalLaw.Side24.image_constant_value` | Ledger.lean | **ALIGNED** | Recomputed 1458·(76·24^6+15)=21175738586478; same constants as anchor E·10^125. Inequality (2) excluded. |
| 9 | `UniversalLaw.Side24.exp_taylor_partial_sum_gt_ten` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | Σ_{k≤20}(288/125)^k/k! >10 cross-multiplied by 125^20·20! gives Σ288^k125^{20−k}(20!/k!) > 10·125^20·20!; 20!/k! exact. Recomputed partial sum ≈10.01415908 >10 (tight). Step partial sum < exp excluded (does_not_claim). |
| 10 | `UniversalLaw.Side24.covariance_relative_bound` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 60E<10^(−108) with E=C·10^(−125) ⇔ 60C<10^17 (multiply by 10^125): correct direction and scaling; 2·10·3=60 matches 2E entry → 20E (dim 10) → ×3 from C_ref≥I/3. Spectral/PSD steps excluded. |
| 11 | `UniversalLaw.Side24.odd_block_eigenvalues` | Ledger.lean | **ALIGNED** | Eigenvalues 8±√58 of [[1,−3],[−3,15]] ⇔ trace 16=2·8 and det 15−9=6=8²−58; Int-typed, signs correct. Statement fully determines the eigenvalue pair. |
| 12 | `UniversalLaw.Side24.odd_block_shifted_minors` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | M−I/3 has minor 2/3, det (2/3)(44/3)−9=7/9; scaled 3M−I=[[2,−9],[−9,44]] minors 2 and 88−81=7 (=9·7/9). Off-diagonal −9=3·(−3) appears only as a literal, not derived; Sylvester excluded. |
| 13 | `UniversalLaw.Side24.smaller_eigenvalue_exceeds_third` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 8−√58>1/3 ⇔ 23/3>√58 ⇔ 529>522 ⇔ 58·9<23²; direction correct. √ monotonicity excluded. |
| 14 | `UniversalLaw.Side24.exponent_bounds` | Ledger.lean | **ALIGNED** | a+2b<14 ⇔ 6a+12b<84; 2a+b<13 ⇔ 12a+6b<78; quantified over d∈{2,3} exactly as anchor. d=3 second is 77<78 (tight). |
| 15 | `UniversalLaw.Side24.exponent_values` | Ledger.lean | **ALIGNED** | m=d−1, n=m(m+1)/2=(d−1)d/2; 6a=6m+4+3n, 6b=6d+3n. d=2: a=13/6,b=5/2 (13,15); d=3: a=25/6,b=9/2 (25,27). Matches anchor definitions. |
| 16 | `UniversalLaw.Side24.ratio_constant_ordering` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 13≤32, 28≤32 match '1−13ε…1+28ε, hence 1−32ε…1+32ε'; only the constant ordering, log/exp bounds excluded. |
| 17 | `UniversalLaw.Side24.density_comparison_below_reported` | Ledger.lean | **ALIGNED AT NARROWER SCOPE** | 32·10^(−108)<10^(−106) ×10^214 ⇔ 32·10^106<10^108; correct. Uses ε and factor 32 from §3–4; the density comparison itself excluded. |
| 18 | `UniversalLaw.Side24.conditional_third_derivative_variance` | ConeMoments.lean | **ALIGNED** | τ²=Var(t)−Cov(t,G1)²/Var(G1)=15−9/1=6 over Int; matches anchor Var(t)=15, Cov=(−3,0,…) and §1 τ²=6. Conditioning formula excluded. |
| 19 | `UniversalLaw.Side24.transverse_covariance_entries_thirds` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | In thirds: Var A_ii 2+3+3=8 (8/3), Var A_12 0+3+0=3 (1), Cov(A_11,A_22) 2+0+0=2 (2/3); delta evaluations checked by hand. Lean is literal sums; the delta evaluation is in docstring only. |
| 20 | `UniversalLaw.Side24.cone_moment_m1_thirds` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | D_1=½·8/3=4/3 ⇔ 8=2·4 in thirds. Symmetry step excluded. |
| 21 | `UniversalLaw.Side24.trace_and_traceless_variances` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Var s=(8/3+8/3+2·2/3)/4=5/3 (8+8+4=20, 20·3=5·12); Var x=(8+8−4)/12=1; Cov(s,x)=(8−8)/4=0. Var y=Var A_12=1 is not in this statement (covered by transverse_covariance_entries_thirds); independence excluded. |
| 22 | `UniversalLaw.Side24.fourth_moment_of_trace` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | E s⁴=3(5/3)²=75/9=25/3 ⇔ 3·5²·3=25·9 (225=225). Fourth-moment identity excluded. |
| 23 | `UniversalLaw.Side24.exponential_moment_of_trace` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | (1+5/3)^(−1/2)=(8/3)^(−1/2)=√(3/8); Lean 3+5=8 in thirds. Exponential-moment identity and the root excluded. |
| 24 | `UniversalLaw.Side24.cone_moment_m2_algebra` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | 4E s²=20/3 (4·5=20 thirds); 25−20+24=29 thirds ⇒ ½·29/3=29/6 (halving not separately stated, trivial); 4√(3/8)=√6 ⇔ 16·3=6·8. Integral/Rayleigh excluded. |
| 25 | `UniversalLaw.Side24.cube_root_simplification` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | Cubing: 6²/24=3/2 ⇔ 6²·2=3·24 (72=72). Real cube roots/injectivity of cubing excluded. |
| 26 | `UniversalLaw.Side24.pin_determinant_and_joint_dimension` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | det diag(3,1,1)=3; dims d+1+d(d+1)/2 = 6 (d=2), 10 (d=3) matching 'at most 10'. Lean det shown only for 3×3 form (d=3); for d=2 trivially same. Covariance structure excluded. |
| 27 | `UniversalLaw.Side24.hessian_block_eigenvalues` | ConeMoments.lean | **ALIGNED AT NARROWER SCOPE** | d+2 = 4,5 and traceless 2; Lean checks 2>1/3 (1<3·2) only, 4,5>1/3 implied a fortiori but not stated. Block decomposition excluded. |
| 28 | `UniversalLaw.Side24.d2_endpoints_adjacent` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | Integers transcribed match ENCLOSURE d=2 lower/upper (checked digit-for-digit in 10^−20 units); adjacency U=L+1. Coefficient enclosure explicitly not claimed. |
| 29 | `UniversalLaw.Side24.d3_endpoints_adjacent` | Endpoints.lean | **ALIGNED AT NARROWER SCOPE** | Matches ENCLOSURE d=3 lower/upper digit-for-digit; U=L+1. Enclosure not claimed. |
| 30 | `UniversalLaw.Side24.d3_below_d2` | Endpoints.lean | **MISALIGNED** | Lean claim d3Upper<d2Lower is a true artifact-consistency fact covered by SCOPE row, but the manifest informal_anchor is '"scientific_acceptance": false', which states nothing about ordering of intervals. Anchor↔statement correspondence fails. Fix: re-anchor to the ENCLOSURE 'dimensions' block (both d=2 and d=3 lower/upper) and mark the ordering as a derived consistency check. |
| 31 | `UniversalLaw.Side24.endpoints_in_unit_tenth` | Endpoints.lean | **MISALIGNED** | Lean claim 0<d3Lower ∧ d2Upper<10^19 (intervals within (0,1/10)) is true for the transcribed endpoints, but informal_anchor '"relative_periodization_bound": "1e-106"' is about the periodization allowance, not the endpoint range. Anchor↔statement correspondence fails. Fix: re-anchor to the 'dimensions' block / PROOF.md interval display. |

## Disposition

**REJECTED (AMEND requested).** 2 targets are MISALIGNED at the anchor level (`d3_below_d2`, `endpoints_in_unit_tenth`): the Lean statements are true and covered by the SCOPE row as artifact-consistency checks, but their manifest `informal_anchor` texts do not state the claims. Per REVIEW_LANE.md a MISALIGNED verdict is corrected at the source (fix the anchor) and re-reviewed. No Lean statement was found to overclaim, flip an inequality, mis-scale, or mis-cross-multiply. The other 29 targets would support acceptance.

Suggested amendment (author/controller action, not performed here): re-anchor both targets to the ENCLOSURE.json `"dimensions"` block (or PROOF.md's interval display), new manifest digest, then fresh review.

## Boundaries

No manuscript, PROOF_INDEX, STATUS, manifest `alignment_status`, Lean, source or evidence file is edited. This record promotes nothing. credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN · Skip #603.

— Grok Bot agent 3
