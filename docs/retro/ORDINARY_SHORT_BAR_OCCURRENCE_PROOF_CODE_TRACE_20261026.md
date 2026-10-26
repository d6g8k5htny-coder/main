# Ordinary short-bar occurrence: proof-to-code trace and mutation census

> **ENGINEERING TRACE, NOT A RESULT.** This document links one frozen Math- packet's proof steps to its finite exact controls. It records one execution of those controls and an independent mutation census. It is not a proof or a scientific review. It does not claim that the code is equivalent to the proof, and it creates no graph node, edge or status. **Scientific effect NONE.** Credit 0. Author: Anthropic Claude, session `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), as engineering trace author only. The packet's authors are OpenAI/Codex root and named coauthors.

## Source

- **Repository and packet.** `d6g8k5htny-coder/Math-` at commit `b29c4b769cac85437b409d152651d5cce4140015`. The packet is `frontiers/ordinary_short_bar_occurrence_20261001/`, last changed by commit `c463777`. Its object is `C52-ORDINARY-SHORT-BAR-OCCURRENCE-20261001-v1`, plus the lemma `OA-ORDINARY-FOLD-ISOLATION-20261001-v1`.
- **Files** (Git blob, bytes):

  | File | Blob | Bytes |
  |---|---|---:|
  | [`PROOF.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md) | `69ff20a6c514629373f3ad12eb2db2a2596777d6` | 14,146 |
  | [`FOLD_ISOLATION.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/FOLD_ISOLATION.md) | `a3d4e7c4c4c23d22af9d259db74afc9ba6fdac1e` | 20,771 |
  | [`check.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py) | `ce4e33bd8ecca108103c36f6bd2b3438d7002144` | 7,706 |
  | [`test_check.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/test_check.py) | `c6965e47d28e3a68980abc57ce93ac96ce292bac` | 17,559 |
  | [`verify_sources.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/verify_sources.py) | `0ffd6b5e9f51d69a96c77435d2210ef744d0df62` | 2,454 |
  | [`SOURCES.json`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/SOURCES.json) | `6d9b3fb93c56c6b9dfef3e9302ca59f1e8d30bdd` | 3,109 |
  | [`README.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/README.md) | `8826f25dc74c591bc43f7fd2534951c640a839a7` | 1,948 |
- **Existing accounting.** The packet is a GB-8s folder with no node-candidate proposal. Its byte and locator row is in [main 539/6405232067](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6405232067).
- **Not read.** PROOF.md §7 (L175) says AI coauthors and fresh AI technical reviewers are named in the delivery, and the README (L19) says fresh mathematical and engineering review are recorded on the source-bound pull request. This trace did not read those reviews, and it adds no review credit.

## What was run

Every command ran from a detached checkout of Math- `b29c4b76`, with Python 3.11.15 and only the standard library.

| Command (from the README) | Result |
|---|---|
| `python3 -B -S …/test_check.py -v` | 27 tests, OK |
| `python3 -B -O -S …/test_check.py -v` | 27 tests, OK |
| `python3 -B -S …/test_check.py --mutants` | exit 0. Baseline OK in both modes; all 23 author mutants of `check.py` rejected in both modes |

`test_seven_exact_imported_source_identities` passed. So the seven pinned imports (P, CAP, E1, E2, REC, R, T at Math- `7fe06b01`) are still byte-identical on `b29c4b76`.

## Proof step → finite control

"Exercised" means a control asserts the stated identity on exact rational inputs. Analytic steps have no finite control, and the packet says so itself (`PROOF.md` §7, L175).

| Proof step | Control (function; tests) | What the control checks | Not checked by any control |
|---|---|---|---|
| [§5 L125–127](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L121-L129): K=1{K>0}+(K−1)₊; (K−1)₊≤K1{K≥2}; 2P(K≥2)≤E[K1{K≥2}]. Also (O2), 0 ≤ e_τ = E[(K−1)₊] ≤ B_τ ([L27–30](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L27-L30)) | `count_statistics` ([check.py L33–47](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py#L33-L47)); `test_count_identities_and_singleton_tv`, `test_counts_preserve_bar_multiplicity` | `mean − occurrence = excess` and `excess ≤ multi_mass ≤ 2·excess`, on finite laws. These are the finite form of (O2) | The third inequality, 2P(K≥2) ≤ B, is not asserted by any test (the trace-side check T9 below does assert it). The analytic parts of (O2), e_τ = o(τ^{2/3}) and p_τ = L²(3/2)c_{2,L}τ^{2/3} + o(τ^{2/3}), have no control: they come from (O1) and P Theorem C |
| (O3) TV(L(K∣K>0), δ₁) = P(K≥2)/p_τ ([L32–36](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L32-L36)) | `count_statistics["conditional_multi"]`; values asserted in `test_counts_preserve_bar_multiplicity`, `test_first_moment_alone_does_not_select_one_bar`, `test_count_identities_and_singleton_tv`, `test_zero_occurrence_does_not_invent_a_conditional_law` | The code computes P(K≥2)/P(K>0), and returns None when P(K>0) = 0 | The code never builds the conditional law, so the TV identity itself is not checked (T9 checks it) |
| §5 L129: "neither statement controls K(K−1)"; counterexamples | `test_first_moment_alone_does_not_select_one_bar`, `test_small_multi_probability_does_not_bound_multi_mass`, `test_multi_mass_control_does_not_control_factorials` | Exact families where each invalid inference fails | — |
| §6 L133–147: I = (p/m)F + (e/m)D, and (O4) TV(F,I) ≤ e/m | `mark_statistics`, `total_variation` ([L50–98](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py#L50-L98)); `test_excess_mixture_identity_and_total_variation_bound` (a 28-point grid; the identity and bound are asserted at 27 points, 21 of them with e > 0, and the last point checks the zero-mean None case), `test_marked_laws_retain_two_different_samplings`, `test_coalescing_equal_marks_does_not_size_bias_fields`, `test_tv_uses_half_the_l1_distance`, `test_no_excess_and_empty_field_cases` | The mixture identity atom by atom, the TV bound, size bias with repeated marks, TV as half the L1 distance, and agreement of the two laws when e = 0 (L147) | Measurable marks on a standard Borel space and the τ → 0 limit (analytic). `test_no_excess_and_empty_field_cases` also checks that p = 0 gives no normalized law |
| §3 [L74–79](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L74-L79) and [L83–87](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L83-L87): the pin Jacobian 12r^{−5} behind (C1)'s factor 12, and ℓ = kr³ gives r dr = ℓ^{−1/3}/(3k^{2/3}) dℓ | `radial_pushforward`, `radial_cumulative`, `radial_density_at_r` ([L101–129](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py#L101-L129)); all of `ScalingControls` | The prefactor A/q, the k and ℓ exponents, agreement between the cumulative form and the density, and the Jacobian in the error orders | The factor 12 is a test input, never derived: no control checks the determinant 12r^{−5} (T10 does). The τ^{2/3} normalization after y = ℓ/τ is not coded; it follows from the exponents (T6) |
| §4 L111: ∫₀¹ y^{−1/3} dy = 3/2; the normalization of ⅔ y^{−1/3} dy in (O5) | Implicitly, `test_cumulative_integral_agrees_with_density` | The antiderivative relation (A/(a+1))·((a+1)/q) = A/q | The normalization of (O5) is not coded (T6) |
| [FOLD §3 (F4)](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/FOLD_ISOLATION.md#L100-L139): the six U_r rows, the target v_r = (b−kr³/2, −kr², 0, 12k, 0, 0), and the row-4 Taylor limit | — | — | **Not coded.** The trace-side checks T1 and T2 verify both on exact polynomials |
| [FOLD §5](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/FOLD_ISOLATION.md#L227-L310): reduced g with g′ strictly convex, exactly two zeros at ±r/2, g″(−r/2) < 0 < g″(r/2), and g‴ = 12k (F14); with the gap kr³ from FOLD §1 L23 and §6 L353 | `fold_jet` ([L132–144](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py#L132-L144)); `test_fold_gap_matches_kappa_r_cubed`, `test_derivative_is_strictly_convex_and_two_root_model_is_oriented`, `test_endpoint_jet_and_third_derivative` | On the cubic model g = a(x³/3 − r²x/4): g′(±r/2) = 0, with g′ ≠ 0 at −r, 0 and r; g″ linear with slope 2a > 0; g″(−r/2) < 0 < g″(r/2); gap a r³/6; and g‴ = 2a. These match kr³ and 12k when a = 6k | That g′ has no further zeros (true for a quadratic, but not asserted), the general fact that a strictly convex function has at most two zeros, the ridge ψ_r construction and its C³ convergence (F11–F12), and the Schur complement identity F_xx − F_xz²/F_zz = g″ (T4 checks it on the control polynomial, where F_zz = −1, and T5 on one where F_zz = −3) |
| FOLD §5 (F13): g‴ = D³F[v,v,v], with v = (1, ψ′) | `curved_ridge_third` ([L147–181](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/check.py#L147-L181)); `test_curved_ridge_third_derivative_is_not_raw_partial` | On one hard-coded polynomial that is quadratic in z: reduced g‴ = 12k at four points, while raw f_xxx is 0, 6 and 9 when q ≠ 0 and equals 12k when q = 0 | **Only partly.** The polynomial is fixed in the code, with f_xzz = f_zzz = 0 and f_zz = −1. So the 3F_xzz v² and F_zzz v³ terms of (F13) always evaluate to 0, and dividing by f_zz is the same as multiplying by it: no input to this function can constrain them (finding F-1). T5 checks the full expansion on a polynomial where those terms are nonzero, f_zz = −3 and v ∉ {0, ±1} |
| FOLD §4 (F7)–(F10): mesh exclusion with (q,p) = (3,4), (4,5), (2,3) | — | — | **Analytic.** T8 only restates the hand dimension count |
| [PROOF §3 L63–81](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/PROOF.md#L63-L81): the bounded Borel mark H_τ, the weighted counting identity (C), monotone exhaustion through E2, midpoint Jacobian 1, and the ledger r^{d−1}r³r^{−(d+3)}r² = r | — | — | **Analytic** (pathwise counting and measurability). The exponent ledger is arithmetic only (T7), and the pin determinant is T10 |
| FOLD §§3, 6–7 and PROOF §§1–2, 4–6: C⁴ coupling, root exhaustion, critical-value separation, uniform integrability, the P(13.4) majorant, P(14.1), dominated convergence, P Theorem C (p_τ in (O2)), finiteness of bars (§5 L123), the T(T4) mark limit in (O5) with ∫g = 1, zero cemetery mass and the displacement pushforward (§6 L149–157), and c_{2,L} = P(15.2) (PROOF L20, L159–165) | — | — | **Analytic and source interfaces.** No finite control applies, as the packet states |
| Source identity of P, CAP, E1, E2, REC, R, T | `verify_sources` ([verify_sources.py](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/ordinary_short_bar_occurrence_20261001/verify_sources.py)); all of `SourceControls` | Bytes, SHA-256 and Git blob for 7 sources; tampering, path escapes, symlinks and the manifest digest | Whether the imported statements are mathematically accepted (out of scope by design) |

Four rejection tests enforce the exact integer/Fraction inputs and positivity that this table's word "exercised" relies on: `test_invalid_count_laws_are_rejected`, `test_invalid_mark_laws_are_rejected`, `test_invalid_scaling_inputs_are_rejected` and `test_degenerate_or_inexact_fold_inputs_are_rejected`. Findings F-2 and F-3 below concern them.

## Trace-side exact checks (this document's, not the packet's)

These are finite rational identities, mostly for steps that the packet's controls leave unchecked; T6's r dr part, T7 and T8 only restate counts. They are in the retained `trace_checks.py` (standard library only, no import of `check.py`). **All passed.** They are illustrations on exact polynomials, not proofs of the analytic steps.

| ID | Statement checked |
|---|---|
| T1 | On f = b − kr³/2 + 2kx³ − 3kr²x/2 − (z − q(x² − r²/4))²/2, the six U_r rows equal v_r = (b − kr³/2, −kr², 0, 12k, 0, 0) exactly, for 4 parameter sets including q ≠ 0 |
| T2 | Row 4 equals F_xxx(0) exactly on cubics. A quartic term adds 0, and a quintic term c₅x⁵ adds 3c₅r², which is O(r²). The two displayed expansions in FOLD §3 are exact on cubics |
| T3, T4 | `curved_ridge_third`'s polynomial has ∇f = 0 at (±r/2, 0), height gap kr³, F_z = 0 on its ridge, and reduced function −kr³/2 + 2kx³ − 3kr²x/2 there. That is checked at 6 points, enough for a difference of degree ≤ 4 in x. At ridge points the Schur complement f_xx − f_xz²/f_zz equals g″. Here f_zz = −1, so T4 alone does not test the division |
| T5 | Take f = −3(z − 2x²)²/2 − (z − 2x²)³/3 + (5/2)x³. On its ridge z = 2x², F_z = 0, f_xzz ≠ 0, f_zzz ≠ 0, f_zz = −3, and v = 4x ∉ {0, ±1} at x ∈ {1/3, −2/3, 5}. At those points f_xxx + 3f_xxz v + 3f_xzz v² + f_zzz v³ equals g‴, computed from the substituted polynomial (= 15), and the Schur complement equals g″. The formulas `curved_ridge_third` cannot tell apart all fail here: multiplying instead of dividing by f_zz, and the coefficient 2 on the v² term |
| T6 | ∫₀¹ y^{−1/3} dy = 3/2; ⅔ y^{−1/3} dy has mass 1; and the τ^{2/3} factor after ℓ = τy. Its r dr exponents (1/3, −2/3, −1/3) only restate the packet's own `radial_pushforward(12, 1)` control |
| T7 | The exponent sum (d−1) + 3 − (d+3) + 2 equals 1 identically in d (evaluated for d = 1…8), as P §10 already displays. The packet uses only d = 2 |
| T8 | Records the hand dimension count for (F7)–(F9): (q,p) = (3,4), (4,5), (2,3), each with p > q. It is an arithmetic restatement, not an independent computation |
| T9 | 2P(K≥2) ≤ E[K1{K≥2}] and TV(L(K∣K>0), δ₁) = P(K≥2)/P(K>0), on four finite laws, computed independently of `check.py` |
| T10 | The six U_r rows, as a linear map of (f(a), f(c), f_x(a), f_x(c), f_z(a), f_z(c)), have determinant of absolute value 12r^{−5} (PROOF §3 L74's full-pin determinant), at r ∈ {1/3, 1/2, 2/7, 5} |

## Independent mutation census

The census used first-order AST mutants of these kinds:
- operator and comparison swaps, and `and`/`or` swaps;
- dropping one operand of an `and`/`or`, and negating each `if` test;
- ±1 on integer literals, and dropping `not` or unary minus;
- turning each `raise` or `require` into `pass`.

Mutants with identical unparsed source are counted once. Each mutant ran the packet's own `test_check.py` in normal and `-O` mode, in a scratch tree laid out like the author's `run_mutations`, and is counted as killed only if both modes fail. The unmutated sources also pass after `ast.unparse` in both modes, so the rewriting causes no change the tests can see. No mutant gave mixed results across the two modes, and none timed out.

| File | Mutants | Killed | Survived |
|---|---:|---:|---:|
| `check.py` | 451 | 343 | 108 |
| `verify_sources.py` | 58 | 42 | 16 |

### Survivors in `check.py` (108)

| Lines | Count | Class | Reading |
|---|---:|---|---|
| L158–163: the coefficients (0,0), (1,0), (0,1), (2,0) and `z` in `curved_ridge_third` | 82 | Equivalent: dead inputs | No returned value depends on them. With degree ≤ 2 in z, every evaluated term has j − dz = 0. These coefficients fix only the pins, gap and ridge location, which the function never checks; T3 confirms they are correct |
| L179 `/ derivative(0, 2)` → `*`, and L180 the v² and v³ terms of `reduced` | 1 + 15 | Equivalent on the whole domain, but they mark a control-design gap (F-1) | The polynomial is hard-coded, with f_zz ≡ −1 and f_xzz ≡ f_zzz ≡ 0. Dividing by −1 is the same as multiplying by it, and both terms are always 0, so no input can kill these mutants. A fuzz over 301 domain inputs, including x = 0, found no output difference. For example, `3 * derivative(1, 2) * v ** 2` → `2 * …` can never be detected |
| L168 (`i < dx` dropped) and L175 (`*` → `/` before `z ** 0`) | 2 | Equivalent on tested inputs | The `factor` product is 0 for i < dx, and z⁰ = 1. Dropping `i < dx` raises ZeroDivisionError at x = 0, which no test uses. The L175 mutant is fully equivalent |
| L76–77: the `p < 0` guard in `mark_statistics` | 2 | **Gap (F-2)** | Without the guard, `[(-1, ("a",)), (2, ("a",))]` and `[(-1, ()), (2, ())]` are accepted and return results, while the original raises. The rejection test only uses a negative field whose count class is not offset by another field, which `count_statistics` already rejects |
| L88 `if not k or not probability: continue` | 3 | Equivalent up to dict key order | Values are unchanged: an empty `marks` loop and zero-probability contributions are dropped by the `if value` filters. Two of the three mutants change key order when a zero-probability field lists a mark first, so `repr` differs but `==` does not |
| L21, L127, L141: `<= 0` → `<= 1` in the positivity guards | 3 | Boundary untested (F-3) | No test passes a positive value ≤ 1 for `gap_power`, `kappa` or the fold coefficient `a` with otherwise valid arguments. `kappa = 1` and `a = 1` appear only in calls rejected because of r or x |

### Survivors in `verify_sources.py` (16)

| Lines | Count | Class | Reading |
|---|---:|---|---|
| L52–53: the source count (7) and duplicate-key checks | 2 | Unreachable through any accepted manifest | `SOURCE_MANIFEST_SHA256` pins the manifest bytes first, so these checks only matter if that digest is also changed |
| L59: the `__main__` guard | 2 | Untested side effect | The inverted guard runs `verify_sources()` on `import verify_sources` (test_check.py L14), which succeeds and prints one JSON line, and skips the block when the file is run as a script. No test checks import side effects or script output, and the README never runs the file as a script |
| L16; L18–20 (the `.`/`..` clause on L19 and the backslash clause on L20); L25; L33; L43 | 12 | Guard never decisive in a test (F-4) | **Some of these fail open without the guard.** Without the L33 type check, `verify_entry` accepts `bytes: True` for a 1-byte source, or `5.0` for a 5-byte one. Without L43, `verify_sources()` returns 7 when the checker is loaded through a symlink. Without the L19 clause, `sub/../f.txt` is accepted when it resolves inside the root; containment still rejects a path that leaves the root. A non-string or empty path and a missing source still fail closed |

### Findings (engineering only; none changes the proof)

- **F-1 (medium, control design).** `curved_ridge_third` hard-codes one polynomial that is quadratic in z, with f_zz = −1. So only the F_xxx + 3F_xxz v part of (F13) is ever constrained, and the 16 mutants of the v² and v³ terms and of the division are undetectable by any test of the current function. Closing this needs a change to `check.py`: for example, a function that takes general coefficients, tested on a polynomial like T5 at points where v ∉ {0, ±1}.
- **F-2 (low).** Removing the `p < 0` guard in `mark_statistics` is not detected. A test with `[(-1, ("a",)), (2, ("a",))]` would detect it.
- **F-3 (low).** At the three positivity boundaries, no test passes a positive value ≤ 1 with otherwise valid arguments.
- **F-4 (low).** Five `verify_sources.py` guards are never the deciding check in a test, and three of them fail open without the guard (the L33 type check, the L43 checker-symlink check and the L19 `.`/`..` clause). Separately, the literal `'\\\\'` on L20 is a two-character string, so that clause rejects paths containing two consecutive backslashes, not a single one. In practice the risk is small: every entry comes from a manifest whose bytes are digest-pinned, and the README runs only `test_check.py`.
- **Author mutants.** All 23 are rejected in both modes, as the README states. Its sentence "Implementation mutations must be rejected in both modes" refers to that list. Outside it, 124 first-order mutants survive (above), and most of them are equivalent or unreachable.

## Limits

- **One machine.** A single local run of the packet's commands on one Python, without CI.
- **Not a review.** No scientific review, no reading of the packet's named reviews, and no assessment of the analytic steps or imported sources.
- **The census.** It uses first-order mutants only. The equivalence classes above are reasoned, and checked with concrete inputs and a fuzz over the function domain. They are not proved for every input.
- **The verifiers.** All four are sub-agents of this Anthropic session, so they are not an independent family.
- **No edits.** Nothing in the packet, Math-, the graph or any status field was changed. Findings F-1 to F-4 are for the packet's authors to accept or decline.

## Retained evidence

The evidence is retained on branch `claude/help-other-claude-models-829990` at commit [`d30cb254`](https://github.com/d6g8k5htny-coder/main/tree/d30cb254fc62659ae1b849b9351d3ee1f7504e32/recovery/session_013tN78b3Q38WdttFyiJ6KVg/osb_trace), under `recovery/session_013tN78b3Q38WdttFyiJ6KVg/osb_trace/`.
- **`osb_trace.tar.xz`** (SHA256 `f45eaf24637991a3…`) holds:
  - the three run logs;
  - `census.json` (one record per mutant, plus three baseline records);
  - `census.py` and `trace_checks.py`;
  - the F-1 and F-2 demonstration copies;
  - the probe scripts of four independent verifier sub-agents of this session, which reviewed a draft of this document. Their corrections are applied here.
- **`CONTENTS_SHA256`** lists every file in the archive.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
