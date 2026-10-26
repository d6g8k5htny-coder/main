# Unique replacement-bar intensity: proof-to-code trace and mutation census

> **ENGINEERING TRACE, NOT A RESULT.** This document links one frozen Math- packet's proof steps to its finite exact controls. It records one execution of those controls, an independent mutation census, and trace-side exact checks for steps the controls take as given. It is not a proof or a scientific review. It does not claim that the code is equivalent to the proof, and it creates no graph node, edge or status. **Scientific effect NONE.** Credit 0. Author: Anthropic Claude, session `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), as engineering trace author only. The packet's authors are OpenAI/Codex `/root`, `/root/c50_multiplicity_feasibility` and its child `boundary_nullity`, under Dylan Roy's delegation (PROOF.md L3, ROOTS.md L3–7, README.md L47). ROOTS states that boundary_nullity's derivation is disclosed author-side work, not independent review.

## Source

- **Repository and packet.** `d6g8k5htny-coder/Math-` at commit `b29c4b769cac85437b409d152651d5cce4140015`. The packet is `frontiers/unique_replacement_bar_intensity_20261001/`, last changed by `b459061`; it was introduced in `b5bf028` and source-bound in `66d7f0f`. Its objects are `OA-UNIQUE-REPLACEMENT-BAR-20261001-v1` (PROOF.md) and `OA-UNIQUE-REPLACEMENT-BAR-ROOTS-20261001-v1` (ROOTS.md).
- **Files** (Git blob, bytes):

  | File | Blob | Bytes |
  |---|---|---:|
  | [`PROOF.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/PROOF.md) | `9c79a672f5d81e04972032b2f1faf185f4664fa5` | 17,545 |
  | [`ROOTS.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/ROOTS.md) | `b27d9ac8111c83f4967d837eee22057583c41eae` | 11,969 |
  | [`check_exact.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/check_exact.py) | `6521bed3c0abbe70f07b53534067df1c7c910240` | 14,327 |
  | [`test_sources.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/test_sources.py) | `37bf1e31beba962b1c0c74b2856184f7c17e74f2` | 7,243 |
  | [`SOURCES.json`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/SOURCES.json) | `210dcacaac4eb0f4a2480959fdf342daff5d41cc` | 2,515 |
  | [`README.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/README.md) | `043cf1830c5ce605406bfaabbbd0581b2b3b74a9` | 3,359 |
- **Existing accounting.** The packet appears in GB-8t's node-candidate proposal (`docs/retro/GB-8t.agent2.node-candidates.json`) and in the post-cut proof-source join (`docs/retro/RETRO_PROOF_SOURCE_JOIN_20261025.md`, landed through #618). Neither maps its proof steps to its code. The join also records byte-identical copies of PROOF.md, ROOTS.md and SOURCES.json under `frontiers/replacement_bar_occurrence_20261001/sources/c50/` (dup-02). This trace covers only the original packet's code.
- **Not read.** PROOF.md L205 sets three review targets (A–C) and requires a fresh nonauthor, source-bound review. README.md L47 says source-bound review is recorded in the associated PR. This trace did not read that PR, and it adds no review credit.

## What was run

Every command ran from a detached checkout of Math- `b29c4b76` with Python 3.11.15.

| Command (from the README) | Result |
|---|---|
| `python3.11 -B -S …/check_exact.py` | exit 0, `"status": "PASS_EXACT_FINITE_FIXTURES"` |
| `python3.11 -B -O -S …/check_exact.py` | exit 0, same status |
| `python3.11 -B -S …/test_sources.py` | 13 tests, OK |
| `python3.11 -B -O -S …/test_sources.py` | 13 tests, OK |

**Environment note.** The README requires Git with `--no-lazy-fetch`, which arrived in Git 2.44. This machine has Git 2.43.0. There, all four commands exit nonzero with `historical Git read failed: rev-parse`; `test_sources.py` reports `FAILED (errors=3)`, with the other 10 tests passing. The runs above used a test-only shim that drops only that flag and then runs the system Git. Dropping it cannot hide a lazy fetch here, for two reasons:
- The local clone is not a partial clone: there is no `extensions.partialClone` and no `remote.*.promisor`. It is shallow (`git rev-parse --is-shallow-repository` is `true`): `.git/shallow` lists only `044d42d5`, from the bounded `git fetch --depth=1 origin 044d42d5…` that README L40 gives. The parent objects of that commit are present.
- `source_git` also sets `GIT_NO_LAZY_FETCH=1` (check_exact.py L43). On a test partial clone, this Git 2.43.0 build honours it: it refused to fetch a missing blob (exit 128, `lazy fetching disabled`), while plain `git cat-file` fetched the blob.

The pinned source commit `044d42d5bd7f31df6f7e6b9034d6711e8fb7ee43` is present locally, and the checker's own historical-blob checks for all seven sources passed. The packet also has its own workflow, `.github/workflows/unique-replacement-bars.yml`, which runs these commands on GitHub Actions. This trace did not read its run results.

## Proof step → finite control

"What the control checks" lists only conditions that a control asserts on exact rational inputs. The guards are explicit `RuntimeError` conditionals, so they stay active under `-O` (ROOTS.md R4 L243–246). Analytic steps have no finite control, and the packet says so itself (PROOF.md L207, ROOTS.md L290–292).

| Proof step | Control (function; fixture) | What the control checks | Not checked by any control |
|---|---|---|---|
| PROOF §3 L78–83 and ROOTS R1 L26–30: the cubic P_θ, B_s, D_s and the typing T_k = {s < −\|B_s\|/2} | `value`, `gradient`, `determinant`, `shear_data` ([check_exact.py L125–153](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/check_exact.py#L125-L153)); `verify_fixture` L204 for typing | Pin heights 0 and −k; gradient zero and Morse at all four listed roots; strict typing; three fixtures, one with nonzero shear a = 12 | The fixtures' extra roots are inputs; the code never solves for critical points (U1). The k-dependence of B_s, D_s and u, because every fixture has k = 1 (UF-1) |
| ROOTS R1 L33–52: besides the pins, every critical point satisfies (R1.1), so there are **at most two extra critical points anywhere**; the elimination (R1.2) | — | — | **Not coded.** Nothing checks that the listed roots are all the critical points. U1 does this for each fixture with an exact resultant; U2 checks (R1.1) at the extras |
| ROOTS R1 L54–78: det Hess = −3k(B_s + 4su) at extras; pin determinants 3k(B_s ∓ 2s); identity (R1.3) H Z² = 3k(B_s + 4su)² | `verify_fixture` L211–216: `determinant` equals stated values; `h*y[1]**2 == 3*p[0]*(b+4*p[1]*u)**2` at each extra; pin determinants only for `below_window` (L255–256) | (R1.3) at the six extra roots; the stated determinant values | The determinant formula −3k(B_s+4su) and the pin formulas are not asserted as formulas. The algebraic derivation of (R1.3) is not checked (U2, U4) |
| ROOTS R1 L80–106: H = H_poly/(1728k³), whose q² coefficient 5184k⁴ makes it a nonzero polynomial; the retained exclusions Σ, Δ, D_s ≠ 0 | `verify_fixture` L205–206: D, Σ, Δ and H are nonzero at each fixture | Nonvanishing of the four exclusions at the fixtures | The Σ and Δ formulas themselves: only nonvanishing is asserted (UF-2). The H/H_poly identity and the q² coefficient (U3). That the zero sets are Lebesgue-null is analytic |
| ROOTS R2 L126–140: scaling (k, η) → (k, kη), P_(k,kη) = kP_(1,η), roots fixed, gap ratios scale by k, typing preserved; Jacobian k⁴ | `main` L283–289: the `below_window` fixture scaled by 7/3 keeps its roots, scales its values, B_s and D_s | One scaling ray at one fixture, with a = 0 | The identity at general (k, η), and preservation of typing (U7). Fubini and nullity are analytic |
| ROOTS R2 L114–152 and PROOF §5.2: radius and gap boundaries are null | `verify_strict_margins` ([L175–184](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/check_exact.py#L175-L184)) on all fixtures | Each saddle root of each fixture (det < 0) lies strictly off the band and gap boundaries at the t values used | The maximum M_0 is skipped, and it lies on the gap-boundary equation (0 = endpoint² · 0). Nullity itself (analytic) |
| PROOF (5) L90–93 and ROOTS R3 L215–218: D_(t,k) counts all saddles with P < 0, Y ≠ Y_*, in the band and gap range, in **original** coordinates | `admissible`, `rejected`, `distance_squared` ([L156–197](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/check_exact.py#L156-L197)); `main` L235–281 | D = 2 for `two_window` at t = 1, and its window-only count is also 2 (L245–247, ROOTS L262–264); D = 1 at t = 8/5 (radius crossing); D = 2 for `below_window`, where the window-only count is 1; there, P(Y_2) < −k (L254, ROOTS L274), \|Y_2 − M_0\|² = 5525/576 and the squared gap ratio is 71289/10793861 (L257–260, ROOTS L279–281); D = 1 for `nonzero_shear` in original coordinates, 2 if sheared; including the actual partner gives 3 | The saddle filter, the P < 0 filter and both gap bounds never decide a fixture count (UF-3; survivors at L164 and L172). U8 independently recomputes the four genuine counts 2, 1, 2, 1; the three wrong-alternative counts are not recomputed |
| ROOTS R3 L227–239 and PROOF L55: D_(t,k) ∈ {1, 2} on the limiting failure population; both values occur | the counts above | D = 1 and D = 2 both occur on the fixtures | That D ≤ 2 for every typed θ (it follows from R1's at-most-two-extras, which no control checks) |
| PROOF §3 L86 and ROOTS R1 L104–106: the unique highest window saddle Y_* | `verify_fixture` L217–222 | A nonempty window and a unique maximum of P on it, per fixture | The window predicate −k < P < 0 is never decisive (no fixture extra lies near either edge). ELDER's identification of Y_* as the actual global partner (source interface) |
| PROOF §4 (7) L113–121: Σ_M Φ(V_h(M)) = Σ_(rejected pairs) Φ(V_h(M))/D_h(M) | `main` L237–239: `sum(F(1, count1) for _ in range(count1)) == 1` and `count1 != 1` | That count1 copies of 1/count1 sum to 1 at count1 = 2 | This is arithmetically trivial for any count1 ≥ 1 and involves no configuration or signed mark. U5 checks the double counting in (7) on random finite configurations |
| PROOF §7 L181–199: r dr · r³ = h⁵t⁴ dt; C_R = ((C_0⁵ − c_0⁵)/5)∫A_0(α_1+α_2); (4) C_R/2 ≤ C_U(1) ≤ C_R | — | — | **Not coded.** U6 computes the substitution and the t-integral. In (4), C_R/2 ≤ C_U(1) ≤ C_R follows from D ∈ {1, 2}; that 0 < C_R < ∞ is analytic (imported bounds, PROOF L199) |
| PROOF §§1–7 and ROOTS R2–R3: the main limit (2)–(3) and the scaled-mark limit (§5.3), Borel measurability of the mark, Kac–Rice and E2 disintegration, the regression coupling [CUB G6], C² root continuation, ELDER Theorem E, the W_r/r⁴ envelope, theta-box exhaustion [ELDER S9, CUB G13], dominated convergence, the normalization z_0 / A_0 (§3 (6) L99–109), the semialgebraic measurability of the (R2.1) set and the persistence of nullity under absolutely continuous measures (ROOTS L121–124, L154–161) | — | — | **Analytic and source interfaces.** No finite control applies, as the packet states |
| SOURCES.json: seven sources pinned at commit `044d42d5`, path containment, bytes, SHA-256, blob, historical commit and blob, Git environment isolation | `verify_sources`, `verify_source_entry`, `verify_historical_source`, `contained_file`, `source_git` ([L21–118](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/unique_replacement_bar_intensity_20261001/check_exact.py#L21-L118)); all 13 tests of `test_sources.py`; the built-in wrong-digest probe L104–112 | Every listed property on the real manifest. On temporary real Git repositories: exact acceptance, current-only path, changed historical blob, missing commit, tree or tag object, historical symlink, commit and blob replacement, hostile environment, external Git directory, parent repository, and a working-file change after the manifest (rejected by the byte-length guard, L82). The SHA-256 guard (L84) is exercised only by the built-in probe | Which guard rejects each negative case: the tests accept any `RuntimeError` (without the shim, 10 of 13 pass with Git unusable). Removing L52, L58, L63, L66, L74, L79 or L87 goes unnoticed, and L79 (entry commit) and L82 (byte length) have no later backstop (UF-4). Mathematical acceptance of imported statements is out of scope by design |

## Trace-side exact checks (this document's, not the packet's)

These are in the retained `trace_checks_urb.py` (standard library only; it does not import `check_exact.py`). **All passed.** They are finite computations, on the packet's own fixtures, on random rational points or on random finite configurations. They are not proofs of the analytic steps.

| ID | Statement checked |
|---|---|
| U1 | **All critical points are listed.** For each fixture, ∇P = (g_1, g_2), where g_1 is quadratic in X with leading coefficient 6k ≠ 0 and g_2 has degree at most 2 in X (linear when a = 0). The resultant Res_X(g_1, g_2)(Z) is c·Z²(Z − z_1)(Z − z_2) with c ≠ 0, where z_1, z_2 are the fixture extras. At Z = 0 the common roots are exactly X = ±1/2, the pins. At each extra Z the common X is unique and equals the fixture's. So each fixture's four roots are all the critical points of its cubic |
| U2 | At each fixture extra, det Hess = −3k(B_s + 4su) with u = X + aZ/(12k), and (R1.1) holds with Z ≠ 0. At every fixture, the pin determinants are 3k(B_s − 2s) at M_0 and 3k(B_s + 2s) at S_0 |
| U3 | H = H_poly/(1728k³) at the fixtures and at 300 random rational points, and H_poly's q² coefficient equals 5184k⁴ there. This is randomized evidence for a polynomial identity, not a symbolic proof |
| U4 | The substitution in ROOTS R1 L72–74 reduces (R1.3) to 12k(s+Bu)² − 12kB²(u²−1/4) + 48ks²(u²−1/4) = 3k(B+4su)², which holds at 300 random rational points; (R1.3) itself holds at the six fixture extras |
| U5 | (7) is a double-counting identity, so U5 checks its bookkeeping. It uses 300 random finite planar configurations. Each has maxima with an actual partner and a signed mark Φ, saddles, and band and gap rules as in (5). D(M) is counted per maximum, and the rejected pairs are enumerated separately, saddle by saddle. The two sides of (7) agree on every configuration, and maxima with D = 0 drop out of both. D took every value from 0 to 8. Dropping 1/D changes the sum on 271 configurations. A count that forgets the partner exclusion breaks the identity on 189, so the check can fail |
| U6 | Substituting r = th: ∫_{c_0h}^{C_0h} r·r³ dr = h⁵∫_{c_0}^{C_0} t⁴ dt = h⁵(C_0⁵ − c_0⁵)/5 at nine rational (c_0, C_0, h). Also ∫_{c_0}^{C_0} t⁴ dt = (C_0⁵ − c_0⁵)/5 on 25 rational bands. Every integral is computed by Boole's rule, which is exact to degree 5; the script confirms that it fails at degree 6 |
| U7 | On 200 random rational (k, η) and points: P_(k,kη) = kP_(1,η) and ∇P scales by k, so roots are fixed; B_s and D_s scale by k; strict typing is unchanged. At 100 random points, the Jacobian of (k, η) → (k, kη), computed by exact central differences, is lower-triangular with determinant k⁴ |
| U8 | Recomputing D_(t,k) from (5) independently gives the packet's counts: 2 (`two_window`, t = 1), 1 (t = 8/5), 2 (`below_window`), 1 (`nonzero_shear`), so D ∈ {1, 2} with both values occurring |

## Independent mutation census

The census applied first-order AST mutants to `check_exact.py`:
- operator, comparison and `and`/`or` swaps;
- dropping one operand of an `and`/`or`, and negating each `if` test;
- ±1 on integer literals, and dropping `not` or unary minus;
- turning each `raise` or `require` into `pass`.

Mutants whose unparsed source is identical are counted once. Each mutant ran both README commands, `check_exact.py` and `test_sources.py`, in normal and `-O` mode. The runs used a dedicated detached Git worktree of Math- at `b29c4b76`, because the source check reads history, and the same Git shim. A mode fails if either command exits nonzero or `check_exact.py` does not print `PASS_EXACT_FINITE_FIXTURES`. A mutant counts as killed only if both modes fail. The baseline and the `ast.unparse` baseline pass in both modes. No mutant gave mixed results across the two modes, and the worktree was clean afterwards.

| File | Mutants | Killed | Survived |
|---|---:|---:|---:|
| `check_exact.py` | 983 | 748 | 235 |

**Reading this census.** `check_exact.py` checks itself. Its controls are its own `require` calls, its five named negative controls (L304–306) and the wrong-digest probe (L104–112). Deleting or weakening a `require` that holds on the fixtures changes nothing a run can see. So many survivors are a property of the design, not missing tests. Unlike the OSB packet, this one has no mutation harness of its own, so nothing confirms that each `require` carries weight. Each class below is defined by line and mutant kind, as the retained `classify2_urb.py` applies it. The table separates the survivors that the design explains from the ones that point at controls missing from the computation.

| Class | Lines (mutant kind) | Count | Reading |
|---|---|---:|---|
| Fixture assertion removed | every `require`/`raise` → `pass` at L175–L289 | 30 | Undetectable by construction, as explained above |
| Fixture assertion weakened | other mutants inside those assertions at L177–L280, and the t values passed to `verify_strict_margins` at L233, L234, L262, L275, L276 (10) | 55 | Undetected weakenings. The margin-t mutants move the strict-margin check away from the t the count uses; an altered band where the original fails passes under them |
| Fixture constants and counted t | int literals at L232, L243, L261, L264, L274, L283 | 28 | Not code defects: band and gap endpoints, the t of counted calls and the scale 7/3 can move without changing any count. The fixtures have margin |
| Window predicate | L217 | 4 | The window −k < P < 0 that selects Y_* is never decisive, because no fixture extra lies near either edge. Not equivalent: a reflected fixture in the documented domain kills `-p[-1]` |
| Source binding: untested path and manifest guards | `contained_file` L22–L33 (16); `verify_sources` manifest shape and key set L98–L103 (10); script symlink L94 (1) | 27 | No test exercises these (UF-4). `..` and symlink components are caught again by `resolved == target` |
| Source binding: redundant history and entry guards | `source_git` L51–L52 (4); `verify_historical_source` L58–L74 (11); `verify_source_entry` L78–L87 (7) | 22 | The tests target these, but each negative case is still rejected by another guard, and the tests accept any `RuntimeError`. The byte-length and entry-commit guards (L79, L82; 5 survivors) have no later backstop (UF-4) |
| Source binding: probe and message constants | L104, L105, L110, L112 | 6 | Equivalent (for example `entries[0]` → `entries[1]`, or `'0' * 64` → `'0' * 65`) |
| **Σ formula** | L150 | 20 | **Finding UF-2.** Σ is only required to be nonzero, so any formula that stays nonzero on the fixtures passes |
| **k placement** | L142, L148, L149, L151, L214, and the L215 operator mutant | 8 | **Finding UF-1(i).** All three fixtures have k = 1 |
| **Floor division in shear data** | `//` at L148, L149 (2), L204 | 4 | **Finding UF-1(ii).** On the nonzero-shear fixture, a²/(12k) = 12, aβ/(4k) = 27 and a³/(72k²) = 24 are integers, so `//` at L148–L149 is invisible. At L204, −\|B_s\|//2 = −2 in place of −3/2 still leaves s = −21/10 strictly typed |
| Negative-control fidelity | L159 (6, `use_shear`), L193 (4, `window_only`) | 10 | Not equivalent. The sheared-distance and window-only alternatives are not pinned to their named formulas u = X + aZ/(12k) and −k < P < 0. Transformed fixtures in the documented domain kill 5 of these: a reflection (a, q, Z) → (−a, −q, −Z) and the R2 scaling by 3. L159 `12 / p[0]` also hides behind k = 1 |
| `admissible` saddle and height filters | L164 | 8 | **Finding UF-3 (low).** No fixture root is a saddle with P ≥ 0, and the only non-saddle root M_0 is excluded anyway by its zero distance |
| `admissible` band and gap bounds | L168, L172 | 11 | Strict versus non-strict (4) is equivalent by design: every `admissible` call is preceded by a margin check with the same arguments. Neither gap bound excludes any fixture root (UF-3). Raising the lower-bound exponent to `d2**4` is killed, but lowering it and the other perturbations of either bound survive |
| Output formatting | L308 | 2 | Equivalent (`indent`) |

### Off-fixture run of the packet's own code

`packet_code_offfixture.py` imports `check_exact.py` read-only, without running `main()`. It evaluates the packet's functions at 392 random rational parameter points with k ≠ 1 and a ≠ 0 (400 draws, of which the 8 with k = 1 are skipped). **All checks pass.**
- `shear_data` equals the ROOTS R1 formulas for B_s, D_s, Σ, Δ and H, transcribed in the script.
- `value`, `gradient`, `B_s` and `D_s` scale exactly as ROOTS R2 requires.
- `gradient` and `determinant` equal the gradient and the Hessian determinant computed from the packet's own `value` by exact finite differences. These are exact for a cubic: central second differences have no error term, and Richardson extrapolation removes the h² term of central first differences. A `determinant` with `12/k` in place of `12*k` fails this check.
- On the nonzero-shear fixture scaled to k = 7/3, the four roots stay critical, and (R1.3) holds with the script's own u.

These are consistency checks at the sampled points.

**One control closes UF-1 and UF-2.** `proposed_control_demo.py` re-runs the packet's own `verify_fixture`, including L204 and L214–L215, on the nonzero-shear fixture scaled to k = 5/7. Heights are scaled by 5/7 and determinants by (5/7)². The control then pins all five `shear_data` outputs (B_s, D_s, Σ, Δ and H) there to ROOTS R1 values computed in the script. The original passes. All 31 survivors in the k-placement, L148–L149 floor-division and Σ classes fail it, and so does the L204 `//` survivor, for 32. The six L159 survivors still pass it. Because the original passes this control at k ≠ 1 with non-integral shear data, the UF-1 and UF-2 survivors are coverage gaps at this fixture, not wrong code.

### Findings (engineering only; none changes the proof)

- **UF-1 (medium, control coverage).** Two causes hide 12 survivors.
  - (i) Every fixture has k = 1, and the only scaling check uses a = 0. So the k-dependence of B_s, D_s, u, Δ and the (R1.3) check is unconstrained: 8 survivors, plus L159 `12 / p[0]` in the negative-control class. ROOTS R2's scaling-nullity argument rests on exactly this homogeneity.
  - (ii) The nonzero-shear fixture's shear terms are integers, and its typing has margin, so floor division passes: 4 survivors.

  The control above closes both, except L159.
- **UF-2 (medium, control coverage).** Σ, the source exclusion that supports window stability (ROOTS L101), is checked only for nonvanishing (20 survivors). Pinning its value at one fixture with k ≠ 1 closes it; the control above uses k = 5/7. Pinning it at any of the current k = 1 fixtures would leave the L150 mutant `12 / k * d * d`.
- **UF-3 (low).** The `admissible` saddle and height filters and both gap bounds never decide a fixture count, and the window predicate at L217 is never decisive. Three additions would exercise them: a fixture with a positive-height or non-saddle root inside the band, candidates just outside each gap bound, and an extra near a window edge.
- **UF-4 (low).** `test_sources.py` does not exercise `contained_file`'s canonical-path, containment and symlink guards, or the manifest-shape, key-set and script-symlink checks (27 survivors). Its negative tests accept any `RuntimeError`, so 22 survivors in the history and entry guards go unnoticed. The byte-length and entry-commit guards (L79, L82) have no later backstop. Unlike the OSB packet's `'\\\\'`, the backslash test here (`'\\'`, one character) is correct.
- **UF-5 (low).**
  - The L159 and L193 negative-control alternatives are not pinned to their named formulas (10 survivors, 5 of them killed by transformed fixtures).
  - The one-maximum reciprocal check at L237–L239 is arithmetically trivial, as the trace table notes. U5 checks the double counting in (7) on random configurations.
  - The script has no mutation harness that would show its `require` calls carry weight.
- **What holds.** The five named negative controls all behave as stated: omit reciprocal, include actual partner, keep a root after a radius crossing, omit below-window candidates, use sheared distances. The wrong-digest probe is rejected, and all 13 history-binding tests pass. No finding contradicts a README or ROOTS R4 claim.

## Limits

- **One machine.** This trace ran the commands once locally, on one Python, through a Git shim, without CI. The packet's own workflow runs them on GitHub Actions; this trace did not read its run results.
- **Not a review.** No scientific review, no reading of the packet's associated PR or its reviews, and no assessment of the analytic steps or imported sources.
- **The census.** It uses first-order mutants only. Equivalence readings are reasoned and spot-checked, not proved for every input. Independent verifiers re-derived the class counts and ran transformed-fixture, margin and source-guard probes; their scripts are retained.
- **The trace-side checks.** U1 settles root completeness only for the three fixtures. U3, U4 and U7 are randomized checks of identities. U5 checks bookkeeping on synthetic configurations, not on fields.
- **No edits.** Nothing in the packet, Math-, the graph or any status field was changed. Findings are for the packet's authors to accept or decline.

## Retained evidence

The evidence is retained on branch `claude/help-other-claude-models-829990` at commit [`413a2cc8`](https://github.com/d6g8k5htny-coder/main/tree/413a2cc8bac26c3cd68370c92d6c74995874e6a7/recovery/session_013tN78b3Q38WdttFyiJ6KVg/urb_trace), under `recovery/session_013tN78b3Q38WdttFyiJ6KVg/urb_trace/`.
- **`urb_trace.tar.xz`** (SHA256 `3e815075c4d71ac0ac27543a0d1dc345ca0f9ade3192db4ca79aab1534c84652`) holds:
  - the run logs of the four README commands, with and without the Git shim, and the shim itself;
  - `census.json` (one record per mutant, plus the baseline records) and `census_urb.py`;
  - `classify2_urb.py`, which defines the survivor classes above, and its output;
  - `trace_checks_urb.py`, `packet_code_offfixture.py` and `proposed_control_demo.py`, with their outputs;
  - the probe scripts and outputs of four independent verifier sub-agents of this session, which reviewed a draft of this document. Their corrections are applied here.
- **`CONTENTS_SHA256`** lists every file in the archive.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
