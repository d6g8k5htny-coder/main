# Replacement-bar occurrence: proof-to-code trace and mutation census

> **ENGINEERING TRACE, NOT A RESULT.** This document links one frozen Math- packet's proof steps to its finite exact controls. It records one execution of those controls, an independent mutation census, and trace-side exact checks for steps the controls take as given. It is not a proof or a scientific review. It does not claim that the code is equivalent to the proof, and it creates no graph node, edge or status. **Scientific effect NONE.** Credit 0. Author: Anthropic Claude, session `session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), as engineering trace author only. PROOF.md's authors are OpenAI/Codex root and `c51_excess_feasibility`, under Dylan Roy's delegation (L4). GLOBAL.md's author/coauthor is OpenAI/Codex `/root/c51_excess_feasibility`, under the delegation of `/root` and Dylan Roy (L4–6). GLOBAL states that it is same-provider author-side work with no independence credit.

## Source

- **Repository and packet.** `d6g8k5htny-coder/Math-` at commit `b29c4b769cac85437b409d152651d5cce4140015`. The packet is `frontiers/replacement_bar_occurrence_20261001/` (tree `bbfb5944108bd3185f9533369ad4dce68e44a983`). It was introduced and last changed by `a38449c30c44733dfb5cd128d11071fac2e0ea98`. Its objects are `OA-REPLACEMENT-BAR-OCCURRENCE-20261001-v1` (PROOF.md) and `OA-SELECTED-BAR-OCCURRENCE-GLOBAL-20261001-v1` (GLOBAL.md).
- **Files** (Git blob, bytes):

  | File | Blob | Bytes |
  |---|---|---:|
  | [`PROOF.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/PROOF.md) | `724258d33408f754ffc21848196e1fdd80c8bf6a` | 12,965 |
  | [`GLOBAL.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/GLOBAL.md) | `e3e4a7ada3abae335b75e7553cbd29f41403faac` | 17,598 |
  | [`check_exact.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py) | `6e8f263e2fa735db298612738d45f2a7f3504b51` | 4,083 |
  | [`test_exact.py`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/test_exact.py) | `4b997e33a49c8afcd714adb5b872e2aa14681b96` | 5,593 |
  | [`SOURCES.json`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/SOURCES.json) | `d764eb81ffe47f842750aa86f711a9bb87e6ae8b` | 4,438 |
  | [`README.md`](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/README.md) | `2e0a5b0b74c96a0c8c5cea63f9270e55e78aede1` | 2,167 |
  | `sources/c50/PROOF.md`, `ROOTS.md`, `SOURCES.json` | `9c79a672…`, `b27d9ac8…`, `210dcaca…` | 17,545; 11,969; 2,515 |

  The three `sources/c50/` files have the same Git blobs as the unique replacement-bar intensity packet's PROOF.md, ROOTS.md and SOURCES.json at `b29c4b76`. The proof-source join records the PROOF.md pair as dup-02. That packet is traced in its own document, `docs/retro/UNIQUE_REPLACEMENT_BAR_INTENSITY_PROOF_CODE_TRACE_20261026.md` (Draft #643).
- **Existing accounting.** The packet is a GB-8s unit with inventory only and no node IDs (`docs/retro/GB-8s.agent1.inventory.json`, at Math- `9f3fb691`, same tree). The post-cut proof-source join lists four implementation references for it: `check_exact.py`, `test_exact.py`, the packet workflow, and the main repository's `tests/test_reader_links.py` (`docs/retro/RETRO_PROOF_SOURCE_JOIN_20261025.md`, landed through #618). Neither maps its proof steps to its code.
- **Not read.** PROOF.md L272 says the analytic steps "require substantive nonauthor review". This trace did not look for or read any review, and it adds no review credit.

## What was run

Every command ran from a detached checkout of Math- `b29c4b76` with Python 3.11.15. No Git history is needed: the checker reads working-tree files only.

| Command | Result |
|---|---|
| `python3.11 -B -S …/test_exact.py -v` (the README command, with `python3` run as Python 3.11) | 18 tests, OK |
| `python3.11 -B -O -S …/test_exact.py -v` (likewise) | 18 tests, OK |
| `python3.11 -B -S …/check_exact.py` | exit 0, `"verified_sources": 10` |
| `python3.11 -B -O -S …/check_exact.py` | exit 0, same output |

The packet's own workflow, `.github/workflows/replacement-bar-occurrence.yml`, runs the two README commands on GitHub Actions. This trace did not read its run results.

## Proof step → finite control

"What the control checks" lists only conditions that a control asserts on exact rational inputs. The guards are explicit `ValueError` conditionals, so they stay active under `-O`. Four of the 18 tests call no function of `check_exact.py`; they assert arithmetic on literals and are listed as such. Analytic steps have no finite control, and the packet says so itself (PROOF.md L268–272, README).

| Proof step | Control (function; test) | What the control checks | Not checked by any control |
|---|---|---|---|
| PROOF §1 L17–31: D_h, A_h, N_h and the mark V_h; N_h counts maxima, not rejected saddles | — | — | **Not coded.** No function computes D_h or N_h from a configuration. R9 checks the bookkeeping of (O8) on random finite configurations |
| PROOF (O1) L33–37: U50's first moment L² C_U(Φ) h⁵ + o(h⁵) | — | — | **Source interface** (U50 §§4–7). The URB trace maps U50's code and lists this limit as analytic, with no finite control there either |
| PROOF L65–72 and SOURCES.json: U50's PROOF.md, ROOTS.md and source map are reproduced unchanged under `sources/c50` from `b5bf0287`; the seven underlying sources are pinned at `044d42d5` and read in place at their original paths | `verify_sources`, `verify_entry`, `contained_file` ([check_exact.py L15–56](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py#L15-L56)); tests L11, L106, L112 | The manifest's SHA-256 equals the pinned constant (L7, L48); ten entries with distinct keys; for each entry, working-tree bytes, SHA-256 and Git blob. Negative tests: wrong manifest digest; wrong byte type, SHA-256, blob; `../x`; a symlink | **The historical binding.** Each manifest entry names a `commit` and `path`, but the code never reads them. It checks only the working-tree file at `local_path`: a copy for the three C50 entries, the original path for the seven. R10 confirms all ten at their declared commits (RF-4) |
| PROOF Theorem O (O2)–(O6) L44–63 | `count_statistics` tests L25, L32, L40 | Exact finite counterexamples: a first moment alone does not fix P(N > 0); a small P(N ≥ 2) does not control E[N; N ≥ 2]; a small E[N; N ≥ 2] does not control E[N(N − 1)] | The theorem is analytic |
| PROOF §3 (O7) L96–119, from GLOBAL Theorem G | — | — | **Analytic.** See the GLOBAL rows below |
| PROOF §4 (O8) L128–137: N_h H_h = Σ_(rejected pairs) H_h/D_h(M) with the global mark H_h = 1{N_h ≥ 2} | — | — | **Not coded.** `mark_statistics` computes a different statistic, over a field law. R9 checks the bookkeeping of (O8) on random configurations |
| PROOF (O9)–(O11) L139–202: marked E2 identity, physical disintegration, dominated convergence, [ELDER S9] tail | — | — | **Analytic and source interfaces** |
| PROOF (O12) L206–219: r dr · r³ = r⁴ dr; the factor L² | tests L64, L70 (literals only) | — (they assert 1 + 3 + 1 = 5, (3⁵ − 1)/5 = 242/5 and L²·c ≠ c on literals) | The substitution. R11 computes the closed form; the exponent count r·r³ = r⁴ is restated |
| PROOF (O13) L221–225: (n − 1)₊ ≤ n·1{n ≥ 2}, n − 1{n > 0} = (n − 1)₊, 2·1{n ≥ 2} ≤ n·1{n ≥ 2} | `count_statistics` ([L64–72](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py#L64-L72)); tests L14, L18, L102 | The five statistics (E N, P(N > 0), E(N − 1)₊, E[N; N ≥ 2], E[N(N − 1)]) of a three-point law; the three relations for point masses n = 0..11. Test L102 rejects a negative count, a bool key, a law summing to 1/2 and a negative probability; it is the only test of the L66 and L68 guards and alone kills 10 census mutants | — (R7 also checks n = 0..200 directly) |
| PROOF (O4)–(O5) L227–234: P(N > 0) = E N − E(N − 1)₊; TV distance = P(N ≥ 2 \| N > 0) ≤ E[N 1{N ≥ 2}]/(2P(N > 0)) | `count_statistics`; test L18 (e − p = x) | P(N > 0) = E N − E(N − 1)₊ for point masses | The total-variation identity and bound. R7 checks both on random finite laws |
| PROOF (O14) L236–245 and (O6): \|S − 1{N > 0}S/N\| ≤ ‖Φ‖∞ (N − 1)₊ | `mark_statistics` ([L75–90](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py#L75-L90)); tests L48, L54, L57 | E S, E[S/N; N > 0], P(N > 0) and E(N − 1)₊ for one three-field law and for the empty field; \|E S − E[S/N]\| ≤ E(N − 1)₊ for that law (‖Φ‖∞ = 1); for a single-bar law, E S/P(N > 0) = E[S/N]/P(N > 0) = 2/3 and E(N − 1)₊ = 0 | The pointwise bound for general ‖Φ‖∞. R8 checks it |
| GLOBAL §2 L56–99: the cubic, the shear, B_s, D_s, Δ, H, Σ, H_poly and its q² coefficient | — | — | **Not coded here.** R1 checks the sheared leading cubic H3. The URB trace covers the shear data, H/H_poly and its q² coefficient |
| GLOBAL Lemma 1 L101–140: under (1) and H ≠ 0, at most four critical points, all nondegenerate; M_0 the only one with negative-definite Hessian; (2), (3), pin determinants 3k(B_s ∓ 2s); at most one strict maximum for degree ≤ 3 | `cubic_hessian` ([L93–96](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py#L93-L96)); test L74; test L82 (literals only) | The Hessian at one point of one cubic with k = 1, and that its value at the midpoint of one segment is the average of the endpoint values. Test L82 asserts on literals that a quartic has two critical points with second derivative −8 | The k-dependence of `cubic_hessian` (RF-1). The rest of Lemma 1. Trace-side: R12 (the gradient factors as conic and Z·line in sheared coordinates), R2 (the elimination), R3 (for constructed cubics: the complete critical set, the pin determinants and typing, (3), and no second negative-definite root) and R4 (the final one-maximum step). Lemma 1's degenerate cases B_s = 0 or D_s = 0 are not checked |
| GLOBAL Lemma 2 L142–155: Δ ≠ 0 ⇒ \|∇H3(x)\| ≥ c\|x\|² | test L90 (literals only) | — (it asserts that x³ has a zero gradient at (0, 1)) | The elimination for B_s ≠ 0 (R5). The B_s = 0 branch (L151), which contains test L90's x³ example, is not checked. Compactness and homogeneity are analytic |
| GLOBAL Lemmas 3–4 L157–245: global C4 contact limit; Morse off the origin | — | — | **Analytic** (Gaussian regression, positive Fourier spectrum, the mesh argument) |
| GLOBAL Lemma 5 L249–293: the root-free annulus; budget C/R² + C/R + ε + Cδ < 3c/4 | `annulus_budget` ([L99–102](https://github.com/d6g8k5htny-coder/Math-/blob/b29c4b769cac85437b409d152651d5cce4140015/frontiers/replacement_bar_occurrence_20261001/check_exact.py#L99-L102)); test L96 | The budget's value at one point (29/128 < 3/4), and that the budget exceeds 1 at C = 3/5, R = 1, δ = ε = 1/1000. The test comment's claim that a budget without C/R would fall below 1 is not asserted | That (8)/\|x\|² is at most the budget on the annulus; R6 checks it. The choice order and the Taylor bound (8) itself are analytic |
| GLOBAL Theorem G L297–350 and §7: C² root continuation, Newton contraction, remote separation; insertion into U50 | — | — | **Analytic** |

## Trace-side exact checks (this document's, not the packet's)

These are in the retained `trace_checks_rbo.py` (standard library only; it does not import `check_exact.py`). **All passed.** They are finite computations on random or fixed rational data, exhaustive small ranges and random finite configurations, plus Git object reads (R10). They are not proofs of the analytic steps. Rows that hold by construction are labelled as such.

| ID | Statement checked |
|---|---|
| R0 | The trace's hand-coded ∇P equals exact finite differences of P (GLOBAL L61–62) at 300 random points (a 5-point stencil, exact to degree 4). R3 and R12 use the finite differences themselves, so they rest on P directly |
| R1 | GLOBAL L74–78: with u = X + aZ/(12k), H3 = 2ku³ + (B_s/2)uZ² + (D_s/3)Z³ at 300 random rational points |
| R2 | Lemma 1 (L111–113): substituting the line of (2) into 2D_s² times the conic gives exactly (12kD_s² + B_s³)u² + 2B_s²su + B_s s² − 3kD_s² (300 random points) |
| R3 | Lemma 1 on 300 random typed cubics with B_s, D_s ≠ 0, built to have two rational extra roots. Each is built from one extra root; the second comes from Vieta on R2's quadratic. Both extras and both pins are critical points of P in original coordinates, by exact finite differences of P. **The four points are the complete critical set** (over C). By R12, exact for each of these tuples and computed from P itself, every critical point is a pin or solves (2). With D_s ≠ 0 the line fixes Z from u, and 2D_s² times the conic is then R2's quadratic Q(u); this identity is also checked exactly on each tuple, at three values of u, since it has degree 2. u_2 is the Vieta partner of u_1, so the linear coefficient of Δ(u − u_1)(u − u_2) matches by construction; the constant coefficient Δu_1u_2 = B_s s² − 3kD_s² is checked, which is equivalent to Q(u_1) = 0. With Δ ≠ 0, Q has no other root, and u_1 ≠ u_2 (checked) makes the four points distinct. M_0 is a strict maximum and S_0 a saddle, with determinants 3k(B_s ∓ 2s). (3) holds at both extras, and neither has a negative-definite Hessian; saddles and local minima both occur. Hessians come from exact finite differences of P |
| R4 | Lemma 1's last step (L132–140): for random general bivariate cubics, (∇P(q) − ∇P(p))·v equals vᵀ Hess P(midpoint) v. The Hessian is affine, so the midpoint rule is exact for the integral. The Hessian formula is checked against finite differences |
| R5 | Lemma 2 (L145–153), B_s ≠ 0 branch only: with Z = 1, 2B_s²(6ku² + B_s/2) at u = −D_s/B_s equals Δ (300 random points). A second part, by construction: with B_s = −12kw² and D_s = 12kw³, Δ = 0 and the sheared gradient vanishes at u = w; each of 300 such cubics has an exact nonzero gradient zero of H3 in original coordinates. This re-checks only the shear, so it shows that Δ ≠ 0 cannot simply be dropped. The B_s = 0 branch (L151) is not checked |
| R6 | Lemma 5 (L272–288): on random rational R r ≤ \|x\| ≤ δ, the bound (8) divided by \|x\|² is at most C/R² + C/R + ε + Cδ. A second part, by construction: with R doubled until C/R² + C/R < c/4, Cδ = c/8 and ε = c/8, the total is below 3c/4. The order of choices, and Rr_i < δ for large i, are not modelled |
| R7 | (O13) for n = 0..200. On 300 random finite count laws, the total-variation distance of Law(N \| N > 0) from δ_1 equals P(N ≥ 2 \| N > 0) and is at most E[N 1{N ≥ 2}]/(2P(N > 0)) |
| R8 | (O14) pointwise for signed rational marks, N = 0..12, 500 draws |
| R9 | Bookkeeping of (O8) on 400 random finite planar configurations. Each has an actual partner per maximum, and §1's distance-window and gap rules at h = 1, with no birth interval B. N·1{N ≥ 2} equals the pair sum of 1{N ≥ 2}/D(M), and N equals the pair sum of 1/D(M); 332 configurations have N ≥ 2. (O8) holds for any pair rule once D(M) counts the same pairs that are summed, so R9 cannot detect a wrong selection rule. Only a mismatched denominator breaks it: dropping the partner exclusion from D alone does so on 218 |
| R10 | For each of the ten manifest entries, `git rev-parse <commit>:<path>` equals the manifest blob, and so does the working-tree file. All ten match (seven at `044d42d5`, three at `b5bf0287`). The manifest's SHA-256 is `320f64d3…`, the constant pinned at check_exact.py L7 |
| R11 | (O12), with the exponent count r·r³ = r⁴ taken as given: h⁵∫_{c_0}^{C_0} t⁴ dt = h⁵(C_0⁵ − c_0⁵)/5 by Boole's rule (exact to degree 5) at nine rational (c_0, C_0, h). The scaling step ∫_{c_0h}^{C_0h} = h⁵∫_{c_0}^{C_0} also passes, but it holds for any quadrature rule with affinely placed nodes, so it is not independent evidence |
| R12 | The step behind GLOBAL L105–109, which GLOBAL asserts without displaying the gradient. In sheared coordinates (u, Z), ∂P/∂u = 6k(u² − 1/4) + (B_s/2)Z² and ∂P/∂Z at fixed u = Z(s + B_s u + D_s Z), at 300 random parameter tuples. Both differences have degree at most 2 in each of u and Z, so vanishing on the 3 × 3 grid {0, 1, 2}² proves the identity for each tuple. The gradient is taken by exact finite differences of P. A control with D_s replaced by D_s + 1 fails on every tuple (`R12_wrong_Ds_rejected`) |

## Independent mutation census

"Independent" here means only that the census was built separately from the packet's own controls. It carries no institutional or model-family independence. The census applied first-order AST mutants to `check_exact.py`:
- operator, comparison and `and`/`or` swaps;
- dropping one operand of an `and`/`or`, and negating each `if` test;
- ±1 on integer literals, and dropping `not` or unary minus;
- turning each `raise` or `require` into `pass`.

Mutants whose unparsed source is identical are counted once. Each mutant ran `test_exact.py -v` (the README command) and `check_exact.py` in normal and `-O` mode, in a dedicated detached worktree of Math- at `b29c4b76`. A mode fails if either command exits nonzero, the `test_exact.py` output has no line beginning `OK`, or `check_exact.py` does not report ten verified sources. A mutant counts as killed only if both modes fail. The baseline and the `ast.unparse` baseline pass in both modes. No mutant gave mixed results, and the worktree was clean afterwards.

| File | Mutants | Killed | Survived |
|---|---:|---:|---:|
| `check_exact.py` | 191 | 146 | 45 |

Per function: `count_statistics` 43 of 48 killed, `mark_statistics` 25 of 29, `cubic_hessian` 19 of 20, `annulus_budget` 20 of 34, `contained_file` 9 of 23, `verify_entry` 10 of 13, `verify_sources` 14 of 17, `rational` 1 of 2, and `require` and `__main__` all killed.

**Reading this census.** Every arithmetic mutant in `count_statistics`, `mark_statistics` and `annulus_budget` is killed, because the tests pin their outputs to exact expected values. In `cubic_hessian` all but one are killed: `12*k` → `12/k` (L96) survives because its only test uses k = 1 (RF-1). The other 44 survivors are guards: 41 input guards and 3 `verify_sources` structure guards (L43, L52, L53). `proposed_controls_rbo.py` runs 18 short extra controls (K1–K11) against the original, which passes all of them, and against every survivor. They close 40 of the 45. A separate demonstration (K-BS1, K-BS2), not a control, documents the backslash literal (RF-3).

| Class | Lines | Count | Closed by | Reading |
|---|---|---:|---:|---|
| **`cubic_hessian` k placement** | L96 | 1 | 1 (K1) | **Finding RF-1.** The only test uses k = 1, so `12*k` → `12/k` passes. K1 compares it with the Hessian of P at k = 5/7 |
| `count_statistics` and `rational` input guards | L60 (1), L65 (4), L68 (1) | 6 | 5 (K2, K3, K7) | **RF-2.** No test passes a non-dict law, a float probability or a zero-probability atom. L65 dropping `bool(law)` is equivalent for built-in dicts: an empty dict fails the sum at L68. Only a dict subclass with an inconsistent `__bool__` or `__len__` separates them |
| `mark_statistics` input guards | L80 (3), L89 (1) | 4 | 4 (K4, K5, K6) | **RF-2.** No test passes a negative probability, a zero-probability field or a law that does not sum to 1 |
| `annulus_budget` parameter guard | L101 | 14 | 14 (K8, K8b) | **RF-2.** No test passes an invalid C, R, δ or ε, or the boundary values C = δ = ε = 0 |
| `contained_file` path guards | L16 (4), L18 (6), L25 (4) | 14 | 13 (K9–K9f, K11) | **RF-2 and RF-3.** The tests try only `../x`, which both L18 and L25 reject, and a symlink, which only the L24 guard catches. So no L16, L18 or L25 conjunct is ever decisive. The controls use a non-string path, `a/./b`, `a/../a/b`, an absolute path inside the root, a directory and a double backslash. K11 calls `contained_file` directly with a relative root: the function does not resolve its own root, so containment at L25 is the only guard that rejects. Through the packet's callers, which pass a resolved root, it is not reachable. L16 dropping `bool(relative)` is equivalent for built-in strings, since an empty path fails L18; only a str subclass with a false `__bool__` separates them |
| `verify_entry` byte-count guard | L33 | 3 | 3 (K10, K10b) | **RF-2.** The test's `bytes=True` fails both conjuncts. A float `3.0` or a wrong integer separates them |
| `verify_sources` structure guards | L43 (1), L52 (1), L53 (1) | 3 | 0 | L52 (ten entries) and L53 (distinct keys) are equivalent while the manifest digest is pinned at L7: only one manifest passes L48. L43 (symlinked checker) is not exercised |

### Findings (engineering only; none changes the proof)

- **RF-1 (low, control coverage).** `cubic_hessian` is tested only at k = 1, so its k placement is unconstrained. The URB packet has the same blind spot (its UF-1). One assertion at k ≠ 1 against the Hessian of P closes it.
- **RF-2 (low, control coverage).** The input guards of `rational`, `count_statistics`, `mark_statistics`, `annulus_budget`, `contained_file` and `verify_entry` leave 41 surviving mutants. That is out of 88 mutants inside their `require` calls; the other 47 are killed, by test L102 and the source negative tests among others. The proposed controls K2–K11 close 39 of the 41 survivors. The other two (L16, L65) are equivalent for built-in types.
- **RF-3 (low, correctness of a guard).** `contained_file`'s backslash test, `'\\\\' not in relative` (L20), uses a two-character literal. So a path with one backslash is accepted as canonical. K-BS1 shows this on the original, and K-BS2 reads the literal's length, 2, from the packet's own source. The OSB packet has the same literal; the URB packet's `'\\'` is correct.
- **RF-4 (low, source binding).** SOURCES.json declares a historical `commit` and `path` for every entry, but `check_exact.py` reads neither. It binds the working-tree files, through the pinned manifest digest and per-entry bytes, SHA-256 and blob. So the packet does not check two PROOF claims: that the `sources/c50` files are reproduced unchanged from `b5bf0287` (L65–67), and that the seven sources are pinned at `044d42d5` (L71–72). R10 checks both: all ten match. The URB packet, by contrast, reads Git history.
- **RF-5 (observation).** Four of the 18 tests (L64, L70, L82, L90) call no packet function. They document counterexamples and arithmetic on literals, and the census cannot see them. R11 computes the closed form h⁵(C_0⁵ − c_0⁵)/5 behind L70's second assertion; the exponent count of its first assertion is taken as given. R4 checks the degree-three step behind L82, and R5 the B_s ≠ 0 discriminant step related to L90, whose own x³ example lies in the unchecked B_s = 0 branch. L64's factor L² is plain arithmetic and has no trace-side check.
- **What holds.** All 18 tests pass in both modes. Test L102 exercises the count and probability-law guards of `count_statistics` (L66, L68). Every arithmetic mutant in `count_statistics`, `mark_statistics` and `annulus_budget` is killed, and all but one in `cubic_hessian`. The manifest digest, byte, SHA-256 and blob checks are each exercised by a negative test, and the symlink guard (L24) is decisive in one. All ten historical source pins match.

## Limits

- **One machine.** This trace ran the commands once locally, on one Python, without CI. The packet's own workflow runs them on GitHub Actions; this trace did not read its run results.
- **Not a review.** No scientific review, no search for the packet's reviews, and no assessment of any analytic step or of the imported sources. The analytic steps include Theorem O and (O7), (O9)–(O11), Lemma 2's compactness step, Lemma 5's bound (8) and choice order, Lemmas 3–4 and Theorem G.
- **The census.** It uses first-order mutants only. Equivalence readings are reasoned and spot-checked, not proved for every input.
- **The trace-side checks.** Most check the algebraic and counting steps on random rational data, so they are evidence, not proofs. R12 is exact for each sampled parameter tuple, but the tuples are random. R3 covers only constructed cubics with B_s, D_s ≠ 0 and two rational extras, not Lemma 1's degenerate cases. R5 covers only B_s ≠ 0. R9 checks bookkeeping on synthetic configurations, not on fields, and cannot detect a wrong selection rule.
- **No edits.** Nothing in the packet, Math-, the graph or any status field was changed. Findings are for the packet's authors to accept or decline.

## Retained evidence

The evidence is retained on branch `claude/help-other-claude-models-829990` at commit [`4c502b54`](https://github.com/d6g8k5htny-coder/main/tree/4c502b54cadf15b70f8cb9f45d894d853de9e9af/recovery/session_013tN78b3Q38WdttFyiJ6KVg/rbo_trace), under `recovery/session_013tN78b3Q38WdttFyiJ6KVg/rbo_trace/`.
- **`rbo_trace.tar.xz`** (SHA256 `0ace7794c836b7dcf62ef7892e8306250b1c5d06efe1fd8e20faa9128f27605a`) holds:
  - the run logs of the README's two commands and the output of `check_exact.py`, each in normal and `-O` mode;
  - `census.json` (one record per mutant, plus the two baseline records) and `census_rbo.py`;
  - `trace_checks_rbo.py` and `proposed_controls_rbo.py`, with their outputs;
  - the probe scripts, outputs and journals of the independent verifier sub-agents of this session, which reviewed drafts of this document in three rounds. Their corrections are applied here.
- **[`CONTENTS_SHA256`](https://github.com/d6g8k5htny-coder/main/blob/4c502b54cadf15b70f8cb9f45d894d853de9e9af/recovery/session_013tN78b3Q38WdttFyiJ6KVg/rbo_trace/CONTENTS_SHA256)** lists every file in the archive.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
