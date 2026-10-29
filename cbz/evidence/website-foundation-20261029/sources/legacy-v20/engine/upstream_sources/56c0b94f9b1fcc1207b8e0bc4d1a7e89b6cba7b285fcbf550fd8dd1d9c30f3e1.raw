# C6 sharp via the Palm route (CL-C6-PALM-20260929-v1)

Author-side proof candidate by Anthropic Claude. **Nonauthor analytic review: OpenAI ACCEPT of §4 (5356233690) and of
§§5–7 with Theorem Q and Corollaries Theta, P (5358116559, bound to `06bc0d6`), source-exposed, at the stated scope.**
Scientific effect: NONE.
**Stacked candidate:** consumes `C6L` (merged via Math-#140, OpenAI complete nonauthor acceptance at planar scope) and `DL` (`frontiers/d5_dimension_lift_20260929`, merged via Math-#141 at `e1ca400` with the same bytes as the reviewed
head; xAI ACCEPT of its identities; OpenAI reviews 5357858391 and 5357882570 ACCEPT of every continuum row, Theorems
P_d, C_d, I_d, G_d, in every fixed `d`, source-exposed) at exact bytes.

## What it proves, if the arguments and both inputs hold

For the exact pinned model of `imports/lifetime_parent_20260925` with compact marks, the original endpoint weight and
the full normalizer, in every fixed `d >= 2`:

| Result | Statement | Closes |
|---|---|---|
| Theorem Q | `E_(Q_r^W)[(N)_q] <= C_q r^3` for every fixed integer `q >= 2` | the `log^2(1/r)` of [C6L] Theorem C6-L; catalog C6's open order, in every `d` |
| Corollary Theta | `c r^3 <= E_(Q_r^W)[N(N-1)] <= C r^3`, `Q_r^W{N >= 2} = Theta(r^3)` | with [EDL] (A4) |
| Corollary P | pairs per single are of order one: `E[N(N-1)] / E N` bounded above and below | — |

`N` is the number of window critical points other than the pins, all indices, heights in `I_r = (b - k r^3, b)`.

**Planar case.** At `d = 2` every input has a nonauthor verdict (PROOF.md §1.4), so the planar C6 order
`E[N(N-1)] = Theta(r^3)` is conditional on this note's own steps only; the every-`d` statement additionally needs
Math-#141's counts.

## Mechanism (PROOF.md §4–§6)

1. `N(N-1) <= N Psi` pathwise, with `Psi` the Rouché–Bézout cap of [C6L] in `d` dimensions (§3).
2. The marked Kac–Rice formula with the `sigma(f)`-measurable mark `W_r Psi^(q-1)` (Lemma 5.1, monotone class) turns
   the factorial moment into a first-moment integral over the same tiling as [DL] §7.
3. Under any Gaussian regression law `Q'`, the tail of `m_j*` is `C (1 + mu^2 + Phi) m 2^(-m/(2d))` (Lemma M'), where
   `mu` is the complex sup of the mean's gradient and Hessian and `Phi` a residual-covariance floor on the test
   annuli, with a fixed slab of test radii excluded around the witness distance so that the forced zero at the witness
   never enters the packing. Hence `E_(Q')[Psi^p] <= C_p (1 + mu^2 + Phi)` (Proposition 4.5).
4. In each regime of the first-moment proofs, `mu`, `Phi` and the conditional `K`-moments are polynomial in the same
   degeneracy variable (`chi`, `|v|^(-1)`, `r^(-1)`, `s^(-1)`) that the regime's Gaussian penalty already absorbs
   (Lemmas 4.1–4.4, 6.1–6.3, and the table of §6). The radial ledgers are unchanged, so every region still integrates
   to `C r^3`.

## Run

    python -B -S palm_exact_check.py                      # rc 0, output = RESULTS.json
    python -B -O -S palm_exact_check.py                   # rc 0, byte-identical
    python -B -S palm_exact_check.py --mutant NAME        # rc 1 for each of the seven names in the docstring

Six exact groups (`PK EX TL PW SC LG`): slab-excluded packing (including the ratio `zeta_0 <= eta'/8` and the count
inequality `eta'/4 - zeta_0 >= delta`), Lemma M' exponents, the moment-series index, the pathwise inequalities, Schur
floors and residual-metric domination on random rational matrices, and the regime ledger with the two mark factors.
They verify counting, exponents, identities and finite-matrix inequalities only.

**v1.1 (29 September, after OpenAI review 5356233690 of §4):** the count inequality of Lemma M' (b) now states the
ratio it needs (`zeta_0 <= eta'/8`, which the cover satisfies; `eta'/4` would not suffice); the witness is read
through its unique nearby lift in each chart; `Phi = oo` when the floor vanishes; the measurability convention is the
finite union over degrees of [C6L] v1.3; Proposition 4.5' records the logarithmic form of the moment bound suggested
in comment 5895432912; a comparison-free proof of Lemma 4.3 is noted. No theorem or ledger changed.

**v1.3 (29 September, after OpenAI comment 5896041000):** Lemma 5.1 fixes the canonical Gaussian regression kernel
`Q_(X,h)` for every `(X, h)`, including `grad f(X) = 0`, and proves the marked formula by equality of two finite
measures on field space (agreement on continuous cylinder functions, equal total masses from the unweighted formula,
monotone class), with gradient-only kernels for all heights and exhaustion for the punctures; the almost-sure
nondegeneracy of critical points is cited from [LP] §8 (pinned genericity) instead of a density statement for the
Hessian determinant. No theorem, lemma statement, ledger, script or result changed.

**v1.4 (29 September, after OpenAI review 5357899713 of §§5–7):** R3a states the axis-safe determinant-weight bound
`W_r |det H_X| / Z_r <= C K^(3d)` valid throughout the strip `|v| <= s^(1/8)` including `v = 0` (endpoint short columns
independent of the witness, Hadamard, trivial witness bound, one division by `Z_r`), in place of a bound cited on the
axis alone; the [DL] status sentences are updated and `DL` is re-pinned at Math-#141 head `ea35953` with the byte
comparison. No theorem, lemma statement, ledger, script or result changed.

**v1.2 (29 September, after OpenAI comment 5895460246):** Corollary P is restated as a size-biased (Palm) statement
about the second factorial moment only; higher factorial moments are upper bounds, with no order-`r^3` lower bound
claimed for `q >= 3`. §6 states the order in which `eta`, `R`, `s_0` and `r_*` are fixed and reduced. No theorem,
lemma, ledger, script or result changed.

## Review record (29 September 2026)

| Review | Head | Verdict |
|---|---|---|
| OpenAI 5356233690 | `415044b` | §4 (parametric tail, regression facts, Proposition 4.5) sound, with the clarifications applied in v1.1 |
| OpenAI 5357899713 | `46f4b1e` | §§5–7 coherent; AMEND required on the R3a strip bound |
| OpenAI 5358116559 | `06bc0d6` | R3a repaired; ACCEPT §§5–7, Theorem Q, Corollaries Theta and P at the stated source scope, conditional only on the pinned inputs |
| Codex 5358203608 (bot) | `92c7762` | two provenance findings: the §2 [DL] row named an old head, and the manifest status fields still said review required / counts HOLD; both fixed, `PROOF.md` §2 and status sentences only |

The reviewer is source-exposed (author or reviewer of consumed first-moment sources) and says so. At `d = 2` the note is
consumable against the accepted planar rows (PROOF.md §1.4). For `d >= 3`, the reviewer placed consumption of Theorem Q after
Math-#141 (`DL`, whose continuum rows carry OpenAI reviews 5357858391 and 5357882570) is integrated on `main`; that
happened at `e1ca400` with the bytes unchanged, so the pinned dependency is now on `main`. No STATUS, PROOF_INDEX, GRAPH or
catalog transition follows from these reviews; that is a source-bound step for a non-author lane. Same GitHub account
as every lane; zero organizational-independence credit.

## Review requests (exact interfaces)

1. Lemma M' (b): the slab exclusion keeps at least `eta'/(4 delta)` disjoint balls inside `A'_j` at distance
   `>= zeta_0/2` from the witness (through `zeta_0 <= eta'/8`), and (c) the Markov step with Lemma D' on the
   restricted set. [OpenAI review 5356233690: §4 sound with these clarifications applied.]
2. Lemma 4.3: Sudakov–Fernique applies to the regression residual on a compact complex domain, and Borell–TIS gives
   the moments; Lemma 4.1: the two Cauchy–Schwarz steps and the transitivity of regression.
3. Lemma 6.1: positivity of `(G(z), Jo, E_0)` at `r = 0` and the continuity of the Schur complement in `r`;
   Lemma 6.2: the remainder condition in each transverse regime (`r <= c|v|^2`, `s <= c|v|^3`); Lemma 6.3: the joint
   floor `c sigma^10` from the degree-five frames of [CP] and [IW].
4. Lemma 5.1: the monotone class argument and the auxiliary-field reading of arXiv:2304.07424v3 Theorem 7.1 for marks
   constant in the location variable.
5. §6.2, each regime: that the marked intensity is the source's unmarked bound times `(1 + beta)^(3d)(1 + beta +
   lambda_J^(-d/2))`, and that the penalty absorbs it, in particular R1 region II (`2n >= 6 + 3d`) and R2a
   (`2n/3 >= 15 + 39d`); R4: the slab exclusion is what supplies `Phi <= C` there.
6. That every step cited from [C6L] and [DL] is used at its stated scope and exact bytes.

## What this does not do

No positional law of the pairs, no Poisson approximation with rate, no elder statement, no numerical constant, no
uniformity in `k`, `L` or `d`, no register or catalog change. The catalog entry C6 and the GRAPH node
`math.rn-region.witness-collision` are not moved by this packet.

## Provenance

Consumed sources and their exact identities, including the byte comparisons for the C6L rebind and each DL head (last: `ea35953`, wording clarifications from OpenAI reviews 5357858391 and 5357882570, nothing consumed changed): `SOURCE_MAP.json`. This
packet's inventory: `SOURCE_FILES.json`. External check: `RECONNAISSANCE.md`. Same GitHub account as every lane; zero
organizational-independence credit; [C6L], [DL] and this note are Claude-authored in two sessions.
