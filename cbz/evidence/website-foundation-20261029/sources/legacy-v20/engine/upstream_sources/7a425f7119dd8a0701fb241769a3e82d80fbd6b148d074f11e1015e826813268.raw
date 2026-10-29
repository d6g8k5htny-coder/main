# Nonauthor analytic review: P15 overlapping blocks (Math-#118) and sharp even tail with weighted loads (Math-#122)

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, `STATUS`,
`PROOF_INDEX`, `GRAPH` or any author source. It records verdicts on two author-side P15 candidates and one
reviewer remark (R1). Integration is a separate act.

## Objects

| Field | Value |
|---|---|
| #118 object | OA-P15-BIPARTITE-OVERLAP-20260928-v1 and OA-P15-EVEN-READ-TWO-20260928-v1 (OpenAI / ChatGPT) |
| #118 head | `chatgpt/p15-overlap-20260928` / `23b7558943f69a7f479251566b19ba58e75cfc26` ([Math-#118](https://github.com/d6g8k5htny-coder/Math-/pull/118)) |
| #118 `PROOF.md` | Git blob `8ad826ecb374cc733e3a547bdba0ba63cad10c08`, 13835 B, SHA256 `21ce7a29ac0971919c0602f556f1aeba48bb14fffd4193864b3f482185bc1698` |
| #118 `EVEN_CAPACITY_EXTENSION.md` | Git blob `ce21e424433100d0307c49bc15cccc748f9438fc`, 7620 B, SHA256 `d28aa700ba83ca139b7b29b6ff06cc39c725470b710af10728867ec10c8ca985` |
| #122 object | OA-P15-TAIL-LOAD-20260928-v1 (OpenAI / ChatGPT) |
| #122 head | `chatgpt/p15-tail-load-20260928` / `245f3d77c9d38f49d13affb9563a266eba92a47b` ([Math-#122](https://github.com/d6g8k5htny-coder/Math-/pull/122)) |
| #122 `PROOF.md` | Git blob `8187908dfd220cb833b94f82c451ef4c0e45e367`, 12828 B, SHA256 `fbca462d4988a06754ded9415ceea6bb1385be5535656d643bdd48edbbee8a20` |
| Requests | [#118 comment 5878164407](https://github.com/d6g8k5htny-coder/Math-/pull/118#issuecomment-5878164407), [#122 comment 5879215663](https://github.com/d6g8k5htny-coder/Math-/pull/122#issuecomment-5879215663) |
| Prior review | OpenAI same-provider review of #118, [5344833684](https://github.com/d6g8k5htny-coder/Math-/pull/118#pullrequestreview-5344833684): no blocker. #122 is its additive follow-up. |

## Provenance

| Field | Value |
|---|---|
| Reviewer | Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3` |
| Relation | Different provider from the OpenAI author. Same GitHub account as every lane, so no organizational independence is claimed. |
| Exposure | I read all three proof files in full, both READMEs, RECONNAISSANCE and SOURCE_MAP files, and the OpenAI review above. I did not write or earlier review `frontiers/full_price_20260924/PROOF.md` or any P15 source. |
| Author code | Replayed from `git archive` of each head: #118 `run_validation.py` passes 29 tests and rejects 10 mutants per mode, with identical output pairs. #122 `verify.py --output <new dir>` passes 15 tests and rejects 7 mutants per mode, with 8 identical pairs. |
| Independent checks | `p15_exact_check.py` is my own suite. Its colouring oracle is a plain backtracking search over colour assignments of the original coordinates, not the authors' constructions. It covers 34 checks and 7 mutants. Finite checks do not prove the continuum steps argued below. |

## Verdicts

### Math-#118

| Interface | Verdict |
|---|---|
| (O1)–(O2) Theorem O1: exact colouring on bipartite overlap with arbitrary capacities | **ACCEPT** |
| (O3)–(O4) Corollary O2: active full blocks form a cover, and their inclusion-minimal distinct members are exactly the minimal `K`-obstructions (an active block containing another is not minimal; see §2); `χ_D(X) = K+1` | **ACCEPT** |
| (O5) separate-concavity price-to-hazard transfer | **ACCEPT** |
| (O6)–(O7) Lemma O3: uniform tail `<= q_5` for all `a >= 1` | **ACCEPT** |
| (O8)–(O10) two-sided Cauchy–Schwarz and sum | **ACCEPT** |
| **Theorem O4 (O11)**: `ρ_bip < 3/4` | **ACCEPT** |
| (O12) enclosure and §7 rational certificate | **ACCEPT**. My rigorous enclosure has width about `3·10^-47` and lies inside (O12). |
| §8 doubled triangle and overlap counterexamples | **ACCEPT**. Both were reproduced by the oracle and by exact measure. |
| Even extension (E1)–(E2) Theorem E1: read-two, even capacities, nonbipartite | **ACCEPT** |
| (E3) Lemma E2: read-`q` entropy inequality, direction | **ACCEPT** |
| (E5)–(E6) `2/log(512/25) < 2/3` | **ACCEPT** |

### Math-#122

| Interface | Verdict |
|---|---|
| (T5)–(T6) cover eligibility in the even/read-two class | **ACCEPT** (same argument as E1) |
| (T7)–(T8) local transfer | **ACCEPT** |
| (T9)–(T11) weighted entropy with actual coordinate loads (sum, not max) | **ACCEPT** |
| **Theorem 2 (T3)–(T4)** instance-adaptive load, once a genuine cover is proved | **ACCEPT** |
| (T12)–(T13) sharp reference maximum `q_9`, attained only at `a=2, K=4` | **ACCEPT** |
| (T14)–(T16) interval directions and margins; `ρ_9 < 1/2` | **ACCEPT** |
| **Theorem 1 (T1)** and (T2) enclosure | **ACCEPT**. My rigorous enclosure (width about `10^-48`) lies inside (T2). |
| Zero or infinite `H_D`; duplicate and superset removal; the active/inactive distinction | **ACCEPT** |

No defect was found in either PR. Remark R1 below slightly extends #122's factor to the bipartite class of #118.

---

## §1 — Colouring: (O2), (E2), (T5)

**Bipartite, arbitrary capacities.** Each original coordinate of `U` becomes exactly one edge: shared coordinates
join the left and right blocks, and private ones join a private degree-one vertex. So the degree of block `i` is
exactly `|U ∩ X_i|`. Splitting `i` into `a_i` vertices gives degree `<= ⌈|U ∩ X_i|/a_i⌉ <= k`. König's theorem,
proved in the source by regularization and Hall, gives `k` colours. Each split vertex sees at most one edge of a
colour, so each colour meets `X_i` in at most `a_i` coordinates. The source's regularization is sound: the total
deficits of the two sides are equal, so a positive deficit on one side always has a partner on the other.

**Read-two, even capacities.** Split `i` into `a_i/2` vertices, so each split degree is `<= 2k`. Pair the odd
vertices; their degrees were `<= 2k-1`. Then take an Euler orientation, so in- and out-degree are both `<= k`, and
apply König to the out/in double. Each colour has at most one out-edge and one in-edge per split vertex. That means
at most `2 · (a_i/2) = a_i` per block, because no edge is a loop.

**Oracle checks.** The formula matches the backtracking oracle on:

- 720 random subsets of 120 random bipartite systems with capacities 1–3;
- 320 random subsets of 80 random read-two systems whose incidence graph is an odd cycle (3 or 5) with capacities
  2 or 4;
- the E4 triangle: capacity 2, blocks of 9, the 12 shared coordinates need 4 colours and all 15 need 5.

**Adversarial cases requested by the owner.**

- *Doubled triangle, capacity one.* The oracle gives `χ = 6` while the formula gives 4, and no block lies inside `U`.
  This confirms that §8 excludes it correctly, and that neither class covers it: it is not bipartite, and its
  capacities are odd.
- *Odd capacities on odd cycles in general.* The `odd-nonbipartite` mutant sets capacity one on the random
  odd-cycle systems, and the formula then fails. The evenness in (E1) and (T5) is genuinely needed, not a proof
  convenience.

## §2 — Obstruction and cover: (O4), (T6), inactive blocks, duplicates

For an inactive block (`d_i < K`), `|X_i| = a_i d_i + 1 <= a_i K`, so it can never exceed the palette. For an active
block, exceeding `a_i K` means containing all of `X_i`.

Brute force over all `2^10` subsets of a bipartite system confirms `O_4(D) = {U ⊇ X_0} ∪ {U ⊇ X_2}` and
`χ_D(X) = 5`. The system has two active capacity-one blocks and an **inactive demand-one block** `{4,5}` that meets
both. The inactive block's full set has `χ = 2 <= K`, so it is not an obstruction. The `charge-inactive` mutant,
which adds it to the obstruction family, fails.

Brute force also confirms two removal rules:

- dropping a duplicate block (`X_0 = X_1`) keeps the cover;
- keeping only the smaller of `X_0 ⊊ X_1` (capacities 1 and 2, both active) keeps the cover.

The `drop-superset-cover` mutant keeps the superset instead, and fails on `U = X_0`.

## §3 — Local transfer: (O5), (T7)–(T8)

With the other coordinates fixed, a decreasing event has `μ(A) = P_1 + (P_0 - P_1)(1 - p_v)` with `P_0 >= P_1 >= 0`.
This was checked exactly on 120 random decreasing events. So `F(t) = -log(A_0 + B_0 e^{-t_v})` has
`∂²F = -A_0B_0e^{-t}/(·)^2 <= 0`. `F >= 0` gives the chord bound `F(t) >= t_v F(t_v = 1)`, and iterating gives
`F(t) >= H_A Π t_v`.

Replacing `p_v` by `1 - e^{-φ(p_v)} <= p_v` only increases the probability of a decreasing event, so the direction
is right. The chord bound was also checked on 60 random capacity events in 60-digit decimal arithmetic. That check
is labelled numerical; it is not exact.

The transfer is applied to the capacity event `A_i = {|U ∩ X_i| <= a_i}`, with reference hazard `H_i`. It is **not**
applied to the containment event. Both sources say this explicitly.

## §4 — Hazard aggregation: (O8)–(O9), (E3), (T9)–(T11)

- **Two-sided Cauchy–Schwarz.** `D ⊆ A_L ∩ A_R`, and `P(A ∩ B)^2 <= P(A)P(B)`. Same-side independence holds only
  because same-side blocks are disjoint. Exact checks on 150 random rational product measures confirm the
  factorization and `μ(D)^2 <= Π μ(A_i)`. The source's overlap example, `5/8 > 9/16` with `(5/8)^2 <= (3/4)^2`, is
  reproduced exactly.
- **Entropy (E3)/(T11).** `KL(ν||μ) = H_D` for `ν = μ(·|D)`. Support gives `KL(ν_S||μ_S) >= h_i`. The chain rule
  with a product `μ`, plus Jensen for KL's first argument, bounds each marginal contribution by the full conditional
  one `b_v`. Summing with weights gives `Σ w_i h_i <= (max_v Σ_{i∋v} w_i) H_D`. The inequality directions are
  correct.
- **Exact exponentiated checks.** `Π μ(A_i) >= μ(D)^q` held on 150 random product measures with **arbitrary**
  (non-monotone) events on read-2 and read-3 incidences. So did `Π μ(A_i)^{n_i} >= μ(D)^{max_v Σ n_i}`. The `q-one`
  mutant (independence-style `q = 1`) fails. The `max-load` mutant fails on the source's counterexample: two copies
  of one fair-coin event, where equality `1/4 = 1/4` holds only with the summed load 2.

## §5 — Reference tails: (O6)–(O7), (E5), (T12)–(T16)

All rational certificates were recomputed exactly:

- `1957/720 < e < 31967/11760 < 87/32 < 11/4`;
- `s = (e+4)/(5e) < 1/2` from `e > 8/3`, with equality exactly at `8/3`;
- `5^a 2^{-(4a+1)} = (1/2)(5/16)^a`, and the chains `<= 25/512` for `a >= 2` and `<= 625/131072` for `a >= 4`;
- `q_5 > (28/3)(4/11)^5`, with the margin `2601239/247374336`;
- `Σ_{j<=6}(7/3)^j/j! - 307/32 = 129055/209952`;
- `512/25 - (87/32)^3 = 314641/819200`;
- the `q_5` and `q_9` identities as polynomial identities in `e`;
- both (T15) margins, digit for digit.

**Interval directions (T15).** `P(z) = 36z^2 - 63z + 28` increases on `[l,u]`, so `P(l)/u^9 < q_9 < P(u)/l^9`. Both
published bounds bracket a rigorous 40-term enclosure of `q_9`. The `interval-swap` mutant fails.

**Enclosures.** I built independent rigorous enclosures of `ρ_bip` and `ρ_9`. `e` comes from the series with the
`1/(N!N)` tail. The log uses binary reduction and the atanh series with its remainder, after outward rounding. The
enclosures have widths of about `3·10^-47` and `10^-48` and lie strictly inside (O12) and (T2).

**Sharpness of the even tail (T13).** Rigorous bounds with `p_* ∈ (1 - 1/e_lo, 1 - 1/e_hi)` show
`P(Bin(aK+1, p_*) <= a) < q_9` for every even `a <= 10`, `4 <= K <= 8`, except `(a, K) = (2, 4)`. The general proof
is the source's: monotonicity in `n` for `a = 2`, and Markov for `a >= 4`.

Capacity one gives `q_5 > q_9`, so the uniform `q_9` reference bound does not reach active capacity-one blocks. In
this coarse read-two assembly they need the `h_5` local reference input instead. That is a failure of a
reference-tail bound, **not** a cover-factor impossibility. For example, a single active capacity-one block is
read-one, so its load is `1/h_5 = ρ_bip/2 < 0.36508 < ρ_9` by the enclosures (O12) and (T2).

### R1 (reviewer remark, not a source claim): the `q_9` reference tail covers every capacity `>= 2`, odd included

In the **bipartite** class of #118, which allows arbitrary capacities, suppose every **active** block has
`a_i >= 2`. Then:

    covercost_c(O_K(D)) <= min(1, ρ_9 H_D),    ρ_9 < 1/2.

Inactive capacity-one blocks are allowed.

*Proof.* (O4), (O5) and (O9) are unchanged. It remains to show `H_i >= -log q_9` for every `a_i >= 2` and `K >= 4`:

- `a = 2`: monotonicity in `n`.
- `a >= 4` of either parity: Markov gives `(1/2)(5/16)^a <= 625/131072`.
- `a = 3`: this case needs a separate step. **Markov's bound `125/8192` exceeds `q_9`**. But `n = 3K+1 >= 13`, and
  monotonicity in `n` gives `P(Bin(3K+1, p_*) <= 3) <= P(Bin(13, p_*) <= 3) ≈ 0.00385 < q_9 ≈ 0.01515`. That last
  inequality is certified exactly by `P(Bin(13, 1 - 1/e_lo) <= 3) < P(Bin(9, 1 - 1/e_hi) <= 2)` with 40-term
  bounds `e_lo < e < e_hi`.

The `tail-a3` mutant, which replaces the direct tail by Markov, fails. ∎

So in this coarse read-two assembly, only **active capacity-one** blocks need the `h_5` reference input, which
yields the uniform factor `ρ_bip`. No necessity or optimality of `ρ_bip` as a cover factor is claimed. When
capacity-one blocks sit alongside larger ones, #122's Theorem 2 already gives the instance-specific load: a
coordinate shared by a capacity-one and a capacity-`>=2` active block carries `1/h_5 + 1/(-log q_9)`. R1 is offered
to the author and changes no source.

## §6 — #122 Theorem 2 edge cases

- **Cover eligibility.** Theorem 2 presupposes a genuine cover and `H_i > 0`. `H_i > 0` holds exactly when
  `|X_i| > a_i`, which is automatic for active blocks.
- **`H_D = ∞`.** The unit-price empty generator applies.
- **`H_D = 0`.** Then `μ(A_i) = 1`. That forces at least `|X_i| - a_i >= 1` coordinates with `p_v = 0`, hence
  `c_v = 0`, so the full-block price is zero, consistent with (T8).
- **Loads.** Loads are per original coordinate, summed. Duplicate removal halves a doubled load, and the cover is
  unchanged (§2).

## Checks run

```
python3 -B -S run_validation.py                       # #118 author runner at 23b7558 (git archive): 29 tests, 10 mutants
python3 -B -S verify.py --output <new dir>            # #122 author runner at 245f3d7 (git archive): 15 tests, 7 mutants
python3 -B -S p15_exact_check.py                      # rc 0, stdout == RESULTS.json
python3 -B -O -S p15_exact_check.py                   # rc 0, byte-identical
python3 -B -S p15_exact_check.py --mutant M           # rc 1 for each of:
    odd-nonbipartite charge-inactive q-one max-load interval-swap tail-a3 drop-superset-cover
```

Python standard library only. The workflow `.github/workflows/p15-overlap-tail-review.yml` verifies the packet tree
against `SOURCE_FILES.json`, replays both modes and requires every mutant to be rejected.

## Not established here

- Optimality of any cover factor. Both sources disclaim it; the tail in (T13) is sharp as a reference tail only.
- Arbitrary macro-clutters, arbitrary read-two with odd capacities, unrestricted downsets, or any governing P15
  status.
- The old `0.84548` sharp factor of `full_price_20260924` is a different class (maximal demand two, disjoint
  blocks). No comparison is asserted.
- Organizational independence: all lanes share one GitHub account.

## Revision history

- **v1**, reviewing #118 at `23b7558` and #122 at `245f3d7`: initial packet.
- **v2**, responding to the OpenAI review
  [5345318332](https://github.com/d6g8k5htny-coder/Math-/pull/123#pullrequestreview-5345318332). Wording only;
  no verdict, check or source changes.
  - The (O3)–(O4) row now says the inclusion-minimal distinct active full blocks are the minimal obstructions, as
    the source does. The reviewer's `X_0 ⊂ X_1` example is the §2 superset case.
  - §5 and R1 no longer say `ρ_9` "cannot cover" active capacity-one blocks or that `ρ_bip` is "forced". A failed
    reference-tail bound is not a cover-factor impossibility; a single active capacity-one block has load
    `1/h_5 < ρ_9`.
