# DYN-CLAIM-SELECTION-01 — source-bound claim/selection transfer diagnostic

**Experimental engineering, NOT A RESULT.** Scientific effect **NONE**. Independence credit **0**, OBL **OPEN**, no flags changed. This is an *opt-in input diagnostic*, not a scientific-status register, a theorem prover, an adopted graph edge, a required gate, or an automatic validator of user-provided mathematical premises.

## Why the mathematical object matters to the architecture

A research dependency cannot be justified merely because two asymptotic expressions contain the same power of `r`. Their **observable, event, conditioning law, selection and count region** must agree. The original RN interface explicitly warns that the cubic event probability bound `P(E_r) <= C r^3` does *not* by itself establish the cubic event-weighted count bound `E[N_r 1_{E_r}] <= C' r^3`. This is a mathematical claim-identity problem that should stay visible when a research architecture displays source/implementation relationships.

The frozen, **existing** mathematics is in `d6g8k5htny-coder/Math-@9f3fb6918b28e6378c034ebc19e76d432f5391cb`, directory `frontiers/three_fronts_20260924/`:

| Original artifact | Git blob | Boundary |
|---|---|---|
| `RN_COUNT_INTERFACE.md` | `371fd6d17920f2eb3b1c5ec30297acf6daf1d385` | Section 2: Hölder N1/N2 and exact conditional identity N3; count integral not evaluated |
| `frontier_math.py` | `d64e119adad0dfb3da5ff033a81234129fccb253` | `holder_exponent` and `rare_count`, finite exact models; not an RN numerical certificate |
| `test_frontiers.py` | `283a469ab831daae537c7f37b36be9523750d1cd` | Tests p=2, p=4, growing moments and a rare-count counterexample |
| `RESULTS.json` | `f0cd87a8ca4f07164d73008667dfbe7e9aa3a016` | `rn.triple_integral_evaluated=false`, `scientific_acceptance=false` |

The existing [two-node source trace on board #539](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6374115953) records this source→program→tests chain. This addition **consumes** that existing trace; it does not claim to independently review/replay the original mathematical proof or add a `math.rn-count-interface` binding. The original graph classification remains `AUTHOR_SIDE_CANDIDATE` at the pinned graph source, with no modification.

## Exact algebra (no generalized or new theorem)

This tool uses only the original source's specific **cubic** setting. Given separately declared estimates

- `P(E_r) <= C r^3`, and
- `E[N_r^p] <= A r^(-beta)`, where integer `2 <= p <= 64` and `beta >= 0`,

the cited finite Hölder algebra gives the exact rational power `gamma=(3(p-1)-beta)/p` for the **event-weighted** count `E[N_r 1_{E_r}]`. Under a separately declared **uniform conditional count-mean** premise, the source's identity `E[N_r 1_{E_r}] = P(E_r) E[N_r|E_r]` gives `gamma=3`. The constants, source hypotheses and validity of any premise are **not** verified by this tool. Rate comparison uses the small-radius convention `0 < r <= 1`: `gamma >= requested_power` is only an algebraic exponent comparison under those asserted premises.

| Declared route | Diagnostic derived rate | Requested cubic count-rate assessment |
|---|---|---|
| Event probability *alone* | none | **Not implied** |
| Uniform second moment, p=2, beta=0 | `3/2` | **Insufficient** |
| Uniform fourth moment, p=4, beta=0 | `9/4` | **Insufficient** |
| Fourth moment grows as `r^-2` | `7/4` | **Insufficient** |
| Event probability plus **actual uniform conditional count mean** | `3` | Exponent comparison meets target **if** both assertions hold |

A source-provided rare-count counterexample has `r=2^-n`, `P(E_r)=r^3`, and `N_r=n` on the event, giving `E[N_r 1_E]/r^3=n`. Finite test values demonstrate the bookkeeping but do not constitute a new all-small-`r` proof or Gaussian model calculation.

## Interface and deliberate refusal boundary

The opt-in command reads **only bounded standard input** (at most 16,384 bytes) and emits deterministic JSON. It performs no disk writes, network activity, remote Git reads, Math module execution, or mutations. The canonical proposed command is:

```sh
python -B -S tools/claim_selection_transfer.py <<'JSON'
{"schema":"claim-selection-transfer.v1","target":{"identity":{"law":"Q_r_weighted_palm","conditioning":"six_pin","event":"E_r","count":"N_B","region":"fixed_B","selection":"event_weighted"},"observable":"event_weighted_count_expectation","requested_power":"3"},"event_bound":{"identity":{"law":"Q_r_weighted_palm","conditioning":"six_pin","event":"E_r","count":"N_B","region":"fixed_B","selection":"event_weighted"},"rate":"r^3"},"count_information":{"kind":"unconditional_p_moment","identity":{"law":"Q_r_weighted_palm","conditioning":"six_pin","event":"E_r","count":"N_B","region":"fixed_B","selection":"event_weighted"},"order":2,"beta":"0"}}
JSON
```

The six identity fields are exact **caller-declared strings**; there is no fuzzy compatibility inference. Only target observable `event_weighted_count_expectation` is supported. The **unconditional whole count** `E N_r`, a different population, conditioning, selected law, event, region, or statistic is refused, not patched with a guessed support relation. The count-information choice is precisely `none`, `unconditional_p_moment` or `uniform_conditional_mean`; a missing count premise cannot silently become a positive theorem. Unknown fields, duplicate JSON keys, float and boolean numerics, malformed/oversize input and unsupported powers yield exit code `2`, zero stdout and the fixed opaque error `CLAIM_SELECTION_TRANSFER_REFUSED` on stderr. No offending private values are echoed.

**Output always reports** `premises_independently_verified=false`, `source_identity_runtime_verified=false`, `scientific_acceptance=false`, `scientific_effect=NONE`, `graph_binding_created=false`, and the immutable source references above. In particular `exponent_comparison=meets_declared_target` is *not* a grade, proof, independent review, adopted graph relation, or discharge of RN's still-open numerical/continuum obligations.

## Verification and promotion firewall

The isolated engineering tests, `tests/test_architecture_claim_selection_transfer.py`, cover event-only refusal, exact finite p powers, growing moments, event/selection/population/scope mismatch, whole-count mismatch, strict input parsing, bounded CLI, source flag, deterministic output and a synthetic rare-count witness. Author-side isolated local execution ran **21/21 methods with normal `-B -S` and 21/21 with `-B -O -S`**, no skips/errors, before publication. A deliberately weakened exponent formula is checked as a negative control separately; any result is recorded on the source PR rather than implied here. The test's `test_architecture_*.py` name makes it eligible for **existing** architecture test discovery; no CI workflow or required-check is changed.

**Integration is separate.** A different source-exposed engineering reviewer must confirm the typed premises, exact N1/N2/N3 algebra, negative controls, source pins, limitations and CLI behavior. A separate integrator then reconciles current default-branch composition, exact-head tested checks, outstanding review findings and protected merge rules. Only later, through a separately reviewed architecture decision, might a consumer opt into this experimental diagnostic. It **never** becomes a mathematical acceptance gate automatically.

[Architecture discussion #275](https://github.com/d6g8k5htny-coder/main/issues/275) · [Current coordination #539](https://github.com/d6g8k5htny-coder/main/issues/539) · [Complete owner mandate](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OWNER_MANDATE_20261015.md) · [Standing delegation](https://github.com/d6g8k5htny-coder/main/blob/main/governance/OP-AUTONOMY-20260923-v2.1.md). Link delivery is not agent acknowledgement.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN · NOT A RESULT
