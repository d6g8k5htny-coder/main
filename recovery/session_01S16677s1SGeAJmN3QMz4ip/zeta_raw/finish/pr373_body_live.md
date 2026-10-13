## Change and purpose

**This PR adds one new directory, `experiments/zeta_fold_exploration/`, an exploratory packet. Scientific effect: NONE. No status or register change.**

Dylan asked this session to examine an external note he forwarded on 12 October 2026, about links between zeta zeros and the project's fold and short-bar law, and to bring in whatever is worth exploring. Pickup: [307/6116750096](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116750096).

The packet checks each link, records what is classical, what is wrong and what is useful, and keeps scripts and results so anyone can re-run them. The note is paraphrased claim by claim in `CLAIMS.md` and never quoted. The packet makes no claim of progress toward RH, and no bound on Λ beyond known results.

**What it finds:**
- **Folds (generic and known).**
  - Two de Bruijn–Newman zero collisions were computed to about 40 digits, at t ≈ −1.878 (zeros 9/10) and t ≈ −0.385 (zeros 34/35). Every collision found among the lowest 46 zeros is at t < 0.
  - Near a nondegenerate double zero, r² = 8τ(1 + O(τ)) and the antiderivative's lobe area is (|A|/6)r³(1 + O(τ)), the project's k = |F‴|/12 fold law.
  - That lobe is a finite superlevel H0 bar only for small τ on a fixed surrounding interval (Proposition 1(c)), not for arbitrary close pairs or at t = 0. The critical-value gap and an actual finite H0 bar are kept distinct.
- **Theta (a footnote).** The project's periodized normalizer is a Jacobi theta value. ζ appears only after a Mellin transform over the torus side, which the project holds fixed.
- **Statistics (useful for the project's own geometry-versus-measure question).**
  - Under the hypotheses of the project's transfer theorem ((D1)), the same cubic fold with r² pair repulsion gives a short-bar density exponent of 0 instead of −1/3.
    - These are theoretical expectations under stated conditions, not calibrated answers. For fixed-N CUE the full bar law is the unproved T-CUE target.
    - GUE behaviour of ζ zeros is conjectural. Pair correlation is neither a nearest-neighbour law nor a persistence-pair law.
  - The measurement uses the specified Hardy-Z primitive: the antiderivative of Z(t)/√(log(t/2π)) in the unfolded variable, with endpoint-born bars dropped. With its windows at heights 10⁴–10⁶ and the Hill estimator:
    - The finite-tail estimates are substantially above −1/3 and near 0. For t ∈ [10⁵, 10⁶] at cut 0.01 the estimate is −0.020 [−0.041, +0.001].
    - The sampling-only block-bootstrap intervals exclude −1/3 at the reported cutoffs.
    - This is exploratory evidence about that finite-window statistic. It does not establish a limiting density exponent or a calibrated rejection accounting for all errors.
    - It says nothing about the ξ-weighted primitive or a de Bruijn–Newman barcode.
  - The data were inspected before any prediction was frozen, so this is exploration, not confirmation.
- **Proposed next targets (§6, not claimed here):** T-CUE, a short-bar theorem for fixed-N CUE, and a frozen, pre-registered zeta comparison on fresh windows.

**Related, not taken:** [307/6116121839](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116121839) is an unclaimed OpenAI-lane proposal for polynomial-first heat-flow fold controls in `experiments/universality/`. This packet touches only its own directory. Prior art on the same antiderivative: Lagarias–Montague, arXiv:1106.4348. Slice B checked it against the primary passage (eqs. (1.2), (1.6), (1.9), Thm 2.1). Its theorem concerns zeros of the primitive, not the critical points used here.

Scope: mathematics and numerics, exploratory only. No other path changes.

## Current disposition

- **Head:** `0969b327`. **Draft.**
  - It is fix commit `479d2450` on the read head `70f3d993`, plus a base-only merge of main `940e4855` that leaves the packet unchanged.
  - The successor note is at [6119395474](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119395474).
- **Author:** Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip`. **No author merge.**
- **Claim:** [307/6116750096](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116750096), lease extended to 12:00Z on 12 October ([307/6118626386](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6118626386)).
- **Nonauthor reads (README §9 slices):**
  - **A and D:** OpenAI dot/Codex `/root/review_zeta_fold_mathematics`, **AMEND** at `70f3d993` ([review 5486843735](https://github.com/d6g8k5htny-coder/main/pull/373#pullrequestreview-5486843735); pickup released at [6118390355](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118390355)). The local normal form and κ_r, κ_ℓ, κ_t check out.
  - **B:** OpenAI dot/Codex `/root/verify_zeta_packet_citations`.
    - **AMEND** at `70f3d993` ([review 5486879914](https://github.com/d6g8k5htny-coder/main/pull/373#pullrequestreview-5486879914)).
    - **Delta PASS (scoped)** at `0969b327` ([review 5487692150](https://github.com/d6g8k5htny-coder/main/pull/373#pullrequestreview-5487692150)); pickup released at [6119816683](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119816683).
  - **Delta readbacks of `0969b327`:** requested at [6119395474](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119395474). B has passed; A/D is in progress ([pickup 6119775263](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6119775263), due 08:00Z).
  - **C:** offered to GROK-H01 ([6118393130](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118393130)); no pickup yet.
  - **Integration owner:** none yet. It must be neither the author nor a reader.
- **Completed review scopes:** all are Anthropic Claude reading Claude work, so organizational-independence credit is 0. None is the nonauthor read.
  - Two review passes, mathematics (M1–M17) and numerics (N1–N12), every finding applied (README §10).
  - An accuracy checker (16 findings) and a reproducibility checker (6), every finding applied.
  - Then independent re-checks of accuracy, paraphrase/privacy and reproducibility/CI. All three PASS on this tree.
- **Unresolved findings:** none open on the author side.
  - ZF-AD-01 to 05 and ZF-B-01 to 05 are applied in `479d2450`; the acceptances are at [6118394964](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118394964) and [6118467693](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118467693).
  - B's delta readback passed. A/D's is pending.
  - **For slice C:** no numerical value changed. Two scripts change labels only, so one SVG and one `.out` differ only in label lines. A C run at `70f3d993` stays applicable to every number.

## Verification

Run locally on this tree with Python 3.13.16, numpy 2.5.3 and mpmath 1.4.1:

| Check | Result |
|---|---|
| `python3 -B -m unittest test_exact test_numeric` (packet dir) | 20 OK (three exact tests added for the review fixes) |
| `python3 -B -S -m unittest test_exact` (stdlib only) | 14 OK; `test_numeric` skips cleanly without numpy/mpmath |
| `python3 -B make_manifest.py --check` | `ok`; 116 files hashed, plus the manifest; `sources.main` = `7588716d` (the base when the packet was written) |
| Repo `python3 -B -S -m unittest discover -s tests` | 545 OK, 6 skipped (unchanged; repo CI does not discover the packet's tests) |
| `tools/workspace_landing_check.py`, `tools/navigation_check.py` | `problems: []` |
| `git diff --check` | clean (CSV writers use LF) |
| Independent re-run (reproducibility checker) | collision fold checks, κ checks, theta checks and the full stats pipeline reproduce the committed outputs; with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, seven regenerated raw samples are byte-identical to their recorded SHA-256 |

Raw `.npz` samples (about 204 MB) are not committed; only their hashes are. The packet is about 520 KB.

**Hosted checks on the exact head are authoritative.** On `0969b327`, all 8 check runs succeeded:
- `verify`, `landing-checks`, `architecture` and `formal-evidence` ([38215567002](https://github.com/d6g8k5htny-coder/main/actions/runs/38215567002));
- `public-shop` and `browser-smoke` ([38215566368](https://github.com/d6g8k5htny-coder/main/actions/runs/38215566368));
- `public-intake` ([38215564326](https://github.com/d6g8k5htny-coder/main/actions/runs/38215564326), [38215632976](https://github.com/d6g8k5htny-coder/main/actions/runs/38215632976)).

`mergeable_state` is clean.

## Evidence and limits

- **Exposure:** author and all pre-PR reviewers are the same provider; credit 0. The note's author is not named; it is attributed only as forwarded by the project owner.
- **Not interval-certified.** High-precision mpmath (collisions agree across independent quadratures to about 1e−43), not a certificate.
- **Citations:** this session could not fetch arXiv or journal pages.
  - The scoped citation table in `CLAIMS.md` (Citation provenance) gives each item's status in slice B's three-term vocabulary. Items checked against a primary passage carry the locator.
  - **Still unverified:** Saouter–Gourdon–Demichel; the 1988–1992 lower-bound originals and the 34/35 identification; Davenport–Heilbronn; de la Vallée Poussin; Jacobi; Mumford; Cerf; Gonek; Hejhal; the KKL full proof; Edwards (bibliographic only).
  - Remaining [snippet] and [memory] tags are not upgraded.
- **The project's theorems are not applied to ζ.** For ζ, the (D1) transfer is heuristic; GUE behaviour of zeta zeros is itself a conjecture. The CUE cut convention used here differs from the T-CUE target and is flagged.
- **Exploratory, not held-out:** data inspected first; no seeds of any frozen protocol were consumed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip

---
_Generated by [Claude Code](https://claude.ai/code)_