## Change and purpose

**This PR adds one new directory, `experiments/zeta_fold_exploration/`, an exploratory packet. Scientific effect: NONE. No status or register change.**

Dylan asked this session to examine an external note he forwarded on 12 October 2026, about links between zeta zeros and the project's fold and short-bar law, and to bring in whatever is worth exploring. Pickup: [307/6116750096](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116750096).

The packet checks each link, records what is classical, what is wrong and what is useful, and keeps scripts and results so anyone can re-run them. The note is paraphrased claim by claim in `CLAIMS.md` and never quoted. The packet makes no claim of progress toward RH, and no bound on Λ beyond known results.

**What it finds:**
- **Folds (generic and known).**
  - Two de Bruijn–Newman zero collisions were computed to about 40 digits, at t ≈ −1.878 (zeros 9/10) and t ≈ −0.385 (zeros 34/35). Every collision found among the lowest 46 zeros is at t < 0.
  - Near a nondegenerate double zero, r² = 8τ and the antiderivative's lobe area is |A|r³/6, the project's k = F‴/12 fold law.
  - Bar pairing is verified only near the collision. The critical-value gap and an actual finite H0 bar are kept distinct.
- **Theta (a footnote).** The project's periodized normalizer is a Jacobi theta value. ζ appears only after a Mellin transform over the torus side, which the project holds fixed.
- **Statistics (useful for the project's own geometry-versus-measure question).**
  - With the same cubic fold, r² pair repulsion (CUE; conjecturally ζ) moves the short-bar density exponent from −1/3 to about 0, as the project's transfer rule predicts.
  - ζ on t ∈ [10⁵, 10⁶] gives −0.020 [−0.041, +0.001] (sampling-only), measured on antiderivatives of Hardy's Z over 1.6M zeros.
  - That is decisive against −1/3 at these heights, and consistent with about 0 given the control bias. On the controls, the same estimator lands −0.017 to −0.065 from the known values.
  - The data were inspected before any prediction was frozen, so this is exploration, not confirmation.
- **Proposed next targets (§6, not claimed here):** T-CUE, a short-bar theorem for fixed-N CUE, and a frozen, pre-registered zeta comparison on fresh windows.

**Related, not taken:** [307/6116121839](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116121839) is an unclaimed OpenAI-lane proposal for polynomial-first heat-flow fold controls in `experiments/universality/`. This packet touches only its own directory. Prior art on the same antiderivative: Lagarias–Montague, arXiv:1106.4348 (snippet-checked only).

Scope: mathematics and numerics, exploratory only. No other path changes.

## Current disposition

- **Head:** `70f3d993` on main `7588716d`, tree `0f621c27`. **Draft.** 117 added files, all under `experiments/zeta_fold_exploration/`.
- **Author:** Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip`. **No author merge.**
- **Claim:** [307/6116750096](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116750096), lease to 08:00Z on 12 October.
- **Nonauthor reads (README §9 slices):**
  - **A and D:** OpenAI dot/Codex `/root/review_zeta_fold_mathematics`, **AMEND** at `70f3d993` ([review 5486843735](https://github.com/d6g8k5htny-coder/main/pull/373#pullrequestreview-5486843735); pickup released at [6118390355](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118390355)). The local normal form and κ_r, κ_ℓ, κ_t check out.
  - **B:** OpenAI dot/Codex `/root/verify_zeta_packet_citations`, in progress ([pickup 6118327380](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118327380), due 06:30Z).
  - **C:** offered to GROK-H01 ([6118393130](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118393130)); no pickup yet.
  - **Integration owner:** none yet. It must be neither the author nor a reader.
- **Completed review scopes:** all are Anthropic Claude reading Claude work, so organizational-independence credit is 0. None is the nonauthor read.
  - Two review passes, mathematics (M1–M17) and numerics (N1–N12), every finding applied (README §10).
  - An accuracy checker (16 findings) and a reproducibility checker (6), every finding applied.
  - Then independent re-checks of accuracy, paraphrase/privacy and reproducibility/CI. All three PASS on this tree.
- **Unresolved findings:** ZF-AD-01 to ZF-AD-05, all accepted ([6118394964](https://github.com/d6g8k5htny-coder/main/pull/373#issuecomment-6118394964)).
  - **Next action:** one successor commit carrying the A/D and slice B fixes once B's verdict is in, then a delta readback request to each reader. The head stays frozen at `70f3d993` until then.
  - **For slice C:** the planned fixes change README/CLAIMS wording and possibly test assertions. No numerical script or result is planned to change, so a C run at `70f3d993` stays applicable; any exception will be named in the successor note.

## Verification

Run locally on this tree with Python 3.13.16, numpy 2.5.3 and mpmath 1.4.1:

| Check | Result |
|---|---|
| `python3 -B -m unittest test_exact test_numeric` (packet dir) | 17 OK |
| `python3 -B -S -m unittest test_exact` (stdlib only) | 11 OK; `test_numeric` skips cleanly without numpy/mpmath |
| `python3 -B make_manifest.py --check` | `ok`; 116 files hashed, plus the manifest; `sources.main` = `7588716d` |
| Repo `python3 -B -S -m unittest discover -s tests` | 545 OK, 6 skipped (unchanged; repo CI does not discover the packet's tests) |
| `tools/workspace_landing_check.py`, `tools/navigation_check.py` | `problems: []` |
| `git diff --check` | clean (CSV writers use LF) |
| Independent re-run (reproducibility checker) | collision fold checks, κ checks, theta checks and the full stats pipeline reproduce the committed outputs; with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, seven regenerated raw samples are byte-identical to their recorded SHA-256 |

Raw `.npz` samples (about 204 MB) are not committed; only their hashes are. The packet is about 520 KB.

**Hosted checks on the exact head are authoritative.** On `70f3d993`, all 8 check runs succeeded:
- `verify`, `landing-checks`, `architecture` and `formal-evidence` ([38205803966](https://github.com/d6g8k5htny-coder/main/actions/runs/38205803966));
- `public-shop` and `browser-smoke` ([38205803766](https://github.com/d6g8k5htny-coder/main/actions/runs/38205803766));
- `public-intake` ([38205803765](https://github.com/d6g8k5htny-coder/main/actions/runs/38205803765), [38205810645](https://github.com/d6g8k5htny-coder/main/actions/runs/38205810645)).

`mergeable_state` is clean.

## Evidence and limits

- **Exposure:** author and all pre-PR reviewers are the same provider; credit 0. The note's author is not named; it is attributed only as forwarded by the project owner.
- **Not interval-certified.** High-precision mpmath (collisions agree across independent quadratures to about 1e−43), not a certificate.
- **Citations:** arXiv and journal pages could not be fetched from this session. Items are tagged [snippet] or [memory] and listed for review slice B.
- **The project's theorems are not applied to ζ.** For ζ, the (D1) transfer is heuristic; GUE behaviour of zeta zeros is itself a conjecture. The CUE cut convention used here differs from the T-CUE target and is flagged.
- **Exploratory, not held-out:** data inspected first; no seeds of any frozen protocol were consumed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01S16677s1SGeAJmN3QMz4ip

---
_Generated by [Claude Code](https://claude.ai/code)_