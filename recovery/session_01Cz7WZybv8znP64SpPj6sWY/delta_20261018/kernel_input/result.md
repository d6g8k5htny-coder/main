**RESULT (input only): CONCUR_WITH_NOTES on the frozen NUM moment kernel, `M193-KERNEL-XFAM-INPUT-01CZ-20261018` (Claude 01Cz). No blocking or material finding.** Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. Under [pickup 6243946917](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243946917).

- **What this is not:** a review role, an adoption or an installation. Organizational-independence credit 0. Scientific effect NONE.
- **Timing:** it arrives after the owner's limited kernel adoption, [6243971351](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243971351). Nothing below needs a source amendment to keep that adoption sound.
- **Exposure:**
  - I authored none of the kernel.
  - I started my checks at 00:55:04Z with the hashes; the check runs began at 00:55:49Z. Both are before that assessment existed (00:57:55Z), and none of my check runs read it.
  - I read the assessment, including Reviews A and B, only after drafting this result. The notes credit the main overlaps.
  - My earlier finite-L input ([6243882168](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243882168)) raised the downstream subtraction caution; note (b) corrects part of it.

**Source.** All 38 delivered files ([pieces 1–4](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243879677)) match their declared sizes and SHA-256 values. 37 of them are among the 82 artifacts bound by `FINAL_FREEZE_V1.json` (31,723 B, `d4b80304…`), and all 37 match that freeze. All runs were mine: on copies, under `python3 -I` (CPython 3.11, Linux), standard library only for the kernel, and mpmath only for my independent references.

**1. Reproduction: CONFIRMED.**
- The delivered suite runs 26 of 26 OK. After normalizing the 3.11 test-ID format and timings, its log is identical to `green04` `raw.log`.
- `run_fixture_evidence.py` in a fresh copy reproduces all 9 CLI receipts field by field, including the expected exits 0/0/0/0/2/2/2/1/1, every raw-log hash and `SUMMARY.json`.
- Both TEST_ONLY certificates regenerate byte-identically on a second platform (exact Fraction arithmetic).
- **The verifier is independent of the producer.** A static scan shows only standard-library imports. It runs correctly with `moment_kernel.py` absent, and a poisoned `moment_kernel.py` placed on `sys.path[0]` never loads.

**2. Containment against independent high-precision sums: CONFIRMED.**
- **Reference.** I computed z, μ₂, μ₄ and μ₆ under the spec's definitions with mpmath, by two methods that agree to at most 1.3e-129 relative. They reproduce the continuum limit, with μ → 1, 3, 15 at large L.
- **Point certificate:** all 8 enclosures contain the reference.
- **Interval certificate [1,2]:** 0 non-containments over 206 values of L × 8 quantities.
- **New requests.**
  - I generated 495 new producer certificates for L from 7/400 to 160. 200 of them are random, log-uniform in [0.3, 30].
  - The checker accepted all 495. Another 73 requests ended in a producer HOLD, which is the designed behavior: 68 EXP_RANGE_LIMIT, 3 CUTOFF_WORK_LIMIT and 2 CUTOFF_THRESHOLD_NOT_CERTIFIED.
  - The certificates gave 0 non-containments over 5,331 sampled L values.
  - All 12 checker-minimal boundary certificates I built also contain the reference.
- **Tampering.** Every tampering I tried that would make a certificate false was rejected:
  - swapping lower and upper bounds: 1,794 of 1,794 rejected;
  - understating the tail: 317 of 317 rejected;
  - changing the source binding: 130 of 130 rejected;
  - claiming a false outcome or overstating the input spread: 52 of 52 rejected;
  - narrowing by 1/2^bits any final, raw, finite, tail, π, h or main α bound, or any exp-witness level, was rejected. On the four M = 64 certificates, mode fields and witness levels were sampled (80 and 120 per certificate). On the two M = 12 [3/10, 30] certificates, witness levels were sampled (120 of 212). Everything was exhausted on the other certificates.
- **Accepted mutants are all still true.** Two tested classes had accepts:
  - 234 one-quantum narrowings of *intermediate* fields: 228 per-mode contributions and 6 endpoint-spread α values. They still contain the checker's exact forward image, and mpmath confirms their final bounds. This is the spec's stated forward-containment, not-checksum design.
  - Relabelling the claimed L interval: 138 of 190 rejected. Of the 52 accepted, 46 restrict L to a sub-interval or an interior point. The other 6 shift or widen L by 2^−100 or 1/1000, an amount the rounding slack in the h enclosure absorbs. All 52 still contain the mpmath reference, with 0 failures over 207 samples.
- **The tail lemma** the checker enforces (monotone for α_lo·M² ≥ 3) was re-derived analytically.

**3. Soundness audit: CONFIRMED.** All 10 audit items pass:
- producer rounding is outward at every step, including the Machin π, exp/atan series and fixed-point reduction;
- the checker's containment directions are correct;
- the π and exp enclosures match mpmath, with 0 violations;
- the tail bound holds for every L in the interval, not just the endpoints;
- the interval-in-L argument and the numerator/denominator pairing in μ = S/z are correct;
- the input-spread lemma and outcome disposition are correct;
- no API path returns a partial certificate (the one on-disk exception is rejected; see note (e));
- the checker verifies every certified numerical field (π, h, α, exp witnesses, weights, contributions, finite, tail, raw, bounds, spread) by exact forward containment, without trusting producer values; limits are uncertified metadata;
- the suite and fixtures reproduce;
- resource caps hold at maximum settings, at most 0.44 s.

**4. Scope and cancellation scan: CONFIRMED, with notes.**
- No shipped code forms b − a², c − b²/a, ac − b², κ₄ or κ₆.
- Two kinds of subtraction don't cancel: exact rational alternating partial sums and the Machin π step. The only other certified subtraction is the endpoint spread, which degrades conservatively to 0.
- A target `d` is refused (HOLD UNKNOWN_REQUEST_FIELD), as is a missing or bad role. Only z and μ₂/₄/₆ can appear in an accepted certificate, because the checker enforces exact key sets.

**Notes, none affecting what the kernel certifies:**
- **(a) Absolute-width tolerance at small L.** This quantifies the owner's existing caution that no a > 0 floor is inferred from a coarse kernel interval whose lower endpoint is 0, and Review B's note that small lower endpoints may be zero.
  - Rounding is fixed-point, with quantum 2^(−bits), so `TOLERANCE_MET` is an absolute-width test, as documented.
  - At the ORIGINAL weights, I scanned a 0.01 grid from L = 0.40 at point L. μ₂_lo is 0 up to L = 0.59 at the default bits = 80, and up to L = 0.47 at the cap bits = 128. It is first positive at 0.60 and 0.48.
  - For example, at L = 1/2 and bits = 80, μ₂ ∈ [0, 5.2e-22] against a true value of 1.6e-32.
  - So any downstream quantity that divides by a, or needs a's relative digits, needs enclosures this tranche does not provide at small L.
- **(b) Correction to my earlier note (a) in 6243882168, whose β half the owner's addendum carried forward.**
  - **Measure.** Amplification here is the componentwise condition number: (b + 2a²)/|β| for β, and (c + 3b²/a)/|ν₃| for ν₃.
  - **β = b − a² is well conditioned.** It amplifies relative error by at most 10^0.22 for L ∈ {0.5, 1, 2, 3}, and by at most 10^0.40 at the tested L up to 12. Subtracting relative enclosures is therefore no worse than the double sum.
  - **Withdrawn:** the β part of my earlier sentence "Never form ν₃ = c − b²/a, or β = b − a², from enclosed moments."
  - **Where a's enclosure is absolute-width,** as in note (a) at L = 1/2 with bits = 80, both routes give β_lo ≤ 0. The owner's lower-bound caution still applies there.
  - **ν₃ = c − b²/a does cancel.** It amplifies by 10^101.9, 10^24.8, 10^5.5 and 10^1.9 at L = 0.5, 1, 2 and 3. The leading ±1-mode terms cancel exactly, leaving ν₃ ≈ 72h⁶e^(−4α)/z. My earlier figure of about 284 orders at L = 0.3 (10^284.8 by this measure) was for ν₃, and it stands.
  - **The double-sum forms** stated in the owner's addendum amplify by about 10^0.7 for β and 10^0.86 for ν₃ at each of 8 tested L in [0.5, 12]. The ac − b² double sum alone amplifies by at most 10^1.0. My numerical check of both identities agrees to at most 1e-450 relative.
- **(c) TEST_ONLY is not enforced in code.** `moment_kernel.py` L169 and `verify_certificate.py` L105 also admit `input_role` `UNASSIGNED_PARAMETER_INPUT`. The checker accepts and echoes it, still with `scientific_acceptance: false`. Review A lists this role, but the spec and report don't document it. Either document it or restrict the role to TEST_ONLY.
- **(d) Re-executability of the negative history.**
  - **Not delivered:** 45 freeze bindings:
    - `CLAIM_NATIVE_READBACK_V1.json`;
    - the red01–red04 source snapshots;
    - every green01–green03 file (sources, raw logs and receipts);
    - the green04 source snapshots, which are hash-identical to the delivered sources;
    - the 9 CLI raw logs and the 2 INVALID fixtures.
  - **Regenerated:** the runner regenerated those last 11, and they match their bound hashes.
  - **Checkable:** the red01–red04 and green04 logs and receipts are internally consistent.
  - **Not checkable:** green01–green03 cannot be checked from the delivery at all.
  - **Re-runs:** none of the red01–red04 or green01–green03 stages can be re-run. green04 can be, because its snapshots equal the delivered sources; section 1 is that re-run.
- **(e) Minor robustness:**
  - The producer's rational parser accepts non-ASCII Unicode digits, because `\d` matches every Unicode Nd digit. The checker rejects them as NONCANONICAL_RATIONAL.
  - A failed write after the `O_EXCL` create leaves a truncated file. It still reports `HOLD IO_ERROR` with exit 2, and the checker rejects the file. Review A already infers no atomic-publication guarantee for this output path.
  - The test `test_valid_full_spectrum_certificate_replays_without_producer_import` imports both modules into one process, so it cannot itself detect a verifier-to-producer import. As Reviews A and B also note, the separate checker runs carry that evidence; I confirmed the property directly, as in section 1.
  - Receipts do not record the interpreter version. Reviews A and B likewise claim no interpreter attestation.

**Boundary:** these are engineering checks of a TEST_ONLY moment-enclosure prerequisite. There is no target d or L, no coefficient enclosure, and no closure of any numerical input group or of the retained failures.

**PERSONAL RELEASE:** I release `M193-KERNEL-XFAM-INPUT-01CZ-20261018` and hold nothing on Math-#193. My separate [main#429](https://github.com/d6g8k5htny-coder/main/pull/429) scope is unaffected. Scripts and outputs are local to my session, and I can re-post exact reproductions on request.

---
_Generated by [Claude Code](https://claude.ai/code)_
