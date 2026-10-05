# Statement alignment — 20-target #272 measure bridge

Dylan Roy — delegated AI review. Actual performer: xAI / Grok 4.7, Cursor Cloud Agent, model `grok-4.7-high-fast`, session `bc-a0687e6b-da3f-4724-86d3-868e01ee676d` (https://cursor.com/agents/bc-a0687e6b-da3f-4724-86d3-868e01ee676d). Transport: Cursor Cloud (`usePrivateWorker: false`, `privateWorkerId: null`). `list-self-hosted-workers` returned `totalCount: 0`. `lake` and `lean` are not on `PATH`; Lean was not re-executed.

Scientific effect: **NONE**. Organizational-independence credit: **0**. Dylan's personal reading: **PENDING**.

**Verdict: PASS.** Finding IDs: none. Alignment disposition for this package: **ACCEPTED**, covering exactly the 20 manifest targets at Math-#272 head `270167900b71539d0e2135131a3a96f1f04289f6`. This record is distinct from the 29-target record at `reviews/lean_probability_alignment_20261005/alignment.json` on `8cf6b74472d77a62f4775a5e3568e79d4407adff`.

## Lane count for this pickup

Roll call `5994946754` had 2 confirmed lanes: the OpenAI coordinator and this Cursor Cloud run. Two local Task helpers then launched and returned inside this same slot:

| ID | Runtime | Scope |
|---|---|---|
| `bc-a0687e6b-da3f-4724-86d3-868e01ee676d` | Cursor Cloud, `grok-4.7-high-fast`, no private worker | Parent auditor; statement check, gate, publication |
| `bc-4ed14bcc-0a62-5c6f-8ecd-9d7e95b45878` | Local Task subprocess, model inherit | Original 13 against preserved sources and `SCOPE.md` |
| `bc-dde77d90-eb4c-5426-ba20-a338aabef307` | Local Task subprocess, model inherit | New 7 statements and new 9 evidence blobs |

Deduplicated confirmed external lanes remain **2**. The helpers are not self-hosted workers and are not separately reachable agents. No nested spawn and no timer.

## Exposure and authorship

Before the comparison finished, this session had read `REVIEW.md` at `8f70c102550a6080c2b8f472df8a601fc7f61081`, alignment JSON at `8cf6b744`, and the earlier seven-target audit `5994130542`. The statement comparison then used the frozen #272 sources and the informal extract below. No Lean file, gate, workflow, manifest, or scientific register was edited. Proof bodies were not graded.

Author of the seven bridge declarations, from `MEASURE_BRIDGE.md` and commit `2701679`: OpenAI / GPT-6 / GPT-6 Astra Pro. Author of the original thirteen companions: OpenAI implementation lane of Math-#92, commit `867d9e34b601` (2026-09-27). That commit does not name GPT-6 Astra Pro. The machine `author` object below is the #272 bridge author, the performer who owns this candidate. The reviewer differs from both OpenAI lanes on provider, family, and agent. Same-provider overlap with the earlier Grok reads is disclosed and earns no organizational-independence credit.

## Bound identity

| Object | SHA-256 |
|---|---|
| `formal/manifest.json` | `845d372a459d32f6b4ffb0a4dceb69363ffe27c4ca6da8838238ce8979d9244d` |
| `formal/SCOPE.md` | `e303550bdd71aa969703fd33dc9808383db13b08072437577d192a7f149a3edb` |
| `formal/originals/Algebra.lean.txt` | `4c196820c4db8fafc288dd35642828ba577e3544d60aba7d1d6d24f14ad1e8ae` |
| `formal/originals/ProbabilityCompanions.lean.txt` | `4ace6600a476859982c8851ae9097c89b082d3c96291b3c8330bd4cc00bae65d` |
| Informal `imports/hardening_ebedb780/P02-LM-008/proof.md.export.txt` at `22876edaaec054d3ab8b0b16668ce3ff8cbf8c73` | `06967d0ba2c2d20550f3bd86ddc97981e324e5f94cd7710996445d507ae2285b` |

`MeasureBridge.lean` blob `11843e5586ec44c7430f077694c90ac8586c50c0` is the same blob at #273 head `dce1b197be3431673289905259be38b450e9b898`. Current PR heads observed during this read: #272 `2701679`, #273 `dce1b197`, #274 `8cf6b744`.

Unified diff of each V2 module against its preserved original is exactly the three `COMPATIBILITY.md` repairs: `noncomputable def foldPotential` with the same right-hand side; deletion of the surplus `ring` after `field_simp [hr]` in `ec014_contact_power`; deletion of the surplus `ring` after `field_simp [hr, hcZ]` in `p02_lm008_r2_cancel`. All thirteen theorem statements, from `theorem` through the character before `:=`, match the preserved originals.

## Per-target verdicts

Each verdict is against the #272 `SCOPE.md` row and the cited informal interface. Informal P02-LM-008, lines 80–124 and 215–219, states a uniform family for `0 < r ≤ r_0` and defines `P_r(E) = E[W_r 1_E] / Z_r`. The Lean ratio bound keeps that inequality direction and the lower-normalizer hypothesis `cZ * r ^ 2 ≤ ∫ W`. Construction of `P_r` as a probability measure is outside these 20 declarations; `SCOPE.md` places it under "Not established" for `p02_lm008_measure_transfer`.

| Target | Verdict | Statement checked |
|---|---|---|
| `ec005_fold_gap` | ALIGNED | For every real `s`, `foldPotential s s - foldPotential s (-s) = (2 * s) ^ 3 / 6`. |
| `ec008_factor_expansion` | ALIGNED | The displayed scalar polynomial identity in unrestricted real `c, s`. |
| `ec010_generic` | ALIGNED | The displayed four-real polynomial identity. |
| `ec010_transverse` | ALIGNED | The displayed three-real polynomial identity. |
| `ec011_scalar_cancellation` | ALIGNED | `(-h * w) * g + w * (h * g) = 0` for real `h, w, g`. |
| `ec014_contact_power` | ALIGNED | For `r ≠ 0`, `(r ^ 3 / 6) * (1 / r ^ 5) * r ^ 2 = 1 / 6`. |
| `p02_lm008_r2_cancel` | ALIGNED | For `r ≠ 0` and `cZ ≠ 0`, the displayed `r ^ 2` cancellation. |
| `p02_lm008_cross_multiplied` | ALIGNED | From `0 < r`, `0 ≤ cW`, `0 < cZ`, `0 ≤ q`, `cZ * r ^ 2 ≤ z`, and `n ≤ sqrt(cW) * r ^ 2 * sqrt(q)`, conclude `n * (cZ * r ^ 2) ≤ (sqrt(cW) * r ^ 2 * sqrt(q)) * z`. |
| `p02_lm008_quotient_bound` | ALIGNED | Those hypotheses plus `0 < z` conclude `n / z ≤ (sqrt(cW) / cZ) * sqrt(q)`. |
| `p02_lm009_bad_event_threshold` | ALIGNED | `0 < r` and `ε < r * R ^ 5` imply `ε / r < R ^ 5`. This is deterministic threshold algebra. It is not a Markov measure bound. |
| `p02_lm009_power40_identity` | ALIGNED | `(R ^ 5) ^ 8 = R ^ 40`. |
| `p02_lm009_r4_le_r3` | ALIGNED | `0 ≤ r ≤ 1` implies `r ^ 4 ≤ r ^ 3`. |
| `p02_lm009_palm_r4_to_r3` | ALIGNED | `0 ≤ C`, `0 ≤ r ≤ 1`, and `p ≤ C * r ^ 4` imply `p ≤ C * r ^ 3`. |
| `p02_lm008_integral_cs` | ALIGNED | One measure, two nonnegative `MemLp` 2 functions, Bochner product bound by the product of L² norms. |
| `p02_lm008_event_cs` | ALIGNED | One probability, measurable `A`, nonnegative `MemLp W 2`, set-integral bound by `sqrt(∫ W^2) * sqrt((μ A).toReal)`. |
| `p02_lm008_event_numerator` | ALIGNED | Those event hypotheses plus `0 ≤ cW` and `∫ W^2 ≤ cW * r ^ 4` give `∫_A W ≤ sqrt(cW) * r ^ 2 * sqrt((μ A).toReal)`. No `r > 0` hypothesis on this step. |
| `p02_lm008_measure_transfer` | ALIGNED | Adds `0 < r`, `0 < cZ`, and `cZ * r ^ 2 ≤ ∫ W`, and concludes the same-law ratio `≤ (sqrt(cW) / cZ) * sqrt((μ A).toReal)`. |
| `p02_lm009_measure_transfer_r8_to_r3` | ALIGNED | Adds `r ≤ 1`, `0 ≤ K`, and `(μ A).toReal ≤ K * r ^ 8`, and concludes the ratio `≤ ((sqrt(cW) / cZ) * sqrt(K)) * r ^ 3`. |
| `p02_lm008_sqrt_counterexample` | ALIGNED | Scalar witness `(1 : ℝ) = sqrt(4) * sqrt(1/4)` and the negation of the missing-square-root form. |
| `p02_lm008_upper_normalizer_counterexample` | ALIGNED | Scalar witness `1 ≤ 2 * 1^2` together with the ratio exceeding the reversed-normalizer expression. |

`0 ≤ cW` and `0 ≤ q` are the stronger real hypotheses `SCOPE.md` already records. The upper-normalizer counterexample matches informal mutation M1: an upper bound `Z ≤ C r^2` does not upper-bound the ratio. GP214's upper bound is not used as the lower bound.

## Fields consumed by `formal/gate.py --alignment`

`check_alignment` reads `disposition`, `manifest_sha256`, `scope_sha256`, `targets`, `author.provider`, `author.family`, `author.agent`, `reviewer.provider`, `reviewer.family`, `reviewer.agent`, `evidence.repository`, `evidence.commit`, `evidence.sha256`, and `evidence.path`. It compares `targets` by set with `manifest["targets"]` and `scope_sha256` with `files["SCOPE.md"]`. It rejects a shared provider, family, or agent, a stale digest, a partial target set, and any disposition other than `ACCEPTED`. It does not read `source_exposure`, `notes`, or `scientific_effect`. Success prints `SOURCE_IDENTITY_PASS` and returns exit 0; a failed alignment prints `FORMAL_GATE_FAIL` and returns 1.

## Boundary

Accepted here: the 20 Lean statements at the #272 scope, including the same-law ratio and the cubic composition of a supplied eighth-order event bound. Left outside this record: the weighted probability construction on #273; a concrete Gaussian or typed Palm law; a Markov bound from an integrable `R^40` moment; uniformity in `r`; discharged moment, normalizer, or tail premises; P0.2; parent persistence; and any scientific-register change. `alignment_status` in `formal/manifest.json` remains `PENDING_INDEPENDENT_REVIEW`.
