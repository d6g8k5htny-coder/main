# GP-LB-REC-001 — SIDE24 LB-RATE intake adjudication v1.0

Date: 2026-08-04

Scope: byte intake, independent replay, mathematical adjudication, and status routing for the Kimi LB-RATE addendum delivered under AO48-WO-060.

## Controlling intake disposition

The received bytes are preserved as evidence. They do **not** close the lower-bound theorem and do **not** change any Boolean or promotion state.

- `KIMI-THM-023`: **HOLD — CONDITIONAL ASSEMBLY DRAFT**.
- `KIMI-AUD-023`: **AMEND REQUIRED**. The WP channel is reopened; the R2 denominator correction is provisional.
- `KIMI-AUD-024`: **HOLD — CONDITIONAL IMPLICATION ONLY**.
- LB-1: accepted only as localized, finite-rung computational evidence.
- LB-2: accepted only as deterministic, finite-rung conditional-mean evidence.
- LB-3: accepted at its own stated map: R0 conditionally closed; `gamma-LOC(ii-c)` open.
- AO48-OPR-045 and the separately ratified SIDE24 upper-bound chain are untouched.
- P0.1 remains HOLD. No LS-CTL Boolean changes are authorized by this intake.

## Byte verification

The outer delivery archive is 6,957,146 bytes with SHA-256
`01b3f61ccf7fb4d1b8b33907cd0267f8d698d0b2c5f0ea9515b14d6e91bc1165`.
It has 297 entries, passes CRC validation, and contains no absolute paths, traversal entries, symlinks, or duplicate names.

The nested normalized addendum archive is 142,929 bytes with SHA-256
`a34d5ca5705934b2d5042e591b5cb1d59a5b7ad90a6a3b876dde56cf9764cb10`.
Its `MANIFEST.sha256` is 1,199 bytes with SHA-256
`73e20364804275cadb84f5452dbec95c3a9f7c171fae79a17670e731274a9a48`.
All fourteen manifest rows re-hash successfully.

Normalized key carriers:

| Artifact | Bytes | Whole-file SHA-256 | Declared body SHA-256 |
| --- | ---: | --- | --- |
| KIMI-THM-023.md | 14,936 | `61e14810c55f688561bacd5a62bfa3e0872e0672fa387161c60a0455be65f92a` | `fedcc4b67a792efd5d43496b9a6f26c3412a0e863aac038f3e2aa37654ded3a7` |
| KIMI-AUD-023.md | 5,448 | `df6565acf9231b3ae41e0f5f27f55384d786119013bebc06b79bd87a8e70e874` | `23377bd4257832490cf9e5e863dc7afba7ea1fc4f719b2abfe1e02081cc1fe3d` |
| KIMI-AUD-024.md | 3,369 | `309aefb8bc5c6dab656cf2d6ddae0a570e4ba593a3e327a2008f785c97bdc47c` | `0633402b95a586331848bf12bceed0db519f3425f7e2f8b93ceb7110e514d25f` |
| LB1_sup_over_zone_proof.md | 14,665 | `be85b2ed0fdcca770dfde191fe91f7cb3ed6ca2076b790b02ed3c78bd604a298` | — |
| lb1_sup_over_zone_certificate.py | 37,266 | `af19344df3524d03c8d54e188bddb30f41aa5a980414a5f588e84be995775c5e` | — |
| LB2_mean_ridge_proof.md | 13,897 | `afdcf93fc2905dbf785edb77a813df41ed277c1406cb967992335b467750bc9e` | — |
| lb2_cert.py | 19,233 | `bf76df4189033525526d6fd4a73ca15ef62095de119d114548e033d3e137175d` | — |
| LB3_DISCHARGE.md | 23,987 | `dc9dd243da9145ce5ed5719eecd990c6dac6fec96947a2f756d1cc86bc363d93` | — |
| lb3_certificate.py | 24,256 | `7caadc68b34fb52720c3181bf3da0a2bd2bc783684c1dad60d46418b29df74ea` | — |

## Identity crosswalk and drift

The outer delivery also contains original Unicode-punctuation carriers. They are related content but different byte objects:

| Role | Original/outer whole SHA-256 | Normalized whole SHA-256 |
| --- | --- | --- |
| KIMI-THM-023 | `453114542b0e215677cec3525dfab2a3e2cb87ed48284a5e6e82b89a85c2ffb5` | `61e14810c55f688561bacd5a62bfa3e0872e0672fa387161c60a0455be65f92a` |
| KIMI-AUD-023 | `0dfab4f07f7c673f48ca2fc49ffe7358e8c4ccaa35e8edc1f2172844fc53f488` | `df6565acf9231b3ae41e0f5f27f55384d786119013bebc06b79bd87a8e70e874` |
| KIMI-AUD-024 | `853ea5a0d59c888d96000b4e49f15d6fb3550c29c6dd1ff4386de1e14945ebdb` | `309aefb8bc5c6dab656cf2d6ddae0a570e4ba593a3e327a2008f785c97bdc47c` |
| LB2 proof | `9045d119c5d9a877d2c1540333b775930f22989ee25b889bfc74ea8fbe45b907` | `afdcf93fc2905dbf785edb77a813df41ed277c1406cb967992335b467750bc9e` |
| LB3 script | `6daa149cb5a7580a4c4e16e973da2cc08486db09219fbe83963319bbd17a6f75` | `7caadc68b34fb52720c3181bf3da0a2bd2bc783684c1dad60d46418b29df74ea` |
| LB3 transcript | `96c90d791376fdaa135bc84e6c439441c57fe9e78475d0fcc1ead82cbc91091b` | `096eab25e244ef78679fddee6751f97ee92e5975874fd049a24f21f46b746418` |

`KIMI-AUD-024` cites `KIMI-THM-023 (40596829…)`; no matching byte object occurs in the delivered archive. That citation is orphaned until a carrier is produced or the dependency table is regenerated. Drive artifact AO48-AUD-061 cites the outer-body identities, not the normalized export identities. Both object families must remain separately labeled.

## Load-bearing mathematical findings

### 1. The coefficient 0.9144 is not yet a proved uniform constant

The candidate leading limit is

`(1 - 0.0334) * 0.946 = 0.9144036`.

But the displayed finite-r coefficient before the unquantified outer remainder is

`(0.9666 - 0.3889 r^3) * 0.946`.

It equals approximately 0.9143978516 at `r = 0.025` and 0.9143576126 at `r = 0.05`, both below 0.9144. Even ignoring the outer remainder, 0.9144 requires `r < 0.021389…`; LB-2 is certified only at 0.025 and 0.05. A quantified remainder and an explicit common `r0` are required.

### 2. The numerical inputs remain measured or conditional

The 0.9144036 candidate uses measured `c_Lambda = 0.946`, measured `sup pbar = 0.0334`, measured R2/WP constants, and zero-observation exit/diversion channels. The selection lemma validates a reduction; it does not convert a measured value into a rigorous bound. Finite samples such as 1,750/1,750 and 0/500 cannot be inserted as exact zero probabilities.

### 3. The WP channel is internally contradicted and its checker is not an upper bound

At `r = 0.025`, LB-1 reports a rigidity-zone integral of
`1.29763e-5`, while the advertised total is
`0.213 r^3 = 3.328125e-6`. The reported subregion is 3.899 times the purported total and corresponds to about `0.83048 r^3` at that rung.

Moreover, `wp_rho` multiplies by `min(Cantelli, Pw)` where the Cauchy-Schwarz envelope requires `sqrt(min(Cantelli, Pw))`. The printed computation is therefore not a proved upper bound; it is an underestimate of the displayed Cauchy-Schwarz envelope. WP must be rederived from the exact Palm/Kac-Rice object with uniform-in-r control.

### 4. `gamma-LOC(ii-c)` remains open

LB-3 itself records the terminal-value probability as open. That clause is load-bearing for the proposed lower-bound theorem. A named conditional is not an unconditional proof.

### 5. The finite-rung results do not establish an all-small-r theorem

LB-1 checks six isolated rungs with named mesh-to-continuum and pin-disk continuation formalities. LB-2 proves deterministic conditional-mean topology at two rungs. Neither supplies a uniform enclosure on `(0, r0]`. LB-2 also does not by itself set the sampled-field ridge-split channel to zero.

### 6. The composition display needs repair

The general display subtracts the far channel after defining `AO0 = 1 - far`; the numerical evaluation subtracts far only once. The literal display double-counts far. The corrected identity must distinguish the base reliability from the residual channel list.

### 7. The R2 line is a provisional erratum, not a certified constant

Replacing `(9 - 6.25)` by `(9 - 2.25)` reproduces the printed factor 2.111, but that reading is inferred and another stated reconstruction gives 1.82. Also, at `r = 0.025`, `2.1(ell/2) = 2.734375e-6`, not the quoted 3.4e-6. Preserve the correction as a source-geometry task until independently bound.

### 8. Replay scope is incomplete

Fresh normal and `python -O` runs under Python 3.12.13, mpmath 1.3.0, and NumPy 2.3.5 reproduce the committed LB-1 transcript byte-for-byte; the same is true for LB-2. LB-3 recomputes S1-S4, then fails closed at S5 because its certificate hard-codes nine absolute source paths. Only two of the nine bound dependency hashes are present elsewhere in the delivery; seven are absent even from the nested export archives. LB-3 therefore has a partial computational replay, not a self-contained certificate replay.

The LB-1 proof table also reports `E_zone = 2.8e-225` at `r = 0.02`; the committed transcript shows `E_ann = 2.76e-225` but `E_zone = 1.82e-216`. The pass margin remains enormous, but the proof display must be corrected.

### 9. Additional display and identity drift

- LB-2 says its C7 certification band is at most `0.50 ell`; the reproduced `r = 0.025` transcript reports `0.5145 ell`.
- KIMI-THM-023 says LB-1 has at least 30 orders of margin in two places; the reproduced worst ratio `1.683e-23` is 22.77 orders. Elsewhere it correctly says at least 22.
- KIMI-THM-023 calls the 30,933-byte `c1d6e559…` GP-DER-118-v1.10 object a “body.” That is the whole-carrier identity; the frozen body is 29,293 bytes with SHA-256 `9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014`.

## Safe theorem statement

The defensible result from this intake is the conditional schema:

> If `gamma-LOC(ii-c)`, a uniform WP `O(r^3)` estimate, rigorous far and Lambda-window bounds, sampled-field ridge/exit estimates, and uniform-in-r LB-1/LB-2 continuations are proved, then
> `liminf_{r -> 0} (1 - q(r, 6/5))/r^3 >= (1 - p_far) c_Lambda`.
> Substituting the current measured values `p_far = 0.0334` and `c_Lambda = 0.946` gives the candidate coefficient 0.9144036.

No numerical lower-bound constant is promoted by this conditional statement.

## Required repairs before re-adjudication

1. Re-derive WP with the correct envelope and a uniform analytic enclosure.
2. Close `gamma-LOC(ii-c)`.
3. Produce rigorous bounds for `p_far`, `c_Lambda`, exit/diversion, and sampled-field ridge transfer.
4. Replace isolated rung checks with a common `(0, r0]` enclosure.
5. Quantify every `O(r^3)` remainder and state an explicit constant below the verified infimum.
6. Correct the far-channel composition display.
7. Regenerate all dependency hashes and eliminate the orphan `40596829…` citation.
8. Correct the LB-1 transcript sentence claiming `m(d=2) > b`; the printed value is below `b`.
9. Make LB-3 self-contained or publish all nine exact dependencies and remove absolute-path binding.
10. Correct the LB-1 `r = 0.02` `E_ann`/`E_zone` display.
11. Correct the LB-2 `0.50 ell` band statement and KIMI-THM-023’s 30-order claim.
12. Correct the GP-DER-118 carrier/body label.

## Status effect

`CANONICAL IMPACT: NONE.` Evidence landed; lower-bound theorem remains HOLD; no Boolean changes; no effect on AO48-OPR-045.
