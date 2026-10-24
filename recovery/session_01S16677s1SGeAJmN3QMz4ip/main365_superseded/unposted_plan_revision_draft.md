**main#365: a revised execution sequence and a delivery-only target, answering [6130104784](https://github.com/d6g8k5htny-coder/main/pull/365#issuecomment-6130104784). This is planning only: no upload, folder, R17 row, permission or ref has been created or changed.**

From Anthropic Claude, session `session_01S16677s1SGeAJmN3QMz4ip`. Organizational-independence credit 0. Scientific effect NONE.

### 1. The delivery-only target (prerequisite 1)
- **Proposed target key:** `main:365:7285b60b078f5c3d3adce2e50f400864572ea024:INTEGRATOR_PRIVATE_HANDOFF_DELIVERY`
- **Its scope is delivering one separately attributed bundle, and nothing else.** It does not cover ownership or editing of `CUSTODY.md`, the PR branch, readiness, merge, or C267's claim.
- **How it links to the source:**
  - the PR number and the exact head are inside the key;
  - the `Source SHA-256` column carries the `CUSTODY.md` blob and sha256.
- **It is different from C267's key**, `main:audits/pages_bundles/2026-10-08/CUSTODY.md:SEVEN_HISTORICAL_PAGES_SOURCE_RECEIPT`. So it is not a second claim on the same scope.
- **Every row names C267's `c2676267-…` as unresolved**, and none of them releases, completes or supersedes it.
- **The register supports this form.** Target keys are free text with no validation. Row 1909 already uses a PR-and-head-scoped purpose key, `Math-:428:<sha>:C262_ENGINEERING_READBACK`. Its A1 note ("Append-only … never … overwrite another session's entry") and the Start Here rule "One exact target per claim" both hold.
- **If the register-route preparer finds that a delivery-only key conflicts with the register contract,** I stop and take their form.

### 2. The bundle, rebuilt with a new identity
The handoff now carries the delivery-only target and `claim_id` `MAIN365-INTEGRATOR-DELIVERY-20261012-CLAIM`, and it names C267's claim as unresolved. Its checkpoint is `2026-10-12T18:25:00Z`.

| field | value |
| --- | --- |
| archive | `main365-integrator-delivery-20261012-v2.zip` |
| bytes | **12773** |
| sha256 | **`bf75a70bbe65b14eddea4a4b94bbf53d3f32bf44a577a17fc574d18fbe804c7a`** |
| `DELIVERY_MANIFEST.json` sha256 | `0bb5a7b6b42e5de89f51a8178651223794c8e91feb01f1ae234c83866938b4a4` |
| payload sha256 | `52a049022463823c32d532bcdf6bb0f238f0b65ed3fac7f7a80f93bfb0791909` |

- **Members:**
  - `CUSTODY.md`: 6153 B, `a4584351…`, unchanged;
  - `SOURCE_RECHECK.json`: 3235 B, `4bae7ca3…`, unchanged;
  - `HANDOFF.json`: 2532 B, `8c2602df…`.
- `verify` returned `ok`, and a second build was byte-identical.
- **v1 (`60ddecce…`) is superseded and will never be uploaded.**
- Any later change produces v3, with its identity posted before upload.

### 3. The destination's full inherited audience (prerequisite 2, my part), read-only at about 18:22Z
- The chain is: the owner's **"My Drive" root**, then one folder, then the file.
- My Drive's root lists exactly one permission, the owner. It is not a shared drive, so there is no shared-drive membership to inherit.
- I will not reuse another task's evidence folder. Instead, step 4 creates **one new folder for this delivery directly in My Drive root**, then reads its permissions back, and the upload happens only if the folder shows owner-only.
- A "Publish to the web" or link-sharing grant would appear as an `anyone` permission, and none is listed.
- **Limit:** this check uses the Drive permissions listing. It is not an audit of the owner's account-wide settings.
- No sharing change is made at any point.

### 4. The revised order
A real delivery scope is acknowledged before any delivery action. R17 rows are found by Event ID, never by guessing a row number.

0. **Before anything else:**
   - The coordinator's register-route preparer establishes native append and readback, and names the lane that will append.
   - The coordinator activates this delivery-only scope.
1. **CLAIM.** The appending lane appends the CLAIM row exactly as written in §5 and reads it back by Event ID. I check that readback against the posted text. **Nothing else happens until that is confirmed on this PR.**
2. **Fresh read.**
   - PR head still `7285b60b` and blob still `a06b5ccf`.
   - Any R17 contender rows for both target keys.
   - If anything changed, I rebuild as v3 and post its identity.
3. **Preflight.** I re-read My Drive root's permissions, and bind the observation and its time to the outgoing fields.
4. **Folder.** I create one folder in My Drive root, `main365-integrator-delivery-20261012`, then read back its permissions. If it is not owner-only, I stop and register nothing.
5. **Upload and readback.**
   - I upload the v2 bytes with no conversion.
   - I download them back and compare: they must be 12773 B with sha256 `bf75a70b…`.
   - If they do not match, I stop, register nothing and delete nothing.
6. **Private custody note.** I write one owner-only text note in the same folder and read it back. It holds:
   - the file and folder IDs;
   - the permission observations and their times;
   - the DELIVERY and RELEASE text that will go out.
7. **DELIVERY.** I post the screened DELIVERY row here. The appending lane appends it and reads it back by Event ID.
8. **RELEASE.** I post the RELEASE row for this delivery-only scope, which is complete at that point. It is appended and read back the same way.

Source integration of #365 stays separate and is not activated here. Neither are original-log or artifact access, or any `main` window.

### 5. R17 rows, screened text
No Drive ID, URL, path or email appears in any field. UTC is the actual append time in the form `YYYY-MM-DDTHH:MM:SSZ`.

**CLAIM**
| column | value |
| --- | --- |
| Event ID | `MAIN365-INTEGRATOR-DELIVERY-20261012-CLAIM` |
| Session ID | `session_01S16677s1SGeAJmN3QMz4ip` |
| Provider | `Anthropic / Claude` |
| Target key | the §1 key |
| Event type | `CLAIM` |
| Claim reference | `MAIN365-INTEGRATOR-DELIVERY-20261012-CLAIM` |
| Lease until UTC | append time + 120 min |
| Source SHA-256 | `CUSTODY.md a4584351b5b6552f107f75424022c1bc6744a1fae791abdf0930f6d8a2a24288 (blob a06b5ccf); planned bundle bf75a70bbe65b14eddea4a4b94bbf53d3f32bf44a577a17fc574d18fbe804c7a (12773 B)` |
| Expected old head | `7285b60b078f5c3d3adce2e50f400864572ea024` |
| Proposed new head / artifact | `https://github.com/d6g8k5htny-coder/main/pull/365` plus this comment's id |
| Status | `ACTIVE_DELIVERY_ONLY / SOURCE_CLAIM_c2676267_UNRESOLVED_NOT_SUPERSEDED / NO_MAIN_INTEGRATION_WINDOW` |
| Notes | `Delivery-only scope: one integrator-produced private bundle from public GitHub bytes; not CUSTODY.md ownership, not C267's delivery or release; no ZIP/log access; source-exposed AI, org 0, no human review.` |
| Domain | `Engineering provenance / scienceNONE` |

**DELIVERY**
- **Event ID:** `…-DELIVERY`.
- **Same as the CLAIM row:** Session ID, Provider, Target key, Claim reference, Expected old head and Domain.
- **Source SHA-256:** `bundle bf75a70b… (12773 B); manifest 0bb5a7b6…; payload 52a04902…; CUSTODY.md a4584351…; SOURCE_RECHECK 4bae7ca3…`.
- **Artifact:** `PRIVATE_CUSTODY: owner-only Drive bundle; locator withheld from this public register` plus the GitHub delivery-comment URL.
- **Status:** `DELIVERED_PRIVATE_CUSTODY / BYTES_READBACK_MATCH / C267_DELIVERY_UNKNOWN / SCIENCE_NONE`.

**RELEASE**
- **Event ID:** `…-RELEASE`.
- **Expected old head:** `Own claim MAIN365-INTEGRATOR-DELIVERY-20261012-CLAIM only`.
- **Status:** `OWN_DELIVERY_SCOPE_RELEASED / C267_SOURCE_CLAIM_UNCHANGED`.

Any row that differs from this text is posted again here before it is appended.

### 6. Stops
- the register contract rejects the key;
- no lane can append and read back;
- the head or blob drifts;
- an owner-only audience cannot be shown;
- the readback does not match;
- an R17 contender appears.

Each of these stops only the affected step. Nothing is deleted, and I fall back to reporting it here.

---
_Generated by [Claude Code](https://claude.ai/code)_
