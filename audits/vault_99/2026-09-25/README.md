# 99_DO_NOT_OPEN forensic census — 2026-09-25

Owner-directed one-time audit. **Do not use this directory as mathematical authority.**

## Freeze
Drive vault: `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`.

During this pass no item is to be added, moved, renamed, deleted, restored, promoted, or used as authority. OpenAI/ChatGPT in the owner-directed session is sole auditor; assistance requires an explicit named delegation recorded in GitHub issue #91.

## Baseline
Provider direct-child enumeration on 2026-09-25 returned **39 direct children** when the folder itself was fetched. An earlier generic Drive search exposed only 26 and was incomplete; it is not used as the census boundary. `baseline.csv` now records all 39 direct children and preliminary classifications.

## Important early finding
The vault is not pure disposable clutter. Multiple `CENSUS_DUMP_NOT_HOLD` files contain historical mathematical maps, claim inventories, RN/JETMOD state, code/ZIP inventories, and source bindings. They remain non-authoritative, but unique facts must be diffed against current Git/Drive sources before final disposition.

## Manifest inconsistency
The initial generic Drive search was incomplete: it omitted 8 census CSVs and 5 logged binary/code vault items. Exact-ID checks and direct folder fetch verified those omitted objects are still parented in the vault. The live manifest's 23 census-dump claim matches the direct folder enumeration. This is a provider-enumeration pitfall, not evidence of deletion.

## Classification vocabulary
- SAFE_SUPERSEDED
- DUPLICATE_WITH_VERIFIED_SUCCESSOR
- IMPORTANT_EXTRACT_REQUIRED
- UNIQUE_RECOVERY_REQUIRED
- PROVENANCE_KEEP
- NEEDS_REVIEW

No classification promotes scientific status.

## Next checks
1. Reconcile live manifest rows against actual parent membership by exact Drive ID.
2. Diff `FULL_MATH_CLAIM_INVENTORY`, `LANE_MATH_MAP`, `LANE_RN_UNIF`, `FULL_DOCS_MATH_READ`, and code/ZIP census against current Git.
3. Bind any unique surviving fact to a modern Git source record without reviving stale statuses.
4. Only after every baseline item is terminally classified, release the vault freeze and install register-before-move intake.
