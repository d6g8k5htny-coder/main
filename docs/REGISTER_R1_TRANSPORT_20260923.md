# R1 register export is now in Git: exact-byte transport complete

This delivery addresses the binary-transfer blocker raised on PR3 and PR21. It stores the immutable R1 XLSX and appends its provenance. The existing Sept18 import selector and canonical JSON/CSV outputs are deliberately unchanged; their coherent migration is next. No scientific claim, proof, review verdict or independence credit changes.

## Exact source

- Path: `registers/source/GP-REG-032_v1.2_export_2026-09-23_R1.xlsx`
- Bytes: 1,979,318
- SHA-256: `6cec54cefa3ee8f9885dcfd763407a2e0274d2887d73bdc23ccc7793cd721469`
- Git blob: `e11a78762c7b3be010dddb864c1c9357e387b09e`
- Immutable Drive export: `1XU1OaiXJzNdzxJc0AwayVP37irnDsC7f`
- Native source: `1O6x8ivmaVUxYqKmCOXmToIHpMXqDI362ibqBl8HY8no`, observed modified 2026-09-23T19:36:14.667Z; export observed 19:37:42Z. This snapshot precedes later Work Events.

## Executed verification

GitHub Actions run `35918474959`, job `107376036654`, completed successfully on Python 3.11.16 / zlib 1.3. It reconstructed the raw export exactly, verified ZIP CRC and all 153 members, then ran the unchanged repository importer from commit `b02efe21d8d7475ac9b4aa25b3dcc26a5b6f4bf4` (SHA-256 `e263f8600571648b84bf575f380cc94fce3395ff12eeb093b9a493ee1e2555cb`). Two generations produced identical 44 JSON plus 44 CSV files. The importer's check passed, rejected altered JSON, missing CSV and extra stale JSON, then passed again after restoration. Source and payload mutations were rejected before reconstruction. Original source and importer were unchanged.

Review Queue contains 40 rows and exactly one each of `RV-H3-SOLVER-20260921-01`, `RV-WITHDRAWAL-20260921-01` and `RV-ROLLOUT-R2-20260921-01`. These keys are present in the successor export, not yet in the default generated registers.

Artifact `10776430936` was downloaded through the authenticated connector; its embedded XLSX matched the original Drive R1 byte for byte. Artifact ZIP: 1,981,837 bytes, SHA-256 `62dcc73811cf0fe06263652db9c55427234639448f528bfe060cb564f7c9887c`. Its raw `REPORT.json` is 2,209 bytes, SHA-256 `755141af41e89baac4aa6cd48c2dce3754a417fbd8a382f4a3140b3df87aeb36`.

The one-off `transport/register-r1-20260923` branch is not to be merged into research or main. Its job created only the expected Git blob and no refs. The transport scripts and chunk payload are operational aids, not mathematical sources.

## Nonauthor integration pickup

The binary capability blocker is resolved. A Git-only agent can now fetch this branch, inspect the exact successor, run `tools/registers_import.py --source` into temporary outputs, and make a tested selector/canonical-output migration. Preserve all earlier export bytes and keep the historic Sept17-to-Sept18 diff bound to those original snapshots. After canonical migration, the held H3 review record can be checked against its actual route key. Do not claim that review filing has already happened.

Authorship: OpenAI / ChatGPT. Scope: source custody and reproducible transport only. Full repository regression of this three-file delivery is not asserted here; its PR CI is separate from the successful transport run.
