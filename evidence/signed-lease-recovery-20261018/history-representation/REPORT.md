# Bounded author-history recovery

Result: one accepted budget representation recovered by JSON decoding from the exact specified author session. Its 6,311 UTF-8 bytes match accepted SHA-256 `9db25a5e4d1c43e17d96fe92de4fa4f0ee33c8e0c13faae506dfe51090241d2c` exactly. The saved artifact is `RECOVERED-BUDGET.utf8`.

Classification: **session-record representation; private original custody UNKNOWN**. The byte/hash equality identifies the accepted representation. It does not authenticate a raw original file, establish author approval, or advance a product, design, harness, runtime, or scientific gate.

## Method and bounds

The filename locator found only the specified Root session. Inspection was restricted to its October 11 and October 13 records and the accepted source references. Inert JSON parsing inspected direct strings, complete nested JSON strings, and complete JSON string tokens. No historical calls or code were executed or evaluated; no patches were replayed; no files were reconstructed or repaired; no other account history was scanned. No original project file was written.

The response-item scan inspected 4,521 eligible records and made 210,285 candidate-string checks. The auxiliary item-completion/compaction scan inspected 6,095 eligible records and made 936,523 candidate-string checks. Counts include duplicate representations. Only the budget matched an accepted target hash in these bounded checks. This is a negative result for the stated decoding route, not proof that no other representation exists anywhere.

## Representation source pin

| Opaque source | UTC timestamp | Stored representation | Raw record SHA-256 |
|---|---|---|---|
| R-BUDGET-01 | 2026-10-13T13:17:00.964Z | Entire JSON-decoded output string, 6,311 bytes | `74462bccac379c5b574b2224b65204cc674ba25b954fc64b9ff0d4acc0feccfe` |

The accepted payload came directly from a single stored string. There was no trimming, newline conversion, concatenation, execution, or inferred content. Fresh readback of the saved file repeated the accepted byte count and SHA-256 and confirmed valid UTF-8/JSON. Private source paths, record identifiers, offsets, and packet locators are retained only in the `PRIVATE-*` tables. The live author session was growing during inspection; exact raw-record pins preserve the selected historical record identity separately from the scan-time session snapshots.

## Route-level status

| Target | Accepted bytes | This history route |
|---|---:|---|
| S01 tests | 179,658 | Private packet locator found; no complete matching literal recovered |
| S01 contract | 101,664 | Private packet locator found; no complete matching literal recovered |
| Amended PLAN | 74,178 | Private packet locator found; no complete matching literal recovered |
| Amended budget | 6,311 | Exact accepted representation recovered; raw original custody UNKNOWN |
| Amendment manifest A | Unspecified | Private packet reference found; no complete matching literal recovered |
| Amendment manifest B | Unspecified | Private packet reference found; no complete matching literal recovered |
| Accepted inventory | 284,340 | Private packet reference found; no complete matching literal recovered |

This route leaves the original-file authentication gate open. The parent reported a separate successful local-file recovery of all seven accepted targets; that separate evidence was outside this subtask and is not independently assessed here. No further history expansion was performed after that coordination update.

The bounded history subtask is complete and released. All broader ownership and locked gates remain under the parent task.
