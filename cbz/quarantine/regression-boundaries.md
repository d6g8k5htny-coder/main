# Exact regression quarantine proposal

This is a source-bound proposal for canonical integration. It does **not** establish G5, CBZ-0, or mathematical closure. No test, source assertion, record or production schema was edited.

## Archived trial

Source `3738c859d519d2dbf1e388fe48fe0ae60c9487a3`; both fresh copies hash `75b85eb0425290f6375d71e583216e75a01c205cb9f6aa86b3a1b1fd38740a00` across6171 tracked files. Unchanged pytest:1355 pass,87 fail,1 skip per mode; exit1 each. All87 failed nodeids and failing assertion strings agree between modes.

| Group | Count | Observed boundary |
|---|---:|---|
| Unavailable live-tip/state assertion | 59 | The isolated execution returned missing live tip/tip_match or a derived state; it cannot validate a live remote observation. This is an environmental observation boundary, not proof of a source algorithm defect. |
| Direct network or network-fallback command failure | 16 | The failing command/log requires GitHub clone, live tip fetch, ls-remote or an API-to-raw fallback while the recorded namespace has no external interface/routes. Source correctness after network admission is not established. |
| Archived README text drift | 3 | Unchanged tests require exact historical README strings absent from the archived current README; this is a reproducible source/fixture mismatch independent of transport. |
| Token-discovery isolation mismatch | 1 | The recorded harness intentionally sets PATH_C_IGNORE_FILE_TOKENS=1 to prevent durable credential discovery. The archived test expects discovery of its fake token file; the observed false return is not evidence that permitted production token discovery is defective. |
| Writable-parent privilege fixture mismatch | 1 | The test chmods a parent to 0555 and expects it unwritable. The user namespace runs as namespace root; the selected output remains under that parent rather than TMPDIR. The observed fixture assumption fails; behavior as an ordinary unprivileged UID is not observed. |
| Conflicting historical fixture pins | 6 | Archived fixtures carry mutually different historical SHA values and tests require their equality/prefix. This is a reproducible fixture inconsistency; no live update or invented receipt is used to replace the historical bytes. |
| Archived idle-state mismatch | 1 | The test requires IDLE_PATH_C_DONE, while the unchanged status result is PATH_C_LANDED. The log establishes the mismatch; a source defect vs derivative unavailable-live-tip classification is not inferred. |

The single existing skip is `tests/test_intent.py::test_batch538_tip_sync_watch_keep_prior_parent_pin`: `portable/BATCH538_TIP_SYNC_IDLE.json` never landed. No receipt is fabricated.

The immutable `trial` repository is already archived. Keep the complete original suite and red evidence recoverable; replay it only in the recorded credential-free namespace, with no external routes and original federation clones mounted read-only. A terminal archival quarantine must remain visibly red and must not be represented as a federation-wide full-suite pass.

## Main real-record coverage

At source `c3377cc8c0fb35844dc7613976d6e746fdc05600`, unchanged39-method module:34 exercised,5 explicit skips per mode. All four real records have owner:null; none has optional evidence URLs, dependencies or artifact citations. `depends_on` and `artifacts` are rejected by production interfacev0, so manufactured records would not close coverage.

| Exact method | Missing observation |
|---|---|
| `RealRecordTests.test_real_records_evidence_id_pattern` | No real record has evidence_url, released_url, or resolution_url. The synthetic URL/fault checks are exercised; URL pattern checks on genuine production evidence are not. |
| `RealRecordTests.test_real_records_owner_released_url_consistency` | All four real records have owner:null. Synthetic owner/fault checks are exercised; owner identity, assignment evidence and release-URL consistency on genuine records are not. |
| `RealRecordTests.test_real_records_dependencies_when_present` | No production record has depends_on. Interface v0 rejects that key; synthetic dependency checks do not establish an admitted real-record dependency graph. |
| `RealRecordTests.test_real_records_cited_artifacts_when_present` | No production record has artifacts. Interface v0 rejects that key; synthetic path checks do not establish artifact resolution for admitted real records. |
| `RealRecordTests.test_real_records_fresh_against_live_source_head` | All four records say fresh, but the method unconditionally skips after listing them; no live-source freshness comparison is executed. A local Git comparison found all cited blobs unchanged at Math8151 except eng.hard-gate cited directory tree715449bbe16b471f68c4913d08e72baac903582b versus132725157175a85dbbd5a1cfc2c668fc7b518283; repository HEAD equality differs for every historical pin. One lifetime source is a mutable comment not observed by local Git. |

The freshness method skips even for records labelled fresh. Local Git evidence found the hard-gate cited directory tree changed, while the source-refresh CLI exits0 even on stale/unknown. No current-live-state claim is admitted.

## Proposed quarantine records

The JSON supplies every §11 field for six records (`Q-9001`–`Q-9006`; root may reindex without changing boundaries), exact87 failed nodeids, assertion source lines, hashes, honest blockers, attempts, resumption requirements and isolation status. Root must bind them to the canonical index; they are not silently terminal merely because this file exists.

The analysis script is one-off evidence assembly, not a gating auditor or proof certifier. Exact source and raw log hashes are recorded in JSON.
