# Research snapshot reconciliation and native export fidelity

This additive delta follows Git head `508ae7e20301df57da0045b13bc7ed9eb0b67ead`.
It retains the 2026-09-19 inventory, source bodies, content-addressed packs
and coverage ledger unchanged. The original 4,456-object inventory remains
immutable. The scoped current view has 4,928 objects: 4,116 stored file
identities, 758 folder records and 54 held contents (40 DO_NOT_OPEN and
14 unrelated personal). This is the research scope, not the entire account.

All 744 permitted folders were relisted between 2026-09-20T00:50:20.927Z
and 2026-09-20T00:50:55.975Z. No prior observed identity or folder membership
was missing. Four added files are the preceding run's delivery outputs;
they were fetched as raw bytes and matched against their custody receipts.
Only the live register had changed metadata. Its new XLSX is stored with a
post-export revision observation, not a revision-pinned or historical backup.
Root membership and four previously mirrored quarantine notices receive
separate metadata-only checks. Protected and personal subtrees were not
opened. Previously preserved quarantine notices remain zero-authority data.

`previous-membership.json` reconstructs the preceding run's observed folder
memberships from its retained normalized metadata; its provenance is explicit
in `reconciliation.json`. The current connector reports null parent fields.
The checker uses the folder in which a file was actually listed, retaining
the distinction between missing, null, empty and populated metadata fields.
Neither absent metadata nor an unobserved membership is called a deletion.
The listings are live observations, not an atomic Drive snapshot.

The export formats are DOCX for 1,930 native Google Docs and XLSX for
39 native Sheets; there are no native Slides in the stored research scope.
The native audit checks every document's tab topology and all 308 workbook
tabs. It reads formula fields across each workbook's full declared grid and
preserves all 14,338 native formula strings in
`native-topology-formulas.json.gz`, alongside the native titles and IDs.
The DOCX/XLSX ZIP containers and their primary XML bodies are reparsed in CI.

An export is not lossless native behavior. One native tab title is shortened
in XLSX; some unshared formulas are rewritten, including open-ended ranges
and array wrappers. Shared-formula XML uses master references and is counted
as such, without calling an empty shared-formula element a lost formula.
Original native strings are preserved even where Excel syntax differs.
The complete deterministic report is `native-fidelity.json`; detailed export
structure is `export-structure.json.gz`. No semantic equivalence of transformed
formulas is asserted. All observed Docs have one tab; this rules out a missing
second-tab defect at the observed topology, but DOCX text is not compared
character by character with the native body. Comments, suggestions, Apps
Script, named native behavior and revision history are not comprehensively
backed up here. A metadata/revision observation is not an atomic export lock.

Run `python tools/drive_reconcile.py` for the derived current view and
`python tools/native_export_check.py --verify-containers` for the native
comparison. Baseline bytes remain independently checked by
`python tools/drive_coverage.py --json`; manifest identity is checked in CI.
Tests mutate missing tabs, formula cells, partial grid reads, source timestamps,
folder enumeration, and unknown metadata to ensure these failures are visible.

The declared membership cutoff excludes this run's reconnaissance memo,
candidate, context synthesis and later delivery receipts. Those are outputs
listed in a separate delivery manifest, avoiding recursive self-ingestion.
Four historical containers with bounded decode failures and one depth-limit
finding remain preserved as bytes in the preceding snapshot, with their
limitations unchanged. Nothing in this delta executes imported archive code,
changes a scientific status, supplies independent review or claims a theorem.
