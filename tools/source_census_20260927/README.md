# Offline source census

Inspect acquired files once by content hash, without executing their code. This is a source-discovery tool, not a proof checker, cloud synchronizer, or security sandbox.

## Run

```sh
python tools/source_census_20260927/source_census.py \
  --root dropbox=/absolute/path/to/downloaded/files \
  --out /absolute/path/to/census-output \
  --cache /absolute/path/to/census-cache \
  --term TRANSVERSE_CONTACT_ASYMPTOTIC \
  --term rnu_env.py
```

Repeat `--root LABEL=PATH` for other already-acquired source directories or individual files. Keep output/cache outside input directories. The CLI resolves the explicitly supplied root path; directory/file symlinks encountered beneath it are not followed. Acquisition, credentials, cloud pagination, and permission handling stay outside this tool.

The first run parses each distinct payload once; subsequent runs reuse SHA-256-keyed extraction records with a parser-profile tag. Every occurrence retains its original provenance, including `outer.zip!/inner.zip!/member`. Raw bytes are rehashed; a cache hit is not independent evidence or proof of semantic correctness. Use a fresh cache for independent extraction or after changing optional parser versions.

## Output and scope

- `inventory.jsonl`: every inspected file/container occurrence, bytes, SHA-256, Git blob identity, content type, text digest and extraction status.
- `term_hits.json`: filename/content hits; references to a missing file do not recover that file.
- `issues.json`: read errors, unsafe or ambiguous members, and size/depth limits.
- `summary.json`: counts and parser availability, with explicit local-only scope.

Exit 2 reports recorded read/container issues. **Exit 0 does not mean every format was understood:** always inspect statuses such as `unsupported_binary`, `parser_unavailable`, `ooxml_partial`, `text_truncated`, and visual-review-pending. A match against a supplied inventory proves no global remote absence; a historical source-map locator is not a fresh byte check.

Signatures, not filename extensions, select ZIP/TAR/gzip, Office Open XML, binary plist, PDF, common images, and UTF-8/UTF-16 text. Office math XML is included in the reading extract; layout/equations may be lossy. Embedded Office media/objects are separately inspected where found. Macro presence is flagged, never executed. Pickles, bytecode, executables, disk images, legacy Office binaries and unsupported archive types are not run or deserialized as executable objects.

Optional PyMuPDF (`fitz`) provides PDF text/page/image metadata; optional Pillow provides image metadata. No OCR, visual understanding, formula evaluation, notebook execution, or mathematical acceptance is supplied. Standard-library operation remains available with missing optional parsers explicitly marked.

Defaults: 128 MiB root-file parse limit, 64 MiB per member, 256 MiB per container, 20,000 members per container, nesting depth 8, and 32 Mi characters per extracted text. These are per-object/container bounds, not a global operating-system resource sandbox. Larger root files are still hashed, but marked uninspected. Enforce additional process/memory limits for adversarial inputs.

## Tests

```sh
python -B -S -m unittest discover -s tools/source_census_20260927 -p 'test_*.py' -v
python -B -O -S -m unittest discover -s tools/source_census_20260927 -p 'test_*.py' -v
```

Twenty fixture tests passed in each interpreter mode locally. Test-first controls exposed missing functionality and three later defects: UTF-16 entity detection, duplicate Office XML members, and a cache placed under the input root. Three deliberate regressions (traversal guard, UTF-16 declaration guard, duplicate-XML guard) were rejected in both modes: six mutation runs, all failing the intended behavior tests rather than imports.

The dedicated PR workflow runs standard-library fixtures on Python 3.11. The actual research census was run separately with optional parsers installed. Neither test suite is a rerun of historical mathematical verification programs or the whole existing repository suite. No archived research code was executed for this census.

[Current dated findings](../../docs/public-math/recovery-20260927/SYSTEMATIC_PASS.md). Full private inventories and preserved historical sources are filed in the linked Drive audit folder, not bulk-published as research claims.
