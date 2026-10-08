# C215 source-directory isolation repair

This narrow engineering increment preserves the pinned graph/gate/source records and has scientific effect NONE. Main307 pickup6069672514 and its exact-filename amendment6069694054 reserve the new casefold probe, external controls and exporter directory traversal. Existing PR320 helper/workflow changes and the Task2 evidence adapter remain separate active scopes.

The prior lexical output guard does not establish physical source-directory isolation on a filesystem that treats different case spellings as the same directory. C215's read-only directory probe demonstrated equal device/inode identities for differently cased paths without running the original exporter. That is a portability finding, not evidence that the first Linux/ext4 hosted export corrupted its pinned input.

The repair will keep the anchored source-directory descriptor open through output promotion. Every anchored output-directory component must be checked against that descriptor's device/inode identity. Open existing components first; only create a genuinely missing component after confirming that its current parent is outside the protected source. Check the newly opened component before advancing. An aliased source ancestor therefore refuses the operation before creating a child, temporary file or replacing graph.json. Preserve source-path parent components so an earlier symlink cannot disappear during lexical normalization before its no-follow check.

The dedicated hosted probe uses the already pinned Python image, a disposable64MiB ext4 casefold filesystem and original exporter CLI inside a nonroot/no-network/read-only-rootfs container with bounded CPU, RAM, processes, wall time and temporary storage. The actual Git checkout is read-only; only TEST source copies and output are writable. The casefold root is configured by the disposable controller before any product execution. Unsupported preparation or unavailable casefold identity fails rather than skipping a control.

Ten external controls cover differently cased same-source and existing/new descendant outputs, exact same/descendant lexical guards, a distinct sibling positive, source/output symlinks, and an earlier symlink followed by a parent component. Before each CLI call, anchored no-follow descriptors must prove directory and GRAPH.json/graph.json alias equality and the three original pinned source hashes. Negative controls preserve raw bytes, original file/directory identities and directory entries. A stdlib audit observer captures attempted writes, mkdirs and promotion; negatives require no such attempts, while the sibling positive requires actual temporary creation and rename to prove that the observer works. No original CLI or test executes on the laptop.

The inline container runner compares the discovered inventory against ten frozen complete IDs before execution. It requires exactly ten executed controls, native success, and no skips, expected failures or unexpected successes in both normal and optimized modes. Both modes share the same frozen inventory. Logs, the original runner script, native identities, runtime image and casefold attributes are retained after independently confirmed container disposal. Publication staging refuses symlinks and special files, hashes a fixed allowlist, preserves failures and requires strict filesystem release.

- [x] Write ten external CLI controls without changing the original implementation.
- [x] Add bounded hosted casefold preparation and exact-discovery acceptance.
- [ ] Complete fresh source/security review of final test/runner bytes.
- [ ] Observe original-implementation hosted RED and retain original identities/logs/archive bytes.
- [ ] Implement physical directory identity and no-follow traversal correction.
- [ ] Verify all ten controls in normal and optimized modes and existing exporter controls at the exact candidate.
- [ ] Complete independent source/runtime review, reconcile actual current main without overwriting peer scopes, and integrate the checked repair.

This increment alone does not complete the eight requested outcomes, enforce a private lease boundary, establish mathematical predicates, or migrate every existing execution entrypoint.
