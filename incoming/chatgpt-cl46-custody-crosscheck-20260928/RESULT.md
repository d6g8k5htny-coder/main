# CL-MIRROR-001: post-publication custody cross-check

Scientific effect: NONE. Review status: REVIEW_REQUIRED.

OpenAI ChatGPT checked the Anthropic-published intake at main commit
`5f22638a8fad05afd3370213943907dc4c050518` against a separately acquired
Dropbox copy of `09152026OKComputer_Project_Gap_Closure.zip`.
The publishers use the same owner's account. This is a different-provider
byte-custody check, not organizational independence or a mathematical review.

## Results

- Archive: 30,148,285 bytes; SHA-256
  `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`;
  all archive members passed the ZIP CRC check.
- All 46 historical manifest expectations matched: 644,546 payload bytes.
- The 45 files in the existing `K3_SIDE24_LB` intake subtree total 636,505 bytes.
  Reconstructing their Git tree from the raw archive members, published names,
  and regular-file mode 100644 yields
  `ad35e8a033c95f2da3f49077bdf3cc712944574e`, exactly the published subtree.
- The remaining `PERC_DECAY.md` is already in the earlier intake: 8,041 bytes,
  Git blob `665db9464f83ab1512e8f1e174a985e43b4bb498`.
- Six mutation controls reject a missing file, extra file, substituted content,
  renamed file, single-byte change, and executable-mode change. Restoring the
  original entries reproduces the expected tree.

`CUSTODY_CROSSCHECK.json` records the complete archive identity, pinned target,
manifest Git blob, separate-file identity, and control outcomes. Hash, size,
and path expectations were transcribed from the pinned historical manifest;
that transcription is not represented as an original manifest byte copy.
No source files are duplicated or edited by this packet. No recovered script
was executed. No probability estimate, proof obligation, historical grade,
scientific status, or review independence is promoted.

## Remaining boundary

The absent September-17 v5 bundle, `rnu_env.py`,
`allcell_fdz_enclosures.json`, numerical 24-jet certificates, and the modern
uniformity/collision obligations are not discharged by this check.
The August-3 publication lane remains with the separate Claude session.

A source-bound reviewer should verify this receipt against the pinned public
manifest and tree. Existing publication and coordination: main PRs #199,
#200, #201 and #202. All historical status labels remain source material,
not new acceptance.
