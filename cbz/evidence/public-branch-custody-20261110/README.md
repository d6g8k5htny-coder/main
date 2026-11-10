# Bounded public branch custody — 10 November 2026

This record preserves selected native Git objects from an earlier, non-atomic
branch-ref observation. It covers 58 captured heads in seven **public**
repositories: Universal-Law-Workspace, meta-framework, query-, google-drive,
governance-, sandbox and d6g8k5htny-coder. It excludes private repository refs
and does not claim a complete federation inventory. See the filtered
`federation-branch-head-trees-20261110.json` for each exact head, default,
observation time and historical ancestry classification. The original
aggregate's SHA-256 is recorded there; the aggregate itself is not published
here. Workspace's captured `d486c55d` head was classified non-ancestor before
it subsequently merged as #66. No classification is a current-ref assertion.

For each repository, the commit/tree pack contains the captured head commits
and their recursively referenced Git tree objects, without blob contents.
The separate blob pack contains **only** head-side blob objects at paths whose
mode or object identity differed from the separately observed default, for
heads then classified non-ancestor. It preserves 301 distinct per-repository
blob identities: 47 Workspace blobs (387,103 raw bytes) and 254 across the
other six repositories (2,188,792 raw bytes). Their Git SHA-1 identities,
SHA-256 hashes and sizes are in the manifests. A Gitlink is a commit reference,
not a blob; this record does not preserve Gitlink target contents. Deletions,
unchanged branch files and default-side blob bytes are not recovered by these
selected blob packs. Neither path changes nor object differences establish
semantic uniqueness, authorship, proof status, logical isolation, branch
ownership or a deletion disposition. No branch or tag was removed.

The workspace inventory covers 12 captured refs and 275 changed path bindings
among six historical non-ancestor heads. The six-repository inventory covers
46 captured refs and 1,976 changed path bindings among 31 historical
non-ancestor heads. The latter's counts and pack digests are in
`small-branch-tree-20261110/summary.json`. All observed ref/default reads were
separate, not atomic; fresh ref and commit checks are necessary before acting.

## Verification

Run from this directory in a fresh temporary copy, keeping the output and exit
of each command. Python standard library and Git are required. The scripts are
bounded to the captured input schema and source identities; they do not audit
the full federation or prove source semantics.

```sh
CBZ_BRANCH_CHECK_DIR=$(mktemp -d) || exit 1
cp -a . "$CBZ_BRANCH_CHECK_DIR/branch-custody" || exit 1
cd "$CBZ_BRANCH_CHECK_DIR/branch-custody" || exit 1
sha256sum -c SHA256SUMS || exit 1
python3 -B -S cold_restore_small_repository_packs.py || exit 1
python3 -B -S test_workspace_branch_tree_adverse.py || exit 1
python3 -B -S test_workspace_changed_blob_adverse.py || exit 1
python3 -B -S test_small_repository_tree_adverse.py || exit 1
python3 -B -S test_small_repository_blob_adverse.py || exit 1
```

The six-repository cold restore checks each tree inventory against Git's
`diff-tree` and each selected blob's Git and SHA-256 identity in normal and
optimized Python: 24 successful readbacks were observed. Each pair of adverse
drivers exercises both modes, clean controls, corrupted inventory or manifest
inputs, a deliberately weakened verifier and missing/corrupted Git packs.
The original Workspace and query campaigns each observed 28 hostile-report or
manifest rejections, four detected mutant executions, four damaged-pack
rejections and six clean controls. These are the specified controls, not a
claim of exhaustive verifier soundness or a final G0/G10 pass. The actual
source scripts, corrupted fixtures, outputs and exits are retained in
`verification-history.tar.gz`. The unfiltered original observation and other
repository evidence remain in the separate durable checkpoint.
