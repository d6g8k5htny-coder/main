# Selected historical Main and Math source custody

This record preserves exact native Git objects selected from an earlier,
non-atomic observation of public Main and Math branch heads. The source file
`source.json` contains the 430 captured refs then classified non-ancestor:
278 Main and 152 Math. It binds each ref to its separately observed default,
head and tree identity. It omits private repositories and records the SHA-256
of the original wider aggregate; the wider aggregate itself is not published
here. Ref ancestry and default tips may have changed since the observation.

The selection is deliberately literal and case-sensitive: changed head-side
Git blob paths ending `.lean` in Main and `.lean` or `.md` in Math. It covers
2,857 Main head/path bindings to 588 distinct blob IDs and 4,010 Math bindings
to 608 distinct blob IDs. The packs preserve 11,530,786 raw blob bytes, along
with the captured head commits and recursively referenced tree objects. Each
binding retains exact path bytes, head, mode, Git blob ID and change status;
per-object SHA-256 and byte lengths are in the manifests. Gitlinks, deleted
default-side content, unchanged branch files, other suffixes and private refs
are outside the selection. A suffix does not classify a file as a theorem,
proof, quarantine, unique work or scientific acceptance. No ref is deleted.

## Verification

From the Main checkout, run the documented commands in a disposable copy. Git
and Python's standard library are required. The cold readback checks the
captured ref/head/tree bindings, independently replays the exact Git tree
comparisons, reselects every supported suffix and checks all selected blob
bytes by Git SHA-1 and SHA-256 in both Python modes. The adverse driver uses
an actual captured Math head with explicitly altered selections and source
records, verifier mutants and damaged packs. Its scope is limited to those
specified controls; it is not a whole-federation auditor.

```sh
CBZ_TEXT_CHECK_DIR=$(mktemp -d) || exit 1
cp -a cbz/evidence/public-selected-text-custody-20261110 "$CBZ_TEXT_CHECK_DIR/text-custody" || exit 1
cd "$CBZ_TEXT_CHECK_DIR/text-custody" || exit 1
sha256sum -c SHA256SUMS || exit 1
python3 -B -S cold_restore_captured_text_blobs.py main || exit 1
python3 -B -S cold_restore_captured_text_blobs.py Math- || exit 1
python3 -B -S test_captured_text_blobs_adverse.py || exit 1
```

Earlier unbundled readbacks passed in both modes for both repositories. Their
specified one-head adverse campaign observed two clean passes, 24 malformed
selection or manifest rejections, two detected verifier-mutant executions and
two damaged-pack rejections. Those original scripts, fixtures, outputs and
exits remain in `verification-history.tar.gz`. A later fresh-copy run of the
packaged source-ref/default and tree-identity verifier passed the four cold
readbacks, then observed two clean passes, 30 specified altered-input
rejections, four detected verifier-mutant executions and two damaged-pack
rejections. These checks do not award G0, G10, proof closure or scientific
review.
