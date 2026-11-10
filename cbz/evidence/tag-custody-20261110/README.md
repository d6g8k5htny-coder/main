# Historical tag source custody — 10 November 2026

This record preserves selected historical Git objects and source files from
seven tags whose commits were not ancestors of the separately observed default
heads. It does not close a proof, admit a quarantine, establish logical
isolation, or assert scientific acceptance. The tags remain in place. The
separate Main custody record already covers the three cusp tags.

At Math `9bdfc54b76173b57912503bfcf901a5f9d43b09c`, a native Git tree
difference from common base `4a0d51bbd06f0d9ac731a1d1df38f25999797135`
matched the separately retrieved comparison for the five annotated tags:

| Tag | Tag object | Target commit | Added paths |
|---|---|---|---:|
| `cbz-quarantine-q-7522` | `f55f2304f4788b759482eee8e3fc70f01211ab82` | `223293d9d26a7a4158026a4225523e8a683a8794` | 17 |
| `cbz-quarantine-q-7523` | `e30449ce79444baaf6607d03659f3c69db19be9a` | `de872d97fa54eb11c0834b8f532543222900d165` | 25 |
| `cbz-quarantine-q-7524` | `adc5c7c9219999b7e952b796b4fb72601c51bcd5` | `5a30296bfc6c9c886d483c87e80e2eb2d9ea047d` | 34 |
| `cbz-quarantine-q-7525` | `e7a6918da1ef61402fd412de9d4e5d2a39456fdd` | `fc5ecb67f4a9d6cfb1dec066499ca2daa1147b3b` | 43 |
| `cbz-quarantine-q-7526` | `86651fa51da980dea8321f8a7b9c3626116b70bf` | `fdddc7e6b1fd584fa566beaccedd64d177104d10` | 52 |

The Math archive stores five tag objects, five commit objects, 32 path-tree
objects, 171 tag-to-path bindings and 52 distinct source paths (46 distinct
Git blobs); the exact 52 paths were absent from the observed Math tree.
The files and their historical statements, authorship and outcomes are
preserved as bytes, without adopting their claims as current dispositions.
The accompanying machine-readable document inventory identifies nine selected
mathematical manuscripts, their 26 tag/commit bindings and 76 literal Markdown
headings. It is navigation data, not a theorem enumeration or an admitted
quarantine; the companion verification files retain the original authorship
and historical outcomes.

At Main `da92c8290d99b7ab1db686ce9294e20c915f01a7`, the lightweight
`archive/drive-migration-20261002` tag pointed to
`b6ccb5e2363dfe859973bf960c494c65488b8bc2`. Its native comparison against
`b1335def2615888463e2d134294ef451895b8d0d` produced nine changed paths, all
absent at those exact paths on the observed default. Annotated tag
`integrate-697-after-am-9e8d634a` had object
`1203b44eeec56e0bc8b4c712368dc39178c7e09e`, pointing to
`e451fa55363df0bf1c1341691c95187322fbb669`; its comparison against
`8958ee09a37d652ad2f533e10e4925367e0361bc` produced two changed paths, one
with the same current blob and one different. The Main archive stores the
tag object, two commits, 13 path-tree objects and all 11 file bindings.
These are selected object and source archives, not complete Git histories.
The JSON records here preserve the bounded comparisons and their locations.

## Reproduction and limits

Run from the Main repository root with Python 3, retaining both output and
exit status:

```sh
python3 -B -S cbz/evidence/tag-custody-20261110/verify_math_tag_native_custody.py cbz/evidence/tag-custody-20261110/math-tag-native-custody-20261110.tar.gz
python3 -B -O -S cbz/evidence/tag-custody-20261110/verify_math_tag_native_custody.py cbz/evidence/tag-custody-20261110/math-tag-native-custody-20261110.tar.gz
python3 -B -S cbz/evidence/tag-custody-20261110/verify_main_tag_native_custody.py cbz/evidence/tag-custody-20261110/main-tag-native-custody-20261110.tar.gz
python3 -B -O -S cbz/evidence/tag-custody-20261110/verify_main_tag_native_custody.py cbz/evidence/tag-custody-20261110/main-tag-native-custody-20261110.tar.gz
```

The one-off verifiers are anchored to the seven recorded tag identities and
check the supplied archive members, source bytes, modes and Git path-tree
bindings. They do not independently prove completeness of the changed-path
comparison or classify the content. `SHA256SUMS` binds the archives and
verifier sources as preserved here. The companion `verification-history.tar.gz`
retains the initial verifier scripts, false-green and red/green logs, two
retained corrupted archives and designated code-mutant sources and outputs.
`canonical-validation.tar.gz` separately retains the first incorrect checksum
command, its correction, reader-link results and STATUS command outputs.

The initial scripts used Python `assert`: optimized execution accepted damaged
source, and certain duplicate or relabeled bindings were also accepted. The
repair uses explicit checks. For the two exact archives, the subsequent
32-child input campaign included four valid controls and 28 specified adverse
inputs in normal and optimized modes; all four controls passed and all 28
adverse inputs were rejected. Two designated helper-code mutations were caught
for both archives in both modes (eight children). These are bounded tests, not
an exhaustive qualification of the verifier or a federation-wide G0 result.
Reproduce the campaigns in a new temporary directory, so generated mutant
archives do not affect the checkout:

```sh
CBZ_TAG_TEST_DIR=$(mktemp -d) || exit 1
CBZ_TAG_SOURCE=cbz/evidence/tag-custody-20261110
for file in math-tag-native-custody-20261110.tar.gz main-tag-native-custody-20261110.tar.gz verify_math_tag_native_custody.py verify_main_tag_native_custody.py tag_custody_verifier_common.py test_tag_custody_adverse.py test_tag_custody_code_mutations.py; do
  cp "$CBZ_TAG_SOURCE/$file" "$CBZ_TAG_TEST_DIR/" || exit 1
done
python3 -B -S "$CBZ_TAG_TEST_DIR/test_tag_custody_adverse.py" after-repair || exit 1
python3 -B -S "$CBZ_TAG_TEST_DIR/test_tag_custody_code_mutations.py" || exit 1
```

Both commands must be observed separately. A successful archive readback or
adverse campaign is source-custody evidence only; it does not validate the
historical mathematical contents or establish independence of any review.
