# Unmerged #695 catalog: exact recovery and non-admitting comparison

The source is Main #695 head `82d9e480a51b946b3e74525ccf32139129d37d28`.
Its tree identifies `cbz/quarantine/catalog.json` as Git blob
`892466e68f90a584e214f60c86c7c493d4f9c42e`; the recovered 2,375,868 bytes
have SHA-256 `6fb095df59726f6ee13b265dd88440e1a5f0c7ca7a5025643fd1bbe9acab3ec7`.
The source README, validation, import-boundary and proof-index blobs are
recovered and individually Git-hash matched. Their historical adoption and
isolation statements describe that unmerged candidate and its recorded source
cuts; this observation does not adopt them at current defaults.

All 184 distinct records have nonempty values for the eleven named fields and
have `owner` equal to `none` and `maintenance` equal to `zero`. This is only a
field-presence/type observation. The source has 178 explicit `QUARANTINED`
status values and six records without that field, Q-9001–Q-9006; those six
refer to regression or real-record coverage, not automatically mathematical
quarantines. No field-presence check establishes a mathematical blocking
reason, dormant isolation, the accuracy of a historical result, or a valid
current disposition. In particular Q-9006 concerns a live peer scope.

`catalog-comparison-20261110.json.gz` decompresses to a report that compares 359 parseable historical location
strings as exact Git tree entries at their recorded commits with the same paths
at separately observed Main `c32e6f27d318e25708939105285366a7a0daf971` and
Math- `9bdfc54b76173b57912503bfcf901a5f9d43b09c`. Of those, 354 match mode,
kind and object identity and five differ. Three free-form locations and six
structured locations remain unassessed. Semantic fragments following `#`, the
completeness of locations, dependencies, proof statements, axioms and status
are not checked. A separate 334-lookup direct Git readback reproduced all 359
parsed comparisons. Each output row explicitly marks current admission
`UNASSESSED` and `cbz_green` stays false.

The three changed directory objects have the same leaf counts but changed
validation inputs: Q-1098 and Q-1099 each have three changed `MANIFEST.json`,
`test_verify.py` and `verify.py` leaves; Q-1114 has changed `SOURCES.json`
and `ecp_check.py`. The two changed file objects are Q-3001's determinant
`QUARANTINE.json` and Q-3002's derivative `SCOPE.md`; both old and current
exact blobs and byte diffs are retained. The current Q-3001 file narrows its
closed status to the 66 exact conditional companions and preserves its parent
application limits. This comparison does not rebind historical receipts or
establish the truth of the newer description. Full adjudication and mapping to
current claims remain open.

The first comparison enumerated only tracked leaves and marked 85 directory
locations unassessed. That result and the original analysis are retained;
the successor enumerates tree entries too. An initial native fetch through the
Math checkout's nonexistent `origin` failed with exit 128; its stderr is
retained. Fetching the four named blobs from the explicit public repository
URL succeeded, and each recovered byte stream was checked against its Git blob.
No default branch or peer scope was modified.

## Explicit archive locations and source bytes

The nine previously unparsed or structured locations received a separate
literal-path comparison, `explicit-locations-20261110.json`, SHA-256
`0bdf7d9bc6e09caf7d8562c7d9a2b0f3938ca1fa12b19a90c379ab1cbcf69bf8`.
The named Q-8336 and Q-8340 proof and review files and the Q-8429 source
directory exist at their recorded PR heads but not at the same paths in Main
`eed4e646565bf670d87b5ec41b9ae18f8a20d35a`. The two descriptions also mention
a README row without naming its path; that part remains unresolved. The three
annotated archive tags were freshly read and peel to those exact original PR
heads. Their historical tag messages do not establish current proof closure.

All 87 files in the three explicitly named source directories were recovered
as exact Git blob bytes: 2, 2 and 83 files respectively, 72 distinct blobs and
697,375 total file bytes. `archive-sources-manifest.json` SHA-256
`054b2cb69258e5909175d7206975f440b1d11586c8dea89ba3bad5a30df553ed`
records each original path, mode, Git blob, byte count and SHA-256. An
independent readback checked the tag targets, Git tree membership, both hash
algorithms, exact byte totals and absence of additional extracted files. The
raw source, authorship, review records and outcomes are preserved without
inferring a new independent review or acceptance. These three tags are listed
in the historical ref inventory but are not named in the five canonical Main
surfaces at this cut; their semantic disposition remains unfinished.

For Q-9001, the four explicitly named trial source entries have identical Git
objects at its recorded and observed archived trial tip. For Q-9002–Q-9006,
the explicitly named Main test file and schema blobs match their recorded
source. Their record glob, live state, skip interpretation and current
admission are not established by that byte comparison. No current proof or
regression gate is awarded.
