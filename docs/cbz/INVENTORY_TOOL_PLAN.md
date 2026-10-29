# CBZ source inventory tool plan

Owner: CBZ-0 consolidation implementer. Scope: deterministic, offline tracked
source inventory and declared classification/binding integrity for the twelve
explicit federation clones. This tool does not certify Layer A, authenticate a
kernel run, prove semantic-claim completeness, or grant CBZ green status.

Implementation is test first: real temporary Git repositories exercise normal,
dirty, missing, substituted, symlink, gitlink and race cases before the scanner
exists. Two source mutations must be killed by the same suite: remove the
file-set completeness check; remove the file-identity comparison.

CLI:

```
python3 -B -S tools/cbz_inventory.py generate --root ROOT --output EXTERNAL.json
python3 -B -S tools/cbz_inventory.py generate --root ROOT --classifications CLASSIFICATIONS.json --output EXTERNAL.json
python3 -B -S tools/cbz_inventory.py verify --root ROOT --manifest EXTERNAL.json
```

The default repository set is main, Math-, governance-, Universal-Law-Workspace,
meta-framework, query-, google-drive, trial, sandbox, d6g8k5htny-coder,
openclaw-custodian-validation, research-hosted-qualification. `--repositories`
accepts an explicit set of at most twelve simple directory names, primarily for
fixture/self-verification and explicitly bounded deployments. Other workspace
directories are outside that declared set; missing or additional manifest
repositories fail verification. No directory or tracked-file read is silently
skipped. Output must be outside those source repositories to avoid self-hashing.

The scanner records HEAD, dirty status, HEAD and staged Git objects/mode and
actual working bytes for the union of HEAD-tracked and index-tracked paths.
It includes staged additions and refuses missing files after staged deletions,
symlinks as link-target bytes without dereference, and explicit gitlink rows
without inventing hydrated submodule bytes. Missing tracked paths, conflicts,
unsupported types, Git errors, changed HEAD/index/status and detected file
replacement/content races fail. Ambient Git overrides are removed; network and
lazy fetch are disabled. Configured Git clean/process filters are refused before
`git status` can execute them. Paths use byte-preserving filesystem decoding and JSON
escaping, including tabs/newlines and non-UTF-8 Git names. No floats are accepted
in control documents.

Top-level manifest keys: `schema`, `certification_scope`,
`semantic_claim_completeness`, `cbz_green`, `repositories`, `obligations`.
Schema is `cbz.inventory.v1`; the two Boolean flags are always false.
Each repository has `name`, `head`, `dirty`, `status_porcelain_base64`, `files`.
Each file row has `path`, `kind`, `mode`, `git_object`, `head_git_object`,
`index_git_object`, `working_mode`,
`head_mode`, `index_mode`,
`working_git_blob`, `bytes`, `sha256`, `materialization`, `submodule_head`.
For gitlinks, byte/hash/working-mode fields are null and materialization is
explicit; the referenced source commit is `git_object`.

Classification input is exactly `{"obligations": [...]}`. Each obligation has
`id`, `status`, `source_bindings` and exactly the applicable quarantine/kernel
fields. Binding shape is `repository`, `commit`, `path`, `git_object`, `sha256`,
with optional `scope: "current"` (the default) or `scope: "historical"`.
Every binding resolves exact immutable regular Git blob bytes offline at the
supplied commit. Current bindings additionally require the current HEAD-tracked
working file to retain those exact bytes. Historical bindings explicitly name
dormant quarantine sources or archived receipt/manifest inputs; they need not
exist in the default working tree. CLOSED proof/source bindings require current
scope. Unavailable historical objects fail; the tool never fetches them.
An immutable ancestral proof binding survives document-only commits without
self-reference. Dirty proof bytes and staged-only new files cannot impersonate
source bytes at an immutable commit. Inventory `git_object` is the HEAD object
when present, otherwise the index object. Status is either CLOSED or
QUARANTINED. Quarantine additionally requires nonblank `exact_boundary`,
`blocking_reason`, `isolation`, and literal `owner: "none"`,
`maintenance: "zero"`. CLOSED requires `kernel_evidence` with `receipt`,
`manifest`, `checked_commit`, `targets`, `source_bindings`. The receipt and
manifest are actual hash-bound immutable Git files. The receipt must expose matching
checked commit, manifest digest and target axiom-report keys with
`formalization_status: "kernel-checked"`. These are structural/source-byte
checks: each target must also be declared by the bound manifest, and that
manifest plus every active proof binding must resolve to the same exact blobs
at the one checked commit in the manifest's repository. Nonexistent checked
commits, later substituted proofs and detached target names fail. Cross-repo
kernel source cuts need separate records and are not inferred. These are
structural/source-byte checks only; an authored receipt can still be false. Actual kernel execution,
trust qualification, statement alignment and exhaustive obligation discovery
remain separate gates.

For every claimed target, the receipt's axiom report must be an actual list of
unique strings drawn only from the declared approved set `propext`,
`Classical.choice`, `Quot.sound`. Explicit unknown axioms such as `sorryAx`,
malformed reports and duplicate names fail. This rejects contradictory recorded
data; it still does not authenticate a kernel execution or independently
reconstruct transitive axioms.

The single optional receipt extension is `kernel_evidence.manifest_overlay` for
the existing Math RF312 source gate. It is permitted only for repository
`Math-`, base `formal/manifest.json`, sibling `formal/manifest.overlay.json`, and
base `source_identity` exactly `{"schema":"rf312-overlay-v1","overlay":"manifest.overlay.json"}`.
That marker requires an overlay binding; it cannot fall back to a raw base
digest. Both bound blobs must resolve unchanged at the checked commit, and
current-scope bindings retain the ordinary current-byte checks. The expected
receipt digest is exactly SHA256(base raw bytes + `\0RF312-OVERLAY\0` + overlay
raw bytes). Overlay keys are exactly `schema_version` and `files`, with integer
schema version 1 and string-to-SHA256 file bindings, matching the source gate's
`_ALLOWED_OVERLAY_KEYS`. The target list comes only from the base manifest's
string targets; the overlay cannot add, replace or remove targets. Raw main
manifest handling and its target-record structure are unchanged. This extension
checks the recorded byte contract; it neither reruns the Math source/kernel gate
nor authenticates the receipt's kernel claim.

Self-verification:

```
python3 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
```

The suite itself executes the two scanner source mutations in isolated
temporary directories and requires both mutated suites to fail. Evidence and
terminal outcomes are recorded in this plan when observed.

The scanner observes multiple consistency checkpoints but does not lock source
repositories. A detected race fails; successful observation cannot promise that
another writer will keep the source frozen afterward. Final CBZ gates must use
the root coordinator's frozen source vector. Inventory success alone never
certifies the semantic completeness of the supplied obligation list.

Observed implementation verification (2026-10-29): Python 3.11.17, Git 2.52.0.
All 38 contract tests passed in normal and optimized modes (82.024s and
82.469s). Each mode also executed both source mutants; removing the complete
file-set check was killed by the missing-manifest-file test, and removing file
identity comparison was killed by the substituted-hash test. The original
unimplemented tool produced 24 expected test failures; subsequent source-cut,
ancestry and review corrections were also observed failing before their fixes.
A final-schema smoke capture and independent verification covered 12 actual
clones and 13,654 tracked entries, with both noncertification flags false.
These are tool tests and an integration smoke observation, not the final
CBZ baseline gate pass. Read-only independent review of the fixes reported no
remaining high or medium findings. Logs are retained under
`/workspace/cbz/discovery/foundation/cbz-inventory-final-{normal,optimized}.log`
and `cbz-real12-final-{generate,verify}.log` in this execution workspace.

RF312 integration extension verification (2026-10-29): the positive composite
receipt test first failed against the prior raw-only validator. Eight new
format/adversarial tests then passed with the original 38 tests. Root's
independent review confirmed exact marker/location/raw-byte concatenation,
strict checked-cut coverage and absence of a raw fallback. Its final recorded
axiom-shape finding produced ten failing subcases before the unique-approved-list
guard was implemented. The complete current suite is now 50 tests: all 50 passed
in both normal and optimized Python 3.11.17 modes, including both scanner source
mutants in each mode (whole-command durations 129.990s and 130.433s). Current
logs are `cbz-inventory-rf312-final-{normal,optimized}.log` under the same local
evidence directory. At actual Math commit `8151f592918a845bf19eb8eafaf8444540a32734`,
the exact base/overlay composite digest is
`481653d73d00f64bf06fefbf24be994f60ebe5c54f826abf899461f6e6937066`, matching the
recorded 63-target core receipt; this is a byte-format observation, not an
independent execution or authentication claim. The source census flags remain
false. No final CBZ baseline success is inferred from these tool tests.


### RF312 effective-map repair (2026-10-29)

The scoped review exposed a valid-hex wrong-digest counterexample. The checker now resolves the base file map with overlay-wins replacement, requires exact checked formal-tree coverage except the two manifest inputs, and compares each effective regular Git blob SHA256 to its declared value. A valid current overlay may replace an older base digest. Seven new behavior methods first produced eight intended assertion failures; initial fixture errors are retained separately. All 50 original test bodies remain unchanged. The final suite passed 58 methods in each Python 3.11.16 mode, including four intended source-mutant kills per mode. Actual 94-binding generation and verification passed and checked all 45 effective formal files. See cbz/evidence/rf312-source-binding-repair/REPORT.json for exact source/log hashes. No semantic completeness, execution authenticity or final CBZ result is inferred.
