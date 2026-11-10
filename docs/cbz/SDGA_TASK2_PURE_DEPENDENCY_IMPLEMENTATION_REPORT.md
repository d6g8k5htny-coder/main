# Task A pure dependency implementation source report

The only production path in this slice is tools/sdga_component.py, blob
372dab47ade30bc2202ebf554ab1def9b1900211. It exposes exactly the two public
pure functions specified by the unchanged dependency contract
7e8d99875bfd3fb8dfdf5d5b60384041a454ed4a. It imports only standard-library
json and performs no filesystem, process, Git, network or CLI operation.
Future Task E compile/observation/combining/controller interfaces remain absent.

Root authorized this implementation after independently observing hosted
run 42742966104, attempt 1, job 129588267695: normal and optimized modes each
ran twelve pure-suite methods with twelve explicit component-file absence
failures, zero errors/skips, before any of the 45 semantic calls. Root reported
native ac0ef848 with the exact test-only tree and preserved the original Task A
six-method/39-subcase native absence results. This worker did not execute or
independently reclassify those runs.

The unchanged offline tests are blob
8809f1974998f9094a0f1239ecca56ddcf272a88: twelve methods, six header methods
with 31 calls plus six topology methods with 14 calls. Existing Task A launcher
d7f72edb1b7dac6458c568e9411edf21f2703476 is untouched.

The parser requires exact nested schema keys and Boolean types, unique JSON
keys, present results, empty errors and exact positional file census. It
preserves legitimate import occurrences. Decoder ValueError/RecursionError
become DEPENDENCY_HEADERS failures without introducing a custom input-size
cap. File order authentication remains the future native acquisition caller's
responsibility because Lean rows contain no filenames.

Topology validates the complete fixed LOCAL vertex census, rejects duplicate
expected vertices and unknown local references, deduplicates repeated import
occurrences only for traversal, and emits the earliest currently ready vertex
in expected order. Incomplete cyclic graphs fail instead of returning a
partial order. Inputs are not mutated. Empty pure boundaries remain {} and ().

A fresh source-exposed OpenAI/Codex child reviewer
/root/sdga_python_contract/pure_api_source_review reviewed the implementation
against the immutable tests and contract. It identified an initial decoder
prefix leak in superseded blob 667242de; the corrected 372dab47 catch was
re-reviewed with no remaining concrete blockers in the scoped source review.
That review provides no external-independence, native-execution or pass credit.

Exact UTF-8 implementation blob readback matched the authored 4,723 characters.
Test and contract readbacks matched their unchanged authored contents. No ref,
tree, workflow, PR, comment, production profile or original failure was changed
by this worker. Hosted normal/optimized pure tests and native compilation/
controls remain pending. scientific_effect=NONE; qualification=BLOCKED;
classification_status=BLOCKED; execution_authenticated=false.
