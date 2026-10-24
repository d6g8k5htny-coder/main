# Public lexical vector index

This is a small retrieval baseline for an agent reading the public
`d6g8k5htny-coder/main` repository. It builds deterministic vectors from word
counts and returns matching source excerpts with pinned GitHub links. It uses
Python 3.9 or newer, Git and the Python standard library. There is no trained
encoder, learned embedding, model download, external model call, service,
credential, package installation or paid storage dependency. Lexical similarity
does not establish semantic understanding or mathematical correctness.

The separately owned semantic-memory work remains a dependency for any learned
semantic capability. [The retained owner record](https://github.com/d6g8k5htny-coder/main/issues/333#issuecomment-6206436926)
preserves Root `01a11d37`; [the current disposition](https://github.com/d6g8k5htny-coder/main/issues/333#issuecomment-6351948287)
keeps Task 2 `REQUEST_CHANGES`, P1/P2 unaccepted, R01/R02 `STOPPED` and the old
author's closeout `UNKNOWN`. This tool does not retry those namespaces, transfer
their ownership or complete that work. Existing #351 and #460 owner scopes and
holds are unchanged. The bounded implementation pickup is
[main #475/6353027897](https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6353027897).
Read the [owner mandate](../governance/OWNER_MANDATE_20261015.md) alongside the
[standing delegation](../governance/OP-AUTONOMY-20260923-v2.1.md).

## Exact source boundary

Only these five already-public paths are selected, in this fixed order:

1. `AGENTS.md`
2. `architecture/node-status.schema.json`
3. `docs/WORKSPACE.md`
4. `docs/site/node-status.json`
5. `governance/OP-WORKFLOW-20260930.md`

There are no user path arguments, globs, recursive document discovery or private
connectors. All private records, local files outside this list, dirty changes
and untracked files are excluded. A source being readable somewhere else does
not authorize adding it. Changing this allowlist requires a separate source and
publication review. The public node file and its existing schema are read as
source text; this tool does not write or reinterpret node status, ownership,
stops, source revisions or scientific effects. It creates no status authority.

`origin` must identify the fixed public repository. The caller supplies a full
lowercase 40-character commit SHA1. The tool reads the commit, each traversed
tree and selected blob through local Git plumbing with replacement objects and
lazy fetch disabled, and verifies each object's SHA1 over its actual Git object
header and bytes. Git subprocesses use argument arrays and `shell=False`.
No source is read from the working tree. Only regular `100644` blob entries
under regular tree parents are accepted; symlinks, executables and submodules
at selected paths fail closed. Missing selected paths are omitted in a full
rebuild, including their former chunks; no previous index is merged or retained.

The remote URL is a scope guard, not proof that a supplied commit was published
or is the current public branch. Obtain the expected commit through the public
repository and its applicable review process. The tool performs no network
requests and does not follow links contained in records. A partial clone must
already contain the selected commit, trees and blobs; missing objects fail
instead of silently fetching.

## Build and query from a pinned public source

A fresh agent can clone the public repository and use a reviewed implementation
commit containing this tool. Set `TOOL_COMMIT` to that exact published commit,
and `SOURCE_COMMIT` to the exact public source revision to consume. The checkout
command below pins the implementation; `--commit` independently pins the data.
Use complete 40-character SHA1 values obtained from the public repository.

```sh
git clone https://github.com/d6g8k5htny-coder/main.git public-main
cd public-main
TOOL_COMMIT='<full-reviewed-implementation-sha1>'
SOURCE_COMMIT='<full-public-source-sha1>'
git checkout --detach "$TOOL_COMMIT"
INDEX_DIR="$(python3 -c 'from pathlib import Path; import tempfile; print(Path(tempfile.mkdtemp(prefix="public-vector-")).resolve())')"
python3 -B -S tools/public_vector_index.py build --commit "$SOURCE_COMMIT" --output "$INDEX_DIR/index.json"
python3 -B -S tools/public_vector_index.py check --index "$INDEX_DIR/index.json" --commit "$SOURCE_COMMIT"
python3 -B -S tools/public_vector_index.py search --index "$INDEX_DIR/index.json" --commit "$SOURCE_COMMIT" --query 'eng.hard-gate' --limit 5
```

`search` returns JSON with cosine scores, exact text, character ranges, stable
chunk IDs and source links. Read the returned pinned source and its linked
evidence before acting. A match is a navigation aid, never an ownership claim,
review disposition or scientific acceptance. Queries are limited to 4,096
characters and result counts to 1–20. Ties sort by path, start character and ID.
Hash collisions can return unrelated matches; tokenization does not understand
synonyms, mathematical meaning or non-ASCII words. Chunks can split a sentence
or JSON node. Adjacent chunks and the full pinned source remain available.

Both `check` and CLI `search` rebuild from the expected source commit and require
byte-exact equality with the provided index. Changed text, vectors, metadata,
unknown fields, different whitespace and stale source commits are refused.
An old commit can still be checked explicitly as a historical source; callers
must choose the expected current revision themselves. Refreshing to a new
revision requires a full rebuild, so deletions cannot survive through an old
index. The in-process `search_index` function expects an index already returned
by `check_index`; direct use of an unchecked dictionary is not a trust boundary.

Output is an explicitly selected file outside the repository. It is written to
a newly created mode-0600 temporary file and atomically replaces the requested
regular destination. Destination symlinks and symlink parent directories are
refused, and a failed exclusive temporary-file creation preserves preexisting
files. Use a canonical directory path: on macOS `/tmp` and `/var` are symlink
aliases, so resolve the directory as above (or use `/private/tmp`). Parent
directories must already exist. The tool does not delete old index files at
other paths or purge historical Git commits and logs.

## Deterministic format and capacity

The artifact is canonical ASCII JSON: sorted keys, compact separators and one
terminal newline. Schema version 1 contains `source`, `algorithm`, `documents`
and `chunks`. `source` identifies the repository and commit; each document
identifies its path, blob SHA1, full raw-byte SHA256 and byte count. Each chunk
references that document path and contains text, a pinned URL, a stable ID,
zero-based half-open Unicode character offsets and one-based inclusive lines.
An unchanged document's ID remains stable across unrelated commits: it is the
SHA256 of repository, path, blob SHA1, start and end joined with NUL characters.
No build timestamp, run ID or mutable source URL is added. Existing timestamps
inside original source records remain source text.

Algorithm `lexical-feature-hashing`, version 1, lowercases text, extracts
`[a-z0-9]+` tokens, hashes each token with SHA256 and maps the first four digest
bytes, interpreted big-endian, modulo 256. Sparse vectors contain sorted
`[dimension, positive-count]` pairs. Search computes cosine similarity using the
same query transform; the index stores integers rather than floating vectors.

| Bound | Limit |
| --- | --- |
| One selected UTF-8 source | 32,768 raw bytes |
| All selected sources combined | 65,536 raw bytes |
| One chunk | 2,048 Unicode characters |
| All chunks | 40 |
| Serialized index | 131,072 bytes (128 KiB) |
| Framed-log input | 262,144 UTF-8 bytes |
| One log payload chunk | 768 raw bytes / 1,024 base64 characters |

Commit objects are bounded to 65,536 bytes and each traversed tree to 2 MiB.
An invalid object, invalid UTF-8 source or exceeded bound rejects the entire
build. It never truncates a source or silently publishes a partial result.
ASCII JSON escaping can exceed the payload cap even when raw inputs fit.
Capacity changes require review along with source-boundary changes.

## Temporary workflow log transport

The manual workflow builds and checks the index in the runner's temporary
directory, then calls `log --index`. That command checks the complete index
against the commit embedded in it before printing any payload. Base64 avoids
interpreting source text as workflow commands. The protocol emits one `BEGIN`
record with raw byte count, SHA256 and chunk count, numbered `CHUNK` records,
and an `END` record repeating the SHA256. The `reconstruct_index` parser requires
the complete ordered sequence; missing, duplicate, reordered, corrupted and
oversized payloads are refused. Standard GitHub UTC timestamp prefixes and
unrelated log lines are accepted. Use the downloaded log for the index step,
not an arbitrarily large archive or combined history.

An agent can consume a saved log using the same expected source SHA1:

```sh
python3 -B -S tools/public_vector_index.py log --index "$INDEX_DIR/index.json" > "$INDEX_DIR/index-step.log"
python3 -B -S tools/public_vector_index.py reconstruct --log "$INDEX_DIR/index-step.log" --commit "$SOURCE_COMMIT" --output "$INDEX_DIR/restored.json"
python3 -B -S tools/public_vector_index.py search --index "$INDEX_DIR/restored.json" --commit "$SOURCE_COMMIT" --query 'eng.hard-gate'
```

Framing hashes establish completeness and byte identity, not authenticity.
Reconstruction also rebuilds from the selected Git commit before writing the
index. Logs are temporary and can expire or be truncated. A fresh consumer can
always rebuild from its pinned public source instead. Logs are no durable
research archive, delivery receipt, mathematical proof or scientific evidence.

The workflow uses a standard public-repository Ubuntu runner, with no artifact
upload, cache action, external model or storage service. GitHub documents
[standard hosted runners for public repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories)
and [Actions billing, including logs and job summaries](https://docs.github.com/en/billing/concepts/product-billing/github-actions).
It requests no larger runner or paid feature; these policy links remain the
source of truth if GitHub changes its terms.

## Verification

```sh
python3 -B -S -m unittest discover -s tests -p test_public_vector_index.py -v
python3 -B -S -O -m unittest discover -s tests -p test_public_vector_index.py -v
```

Tests use local disposable Git repositories and real blobs, rebuilds, CLI
queries and output files. They cover dirty/untracked exclusions, original
source identities, deletion, stale and forged inputs, nonregular entries,
corrupt and replaced objects, size limits, ranking and tie order, safe output,
and complete/corrupt/missing/duplicate/reordered/oversized log transport.
They perform no mathematical experiment and establish no scientific status.
