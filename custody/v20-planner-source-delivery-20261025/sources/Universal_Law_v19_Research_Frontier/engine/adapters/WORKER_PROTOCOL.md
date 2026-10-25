# Worker protocol v17.1

`workers.make_job(data, record_id, triggering_event)` captures the exact statement,
scope, equations, proof hyperedge, recursive prerequisites, side conditions,
imports, corrections, historical evidence, proof-note hashes and source-custody
descriptors. A binding excludes the global working-dataset hash so an unrelated
branch change does not make this obligation stale. The full snapshot hash remains
in the job as enqueue provenance. The receipt's `execution_dataset_sha256` pins
the actual snapshot at execution; an unrelated approval between queueing and
execution may change that global snapshot while preserving the input binding.

`workers.execute_job(job, data, empty_attempt_directory)` writes immutable input
files, executes the real Python diagnostic in a child process, and writes
`diagnostic.json`, bounded stdout/stderr, and `receipt.json`. Every retained file
has its byte count and SHA-256 in the receipt. Worker source files and the
validator have independent SHA-256 pins, and the receipt records the Python
version, executable digest, platform and minimal child environment policy.

The parent enforces wall time and kills the process group on timeout. The child
enforces CPU time, address-space, file-size and core-dump bounds using POSIX
resource limits. A platform without these child limits cannot pass this worker.
Output truncation fails the attempt. Resource bounds are execution controls;
this local worker is intended for trusted code and is not a general code sandbox.

Results are `diagnostic_completed`, `failed`, or `blocked`. A retryable failure
indicates a transient execution problem; the durable coordinator controls capped
retry, leases and escalation. An integrity-valid failed receipt still describes
a failure. Neither result can discharge a mathematical obligation or award a
formal theorem certificate.

Run `python workers.py detect` to inspect installed adapters. Unconfigured Lean
and Coq tools are detected by executable lookup and are not invoked. A submitted
job for either returns a blocked receipt. Installing a tool does not configure a
kernel adapter. The profile examples specify the fields a future reviewed
adapter must fill; no fake proof-success path exists.

Run `python workers.py replay PATH/TO/receipt.json` to validate every recorded
artifact digest and execute the diagnostic again from its captured snapshot.
Replay compares the deterministic diagnostic result, including its input binding
and unresolved obligations. Runtime artifact paths are local execution locations;
after moving a runtime, replay resolves files relative to the supplied receipt
file. Python callers passing a receipt dictionary can supply `workdir=...` to
`verify_receipt`. The original absolute directory remains hashed historical
execution context. Source-custody descriptors and the checker implementation must
still match the pinned release.

Receipt hashes detect inconsistent bytes. They do not authenticate an external
publisher or protect a trusted machine against its owner rewriting both files
and their hashes. A full mathematical certificate requires the separate
certificate gate and a genuine reviewed kernel replay.
