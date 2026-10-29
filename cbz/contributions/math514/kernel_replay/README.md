# Bounded exact-module Lean kernel replay

This adapter invokes the installed `LeanChecker.replayFromImports` once in each fresh Lean process. It removes the upstream CLI's prefix expansion and concurrent per-module tasks while preserving its kernel replay routine. It is a scoped G4 tool, not an independent proof checker or a complete CBZ gate suite.

Requirements: Linux, Python 3 standard library, and a complete Lean 4.34.1 toolchain containing `bin/lean`, `lib/lean/LeanChecker.olean`, `lib/lean/Lean/Replay.olean`, and their corresponding installed source files. No Lake invocation, package download or native build is required. The driver is evaluated by plain `lean -j1 ExactReplay.lean`, because importing `LeanChecker` already imports upstream's root `main`; using `lean --run` would invoke its unsuitable CLI.

Use absolute, canonical paths without symlinks for artifact roots and olean files/parts. The sysroot library takes precedence over supplied `LEAN_PATH` entries so a project cannot shadow the imported checker. Only explicit ASCII Lean module identifiers are accepted. Unknown arguments, including `--help`, fail; `python3 kernel_replay.py help` prints usage.

From a fresh clone's repository root, use the published contribution path below. This example requires `elan` and an already installed Lean 4.34.1 toolchain selected by the repository's pinned toolchain setting. Without `elan`, set `LEAN=/absolute/lean-4.34.1/bin/lean` instead. Historical G0 receipts retain the actual author-workspace paths recorded when those checks ran; these portable usage examples do not rewrite that evidence.

```bash
TOOL="$PWD/cbz/contributions/math514/kernel_replay"
LEAN="$(elan which lean)"

# Include every module from the caller's independently complete owned-module inventory.
python3 "$TOOL/kernel_replay.py" prepare \
  --lean "$LEAN" \
  --lean-path /absolute/project/.lake/build/lib/lean \
  --lean-path /absolute/mathlib/.lake/build/lib/lean \
  --module ExactModule=/absolute/project/ExactModule.lean \
  --module AnotherModule=/absolute/project/AnotherModule.lean \
  --context /absolute/project/lean-toolchain \
  --context /absolute/project/lake-manifest.json \
  --output /absolute/evidence/replay-manifest.json

python3 "$TOOL/kernel_replay.py" run /absolute/evidence/replay-manifest.json \
  --memory-mib 16384 --timeout-seconds 600 \
  --output /absolute/evidence/replay-report.json

# The one-module convenience command uses the same manifest and replay checks.
python3 "$TOOL/kernel_replay.py" one ExactModule \
  --lean "$LEAN" --lean-path /absolute/project/.lake/build/lib/lean \
  --source /absolute/project/ExactModule.lean \
  --memory-mib 16384 --timeout-seconds 600 \
  --output /absolute/evidence/one-module-report.json
```

Repeat `--lean-path`, `--module`, and `--context` as needed. No glob, prefix expansion, implicit module or skip option exists. `prepare` binds every source, available exported/server/private olean part, executable and checker implementation. `run` checks exact nonempty coverage, validates bindings before/after execution, and waits for each process before starting the next. It requires exit zero and exactly one matching success marker. A failed child remains fatal to overall status even if other modules pass. Reports contain commands, times, bounds, hashes and output tails. CLI exit codes: 0 successful scoped replay/preparation, 1 non-green replay report, 2 malformed CLI/input/preparation failure. Preserve failed reports.

`--memory-mib` limits **virtual address space** with Linux `RLIMIT_AS`; it is not an RSS measurement or a replacement for the enclosing cgroup's physical memory limit. Tiny-driver diagnostics observed about 6.2 GiB virtual reservation and 614 MiB RSS. A 2 GiB address-space limit aborted during thread creation before checking; fixture replay passed with an explicit 8192 MiB cap. Individual real modules can still exceed memory or time bounds. The initial Math core exit-137 OOM at16GiB remains retained failed-run evidence. The canonical handoff already records a later successful same-source complete Math-core gate with `LEAN_NUM_THREADS=2` at8151; see [the immutable component receipt](https://github.com/d6g8k5htny-coder/main/blob/94747663830ca6645ea27c8820058168c08cf170/cbz/evidence/kernel/math-core/receipt.json). Reuse that completed evidence: this adapter is a separately verified exact-scheduling option and does not require another Math-core replay. Its tiny fixtures alone do not establish every project module's resource requirement.

The source and compiled hashes are separate bindings; G3 build evidence must establish that the compiled artifacts came from those sources. The report covers only declared targets and explicit context files. Supply all dependency bindings required by the intended evidence claim, and use a stable input tree. A hand-selected subset does not establish complete project coverage. Symlinked artifact roots/components/files/parts are rejected because upstream selects private/server siblings by lexical filename; canonicalizing a single file could otherwise bind different parts from those replayed.

Upstream reads all available olean parts, uses the most private constant map, and checks safe total declarations in an **import-only** environment. It handles inductive declarations and checks the generated constructors/recursors. Its intentional exclusion of unsafe and partial definitions remains in effect. Imported dependencies require their own coverage/evidence. Kernel validity permits axioms; `g2_status` always remains `NOT_ASSESSED`. The included axiom fixture is kernel-valid and is rejected by a separate explicit fixture-policy check. That check illustrates the G2/G4 distinction; it is not a production all-declaration approved-axiom auditor. G2 is still a separate blocking obligation.

Run G0 only for changed tooling or uncovered obligations; reuse applicable evidence on unchanged material:

```bash
export CBZ_TEST_LEAN="$LEAN"
python3 "$TOOL/run_tests.py" --output "$TOOL/evidence/g0-tests.json"
python3 "$TOOL/mutations.py" --output "$TOOL/evidence/g0-mutations.json"
```

The real fixtures cover public/private proofs, data, inductives, exact-module behavior, installed-checker import trust and module-file symlink rejection. `BadPublic.lean` and `BadPrivate.lean` intentionally inject self-referential proofs through the public declaration API with fixture-local `debug.skipKernelTC`. They and `UnapprovedAxiom.lean` are quarantined test data, never closed mathematical modules. The exported-only and target-preloading mutants demonstrate why private parts and an import-only replay environment are required. Mutation copies are isolated and disposable; actual assertion failures alone count as kills. Errors, skips, empty runs and timeouts are non-green harness outcomes. Initial survivors and erroneous historical adapter receipts are retained as failed tooling evidence, never certification.

Ownership, gate boundaries, publication mapping and canonical registry/ledger payload are in `REGISTRY.json`. The canonical writer integrates that payload; this contribution does not edit shared canonical surfaces or publish externally.
