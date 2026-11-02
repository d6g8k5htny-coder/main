# Bounded Lean proof-closure replay

This self-contained tool generates an audit program for existing Lean 4.34.1
packages. Lean enumerates every compiled constant belonging to the explicitly
listed modules, follows all type/value dependencies, rejects every axiom except
`Classical.choice`, `Quot.sound`, and `propext`, rejects unsafe/partial constants,
and replays the selected declarations into an empty kernel environment with
trust level zero. It checks that every selected module's owned constants remain
present after replay. Inductives, constructors and recursors are handled by the
pinned Lean `Kernel.Environment.replay`, rather than a replacement kernel.

`replay_tool.py` also compares the complete observed `(module, constant)` census
with an independently produced compiled census. It rejects missing, additional,
duplicate, malformed, misplaced and contradictory records. The Python reader
does **not** verify Lean proofs or authenticate a claimed execution: a caller
must preserve the actual execution receipt, stdout, stderr, program identity,
source/object identities and toolchain pins. Passing supplied text to the reader
alone is not mathematical evidence. Its output states this limitation.

The owned modules' full proof dependency closure is the scope. Unrelated
imported declarations, adequacy of a mathematical model, satisfiability of all
hypotheses, and correctness of prose are outside this tool's claim. The tool
accepts simple dotted module names, and constant names with no whitespace,
brackets or commas. Unsupported names fail closed. It does not silently omit
them. No floating-point decision is used.

## Provenance and use

The dependency collector is extracted **without executing Python** from the
single string literal containing the existing `inventory_source` collector in
`Math-/formal/gate_helper.py`. The entire helper file's SHA256 must equal the
explicit expected value. For the qualified source cut
`8151f592918a845bf19eb8eafaf8444540a32734`, that SHA256 is
`ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4`.
The generated Lean program contains that collector verbatim.

The CLI has no required `/workspace` paths and supports repeated module/import
arguments. Both output files must be new, with existing parent directories.
Outputs are written through temporary files and atomic exclusive hard links;
existing files and input/output aliases are rejected.

```sh
python3.11 -B -S replay_tool.py generate \
  --collector /path/to/Math-/formal/gate_helper.py \
  --collector-sha256 ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4 \
  --module Lifetime --module RegularPointChart \
  --module RegularSuperlevel --module RegularSuperlevelLiteral \
  --import RegularSuperlevelLiteral \
  --program /new/output/Replay.lean --metadata /new/output/program.json
```

Build the existing package from its exact pinned sources and execute the emitted
program using the package's resolved Lean path and the pinned runtime. The
qualification used Lean 4.34.1 release commit
`5045d0056413266e57c625dcd7c365b10e377c52` and Mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`, with `LEAN_NUM_THREADS=2`.
The surrounding package recorder is responsible for actual exit status and
before/after source, object, dependency and runtime identities. This tool does
not replace a package build/receipt orchestrator.

The expected-census JSON contains nonempty `module_counts` and
`module_declarations` objects from the independently executed compiled census.
Each owned name must occur once in exactly one module, and each integer count
must match its full list. An optional `allowed_axioms` list must equal the fixed
three names above. Duplicate JSON keys are rejected. Supplying an incomplete
census would narrow coverage; the caller must establish its completeness from
the actual compiled module tables.

After the recorder observes exit zero and empty stderr, check the raw files:

```sh
python3.11 -B -S replay_tool.py validate \
  --transcript /actual/run/stdout.log --stderr /actual/run/stderr.log \
  --exit-code 0 --expected-census /actual/run/compiled-census.json \
  --program /actual/run/Replay.lean --output /new/output/consistency.json
```

`--exit-code` is supplied execution data, not proof that a process actually ran.
Raw file hashes preserve line endings. The output's
`execution_authentication: false` and `kernel_verification_by_python: false`
prevent treating this parser as an execution or kernel oracle.

## Self-verification

Use Python 3.11.16, matching the repository's exact CI version:

```sh
python3.11 -B -S -m unittest -v test_replay_tool.py
python3.11 -B -S -O -m unittest -v test_replay_tool.py
python3.11 -B -S run_mutations.py --output /new/verification-run
```

The mutation runner first requires the complete unmodified suite to pass in
both modes, then removes three distinct, uniquely anchored validator guards.
Every mutant must produce the named assertion failure; import/syntax/launch
failures cannot count as kills. The declared threshold is all three mutants in
both modes. The owned-name guard has a redundant module-assignment guard behind
it: its test requires the specific owned-name rejection, so removal is detected
even though the second guard still rejects the malformed input. The other two
mutants disable exact module membership and the fixed axiom allowlist.

`verification-v2/REPORT.json` records 31/31 unit tests per mode and all three
declared source mutants killed in each mode on Python 3.11.16. Earlier results
are retained with their source hashes; they are not substituted for this run.
Actual Lean adverse controls and source mutations also have a runnable,
independent finite fixture harness. It requires only the pinned Lean runtime,
the expected collector and Python, with no Mathlib build:

```sh
python3.11 -B -S run_lean_selftest.py \
  --lean /path/to/lean-4.34.1/bin/lean \
  --collector /path/to/Math-/formal/gate_helper.py \
  --collector-sha256 ff8a5374597772b2d37054fd722c2243c4b552a63802c69f7c605519832be6c4 \
  --output /new/actual-lean-selftest
```

Run the same command with Python `-O` and a different new output directory to
verify the recorder in optimized mode. The harness generates every violating
fixture in a temporary directory and removes it afterward. It retains raw
streams, executable/program hashes and observed exit codes. It compares its
positive owned census with a separate compiled module-table enumeration.

The six adverse cases test an ill-typed theorem value, omitted owned theorem,
empty selected map, forbidden axiom, and unsafe/partial `ConstantInfo` values.
Six uniquely anchored mutations change generated auditor source: reusing the
imported environment, removing owned-presence checks, bypassing kernel replay,
removing the axiom guard, and separately removing unsafe/partial rejection.
The first four yield actual false completion markers. The last two still
reject at the independent presence check after upstream Replay skips the
unsafe/partial declaration, but lose the required safety diagnostic. The
report distinguishes these outcomes. Syntax/import/launch failures cannot
count as kills. The declared threshold is all six registered mutations.
`lean-verification-v3-normal/REPORT.json` and
`lean-verification-v3-optimized/REPORT.json` each observed all six controls and
all six mutations pass on the final self-test source; the latter records Python
optimized mode explicitly. Their runtime, collector and auditor hashes were
checked again after execution.
Additional controls on the actual large packages are the qualification
caller's responsibility. Finite tests establish the stated coverage, not a
formal proof of Python or Lean implementation correctness.

Maintenance owner: none — bounded, self-contained artifact. Serves CBZ criteria
3, 6, 9 and 11, within the stated scope. Root canonical GOVERNANCE/TOOLING/STATUS
registration is the integrating agent's responsibility; no global gate is green
merely because this component passes.
