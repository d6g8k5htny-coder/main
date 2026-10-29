# Determinant literal theorem evidence

This bundle closes exactly the 66 public theorem declarations in
`frontiers/determinant_moment_20261025` at Math commit
`74565f013575b7b05ecfc4034cd5d638ee355353`. Its mathematical inputs are 14 Lean
files plus the exact package Lake configuration, lock and toolchain. The proof
files are unchanged from the preceding snapshot. Kernel names must be identified
with their package and source path: this isolated packet reuses 13 names from the
separate formal core and is not interchangeable with that core.

The observed positive run rebuilt the packet, replayed all 157 owned kernel
constants and all 58,421 transitive type/value dependencies into the official
empty Lean kernel at trust level zero, printed each of the 66 axiom closures,
and checked every elaborated literal signature. Only `Classical.choice`,
`Quot.sound` and `propext` occur. Real hidden-sorry and new-custom-axiom leaf
mutations first compiled successfully and then failed for their intended
forbidden axiom. An earlier mutation launch failed to resolve a package object;
that attempt was retained outside this accepted bundle and was not counted as
an intended rejection.

`receipt.json` is a new observation index over actual raw executions, not an
alteration of a historical package receipt. Python checks supplied transcript
consistency; it does not authenticate an execution or verify Lean proofs.
The original packet's admission auditor at this source cut does not support its
legitimate private/instance/generated declarations and remains a separate
engineering quarantine. This literal receipt does not claim a passing repaired
package gate, scientific acceptance, a concrete Gaussian/Palm identification,
or a parent lifetime theorem. Every theorem retains its complete premises.

## Reproduce the literal kernel checks

Use the documented federation checkout layout, Python 3.11.16 and the pinned
Lean 4.34.1 release (`5045d0056413266e57c625dcd7c365b10e377c52`). The lock pins
Mathlib to `d13f23b723b8a846827a245b89c10fc7d3f11612`; provision dependencies
using the federation bootstrap instructions, preserving that lock. These
commands assume the exact source files/configuration already passed the
verified federation census. They create output outside the source trees.

```sh
MATH_ROOT="$(pwd)/Math-"
EVIDENCE="$(pwd)/main/cbz/evidence/kernel/determinant"
RUN="$(mktemp -d "${TMPDIR:-/tmp}/cbz-determinant.XXXXXX")"
export LEAN_NUM_THREADS=2
cd "$MATH_ROOT/frontiers/determinant_moment_20261025"
lake env lean --version
lake build
if lake env lean "$EVIDENCE/literal-qualification/Replay.lean" > "$RUN/replay.stdout" 2> "$RUN/replay.stderr"; then
  replay_status=0
else
  replay_status=$?
fi
python3.11 -B -S "$EVIDENCE/replay-tool/replay_tool.py" validate \
  --transcript "$RUN/replay.stdout" --stderr "$RUN/replay.stderr" \
  --exit-code "$replay_status" \
  --expected-census "$EVIDENCE/literal-qualification/expected-census.json" \
  --program "$EVIDENCE/literal-qualification/Replay.lean" \
  --output "$RUN/transcript-consistency.json"
lake env lean "$EVIDENCE/literal-qualification/Axioms.lean"
lake env lean "$EVIDENCE/literal-qualification/Types.lean"
```

The complete raw original execution phases and hashes are in
`literal-qualification/RESULT.json`, `actual-axiom-mutations-v2/RESULT.json`,
and the corresponding stream files. `kernel-targets.json` gives every exact
source path and its SHA256. `closed-declarations.json` records each literal
source header, elaborated signature, source Git object and actual axiom set.
A supplied consistency JSON or a zero process code without these source and
execution bindings is insufficient evidence.

## Recheck the auditor

The [bounded replay tool instructions](replay-tool/README.md) contain portable
commands for its ordinary/optimized unit suite, three Python auditor source
mutants, and six actual Lean adverse controls plus six generated-auditor source
mutants in each Python mode. Fixtures are generated in a disposable temporary
directory. The source/runtime/collector identities are checked before and after.
The qualification reused this finished tool without changing it; its accepted
code hash is `b97bee53a590651b83d68056c090ba70e8a055157e65d2bd3fcc5cd0219840b3`.

## Retained mathematical contracts

`retained-controls-v2/RESULT.json` records all nine retained cases: two positive
examples compile, seven false bound/identification claims fail at their actual
mathematical goals. One historical positive fixture used `Measure.map_dirac`
where the pinned library requires `Measure.map_dirac'`; the existing successor
already contained the correct proof. The original fixture received that single
character correction, retaining exactly the same asserted statements and all
nine cases. Its failing run and original Git blob remain preserved. This test
repair changes no production theorem, bound, hypothesis or dependency pin.
