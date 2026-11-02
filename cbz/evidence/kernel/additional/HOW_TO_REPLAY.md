# Replaying the three additional literal proof families

This delivery records 207 literal declarations: 78 specification proofs, 116
smooth-source theorems, and 13 C297 theorems. The source cut is Math-
`8151f592918a845bf19eb8eafaf8444540a32734`. Every parameter and hypothesis in
the indexed declaration remains part of its statement. No informal model or
larger conjecture is promoted by these receipts.

The specification package has 143 registered **Prop definitions**. Its 78
registered proofs cover 77 distinct Props. The other 66 definitions are not
theorem obligations merely because they define propositions; defining a Prop
does not prove it. Those 66 are outside the 207-declaration proof index. A
particular intended claim about one of them requires its own explicit statement
and proof boundary.

The three final generated programs and their complete expected compiled
censuses are already included under `verified-replay-20261029/`. Their observed
empty-kernel replays covered:

| Family | Indexed literal theorems | All owned compiled constants | Type/value dependency closure |
|---|---:|---:|---:|
| Specification | 78 | 901 | 65,210 |
| Smooth source | 116 | 222 | 55,236 |
| C297, including Lifetime | 13 | 131 | 38,531 |

The additional constants include definitions and compiler-generated helpers;
they are not additional mathematical claims. The proof dependency closures may
overlap. Unrelated imported declarations remain outside these bounded replays.

## Preparation and immutable inputs

Run the following blocks in one Bash session. Replace the first two paths with
absolute paths. `task_math` must be a clean isolated checkout at the exact cut
above; the delivery may live anywhere outside that checkout. Use Python
3.11.16 and Lean 4.34.1 release commit
`5045d0056413266e57c625dcd7c365b10e377c52`. The pinned Mathlib revision is
`d13f23b723b8a846827a245b89c10fc7d3f11612`.

```bash
set -euo pipefail
task_math=/absolute/path/to/Math-source-at-8151
task_delivery=/absolute/path/to/this/delivery
task_math="$(realpath "$task_math")"
task_delivery="$(realpath "$task_delivery")"
task_python="$(command -v python3.11)"
task_run="$(mktemp -d "${TMPDIR:-/tmp}/cbz-additional-replay.XXXXXXXX")"
export LEAN_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
test "$("$task_python" -c 'import sys; print(".".join(map(str,sys.version_info[:3])))')" = 3.11.16
test "$(git -C "$task_math" rev-parse HEAD)" = 8151f592918a845bf19eb8eafaf8444540a32734
test -z "$(git -C "$task_math" status --porcelain --untracked-files=all)"
(cd "$task_delivery" && sha256sum -c SHA256SUMS)
"$task_python" -B -S - "$task_math" "$task_delivery/SOURCE_MANIFEST.json" <<'PY'
import hashlib, json, pathlib, sys
root, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for entry in json.loads(manifest.read_text())['sources']:
    path = root / entry['path']
    data = path.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise SystemExit('source identity mismatch: ' + entry['path'])
print('All delivered source/config identities match.')
PY
```

Resolve the existing Lake manifests and fetch their exact pinned dependencies
and available compiled caches. Do not run `lake update`, change a lock file,
substitute a newer runtime, or use `--allow-dirty` to obtain a green result.
Network/cache availability affects setup time, not the proof boundary.

```bash
for task_package in \
  specifications/lean_statement_proofs_20261011 \
  frontiers/source_smooth_realization_20261011 \
  formal
do
  (cd "$task_math/$task_package" && lake exe cache get)
done
```

Check all resolved dependency revisions and tracked dependency worktrees before
continuing. Repeat this same block after the last replay. Relative Lake package
directories are resolved from their package root; no environment-specific
`/workspace` path is assumed.

```bash
for task_package in \
  specifications/lean_statement_proofs_20261011 \
  frontiers/source_smooth_realization_20261011 \
  formal
do
  "$task_python" -B -S - "$task_math/$task_package" <<'PY'
import json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
for package in json.loads((root/'lake-manifest.json').read_text())['packages']:
    if package['type'] != 'git':
        raise SystemExit('unexpected non-Git dependency: ' + package['name'])
    directory = root/'.lake/packages'/package['name']
    def git(*args):
        return subprocess.check_output(['git','-C',str(directory),*args],text=True).strip()
    if git('rev-parse','HEAD') != package['rev'] or git('status','--porcelain','--untracked-files=no'):
        raise SystemExit('dependency revision/worktree mismatch: ' + package['name'])
print('Pinned dependency revisions and tracked worktrees match:', root)
PY
done
```

## Specification proofs

The unchanged package checker performs its fresh package build, existing kernel
check, signature/axiom/census audits, and existing adverse controls. Keep its
new `.lake/evidence/` receipt and raw streams with the new run; do not overwrite
the delivered historical receipts.

```bash
task_spec="$task_math/specifications/lean_statement_proofs_20261011"
(
  cd "$task_spec"
  "$task_python" -B -S check.py --execute
) >"$task_run/spec-existing-gate.stdout" 2>"$task_run/spec-existing-gate.stderr"
(
  cd "$task_spec"
  lake env lean --version
) >"$task_run/spec-version.stdout" 2>"$task_run/spec-version.stderr"
```

Run the delivered generated program, preserving its actual exit code and empty
stderr. The validator requires a pure replay stream:

```bash
if (cd "$task_spec" && lake env lean "$task_delivery/verified-replay-20261029/spec-kernel-replay.lean") \
  >"$task_run/spec.stdout" 2>"$task_run/spec.stderr"
then task_spec_exit=0
else task_spec_exit=$?
fi
printf '%s\n' "$task_spec_exit" >"$task_run/spec.exit"
"$task_python" -B -S "$task_delivery/replay-tool/replay_tool.py" validate \
  --transcript "$task_run/spec.stdout" --stderr "$task_run/spec.stderr" \
  --exit-code "$task_spec_exit" \
  --expected-census "$task_delivery/verified-replay-20261029/spec-expected-census.json" \
  --program "$task_delivery/verified-replay-20261029/spec-kernel-replay.lean" \
  --output "$task_run/spec-consistency.json"
```

## Smooth-source realization

Build all 13 audited modules from their existing package. The axiom-audit file
is executed directly; it is not a declared Lake library target. Its full module
census includes private and generated constants, not just the 116 indexed
public theorem names.

```bash
task_smooth="$task_math/frontiers/source_smooth_realization_20261011"
(
  cd "$task_smooth"
  sha256sum -c SOURCE_SHA256SUMS
  lake build GaussianSeriesFoundation SpectralWeights SourceFourierSeries \
    SourceC4 SourceGaussianLaw SourceLattice SourceKernel SourceCovariance \
    SourceTorus SourceTorusC4 SourceSmooth SourceSmoothCarrier SourceSmoothControls
  lake env lean SourceSmoothAxiomAudit.lean
) >"$task_run/smooth-build-and-audit.stdout" 2>"$task_run/smooth-build-and-audit.stderr"
if (cd "$task_smooth" && lake env lean "$task_delivery/verified-replay-20261029/smooth-kernel-replay.lean") \
  >"$task_run/smooth.stdout" 2>"$task_run/smooth.stderr"
then task_smooth_exit=0
else task_smooth_exit=$?
fi
printf '%s\n' "$task_smooth_exit" >"$task_run/smooth.exit"
"$task_python" -B -S "$task_delivery/replay-tool/replay_tool.py" validate \
  --transcript "$task_run/smooth.stdout" --stderr "$task_run/smooth.stderr" \
  --exit-code "$task_smooth_exit" \
  --expected-census "$task_delivery/verified-replay-20261029/smooth-expected-census.json" \
  --program "$task_delivery/verified-replay-20261029/smooth-kernel-replay.lean" \
  --output "$task_run/smooth-consistency.json"
```

## C297 native packet, including Lifetime

Use the unchanged native packet files and the resolved pinned `formal/` Lake
environment. Build into a new external directory so previously compiled formal
modules cannot substitute for this packet. The resolved formal search path is
made absolute before changing directories. `Lifetime` is built and included in
the empty-kernel replay even though the older hosted runner's separate replay
loop omitted it.

```bash
task_c297="$task_math/frontiers/regular_superlevel_geometry_20261020"
"$task_python" -B -S "$task_c297/c297_hosted_verify.py" source \
  --manifest-sha256 c339a9d1ca5b9ef70f8d581ea0760f11922aba2cd16cbeeb7aba02092b476e94 \
  >"$task_run/c297-source.stdout" 2>"$task_run/c297-source.stderr"
task_formal_path="$(cd "$task_math/formal" && lake env printenv LEAN_PATH)"
task_formal_path="$("$task_python" -B -S -c 'import os,pathlib,sys; root=pathlib.Path(sys.argv[1]); print(os.pathsep.join(str((root/p).resolve()) for p in sys.argv[2].split(os.pathsep)))' "$task_math/formal" "$task_formal_path")"
task_lean_prefix="$(cd "$task_math/formal" && lake env lean --print-prefix)"
task_native_lean="$task_lean_prefix/bin/lean"
mkdir "$task_run/c297-objects"
task_c297_path="$task_run/c297-objects:$task_formal_path"
for task_module in Lifetime RegularPointChart RegularSuperlevel \
  RegularSuperlevelLiteral RegularSuperlevelAxiomAudit
do
  task_warning_flags=()
  if test "$task_module" != Lifetime
  then task_warning_flags=(-DwarningAsError=true)
  fi
  (
    cd "$task_c297/project"
    LEAN_PATH="$task_c297_path" "$task_native_lean" -j2 -M14000 \
      "${task_warning_flags[@]}" \
      -o "$task_run/c297-objects/$task_module.olean" "$task_module.lean"
  ) >"$task_run/c297-build-$task_module.stdout" 2>"$task_run/c297-build-$task_module.stderr"
done
if (cd "$task_c297/project" && LEAN_PATH="$task_c297_path" "$task_native_lean" -j2 -M14000 "$task_delivery/verified-replay-20261029/c297-kernel-replay.lean") \
  >"$task_run/c297.stdout" 2>"$task_run/c297.stderr"
then task_c297_exit=0
else task_c297_exit=$?
fi
printf '%s\n' "$task_c297_exit" >"$task_run/c297.exit"
"$task_python" -B -S "$task_delivery/replay-tool/replay_tool.py" validate \
  --transcript "$task_run/c297.stdout" --stderr "$task_run/c297.stderr" \
  --exit-code "$task_c297_exit" \
  --expected-census "$task_delivery/verified-replay-20261029/c297-expected-census.json" \
  --program "$task_delivery/verified-replay-20261029/c297-kernel-replay.lean" \
  --output "$task_run/c297-consistency.json"
```

This is current local kernel execution. It does not recreate a historical
workflow event, satisfy an expired date guard, or manufacture hosted execution
credit. The original package controls and their intended missing-premise
rejections remain in `resumed-20261029/c297/`; actual sorry/custom-axiom controls
for both native packets remain in their corresponding immutable raw receipts.
The portable auditor's own positive/adverse/source-mutation checks are documented
in [the tool README](replay-tool/README.md), with its
[registry addendum](replay-tool/REGISTRY_ADDENDUM.md).

## Final identities and interpretation

Repeat the dependency revision/worktree check from preparation, repeat the
source-manifest identity check, then require a clean tracked/untracked status:

```bash
test -z "$(git -C "$task_math" status --porcelain --untracked-files=all)"
(cd "$task_delivery" && sha256sum -c SHA256SUMS)
printf 'New execution logs and consistency records: %s\n' "$task_run"
```

Preserve the new execution logs, actual exit files, compiled objects and their
hashes with their source/runtime identities. The Python consistency output is
not an execution authenticator: mathematical evidence is the actual pinned
Lean build/replay together with those bindings. Never copy a historical green
result into a new run without executing it.

The delivered top-level `receipt.json` is a **new observation index** binding
the 207 literal declarations to the manifest, axiom observations and actual
execution artifacts. It does not rewrite the original package receipts or
their historical status language. `spec-proof-existing-gate/`,
`spec-kernel-closure/`, `resumed-20261029/`,
`all-literal-axioms-20261029/`, and `verified-replay-20261029/` retain the original
raw observations. An interrupted broad full-import experiment remains labeled
interrupted and receives no passing credit. These component receipts do not
declare global CBZ‑0 or close any semantic or vacuity boundary beyond the exact
indexed statements.
