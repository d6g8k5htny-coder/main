# Universal Law — pinned Math verification, run004

Observed 29 October 2026. This is a completed **scoped formal verification delivery**, not a federation CBZ-0 declaration. Preserve the frozen `sealed/` directory. It contains315 hash-indexed files plus its own MANIFEST.json; the package manifest covers every delivered file except itself.

## What passed

Exact Math- commit8151f592918a845bf19eb8eafaf8444540a32734: nine module builds plus default build, nine ordinary exact-module kernel replays,63 public target axiom/type reports,116 compiled declarations, four hidden-admission controls, two compiled-injection sequences and five original negative controls. All82 command outcomes, source/runtime/dependency identities and log hashes are retained. The receipt SHA256 is7711c528031f26bb094718db736f960c546fcbb5c00f345fae1b201d4d85f561. The original seal manifest SHA256 is63d0c9e0268ffec38eeb2fe83a0d31850e740aaba39dbdf3caf76ef6970f6acf.

116 constants include63 public theorem targets,43 compiler auxiliary theorem constants and10 other declarations; these are not116 new research results. Approved target axioms are propext, Classical.choice and Quot.sound. Original target statements, bounds, predicates and source files were not weakened or changed.

## Boundaries

This proves the recorded literal formal statements against the pinned imported environment. It does not establish paper/model alignment, applicability to the intended Gaussian/Palm model, nonvacuity or model existence, or fresh replay of every imported Mathlib dependency. All63 targets retain UNASSESSED nonvacuity at this delivery scope.45 have Prop-premise binders; absence of such binders in18 others does not establish inhabitation of every parameter domain. The typed statements are in supplement/math-type-diagnostic/public-target-types.json.

Intentional sorry/axiom/native/metaprogram inputs in sealed/formal-evidence and execution control directories are rejection fixtures, never accepted proofs. Historical resource stops and earlier review failures remain at their original scopes outside this focused successful-run package; their existence is described in the final report. PRIVATE_TEST_FAILURES and all unselected parent-directory files are deliberately excluded. No inherited environment dump is included; selected Lake toolchain/search-path fields are recorded for reproducibility.

## Why the adapter exists

The normal prefix checker imported many environments concurrently and exceeded available memory. The unchanged gate was rerun through the pinned upstream LeanChecker.replayFromImports API one exact module/process at a time. The same pinned Lean executable uses ten strictly validated Lake path/toolchain fields, avoiding a resident Lake parent for Lean jobs. No original check or negative outcome was omitted. LEAN_NUM_THREADS=1, a sampled7GiB process-group RSS cap and900second per-command timeout fail closed. The monitor is sampled/same-process-group scoped, not a hard per-job kernel cgroup.

Adapter tests:21 cases and19 strict source mutants each in normal and optimized Python, with positive unmutated harnesses. Independent engineering review added14 inert launch probes per mode and checked original monitor-error/receipt failure controls. These finite engineering checks do not certify full Layer A or theorems. See sealed/tool-review/ and supplement/MATH_DRIVER_TOOLING_ADDENDUM.json. Maintenance: none for the frozen delivery; canonical adoption or path-porting requires its own verification.

## Offline identity verification

Python3.12 stdlib is sufficient. From this extracted directory, run:

```sh
python -B -S verify_identity.py
python -O -B -S verify_identity.py
```

The checker verifies package paths/sizes/SHA256s, original seal identity, all315 original sealed file entries and the successful receipt identity. This is file identity checking only, not another kernel run. The verification script is itself hash-indexed in PACKAGE_MANIFEST.json; whole-ZIP SHA256 is supplied separately in the delivery receipt.

## Repeat adapter self-tests without altering frozen evidence

Use a new disposable directory (replace `/tmp/ul-math-driver-check` if it exists):

```sh
mkdir /tmp/ul-math-driver-check
cp sealed/run_math_sequential_gate.py sealed/test_math_sequential_driver.py sealed/mutate_math_sequential_driver.py /tmp/ul-math-driver-check/
cd /tmp/ul-math-driver-check
python -B -S -m unittest test_math_sequential_driver -v
python -O -B -S -m unittest test_math_sequential_driver -v
python -B -S mutate_math_sequential_driver.py
python -O -B -S mutate_math_sequential_driver.py
```

The mutation runner writes its own logs beside the disposable copies. These tests inspect/exercise isolated driver functions; they do not import or execute the heavy Math gate.

## Exact full-gate setup and execution

This frozen execution recipe intentionally retains its original absolute paths. Linux x86_64, Python3.12, git, curl, tar with zstd support, ps and sufficient RAM/disk are required. Allocate at least16GiB system memory, keep unrelated heavy jobs stopped, and allow20GiB+ free disk for toolchain/dependencies/cache. Network is needed for initial acquisition unless the exact pinned bytes are already available. This ZIP does not include the580MB Lean release or multi-GB Mathlib cache and does not claim an offline full build.

Run the following on a fresh machine/empty designated paths, never over another checkout. Set PACKAGE to the absolute extracted directory. The read-only recorded identities are independent of this optional replay:

```sh
PACKAGE=/absolute/path/Universal_Law_Math_8151_Run004
mkdir -p /workspace/cbz0/lean-foundation /workspace/cbz0/repos /workspace/research/audit/formal/toolchain /workspace/universal-law
curl -fL https://github.com/leanprover/lean4/releases/download/v4.34.1/lean-4.34.1-linux.tar.zst -o /tmp/ul-lean-4.34.1-linux.tar.zst
printf '%s  %s\n' 47bf4bbd78f70c2e9670598ab7124d92b6efb7330ff33e5fbb4030f6fd72e4e4 /tmp/ul-lean-4.34.1-linux.tar.zst | sha256sum -c -
tar --zstd -xf /tmp/ul-lean-4.34.1-linux.tar.zst --strip-components=1 -C /workspace/research/audit/formal/toolchain
export PATH=/workspace/research/audit/formal/toolchain/bin:$PATH
export LEAN_NUM_THREADS=1
lean --version
# Expected Lean4.34.1; upstream commit5045d0056413266e57c625dcd7c365b10e377c52.
git clone https://github.com/d6g8k5htny-coder/Math-.git /workspace/cbz0/repos/Math-
git -C /workspace/cbz0/repos/Math- checkout --detach 8151f592918a845bf19eb8eafaf8444540a32734
git clone https://github.com/d6g8k5htny-coder/main.git /workspace/universal-law/website-repair-20261028
git -C /workspace/universal-law/website-repair-20261028 checkout --detach 22567eeff95f3184048b3e2190ecab908bfc115a
cp "$PACKAGE/sealed/run_math_sequential_gate.py" "$PACKAGE/sealed/OneModuleReplay.lean" "$PACKAGE/sealed/LeanChecker-pinned.lean" /workspace/cbz0/lean-foundation/
# Existing Main endpoint is used solely as the API replay positive control.
(cd /workspace/universal-law/website-repair-20261028/formal && lake build)
(cd /workspace/cbz0/repos/Math-/formal && lake exe cache get)
python -B -S /workspace/cbz0/lean-foundation/run_math_sequential_gate.py
```

Mathlib is pinned to d13f23b723b8a846827a245b89c10fc7d3f11612; all nine dependency revisions are in sealed/source/lake-manifest.json and the original gate verifies the pinned sources/dependencies. Do not run lake update or repin dependencies to bypass failures. The script automatically chooses a new math-run-NNN, preserves earlier receipts, verifies adapter/runtime/source hashes before and after, and returns nonzero on any refused phase. Full replay is optional for consuming the frozen evidence but required for claiming a newly observed gate. A machine with missing binaries/cache, network restrictions or insufficient resources must report the actual failure; do not infer success from this historical run.
