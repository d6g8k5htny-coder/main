# Q0-C096 Reproduction and Verification Guide

## Release

```text
Q0-C096-RECONCILED-GATE
```

This release contains the reconciled Gate Framework Master v1.2, Gate Kernel
2.0, the 36-case independent validation battery, the migrated legacy registry,
the UB-G residual audit, the conditional Q0 repair contract, the BR-MARK
adjudication, and immutable copies of the five user-supplied source artifacts.

## Environment

The pinned Python environment is recorded in:

```text
C096_ENVIRONMENT.json
requirements-c096.txt
```

Internal validation does not require network access. The historical file-library
source review used connected retrieval before this release was assembled.

## Important separation: immutable verification root versus execution copy

Several executable instruments produce versioned result JSON files beside their
source code. Running such a writer inside the immutable extraction would alter a
file whose original bytes are listed in `C096_MANIFEST.json`.

Therefore the clean-extraction protocol uses two directories:

```text
immutable_root/
    extracted release; used only for manifest verification

execution_copy/
    disposable byte-for-byte copy; used to rerun writers and audits
```

This separation is deliberate. A release verifier should never be invalidated by
the verification process itself.

## ZIP-mode verification

With the ZIP and manifest in the same directory:

```bash
python3 verify_q0_c096_release.py
```

Expected result:

```text
mode: zip
valid: true
declared_files: 44
verified_files: 44
zip_entries_expected: 45
```

The manifest is self-excluded and is the forty-fifth ZIP entry.

## True fresh-extraction protocol

### 1. Extract into an empty immutable root

```bash
mkdir immutable_root
cd immutable_root
unzip ../Q0_C096_RECONCILED_GATE_RELEASE.zip
python3 verify_q0_c096_release.py
```

The first verifier invocation must pass before any executable writer is run.

### 2. Create a disposable execution copy

From the parent directory:

```bash
cp -a immutable_root execution_copy
cd execution_copy
python3 audit_q0_c096_release.py
```

The deep audit:

- compiles every current Python file;
- parses every JSON artifact;
- checks control characters and hard-coded active output paths;
- reruns the Gate Kernel v1.2 red-team;
- reruns Gate Kernel 2.0;
- reruns all 36 independent validation cases;
- reruns the UB-G residual arithmetic instrument;
- checks the migrated registry, conditional Q0 contract, BR-MARK adjudication,
  PDF preflight, release state, and immutable input hashes.

It is expected to rewrite generated reports **inside the disposable execution
copy only**.

### 3. Reverify the immutable root

```bash
cd ../immutable_root
python3 verify_q0_c096_release.py
```

The second verifier invocation must also pass. This proves that rerunning every
check did not mutate the immutable verification root.

### 4. Local-import check

Inside the execution copy:

```bash
python3 -c \
  'import gate_kernel_v2_0; print(gate_kernel_v2_0.__file__)'
```

The printed module path must lie inside `execution_copy`, not in its parent or a
global project directory.

## Principal executable checks

```bash
python3 C096_GATE_KERNEL_V1_2_REDTEAM.py
python3 gate_kernel_v2_0.py
python3 validate_gate_kernel_v2_0.py
python3 C096_UBG_RESIDUAL_SOURCE_AUDIT.py
python3 audit_q0_c096_release.py
```

Expected high-level outcomes:

```text
Gate Kernel v1.2 red-team:
    17 tests
    17 reproduced implementation gaps
    0 harness errors

Gate Kernel 2.0 validation:
    36 cases
    36 passed
    0 failed

Conditional Q0 contract:
    shell_valid = true
    conditional_promotable = true
    unconditional_promotable = false

Legacy Q0 registry migration:
    archive admission = true
    candidate admission = false
    promotion admission = false
```

## What is established

- The uploaded Gate Kernel v1.2 has the seventeen recorded shell and registry
  gaps under the frozen adversarial battery.
- Gate Kernel 2.0 returns all 36 preregistered positive and negative outcomes.
- The migrated legacy registry is structurally valid as an archive and is not
  promotable as a theorem.
- The Q0 upper-rate and q0 implications have a shell-valid
  `Proven-Modulo` contract under three explicit residual conditions.
- The exact six-pin candidate-mark jet reduction is derived; the simple uniform
  mark-density transfer is falsified on the tested pair-scaled charts.

## What is not claimed

- A gate PASS does not prove mathematical truth.
- The three residual conditions in the Q0 contract are not discharged here.
- The decimal upper coefficient 4.35 is not promoted.
- The Q0 selection limit is not claimed unconditionally by this release.
- The BR-MARK theorem is not closed.
- The migrated `q0 registry.json` is not the frozen Q0 theorem registry.
- Diagnostic two-rung scaling is not mislabeled as a continuum theorem.

## Integrity

The final bundle hash and detached fresh-extraction result are recorded in:

```text
C096_ATTESTATION.json
C096_FRESH_EXTRACTION_TEST.json
```

Those two detached artifacts are created after the immutable ZIP is finalized;
they are not included inside the ZIP whose hash they attest.
