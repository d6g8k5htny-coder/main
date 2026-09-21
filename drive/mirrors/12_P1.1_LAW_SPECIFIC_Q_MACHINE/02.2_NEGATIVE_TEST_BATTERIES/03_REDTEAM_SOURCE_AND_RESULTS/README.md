# `02.2_NEGATIVE_TEST_BATTERIES/03_REDTEAM_SOURCE_AND_RESULTS`

Drive folder id `1-mFW7lEhiL9hiv5q3_zAXJS7WL3sXwEL`. The 2026-09-17 inventory gives this
folder **2 items** — one Python source and its JSON result record — and both are held
byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C096_GATE_KERNEL_V1_2_REDTEAM.json` | byte-exact | 28,267 |
| `C096_GATE_KERNEL_V1_2_REDTEAM.py` | byte-exact | 16,227 |

## The status banners, verbatim

The Python source states its own discipline in its opening docstring:

> Adversarial regression suite for uploaded `gate kernel v1_2.py`.

> This file does not modify the uploaded kernel. It constructs proof objects that should be rejected under the uploaded Master v1.1's own shell/core, bridge-node, hash-drift, and dependability language, then records whether the kernel rejects or accepts them.

> Every test is frozen by name before execution. A "vulnerability reproduced" result means the kernel accepted a structurally malformed or over-promoted object, or failed to reject a bad registry load.

The JSON result records seventeen tests, seventeen vulnerabilities reproduced, zero secure
rejections and zero harness errors, and it bounds its own reading explicitly:

> A reproduced vulnerability is a shell/registry implementation gap. It does not assert that any mathematical bridge is false; it shows that the kernel cannot enforce the uploaded Master v1.1 contract as written.

## What this does not establish

The battery was **not run here**. The seventeen reproduced vulnerabilities are the source's
own record of a run performed elsewhere, at a time the file records, against a kernel file
whose path inside the record is the author's temporary directory and not this repository.
The source's own interpretation line is the right reading and this repository adds nothing
to it: a reproduced vulnerability is a statement about an implementation, not about any
mathematics. Nothing here grades the kernel, the framework, or any claim either of them
touches.
