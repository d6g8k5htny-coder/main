# Execution and review evidence, 8 October 2026

This is software/mechanism evidence for the new suite. It does not establish a
general-field theorem, actual ideal IID inputs, a continuum error/window, a
coefficient prediction, a blinded experiment or scientific acceptance.

## Retained complete pilot

[observations.json](results/exploratory50/observations.json) is the actual default
run retained from the model worker and independently inspected by root and the
fresh reviewer. It contains 1,751,085 bytes; SHA-256:

    99b8aef1af864f320cd79148cc462d05ea30ef23f0e79d1fb6d6a72b94b0c51c

There are fifty model definitions and one hundred retained field rows, two per
model, at side24, cutoff3, grid16² and quantization scale65536. All one hundred
quantized-grid barcodes passed independent connectivity verification. The data
contain 309 positive finite bars, 100 essential classes and 25,191 zero pairs,
with zero failed rows. Combined bin totals are [26,29,46,45,39,43,41,3]; this is
coverage across different laws, not a pooled scientific estimand.

The worker generated the default run twice with byte-identical output. Root
verified the retained SHA, definitions/row totals and the saved observation
checker after copying it here. The review independently verified the same
artifact and exact arithmetic. Generator authenticity is not inferred from
the retained-record checker; fresh execution replay is separate evidence.

Recheck the stored artifact with:

```sh
python3 -B -S experiments/universality/run_pilot.py --verify experiments/universality/results/exploratory50/observations.json
```

## Tests and repository boundary

The complete new suite passed 26 tests in normal and optimized Python. The
existing exact-H0 suite passed 9 tests. Test-first missing-feature failures and
subsequent boolean-schema/convention/provenance regression failures were
observed before their fixes. A completely failed injected fifty-row run retains
all rows and exits1; injected inputs are labelled arithmetic/test inputs.

The existing repository suite passed 429 tests with one existing skip under
the bundled Python3.12 runtime, its executable directory on PATH, and
TMPDIR=/private/tmp. The initial system-Python3.9 baseline reported two failures
and22 errors: missing hashlib.file_digest, missing python executable in the
shell control, and /var versus /private/var temporary-path identities. These
were environment differences; no existing production source was changed to
conceal them. Original and configured execution logs are retained in the task
workspace outside the checkout.

The existing landing/navigation/source-only formal gates passed. The latter
checked the unchanged31 arithmetic targets and reports alignment pending; no
new local Lean execution is claimed. Twelve local links in the new design,
plan, program and protocol were checked separately. The new workflow was parsed
with Ruby/Psych as full YAML. Hosted CI is separate evidence and must be read
on the actual proposal commit.

## Fresh review and repairs

Actual author roles: native OpenAI/Codex root wrote the program/protocol/fold
controls/workflow; internal helper benchmark_formal_audit wrote the catalog,
sampler/runner and their tests. Internal helper universality_review performed a
fresh read-only technical/source review. It independently attacked thirteen
record alterations, including omissions/duplicates, units/rounding/bin changes,
boolean parameters, source/scope/seed changes, quantized nodes and summaries;
the final code refused them.

The reviewer found one important hosted-custody gap: temporary rows would have
been lost after job cleanup, particularly after a failed pilot. The repaired
workflow preserves failure exit with pipefail, retains run/verify logs, and
uploads observations/logs with always() using the existing pinned artifact
action. The summary identifies the job outcome as the evidence for success.
The reviewer inspected the repaired YAML and found no remaining critical or
important source issue. Hosted upload behavior still requires actual execution.

Technical review is same-provider/source-exposed and carries organizational
independence credit zero. It is not an independent-human alignment verdict or
a scientific-status transition. The full research objective remains active.
