# G11 browser entrypoint contract plan

## Boundary

Close only the 28 G11 tool-reference candidates that are documented browser
check command-line programs but have no executable test or workflow reference.
Do not claim that command-line parser coverage executes a browser scenario,
qualifies the browser assertions, or replaces the tools' existing source-bound
real-browser evidence. Do not change mathematical statements, proof outcomes,
bounds, axioms, authorship, historical failures, site assets, hosted refs, or
canonical federation surfaces.

The red baseline is the exact `59b27b62352ce006411d1aad9d7ab617b3477282`
tree: G11 exits 1 with zero production findings and exactly 28 candidates, all
of them browser-check CLIs. A direct diagnostic run observed each CLI's
`--help` contract exit 0 with nonempty stdout and empty stderr in normal and
optimized modes, without changing tracked source.

## Implementation

1. Add one bounded unittest module with an explicit literal path at every call
   site so G11's static-reference result is reviewable and does not rely on
   loop or manifest inference.
2. For every CLI, execute the actual Python file with `--help`, assert exit 0,
   empty stderr, a `usage:` banner, and that the advertised program name is
   the selected file name.
3. Preserve optimization mode in child processes and remove inherited
   `PYTHONOPTIMIZE` so normal, `-O`, and `-OO` parent runs exercise the intended
   child mode deterministically.
4. Add adverse controls that prove a missing tool path and a CLI that refuses
   `--help` cannot be converted into green.
5. Run the focused module in normal, `-O`, and `-OO`; run the full G11 unit and
   mutation suites in normal and optimized modes; then run fresh exact-head
   censuses in normal and optimized modes and compare their bytes.

## Truthful outcome

A clean census after this change establishes only the bounded G11 static
documentation/reference criterion at the exact tested tree. It does not
establish runtime browser coverage, whole-tool verification, G0-G10, the
federation gate matrix, or CBZ-0.
