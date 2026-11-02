# Test-body identity preflight

New obligation: determine whether a supported ordinary synchronous None-return
`load_tests` hook can replace the declared body population without detection.
No prior tests, Lean work, scanner campaigns or hosted checks were rerun.

Both exact current runners have the same concrete gap. Default discovery calls
`load_tests` before capturing the selected population. A hook retaining both
ordinary method declarations but aliasing `test_second` to `test_first` produces
two distinct expected names and complete body events, while executing the first
original body twice and omitting the second. The passing variant returns exit 0
and `SURVIVED`. The assertion variant returns `KILLED` with two assertion failures,
but still omits the second original body. A same-name replacement subclass also
returns exit 0 and `SURVIVED` while both original body sentinels are absent.

These controls do not fabricate receipts or override TestCase.run. All selected
functions are ordinary synchronous functions and their ordinary successful
return is None. This is a population-identity defect inside the advertised
profile, not an arbitrary-code sandbox claim.

The explicit class-selector negative controls bypass the module hook and execute
both original bodies, including the second body's retained failure. Thus the gap
affects default/module discovery. The current kernel production mutation plan
uses class or individual-method selectors, while the layer-a runner's default
baseline and `run_one` discovery are exposed to module hooks. Current five test
modules contain no explicit top-level `load_tests` definition or assignment;
completed applicable source-specific evidence remains valid.

`RESULTS.json` records six new default-discovery executions and two named-class
controls in normal mode (fourteen subprocess commands including population
listing). Each fixture, unchanged copied runner, full structured result, stdout,
stderr and body trace is retained under `fixtures/`. `SOURCE_BINDINGS.json` binds
the active source generation. `EXISTING_APPLICABILITY.json` records the narrow
existing-source hook check and its limits.

Minimal repair: derive the canonical selected population from trusted standard
module/class declarations before module hooks; retain original class, function
and code identities in process; refuse unsupported population factories before
fixtures or bodies; compare identities after loading and at starts/body exits.
Explicit class and method selectors must retain their exact supported behavior.
The implementation must have actual omission controls and source guard mutations.
