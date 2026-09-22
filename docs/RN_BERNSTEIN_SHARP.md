# RN Bernstein and sharp-variance successor

The immutable [successor packet](../research/campaigns/rn_bernstein_sharp_variance_20260921_v1.zip)
improves the existing local RN wedge bound. It does not enlarge its domain or
replace the original inner-wedge result. Both versions remain byte-for-byte.

At fixed normalized SIDE24 pin radius `r=1/20`, birth `b=6/5`, and the original
x-axis six pins, the domain is `rho in [1/10,11/100]`, signed turns
`[-1/1024,1/1024]`, with exact area `21*pi/5120000`. The full mark interval
`[-1/96000,1/96000]` gives the height window `[57599/48000,6/5]`.

One auxiliary rectangle `[9999/100000,11/100] x [-7/10000,7/10000]` now suffices.
The degree-36 tensor Bernstein enclosure retains common polynomial coefficients,
Gaussian image tails, Taylor and conditioning remainders, and exact preconditioner
determinant conversion. It proves `D > 8/10^25`. The shared polynomial
`mu_fx^2 - 103 Var(fx)` with mean and variance error corrections proves `Q >= 103`
throughout the rectangle and full height window. The old one-rectangle refusal
is replayed and retained alongside the success; it is not a mathematical
counterexample.

The sharp stationary-variance argument replaces moment coefficient `512` with
`64.000001`, using `(m4+m2^2)^3 < 64.000001`. The exact normalized periodized
kernel matters: this coefficient is slightly greater than `64`. Composition gives
`I(y)<1/1000000` and `integral_W I < 33/2560000000000`, conditional on the same
imported fixed-r H3 floor `0.0077592917375327855` and six-pin energy allowance `<8`.

The packet does not reprove those imported premises, certify the full annulus,
all radii or pin orientations, identify the weighted Palm/event interface, or
close q0 or an original prize problem. No scientific status changes. Reviews
are source-exposed, same-provider technical checks with zero organizational
independence credit.

## Reproduction and custody

Run with normal Python 3.11:

```bash
python tools/rn_bernstein_sharp_check.py
python tools/rn_bernstein_sharp_check.py --repo /absolute/checkout --output-dir /outside/checkout/fresh-run
```

Default output is temporary. An explicit output directory must be fresh, outside
both the selected checkout and this wrapper's checkout, with no symlink alias.
The whole run has a 180-second default budget (maximum 300 seconds).

The wrapper pins the 157355-byte ZIP SHA-256
`868682c92d41015fe9a41714aeef243d9a8ff4713bba29d613ca13529e797ce9`, all 48 members,
three exact manifests, and the conflict-free union of 36 repository dependencies.
Current exclusions are checked before packet bodies are materialized. Pinned
exclusion, inventory and coverage metadata and Python inputs are checked before
loading the N6 admission gate; any changed policy snapshot refuses before original
source reads, including new upstream exclusions. That gate additionally checks
current RN5 inventory, coverage and hold metadata before reading original RN5
source bodies. Historical RN5 programs remain data. The authenticated authored
N6 Taylor implementation is executed by the numerical children.

Only `close_local_wedge.py` is launched, once normally and once with `-O`; its
literal reviewed commands run `bernstein/check_v2.py --pieces 1` and
`sharp_variance/check.py`. No command comes from `RUN.json`, and the archived
draft successor and historical author tests are not executed. There are two
composition executions and four numerical child executions. All three
mathematical reports per mode must match the frozen bytes exactly; both attempt
files per mode must match canonical entries of the frozen report. The complete
ten-file output set per mode is checked. Wall-clock timings are NON-CERTIFYING,
typed and bounded, retained with actual identities, and excluded from historical
byte comparisons.

Before and after snapshots bind the archive, all extracted files, dependencies,
source admission, wrapper/runtime, outputs and logs. New controls test exclusions
and holds before source reads, malformed archives and duplicate JSON keys,
altered proofs/refusals, boolean aliases, stale inputs, fresh-output protection
and failed children. Archived author-test counts are historical; they are not
added to current pytest/JUnit counts. A replay receipt is unsigned local execution
evidence, not trusted time, a security sandbox, an environment lock, or detection
of a concurrent change followed by reversal.
