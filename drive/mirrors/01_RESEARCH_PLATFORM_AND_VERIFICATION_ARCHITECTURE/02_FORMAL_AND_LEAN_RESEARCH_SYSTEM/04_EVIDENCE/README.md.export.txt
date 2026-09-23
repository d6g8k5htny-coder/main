# 04_EVIDENCE


Proofs and evidence are separated.


Every experiment must record:


- preregistration;
- exact source/code hash;
- environment and dependency lock;
- seeds;
- raw outputs;
- derived outputs;
- exclusions and failed runs;
- estimator definition;
- uncertainty method;
- stopping rule;
- whether the experiment can falsify a theorem, a finite-range model, or neither.


Empirical agreement never promotes a theorem. A reproducible counterexample can kill one.




## Active evidence layout
- `00_PREREG` — frozen protocols and numbered amendments.
- `01_RAW_OUTPUTS` — immutable raw results and failure logs.
- `02_DERIVED_RESULTS` — analyses derived from identified raw objects.
- `03_REPLICATION` — independent reruns, object cards, executable packages, and discrepancy audits.
- `04_ENVIRONMENTS_AND_SEEDS` — environments, dependencies, precision settings, seeds, and hardware/runtime metadata.


## FS2-EXP-001
The exact conditioned and determinant-weighted same-maximum loop experiment has a dedicated replication workspace under `03_REPLICATION/FS2-EXP-001 — CONDITIONED_AND_WEIGHTED_LOOP`. Its object card defines required executables, raw schema, practical Track A, theorem-sensitive Track B, and the direct good-event counterexample criterion.


Track A finite-r evidence may compare models but cannot falsify the present asymptotic theorem. Track B receives theorem-sensitive evidence credit only when conditioning, Taylor, H5, floating-point, and ODE errors are rigorously controlled. Failure to enter the certified regime is `INACCESSIBLE WITH CURRENT CONSTANTS`, not `SURVIVES`.


## Evidence-to-math transition
A reproducible counterexample or new empirical pattern enters `01_PREPPING_GROUNDS` as a new atomic correction/falsifier candidate. It never edits a Core object or theorem graph directly.