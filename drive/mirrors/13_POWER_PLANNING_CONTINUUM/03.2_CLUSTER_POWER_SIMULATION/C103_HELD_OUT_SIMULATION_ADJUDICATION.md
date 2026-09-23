# C103 Held-Out Simulation Adjudication

**Grade:** `PRE-REGISTERED-DIAGNOSTIC-FAILURE`

## Registered verdict

The primary gate failed.

At grid \(256\), on the frozen window

\[
[0.005,0.05],
\]

the cumulative slope estimate was

\[
0.1487
\]

with field-bootstrap 95% interval

\[
[0.1370,0.1616],
\]

which excludes the predicted value

\[
\frac23.
\]

Every pre-registered interval at both grids excluded \(2/3\) in the same
direction. The frozen kill signal therefore triggered.

This result may not be relabeled a pass.

## Grid results

| grid | window | slope | 95% interval |
|---:|---:|---:|---:|
| 192 | 0.005–0.05 | 0.1969 | [0.1844, 0.2103] |
| 192 | 0.01–0.08 | 0.1743 | [0.1613, 0.1870] |
| 192 | 0.003–0.03 | 0.2906 | [0.2724, 0.3097] |
| 256 | 0.005–0.05 | 0.1487 | [0.1370, 0.1616] |
| 256 | 0.01–0.08 | 0.1673 | [0.1530, 0.1804] |
| 256 | 0.003–0.03 | 0.1696 | [0.1546, 0.1840] |

## Instrument diagnosis

The first diagnostic is the scale of the smallest bars.

For grid spacing \(h=24/n\):

```text
n=192:
    q25(lifetime)/h^2 = 0.571

n=256:
    q25(lifetime)/h^2 = 0.654
```

A large population of bars collapses toward the diagonal at an
\(h^2\)-class scale. This is the expected scale of smooth-field interpolation
error and strongly suggests a discretization layer.

The pre-registered estimator used the total cumulative count

\[
N_h(\ell'\le\ell)
\]

without separating bars that are unstable under grid refinement. An additive
population collapsing to zero flattens the log-log slope.

However, this does **not** close the diagnosis. Exploratory windows above the
visible \(h^2\) layer still produced slopes below \(2/3\). Therefore the
current evidence does not establish that discretization explains the full
discrepancy.

## Adjudication

```text
pre-registered empirical gate:
    FAILED

kill signal:
    TRIGGERED

mathematical -1/3 derivation:
    NOT AUTOMATICALLY KILLED BY THIS DISCRETE PROXY

continuum validation:
    OPEN

Q0-B project terminal status:
    NOT CLOSED
```

The correct status is:

```text
INSTRUMENT-OR-THEORY-UNRESOLVED
```

not “the theorem survived” and not “the theorem is false.”

## Required C104 repair

The next experiment must be frozen before execution and must use:

1. a Freudenthal triangulation rather than a four-neighbor vertex graph;
2. coupled multi-resolution samples generated from identical Fourier
   coefficients;
3. cross-resolution persistence-pair matching;
4. removal or separate accounting of bars not stable under refinement;
5. fresh seeds;
6. raw and continuum-stable results reported together.

Only a continuum-consistent held-out test can discharge the failed empirical
gate.
