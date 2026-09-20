# C104 Continuum-Persistence Validation Adjudication

**Grade:** `PRE-REGISTERED-MEASURED-DIAGNOSTIC`  
**Primary verdict:** `SURVIVES`  
**Kill signal:** `NOT TRIGGERED`

## 1. Frozen primary gate

The primary gate was fixed before execution:

```text
resolution pair:
    256 -> 512

upper lifetime cutoff:
    U = 0.2

minimum stable bars:
    500

target density exponent:
    alpha = -1/3
```

The observed result was

\[
\widehat\alpha=-0.48144,
\]

using

\[
543
\]

refinement-stable bars. The field-bootstrap 95% interval was

\[
\boxed{
[-0.71150,-0.24375].
}
\]

It contains

\[
-\frac13.
\]

All 1000 primary bootstrap replicates were valid. Therefore the
pre-registered primary gate **survives**.

This is not a claim that the point estimate equals \(-1/3\). It is the exact
verdict licensed by the frozen criterion.

## 2. Continuum-consistency diagnostics

The mean coarse-to-fine interpolation error decreased across the nested
resolution pairs:

| pair | mean \(\varepsilon_h\) |
|---:|---:|
| 128→256 | 0.05482 |
| 192→384 | 0.02495 |
| 256→512 | 0.01418 |

The fraction of fine bars classified as refinement-stable increased:

| pair | stable fraction |
|---:|---:|
| 128→256 | 0.5555 |
| 192→384 | 0.6804 |
| 256→512 | 0.7336 |

This is the direction required of a continuum-consistent reconstruction:
interpolation error falls while more of the diagram survives refinement.

## 3. All frozen exponent fits

| pair | \(U\) | \(\widehat\alpha\) | 95% interval | bars |
|---:|---:|---:|---:|---:|
| 128→256 | 0.1 | inconclusive | — | 0 |
| 128→256 | 0.2 | boundary/insufficient | broad | 5 |
| 128→256 | 0.3 | boundary estimate | broad | 195 |
| 192→384 | 0.1 | boundary/weak | broad | 15 |
| 192→384 | 0.2 | -0.5821 | [-0.9500,-0.0240] | 320 |
| 192→384 | 0.3 | -0.5171 | [-0.7590,-0.2693] | 565 |
| 256→512 | 0.1 | -0.5519 | [-0.9500,0.3115] | 202 |
| 256→512 | 0.2 | -0.4814 | [-0.7115,-0.2437] | 543 |
| 256→512 | 0.3 | -0.5005 | [-0.6385,-0.3791] | 805 |

The coarsest fits are data-limited because the stability threshold
\(4\varepsilon_h\) leaves few bars below the smaller upper cutoffs.

## 4. Preserved secondary tension

At the finest pair with \(U=0.3\),

\[
\widehat\alpha=-0.50047,
\]

with interval

\[
[-0.63855,-0.37905],
\]

which excludes \(-1/3\).

This result is preserved. Possible explanations include:

- the \(U=0.3\) window is outside the asymptotic regime;
- residual finite-resolution bias;
- misspecification of a single power law on the entire truncated window;
- a genuine deviation from the proposed exponent.

Because the primary window was frozen at \(U=0.2\), the \(U=0.3\) result does
not retrospectively replace the primary verdict. It does prohibit describing
C104 as a precise empirical confirmation.

## 5. Relation to the failed C103 proxy

C103 used a four-neighbor vertex graph and raw cumulative small-bar counts.
Its gate failed, with slopes around \(0.15\)–\(0.29\), and its kill signal
triggered.

C104 changed only through a frozen repair protocol:

1. periodic Freudenthal triangulation;
2. identical Fourier realizations on nested grids;
3. explicit coarse-to-fine interpolation error;
4. diagram matching with diagonal options;
5. exclusion from the exponent fit only when bars failed the frozen
   stability criteria;
6. per-realization left truncation in the likelihood.

The C103 failure remains a real failure of that proxy. C104 is the
pre-registered replacement validation, not a relabeling of C103.

## 6. Project disposition

```text
CONTINUUM-PERSISTENCE-VALIDATION:
    DISCHARGED

Q0-B mathematical core:
    CORE-CLOSED AT PROGRAM GRADE

Q0-B empirical gate:
    SURVIVES

Q0-B terminal state:
    EXTERNAL-REVIEW-TRACK

external dependency:
    INDEPENDENT SARD-G SPECIALIST REVIEW

numerical C_*:
    NOT-CLAIMED
```

The experiment supports the theorem at the level specified by the frozen
gate. It does not prove the theorem and does not remove the secondary
finite-window tension.
