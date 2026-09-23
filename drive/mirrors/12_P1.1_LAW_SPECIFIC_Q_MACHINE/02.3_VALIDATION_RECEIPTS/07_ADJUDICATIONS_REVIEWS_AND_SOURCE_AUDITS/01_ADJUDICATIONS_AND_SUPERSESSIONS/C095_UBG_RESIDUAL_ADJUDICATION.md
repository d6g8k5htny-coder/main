# C095 UB-G Residual-Uniformity Adjudication

**Amendment type:** CORRECTION  
**Affected claims:** `UPPER_RATE_PROGRAM_GRADE`, `Q0_LIMIT`  
**Frozen C094 bytes:** untouched

## Gate finding

The C094 base assembly is

\[
\frac{0.657+
2.8185310984873743106223995278378565957688812716057680797457}{0.80}
=
4.344413873109217888277999409797320744711101589507210099682125.
\]

The coefficient margin to \(4.35\) is

\[
4.35-
4.344413873109217888277999409797320744711101589507210099682125
=
0.005586126890782111722000590202679255288898410492789900317875.
\]

But the source-level inequality also names positive residuals:

\[
\Gamma(r)+\operatorname{Collar}(r).
\]

Those residuals were not inputs to the C094 arithmetic expression. Gate 14
therefore fails by construction.

The rounded display inputs

\[
(0.66+2.82)/0.80=4.35
\]

leave **zero** displayed-input margin.

## Exact replacement

The live structural inequality is

\[
\boxed{
D(r)
\le
\frac{C_{\rm near}(r)+C_{\rm ext}(r)}
{P_{\rm typed}(r)}r^3
+
\Gamma(r)
+
\operatorname{Collar}(r).
}
\]

A uniform \(4.35r^3\) display requires the additional certificate

\[
\sup_{0<r\le0.025}
\left[
\frac{C_{\rm near}(r)+C_{\rm ext}(r)}
{P_{\rm typed}(r)}
+
\frac{\Gamma(r)+\operatorname{Collar}(r)}{r^3}
\right]
\le4.35.
\]

No such complete arithmetic object is present in the available C091–C094
artifacts.

## Status

```text
structural UB-G decomposition:
    DERIVED

uniform decimal 4.35:
    BLOCKED on UB_G_RESIDUAL_UNIFORM

q0 limit derived from the cubic UB-G rate:
    PROVEN-MODULO UB_G_RESIDUAL_UNIFORM
```

The unblock artifact is a full-domain Gamma/collar coefficient certificate or
a newly assembled larger coefficient with every residual present and rounded
up.
