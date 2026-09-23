# C094 UB-G DISPLAY ADJUDICATION

**Item:** Directive v2.1 §2.2  
**Disposition:** **CORRECTION branch**  
**Frozen C093 archive:** unchanged  
**Affected claim IDs:** `UB_G`, `RATE_PROGRAM_GRADE`, `Q0_LIMIT`

## Finding

At 60 decimal digits,

\[
\frac{0.030449}6(576-9\pi)(1.014)
=
2.8185310984873743106224.
\]

Using the registered near coefficient \(0.657\),

\[
\frac{0.657+2.8185310984873743106}{0.80}
=
4.344413873109217888277999.
\]

Using the displayed rounded inputs,

\[
\frac{0.66+2.82}{0.80}=4.35.
\]

The frozen display \(4.3\) is therefore a downward rounding by \(0.05\) from
the displayed-input assembly and by
\(0.0444138731092178883\) from the unrounded-input
assembly. No C092 correction-ledger entry or displayed certificate licenses
that downward movement.

## Correction

Replace every live occurrence of the upper coefficient \(4.3\) by \(4.35\)
in the C094 successor release.

The corrected program-grade theorem is

\[
\boxed{
0.8411r^3
\le
1-q(r,6/5)
\le
4.35r^3,
\qquad
0<r\le0.025,
\quad L=24.
}
\]

The inherited COL and Gamma-kill conditions are unchanged. Relative to the
unrounded near/exterior inputs, \(4.35\) leaves coefficient margin

\[
4.35-4.344413873109217888278
=
0.005586126890782111722001.
\]

At \(r=0.025\), this is absolute probability margin

\[
8.728323266847049565626e-8,
\]

far above the recorded \(e^{-92}\) and \(10^{-46}\)-class station terms.
The derived COL \(O(r^3)\) backstop remains part of the same program-grade
condition set.

## Corrected endpoint

At \(r=0.025\),

\[
1-q\le4.35(0.025)^3
=
0.00006796875,
\]

hence

\[
q\ge0.99993203125.
\]

The lower side is unchanged, so

\[
q\le0.9999868578125.
\]

## Root replacement

```text
RATE_PROGRAM_GRADE
old: e24569bfedfac0d02a30ea606bcf271ad9a06650b2e935f8bdd81aa917ef5d48
new: 2521f134f94c61c7f7b27917f1ca591224acc89fb3010c1f802ae2ca52138d08

Q0_LIMIT
old: 0c643c9d26fdda9e87065b040c6d55d91c932a8748f4a5eb223f1babf10cb4c6
new: 8974f6cb3252fccd16b1e2af5a978b55090ee451bccb1f183667822bf44cd01b
```

The statement \(q(r,6/5)\to1\) is unchanged; its hash changes because its
`UB_G` dependency changed.

## Amendment classification

```text
type: CORRECTION
cause: non-conservative downward rounding of an upper bound
review status: internally re-derived at 60 dps
frozen C093 bytes: untouched
replacement release: C094 successor-era release
```
