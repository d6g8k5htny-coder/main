# C094 LOWER-COEFFICIENT DIMENSIONAL ADJUDICATION

**Directive row:** §2.3(3)  
**Disposition:** `ADJUDICATED-CONFLICT` → **CORRECTION**  
**Frozen C092/C093 archives:** untouched

## Source-level formula

The lower-side source states

\[
C^*(r)\ge0.8411
\]

and then assembles

\[
1-q(r,6/5)
\ge
C^*(r)r^3\,AO(r)\,(1-O(r^3)).
\]

It separately records

\[
AO(r)\ge1-2.24r^3-2e^{-111}-\text{named residues}.
\]

Therefore \(0.8411\) is a lower coefficient for the qualifying first moment,
not automatically the final defect probability.

At \(r=0.025\), even before Bonferroni and the other named residues,

\[
0.8411\left(1-2.24(0.025)^3\right)
=
0.8410705615
<
0.8411.
\]

The displayed finite lower theorem cannot follow from the displayed inputs.

## Correction

Withdraw the live core claim

\[
1-q(r,6/5)\ge0.8411r^3.
\]

Preserve the correctly typed statement

\[
C^*(r)\ge0.8411,
\qquad0<r\le0.025,
\]

at the inherited program/verification grade.

Preserve the rung-tagged measured coefficient \(0.946\) at \(r=0.025\).

Move every finite probability lower sharpening to `Q0-SHARP`, where it
requires the complete chain

```text
BONFERRONI
→ FAR_DECORRELATION
→ BR-REP
→ BR-MARK
→ SIX_PIN_PALM_UNIFORMITY
→ exact AO and loss propagation
→ full-domain infimum certificate
```

## Corrected live core

The corrected C094 core theorem is one-sided:

\[
\boxed{
0\le1-q(r,6/5)\le4.35r^3,
\qquad0<r\le0.025,
\quad L=24,
}
\]

at the same inherited program/verification grade.

It still implies

\[
\boxed{q(r,6/5)\to1}.
\]

## Root replacement

```text
old two-sided root:
e24569bfedfac0d02a30ea606bcf271ad9a06650b2e935f8bdd81aa917ef5d48

new upper-rate root:
3d6d82aed7661c388528755c8bec12ad73a991a76995275915e88c2bbcf2d0e5

old Q0_LIMIT root:
0c643c9d26fdda9e87065b040c6d55d91c932a8748f4a5eb223f1babf10cb4c6

new Q0_LIMIT root:
772f8b41b2eb13a0dfec415f415ff74049efbed33783ed24af46f6cdcbc38d73
```
