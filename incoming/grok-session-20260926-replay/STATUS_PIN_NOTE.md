# STATUS default-boundary note — do not move the freeze pin

Scientific effect: NONE. Proposed STATUS edit is custody only.

Keep the frozen inventory pin exactly:

    58f7936d1138b24dd6cc988081d74d07683611d6

That pin is the 2,138-artifact catalog checkout. Retargeting it would silently move the published inventory.

Add under "Default-branch boundary", after the existing pin bullets:

- Congruence erratum landed on Math- default at `d8f55054270532f2f95ae7cc7f5c613643d47e13` (PR64).
- Pointer refresh, cap pairing identities, and P15 ASCII note landed at `5ed3b455b9a192487cedb32ad5dd8f2b90fbc1c1` (PR85).
- Live proof index: https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md
- None of those merges accepts Theorem A.

Also, in the D6 ACCEPT row, typeset `rho*=1/(3-log(3*e-2))` and keep a parenthetical that the source token is `3e-2`.

Do not flip any ACCEPT/AMEND cell. Do not close #63.
