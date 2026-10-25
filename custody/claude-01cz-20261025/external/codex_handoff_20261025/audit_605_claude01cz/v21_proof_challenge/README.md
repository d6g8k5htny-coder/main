# Claude01Cz v21 proof challenge: evidence (main#605)

These scripts are independent symbolic checks. They were written from the definitions stated in the three v21 proof notes, with none of the package's own checkers read or run. They ran in an isolated virtual environment with SymPy 1.14.0 installed from PyPI.

- `independent_algebra.py` / `.out` covers two notes:
  - **SEPARATED_COMPACT_COVER §1:** pin form, generality, witness equations, rank determinant and both compatibility conditions.
  - **ANTIPODAL_UNFOLDING:** the row limits (8), the expansion of E in (9)–(11), q13 (12), β_g (13), the targets (19)–(20) and the pin Jacobian factor in (21). All rows are computed on a general degree-5 field.
- `independent_hermite.py` / `.out` checks HERMITE_POLYNOMIAL_FLOOR (4)–(5) and the degree bound, on one generic and one fully collinear rational configuration.

The members reviewed were verified against these SHA-256 values before reading:
ANTIPODAL_UNFOLDING `3faa6fc6…9fccb`, HERMITE_POLYNOMIAL_FLOOR `c4493b19…b482d`, SEPARATED_COMPACT_COVER `b5aef6e1…fd36`, OPEN_OBLIGATIONS `0edc0d06…ac352` and README `1d9d905e…118b`.

Scientific effect NONE. The result and findings are on main#605.
