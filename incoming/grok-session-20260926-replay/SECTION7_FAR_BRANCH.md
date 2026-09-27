# Parent §7 m=1 far branch

**Object:** GROK-HEAVY-S7-FAR-20260926-v1
**Scientific effect:** NONE. Does not accept Theorem A.
**Source:** `UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (7.5)–(7.6).

## Split

For `m=1`, `lambda=-A_M>0`, `h<=K(J+lambda)`. Depth failure

    lambda <= D r (J+lambda)^2 <= 2 D r J^2 + 2 D r lambda^2

lives in

    {0 < lambda <= 4 D r J^2}  union  {lambda > 1/(4 D r)}.

The near piece integrates like (7.3) and is `O(r^5)` in `E[W 1_near]`.

## Far Markov

On the far set, `4 D r lambda > 1`, so for any positive power

    1_far <= (4 D r lambda)^4.

Hence

    E[W/r^2 1_far] <= (4 D r)^4 E[(W/r^2) lambda^4].

For `m=1`, `W/r^2 <= (h^2/4) lambda(lambda + 3 r h / 2)` is a polynomial in `(J,lambda)`. Against (3.5) and uniform `J` moments the expectation is finite. The displayed bound is therefore `O(r^4)` on `W/r^2`, i.e. `O(r^6)` on `W`.

After dividing by `Z=Theta(r^2)` this is `O(r^4)`, smaller than the near `O(r^3)`.

## What this does not close

The far branch is why the all-dimension statement cannot drop `lambda` large. It does not replace A3, the cap implication, or Morse/§8. Theorem A remains AMEND.
