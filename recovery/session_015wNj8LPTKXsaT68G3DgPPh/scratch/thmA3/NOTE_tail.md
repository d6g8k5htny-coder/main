## 10. Controls (floating point; not part of the certificate)

@@MC@@

## 11. What this does not do

- **`d = 3` only.** The method extends to every `d` with `m = d - 1` transverse directions: `GOE_m` eigenvalue coordinates, and
  `m`-variable blocks for `M_3` and `M_4`. This packet certifies `d = 3` only.
- **The band only.** No `k < 1/2`. In particular there is no three-dimensional SIDE24 point: the [CAP] SIDE24 law is planar.
- **No lower bound for the constant of (1.1).** Every bound here is for the cap route (7.8). As in [G3] Corollary 2, the
  cap criterion overstates the true failure rate by the planar factor of Math-#203 (about `2.4 x 10^6` at `(0, 2)`).
- **Other C8 constants.** `c_{B,K}` and `c_{d,L}` are not addressed (see Math-#205 and Math-#209 for `c_{B,K}` in `d = 3, 4`).
- **Not a review.** This is not a review of [LP]'s proof or of [CAP]. The [LP] §8 implication and [CAP]'s deterministic
  theorem are used as statements, only for the consequence on `1 - p_r`.
- **Precision.** The cap-route constants for `r_* >= 1/256` are valid but far from sharp, because the rare branches at small
  `k` dominate there. A finer sweep or sign-aware weight bounds could improve them, but they cannot become informative,
  since `c_G^(3) (1/256)^3 > 1`.
- **Status.** C8 stays OPEN.
- **Independence.** Same GitHub account as every lane; zero organizational-independence credit. The author will not merge.
