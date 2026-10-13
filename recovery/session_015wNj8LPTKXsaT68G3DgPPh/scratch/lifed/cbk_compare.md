**What the refinement and the cap route contribute** (`r_* = 1/4096`):
- **Box resolution.** On their own boxes, the per-band bounds of [E] and [E3] give `2325.98` in the plane and `4999.57` in
  `d = 3`. The refined boxes improve these by factors `5.8` and `4.1`. [E] has `28` boxes, the coarsest `k`-box being
  `[1/2, 1]`; [E3] has `39`.
- **A single constant.** Using one sup constant `C(r_*)` of [E] or [E3] times the total weight gives `22467.8` and `38970.6`.
- **The cap-route limit.** As `r_* -> 0`, the cap route can give no less than
  `Gamma_{B,K} = |S^(d-1)| int A_0^ref c_G / (3 k^(5/3))`, with the cap-failure coefficients `c_G` of [G] and [G3]. In
  floating point, `Gamma_2 = 307.5` and `Gamma_3 = 541` (section 9).
  - `C_{B,K}(1/4096)` is therefore within a factor `1.31` of that limit in the plane and `2.23` in `d = 3`.
  - The `d = 3` excess comes from the [E3] box bounds near `k = 1/2`, which are about `2.2 c_G^(3)` there.
- **The cap route itself is loose.** By [G], the cap criterion overstates the actual elder-pairing failure rate by
  `7.4 x 10^4` at `k = 1/2`. `C_{B,K}` bounds the [LP] quantity as [LP] defines it; it is not an estimate of the true
  difference.
- **Above `r_* = 1/512`** the certified per-band bounds of [E] and [E3] grow quickly; their own `C(r_*)` tables show the same
  jump. The useful range is `r_* <= 1/512`.
