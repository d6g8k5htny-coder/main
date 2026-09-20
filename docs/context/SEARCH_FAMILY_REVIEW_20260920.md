# Search and Gaussian-family review — 2026-09-20

Root performed read-only code/source review and the bounded checks below while the existing repository writer implements this pass. This is a same-provider review, with zero organizational independence credit and no scientific authority. It does not replace the writer's full validation or final Drive readbacks.

## Scope and identity

The derived search view returns the preferred DS3 hash `bd3074fd900fc80bebdd430e34ae953f0a78685fff672758e7b927cc1e6d9421` as both live-file record `14FIaDPHfbTTGQWzxZi8xKAzlKub5Iud8` and its indexed archive occurrence. The distinct 7,201-byte superseded DS3 remains metadata-only. The RN5 hash returns its live record and archive occurrence. The reconstruction hash `859e1963b1b7c84f2c1f14b4b163fb5575ad41c5394d27c51c0d1ceff70d2dc3` returns all four archive-table occurrences; these are locations of the author-side reconstruction, not evidence that the missing historical Claude module was recovered.

At review time the view held 4,934 file/folder records: 4,928 reconciled objects and six explicitly dated delivery records. Text coverage reported 3,140 extracted texts, 1,012 scope exclusions, 461 records without stored payloads, 301 unsupported formats, and 20 objects beyond the five-million-byte text extraction limit. Archive members were metadata-searchable only. Search coverage must remain separate from transfer coverage and whole-account completeness.

## Algebra checks

For independent centered X,Y with variance one and Z with variance s in [0,1], set D=XY-Z². Expanding directly using one-dimensional Gaussian moments gives E[D²]=1+3s² and E[D⁴]=9+18s²+105s⁴. A read-only Python check of the new family engine returned caps 4 and 132, equal to the endpoint maxima. Midpoint-only values were 7/4 and 321/16, strictly too small for the family. This is an analytic negative control, not a sampled uniformity claim.

For the all-ones 3×3 covariance (PSD, rank one, and failing diagonal dominance), affine intercept (1/2,0,1) and slope (1,2,3), both second/fourth polynomial coefficients and mark-interval caps recovered the existing exact engine without widening. A negative-definite singleton was refused. A diagonal interval box containing negative and nonnegative variances returned PSD_MEMBERS_ONLY.

Code review found that symmetric covariance entries represent the same uncertain scalar, that the sufficient diagonal-dominance criterion is not treated as necessary, and that failed midpoint validation does not establish emptiness. Bernstein recursion takes the hull of both children. The existing conditioned-law and source-byte checks are reused. These are appropriate distinctions for this candidate.

The primary-source context was checked directly against Mamis, https://arxiv.org/html/2202.00189v6 (multivariate Stein identity), and Rump, https://www.tuhh.de/ti3/rump/intlab/ActaNumerica2010.pdf (interval inclusion and dependency). The generic implementation does not establish novelty over those methods or RN5's Arb spatial calculations.

## Reviewed bytes

- research/rn/gaussian_families.py: ecd1958dd9e02f829e93394d362310e3df0c0267c324f62c9fac7c6ecd2522b6
- engine/operations/rn_family_applicability.py: 3a55870d2129a3b41ba909bf3b2e7a0e288a10b87598b8084ca711d970aef238
- tools/drive_search.py: ecf39fe570970551cd62a1bddd51dbe30fb0ad0677b1845c469953a6aa3b13ca
- tools/drive_index.py: e6302e9640587d4749170550128892f6688746a634b150361832e60a52f17bf1

Later edits require their own validation; this record does not automatically cover different bytes. At this review point, complete tests, the Drive-side navigator, publication, final artifact readbacks and claim release remained with the writer. No proof status, utility, held-out evaluation or field-law certification is asserted.
