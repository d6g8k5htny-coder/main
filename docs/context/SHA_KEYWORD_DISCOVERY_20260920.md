# Keyword and SHA discovery — 2026-09-20

This is an exposed same-provider retrieval and source-context review. It
records observed search behavior, source identities and mathematical scope.
It does not change scientific status or establish organizational independence.

## Observed retrieval failures and useful controls

1. Live keyword query `RN5_NEAR_MOMENT_REPAIR` returned 13 metadata results.
   Reading copies occupied the first three positions; the original source was
   fourth. An exact-name filter for `RN5_NEAR_MOMENT_REPAIR.md` selected the
   original file `1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5`.
2. A full-hash query for RN5
   `ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`
   returned 12 references, including the register, reading copies and the
   recent proof/recon memo. RN5's own source file was absent from those hits.
   A hash *mentioned in text* is not the containing file's byte identity.
3. Both legacy and explicit-document exact-name searches for `rnu_ds3.py`
   returned zero hits. Direct metadata for `14FIaDPHfbTTGQWzxZi8xKAzlKub5Iud8`
   nevertheless returned that exact name, MIME `text/x-python`, 9,704 bytes,
   and parent `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG`.
4. Direct `list_folder` of that parent, with cap 1,000, returned 13 entries
   including the preferred Python file, the superseded Python file, receipts
   and other code. A document-filtered parent search returned only five text
   reports. These search modes are not interchangeable inventories.
5. The preferred DS3 hash is
   `bd3074fd900fc80bebdd430e34ae953f0a78685fff672758e7b927cc1e6d9421`.
   The old `tools/drive_index.py sha` returned the original inventory file
   only. The already-present Archive_Members table also contains the same
   payload as `MATH-20260917-b9c2/intake/rnu_ds3.py` in carrier
   `1TJr1awRjB0D9niq1Jyi6XGnJKuvW9Wmx` (carrier SHA
   `3b16c3712ed9bfe69f82e0fd288a1af72a7b7a5c4de8a7ece23935eb270bdbcb`).
   `cmd_sha` suppresses archive matches whenever an inventory match exists.
   Both occurrences should be returned without conflating carrier and member.
6. The 7,201-byte version is a distinct historical source:
   `1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21`, now titled
   `rnu_ds3_scalar_SUPERSEDED.py`, recorded payload SHA
   `c3d5adc8b88788b7eed43bd68efe9381f7076775a7ec60817a20a8267d9116da`.
   Its inventory context is `QUARANTINE_OR_HISTORY`. Only its metadata was
   inspected in this pass; it must not become current mathematical evidence.
7. `rnu_ds3` hit the requested legacy cap of 30, without a pagination token.
   Treat that result as potentially partial. Queries for
   `CL_ANTHROPIC_BUNDLE_2026-09-17_v5` and broader `CL_ANTHROPIC_BUNDLE` were
   empty. Given the demonstrated Python-file false negative, these empty
   searches do not establish that an archive is absent from all Drive storage.

## Sources read for mathematical context

- [RN5 corrected near-moment source](https://drive.google.com/file/d/1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5/view),
  known exact source SHA `ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`.
- [DS3 receipt drift](https://drive.google.com/file/d/1ZV5nA9IASm9xAAQyp211rKqUR0XpI9E6/view),
  6,122-byte report; same-title/numbered copy also appears as
  `1tkSPeg9k_xJWQvXACD_FZvOoacQfktEQ`.
- [CL-RNU-002 third derivatives](https://drive.google.com/file/d/13w34AGF17N-gbQVI9GFE0qfmyrMJC0Bt/view),
  4,062-byte proposed lane progress report, no canonical impact.
- [CL-RNU-003 Piece 1/2 report](https://drive.google.com/file/d/1aCa-QG9CSrNUB9SUFKISifghSf-41fRy/view),
  7,407-byte report whose source SHA RN5 records as
  `59b8f002f1b1e81e7829416baf16ef1d84b1326d4b3b4604b833e5a55a9d6d53`.
- [Whitened env_form smoke report](https://drive.google.com/file/d/1pnnSC2E-FY5tjLa4z60UOSjOGdiKcAIe/view),
  3,564-byte report identifying the preferred DS3 hash and explicitly marking
  its orders 2–4 calculation as a proxy.

The fresh report reads used readable-text responses. They do not by themselves
re-verify source bytes. Hashes above are attributed to the existing inventory,
archive tables or pinned RN5 source unless a later custody record says otherwise.

## Mathematical implications

The older CL-RNU-003 report's third determinant factor uses the fourth moment
under a square root. RN5 later corrects this to the second moment. Its older
integrand-certification wording and forecast therefore cannot be carried
forward as an accepted result. High-precision derivative scans and smoke
comparisons also remain numerical observations, not interval certificates.

RN5 describes affine conditional mean in the centered mark
`t = v - (b - r^3/12)` with mark-independent conditional covariance. It groups
equal spatial monomials before interval substitution to retain cancellation,
and uses separate marginal Hessian whitening; the three Hessian blocks need
not be mutually independent. For its actual spatial computation it requires
positive interval Cholesky pivots for the conditioning block and each
conditional marginal. Its 65 point-law and 10 spatial-box certificates are
not a complete annulus cover.

The generic exact affine-moment implementation from the preceding pass still
has a useful bounded extension: enclose uncertain affine means and covariance
parameters over complete mark intervals. The proof should explicitly separate
polynomial interval inclusion from the question of whether every matrix in a
covariance box is PSD. A conditional enclosure for every PSD member of a box
must not be presented as certifying that all box members are Gaussian laws or
identifying the actual RN field law. Midpoint substitutions, dropped covariance
terms and one-child subdivision are useful negative controls. Exact singleton
input families should recover the preceding fixed-law result.

RN5 pins `closure_round2/rn_field.py` at payload SHA
`d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8`.
The archive-member index has four occurrences: in RN5 repair, RN3 proof,
nested RN3 prior-close intake, and the prior CLOSE bundle. Each is a 16,701-byte
member. These occurrences must retain distinct carrier identities. The source
RN5 proof names another scientific carrier
`1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`, SHA
`a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`;
source-named and reading-copy/archive carriers are not interchangeable without
the recorded member identity comparison.

## Delivery requirements suggested by these observations

Publish a usable additive Drive navigation/search artifact with exact source
links, digest kinds, query examples and the above negative-search warning.
Include current reconciled file records and archive occurrences, not just the
old 4,456-item inventory. Keep all same-hash occurrences and all different-hash
versions. Apply personal/vault/quarantine scope filters before content search;
metadata visibility never grants content authority. Do not rename, move,
delete or rewrite source files merely to improve search rankings.
