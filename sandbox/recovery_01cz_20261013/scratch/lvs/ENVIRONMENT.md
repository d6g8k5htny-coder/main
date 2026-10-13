# Measured environment facts for the Lean verification specification (recorded 2026-10-10, this sandbox)

Sandbox: Ubuntu 24.04.4 LTS, x86_64, 4 CPUs, 15 GB RAM, ~17 GB writable disk; outbound HTTPS via an egress proxy with an allow-list.

## Toolchain actually installed and used here
- elan 4.1.2 (58e8d545e 2025-05-26), from the GitHub release tarball `elan-x86_64-unknown-linux-gnu.tar.gz` (sha256 f81c2e48c1588d4612cd2c8851947898a45ac8d72748a07dff3a5694f1cf589b as downloaded here). CI uses elan v4.2.4 via `https://raw.githubusercontent.com/leanprover/elan/v4.2.4/elan-init.sh` with sha256 `a620ff1641616222c8d37c54845492004bb84d6877cdbc944dd65c1aa685bf53` (pinned in main `.github/workflows/formal-verification.yml`).
- Lean release tarball `https://github.com/leanprover/lean4/releases/download/v4.34.1/lean-4.34.1-linux.tar.zst`, sha256 `47bf4bbd78f70c2e9670598ab7124d92b6efb7330ff33e5fbb4030f6fd72e4e4` (observed here; not an official checksum publication), unpacked size 3.0 GB; linked with `elan toolchain link leanprover/lean4:v4.34.1 <dir>` and set default.
- `lean --version` → `Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release)`
- `lake --version` → `Lake version 5.0.0-src+5045d00 (Lean version 4.34.1)`
- Binaries present: lean, lake, leanchecker, leanc, clang, cadical (used by `bv_decide`, not by this project).
- Mathlib rev `d13f23b723b8a846827a245b89c10fc7d3f11612` cloned from GitHub; dependency lock identical to Math- `formal/lake-manifest.json` (plausible 118aa17ee84656b8bd727fef7c458ee8c833385c, LeanSearchClient ddf04cf3949fa556442341e87d47f9f6e6074707, importGraph e928b72544873815af278d38681b31c0293588e3, proofwidgets 106ff4fafc74ef4ac99d81dbf3ab399118f497a5, aesop 355695d523e41d0554926416cba2a2b3544fbbc9, Qq 6a489d9af5d0c47e5b259e2e8bcdfc1811b5a259, batteries f2effa3d803fda822b1f97b806c47cf2adfbcbc2, Cli e92c9f15fdfacc8536f31cfb3b7ad26c3c8cd204, mathlib d13f23b723b8a846827a245b89c10fc7d3f11612).

## Network facts (this sandbox)
Works: `https://github.com/<owner>/<repo>/releases/download/...` (302 → release-assets.githubusercontent.com), `git clone https://github.com/...`, raw.githubusercontent.com (CI path), PyPI, Ubuntu apt mirrors.
Blocked by the egress policy (CONNECT refused): cache.mathlib.org (so `lake exe cache get` cannot fetch the Mathlib oleans), mathlib4.lean-lang.org, lakecache.blob.core.windows.net, release.lean-lang.org (elan's toolchain download host), reservoir.lean-lang.org (Lake's package index). codeload.github.com and GitHub HTML pages return 403.
Consequence: Mathlib must be built from source here (full `lake build` of the import closure, hours on 4 cores) unless the environment's network allow-list is extended with: cache.mathlib.org, release.lean-lang.org, reservoir.lean-lang.org (and lakecache.blob.core.windows.net if `lake` is to use its remote cache). GitHub-hosted CI is unaffected (lean-action with `use-mathlib-cache: 'true'` downloads the cache in a few minutes).

## Timings measured here
- main `formal/` (dependency-free core-Lean package, 31 targets): `python3 tools/formal_gate_check.py --run-lean` ≈ 12 s end to end (build + leanchecker + axiom audit + 10 negative controls + receipt).
- Mathlib from source (`lake build ResearchFormalCoreR1` in a copy of Math- `formal/`, which imports all of Mathlib): started 17:24:45Z; 539 of 8529 module oleans after 7 min, 1389 after 23 min; `.lake` at 2.8 GB at that point. (Final figures appended when the build finishes.)

## Pinned Layer 0 sources used by the new Lean content
Math- commit `760340e921ac4ceda296b8118da936f1133e956e` (the LANDING_CLAIMS audited default commit), Math- commit `6c020d6a72936632f2055122e71a7818a4ff497e` (tip on 2026-10-09 for packets landed after 760340e9), Math- commit `c39164254ad1529ed49543743598d5dcb24bab98` (lifetime parent imports). Identities (bytes, sha256, blob) in `specifications/lean_statements_20261010/sources/SOURCES.json` (Math-) and `formal/manifest.json` (main).

## Final Mathlib source-build figures (measured here)
- `lake build ResearchFormalCoreR1` in a copy of Math- `formal/` (imports all of Mathlib): started 17:24:45Z, `Build completed successfully (8939 jobs)` at 20:42:15Z → 3 h 17 min 30 s on 4 CPUs / 15 GB RAM, with other agents competing for CPU for most of that time. 8,548 Mathlib oleans (8,529 `Mathlib/**/*.lean` modules at this revision plus generated modules).
- Disk: `.lake` reached 15 GB at completion (7.4 GB of Lake `*.setup.json` files, 1.8 GB oleans, 0.3 GB ileans, 0.4 GB generated C). After deleting the setup JSON and C files (not needed to import or elaborate) it is 7.4 GB. Budget at least 16 GB free for a from-source build; the Mathlib cache route needs about 6 GB.
- Elaboration of the five statement-only modules (`lake env lean -DwarningAsError=true` against the built Mathlib): Field 41 s (cold), Lifetime 10 s, Side24 9 s, RN 10 s, P15 9 s; all exit 0 with zero warnings. Negative control: an injected type error fails with exit 1.
