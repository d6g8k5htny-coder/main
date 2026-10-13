`lake build ResearchFormalCoreR1` in a copy of Math- `formal/` started at 17:24:45Z on 2026-10-10 and ended with
`Build completed successfully (8939 jobs)` at 20:42:15Z, i.e. **3 h 17 min 30 s** of wall clock on four cores that
other agents were competing for during most of that time (this session's 5,603-line Lake log, not part of any
repository; the intermediate readings were 539 of 8,529 module oleans after 7 min and 1,389 after 23 min). Re-counted after the
build, at 22:06Z: `find .lake/packages/mathlib/.lake -name '*.olean' | wc -l` gives 8,548 (the 8,529
`Mathlib/**/*.lean` modules of this revision plus generated modules), and `du -sh .lake` gives 7.5 GB after the
Lake `*.setup.json` files (7.4 GB) and the generated C sources (0.4 GB) were deleted; the tree peaked at 15 GB at
completion (1.8 GB of oleans, 0.3 GB of ileans), so a from-source build needs at least 16 GB free and the cache
route about 6 GB. Elaborating the five statement-only modules against the built Mathlib
(`lake env lean -DwarningAsError=true`) took 41 s for `Field` (cold), 10 s `Lifetime`, 9 s `Side24`, 10 s `RN`,
9 s `P15`, each exiting 0 with zero warnings; the injected-type-error control exits 1