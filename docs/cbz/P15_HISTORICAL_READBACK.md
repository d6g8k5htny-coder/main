# P15 historical artifact readback

This **pending**, isolated candidate checks the downloaded 38096827443 artifact against independently recorded ZIP, receipt, manifest, checked-commit and run identities. It verifies every receipt-bound log, every manifest-bound source file, the complete 59-target receipt inventory and all 28 P15 axiom lists. It preserves the original ZIP and literal elaborated-type log. This is byte/identity consistency for a historical run; it executes no Lean and establishes no informal alignment or current-baseline admission.

Tool: `tools/p15_historical_artifact.py --archive PATH --manifest HISTORICAL/formal/manifest.json --source-root HISTORICAL --output NEW.json`. Its caller must check out main commit `3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c` cleanly and obtain artifact 11686340530. The workflow performs these steps.

Self-verification: `python3.11 -B -S tools/p15_historical_mutations.py --output NEW_DIRECTORY` runs 15 ZIP/source and harness controls, seven named intended-failure source mutants, and one mutation of the harness in both normal and optimized modes. A test setup failure, exception or unrelated error is not a mutant kill. All results are pending until observed on this exact source. Owner none; scope criteria 3, 6, 7 and 9. The canonical integrator must register any adopted tool in GOVERNANCE.md and TOOLING.md after observing its results.
