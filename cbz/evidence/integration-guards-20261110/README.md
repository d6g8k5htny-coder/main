# Scoped Main #801 pre-integration observation

`main-801-before.json` records sequential public ref and current check observations at the stated time. It is a guard snapshot, not an atomic federation cut, a successful receipt recovery or final G0–G11 evidence. The exact tested merge tree and parents are recorded. The PR-only `recover` job was intentionally skipped; it is not credited as a recovery execution. The earlier full discovery had explicit skips.

The `main-802-before.json` record contains the separate fresh pre-merge reads for #802 at the recorded time. It reports all 26 observed checks, including the PR-only build, deploy and recovery skips; it is not an atomic federation snapshot or live recovery result.
