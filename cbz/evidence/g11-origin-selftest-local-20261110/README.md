# G11 origin repair: local adverse and mutation checks

These observations use Python 3.12.14 in the restored executor. The exact
original hosted false-kill evidence is in the sibling
`g11-origin-adverse-20261110` directory. The G11 subject census is unchanged.

- Against provisional harness blob `e4ee273279908d8b6952a8de61db7f0d7d2f986b`,
  a subject pre-created a forged origin report. The dedicated regression
  failed in both modes; the original logs are retained as
  `forged-report-*-red.log`. A completion marker and separate report validation
  were then added. This is an accidental/mutational fault boundary, not a
  security sandbox against arbitrary hostile Python in the same process.
- The original 263 G11 census unit methods passed in each mode with no skips in
  `g11-base-*-20261110.log`. Nine origin methods passed in each mode with no
  skips in `origin-*.log`.
- `g11-origin-mut-*-20261110.json` records the first local origin mutation
  self-test failure: three of four were credited, while a mutant produced three
  subtest failures and the grader correctly refused to call it one kill.
  The selected adverse test was then narrowed to one real subject assertion;
  no failure was removed. The v2 outputs record four of four killed in each
  mode. After the forged-report regression, the v3 outputs record five of five
  authorized selected-body kills per mode, including the report-completion
  mutant. Every result records a clean baseline, exact test and assertion line,
  and source hashes before/after.

These observations do not claim a final G11 gate, a federation-wide CBZ pass,
or the current hosted 328-mutant result. The historical runs keep their original
outcomes and do not establish origin soundness for the old grader.
