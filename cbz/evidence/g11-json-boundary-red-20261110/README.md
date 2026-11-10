# JSON no-op and malformed-workflow checks: observed red

Hosted run 42762787724, job 129652192212, exercised debug commit 4ef18b1f88b78a01a6c9efc9179024caa6f97d62. Seven workflow-format methods ran in each Python 3.11.18 mode. The five earlier workflow-reference controls passed, while a JSON no-op step and a duplicate-key JSON workflow were silently accepted by the provisional census; those two checks failed in each mode. The original job log and exact test are retained. This is scoped tooling evidence, not a final G11 result.
