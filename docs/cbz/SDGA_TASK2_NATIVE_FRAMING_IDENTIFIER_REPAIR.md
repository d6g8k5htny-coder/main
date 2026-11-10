# SDGA native binary-read family: author fix round 2

Date: 2026-11-10 UTC. Identifier-only source correction.

The actual hosted run 42749115104, attempt 1, job 129608245848 rejected candidate a46ec37234825ead870b8166ef28d8fbb4997e6f before controls. Primary logs contain the parser error at Component.lean:400:7 for reserved token `prefix`, its dependent parse errors at 400:14/409:32/410:32, and original 477-byte compiler stdout SHA-256 7753377c06b853292d82387f5a220fa9ad79ce4c4c75cf513ab23d0c7cb0037c. Both modes have six bootstrap method failures and all 39 subcases not_run; no inventory or OBJECT_READ result was produced. Pure dependency tests passed 12/12 per mode; this does not credit the native guard.

Intended file: tools/sdga/Component.lean.
Corrected UTF-8 blob: 336a35a3b916196e4f5d90d7c2c55be63d073c90.

Exactly five lexical uses of the local identifier `prefix` (declaration and four references) were renamed to `headerBytes`. Exact UTF-8 readback matched the authored source. Replacing those five `headerBytes` tokens back with `prefix` reconstructs a46ec37234825ead870b8166ef28d8fbb4997e6f byte-for-byte.

No guard behavior, model, framing criterion, test, fixture, assertion or other source is changed. Prior bounded format rationale/review 03ff8094d546d770c3a3c51778c6d51c80556806 still applies, subject to its stated limits. No ref, tree, commit, PR, comment, filesystem/local execution or rerun occurred here.

Fresh native compilation and unchanged six-method/39-subcase execution remain pending; no compilation or runtime fix is claimed. Scientific effect NONE; qualification/classification BLOCKED; execution_authenticated=false.
