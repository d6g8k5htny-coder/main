# Q0 Composite Migration Audit — 2026-07-30

Author/executor: OpenAI GPT-5.6 / same organizational line as the GP verifier work  
Authority: technical reconstruction and noncanonical candidate preparation only  
Canonical impact: none  
Installation impact: none — the active `q0_machine.json` and `q0_verify.py` were not modified  
Status: technical gates 3–5 evidenced at candidate scope; independent review, status synchronization, promotion, and post-install verification remain open

## 1. Controlling objects and successor check

No machine or verifier successor later than the noncanonical 1.2.1 hardening package was found before this work began.

| Object | Drive ID | Bytes | SHA-256 |
|---|---|---:|---|
| Active `q0_machine.json` | `1PXa-ZqCrICicUUDIy37PafHgdjCOb3cj` | 5,910,703 | `c3a93bd250c49e256467939bf714c18488fa506e87ae10636608fae99fb9cd38` |
| Active `q0_verify.py` | `1FxdPkPmK-WhTm-9hQwyJsuJiscCJoy-7` | 957,321 | `eca1755d25375adfe4ccd3366b0c696ea5214d3267cffe7df9160ed6136d34f4` |
| Law-specific machine 1.2 | `1KWNv1Yz95o42tMocUA_G7qocQrioS52o` | 25,524 | `65af52096aa40a4ab49f0a7b0fca6d29876aaafa153139da8de351ab3b632219` |
| Verifier 1.2 predecessor | `1COFvunwQWUWxET6AB0j0kYd-SfUO0BSU` | 11,496 | `10ac7b82923c79d491923a9c4a840797004056326b02cd48aa2e40c51ed4d2a3` |
| Verifier 1.2.1 reconstructed source | capsule `1nkKiFCt5KDSuSzlC01CBsO8TD6N_lsUCQMGywhCCYUM` | 25,843 | `5202c5fa33b1f08d502220943443573cc8c8e3a31682e6f3079443647fe0f6b9` |

The active machine is schema `q0-consolidated/1.0`; it is a complete archive carrier, not merely a theorem-root graph. Its payload contains:

- 9 live roots and 4 named historical roots;
- 227 JSON artifacts;
- 5 CSV artifacts;
- 6 Base64-encoded PNG artifacts;
- 1 text artifact;
- 488 manifest entries;
- 1 legacy fixture.

The raw active file contains one non-standard JSON constant:

`json_artifacts/C104_COUPLED_PERSISTENCE.json/pairs/128_256/fits/0.1/alpha_hat = NaN`.

The exact raw identity is preserved above. The strict-JSON composite represents that one value as `null` and records the path and replacement.

## 2. Defects found in the 1.2.1 replacement path

### M1 — incomplete migration coverage

The 1.2 machine mapped 5 of the 13 named active/historical roots. It omitted:

- `Q0_CUBIC_RATE_FULL_GAUSSIAN_PROGRAM_C102`;
- `Q0_LIMIT_FULL_GAUSSIAN_PROGRAM_C102`;
- `Q0_B_PROJECT_CLOSE_C104`;
- `Q0_PORTFOLIO_CLOSE_C108`;
- `RATE_PROGRAM_GRADE_C092`;
- `Q0_LIMIT_C092`;
- `RATE_PROGRAM_GRADE_C094_4p35`;
- `UPPER_RATE_ONLY_C094`.

Replacing the active machine by the 25,524-byte 1.2 object would also discard the legacy artifact payload needed by the active verifier.

### M2 — backward-compatibility failure

The active verifier passes on the active machine:

- kernel 2.0: 36/36;
- verifier v4: pass;
- verifier v5: 22/22.

With the 1.2 machine substituted for the active machine, the active verifier falls to 34/36. The failing cases are `legacy_schema_rejected` and `migrated_registry_classified`, both through missing legacy fixture files.

Running verifier 1.2.1 directly on the active machine raises an uncaught `KeyError: 'claims'` after schema verification, because the driver enters law-specific mutation tests on an incompatible parsed object.

### M3 — strict-loader gap

Verifier 1.2.1 rejects duplicate keys but still accepts Python's non-standard `NaN`/`Infinity` constants. The active machine therefore reaches semantic verification despite not being RFC-8259 JSON.

## 3. Composite candidate

The noncanonical composite candidate:

- keeps the 1.2 law-specific claims, objects, and five proposed roots;
- preserves all active legacy payload fields required by `q0_verify.py`;
- maps all 13 named legacy roots;
- maps the eight previously omitted roots only to explicit `SUPERSEDED` provenance claims;
- does not convert any omitted historical or governance root into theorem proof;
- uses strict JSON and normalizes only the single recorded `NaN` to `null`;
- records the exact active, active-verifier, and 1.2-predecessor identities inside the object.

Candidate identity:

Audit record Drive ID: `1kc_7XBNjwv3WsXiNPI7KKyYwhW30cREp`.

| Artifact | Drive ID | Bytes | SHA-256 |
|---|---|---:|---|
| `q0_machine_composite_candidate.json` | `18fpjSh8p21B1ZyQwGgAkXQBFOE9HVl3y` | 7,176,671 | `3e66e41b316244d291a0b660c610f61f43d3f3a6c4f55e452a413bb27932a427` |
| `q0_verifier_composite_candidate.py` | `10N2QU9BlVztDmrHUUOYv2OkzdfqcYQwA` | 27,110 | `9045924d2927a4015c47ccdd3405208e99d412ad2c44ae75bc2eab2d817074b5` |
| Composite verification report | `1it6jzWemyt8-u7gIpJib6OSy4CPtlcUd` | 8,834 | `f15b7e58cf3026ebaf35091d460e00c5f94a48fc09bb38a0f2df648532b48705` |
| Active-byte rejection report | `1qQ6XJVOLIehejlg-1pY0tN2qyb4l3l_G` | 1,665 | `ca1902d7144b962eca4594be21d111092e62db21fb3804a794b8b0525a15e784` |

## 4. Verification results

Against the exact composite candidate:

- structural verification: valid, zero errors;
- predecessor negative tests: 12/12;
- CL mutation tests: 9/9;
- all five proposed roots: nonpromotable;
- named-root migration coverage: 13/13;
- legacy kernel: 36/36;
- legacy verifier v4: pass;
- legacy verifier v5: 22/22.

The hardened composite verifier additionally:

- rejects the raw active machine deterministically with `INPUT_REJECTED:NONSTANDARD_JSON_CONSTANT:NaN`;
- rejects a strict but incompatible `q0-consolidated/1.0` fixture without crashing;
- does not run schema-specific mutation tests after a structural rejection.

These are same-line technical results. They are not an organizationally independent review and do not authorize installation.

## 5. Promotion-gate disposition

| Gate | Disposition |
|---|---|
| Recover exact active machine and verifier bytes | PASS at same-line technical scope |
| Complete migration report for every named active/historical root | PASS in the composite candidate, 13/13 |
| Preserve existing Q0 manifests and verification scripts | PASS for the registered legacy selftests on composite bytes |
| Synchronize machine status with later mathematics | OPEN — the 1.2 theorem statuses predate the terminal side-24 Theorem B review chain and must not be silently upgraded |
| Proposed Q0 ledger append | PREPARED below |
| Independent review of the exact composite bytes and verifier | OPEN |
| Canonical promotion/install decision | OPEN |
| New immutable Drive identities and round-trip hashes | PASS — all five uploaded objects were downloaded from Drive and matched their pre-upload byte counts and SHA-256 digests exactly |
| Post-install verifier run | OPEN; installation has not occurred |

## 6. Proposed Q0_LEDGER append — do not apply as a promotion entry

> 2026-07-30 — Q0 composite migration candidate prepared. Active machine `1PXa-ZqCrICicUUDIy37PafHgdjCOb3cj`, 5,910,703 bytes, SHA-256 `c3a93bd250c49e256467939bf714c18488fa506e87ae10636608fae99fb9cd38`, and active verifier `1FxdPkPmK-WhTm-9hQwyJsuJiscCJoy-7`, 957,321 bytes, SHA-256 `eca1755d25375adfe4ccd3366b0c696ea5214d3267cffe7df9160ed6136d34f4`, remain unchanged. Candidate machine `18fpjSh8p21B1ZyQwGgAkXQBFOE9HVl3y`, SHA-256 `3e66e41b316244d291a0b660c610f61f43d3f3a6c4f55e452a413bb27932a427`, preserves the complete legacy payload, maps all 13 named roots, and passes the law-specific 12/12 + 9/9 battery and the legacy 36/36 + v4 + v5 batteries. Eight previously omitted roots map only to nonterminal provenance claims. One legacy `NaN` is normalized to `null`, with the raw active identity preserved. Candidate verifier `10N2QU9BlVztDmrHUUOYv2OkzdfqcYQwA`, SHA-256 `9045924d2927a4015c47ccdd3405208e99d412ad2c44ae75bc2eab2d817074b5`, rejects nonstandard constants and incompatible schemas deterministically. Machine report `1it6jzWemyt8-u7gIpJib6OSy4CPtlcUd` and active-byte rejection report `1qQ6XJVOLIehejlg-1pY0tN2qyb4l3l_G` were round-trip verified. No active replacement, theorem-status upgrade, canonical promotion, or release is authorized. Independent exact-byte review and current-status synchronization remain required.

## 7. Remaining decisive work

1. A distinct-family or qualified human reviewer must reproduce both candidate hashes and rerun all batteries.
2. The current fixed-side-24 Theorem B closure must be represented by new source-addressed machine claims, not by upgrading stale 1.2 claim labels.
3. P0.1 and P0.2 statuses must be synchronized only from their newest controlling exact objects.
4. Only after review and status synchronization should a promotion proposal identify the exact installed Drive IDs and require a post-install rerun.

No active or canonical object was changed by this audit.
