# V20-PLANNER-SOURCE-DELIVERY inventory (prep only; nothing posted/branched)
Generated 2026-10-25 15:06 CDT.

## Outer zip
- /tmp/v20/Universal_Law_v20_Conditional_Mathematics.zip, 290395573 bytes
- sha256 71b6b21679ab970c2d876268ab29322373e0444af4ece6d248d5f5cc57e95d89: **VERIFIED**
- 117 file members. Nested zips: `Universal_Law_v20_Conditional_Mathematics/foundation/Universal_Law_v19_Research_Frontier.zip` (v19, **6564 members are listed in file_list.tsv**), `Universal_Law_v20_Conditional_Mathematics/sources/Next_Gate_Literature.zip` (only listed as an outer member, not expanded)
- file_list.tsv columns: scope, path, bytes (uncompressed), sha256. I added a scope column (outer | nested:<zip>) so the two sets can be told apart.

## Requested planner/protocol sources
**I found no dedicated source files for shared plans, effort budgets, work leases, typed failure reports, quarantined proposals, or persistence** in the v20 zip, the nested v19 zip, or the handoff archive. I checked filenames and grepped the contents of every .py/.md/.json file outside vendor/ and toolchains/. Mentions show up only in docs, JSON state, and the related modules listed below.

Protocol/API files that exist:
| path | bytes | sha256 |
|---|---|---|
| `Universal_Law_v20_Conditional_Mathematics/API_V20.md` | 4459 | `65f920e847888c0200c24628305313851170916338ff8aa0d9464665bcf77c19` |
| `Universal_Law_v19_Research_Frontier/engine/RETAINED_PROTOCOL_COMPATIBILITY.md` | 2239 | `d99071d5722c8e8beb977c15acd0c0662abeda8df398c8fadd83672e165c5329` |
| `Universal_Law_v19_Research_Frontier/engine/adapters/worker_protocol.schema.json` | 3277 | `7e2d5a030c577cbad43f6a5951c2d2c1a4c67ce982d62fff39931c5d96dea74e` |
| `Universal_Law_v19_Research_Frontier/engine/adapters/WORKER_PROTOCOL.md` | 3194 | `d87b10f79272534e554ed1e1c0d3fb6569c35646efd3db28f178d93c26cff25c` |
| `Universal_Law_v19_Research_Frontier/engine/AGENT_PROTOCOL.md` | 3825 | `a38c860cb75ee7c5907a4cc30684414be4aff1842f06624e247cf24a0fb9fa9c` |
| `Universal_Law_v19_Research_Frontier/engine/retained_worker_compat.py` | 5392 | `88e59a603ce620983a4b18b55eb886f805a58da9e1ede24e3f0f68835b84ae02` |
| `Universal_Law_v19_Research_Frontier/engine/tests/test_retained_worker_compat.py` | 3955 | `1108b09a28ba24081a2a109563d17e6fd5c2ff7618d750efa859b8560fdb62e5` |
| `Universal_Law_v19_Research_Frontier/engine/workers.py` | 23758 | `a56cf42be8a49b3309e7d9fbde40e88c1562608a40273856346c9bc8e18b20ea` |
| `Universal_Law_v19_Research_Frontier/engine/worker_process.py` | 5148 | `fab28e600e84a0d3c0a684875c1d0d2a6fa091669223e16c8739a95187b3262c` |
| `Universal_Law_v19_Research_Frontier/engine/tests/test_workers_v17.py` | 3950 | `a404d4338330901e161cadfa2eeda0957b75644c9d702d4a03989a28d9e889cf` |
| `Universal_Law_v19_Research_Frontier/API_V19.md` | 4148 | `5bb79e0d291e4f63d3987a916a1572807716605bf09b04486dad24759b01f4e0` |
| `Universal_Law_v19_Research_Frontier/frontier_runtime.py` | 42316 | `f74eede7d9c54abe46185c9219305384567c77d07bd6ac359de4ef49afcbe364` |
| `Universal_Law_v19_Research_Frontier/proof_conductor.py` | 49920 | `9570783ad05d25a2d8cb4caa1dad73b11e4c0d8f5081be5af8d2b3e31deeea4d` |

The last three rows are loosely related: frontier_runtime.py and proof_conductor.py mention "failure report". RETAINED_PROTOCOL_COMPATIBILITY.md and worker_protocol.schema.json are in the **nested v19** zip under engine/. They are not at the v20 top level.

## Handoff parts
/workspace/dylan_audit_phone_chatgpt_20261025/parts/*.part001-004 joined together make Universal_Law_Project_Handoff_2026-10-25.zip (sha256 6ca2a0c1b29585a048f7ea153af2276af464913ea364ccdd370926a1338261b3, 29 files). That is a **different archive**. It contains packages/Universal_Law_v20_Conditional_Mathematics.zip, which has the **same bytes** (sha256 71b6b216…). It also has the v21 supplement, the v24 Research Router, and v20–v24 AuditEvidence zips. None of them contain the planner source files listed above. I joined the parts only into /tmp/handoff.zip, for checking.
