# C096 Q0 Registry Migration

**Source:** `q0 registry.json`  
**Target:** `q0_registry_v2_0.json`  
**Source bytes:** untouched

## Authority decision

The uploaded registry is a **Gate Kernel demonstration**, not the source-of-truth theorem registry for the frozen q0 Rate Program. Its coefficient, domain, generic region claims, and use of `R0` do not match the canonical program objects.

## Critical name collision

The legacy registry's `R0` means an unbounded remainder in a toy pairing-failure expansion. In the mathematical Q0 program, `R0` names the Gaussian-Sard/Morse-Smale condition. The migrated object is therefore:

```text
PAIRING_REMAINDER_CONDITION
```

and never bare `R0`.

## Finding table

| ID | Finding | Migration repair |
|---|---|---|
| REG-01 | The legacy ID R0 denotes an unbounded expansion remainder, not the canonical Q0 Gaussian-Sard/Morse-Smale condition R0. | Renamed PAIRING_REMAINDER_CONDITION. |
| REG-02 | The coefficient 0.4127, domain r∈(0,1], b∈[0,3], and three generic region bounds do not match the frozen Q0 theorem. | Registry metadata labels the graph as a legacy schema demonstration. |
| REG-03 | inner_bound, far_bound, and bdry_bound are labeled Proven without proof/evidence artifacts or numerical statements. | Migrated as explicit open/Plausible conditions rather than nonconditional theorem support. |
| REG-04 | C_const says 'closed form + numerical value' but provides only 0.4127 and an agreement-based tag with no independent-error certificate. | Migrated as an open Measured condition. |
| REG-05 | pinned_input is disconnected, while cm_node is attached to T_main despite no Gaussian-pinned support edge requiring the bridge. | Both are retained as disconnected fixtures; the bridge no longer launders into the root. |
| REG-06 | The union witness says the regions are disjoint, although a union bound requires neither disjointness nor independence. | Typed as an exact UNION_BOUND witness. |
| REG-07 | Endpoint truth is stored as Boolean values rather than as recomputable endpoint certificates. | ENDPOINT-FIDELITY remains an active open gate. |
| REG-08 | T_main prints PASS at Proven-Modulo under v1.2 while R0 is open and C_const is provisional. | The v2.0 root report separates archive validity, candidate shell validity, conditional promotion, and unconditional promotion. |
| REG-09 | Human labels such as C#1 are not cryptographic content hashes. | Every target claim and edge uses canonical SHA-256. |
| REG-10 | No coefficient assembly or full-domain uniformity certificate is present. | COEFFICIENT-ASSEMBLY-SOURCE and UNIFORM-COEFFICIENT-CERTIFICATE remain open blockers. |

## Root disposition

- structural archive admission: **True**
- candidate admission: **False**
- promoted admission: **False**

The migrated root has five explicit conditions and remains blocked by:

- `DOMAIN-INFIMUM`
- `COEFFICIENT-ASSEMBLY-SOURCE`
- `UNIFORM-COEFFICIENT-CERTIFICATE`
- `ENDPOINT-FIDELITY`

This is a successful migration outcome. It preserves the demonstration without presenting it as completed mathematics.