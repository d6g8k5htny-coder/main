"""Parity probe: Python contract validators vs a stdlib evaluator of the subset of
JSON Schema 2020-12 keywords the four published schemas actually use.

Read-only. Standard library only. Run with:  python3 -I parity_probe.py <meta-framework-clone>
The subset evaluator covers: type, const, enum, required, properties, additionalProperties
(true/false), items, minItems, maxItems, uniqueItems, minLength, minimum, pattern, oneOf.
It is a reading aid for this decision note, not a replacement for a full validator.
"""
from __future__ import annotations
import importlib.util, json, re, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("contract_validation", ROOT / "tools" / "contract_validation.py")
cv = importlib.util.module_from_spec(spec); spec.loader.exec_module(cv)

SCHEMAS = {name: json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text()) for name in
           ("repository_role", "source_ref", "source_manifest", "workspace_snapshot")}
EXAMPLES = {p.name: json.loads(p.read_text()) for p in sorted((ROOT / "examples").glob("*.json"))}

def js_type(v):
    if v is None: return "null"
    if isinstance(v, bool): return "boolean"
    if isinstance(v, int): return "integer"
    if isinstance(v, float): return "integer" if v == int(v) else "number"
    if isinstance(v, str): return "string"
    if isinstance(v, list): return "array"
    if isinstance(v, dict): return "object"
    raise TypeError(type(v))

def ok(schema, v) -> bool:
    if "oneOf" in schema:
        return sum(1 for s in schema["oneOf"] if ok(s, v)) == 1
    if "type" in schema:
        t = schema["type"]; ts = t if isinstance(t, list) else [t]
        jt = js_type(v)
        if jt == "integer":
            if not ("integer" in ts or "number" in ts): return False
        elif jt not in ts: return False
    if "const" in schema and not (v == schema["const"] and type(v) is type(schema["const"])): return False
    if "enum" in schema and v not in schema["enum"]: return False
    if isinstance(v, str):
        if "minLength" in schema and len(v) < schema["minLength"]: return False
        if "pattern" in schema and re.search(schema["pattern"], v) is None: return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in schema and v < schema["minimum"]: return False
    if isinstance(v, dict):
        for r in schema.get("required", []):
            if r not in v: return False
        props = schema.get("properties", {})
        for k, sub in props.items():
            if k in v and not ok(sub, v[k]): return False
        ap = schema.get("additionalProperties", True)
        if ap is False and any(k not in props for k in v): return False
    if isinstance(v, list):
        if "minItems" in schema and len(v) < schema["minItems"]: return False
        if "maxItems" in schema and len(v) > schema["maxItems"]: return False
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in v}) != len(v): return False
        if "items" in schema and not all(ok(schema["items"], x) for x in v): return False
    return True

def py(fn, doc):
    try:
        fn(doc); return "accept"
    except cv.ContractError as e:
        return f"reject ({e})"

VALIDATORS = {"repository_role": cv.validate_repository_role, "source_ref": cv.validate_source_ref,
              "source_manifest": cv.validate_source_manifest, "workspace_snapshot": cv.validate_workspace_snapshot}
EX_FOR = {"repository_role": "repository_role.json", "source_ref": "source_ref.git.json",
          "source_manifest": "source_manifest.json", "workspace_snapshot": "workspace_snapshot.json"}

rows = []
def case(contract, label, doc):
    s = "accept" if ok(SCHEMAS[contract], doc) else "reject"
    p = py(VALIDATORS[contract], doc)
    rows.append((contract, label, s, p, "SAME" if s == p.split(" ")[0] else "DIFF"))

# 1. the five examples as published
for name, v in VALIDATORS.items():
    case(name, f"example {EX_FOR[name]} as published", EXAMPLES[EX_FOR[name]])
case("source_ref", "example source_ref.drive.json as published", EXAMPLES["source_ref.drive.json"])

# 2. schema_version spellings, on every contract (the five declarations)
for ver in ["1.0", "1.1", "1.10", "01.0", "1.00", "1", "1.0.0", "2.0", "1.0\n", " 1.0", 1, 1.0, None]:
    for name in VALIDATORS:
        doc = dict(EXAMPLES[EX_FOR[name]]); doc["schema_version"] = ver
        case(name, f"schema_version={ver!r}", doc)
    doc = dict(EXAMPLES["source_ref.drive.json"]); doc["schema_version"] = ver
    case("source_ref", f"(external) schema_version={ver!r}", doc)

# 3. runtime_package presence / type
base = EXAMPLES["repository_role.json"]
d = dict(base); d.pop("runtime_package"); case("repository_role", "runtime_package omitted", d)
case("repository_role", "runtime_package=null", dict(base, runtime_package=None))
case("repository_role", "runtime_package='pkg'", dict(base, runtime_package="pkg"))
case("repository_role", "runtime_package=''", dict(base, runtime_package=""))
case("repository_role", "runtime_package=1", dict(base, runtime_package=1))
case("repository_role", "runtime_package=['a']", dict(base, runtime_package=["a"]))

# 4. direction check: places where the Python reader is stricter than the schema
g = EXAMPLES["source_ref.git.json"]
case("source_ref", "git path 'sandbox/x.md'", dict(g, path="sandbox/x.md"))
case("source_ref", "git path '../x.md'", dict(g, path="../x.md"))
case("source_ref", "git path 'a\\\\b.md' (backslash)", dict(g, path="a\\b.md"))
case("source_ref", "git role ' proof ' (padded)", dict(g, role=" proof "))
case("source_ref", "git size_bytes=1.0 (float)", dict(g, size_bytes=1.0))
case("source_ref", "git size_bytes=true", dict(g, size_bytes=True))
case("source_ref", "git commit uppercase hex", dict(g, commit="A" * 40))
e = EXAMPLES["source_ref.drive.json"]
case("source_ref", "external visibility='private' (public reader)", dict(e, visibility="private"))
case("repository_role", "canonical_authority_for=['']", dict(base, canonical_authority_for=[""]))
case("repository_role", "role_id=' ' (whitespace)", dict(base, role_id=" "))
w = EXAMPLES["workspace_snapshot.json"]
dup = json.loads(json.dumps(w)); dup["repositories"][7] = dict(dup["repositories"][0]); case("workspace_snapshot", "8 entries, Math- twice, trial missing", dup)
m = EXAMPLES["source_manifest.json"]
dm = json.loads(json.dumps(m)); dm["files"].append(dict(dm["files"][0])); case("source_manifest", "duplicate manifest path", dm)
dm2 = json.loads(json.dumps(m)); dm2["files"][0]["path"] = "sandbox/private.py"; case("source_manifest", "manifest path under sandbox/", dm2)

# 5. extra keys (both tolerate at top level)
case("repository_role", "extra key lemma_closed=true at top level", dict(base, lemma_closed=True))
case("source_ref", "git extra key status='accepted'", dict(g, status="accepted"))

w1 = max(len(r[1]) for r in rows)
print(f"{'contract':18} {'case':{w1}}  schema   python")
for c, l, s, p, same in rows:
    print(f"{c:18} {l:{w1}}  {s:8} {p:40.40} {same}")
diff = [r for r in rows if r[4] == "DIFF"]
print(f"\n{len(rows)} cases; {len(diff)} differ.")
print("schema stricter than Python:", sum(1 for r in diff if r[2] == "reject"))
print("Python stricter than schema:", sum(1 for r in diff if r[2] == "accept"))
print("returned object keeps omission:", "runtime_package" in cv.validate_repository_role({k: v for k, v in base.items() if k != "runtime_package"}))
print("VER regex:", cv.VER.pattern, "| fullmatch('1.0\\n') ->", cv.VER.fullmatch("1.0\n"))
