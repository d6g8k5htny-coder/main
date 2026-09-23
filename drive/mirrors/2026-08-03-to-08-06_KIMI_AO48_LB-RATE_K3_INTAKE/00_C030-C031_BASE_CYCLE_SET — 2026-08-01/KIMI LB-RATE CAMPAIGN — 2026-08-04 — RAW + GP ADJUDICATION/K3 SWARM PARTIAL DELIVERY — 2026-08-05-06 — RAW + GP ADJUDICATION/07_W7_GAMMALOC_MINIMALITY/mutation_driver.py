#!/usr/bin/env python3
"""
mutation_driver.py -- executable falsifier harness for the K3-W7
gamma-LOC certificate. Writes deliberately mutated copies of the
station table of record into ./mutation_workspace/ and runs
verify_k3_w7_gammaloc_v1.py --json on each. The pristine copy must
PASS (exit 0); every mutated copy must FAIL (nonzero exit). Any
deviation raises SystemExit(1). Deterministic output, byte-identical
under `python3` and `python3 -O`.
"""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIER = os.path.join(HERE, "verify_k3_w7_gammaloc_v1.py")
SRC = "/mnt/agents/upload/c031 verify.json"
WS = os.path.join(HERE, "mutation_workspace")


def ck(cond, msg="check"):
    if not cond:
        raise SystemExit("CHECK FAILED: " + str(msg))


def run(path):
    cmd = [sys.executable, VERIFIER, "--json", path]
    if sys.flags.optimize:
        cmd.insert(1, "-O")
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p.returncode


os.makedirs(WS, exist_ok=True)
base = json.loads(open(SRC, "rb").read().decode())

variants = []

# M1: pristine content (re-serialized; hash binding skipped in --json mode,
# so this tests that the content checks accept the true record)
variants.append(("M1_pristine_accept", base, 0))

# M2: total above gate
v = json.loads(json.dumps(base)); v["total"] = 0.5
variants.append(("M2_total_above_gate", v, 1))

# M3: gate shrunk below total
v = json.loads(json.dumps(base)); v["gate"] = 0.001
variants.append(("M3_gate_shrunk", v, 1))

# M4: one station's variance ratio corrupted
v = json.loads(json.dumps(base)); v["stations"][0]["v"] = 0.9
variants.append(("M4_station_v_corrupt", v, 1))

# M5: a station deleted
v = json.loads(json.dumps(base)); v["stations"] = v["stations"][:-1]
variants.append(("M5_station_dropped", v, 1))

# M6: proj_c1 perturbed so proj+mean != total beyond tolerance
v = json.loads(json.dumps(base)); v["proj_c1"] = v["proj_c1"] + 1e-12
variants.append(("M6_sum_inconsistent", v, 1))

# M7: rice constant corrupted
v = json.loads(json.dumps(base)); v["rice"] = 2.435
variants.append(("M7_rice_corrupt", v, 1))

# M8: closes flag flipped
v = json.loads(json.dumps(base)); v["closes"] = False
variants.append(("M8_closes_flipped", v, 1))

print("=== mutation_driver.py -- executable falsifier harness (K3-W7) ===")
for name, obj, expect_fail in variants:
    path = os.path.join(WS, name + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    rc = run(path)
    if expect_fail:
        ck(rc != 0, "%s: mutated artifact ACCEPTED (exit 0) -- verifier is vacuous" % name)
        print("%-24s exit=%d expected=FAIL  OK (rejected)" % (name, rc))
    else:
        ck(rc == 0, "%s: pristine artifact REJECTED (exit %d)" % (name, rc))
        print("%-24s exit=%d expected=PASS  OK (accepted)" % (name, rc))

# M9: byte-corrupt (truncated) file must fail at parse
path = os.path.join(WS, "M9_truncated.json")
raw = open(SRC, "rb").read()
with open(path, "wb") as fh:
    fh.write(raw[: len(raw) // 2])
rc = run(path)
ck(rc != 0, "M9_truncated: corrupt artifact ACCEPTED")
print("%-24s exit=%d expected=FAIL  OK (rejected)" % ("M9_truncated", rc))

print("MUTATION_DRIVER_PASS")
