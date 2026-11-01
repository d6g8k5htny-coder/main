"""Capture, never reinterpret, the original A8 baseline and all 89 mutants."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record(proc, label, expected=None):
    output = proc.stdout + proc.stderr
    data = {"label": label, "returncode": proc.returncode,
            "expected_method_historical": expected,
            "stdout": proc.stdout, "stderr": proc.stderr,
            "failure_headers": re.findall(r"^FAIL: (.*)$", output, re.M),
            "errors": re.findall(r"^ERROR: (.*)$", output, re.M)}
    (OUT / (label + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    print("A8_OBSERVATION " + json.dumps({
        "label": label, "returncode": proc.returncode,
        "expected_method_historical": expected,
        "failure_headers": data["failure_headers"], "errors": data["errors"],
        "trace": re.findall(r'(?m)^  File .*$|^    self\..*$|^AssertionError:.*$', output)
    }), flush=True)
    return data


observations = []
for kind, testfile, tool, count in [
    ("scanner", "test_cbz_layer_a_scan_mutants.py", "layer_a_scan.py", 66),
    ("census", "test_cbz_a3_float_census.py", "a3_float_census.py", 23),
]:
    mod = load(ROOT / "tests" / testfile, "capture_" + kind)
    if len(mod.MUTANTS) != count:
        raise RuntimeError(f"{kind}: original mutant population drifted")
    path = ROOT / "tools" / "cbz" / tool
    observations.append(record(mod.run_suite(str(path)), kind + "_baseline"))
    source = path.read_text()
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, old, new) in enumerate(mod.MUTANTS):
            if source.count(old) != 1:
                raise RuntimeError(f"{name}: mutation point drifted")
            mutant = Path(tmp) / f"mutant_{i:02d}.py"
            mutant.write_text(source.replace(old, new))
            data = record(mod.run_suite(str(mutant)), f"{kind}_{i:02d}",
                          mod.EXPECTED_FAILURE[name])
            data["mutant"] = name
            observations.append(data)
index = {"checked_commit": os.environ.get("GITHUB_SHA"),
         "python": sys.version, "optimize": sys.flags.optimize,
         "source_base": "32ab7d9013264d2874aa4b778a90eede0cda0cdf",
         "observations": observations}
(OUT / "observations.json").write_text(json.dumps(index, indent=2) + "\n")
if any(d["returncode"] != 0 for d in observations if d["label"].endswith("_baseline")):
    raise SystemExit("original baseline failed; characterization is not qualification")
if any(d["returncode"] != 1 or d["errors"] or not d["failure_headers"]
       for d in observations if not d["label"].endswith("_baseline")):
    raise SystemExit("original mutant had an unsupported outcome; inspect exact record")
print("A8_CAPTURE_COMPLETE", len(observations), flush=True)
