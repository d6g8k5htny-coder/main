"""Kill seeded one-point mutations of the strict package validator."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
VALIDATOR = HERE / "validator.py"


MUTANTS = {
    "allow_extra_field": (
        "set(document) == expected",
        "set(document) >= expected",
        "extra",
    ),
    "allow_scientific_status_promotion": (
        'document["scientific_status_effect"] == "NONE"',
        'document["scientific_status_effect"] in ("NONE", "ACCEPTED")',
        "status",
    ),
    "allow_theorem_consumption": (
        'source["theorem_conclusion_consumed"] is False',
        'source["theorem_conclusion_consumed"] in (False, True)',
        "theorem",
    ),
    "allow_incomplete_closure": (
        "set(closure) == REQUIRED_CLOSURE",
        "set(closure) <= REQUIRED_CLOSURE",
        "closure",
    ),
}


PROBE = r'''import importlib.util, json, pathlib, sys, traceback
spec = importlib.util.spec_from_file_location("mutated_validator", sys.argv[1])
module = importlib.util.module_from_spec(spec)
here = pathlib.Path(sys.argv[2])
kind = sys.argv[3]
control = json.loads((here / "control.json").read_text())
sources = json.loads((here / "source_manifest.json").read_text())
try:
    spec.loader.exec_module(module)
    if kind == "extra":
        control["unexpected"] = False
        module.validate_document(control)
    elif kind == "status":
        control["scientific_status_effect"] = "ACCEPTED"
        module.validate_document(control)
    elif kind == "theorem":
        sources["sources"][0]["theorem_conclusion_consumed"] = True
        module._validate_sources(sources)
    elif kind == "closure":
        module.verify_closure(here, {})
    else:
        raise RuntimeError("unknown probe")
except ValueError:
    raise SystemExit(0)
except Exception:
    traceback.print_exc()
    raise SystemExit(4)
raise SystemExit(3)
'''


def run():
    source = VALIDATOR.read_text(encoding="utf-8")
    killed = []
    for name, (old, new, probe) in MUTANTS.items():
        if source.count(old) != 1:
            raise RuntimeError(name + " mutation point is not unique")
        mutated = source.replace(old, new, 1)
        with tempfile.TemporaryDirectory(prefix="reactor-v0-mutant-") as temporary:
            path = Path(temporary) / "validator.py"
            path.write_text(mutated, encoding="utf-8")
            command = [
                sys.executable, "-O", "-B", "-c", PROBE,
                str(path), str(HERE), probe,
            ]
            baseline = subprocess.run(
                command[:5] + [str(VALIDATOR)] + command[6:],
                text=True, capture_output=True, check=False,
            )
            if baseline.returncode != 0:
                raise RuntimeError(
                    name + " probe does not reject under the original validator: " +
                    baseline.stderr
                )
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False,
            )
        if completed.returncode == 3:
            killed.append(name)
        elif completed.returncode == 4:
            raise RuntimeError(name + " crashed instead of accepting its probe: " +
                               completed.stderr)
        else:
            raise RuntimeError(name + " returned unexpected probe status " +
                               str(completed.returncode))
    return killed


if __name__ == "__main__":
    print(json.dumps({"killed": run()}, sort_keys=True))
