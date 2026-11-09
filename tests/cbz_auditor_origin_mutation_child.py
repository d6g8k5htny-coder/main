"""Run one auditor-origin control with an explicitly selected helper copy."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import cbz_inventory_origin as recorder
from tests import test_cbz_auditor_origin as controls
from tests import test_cbz_a3_float_census as a3
from tests import test_cbz_layer_a_scan_mutants as lean


def main():
    if len(sys.argv) != 3:
        raise SystemExit('helper copy and exact selected test are required')
    spec = importlib.util.spec_from_file_location('auditor_origin_mutant', sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The copied adapter lives outside the checkout; its tests still run
    # against the exact checkout that owns the designated test bodies.
    module.ROOT = ROOT
    controls.cbz_auditor_origin = module
    a3.cbz_auditor_origin = module
    lean.cbz_auditor_origin = module
    report = recorder._child(ROOT, sys.argv[2])
    print(json.dumps(report, sort_keys=True))
    return 0 if (report['failure_count'] == 0 and report['error_count'] == 0 and
                 report['unexpected_success_count'] == 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
