from importlib import util as _util
from pathlib import Path as _Path
_spec = _util.spec_from_file_location(
    "_evidence_path_source_tests", _Path(__file__).with_name("test_evidence_path.py"))
_source = _util.module_from_spec(_spec)
_spec.loader.exec_module(_source)

class EvidencePathArchitectureTests(_source.EvidencePathTests):
    """Run the original case methods under an explicit architecture adapter."""


def load_tests(loader, standard_tests, pattern):
    original = loader.getTestCaseNames(_source.EvidencePathTests)
    adapted = loader.getTestCaseNames(EvidencePathArchitectureTests)
    if not original or adapted != original:
        raise RuntimeError("evidence-path adapter must preserve a nonempty method set")
    suite = loader.loadTestsFromTestCase(EvidencePathArchitectureTests)
    if suite.countTestCases() != len(original):
        raise RuntimeError("evidence-path adapter case count mismatch")
    return suite
