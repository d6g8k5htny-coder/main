"""Adverse tests for G11 workflow reference recognition."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import g11_census as g11

TOOL = "tools/cbz/g11_origin_mutation_selftest.py"
PUBLISHER = "tools/cbz_site_publisher_cas_mutations.py"
RECOVERY = "tools/cbz_site_recovery_mutations.py"
TOOLS = {TOOL, PUBLISHER, RECOVERY}

def yaml_run(command):
    return "jobs:\n  check:\n    steps:\n      - run: " + command + "\n"

class WorkflowFormats(unittest.TestCase):
    def test_env_unset_preserves_direct_execution_reference(self):
        text = yaml_run("env -u PYTHONPATH --unset PYTHONHOME -- python3 -B -S " + TOOL)
        self.assertEqual(g11.workflow_tool_references(text, TOOLS), {TOOL})

    def test_env_unsupported_or_nonexecution_is_not_credited(self):
        for command in (
            "env -u",
            "env -u PYTHONPATH python3 -c 'print(\"" + TOOL + "\")'",
            "env -u PYTHONPATH echo " + TOOL,
            "env --split-string='python3 " + TOOL + "'",
            "env NAME=value python3 " + TOOL,
        ):
            with self.subTest(command=command):
                self.assertEqual(g11.workflow_tool_references(yaml_run(command), TOOLS), set())

    def test_json_workflow_recognizes_only_job_step_runs(self):
        text = json.dumps({"on": {"push": {"paths": [PUBLISHER]}},
                           "jobs": {"real": {"steps": [
                               {"with": {"run": "python3 " + RECOVERY}},
                               {"run": "set -e\npython3 -B -S " + PUBLISHER + "\n"}]}}})
        self.assertEqual(g11.workflow_tool_references(text, TOOLS), {PUBLISHER})

    def test_json_duplicate_keys_and_non_run_paths_are_refused(self):
        duplicate = ('{"jobs":{"one":{"steps":[]}},"jobs":{"two":{"steps":[{"run":"python3 '
                     + PUBLISHER + '"}]}}}')
        metadata = json.dumps({"jobs": {"a": {"steps": [{"with": {"path": RECOVERY}}]}}})
        self.assertEqual(g11.workflow_tool_references(duplicate, TOOLS), set())
        self.assertEqual(g11.workflow_tool_references(metadata, TOOLS), set())

    def test_current_native_workflows_expose_real_static_references(self):
        cases = ((".github/workflows/cbz-g11-auditor.yml", TOOL),
                 (".github/workflows/cbz-publisher-cas.yml", PUBLISHER),
                 (".github/workflows/cbz-site-recovery.yml", RECOVERY))
        for path, expected in cases:
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn(expected, g11.workflow_tool_references(text, TOOLS))


    def test_json_noop_step_is_failing(self):
        text = json.dumps({"jobs": {"a": {"steps": [{"run": "true"}]}}})
        report = g11.census({".github/workflows/noop.yml": text.encode()})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([(finding["class"], finding["line"]) for finding in report["failing"]],
                         [("noop", 0)])

    def test_duplicate_json_workflow_is_unparseable(self):
        text = '{"jobs":{"one":{"steps":[]}},"jobs":{"two":{"steps":[]}}}'
        report = g11.census({".github/workflows/duplicate.yml": text.encode()})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([finding["class"] for finding in report["failing"]], ["unparseable"])


    def test_json_array_workflow_is_unparseable(self):
        report = g11.census({".github/workflows/array.yml": b"[]"})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([finding["class"] for finding in report["failing"]], ["unparseable"])

    def test_nonstandard_json_constant_is_unparseable(self):
        text = '{"jobs":{},"unsupported":NaN}'
        report = g11.census({".github/workflows/nonstandard.yml": text.encode()})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([finding["class"] for finding in report["failing"]], ["unparseable"])

    def test_non_noop_json_run_is_not_a_noop(self):
        text = json.dumps({"jobs": {"a": {"steps": [{"run": "python3 -V"}]}}})
        report = g11.census({".github/workflows/normal.yml": text.encode()})
        self.assertEqual(report["failing"], [])


    def test_nonstring_json_run_is_unparseable(self):
        text = json.dumps({"jobs": {"a": {"steps": [{"run": False}]}}})
        report = g11.census({".github/workflows/invalid-run.yml": text.encode()})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual([finding["class"] for finding in report["failing"]], ["unparseable"])

    def test_invalid_json_job_step_shapes_are_unparseable(self):
        cases = ({"jobs": {"a": None}}, {"jobs": {"a": {"steps": "not a list"}}},
                 {"jobs": {"a": {"steps": ["not an object"]}}})
        for workflow in cases:
            with self.subTest(workflow=workflow):
                report = g11.census({".github/workflows/invalid-shape.yml": json.dumps(workflow).encode()})
                self.assertEqual(report["status"], "FAIL")


    def test_env_unset_equals_preserves_direct_reference(self):
        text = yaml_run("env --unset=PYTHONPATH python3 -B -S " + TOOL)
        self.assertEqual(g11.workflow_tool_references(text, TOOLS), {TOOL})


    def test_interpreter_early_exit_or_alternate_code_is_not_a_tool_reference(self):
        import shlex
        import subprocess
        import tempfile

        for option in ("-V", "--version", "--help", "-cprint(1)", "-mjson.tool"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as root:
                path = Path(root) / TOOL
                path.parent.mkdir(parents=True)
                path.write_text("from pathlib import Path\nPath('tool_was_executed').touch()\n", encoding="utf-8")
                subprocess.run([sys.executable, option, TOOL], cwd=root, capture_output=True, text=True,
                               timeout=10, check=False)
                self.assertFalse((Path(root) / "tool_was_executed").exists())
                text = yaml_run("python3 " + shlex.quote(option) + " " + TOOL)
                self.assertEqual(g11.workflow_tool_references(text, TOOLS), set())


    def test_shell_no_execution_options_are_not_tool_references(self):
        import subprocess
        import tempfile

        shell_tool = "tools/cbz/example_guard.sh"
        for interpreter, option in (("bash", "-n"), ("sh", "-n"),
                                    ("bash", "--version"), ("sh", "--version")):
            with self.subTest(interpreter=interpreter, option=option), tempfile.TemporaryDirectory() as root:
                path = Path(root) / shell_tool
                path.parent.mkdir(parents=True)
                path.write_text("touch tool_was_executed\n", encoding="utf-8")
                subprocess.run([interpreter, option, shell_tool], cwd=root, capture_output=True, text=True,
                               timeout=10, check=False)
                self.assertFalse((Path(root) / "tool_was_executed").exists())
                text = yaml_run(interpreter + " " + option + " " + shell_tool)
                self.assertEqual(g11.workflow_tool_references(text, {shell_tool}), set())


    def test_python_module_direct_reference_with_supported_options(self):
        text = yaml_run("python3 -B -S -m tools.cbz.g11_origin_mutation_selftest")
        self.assertEqual(g11.workflow_tool_references(text, TOOLS), {TOOL})

    def test_python_module_after_early_exit_is_not_a_reference(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / TOOL
            path.parent.mkdir(parents=True)
            path.write_text("from pathlib import Path\nPath('tool_was_executed').touch()\n", encoding="utf-8")
            subprocess.run([sys.executable, "-V", "-m", "tools.cbz.g11_origin_mutation_selftest"],
                           cwd=root, capture_output=True, text=True, timeout=10, check=False)
            self.assertFalse((Path(root) / "tool_was_executed").exists())
            text = yaml_run("python3 -V -m tools.cbz.g11_origin_mutation_selftest")
            self.assertEqual(g11.workflow_tool_references(text, TOOLS), set())


    def test_invalid_interpreter_option_operands_do_not_execute_tool(self):
        import shlex
        import subprocess
        import tempfile

        shell_tool = "tools/cbz/example_guard.sh"
        cases = (("python3", "--check-hash-based-pycs", "invalid", TOOL),
                 ("bash", "-o", "not_a_real_option", shell_tool),
                 ("sh", "-o", "not_a_real_option", shell_tool))
        for interpreter, option, operand, tool in cases:
            with self.subTest(interpreter=interpreter), tempfile.TemporaryDirectory() as root:
                path = Path(root) / tool
                path.parent.mkdir(parents=True)
                content = ("from pathlib import Path\nPath('tool_was_executed').touch()\n"
                           if tool.endswith(".py") else "touch tool_was_executed\n")
                path.write_text(content, encoding="utf-8")
                command = [sys.executable if interpreter == "python3" else interpreter, option, operand, tool]
                subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
                self.assertFalse((Path(root) / "tool_was_executed").exists())
                text = yaml_run(shlex.join([interpreter, option, operand, tool]))
                self.assertEqual(g11.workflow_tool_references(text, {tool}), set())

if __name__ == "__main__":
    unittest.main()
