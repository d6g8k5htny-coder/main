"""Historical v1 fixture wrappers; original assertion bodies remain byte-exact.

Each canonical outer case runs only its identical original method in a fresh
same-interpreter external fixture. This supplies no current-v2 or HTTP credit.
"""
import hashlib
from pathlib import Path
import types
import unittest

_runner_path = Path(__file__).resolve().parent / "node_status_legacy_v1_runner.py"
_runner_raw = _runner_path.read_bytes()
_runner_blob = hashlib.sha1(b"blob " + str(len(_runner_raw)).encode("ascii")
                            + b"\0" + _runner_raw).hexdigest()
if _runner_blob != "77e036b5e8e816cec67720318d61fb95d445bc47":
    raise ImportError("legacy fixture runner source identity mismatch")
_runner = types.ModuleType("_node_status_legacy_v1_checked_runner")
_runner.__file__ = str(_runner_path)
_runner.__package__ = ""
exec(compile(_runner_raw, str(_runner_path), "exec", dont_inherit=True), _runner.__dict__)
run_legacy_case = _runner.run_legacy_case


class NodeStatusLiveCLIContract(unittest.TestCase):
    def test_help_is_offline_and_describes_the_actual_root_output_interface(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_help_is_offline_and_describes_the_actual_root_output_interface")

    def test_missing_required_output_argument_is_not_success(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_required_output_argument_is_not_success")

    def test_missing_native_run_context_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_native_run_context_refuses")

    def test_wrong_repository_context_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_wrong_repository_context_refuses")

    def test_context_commit_must_equal_the_actual_selected_commit(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_context_commit_must_equal_the_actual_selected_commit")

    def test_invalid_native_run_and_attempt_values_refuse(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_invalid_native_run_and_attempt_values_refuse")

    def test_selected_bytes_must_match_git_even_when_assume_unchanged(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_selected_bytes_must_match_git_even_when_assume_unchanged")

    def test_changed_dependency_cannot_execute_before_source_binding_refusal(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_changed_dependency_cannot_execute_before_source_binding_refusal")

    def test_caller_cannot_select_another_checkout_even_with_identical_source(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_caller_cannot_select_another_checkout_even_with_identical_source")

    def test_default_root_is_the_executed_checkout_not_callers_cwd(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_default_root_is_the_executed_checkout_not_callers_cwd")

    def test_existing_output_is_not_overwritten(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_existing_output_is_not_overwritten")

    def test_output_inside_the_selected_source_is_refused(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_inside_the_selected_source_is_refused")

    def test_output_symlink_is_refused_without_touching_its_target(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_symlink_is_refused_without_touching_its_target")

    def test_output_parent_symlink_is_refused(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_parent_symlink_is_refused")

    def test_missing_token_retains_real_source_and_run_identity_without_live_credit(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_token_retains_real_source_and_run_identity_without_live_credit")

    def test_cli_reports_actual_interpreter_mode_in_each_child(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_cli_reports_actual_interpreter_mode_in_each_child")

    def test_ambient_git_path_redirects_cannot_replace_the_selected_checkout(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_ambient_git_path_redirects_cannot_replace_the_selected_checkout")

    def test_api_or_transcript_override_is_not_a_supported_cli_input(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_api_or_transcript_override_is_not_a_supported_cli_input")

    def test_unbound_package_cannot_replace_either_bound_project_dependency(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_unbound_package_cannot_replace_either_bound_project_dependency")

    def test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency(self):
        run_legacy_case(self, "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency")


if __name__ == "__main__":
    unittest.main()
