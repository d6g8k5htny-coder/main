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


class NodeStatusLiveContract(unittest.TestCase):
    def test_complete_real_input_cohort_is_checked_without_rewriting_pins(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_complete_real_input_cohort_is_checked_without_rewriting_pins")

    def test_missing_any_of_the_four_real_record_files_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_any_of_the_four_real_record_files_refuses")

    def test_invalid_schema_refuses_before_source_observation(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_invalid_schema_refuses_before_source_observation")

    def test_missing_schema_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_schema_refuses")

    def test_duplicate_record_json_keys_refuse(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_duplicate_record_json_keys_refuse")

    def test_unexpected_record_file_is_not_ignored(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_unexpected_record_file_is_not_ignored")

    def test_schema_byte_drift_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_schema_byte_drift_refuses")

    def test_graph_byte_drift_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_graph_byte_drift_refuses")

    def test_no_declared_fresh_record_is_not_an_empty_success(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_no_declared_fresh_record_is_not_an_empty_success")

    def test_changed_cited_git_object_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_git_object_refuses")

    def test_changed_cited_discussion_timestamp_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_discussion_timestamp_refuses")

    def test_unavailable_source_cannot_become_fresh(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_unavailable_source_cannot_become_fresh")

    def test_record_byte_drift_even_with_unchanged_json_value_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_byte_drift_even_with_unchanged_json_value_refuses")

    def test_record_set_drift_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_set_drift_refuses")

    def test_success_without_observation_counter_coverage_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_success_without_observation_counter_coverage_refuses")

    def test_correct_count_with_wrong_source_observation_urls_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_correct_count_with_wrong_source_observation_urls_refuses")

    def test_observation_without_captured_default_head_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_without_captured_default_head_refuses")

    def test_observation_head_must_match_the_captured_repository_cut(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_head_must_match_the_captured_repository_cut")

    def test_same_count_and_url_set_with_wrong_multiplicities_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_same_count_and_url_set_with_wrong_multiplicities_refuses")

    def test_retained_discussion_timestamp_must_match_the_captured_source_revision(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_retained_discussion_timestamp_must_match_the_captured_source_revision")

    def test_malformed_run_identity_refuses(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_malformed_run_identity_refuses")

    def test_live_entry_refuses_missing_authentication_before_constructing_observer(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_refuses_missing_authentication_before_constructing_observer")

    def test_live_entry_uses_one_default_observer_for_the_whole_mode(self):
        run_legacy_case(self, "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_uses_one_default_observer_for_the_whole_mode")


if __name__ == "__main__":
    unittest.main()
