"""Run only the frozen historical v1 fixture cases; no HTTP/current-v2 credit.

The original eleven payload bodies and each original test method are unchanged.
Current outer wrappers retain their names, while fresh children execute one
identical original method with its normal unittest setup and cleanup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

_ROOT = Path(os.path.abspath(__file__)).parent.parent
_BUNDLE_PATH = "tests/fixtures/node_status_live_v1/bundle.json"
_RUNNER_PATH = "tests/node_status_legacy_v1_runner.py"
_WRAPPERS = ("tests/test_node_status_live_check.py", "tests/test_node_status_live_cli.py")
_LIMIT = 2_000_000
_STREAM_LIMIT = 4 * 1024 * 1024
_EXPECTED = {
    "schema": "node-status-legacy-v1.bundle.v1",
    "source_repository": "d6g8k5htny-coder/main",
    "source_commit": "d09f3252e4f9eb91d624f42357e81e3e43a658e5",
    "source_tree": "111ec264e36b15a03bb3c5150247003198fc1609",
    "members": [
        {
            "source_path": "architecture/node-status.schema.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/architecture/node-status.schema.json.txt",
            "materialized_path": "architecture/node-status.schema.json",
            "git_mode": "100644",
            "git_blob": "2bad95d30bcb89ea686d1ac5a50d52113dd730be",
            "bytes": 5018,
            "sha256": "b82f1aee34097071bc47def5e3d371a3ced2df1332fb29779fe1d56e38676e45"
        },
        {
            "source_path": "docs/site/current.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/docs/site/current.json.txt",
            "materialized_path": "docs/site/current.json",
            "git_mode": "100644",
            "git_blob": "a64fb6d9768ada12b9884adac854e0ccdd5a910e",
            "bytes": 962639,
            "sha256": "560d3d4e77369835c832dbb4fd0d12f4988744045b9cbc4c7978beddb6e9d20f"
        },
        {
            "source_path": "records/node_status/eng.d0-packet-allowlist-fix.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/records/node_status/eng.d0-packet-allowlist-fix.json.txt",
            "materialized_path": "records/node_status/eng.d0-packet-allowlist-fix.json",
            "git_mode": "100644",
            "git_blob": "0a7f209b58f4d309f8c8ae36d8dce1696c22ffbd",
            "bytes": 988,
            "sha256": "8a5bfbc046147c5bf4a95e8ff912abcdef5f61761583e5654a7327772c72d3b3"
        },
        {
            "source_path": "records/node_status/eng.hard-gate.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/records/node_status/eng.hard-gate.json.txt",
            "materialized_path": "records/node_status/eng.hard-gate.json",
            "git_mode": "100644",
            "git_blob": "9889637191bc01e26a6ad6dfec19b35de8b72a0b",
            "bytes": 1273,
            "sha256": "f476f09dcdf9f691c1579d8396590afb02dac86aa9bfb207a7331b872a8e5846"
        },
        {
            "source_path": "records/node_status/math.lifetime-remainder.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/records/node_status/math.lifetime-remainder.json.txt",
            "materialized_path": "records/node_status/math.lifetime-remainder.json",
            "git_mode": "100644",
            "git_blob": "a6f82a36e4f9a31abf656661f7c8ee5201874522",
            "bytes": 4235,
            "sha256": "e672df10ee7f6cd75d86969e899b7d00bad23884b147fddf5bdd93b8db295180"
        },
        {
            "source_path": "records/node_status/math.side24-coefficient.json",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/records/node_status/math.side24-coefficient.json.txt",
            "materialized_path": "records/node_status/math.side24-coefficient.json",
            "git_mode": "100644",
            "git_blob": "ecf9c42723ee72b8e7d1a4d5b36470f158d6af0e",
            "bytes": 938,
            "sha256": "0cc919fde0c9f57cdaab45d10c6eec8fc7d8d417bc33bafcd00cc53c7b6b2be7"
        },
        {
            "source_path": "tests/test_node_status_live_check.py",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/tests/test_node_status_live_check.py.txt",
            "materialized_path": "tests/test_node_status_live_check.py",
            "git_mode": "100644",
            "git_blob": "c5d578ddb9b7699f24966784ebfcb518d2d7284a",
            "bytes": 16036,
            "sha256": "62c0ae5108c7bc36b6e41f2d8fae1e168589f326480ca9982fe57066e2348b33"
        },
        {
            "source_path": "tests/test_node_status_live_cli.py",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/tests/test_node_status_live_cli.py.txt",
            "materialized_path": "tests/test_node_status_live_cli.py",
            "git_mode": "100644",
            "git_blob": "14d0c84b1236fdc0d01083c62b791117b663c682",
            "bytes": 20697,
            "sha256": "fa3d35decb2c377d75c9e8f5aba9959ca4fada12ada786777550efcae1074f42"
        },
        {
            "source_path": "tools/node_status_live_check.py",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/tools/node_status_live_check.py.txt",
            "materialized_path": "tools/node_status_live_check.py",
            "git_mode": "100644",
            "git_blob": "1e4852188f15060176b4458f58b27815f56845fd",
            "bytes": 21374,
            "sha256": "85bb8ca6821d376655a629447940bdfa057a3567eaa72f734a4d7261ddd9a5b5"
        },
        {
            "source_path": "tools/node_status_refresh.py",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/tools/node_status_refresh.py.txt",
            "materialized_path": "tools/node_status_refresh.py",
            "git_mode": "100644",
            "git_blob": "5fa7c8b6647938a0aca538a7735edede65de9ed5",
            "bytes": 23238,
            "sha256": "ed018809b290046f9155487c41bbe7ba594dc4fba7941256b7ebb5abff09cac0"
        },
        {
            "source_path": "tools/validate_node_status.py",
            "payload_path": "tests/fixtures/node_status_live_v1/payload/tools/validate_node_status.py.txt",
            "materialized_path": "tools/validate_node_status.py",
            "git_mode": "100644",
            "git_blob": "e9d44b7620536b389a35ff6fad8aef2b0464b5d2",
            "bytes": 16375,
            "sha256": "9ccadda867e00617b42177e8aee067c64bd6e52fbc79ec8fddc82653dfbc6375"
        }
    ]
}
HELPER_IDS = (
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_complete_real_input_cohort_is_checked_without_rewriting_pins",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_any_of_the_four_real_record_files_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_invalid_schema_refuses_before_source_observation",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_schema_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_duplicate_record_json_keys_refuse",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_unexpected_record_file_is_not_ignored",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_schema_byte_drift_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_graph_byte_drift_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_no_declared_fresh_record_is_not_an_empty_success",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_git_object_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_discussion_timestamp_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_unavailable_source_cannot_become_fresh",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_byte_drift_even_with_unchanged_json_value_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_set_drift_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_success_without_observation_counter_coverage_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_correct_count_with_wrong_source_observation_urls_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_without_captured_default_head_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_head_must_match_the_captured_repository_cut",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_same_count_and_url_set_with_wrong_multiplicities_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_retained_discussion_timestamp_must_match_the_captured_source_revision",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_malformed_run_identity_refuses",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_refuses_missing_authentication_before_constructing_observer",
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_uses_one_default_observer_for_the_whole_mode",
)
CLI_IDS = (
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_help_is_offline_and_describes_the_actual_root_output_interface",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_required_output_argument_is_not_success",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_native_run_context_refuses",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_wrong_repository_context_refuses",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_context_commit_must_equal_the_actual_selected_commit",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_invalid_native_run_and_attempt_values_refuse",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_selected_bytes_must_match_git_even_when_assume_unchanged",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_changed_dependency_cannot_execute_before_source_binding_refusal",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_caller_cannot_select_another_checkout_even_with_identical_source",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_default_root_is_the_executed_checkout_not_callers_cwd",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_existing_output_is_not_overwritten",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_inside_the_selected_source_is_refused",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_symlink_is_refused_without_touching_its_target",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_parent_symlink_is_refused",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_token_retains_real_source_and_run_identity_without_live_credit",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_cli_reports_actual_interpreter_mode_in_each_child",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_ambient_git_path_redirects_cannot_replace_the_selected_checkout",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_api_or_transcript_override_is_not_a_supported_cli_input",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_unbound_package_cannot_replace_either_bound_project_dependency",
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency",
)
_CASES = {
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_complete_real_input_cohort_is_checked_without_rewriting_pins": {
        "path": "tests/test_node_status_live_check.py",
        "line": 136
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_any_of_the_four_real_record_files_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 155
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_invalid_schema_refuses_before_source_observation": {
        "path": "tests/test_node_status_live_check.py",
        "line": 169
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_schema_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 178
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_duplicate_record_json_keys_refuse": {
        "path": "tests/test_node_status_live_check.py",
        "line": 184
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_unexpected_record_file_is_not_ignored": {
        "path": "tests/test_node_status_live_check.py",
        "line": 192
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_schema_byte_drift_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 198
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_graph_byte_drift_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 205
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_no_declared_fresh_record_is_not_an_empty_success": {
        "path": "tests/test_node_status_live_check.py",
        "line": 212
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_git_object_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 219
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_discussion_timestamp_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 231
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_unavailable_source_cannot_become_fresh": {
        "path": "tests/test_node_status_live_check.py",
        "line": 236
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_byte_drift_even_with_unchanged_json_value_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 248
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_record_set_drift_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 255
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_success_without_observation_counter_coverage_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 261
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_correct_count_with_wrong_source_observation_urls_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 267
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_without_captured_default_head_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 272
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_head_must_match_the_captured_repository_cut": {
        "path": "tests/test_node_status_live_check.py",
        "line": 277
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_same_count_and_url_set_with_wrong_multiplicities_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 288
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_retained_discussion_timestamp_must_match_the_captured_source_revision": {
        "path": "tests/test_node_status_live_check.py",
        "line": 303
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_malformed_run_identity_refuses": {
        "path": "tests/test_node_status_live_check.py",
        "line": 315
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_refuses_missing_authentication_before_constructing_observer": {
        "path": "tests/test_node_status_live_check.py",
        "line": 326
    },
    "tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_uses_one_default_observer_for_the_whole_mode": {
        "path": "tests/test_node_status_live_check.py",
        "line": 333
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_help_is_offline_and_describes_the_actual_root_output_interface": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 219
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_required_output_argument_is_not_success": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 230
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_native_run_context_refuses": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 236
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_wrong_repository_context_refuses": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 241
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_context_commit_must_equal_the_actual_selected_commit": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 246
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_invalid_native_run_and_attempt_values_refuse": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 251
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_selected_bytes_must_match_git_even_when_assume_unchanged": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 261
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_changed_dependency_cannot_execute_before_source_binding_refusal": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 276
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_caller_cannot_select_another_checkout_even_with_identical_source": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 288
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_default_root_is_the_executed_checkout_not_callers_cwd": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 295
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_existing_output_is_not_overwritten": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 299
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_inside_the_selected_source_is_refused": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 304
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_symlink_is_refused_without_touching_its_target": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 309
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_parent_symlink_is_refused": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 316
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_token_retains_real_source_and_run_identity_without_live_credit": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 324
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_cli_reports_actual_interpreter_mode_in_each_child": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 340
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_ambient_git_path_redirects_cannot_replace_the_selected_checkout": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 348
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_api_or_transcript_override_is_not_a_supported_cli_input": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 359
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_unbound_package_cannot_replace_either_bound_project_dependency": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 414
    },
    "tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency": {
        "path": "tests/test_node_status_live_cli.py",
        "line": 433
    }
}
_BOOTSTRAP = "import hashlib\nimport importlib\nimport json\nimport os\nfrom pathlib import Path\nimport stat\nimport subprocess\nimport sys\nimport unittest\n\nTARGETS = {\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_complete_real_input_cohort_is_checked_without_rewriting_pins\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 136\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_any_of_the_four_real_record_files_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 155\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_invalid_schema_refuses_before_source_observation\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 169\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_missing_schema_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 178\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_duplicate_record_json_keys_refuse\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 184\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_unexpected_record_file_is_not_ignored\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 192\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_schema_byte_drift_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 198\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_graph_byte_drift_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 205\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_no_declared_fresh_record_is_not_an_empty_success\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 212\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_git_object_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 219\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_changed_cited_discussion_timestamp_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 231\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_unavailable_source_cannot_become_fresh\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 236\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_record_byte_drift_even_with_unchanged_json_value_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 248\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_record_set_drift_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 255\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_success_without_observation_counter_coverage_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 261\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_correct_count_with_wrong_source_observation_urls_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 267\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_without_captured_default_head_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 272\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_observation_head_must_match_the_captured_repository_cut\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 277\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_same_count_and_url_set_with_wrong_multiplicities_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 288\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_retained_discussion_timestamp_must_match_the_captured_source_revision\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 303\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_malformed_run_identity_refuses\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 315\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_refuses_missing_authentication_before_constructing_observer\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 326\n  },\n  \"tests.test_node_status_live_check.NodeStatusLiveContract.test_live_entry_uses_one_default_observer_for_the_whole_mode\": {\n    \"path\": \"tests/test_node_status_live_check.py\",\n    \"line\": 333\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_help_is_offline_and_describes_the_actual_root_output_interface\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 219\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_required_output_argument_is_not_success\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 230\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_native_run_context_refuses\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 236\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_wrong_repository_context_refuses\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 241\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_context_commit_must_equal_the_actual_selected_commit\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 246\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_invalid_native_run_and_attempt_values_refuse\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 251\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_selected_bytes_must_match_git_even_when_assume_unchanged\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 261\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_changed_dependency_cannot_execute_before_source_binding_refusal\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 276\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_caller_cannot_select_another_checkout_even_with_identical_source\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 288\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_default_root_is_the_executed_checkout_not_callers_cwd\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 295\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_existing_output_is_not_overwritten\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 299\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_inside_the_selected_source_is_refused\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 304\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_symlink_is_refused_without_touching_its_target\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 309\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_output_parent_symlink_is_refused\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 316\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_missing_token_retains_real_source_and_run_identity_without_live_credit\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 324\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_cli_reports_actual_interpreter_mode_in_each_child\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 340\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_ambient_git_path_redirects_cannot_replace_the_selected_checkout\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 348\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_api_or_transcript_override_is_not_a_supported_cli_input\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 359\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_unbound_package_cannot_replace_either_bound_project_dependency\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 414\n  },\n  \"tests.test_node_status_live_cli.NodeStatusLiveCLIContract.test_timestamp_valid_bytecode_cannot_replace_either_bound_project_dependency\": {\n    \"path\": \"tests/test_node_status_live_cli.py\",\n    \"line\": 433\n  }\n}\nSELECTED = [\n  {\n    \"source_path\": \"architecture/node-status.schema.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/architecture/node-status.schema.json.txt\",\n    \"materialized_path\": \"architecture/node-status.schema.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"2bad95d30bcb89ea686d1ac5a50d52113dd730be\",\n    \"bytes\": 5018,\n    \"sha256\": \"b82f1aee34097071bc47def5e3d371a3ced2df1332fb29779fe1d56e38676e45\"\n  },\n  {\n    \"source_path\": \"docs/site/current.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/docs/site/current.json.txt\",\n    \"materialized_path\": \"docs/site/current.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"a64fb6d9768ada12b9884adac854e0ccdd5a910e\",\n    \"bytes\": 962639,\n    \"sha256\": \"560d3d4e77369835c832dbb4fd0d12f4988744045b9cbc4c7978beddb6e9d20f\"\n  },\n  {\n    \"source_path\": \"records/node_status/eng.d0-packet-allowlist-fix.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/records/node_status/eng.d0-packet-allowlist-fix.json.txt\",\n    \"materialized_path\": \"records/node_status/eng.d0-packet-allowlist-fix.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"0a7f209b58f4d309f8c8ae36d8dce1696c22ffbd\",\n    \"bytes\": 988,\n    \"sha256\": \"8a5bfbc046147c5bf4a95e8ff912abcdef5f61761583e5654a7327772c72d3b3\"\n  },\n  {\n    \"source_path\": \"records/node_status/eng.hard-gate.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/records/node_status/eng.hard-gate.json.txt\",\n    \"materialized_path\": \"records/node_status/eng.hard-gate.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"9889637191bc01e26a6ad6dfec19b35de8b72a0b\",\n    \"bytes\": 1273,\n    \"sha256\": \"f476f09dcdf9f691c1579d8396590afb02dac86aa9bfb207a7331b872a8e5846\"\n  },\n  {\n    \"source_path\": \"records/node_status/math.lifetime-remainder.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/records/node_status/math.lifetime-remainder.json.txt\",\n    \"materialized_path\": \"records/node_status/math.lifetime-remainder.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"a6f82a36e4f9a31abf656661f7c8ee5201874522\",\n    \"bytes\": 4235,\n    \"sha256\": \"e672df10ee7f6cd75d86969e899b7d00bad23884b147fddf5bdd93b8db295180\"\n  },\n  {\n    \"source_path\": \"records/node_status/math.side24-coefficient.json\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/records/node_status/math.side24-coefficient.json.txt\",\n    \"materialized_path\": \"records/node_status/math.side24-coefficient.json\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"ecf9c42723ee72b8e7d1a4d5b36470f158d6af0e\",\n    \"bytes\": 938,\n    \"sha256\": \"0cc919fde0c9f57cdaab45d10c6eec8fc7d8d417bc33bafcd00cc53c7b6b2be7\"\n  },\n  {\n    \"source_path\": \"tools/node_status_live_check.py\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/tools/node_status_live_check.py.txt\",\n    \"materialized_path\": \"tools/node_status_live_check.py\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"1e4852188f15060176b4458f58b27815f56845fd\",\n    \"bytes\": 21374,\n    \"sha256\": \"85bb8ca6821d376655a629447940bdfa057a3567eaa72f734a4d7261ddd9a5b5\"\n  },\n  {\n    \"source_path\": \"tools/node_status_refresh.py\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/tools/node_status_refresh.py.txt\",\n    \"materialized_path\": \"tools/node_status_refresh.py\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"5fa7c8b6647938a0aca538a7735edede65de9ed5\",\n    \"bytes\": 23238,\n    \"sha256\": \"ed018809b290046f9155487c41bbe7ba594dc4fba7941256b7ebb5abff09cac0\"\n  },\n  {\n    \"source_path\": \"tools/validate_node_status.py\",\n    \"payload_path\": \"tests/fixtures/node_status_live_v1/payload/tools/validate_node_status.py.txt\",\n    \"materialized_path\": \"tools/validate_node_status.py\",\n    \"git_mode\": \"100644\",\n    \"git_blob\": \"e9d44b7620536b389a35ff6fad8aef2b0464b5d2\",\n    \"bytes\": 16375,\n    \"sha256\": \"9ccadda867e00617b42177e8aee067c64bd6e52fbc79ec8fddc82653dfbc6375\"\n  }\n]\nroot = Path(sys.argv[1])\ntarget = sys.argv[2]\nevidence = Path(sys.argv[3])\nstate = {\"schema\": \"node-status-legacy-v1.inner.v1\", \"target\": target,\n         \"status\": \"STARTED\", \"python_optimize\": sys.flags.optimize,\n         \"scientific_effect\": \"NONE\", \"http_credit\": False,\n         \"current_v2_qualified\": False, \"program_captures\": []}\n(evidence / \"inner-outcome.json\").write_text(json.dumps(state, sort_keys=True) + \"\\n\")\nspec = TARGETS[target]\noriginal_file = root / spec[\"path\"]\n\ndef deny_network(event, arguments):\n    if event.startswith(\"socket.\") or event.startswith(\"urllib.\"):\n        raise RuntimeError(\"LEGACY_FIXTURE_NETWORK_REFUSED\")\n\nsys.addaudithook(deny_network)\nsys.path[:] = [str(root), str(root / \"tools\")] + [\n    value for value in sys.path if value and not Path(value).is_relative_to(root)]\nsys.dont_write_bytecode = True\n\ndef git(repo, *args):\n    environment = {\"PATH\": os.environ.get(\"PATH\", os.defpath), \"LANG\": \"C.UTF-8\",\n                   \"LC_ALL\": \"C.UTF-8\", \"GIT_CONFIG_NOSYSTEM\": \"1\",\n                   \"GIT_CONFIG_GLOBAL\": os.devnull, \"GIT_OPTIONAL_LOCKS\": \"0\",\n                   \"GIT_TERMINAL_PROMPT\": \"0\", \"GIT_NO_REPLACE_OBJECTS\": \"1\",\n                   \"GIT_NO_LAZY_FETCH\": \"1\"}\n    result = subprocess.run(\n        [\"git\", \"--no-replace-objects\", \"--literal-pathspecs\", \"-c\",\n         \"core.fsmonitor=false\", \"-C\", str(repo), *args],\n        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,\n        timeout=30, check=False)\n    if result.returncode:\n        raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n    return result.stdout\n\ndef program_snapshot(tool):\n    repo = tool.parent.parent\n    commit = git(repo, \"rev-parse\", \"--verify\", \"HEAD^{commit}\").decode(\"ascii\").strip()\n    tree = git(repo, \"rev-parse\", \"--verify\", \"HEAD^{tree}\").decode(\"ascii\").strip()\n    paths = [row[\"source_path\"] for row in SELECTED]\n    raw = git(repo, \"ls-tree\", \"-z\", \"-r\", \"--full-tree\", commit, \"--\", *paths)\n    entries = {}\n    for entry in raw.split(b\"\\0\"):\n        if not entry:\n            continue\n        metadata, name = entry.split(b\"\\t\", 1)\n        mode, kind, oid = metadata.decode(\"ascii\").split()\n        name = name.decode(\"utf-8\")\n        if name in entries or kind != \"blob\":\n            raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n        entries[name] = (mode, oid)\n    if set(entries) != set(paths):\n        raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n    bindings = {}\n    for row in SELECTED:\n        path = repo / row[\"source_path\"]\n        if entries[row[\"source_path\"]] != (row[\"git_mode\"], row[\"git_blob\"]):\n            raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n        before = path.stat()\n        if path.is_symlink() or not stat.S_ISREG(before.st_mode):\n            raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n        data = path.read_bytes()\n        after = path.stat()\n        fields = (\"st_dev\", \"st_ino\", \"st_mode\", \"st_size\", \"st_mtime_ns\", \"st_ctime_ns\")\n        if tuple(getattr(before, key) for key in fields) != tuple(getattr(after, key) for key in fields):\n            raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n        working_blob = hashlib.sha1(\n            b\"blob \" + str(len(data)).encode(\"ascii\") + b\"\\0\" + data).hexdigest()\n        bindings[row[\"source_path\"]] = {\n            \"git_blob\": row[\"git_blob\"], \"git_mode\": row[\"git_mode\"],\n            \"working_blob\": working_blob, \"working_bytes\": len(data),\n            \"working_sha256\": hashlib.sha256(data).hexdigest(),\n            \"working_equals_committed\": working_blob == row[\"git_blob\"]}\n    if git(repo, \"rev-parse\", \"--verify\", \"HEAD^{commit}\").decode(\"ascii\").strip() != commit:\n        raise ValueError(\"LEGACY_PROGRAM_GIT_CAPTURE_FAILED\")\n    return {\"program_root\": str(repo), \"checked_commit\": commit,\n            \"checked_tree\": tree, \"selected_inputs\": bindings,\n            \"scope\": \"OWNED_GIT_NINE_INPUT_FIXTURE_ONLY\"}\n\ndef capture_program(frame, event, returned):\n    if event != \"return\" or not isinstance(returned, subprocess.CompletedProcess):\n        return\n    caller = frame.f_back\n    if caller is None or caller.f_code.co_filename != str(original_file):\n        return\n    if caller.f_code.co_name not in (\"invoke\", \"_invoke_import_probe\"):\n        return\n    case = caller.f_locals.get(\"self\")\n    tool = getattr(case, \"tool\", None)\n    if not isinstance(tool, Path):\n        state[\"program_capture_error\"] = \"MISSING_FIXED_PROGRAM_TOOL\"\n        return\n    try:\n        if not isinstance(returned.stdout, bytes) or not isinstance(returned.stderr, bytes):\n            raise ValueError(\"LEGACY_PROGRAM_STREAMS_NOT_ORIGINAL_BYTES\")\n        if max(len(returned.stdout), len(returned.stderr)) > 4 * 1024 * 1024:\n            raise ValueError(\"LEGACY_PROGRAM_STREAM_CEILING\")\n        number = len(state[\"program_captures\"])\n        directory = evidence / (\"program-\" + str(number).zfill(3))\n        directory.mkdir()\n        (directory / \"stdout.bin\").write_bytes(returned.stdout)\n        (directory / \"stderr.bin\").write_bytes(returned.stderr)\n        record = {\n            \"command\": returned.args, \"returncode\": returned.returncode,\n            \"stdout\": {\"bytes\": len(returned.stdout),\n                       \"sha256\": hashlib.sha256(returned.stdout).hexdigest()},\n            \"stderr\": {\"bytes\": len(returned.stderr),\n                       \"sha256\": hashlib.sha256(returned.stderr).hexdigest()},\n            \"caller\": caller.f_code.co_name, \"source_file\": str(original_file),\n            \"source_line\": caller.f_lineno, \"http_credit\": False,\n            \"current_v2_qualified\": False, \"scientific_effect\": \"NONE\",\n            \"owned_git\": program_snapshot(tool)}\n        (directory / \"observation.json\").write_text(json.dumps(record, sort_keys=True) + \"\\n\")\n        state[\"program_captures\"].append(str(directory.name))\n    except Exception as error:\n        state[\"program_capture_error\"] = type(error).__name__\n\ncode = 1\ntry:\n    module_name, class_name, method_name = target.rsplit(\".\", 2)\n    module = importlib.import_module(module_name)\n    if Path(module.__file__) != original_file:\n        raise ValueError(\"LEGACY_INNER_MODULE_ORIGIN_MISMATCH\")\n    original_class = getattr(module, class_name)\n    original_method = getattr(original_class, method_name)\n    if (original_method.__code__.co_filename != str(original_file)\n            or original_method.__code__.co_firstlineno != spec[\"line\"]\n            or original_method.__module__ != module_name):\n        raise ValueError(\"LEGACY_INNER_METHOD_ORIGIN_MISMATCH\")\n    suite = unittest.defaultTestLoader.loadTestsFromName(target)\n    if suite.countTestCases() != 1:\n        raise ValueError(\"LEGACY_INNER_POPULATION_MISMATCH\")\n    observed = []\n    def walk(item):\n        if isinstance(item, unittest.TestSuite):\n            for child in item:\n                walk(child)\n        else:\n            observed.append(item.id())\n    walk(suite)\n    if observed != [target]:\n        raise ValueError(\"LEGACY_INNER_ID_MISMATCH\")\n    state.update(observed_ids=observed, original_source_file=str(original_file),\n                 original_method_first_line=spec[\"line\"])\n    sys.setprofile(capture_program)\n    try:\n        result = unittest.TextTestRunner(verbosity=2, descriptions=False).run(suite)\n    finally:\n        sys.setprofile(None)\n    state.update(status=\"COMPLETE\", tests_run=result.testsRun,\n                 failures=len(result.failures), errors=len(result.errors),\n                 skipped=len(result.skipped),\n                 expected_failures=len(result.expectedFailures),\n                 unexpected_successes=len(result.unexpectedSuccesses))\n    clean = (result.wasSuccessful() and result.testsRun == 1\n             and not result.skipped and not result.expectedFailures\n             and not result.unexpectedSuccesses\n             and \"program_capture_error\" not in state)\n    code = 0 if clean else 1\nexcept BaseException as error:\n    sys.setprofile(None)\n    state.update(status=\"HOLD\", exception_type=type(error).__name__)\nfinally:\n    state[\"exit_code\"] = code\n    (evidence / \"inner-outcome.json\").write_text(json.dumps(state, sort_keys=True) + \"\\n\")\nraise SystemExit(code)\n"
_SOURCE_PATHS = tuple(sorted((_BUNDLE_PATH, _RUNNER_PATH, *_WRAPPERS,
                             *(row["payload_path"] for row in _EXPECTED["members"]))))

def _require(condition, code="LEGACY_BUNDLE_MISMATCH"):
    if not condition:
        raise ValueError(code)

def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def _blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _read(root, relative, limit=_LIMIT):
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        _require(parts and all(part not in ("", ".", "..") for part in parts))
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                  dir_fd=descriptor)
        with os.fdopen(file_descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit)
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
                                 item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        _require(len(raw) == before.st_size <= limit and identity(before) == identity(after))
        return raw, ("100755" if before.st_mode & stat.S_IXUSR else "100644")
    finally:
        os.close(descriptor)

def _git(*arguments):
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8",
                   "LC_ALL": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1",
                   "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0",
                   "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1",
                   "GIT_NO_LAZY_FETCH": "1"}
    result = subprocess.run(
        ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
         "core.fsmonitor=false", "-C", str(_ROOT), *arguments],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False)
    _require(result.returncode == 0)
    return result.stdout

def _capture():
    _require(_ROOT.is_absolute() and _ROOT.resolve(strict=True) == _ROOT)
    for parent in (_ROOT, *_ROOT.parents):
        _require(not parent.is_symlink() and parent.is_dir())
    head = _git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    tree = _git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", head) is not None
             and re.fullmatch(r"[0-9a-f]{40}", tree) is not None)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        _require(os.environ.get("GITHUB_REPOSITORY") == _EXPECTED["source_repository"]
                 and os.environ.get("GITHUB_SHA") == head)
    bindings, bodies = {}, {}
    for path in _SOURCE_PATHS:
        entry = _git("ls-tree", "--full-tree", "-z", head, "--", path)
        items = entry.split(b"\0")
        _require(len(items) == 2 and items[-1] == b"")
        metadata, name = items[0].split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        _require(name == path.encode("utf-8") and kind == "blob"
                 and mode in ("100644", "100755"))
        raw, actual_mode = _read(_ROOT, path)
        committed = _git("cat-file", "blob", oid)
        _require(raw == committed and _blob(raw) == oid and actual_mode == mode)
        bodies[path] = raw
        bindings[path] = {"git_blob": oid, "git_mode": mode,
                          "bytes": len(raw), "sha256": _sha(raw)}
    _require(_git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() == head
             and _git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip() == tree)
    return {"checked_commit": head, "checked_tree": tree,
            "source_bindings": bindings}, bodies

def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("LEGACY_BUNDLE_MISMATCH")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=constant, parse_float=constant)

def verify_bundle() -> dict:
    """Verify fixed tracked originals; the returned record authenticates no HTTP."""
    try:
        captured, bodies = _capture()
        manifest_raw = bodies[_BUNDLE_PATH]
        _require(len(manifest_raw) <= 65536 and _parse(manifest_raw) == _EXPECTED)
        for row in _EXPECTED["members"]:
            raw = bodies[row["payload_path"]]
            _require(len(raw) == row["bytes"] and _sha(raw) == row["sha256"]
                     and _blob(raw) == row["git_blob"]
                     and captured["source_bindings"][row["payload_path"]]["git_mode"] == row["git_mode"])
        result = dict(schema="node-status-legacy-v1.bundle-check.v1",
                      source_repository=_EXPECTED["source_repository"],
                      source_commit=_EXPECTED["source_commit"],
                      source_tree=_EXPECTED["source_tree"],
                      members=json.loads(json.dumps(_EXPECTED["members"])),
                      helper_ids=list(HELPER_IDS), cli_ids=list(CLI_IDS), **captured)
        return result
    except (OSError, ValueError, TypeError, UnicodeError, KeyError, RecursionError,
            subprocess.SubprocessError):
        raise ValueError("LEGACY_BUNDLE_MISMATCH") from None

def _destination(destination):
    path = Path(destination)
    _require(path.is_absolute() and path.resolve(strict=True) == path
             and path.is_dir() and not path.is_relative_to(_ROOT),
             "LEGACY_DESTINATION_REFUSED")
    for parent in (path, *path.parents):
        _require(not parent.is_symlink() and parent.is_dir(), "LEGACY_DESTINATION_REFUSED")
    return path

def verify_materialized(destination: Path) -> dict:
    """Check every copied original; this is fixture identity, not current authority."""
    try:
        path = _destination(destination)
        for row in _EXPECTED["members"]:
            raw, mode = _read(path, row["materialized_path"])
            _require(mode == row["git_mode"] and len(raw) == row["bytes"]
                     and _sha(raw) == row["sha256"] and _blob(raw) == row["git_blob"])
        return {"schema": "node-status-legacy-v1.materialized.v1",
                "members": json.loads(json.dumps(_EXPECTED["members"]))}
    except (OSError, ValueError, TypeError, UnicodeError):
        raise ValueError("LEGACY_BUNDLE_MISMATCH") from None

def materialize_bundle(destination: Path) -> dict:
    """Copy only fixed originals to an empty owned external fixture directory."""
    try:
        path = _destination(destination)
        _require(not any(path.iterdir()), "LEGACY_DESTINATION_REFUSED")
        before = verify_bundle()
        for row in _EXPECTED["members"]:
            raw, _ = _read(_ROOT, row["payload_path"])
            target = path / row["materialized_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o644)
            with os.fdopen(descriptor, "wb") as stream:
                _require(stream.write(raw) == len(raw))
            target.chmod(0o644)
        result = verify_materialized(path)
        _require(verify_bundle() == before)
        return result
    except (OSError, TypeError, UnicodeError):
        raise ValueError("LEGACY_DESTINATION_REFUSED") from None

def _temporary_parent():
    value = os.environ.get("RUNNER_TEMP")
    return _destination(Path("/tmp") if value is None else Path(value))

def _evidence_root():
    value = os.environ.get("NODE_STATUS_LEGACY_V1_EVIDENCE")
    if value:
        path = Path(value)
        _require(path.is_absolute() and not path.is_relative_to(_ROOT),
                 "LEGACY_EVIDENCE_REFUSED")
        if os.path.lexists(path):
            return _destination(path)
        _destination(path.parent)
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
        return _destination(path)
    parent = _temporary_parent()
    path = Path(tempfile.mkdtemp(prefix="node-status-legacy-v1-evidence-", dir=parent))
    return _destination(path)

def _retain_shared(root, name, raw):
    """Retain one complete immutable runtime original, never a prefix."""
    _destination(root)
    _require(name == "bootstrap.py" or re.fullmatch(r"source-[0-9a-f]{64}\.json", name),
             "LEGACY_SHARED_REFUSED")
    shared = root / "_shared"
    if os.path.lexists(shared):
        _destination(shared)
    else:
        shared.mkdir(mode=0o700, exist_ok=False)
        _destination(shared)
    relative = "_shared/" + name
    path = root / relative
    if not os.path.lexists(path):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600)
        with os.fdopen(descriptor, "wb") as stream:
            _require(stream.write(raw) == len(raw))
    retained, _ = _read(root, relative)
    _require(retained == raw)
    return {"path": relative, "bytes": len(raw), "sha256": _sha(raw)}

def _source_reference(root, capture):
    raw = _encode(capture)
    reference = _retain_shared(root, "source-" + _sha(raw) + ".json", raw)
    return dict(reference, observed_at=_utc())

def _environment():
    # Only harmless execution settings enter this fixture child. No actual token,
    # native run identity, PYTHONPATH or Git selection/credential override is copied.
    result = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ") if key in os.environ}
    result.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    return result

def _group_absent(group):
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    return False

def _supervise(command, root, evidence):
    # No new per-case budget is invented. Existing outer workflow job limits and
    # unchanged original Git/CLI30s subcall limits remain the runtime boundary.
    record = {"command": command, "cwd": str(root), "started_at": _utc(),
              "timeout_policy": "EXISTING_OUTER_WORKFLOW_JOB_LIMIT_ONLY",
              "timeout_seconds": None, "timed_out": False, "status": "STARTED",
              "stdout_complete": False, "stderr_complete": False}
    (evidence / "supervisor.json").write_bytes(_encode(record))
    streams = {name: (evidence / (name + ".bin")).open("xb") for name in ("stdout", "stderr")}
    counts = {"stdout": 0, "stderr": 0}
    selector = selectors.DefaultSelector()
    process = None
    started = time.monotonic()
    failure = None
    try:
        process = subprocess.Popen(command, cwd=root, env=_environment(),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        record.update(pid=process.pid, pgid=process.pid)
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            for key, _ in selector.select(0.5):
                name = key.data
                raw = os.read(key.fileobj.fileno(),
                              min(65536, _STREAM_LIMIT + 1 - counts[name]))
                if not raw:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    record[name + "_complete"] = True
                else:
                    streams[name].write(raw)
                    counts[name] += len(raw)
                    if counts[name] > _STREAM_LIMIT:
                        raise ValueError("LEGACY_ORIGINAL_STREAM_CEILING")
        record["returncode"] = process.wait(timeout=5)
        record["status"] = "COMPLETE"
    except BaseException as error:
        failure = type(error).__name__
        record.update(status="HOLD", exception_type=failure)
    finally:
        if process is not None:
            if process.poll() is None or not _group_absent(process.pid):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                record["unexpected_survivor_or_interruption"] = True
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                record["cleanup_error"] = "LEADER_NOT_REAPED"
            record["returncode"] = process.returncode
            record["group_absent"] = _group_absent(process.pid)
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        selector.close()
        for stream in streams.values():
            stream.close()
        for name in ("stdout", "stderr"):
            raw = (evidence / (name + ".bin")).read_bytes()
            record[name] = {"bytes": len(raw), "sha256": _sha(raw)}
        record.update(finished_at=_utc(), elapsed_seconds=time.monotonic() - started)
        (evidence / "supervisor.json").write_bytes(_encode(record))
    return record

def run_legacy_case(case, target: str) -> None:
    """Outer unittest harness only: run its one fixed unchanged original case."""
    _require(target in _CASES and target.rsplit(".", 1)[1] == case._testMethodName,
             "LEGACY_TARGET_REFUSED")
    expected_class = target.rsplit(".", 2)[1]
    _require(type(case).__name__ == expected_class, "LEGACY_TARGET_REFUSED")
    before = verify_bundle()
    evidence_root = _evidence_root()
    evidence = evidence_root / (target + "-opt" + str(sys.flags.optimize))
    evidence.mkdir(mode=0o700, exist_ok=False)
    bootstrap_raw = _BOOTSTRAP.encode("ascii")
    bootstrap_reference = _retain_shared(evidence_root, "bootstrap.py", bootstrap_raw)
    before_reference = _source_reference(evidence_root, before)
    receipt = {"schema": "node-status-legacy-v1.case.v1",
               "scope": "HISTORICAL_NODE_STATUS_LIVE_V1_FIXTURE_CONTRACT_ONLY",
               "target": target, "outer_id": case.id(),
               "source_cut": _EXPECTED["source_commit"],
               "original_program_schema": "cbz.node-status-live-cli.v1",
               "current_v2_qualified": False, "http_credit": False,
               "scientific_effect": "NONE", "source_before": before_reference,
               "bootstrap": bootstrap_reference, "status": "STARTED"}
    (evidence / "observation.json").write_bytes(_encode(receipt))
    failure = None
    supervisor = None
    try:
        with tempfile.TemporaryDirectory(prefix="node-status-legacy-v1-case-",
                                         dir=_temporary_parent()) as temporary:
            root = Path(temporary)
            materialize_bundle(root)
            (root / "tests/__init__.py").write_bytes(b"")
            command = [sys.executable, "-B", *(["-O"] if sys.flags.optimize else []),
                       "-S", str(evidence_root / bootstrap_reference["path"]),
                       str(root), target, str(evidence)]
            supervisor = _supervise(command, root, evidence)
            receipt["supervisor"] = supervisor
            _require(_retain_shared(evidence_root, "bootstrap.py", bootstrap_raw)
                     == bootstrap_reference)
            receipt["materialized_after"] = verify_materialized(root)
            raw = (evidence / "inner-outcome.json").read_bytes()
            _require(len(raw) <= 65536)
            inner = _parse(raw)
            receipt["inner"] = inner
            captures = inner.get("program_captures")
            _require(type(captures) is list and len(set(captures)) == len(captures))
            if target in CLI_IDS:
                _require(bool(captures))
            expected_programs = {
                row["source_path"]: row for row in _EXPECTED["members"]
                if not row["source_path"].startswith("tests/")}
            for number, name in enumerate(captures):
                _require(name == "program-" + str(number).zfill(3))
                program_root = evidence / name
                program_raw = (program_root / "observation.json").read_bytes()
                _require(len(program_raw) <= 65536)
                program = _parse(program_raw)
                bindings = program["owned_git"]["selected_inputs"]
                _require(set(bindings) == set(expected_programs))
                for path, specification in expected_programs.items():
                    _require(bindings[path]["git_blob"] == specification["git_blob"]
                             and bindings[path]["git_mode"] == specification["git_mode"])
                for label in ("stdout", "stderr"):
                    original = (program_root / (label + ".bin")).read_bytes()
                    _require(len(original) <= _STREAM_LIMIT
                             and program[label]["bytes"] == len(original)
                             and program[label]["sha256"] == _sha(original))
            stderr = (evidence / "stderr.bin").read_text(encoding="utf-8")
            method = target.rsplit(".", 1)[1]
            expected_line = method + " (" + target + ") ... ok"
            outcomes = [line for line in stderr.splitlines() if re.search(r" \.\.\. ", line)]
            _require(outcomes == [expected_line])
            _require(len(re.findall(r"^Ran 1 test in [0-9.]+s$", stderr, re.MULTILINE)) == 1
                     and stderr.rstrip().endswith("\nOK"))
            _require(inner.get("target") == target and inner.get("observed_ids") == [target]
                     and inner.get("status") == "COMPLETE" and inner.get("tests_run") == 1
                     and all(inner.get(key) == 0 for key in
                             ("failures", "errors", "skipped", "expected_failures",
                              "unexpected_successes", "exit_code"))
                     and inner.get("python_optimize") == sys.flags.optimize
                     and "program_capture_error" not in inner)
            _require(supervisor["status"] == "COMPLETE" and supervisor["returncode"] == 0
                     and supervisor["group_absent"]
                     and supervisor["stdout_complete"] and supervisor["stderr_complete"]
                     and not supervisor.get("unexpected_survivor_or_interruption")
                     and "cleanup_error" not in supervisor)
            receipt["status"] = "PASS"
    except BaseException as error:
        failure = type(error).__name__
        receipt.update(status="HOLD", exception_type=failure)
    finally:
        try:
            after = verify_bundle()
            receipt["source_after"] = _source_reference(evidence_root, after)
            if after != before:
                failure = "SOURCE_CHANGED"
                receipt.update(status="HOLD", exception_type=failure)
        except Exception as error:
            failure = type(error).__name__
            receipt.update(status="HOLD", source_recheck_error=failure)
        (evidence / "observation.json").write_bytes(_encode(receipt))
    case.assertIsNone(failure, "retained legacy fixture evidence: " + str(evidence))
    case.assertEqual(receipt["status"], "PASS", str(evidence))
