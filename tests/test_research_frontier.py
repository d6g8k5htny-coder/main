"""Mutation controls for the bounded derived view; no authority or review credit.

Every mutation below changes a temporary repository or an in-memory checkpoint.
The committed graph, register exports and historical records are never edited.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import research_frontier as F  # noqa: E402


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Existing trial records bind field order, so retain their insertion order.
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    graph = {
        "_comment": "Synthetic mutation fixture; no scientific claim.",
        "as_of": "fixture", "tracks": {"TEST": "synthetic"}, "firewalls": [],
        "premises": {"P": {"track": "TEST", "source": "synthetic fixture",
                            "status_frozen_v2_2": "OPEN", "status_register_note": "CLOSED"}},
        "claims": {
            "A": {"track": "TEST", "source": "synthetic fixture", "grade": "CONDITIONAL",
                  "depends_on": ["B", "C"]},
            "B": {"track": "TEST", "source": "synthetic fixture", "grade": "CONDITIONAL",
                  "depends_on": ["P"]},
            "C": {"track": "TEST", "source": "synthetic fixture", "grade": "CONDITIONAL",
                  "sub_obligations": ["P"]},
        },
    }
    write_json(root / F.GRAPH, graph)
    (root / F.CATALOG).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / F.CATALOG, root / F.CATALOG)
    for command in (("init", "-q"), ("add", "."),
                    ("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "-qm", "Synthetic frontier fixture")):
        subprocess.run(["git", "-C", str(root), *command], check=True, capture_output=True)
    return root


def change_graph(repo, mutation):
    graph = F.strict_json((repo / F.GRAPH).read_bytes())
    mutation(graph)
    write_json(repo / F.GRAPH, graph)


def rehash(checkpoint):
    checkpoint["payload_sha256"] = F.digest(F.canonical(checkpoint["payload"]))
    return checkpoint


def test_layer_separation_typed_diamond_paths_and_no_invented_refusal(repo):
    checkpoint = F.snapshot(repo)
    payload = F.check(checkpoint, repo)
    paths = [p for p in payload["unresolved_dependency_paths"] if p["nodes"][0] == "A"]
    assert [(p["nodes"], p["edge_types"]) for p in paths] == [
        (["A", "B", "P"], ["depends_on", "depends_on"]),
        (["A", "C", "P"], ["depends_on", "sub_obligations"]),
    ]
    assert {p["layer"] for p in paths} == {"status_frozen_v2_2"}
    assert payload["nodes"]["P"]["status_register_note"] == "CLOSED"
    assert payload["recorded_diagnostics"] == []
    assert payload["authority"] == "NONE" and payload["independence_credit"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda g: g["claims"]["A"].update(depends_on=["MISSING"]), "missing graph reference"),
    (lambda g: g["claims"]["B"].update(depends_on=["A"]), "cycle"),
    (lambda g: g["claims"]["A"].update(depends_on=["B", "B"]), "repeated"),
    (lambda g: g["claims"]["A"].update(depends_on_any=["B", "C"]), "unsupported dependency"),
    (lambda g: g["premises"]["P"].update(status_register_note=True), "invalid"),
    (lambda g: g["claims"]["A"].update(track="MISSING"), "unknown track"),
])
def test_graph_mutations_refuse_instead_of_partial_view(repo, mutation, match):
    change_graph(repo, mutation)
    with pytest.raises(F.Refused, match=match):
        F.snapshot(repo)


@pytest.mark.parametrize("data,match", [
    (b'{"claims":{},"claims":{}}', "duplicate JSON key"),
    (b'{"x":{"a":1,"a":2}}', "duplicate JSON key"),
    (b'{"x":NaN}', "non-finite"),
])
def test_ambiguous_json_is_refused(data, match):
    with pytest.raises(F.Refused, match=match):
        F.strict_json(data)


def test_source_bytes_and_derived_values_both_bound(repo):
    checkpoint = F.snapshot(repo)
    changed = copy.deepcopy(checkpoint)
    changed["payload"]["nodes"]["P"]["source"] = "forged citation"
    with pytest.raises(F.Refused, match="hash mismatch"):
        F.check(changed, repo)
    rehash(changed)
    with pytest.raises(F.Refused, match="altered checkpoint field: nodes"):
        F.check(changed, repo)
    # A whitespace-only edit leaves the graph meaning unchanged, but its exact
    # input identity must change. A semantic comparison alone would miss this.
    with (repo / F.GRAPH).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(F.Refused, match="stale or altered"):
        F.check(checkpoint, repo)
    newer = F.snapshot(repo)
    delta = F.semantic_diff(checkpoint, newer)
    assert delta["known_field_changes"] == {}
    assert "sources" in delta["identity_changes"]


def test_source_membership_is_bound_even_when_no_diagnostic_changes(repo):
    checkpoint = F.snapshot(repo)
    path = next((ROOT / "engine/receipts").glob("*/*.json"))
    target = repo / path.relative_to(ROOT)
    target.parent.mkdir(parents=True)
    shutil.copyfile(path, target)
    assert F.snapshot(repo)["payload"]["recorded_diagnostics"] == []
    with pytest.raises(F.Refused, match="stale or altered"):
        F.check(checkpoint, repo)


def test_refusal_and_not_run_are_observed_records_not_caveat_words(repo):
    source = next((ROOT / "engine/receipts").glob("*/*.json"))
    record = F.strict_json(source.read_bytes())
    record["notes"] = ["total() refuses while a cell is pending", "total() refused: synthetic pending-cell fixture"]
    body = {k: v for k, v in record.items() if k != "body_sha256"}
    record["body_sha256"] = F.R.body_sha256(body)
    target = repo / source.relative_to(ROOT)
    write_json(target, record)
    catalog = F.strict_json((repo / F.CATALOG).read_bytes())
    operation = next(e["operation_id"] for e in catalog["entries"]
                     if e["git_side"]["trial_kind"] == "NOT_MACHINE_CHECKABLE_HERE")
    trial = F.T.build_trial(operation, catalog, F.digest((repo / F.CATALOG).read_bytes()), root=str(repo))
    trial_path = repo / "engine/operations/trials" / (trial["Trial ID"] + ".json")
    write_json(trial_path, trial)
    diagnostics = F.snapshot(repo)["payload"]["recorded_diagnostics"]
    assert {d["outcome"] for d in diagnostics} == {"RECORDED_REFUSAL", "NOT_RUN"}
    refusal = next(d for d in diagnostics if d["outcome"] == "RECORDED_REFUSAL")
    assert refusal["field"] == "notes/1" and refusal["source"] == target.relative_to(repo).as_posix()
    # The test must reject stale catalog identity even when the trial's prose
    # and outcome look plausible.
    trial["Library / catalog SHA-256"] = "0" * 64
    write_json(trial_path, trial)
    with pytest.raises(F.Refused, match="catalog identity"):
        F.snapshot(repo)


def test_failed_receipt_and_invalid_receipt_hash(repo):
    source = next((ROOT / "engine/receipts").glob("*/*.json"))
    record = F.strict_json(source.read_bytes())
    record.update(outcome="FAILED", error="Synthetic test failure", notes=[], results=[])
    record["body_sha256"] = F.R.body_sha256({k: v for k, v in record.items() if k != "body_sha256"})
    target = repo / source.relative_to(ROOT)
    write_json(target, record)
    assert F.snapshot(repo)["payload"]["recorded_diagnostics"][0]["outcome"] == "FAILED"
    record["error"] = "Changed without updating body hash"
    write_json(target, record)
    with pytest.raises(F.Refused, match="body_sha256 mismatch"):
        F.snapshot(repo)


def test_checkpoint_cannot_hide_a_layer_or_add_unknown_fields_even_rehashed(repo):
    checkpoint = F.snapshot(repo)
    checkpoint["payload"]["unresolved_dependency_paths"] = []
    with pytest.raises(F.Refused, match="dependency projection mismatch"):
        F.semantic_diff(rehash(checkpoint), F.snapshot(repo))
    checkpoint = F.snapshot(repo)
    checkpoint["payload"]["admitted"] = True
    with pytest.raises(F.Refused, match="checkpoint fields"):
        F.validate_checkpoint(rehash(checkpoint))
    checkpoint = F.snapshot(repo)
    checkpoint["payload"]["scope"]["exclusions"] = []
    with pytest.raises(F.Refused, match="scope/authority"):
        F.validate_checkpoint(rehash(checkpoint))


def test_output_exclusive_outside_repository_and_symlink_safe(repo, tmp_path):
    checkpoint = F.snapshot(repo)
    outside = tmp_path / "frontier.json"
    F.write_snapshot(checkpoint, outside, repo)
    assert F.check(F.strict_json(outside.read_bytes()), repo)
    with pytest.raises(FileExistsError):
        F.write_snapshot(checkpoint, outside, repo)
    for destination in (repo / F.GRAPH, repo / "new-checkpoint.json"):
        with pytest.raises(F.Refused, match="outside the repository"):
            F.write_snapshot(checkpoint, destination, repo)
    alias = tmp_path / "source-link"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(F.Refused, match="outside the repository"):
        F.write_snapshot(checkpoint, alias / "new.json", repo)


def test_source_symlink_is_refused(repo, tmp_path):
    source = repo / F.GRAPH
    real = tmp_path / "graph.json"
    real.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(real)
    with pytest.raises(F.Refused, match="symlink"):
        F.snapshot(repo)


def test_cli_self_check_and_failure_exit(repo):
    command = [sys.executable, str(ROOT / "tools/research_frontier.py"), "--root", str(repo), "self-check"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and "recorded_diagnostics=0" in result.stdout
    assert '"payload"' not in result.stdout
    change_graph(repo, lambda g: g["claims"]["A"].update(depends_on=["MISSING"]))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "REFUSED" in result.stderr and not result.stdout


def test_known_field_diff_retains_layer_change(repo):
    before = F.snapshot(repo)
    change_graph(repo, lambda g: g["premises"]["P"].update(status_register_note="OPEN"))
    after = F.snapshot(repo)
    delta = F.semantic_diff(before, after)["known_field_changes"]
    assert set(delta["nodes"]) == {"P"}
    paths = delta["unresolved_dependency_paths"]
    assert not paths["removed"]
    assert {path["layer"] for path in paths["added"]} == {"status_register_note"}


def test_cli_snapshot_check_diff_and_overwrite(repo, tmp_path):
    command = [sys.executable, str(ROOT / "tools/research_frontier.py"), "--root", str(repo)]
    out = tmp_path / "checkpoint.json"
    result = subprocess.run([*command, "snapshot", "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0 and "UNSIGNED" in result.stdout
    result = subprocess.run([*command, "check", str(out)], capture_output=True, text=True)
    assert result.returncode == 0
    result = subprocess.run([*command, "diff", str(out), str(out)], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)["known_field_changes"] == {}
    result = subprocess.run([*command, "snapshot", "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 1 and "REFUSED" in result.stderr


def test_committed_inputs_pass_bounded_self_check():
    # No expected count is hard-coded: legitimate append-only records may grow.
    result = F.derive(ROOT)
    assert "D1-v2.2(2)" in result["nodes"]
    assert all(item["sha256"] and item["bytes"] > 0 for item in result["sources"])
