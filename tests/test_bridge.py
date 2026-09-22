"""The bridge records, their validators, the store and the checker, with
negative controls.

Every test below is a NEGATIVE CONTROL unless its name says otherwise: it
takes a record that passes, breaks it in exactly one way, and asserts that the
validator, the store or `tools/bridge_check.py` refuses it. Each control is
paired with the positive case it mutates, because a checker that cannot fail
is decoration. The checker is driven through its CLI flags against temporary
directories, and it is asserted to write nothing.

The properties under test, in the order the task statement puts them:

* no work order can authorize a status change; NOT_VERIFIED is PREPARED-only;
  an order is never its own authority (in any spelling, and never a repository
  path, a file name or a pull request); every governing source carries its
  five identity fields; scope paths are literal (no `..`, no glob, no absolute
  path, casefolded); the bridge, the checkers and the agent instructions are
  never in scope; policy surfaces in scope are refused without both references
  and only RECORDED with them; allowed commands never mutate git or write a
  surface;
* a receipt's status vocabulary and what each status requires; ACCEPTED is
  refused without the Drive record it attributes the acceptance to, and is
  transcribed unverified with it; `scientific_status_change` is always
  UNCHANGED; missing or placeholder evidence is never PASS and never EXECUTED;
  a nonzero exit code is never zero failures; a dirty worktree never passes;
  independence credit is 0 without a record id that names a record;
* the idempotency key changes when any of its four inputs changes;
* a receipt agrees with its order on task, repository, commands and outputs,
  and never reports an execution under a PREPARED-only order;
* the store never overwrites, and writes nowhere in the repository but
  `receipts/`: an identical redelivery writes nothing and a conflicting one is
  held beside the first;
* the checker resolves digests, checks names, refuses examples in the live
  directories, skips nothing silently, reads the bytes as the record, writes
  nothing, and compares against git history as well as git HEAD;
* what the text says about itself: no sentence claims a verification this
  repository cannot do.

Each new control was verified by reverting its fix in a scratch copy and
watching it fail. None of this is mathematics, and this suite does
not establish anything mathematical. A passing suite says the record machinery
behaves as documented; it says nothing about any run, any review or any
claim, and it moves no obligation. The contract these records implement is
PROPOSED / NOT DEPLOYED. `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side),
`OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` and both Pieces of
`D3-LEMMA-RN-UNIF` are OPEN.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.bridge import common as C  # noqa: E402
from engine.bridge import run_receipt as RR  # noqa: E402
from engine.bridge import work_order as WO  # noqa: E402
from tools import bridge_check as BC  # noqa: E402

CHECKER = os.path.join(ROOT, "tools", "bridge_check.py")
BRIDGE = os.path.join(ROOT, "engine", "bridge")
EXAMPLES = os.path.join(BRIDGE, "examples")
EX_ORDER = os.path.join(EXAMPLES, "work_order.example.json")
EX_RECEIPT = os.path.join(EXAMPLES, "run_receipt.example.json")

DRIVE_ID = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcd"
POLICY_ID = "1PolicyChangeAuthorizedByRecordFixture000"
SOURCE_URL = "https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789abcd/view"
REVIEW_ID = "REV-BRIDGE-FIXTURE-0"
COMMIT = "0" * 40
TREE = "1" * 40
SHA = "2" * 64
TS = "2026-09-18T12:00:00.000000Z"
TS_LATER = "2026-09-18T12:05:00.000000Z"
COMMANDS = ["python3 -m pytest -q tests/test_bridge.py", "python3 tools/bridge_check.py"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_path(obj, dotted, value):
    """Deep-set ``obj[a][b]... = value`` for a dotted path; returns obj."""
    cur = obj
    parts = dotted.split(".")
    for k in parts[:-1]:
        cur = cur[k]
    cur[parts[-1]] = value
    return obj


def del_path(obj, dotted):
    cur = obj
    parts = dotted.split(".")
    for k in parts[:-1]:
        cur = cur[k]
    del cur[parts[-1]]
    return obj


def good_order(**over):
    """A RECORD-kind, PREPARED order derived from the example. Passes."""
    o = copy.deepcopy(load(EX_ORDER))
    o["record_kind"] = WO.RECORD_RECORD
    o["task_id"] = "FIXTURE-TASK-001"
    o["scope"]["objective"] = "FIXTURE: a well-formed prepared order that authorizes nothing"
    for k, v in over.items():
        set_path(o, k, v)
    return WO.with_digest(o)


def executable_order(**over):
    """A fully-formed EXECUTABLE order. The positive path of the strict branch."""
    o = good_order()
    o["task_id"] = "FIXTURE-TASK-EXEC"
    o["status"] = WO.STATUS_EXECUTABLE
    o["repository"]["base_commit_sha"] = COMMIT
    o["authorization"].update({
        "approved_principal_ids": ["github-app-installation:0 (fixture)"],
        "source_ref": SOURCE_URL,
        "verification_status": WO.VERIFIED_BY_OWNER_BOUNDARY,
        "authorizing_record_drive_id": DRIVE_ID,
    })
    o["claim"] = {"work_events_event_id": "FIXTURE-EVENT-0", "lease_until_utc": TS}
    o["scope"]["allowed_commands"] = list(COMMANDS)
    for k, v in over.items():
        set_path(o, k, v)
    return WO.with_digest(o)


def good_receipt(order=None, **over):
    """A RECORD-kind NOT_RUN receipt for ``order``. Passes."""
    order = order or good_order()
    r = copy.deepcopy(load(EX_RECEIPT))
    r["record_kind"] = RR.RECORD_RECORD
    r["task_id"] = order["task_id"]
    r["work_order_digest"] = order["work_order_digest"]
    r["run_id"] = "FIXTURE-RUN-0"
    for k, v in over.items():
        set_path(r, k, v)
    return RR.finalize(r)


def executed_receipt(order=None, **over):
    """An EXECUTED receipt with every piece of evidence present, for an
    EXECUTABLE order by default. Passes, and agrees with its order."""
    order = order or executable_order()
    r = good_receipt(order)
    r["status"] = RR.EXECUTED
    r["actor"].update({
        "authenticated_principal_id": "github-app-installation:0 (fixture)",
        "declared_provider": "fixture", "declared_model": "fixture",
        "session_id": "fixture-session", "exposure": "fixture: nothing was read",
    })
    r["execution"].update({
        "commit_sha": COMMIT, "tree_sha": TREE, "dirty_worktree": False,
        "environment_identity": "fixture runner (no real environment)",
        "commands": list(COMMANDS),
        "start_utc": TS, "end_utc": TS_LATER, "exit_codes": [0, 0],
    })
    r["verification"].update({
        "tests_passed": 3, "tests_failed": 0,
        "negative_controls": ["a receipt with tests_passed null cannot PASS"],
        "coverage": "tests/test_bridge.py", "excluded_scope": ["all mathematics"],
    })
    for k, v in over.items():
        set_path(r, k, v)
    return RR.finalize(r)


def rehash_order(o):
    return WO.with_digest(o)


def rehash_receipt(r):
    return RR.finalize(r)


def rehash_receipt_body(r):
    """Recompute body_sha256 only, as a forger would, leaving the key alone."""
    body = RR.canonical_body(r)
    return {**body, "body_sha256": RR.receipt_body_digest(body)}


def write(path, obj, text=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else RR.to_json(obj))
    return path


def workspace(tmp_path, orders=(), receipts=(), examples=None):
    """Write orders (named by task_id) and receipts (named by key) to tmp dirs."""
    od, rd = tmp_path / "orders", tmp_path / "receipts"
    od.mkdir(exist_ok=True)
    rd.mkdir(exist_ok=True)
    for o in orders:
        write(str(od / f"{o['task_id']}.json"), o)
    for r in receipts:
        write(str(rd / f"{r['idempotency_key']}.json"), r)
    ed = None
    if examples is not None:
        ed = tmp_path / "examples"
        ed.mkdir(exist_ok=True)
        for i, e in enumerate(examples):
            write(str(ed / f"example-{i}.json"), e)
    return str(od), str(rd), (str(ed) if ed else "")


def run_checker(orders, receipts, examples="", extra=()):
    return subprocess.run(
        [sys.executable, CHECKER, "--orders", orders, "--receipts", receipts,
         "--examples", examples, "--no-git", *extra],
        capture_output=True, text=True)


def snapshot(directory):
    out = {}
    for dirpath, _dirs, files in os.walk(directory):
        for fn in files:
            p = os.path.join(dirpath, fn)
            with open(p, "rb") as f:
                out[os.path.relpath(p, directory)] = hashlib.sha256(f.read()).hexdigest()
    return out


def git_repo(tmp_path):
    """A scratch git repository laid out like the bridge, with the checker's
    directories inside it. Returns ``(repo, orders_dir, receipts_dir, examples_dir)``."""
    repo = tmp_path / "repo"
    odir = repo / "engine" / "bridge" / "orders"
    rdir = repo / "engine" / "bridge" / "receipts"
    edir = repo / "engine" / "bridge" / "examples"
    for d in (odir, rdir, edir):
        d.mkdir(parents=True)
    for cmd in (["git", "init", "-q"],
                ["git", "config", "user.email", "t@example.invalid"],
                ["git", "config", "user.name", "t"]):
        assert subprocess.run(cmd, cwd=repo, capture_output=True).returncode == 0
    return repo, str(odir), str(rdir), str(edir)


def git_commit(repo, message):
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    out = subprocess.run(["git", "commit", "-qm", message], cwd=repo, capture_output=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# 1. the positive cases
# ---------------------------------------------------------------------------

def test_example_order_validates_and_says_it_is_an_example():
    o = load(EX_ORDER)
    assert WO.validate_work_order(o) == []
    assert o["record_kind"] == "EXAMPLE"
    assert o["status"] == "PREPARED"
    assert o["authorization"]["verification_status"] == "NOT_VERIFIED"
    assert o["scientific_status_change_authorized"] is False
    assert WO.work_order_digest(o) == o["work_order_digest"]


def test_example_receipt_validates_and_resolves_to_the_example_order():
    o, r = load(EX_ORDER), load(EX_RECEIPT)
    assert RR.validate_run_receipt(r) == []
    assert r["record_kind"] == "EXAMPLE" and r["status"] == "NOT_RUN"
    assert r["scientific_status_change"] == "UNCHANGED"
    assert r["review"]["independence_credit"] == 0
    assert r["work_order_digest"] == o["work_order_digest"]
    assert RR.check_receipt_against_order(r, o) == []
    assert r["idempotency_key"] == RR.idempotency_key(
        r["task_id"], r["work_order_digest"], None, r["run_id"])
    assert RR.receipt_body_digest(r) == r["body_sha256"]


def test_repository_checker_passes_with_empty_orders_and_receipts():
    """The committed state: empty live directories, two examples, no problems."""
    for d in ("orders", "receipts"):
        names = sorted(os.listdir(os.path.join(BRIDGE, d)))
        assert names == ["README.md"], names
    out = subprocess.run([sys.executable, CHECKER], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "orders=0 receipts=0 held=0 example_orders=1 example_receipts=1" in out.stdout
    assert "problems=0" in out.stdout


def test_a_record_order_and_receipt_pass_the_checker(tmp_path):
    o, e = good_order(), executable_order()
    od, rd, _ = workspace(tmp_path, [o, e], [good_receipt(o), executed_receipt(e)])
    out = run_checker(od, rd)
    assert out.returncode == 0, out.stdout
    assert "orders=2 receipts=2 held=0" in out.stdout and "problems=0" in out.stdout


def test_an_executable_verified_order_passes_and_is_recorded_not_granted():
    o = executable_order()
    assert WO.validate_work_order(o) == []
    notes = WO.work_order_notes(o)
    assert any("transcribed" in n and DRIVE_ID in n for n in notes), notes
    assert any("did not verify" in n for n in notes)


def test_an_executed_receipt_with_all_evidence_passes():
    r = executed_receipt()
    assert RR.validate_run_receipt(r) == []
    assert RR.check_receipt_against_order(r, executable_order()) == []


def test_pass_technical_with_evidence_passes():
    r = executed_receipt(**{"review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "fixture exposure"})
    assert RR.validate_run_receipt(r) == []


# ---------------------------------------------------------------------------
# 2. no work order can authorize a status change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [True, "false", None, 0, 1, "no"])
def test_order_refuses_any_scientific_status_change_authorized_but_false(value):
    o = good_order(scientific_status_change_authorized=value)
    problems = WO.validate_work_order(o)
    assert any("scientific_status_change_authorized" in p and "only permitted value is false" in p
               for p in problems), problems


def test_order_refuses_a_missing_scientific_status_change_authorized():
    o = rehash_order(del_path(good_order(), "scientific_status_change_authorized"))
    assert any("scientific_status_change_authorized is missing" in p
               for p in WO.validate_work_order(o))


# ---------------------------------------------------------------------------
# 3. NOT_VERIFIED is PREPARED-only
# ---------------------------------------------------------------------------

def test_not_verified_order_cannot_be_marked_executable():
    o = executable_order(**{"authorization.verification_status": WO.NOT_VERIFIED})
    problems = WO.validate_work_order(o)
    assert any("PREPARED-only" in p for p in problems), problems


@pytest.mark.parametrize("value", ["VERIFIED", "verified_by_owner_boundary", "", None, True])
def test_verification_status_outside_the_vocabulary_is_refused(value):
    o = good_order(**{"authorization.verification_status": value})
    assert any("verification_status" in p for p in WO.validate_work_order(o))


def test_verified_order_without_authorizing_drive_record_is_refused():
    o = executable_order(**{"authorization.authorizing_record_drive_id": None})
    problems = WO.validate_work_order(o)
    assert any("authorizing_record_drive_id names no Drive record" in p for p in problems), problems


def test_verified_order_without_principals_or_source_ref_is_refused():
    o = executable_order(**{"authorization.approved_principal_ids": []})
    assert any("approved_principal_ids is empty" in p for p in WO.validate_work_order(o))
    o = executable_order(**{"authorization.source_ref": None})
    assert any("source_ref is null" in p for p in WO.validate_work_order(o))


@pytest.mark.parametrize("ref", ["FIXTURE Drive record; not a real authorization",
                                 "the owner said so", "https://example.com/record/1"])
def test_verified_order_source_ref_must_be_a_drive_id_or_drive_url(ref):
    o = executable_order(**{"authorization.source_ref": ref})
    problems = WO.validate_work_order(o)
    assert any("neither a Drive id nor a Drive URL" in p for p in problems), problems


@pytest.mark.parametrize("ref", [DRIVE_ID, SOURCE_URL,
                                 "https://docs.google.com/document/d/1aBcDeFgHiJkLmNoPq/edit"])
def test_verified_order_source_ref_that_names_a_drive_record_passes(ref):
    assert WO.validate_work_order(executable_order(**{"authorization.source_ref": ref})) == []


# ---------------------------------------------------------------------------
# 4. an order is never its own authority
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_order_must_be_authorized_outside_this_repository(value):
    o = good_order(**{"authorization.authorized_outside_this_repository": value})
    problems = WO.validate_work_order(o)
    assert any("cannot authorize itself" in p for p in problems), problems


@pytest.mark.parametrize("field", ["source_ref", "authorizing_record_drive_id",
                                   "policy_change_authorized_by"])
@pytest.mark.parametrize("ref", ["engine/bridge/orders/FIXTURE-TASK-001.json",
                                 "./engine/bridge/orders/X.json",
                                 "/engine/bridge/orders/X.json",
                                 "see engine/bridge/orders/X.json"])
def test_order_naming_its_own_directory_as_authority_is_refused(field, ref):
    o = good_order(**{f"authorization.{field}": ref})
    problems = WO.validate_work_order(o)
    assert any("cannot be its own authority" in p and field in p for p in problems), problems


@pytest.mark.parametrize("ref", [
    "engine/bridge/orders",                             # no trailing slash
    "engine/bridge//orders/FIXTURE-TASK-001.json",      # doubled slash
    "engine/bridge/./orders/FIXTURE-TASK-001.json",     # dot segment
    "Engine/Bridge/Orders/FIXTURE-TASK-001.json",       # case
    "engine\\bridge\\orders\\X.json",                   # backslashes
    "orders/FIXTURE-TASK-001.json",                     # a file name
    "engine/bridge/examples/work_order.example.json",   # another repository path
    "tests/test_bridge.py",
    "see CLAUDE.md",
    "drive/deltas/2026-09-18/x",
    "this pull request (the PR that adds this order)",
    "https://github.com/d6g8k5htny-coder/main/pull/14",  # the contract's own example
    "PR #14",
    "merge request !3",
])
def test_source_ref_naming_a_repository_path_or_pull_request_is_refused(ref):
    """Rule 3 in every spelling: a path here or a PR is never an owner-side record."""
    o = executable_order(**{"authorization.source_ref": ref})
    problems = WO.validate_work_order(o)
    assert any("cannot be its own authority" in p and "source_ref" in p for p in problems), (ref, problems)
    assert WO.names_repository_path(ref) is not None


@pytest.mark.parametrize("ref", [
    "engine/bridge/orders", "engine/bridge/orders/", "engine/bridge//orders/X.json",
    "engine/bridge/./orders/X.json", "engine//bridge/./orders", "Engine/Bridge/Orders/X.json",
    "engine\\bridge\\orders\\X.json", "see engine/bridge/orders/X.json", "/engine/bridge/orders",
])
def test_own_orders_directory_is_recognized_in_every_spelling(ref):
    """The self-authority reason names the orders directory itself, not merely
    'a repository path': the specific rule, not only the general one, holds."""
    why = WO.names_repository_path(ref)
    assert why is not None and why.startswith(f"names a path under {WO.OWN_ORDERS_DIR}"), (ref, why)


@pytest.mark.parametrize("ref", ["engine/bridge/orders_archive/X", "engine/bridge/ordersX"])
def test_own_orders_rule_matches_whole_segments_only(ref):
    why = WO.names_repository_path(ref)
    assert why == "names a path in this repository", (ref, why)   # engine/, but not orders/


def test_repository_top_level_list_matches_the_checkout():
    """The static list the self-authority rule uses must not go stale."""
    ignore = {".git", ".pytest_cache", "__pycache__", ".venv", ".idea", ".vscode",
              ".DS_Store", ".claude", ".mypy_cache", ".ruff_cache"}
    actual = {n for n in os.listdir(ROOT) if n not in ignore}
    assert actual - set(WO.REPOSITORY_TOP_LEVEL) == set(), \
        f"top-level entries missing from REPOSITORY_TOP_LEVEL: {actual - set(WO.REPOSITORY_TOP_LEVEL)}"
    for name in WO.REPOSITORY_TOP_LEVEL:
        assert os.path.exists(os.path.join(ROOT, name)), name


def test_repository_path_is_not_a_drive_id_for_the_authorizing_record():
    o = executable_order(**{"authorization.authorizing_record_drive_id": "governance/X.md"})
    problems = WO.validate_work_order(o)
    assert any("not a Drive id" in p or "cannot be its own authority" in p for p in problems), problems


@pytest.mark.parametrize("value", ["a" * 64, "0" * 40])
def test_a_hex_digest_is_not_a_drive_id(value):
    """A 40- or 64-hex string is a hash; a receipt's own key is never the record that accepted it."""
    assert not C.is_drive_id(value)
    o = executable_order(**{"authorization.authorizing_record_drive_id": value})
    assert any("authorizing_record_drive_id" in p for p in WO.validate_work_order(o))


# ---------------------------------------------------------------------------
# 5. governing sources carry all five identity fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", list(WO.SOURCE_FIELDS))
def test_governing_source_missing_any_identity_field_is_invalid(field):
    o = good_order()
    del o["governing_sources"][0][field]
    o = rehash_order(o)
    problems = WO.validate_work_order(o)
    assert any(f"governing_sources[0].{field} is missing" in p for p in problems), problems


@pytest.mark.parametrize("field,value,fragment", [
    ("sha256", "abc", "not a sha256"),
    ("sha256", None, "not a sha256"),
    ("bytes", -1, "not a byte count"),
    ("bytes", "13535", "not a byte count"),
    ("extraction_rule", "", "extraction_rule is empty"),
    ("drive_id", "drive/deltas/x.md", "not a Drive id"),
    ("revision_if_native", "", "revision_if_native"),
])
def test_governing_source_with_a_bad_identity_value_is_invalid(field, value, fragment):
    o = good_order()
    o["governing_sources"][0][field] = value
    problems = WO.validate_work_order(rehash_order(o))
    assert any(fragment in p for p in problems), problems


def test_governing_source_with_an_extra_field_is_invalid():
    o = good_order()
    o["governing_sources"][0]["title"] = "a title is not identity"
    assert any("unexpected field governing_sources[0].'title'" in p
               for p in WO.validate_work_order(rehash_order(o)))


def test_executable_order_needs_at_least_one_governing_source():
    o = executable_order(governing_sources=[])
    assert any("governing_sources is empty" in p for p in WO.validate_work_order(o))


# ---------------------------------------------------------------------------
# 6. policy/status surfaces in scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    ".github/", ".github/workflows/ci.yml", "governance/", "governance/protocols/x.md",
    "registers/", "registers", "registers/json/work_events.json", "claims/graph.json",
    "claims/", "docs/OPEN_PROBLEMS.md", "docs/", ".", "", "/", "**", "./.github/",
])
def test_policy_surface_in_scope_is_refused_without_both_references(path):
    o = good_order(**{"scope.allowed_paths": ["research/x.py", path]})
    problems = WO.validate_work_order(o)
    assert any("policy/status surface" in p for p in problems), (path, problems)


@pytest.mark.parametrize("path", [
    "engine/../governance/", "sandbox/../claims/graph.json", "../governance/",
    "..\\..\\governance\\", "/home/user/main/governance/", "~/main/governance/",
    "C:\\governance\\", "Governance/", "GOVERNANCE/README.md", "docs/OPEN_PROBLEMS.MD",
    "claims/**", "claims/*", "*/graph.json", "cl*/graph.json", "docs/*", "**/",
    "**/*.json", ".github/**", "governance/**", "docs/{OPEN_PROBLEMS,x}.md",
    "governance//protocols/", "governance/./x", " governance/", "governance/\n",
    "registers/json/../json/x.json", "docs/OPEN_PROBLEMS.md/..",
])
def test_policy_surface_reached_by_a_disguised_spelling_is_refused(path):
    """Traversal, absolute, home, case, globs and empty segments all reach the surface."""
    o = good_order(**{"scope.allowed_paths": [path]})
    problems = WO.validate_work_order(o)
    assert any("policy/status surface" in p for p in problems), (path, problems)


@pytest.mark.parametrize("path", ["engine/../governance/", "../governance/", "claims/**",
                                  "/home/user/main/governance/", "~/x", "docs/*", "**/",
                                  "governance//protocols/", "governance/./x",
                                  "docs/OPEN_PROBLEMS.md/.", " governance/", "governance/\t"])
def test_malformed_scope_path_is_refused_even_with_both_references(path):
    """An unnormalizable entry is refused outright; no reference unlocks it."""
    o = good_order(**{"scope.allowed_paths": [path],
                      "public_disclosure.approved": True,
                      "authorization.policy_change_authorized_by": POLICY_ID})
    problems = WO.validate_work_order(o)
    assert any("refused, not normalized" in p for p in problems), (path, problems)
    assert WO.malformed_scope_paths([path]), path


@pytest.mark.parametrize("path,norm", [
    ("research/x.py", "research/x.py"), ("./research/x.py", "research/x.py"),
    ("Research/X.py", "research/x.py"), ("sandbox/t/", "sandbox/t"),
    ("sandbox\\t\\", "sandbox/t"), (".", "."), ("./", "."),
])
def test_normalize_scope_path_positive(path, norm):
    assert C.normalize_scope_path(path) == (norm, "")


@pytest.mark.parametrize("path", ["research/x.py", "sandbox/t/", "docs/RESEARCH_MAP.md",
                                  "claims/README.md", "packages/x/", "engine/rn_engine/x.py",
                                  "tests/test_rn_moment_envelope.py", "research/"])
def test_non_policy_paths_are_not_refused(path):
    o = good_order(**{"scope.allowed_paths": [path]})
    problems = WO.validate_work_order(o)
    assert not any("surface" in p for p in problems), (path, problems)


def test_policy_surface_with_only_disclosure_is_still_refused():
    o = good_order(**{"scope.allowed_paths": [".github/workflows/ci.yml"],
                      "public_disclosure.approved": True})
    assert any("policy/status surface" in p for p in WO.validate_work_order(o))


def test_policy_surface_with_only_policy_reference_is_still_refused():
    o = good_order(**{"scope.allowed_paths": [".github/workflows/ci.yml"],
                      "authorization.policy_change_authorized_by": POLICY_ID})
    assert any("policy/status surface" in p for p in WO.validate_work_order(o))


def test_policy_surface_with_both_references_is_recorded_not_granted():
    o = good_order(**{"scope.allowed_paths": [".github/workflows/ci.yml"],
                      "public_disclosure.approved": True,
                      "authorization.policy_change_authorized_by": POLICY_ID})
    assert WO.validate_work_order(o) == []
    notes = WO.work_order_notes(o)
    assert any("RECORDED" in n and "Nothing is granted" in n and POLICY_ID in n
               for n in notes), notes
    # and nothing about the order changed: it is still PREPARED and unverified
    assert o["status"] == WO.STATUS_PREPARED
    assert o["authorization"]["verification_status"] == WO.NOT_VERIFIED


def test_disclosure_paths_without_approval_are_refused():
    o = good_order(**{"public_disclosure.approved_paths": ["research/x.py"]})
    assert any("approved is false but approved_paths" in p for p in WO.validate_work_order(o))


# ---------------------------------------------------------------------------
# 6b. protected surfaces: never in scope, whatever the references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "engine/bridge/orders/", "engine/bridge/receipts/", "engine/bridge/examples/",
    "engine/bridge/", "engine/bridge/work_order.py", "engine/", "tools/bridge_check.py",
    "tools/claims_check.py", "tools/", "tests/test_bridge.py", "tests/", "engine/lanes/",
    "engine/lanes/D.json", "engine/receipts/", "CLAUDE.md", "AGENTS.md", "quarantine/",
    "drive/", "drive/inventory.jsonl", "Engine/Bridge/Orders/", "claude.md", ".",
])
def test_scope_reaching_a_protected_surface_is_refused_even_with_both_references(path):
    """An order that could authorize writing an order, a receipt, the checker or
    its controls could authorize itself. No reference unlocks these."""
    o = executable_order(**{"scope.allowed_paths": [path],
                            "public_disclosure.approved": True,
                            "public_disclosure.approved_paths": [],
                            "authorization.policy_change_authorized_by": POLICY_ID})
    problems = WO.validate_work_order(o)
    assert any("protected surface" in p for p in problems), (path, problems)
    assert WO.protected_surfaces_touched([path]), path


@pytest.mark.parametrize("paths,fragment", [
    (["engine/bridge/receipts/"], "protected surface"),
    (["governance/"], "not within scope.allowed_paths"),
    (["research/other.py"], "not within scope.allowed_paths"),
    (["sandbox/../research/"], "refused, not normalized"),
])
def test_approved_paths_are_bounded_by_scope_and_surfaces(paths, fragment):
    o = good_order(**{"scope.allowed_paths": ["research/x.py", "sandbox/t/"],
                      "public_disclosure.approved": True,
                      "public_disclosure.approved_paths": paths})
    problems = WO.validate_work_order(o)
    assert any(fragment in p and "approved_paths" in p for p in problems), (paths, problems)


def test_approved_paths_within_scope_pass():
    o = good_order(**{"scope.allowed_paths": ["research/x.py", "sandbox/t/"],
                      "public_disclosure.approved": True,
                      "public_disclosure.approved_paths": ["sandbox/t/out.json", "research/x.py"]})
    assert WO.validate_work_order(o) == []


# ---------------------------------------------------------------------------
# 6c. allowed commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd,fragment", [
    ("git push origin HEAD:main", "git push"),
    ("git commit -am x", "git commit"),
    ("git add -A && git commit -m x", "git add"),
    ("gh pr create --fill", "gh pr"),
    ("python3 -c \"open('claims/graph.json','w').write('{}')\"", "writes into"),
    ("echo x > governance/GIT_ADAPTATION.md", "writes into"),
    ("tee engine/bridge/orders/NEW.json < x", "writes into"),
    ("cp x registers/json/work_events.json", "writes into"),
    ("sed -i s/a/b/ docs/OPEN_PROBLEMS.md", "writes into"),
    ("python3 write.py --out tools/bridge_check.py", "writes into"),
    (":", "placeholder"), ("true", "placeholder"), ("", "blank"), ("  ", "blank"),
    ("-", "placeholder"), ("TODO", "placeholder"),
])
def test_allowed_command_that_mutates_git_or_writes_a_surface_is_refused(cmd, fragment):
    o = executable_order(**{"scope.allowed_commands": [cmd]})
    problems = WO.validate_work_order(o)
    assert any("allowed_commands[0]" in p and fragment in p for p in problems), (cmd, problems)


@pytest.mark.parametrize("cmd", [
    "python3 tools/bridge_check.py", "python3 -m pytest -q tests/test_bridge.py",
    "python3 tools/claims_check.py > sandbox/t/out.txt", "cat governance/GIT_ADAPTATION.md",
    "git rev-parse HEAD", "git status --short", "python3 research/rn/moment_envelope.py",
])
def test_allowed_command_that_only_reads_or_runs_a_checker_is_not_refused(cmd):
    assert WO.command_problems(cmd) == [], cmd


# ---------------------------------------------------------------------------
# 7. the order is frozen and closed
# ---------------------------------------------------------------------------

def test_order_edited_after_its_digest_was_taken_is_refused():
    o = good_order()
    o["scope"]["allowed_paths"].append("research/extra.py")   # no re-digest
    problems = WO.validate_work_order(o)
    assert any("work_order_digest mismatch" in p for p in problems), problems


def test_order_with_an_unknown_top_level_field_is_refused():
    o = rehash_order({**good_order(), "approved": True})
    assert any("unexpected field work_order.'approved'" in p for p in WO.validate_work_order(o))


@pytest.mark.parametrize("field", ["task_id", "authorization", "governing_sources", "claim",
                                   "scope", "limits", "public_disclosure", "does_not_establish"])
def test_order_missing_a_required_field_is_refused(field):
    o = rehash_order(del_path(good_order(), field))
    assert any(f"work_order.{field} is missing" in p for p in WO.validate_work_order(o))


def test_order_missing_its_digest_is_refused():
    o = del_path(good_order(), "work_order_digest")
    assert any("work_order_digest is missing" in p for p in WO.validate_work_order(o))


def test_order_with_the_wrong_schema_id_is_refused():
    o = good_order(schema="q0.bridge.work_order/v0")
    assert any("schema is" in p for p in WO.validate_work_order(o))


@pytest.mark.parametrize("value", ["", "   ", "n/a", "TBD", "nothing much"])
def test_order_without_a_real_does_not_establish_is_refused(value):
    o = good_order(does_not_establish=value)
    assert any("does_not_establish" in p for p in WO.validate_work_order(o))


def test_example_order_cannot_be_verified_or_executable():
    o = good_order(record_kind=WO.RECORD_EXAMPLE,
                   **{"authorization.verification_status": WO.VERIFIED_BY_OWNER_BOUNDARY,
                      "authorization.authorizing_record_drive_id": DRIVE_ID,
                      "authorization.approved_principal_ids": ["x"],
                      "authorization.source_ref": DRIVE_ID})
    assert any("an example is not an authorization" in p for p in WO.validate_work_order(o))
    o = good_order(record_kind=WO.RECORD_EXAMPLE, status=WO.STATUS_EXECUTABLE)
    assert any("an example is never executable" in p for p in WO.validate_work_order(o))


@pytest.mark.parametrize("field,fragment", [
    ("claim.work_events_event_id", "work_events_event_id is null"),
    ("claim.lease_until_utc", "lease_until_utc is null"),
    ("scope.required_negative_controls", "required_negative_controls is empty"),
    ("scope.acceptance_tests", "acceptance_tests is empty"),
    ("scope.allowed_paths", "allowed_paths is empty"),
    ("scope.allowed_commands", "allowed_commands is empty"),
    ("repository.base_commit_sha", "base_commit_sha is not pinned"),
    ("limits.network_policy", "limits are not all set"),
])
def test_executable_order_missing_a_required_element_is_refused(field, fragment):
    empty = [] if field.startswith("scope.") else None
    o = executable_order(**{field: empty})
    problems = WO.validate_work_order(o)
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize("value", ["2026-99-99T99:99:99Z", "2026-02-30T00:00:00Z",
                                   "2026-09-18T24:00:00Z"])
def test_lease_with_the_shape_but_not_the_calendar_is_refused(value):
    o = executable_order(**{"claim.lease_until_utc": value})
    assert any("lease_until_utc" in p for p in WO.validate_work_order(o))
    assert not C.is_timestamp(value)


@pytest.mark.parametrize("garbage", [None, [], "x", 3, {}, {"schema": 1},
                                     {"schema": WO.SCHEMA, "work_order_digest": 5},
                                     {"work_order_digest": "z" * 64}])
def test_order_validator_never_raises_on_garbage(garbage):
    problems = WO.validate_work_order(garbage)
    assert problems and all(isinstance(p, str) for p in problems)
    assert isinstance(WO.work_order_notes(garbage), list)


# ---------------------------------------------------------------------------
# 8. receipts: ACCEPTED is owner-side
# ---------------------------------------------------------------------------

def recorded_receipt(**over):
    r = executed_receipt()
    r["status"] = RR.RECORDED
    r["drive_return"].update({"delivery_id": "FIXTURE-DELIVERY", "receipt_file_id": DRIVE_ID,
                              "readback_sha256": SHA, "readback_utc": TS})
    for k, v in over.items():
        set_path(r, k, v)
    return rehash_receipt(r)


def accepted_receipt(**over):
    r = recorded_receipt()
    r["status"] = RR.ACCEPTED
    r["drive_return"]["acceptance_status"] = RR.ACCEPTED
    r["accepted_by_drive_record"] = {"drive_id": DRIVE_ID, "sha256": SHA}
    for k, v in over.items():
        set_path(r, k, v)
    return rehash_receipt(r)


def test_accepted_receipt_without_the_drive_record_is_refused():
    r = accepted_receipt(accepted_by_drive_record=None)
    problems = RR.validate_run_receipt(r)
    assert any("this repository cannot verify" in p for p in problems), problems


def test_accepted_receipt_with_the_drive_record_is_transcribed_not_granted():
    r = accepted_receipt()
    assert RR.validate_run_receipt(r) == []
    notes = RR.run_receipt_notes(r)
    assert any("transcribed" in n and "cannot verify" in n and "unverified" in n
               for n in notes), notes


@pytest.mark.parametrize("field,value", [("drive_id", "engine/bridge/receipts/x.json"),
                                         ("drive_id", "e" * 64),
                                         ("sha256", "abc")])
def test_accepted_receipt_with_a_malformed_drive_record_is_refused(field, value):
    r = accepted_receipt(**{f"accepted_by_drive_record.{field}": value})
    assert any(f"accepted_by_drive_record.{field}" in p for p in RR.validate_run_receipt(r))


def test_acceptance_status_accepted_on_a_non_accepted_receipt_is_refused():
    r = recorded_receipt(**{"drive_return.acceptance_status": RR.ACCEPTED})
    assert any("acceptance_status is ACCEPTED" in p for p in RR.validate_run_receipt(r))
    r = recorded_receipt(accepted_by_drive_record={"drive_id": DRIVE_ID, "sha256": SHA})
    assert any("accepted_by_drive_record is set" in p for p in RR.validate_run_receipt(r))


def test_accepted_receipt_needs_acceptance_status_accepted():
    r = accepted_receipt(**{"drive_return.acceptance_status": RR.ACCEPTANCE_NOT})
    assert any("acceptance_status is not ACCEPTED" in p for p in RR.validate_run_receipt(r))


# ---------------------------------------------------------------------------
# 9. scientific_status_change is always UNCHANGED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["PROMOTED", "CHANGED", "unchanged", "CLOSED", "", None, True])
def test_any_scientific_status_change_but_unchanged_is_refused(value):
    r = good_receipt(scientific_status_change=value)
    problems = RR.validate_run_receipt(r)
    assert any("FW-NO-RECEIPT-PROMOTION" in p for p in problems), problems


def test_a_missing_scientific_status_change_is_refused():
    r = rehash_receipt(del_path(good_receipt(), "scientific_status_change"))
    assert any("scientific_status_change is missing" in p for p in RR.validate_run_receipt(r))


# ---------------------------------------------------------------------------
# 10. the idempotency key
# ---------------------------------------------------------------------------

def test_idempotency_key_changes_when_any_of_its_four_inputs_changes():
    base = ("T-1", "a" * 64, COMMIT, "run-1")
    k0 = RR.idempotency_key(*base)
    variants = [
        ("T-2", "a" * 64, COMMIT, "run-1"),
        ("T-1", "b" * 64, COMMIT, "run-1"),
        ("T-1", "a" * 64, "f" * 40, "run-1"),
        ("T-1", "a" * 64, None, "run-1"),
        ("T-1", "a" * 64, COMMIT, "run-2"),
    ]
    keys = [RR.idempotency_key(*v) for v in variants]
    assert all(k != k0 for k in keys)
    assert len(set(keys)) == len(keys)
    assert RR.idempotency_key(*base) == k0          # deterministic
    # a null commit and the string "null" are different inputs
    assert RR.idempotency_key("T", "a" * 64, None, "r") != RR.idempotency_key("T", "a" * 64, "null", "r")


def test_receipt_with_a_wrong_idempotency_key_is_refused():
    r = good_receipt()
    r["idempotency_key"] = RR.idempotency_key(r["task_id"], r["work_order_digest"], None, "other")
    r = {**r, "body_sha256": RR.receipt_body_digest(r)}     # self-consistent hash, wrong key
    problems = RR.validate_run_receipt(r)
    assert any("idempotency_key mismatch" in p for p in problems), problems


def test_finalize_recomputes_the_key_from_the_tested_commit():
    r = executed_receipt()
    assert r["idempotency_key"] == RR.idempotency_key(
        r["task_id"], r["work_order_digest"], COMMIT, r["run_id"])
    r2 = executed_receipt(**{"execution.commit_sha": "f" * 40})
    assert r2["idempotency_key"] != r["idempotency_key"]


# ---------------------------------------------------------------------------
# 11. missing evidence is never PASS
# ---------------------------------------------------------------------------

def test_pass_technical_with_null_tests_passed_is_refused():
    r = executed_receipt(**{"verification.tests_passed": None,
                            "review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "fixture exposure"})
    problems = RR.validate_run_receipt(r)
    assert any("never PASS" in p for p in problems), problems


def test_pass_technical_with_zero_tests_passed_is_refused():
    """A vacuous pass: nothing ran, nothing failed, PASS. Refused."""
    r = executed_receipt(**{"verification.tests_passed": 0, "verification.tests_failed": 0,
                            "review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "author-side, fixture"})
    problems = RR.validate_run_receipt(r)
    assert any("tests_passed is 0" in p and "never PASS" in p for p in problems), problems


def test_pass_technical_with_null_tests_failed_is_refused():
    r = executed_receipt(**{"verification.tests_failed": None,
                            "review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "fixture exposure"})
    assert any("tests_failed is null" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("status", [RR.NOT_RUN, RR.CANNOT_VERIFY, RR.PREPARED])
def test_pass_technical_on_a_receipt_that_did_not_execute_is_refused(status):
    r = good_receipt(status=status, **{"review.technical_verdict": RR.PASS_TECHNICAL,
                                       "review.authorship_exposure": "fixture exposure",
                                       "verification.tests_passed": 3})
    problems = RR.validate_run_receipt(r)
    assert any("nothing that did not execute can pass" in p for p in problems), problems


def test_pass_technical_without_exposure_is_refused():
    r = executed_receipt(**{"review.technical_verdict": RR.PASS_TECHNICAL})
    assert any("authorship_exposure is null" in p for p in RR.validate_run_receipt(r))


def test_pass_technical_with_failures_or_nonzero_exit_is_refused():
    r = executed_receipt(**{"review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "fixture exposure",
                            "verification.tests_failed": 1})
    assert any("PASS_TECHNICAL with tests_failed > 0" in p for p in RR.validate_run_receipt(r))
    r = executed_receipt(**{"review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "fixture exposure",
                            "execution.exit_codes": [0, 7],
                            "verification.tests_failed": 1})
    assert any("PASS_TECHNICAL with a nonzero exit code" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("field,value,fragment", [
    ("execution.commit_sha", None, "commit_sha is null"),
    ("execution.repository_id", None, "repository_id is null"),
    ("execution.tree_sha", None, "tree_sha is null"),
    ("execution.dirty_worktree", None, "dirty_worktree is null"),
    ("execution.environment_identity", None, "environment_identity is null"),
    ("execution.commands", [], "commands is empty"),
    ("execution.exit_codes", [], "exit_codes is empty"),
    ("verification.negative_controls", [], "negative_controls is empty"),
    ("verification.tests_passed", None, "test counts are null"),
    ("verification.tests_failed", None, "test counts are null"),
    ("verification.coverage", None, "coverage is null"),
    ("execution.start_utc", None, "start_utc/end_utc"),
    ("actor.authenticated_principal_id", None, "authenticated_principal_id is null"),
])
def test_executed_receipt_missing_evidence_is_refused(field, value, fragment):
    r = executed_receipt(**{field: value})
    problems = RR.validate_run_receipt(r)
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize("field,value", [
    ("execution.commands", [":"]), ("execution.commands", [""]), ("execution.commands", [" "]),
    ("execution.commands", ["-"]), ("execution.commands", ["TODO"]), ("execution.commands", ["true"]),
    ("verification.negative_controls", ["n/a"]), ("verification.negative_controls", ["-"]),
    ("verification.negative_controls", ["."]), ("verification.negative_controls", ["none"]),
    ("verification.negative_controls", ["TODO"]), ("verification.negative_controls", ["x"]),
    ("verification.negative_controls", ["short"]),
    ("actor.authenticated_principal_id", "unknown"), ("actor.authenticated_principal_id", "-"),
    ("actor.authenticated_principal_id", "n/a"), ("actor.authenticated_principal_id", "anonymous"),
    ("actor.authenticated_principal_id", "claude"),
    ("execution.environment_identity", "n/a"), ("execution.environment_identity", "TBD"),
    ("verification.coverage", "-"), ("verification.coverage", "none"),
])
def test_executed_receipt_with_placeholder_evidence_is_refused(field, value):
    """Evidence in shape only: a placeholder where a statement belongs."""
    r = executed_receipt(**{field: value})
    problems = RR.validate_run_receipt(r)
    assert any(field.split(".")[-1] in p and ("placeholder" in p or "blank" in p or
                                              "shorter than" in p or "no letter" in p)
               for p in problems), (field, value, problems)


@pytest.mark.parametrize("value", ["none", "x", "n/a", "-"])
def test_pass_technical_with_a_placeholder_exposure_is_refused(value):
    r = executed_receipt(**{"review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": value})
    assert any("authorship_exposure" in p and "placeholder" in p for p in RR.validate_run_receipt(r))


def test_executed_receipt_that_counted_no_test_is_refused():
    r = executed_receipt(**{"verification.tests_passed": 0, "verification.tests_failed": 0})
    problems = RR.validate_run_receipt(r)
    assert any("counted no test" in p for p in problems), problems


def test_executed_receipt_with_zero_passed_and_some_failed_is_noted():
    r = executed_receipt(**{"verification.tests_passed": 0, "verification.tests_failed": 2,
                            "execution.exit_codes": [1, 0]})
    assert RR.validate_run_receipt(r) == []
    assert any("tests_passed is 0" in n for n in RR.run_receipt_notes(r))


def test_executed_receipt_needs_one_exit_code_per_command():
    r = executed_receipt(**{"execution.exit_codes": [0]})
    assert any("one exit code per command" in p for p in RR.validate_run_receipt(r))


def test_nonzero_exit_code_cannot_claim_zero_failures():
    r = executed_receipt(**{"execution.exit_codes": [0, 7]})
    problems = RR.validate_run_receipt(r)
    assert any("cannot be reported as zero failures" in p for p in problems), problems
    # the honest version passes
    r = executed_receipt(**{"execution.exit_codes": [0, 7], "verification.tests_failed": 2})
    assert RR.validate_run_receipt(r) == []


def test_a_receipt_that_did_not_execute_cannot_report_exit_codes():
    r = good_receipt(**{"execution.exit_codes": [0]})
    assert any("exit_codes is not empty" in p for p in RR.validate_run_receipt(r))


def test_pass_technical_on_a_dirty_worktree_is_refused_and_dirty_is_noted():
    r = executed_receipt(**{"execution.dirty_worktree": True,
                            "review.technical_verdict": RR.PASS_TECHNICAL,
                            "review.authorship_exposure": "author-side, fixture"})
    problems = RR.validate_run_receipt(r)
    assert any("dirty_worktree is true" in p and "PASS_TECHNICAL" in p for p in problems), problems
    r = executed_receipt(**{"execution.dirty_worktree": True})
    assert RR.validate_run_receipt(r) == []
    assert any("dirty_worktree is true" in n and "not the named commit" in n
               for n in RR.run_receipt_notes(r))


def test_run_that_ends_before_it_starts_is_refused():
    r = executed_receipt(**{"execution.start_utc": TS_LATER, "execution.end_utc": TS})
    assert any("before start_utc" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("value", ["2026-99-99T99:99:99Z", "2026-02-30T12:00:00Z",
                                   "2026-09-18T12:60:00Z"])
def test_timestamp_with_the_shape_but_not_the_calendar_is_refused(value):
    r = executed_receipt(**{"execution.start_utc": value, "execution.end_utc": value})
    assert any("not a UTC timestamp" in p for p in RR.validate_run_receipt(r))
    assert C.parse_timestamp(value) is None


# ---------------------------------------------------------------------------
# 12. independence credit
# ---------------------------------------------------------------------------

def test_independence_credit_without_a_record_is_refused():
    r = executed_receipt(**{"review.independence_credit": 1,
                            "review.technical_verdict": RR.AMEND_REQUIRED})
    problems = RR.validate_run_receipt(r)
    assert any("cannot be manufactured" in p for p in problems), problems


@pytest.mark.parametrize("value", [2, -1, "1", 1.0, True])
def test_independence_credit_outside_zero_or_one_is_refused(value):
    r = good_receipt(**{"review.independence_credit": value,
                        "review.organizational_independence_record_id": REVIEW_ID})
    assert any("independence_credit" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("record_id", ["none", "n/a", "-", "TBD", "x", "unknown",
                                       "same-provider-reviewer", "same provider (Anthropic)",
                                       "REC-1", "short", "reviews/records/REV-P15-A.json"])
def test_independence_credit_with_a_placeholder_record_id_is_refused(record_id):
    """CLAUDE.md rule 6: independence cannot be manufactured, least of all by
    writing 'none' in the record-id field."""
    r = executed_receipt(**{"review.independence_credit": 1,
                            "review.technical_verdict": RR.AMEND_REQUIRED,
                            "review.organizational_independence_record_id": record_id})
    problems = RR.validate_run_receipt(r)
    assert any("cannot be manufactured" in p for p in problems), (record_id, problems)
    assert not RR.run_receipt_notes(r) or all("awards no independence" in n
                                              for n in RR.run_receipt_notes(r))


def test_independence_credit_on_a_not_reviewed_verdict_is_refused():
    r = executed_receipt(**{"review.independence_credit": 1,
                            "review.organizational_independence_record_id": REVIEW_ID})
    assert r["review"]["technical_verdict"] == RR.NOT_REVIEWED
    problems = RR.validate_run_receipt(r)
    assert any("NOT_REVIEWED verdict" in p and "manufactured" in p for p in problems), problems


@pytest.mark.parametrize("record_id", [REVIEW_ID, DRIVE_ID, "OP-PROT-012-5-FIXTURE"])
def test_independence_credit_with_a_record_is_recorded_not_awarded(record_id):
    r = executed_receipt(**{"review.independence_credit": 1,
                            "review.technical_verdict": RR.AMEND_REQUIRED,
                            "review.organizational_independence_record_id": record_id})
    assert RR.validate_run_receipt(r) == []
    notes = RR.run_receipt_notes(r)
    assert any("awards no independence" in n and "not moved" in n for n in notes), notes


# ---------------------------------------------------------------------------
# 13. RECORDED needs a readback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["delivery_id", "receipt_file_id", "readback_sha256", "readback_utc"])
def test_recorded_receipt_without_a_readback_element_is_refused(field):
    r = recorded_receipt(**{f"drive_return.{field}": None})
    problems = RR.validate_run_receipt(r)
    assert any("RECORDED requires a Drive readback" in p and field in p for p in problems), problems


def test_recorded_receipt_with_a_readback_passes():
    assert RR.validate_run_receipt(recorded_receipt()) == []


# ---------------------------------------------------------------------------
# 14. the receipt is closed and hashed
# ---------------------------------------------------------------------------

def test_receipt_edited_after_it_was_hashed_is_refused():
    r = good_receipt()
    r["verification"]["tests_passed"] = 99                    # no rehash
    assert any("body_sha256 mismatch" in p and "append-only" in p
               for p in RR.validate_run_receipt(r))


def test_receipt_with_an_unknown_field_is_refused():
    r = rehash_receipt({**good_receipt(), "verdict": "PASS"})
    assert any("unexpected field run_receipt.'verdict'" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("field", ["status", "actor", "execution", "verification", "review",
                                   "drive_return", "does_not_establish", "idempotency_key"])
def test_receipt_missing_a_required_field_is_refused(field):
    r = rehash_receipt_body(del_path(good_receipt(), field))
    assert any(f"run_receipt.{field} is missing" in p for p in RR.validate_run_receipt(r))


def test_receipt_with_the_wrong_schema_id_is_refused():
    r = good_receipt(schema="q0.engine.receipt/v1")
    assert any("schema is" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("value", ["PASS", "DONE", "not_run", "", None])
def test_receipt_status_outside_the_vocabulary_is_refused(value):
    r = good_receipt(status=value)
    assert any("status" in p and "is not one of" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("value", ["PASS", "APPROVED", "pass_technical", None])
def test_technical_verdict_outside_the_vocabulary_is_refused(value):
    r = good_receipt(**{"review.technical_verdict": value})
    assert any("technical_verdict" in p and "is not one of" in p for p in RR.validate_run_receipt(r))


@pytest.mark.parametrize("value", ["", "n/a", "see above", "too short"])
def test_receipt_without_a_real_does_not_establish_is_refused(value):
    r = good_receipt(does_not_establish=value)
    assert any("does_not_establish" in p for p in RR.validate_run_receipt(r))


def test_example_receipt_cannot_claim_execution_or_review():
    r = executed_receipt(record_kind=RR.RECORD_EXAMPLE)
    assert any("an example is NOT_RUN" in p for p in RR.validate_run_receipt(r))
    r = good_receipt(record_kind=RR.RECORD_EXAMPLE, **{"review.technical_verdict": RR.CANNOT_VERIFY})
    assert any("EXAMPLE but technical_verdict" in p for p in RR.validate_run_receipt(r))


def test_output_records_carry_path_bytes_and_digest():
    r = executed_receipt(outputs=[{"path": "sandbox/EXAMPLE-DG-EXEC-NOT-A-TASK/x.json",
                                   "bytes": 3, "sha256": SHA}])
    assert RR.validate_run_receipt(r) == []
    for bad in ({"path": "", "bytes": 3, "sha256": SHA},
                {"path": "x", "bytes": -3, "sha256": SHA},
                {"path": "x", "bytes": 3, "sha256": "nope"},
                {"path": "../x", "bytes": 3, "sha256": SHA},
                {"path": "/x", "bytes": 3, "sha256": SHA},
                {"path": "x/*", "bytes": 3, "sha256": SHA},
                {"path": ".", "bytes": 3, "sha256": SHA},
                {"path": "x", "bytes": 3}):
        assert RR.validate_run_receipt(executed_receipt(outputs=[bad])), bad


@pytest.mark.parametrize("garbage", [None, [], "x", 3, {}, {"schema": 1},
                                     {"schema": RR.SCHEMA, "body_sha256": 5},
                                     {"body_sha256": "z" * 64}])
def test_receipt_validator_never_raises_on_garbage(garbage):
    problems = RR.validate_run_receipt(garbage)
    assert problems and all(isinstance(p, str) for p in problems)


@pytest.mark.parametrize("garbage", [
    None, [], "x", {}, {"idempotency_key": None, "review": {"independence_credit": 1}},
    {"idempotency_key": 5, "status": "ACCEPTED", "accepted_by_drive_record": {"drive_id": None}},
    {"idempotency_key": None, "status": "EXECUTED", "execution": {"dirty_worktree": True},
     "verification": {"tests_passed": 0}},
    {"idempotency_key": [], "review": "x", "execution": 3, "verification": None},
])
def test_notes_functions_never_raise_on_garbage(garbage):
    assert isinstance(RR.run_receipt_notes(garbage), list)
    assert isinstance(WO.work_order_notes(garbage), list)


# ---------------------------------------------------------------------------
# 14b. a receipt against its order
# ---------------------------------------------------------------------------

def test_receipt_task_id_must_match_the_referenced_order():
    o = good_order()
    r = good_receipt(o, task_id="SOMETHING-ELSE")
    assert any("differs from the referenced order" in p for p in RR.check_receipt_against_order(r, o))
    assert any("resolves to no committed work order" in p for p in RR.check_receipt_against_order(r, None))


def test_a_record_receipt_may_not_reference_an_example_order():
    r = good_receipt(load(EX_ORDER))
    assert any("examples authorize nothing" in p
               for p in RR.check_receipt_against_order(r, load(EX_ORDER)))


@pytest.mark.parametrize("status", [RR.EXECUTED, RR.RECORDED, RR.ACCEPTED])
def test_executed_receipt_against_a_prepared_only_order_is_refused(status, tmp_path):
    """A NOT_VERIFIED order is PREPARED-only. A receipt that says something ran
    under it records a run the contract did not permit, and the checker fails."""
    o = good_order()
    builder = {RR.EXECUTED: executed_receipt,
               RR.RECORDED: lambda order: rehash_receipt({**recorded_receipt(), "task_id": order["task_id"],
                                                          "work_order_digest": order["work_order_digest"]}),
               RR.ACCEPTED: lambda order: rehash_receipt({**accepted_receipt(), "task_id": order["task_id"],
                                                          "work_order_digest": order["work_order_digest"]})}
    r = builder[status](o)
    assert RR.validate_run_receipt(r) == []
    problems = RR.check_receipt_against_order(r, o)
    assert any("PREPARED-only" in p and "did not permit" in p for p in problems), problems
    od, rd, _ = workspace(tmp_path, [o], [r])
    out = run_checker(od, rd)
    assert out.returncode == 1 and "PREPARED-only" in out.stdout


@pytest.mark.parametrize("status", [RR.NOT_RUN, RR.CANNOT_VERIFY, RR.PREPARED])
def test_non_executed_receipt_against_a_prepared_order_passes(status):
    o = good_order()
    assert RR.check_receipt_against_order(good_receipt(o, status=status), o) == []


def test_receipt_repository_id_must_match_the_order():
    o = executable_order()
    r = executed_receipt(o, **{"execution.repository_id": 1})
    problems = RR.check_receipt_against_order(r, o)
    assert any("repository_id 1 differs" in p for p in problems), problems


def test_receipt_command_not_in_allowed_commands_is_refused():
    o = executable_order()
    r = executed_receipt(o, **{"execution.commands": ["git push --force origin HEAD:main"],
                               "execution.exit_codes": [0]})
    problems = RR.check_receipt_against_order(r, o)
    assert any("not present verbatim" in p for p in problems), problems
    # a whitespace variant of an allowed command is not that command
    r = executed_receipt(o, **{"execution.commands": [COMMANDS[0] + " ", COMMANDS[1]]})
    assert any("not present verbatim" in p for p in RR.check_receipt_against_order(r, o))


@pytest.mark.parametrize("path,fragment", [
    ("claims/graph.json", "policy or protected surface"),
    ("engine/bridge/orders/NEW-ORDER.json", "policy or protected surface"),
    ("governance/GIT_ADAPTATION.md", "policy or protected surface"),
    ("tools/bridge_check.py", "policy or protected surface"),
    ("research/elsewhere.py", "not within the order's scope.allowed_paths"),
    ("sandbox/OTHER/x.json", "not within the order's scope.allowed_paths"),
])
def test_receipt_output_outside_scope_or_on_a_surface_is_refused(path, fragment, tmp_path):
    o = executable_order()
    r = executed_receipt(o, outputs=[{"path": path, "bytes": 10, "sha256": SHA}])
    problems = RR.check_receipt_against_order(r, o)
    assert any(fragment in p for p in problems), (path, problems)
    od, rd, _ = workspace(tmp_path, [o], [r])
    assert run_checker(od, rd).returncode == 1


def test_receipt_output_within_scope_passes():
    o = executable_order()
    r = executed_receipt(o, outputs=[{"path": "sandbox/EXAMPLE-DG-EXEC-NOT-A-TASK/out.json",
                                      "bytes": 10, "sha256": SHA}])
    assert RR.check_receipt_against_order(r, o) == []


# ---------------------------------------------------------------------------
# 15. the store is append-only
# ---------------------------------------------------------------------------

def test_store_writes_a_content_addressed_file(tmp_path):
    r = executed_receipt()
    res = RR.store_receipt(r, str(tmp_path))
    assert res.disposition == "STORED" and res.wrote
    assert os.path.basename(res.path) == r["idempotency_key"] + ".json"
    assert load(res.path) == r


def test_identical_redelivery_writes_nothing(tmp_path):
    r = executed_receipt()
    first = RR.store_receipt(r, str(tmp_path))
    before = snapshot(str(tmp_path))
    again = RR.store_receipt(copy.deepcopy(r), str(tmp_path))
    assert again.disposition == "IDENTICAL_RETRY" and not again.wrote
    assert again.path == first.path
    assert snapshot(str(tmp_path)) == before


def test_conflicting_redelivery_is_held_beside_and_never_replaces_the_first(tmp_path):
    r1 = executed_receipt()
    r2 = executed_receipt(**{"verification.tests_passed": 4})    # same key, different content
    assert r1["idempotency_key"] == r2["idempotency_key"]
    rd = str(tmp_path / "receipts")
    p1 = RR.store_receipt(r1, rd).path
    with open(p1, "rb") as f:
        first_bytes = f.read()
    res = RR.store_receipt(r2, rd, now=TS)
    assert res.disposition == RR.DISPOSITION_NEEDS_RECONCILIATION and res.wrote
    assert res.path != p1
    with open(p1, "rb") as f:
        assert f.read() == first_bytes                         # the first is untouched
    held = load(res.path)
    assert held["schema"] == RR.HELD_SCHEMA
    assert held["delivered"] == r2                              # preserved verbatim
    assert held["first_body_sha256"] == r1["body_sha256"]
    assert RR.validate_held_delivery(held, r1) == []
    assert RR.parse_receipt_file_name(os.path.basename(res.path))[0] == "held"
    # delivering the same conflict a third time writes nothing new
    before = snapshot(rd)
    res3 = RR.store_receipt(r2, rd, now=TS)
    assert res3.disposition == RR.DISPOSITION_NEEDS_RECONCILIATION and not res3.wrote
    assert snapshot(rd) == before
    # and the checker sees one receipt, one held delivery, no problems, a note
    od, _rd, _ = workspace(tmp_path, [executable_order()])
    out = run_checker(od, rd)
    assert out.returncode == 0, out.stdout
    assert "receipts=1 held=1" in out.stdout and "NOTE:" in out.stdout


def test_store_refuses_an_invalid_receipt_and_writes_nothing(tmp_path):
    r = executed_receipt(scientific_status_change="PROMOTED")
    with pytest.raises(RR.ReceiptRefused) as exc:
        RR.store_receipt(r, str(tmp_path))
    assert "FW-NO-RECEIPT-PROMOTION" in str(exc.value)
    assert os.listdir(tmp_path) == []


def test_store_refuses_an_example(tmp_path):
    with pytest.raises(RR.ReceiptRefused) as exc:
        RR.store_receipt(load(EX_RECEIPT), str(tmp_path))
    assert "EXAMPLE" in str(exc.value)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("sub", ["engine/bridge/orders", "engine/bridge/examples",
                                 "engine/lanes", "engine/receipts", "claims", "registers",
                                 "governance", "docs", ".github", "drive"])
def test_store_refuses_governed_destinations(sub):
    target = os.path.join(ROOT, sub)
    before = snapshot(target) if os.path.isdir(target) else None
    with pytest.raises(RR.ForbiddenDestination):
        RR.store_receipt(executed_receipt(), target)
    if before is not None:
        assert snapshot(target) == before


@pytest.mark.parametrize("sub", ["tools", "tests", "sandbox", "research", "engine/bridge",
                                 "engine", ".", "reviews", "quarantine", "packages/_probe"])
def test_store_refuses_every_in_repository_destination_but_the_receipts_root(sub):
    """Inside the repository the store writes under receipts/ only; a receipt
    anywhere else in the tree is a record out of place."""
    target = os.path.join(ROOT, sub)
    existed = os.path.isdir(target)
    before = snapshot(target) if existed else None
    with pytest.raises(RR.ForbiddenDestination) as exc:
        RR.store_receipt(executed_receipt(), target)
    assert "receipts" in str(exc.value)
    if existed:
        assert snapshot(target) == before
    else:
        assert not os.path.exists(target)


def test_store_destination_predicate_accepts_the_receipts_root_and_outside(tmp_path):
    """Positive side of the predicate, checked without writing into the repository."""
    RR._assert_destination_allowed(os.path.join(RR.RECEIPTS_ROOT, "k.json"), RR.RECEIPTS_ROOT)
    RR._assert_destination_allowed(str(tmp_path / "k.json"), str(tmp_path))


def _open_modes(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    modes, attrs, imports = [], set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            modes.append(mode)
        if isinstance(node, ast.Attribute):
            # os.replace / shutil.rmtree are destructive; str.replace is not,
            # so the base name is part of the signature.
            base = node.value.id if isinstance(node.value, ast.Name) else None
            if base in ("os", "shutil", "pathlib", "Path"):
                attrs.add(node.attr)
            elif node.attr in ALWAYS_BANNED:
                attrs.add(node.attr)
        if isinstance(node, ast.Import):
            imports.update(n.name for n in node.names)
    return modes, attrs, imports


ALWAYS_BANNED = {"write_text", "write_bytes", "rmtree", "truncate", "unlink", "rmdir"}
BANNED_ATTRS = ALWAYS_BANNED | {"remove", "removedirs", "rename", "replace"}


def test_bridge_modules_have_no_destructive_call():
    """STRUCTURAL: the store creates with mode 'x' only; nothing else writes."""
    modes, attrs, imports = _open_modes(os.path.join(BRIDGE, "run_receipt.py"))
    assert "x" in modes, "the exclusive-creation write is gone"
    assert all(m in (None, "r", "rb", "x") for m in modes), modes
    for path in ("work_order.py", "common.py", "__init__.py"):
        m, a, i = _open_modes(os.path.join(BRIDGE, path))
        assert all(x in (None, "r", "rb") for x in m), (path, m)
        attrs |= a
        imports |= i
    assert not (attrs & BANNED_ATTRS), attrs & BANNED_ATTRS
    assert "shutil" not in imports


# ---------------------------------------------------------------------------
# 16. the checker
# ---------------------------------------------------------------------------

def test_checker_fails_when_a_receipt_digest_resolves_to_no_order(tmp_path):
    o = good_order()
    other = good_order(task_id="FIXTURE-TASK-002")
    od, rd, _ = workspace(tmp_path, [o], [good_receipt(other)])
    out = run_checker(od, rd)
    assert out.returncode == 1
    assert "resolves to no committed work order" in out.stdout


def test_checker_fails_when_receipt_and_order_disagree_on_task_id(tmp_path):
    o = good_order()
    od, rd, _ = workspace(tmp_path, [o], [good_receipt(o, task_id="FIXTURE-TASK-002")])
    out = run_checker(od, rd)
    assert out.returncode == 1 and "differs from the referenced order" in out.stdout


def test_checker_fails_when_a_receipt_file_stem_is_not_its_key(tmp_path):
    o = good_order()
    od, rd, _ = workspace(tmp_path, [o])
    write(os.path.join(rd, "latest.json"), good_receipt(o))
    out = run_checker(od, rd)
    assert out.returncode == 1 and "named by its idempotency_key" in out.stdout
    os.remove(os.path.join(rd, "latest.json"))
    r = good_receipt(o)
    write(os.path.join(rd, ("f" * 64) + ".json"), r)
    out = run_checker(od, rd)
    assert out.returncode == 1 and "file stem does not equal idempotency_key" in out.stdout


def test_checker_fails_when_an_order_file_stem_is_not_its_task_id(tmp_path):
    od, rd, _ = workspace(tmp_path)
    write(os.path.join(od, "current.json"), good_order())
    out = run_checker(od, rd)
    assert out.returncode == 1 and "is not its task_id" in out.stdout


def test_checker_refuses_example_records_in_the_live_directories(tmp_path):
    ex_o, ex_r = load(EX_ORDER), load(EX_RECEIPT)
    od, rd, _ = workspace(tmp_path, [ex_o], [ex_r])
    out = run_checker(od, rd)
    assert out.returncode == 1
    assert out.stdout.count("holds RECORD records only") == 2, out.stdout


def test_checker_refuses_record_kind_records_under_examples(tmp_path):
    o = good_order()
    od, rd, ed = workspace(tmp_path, examples=[o, good_receipt(o)])
    out = run_checker(od, rd, ed)
    assert out.returncode == 1
    assert out.stdout.count("holds EXAMPLE records only") == 2, out.stdout


def test_checker_passes_the_shipped_examples_through_the_examples_flag(tmp_path):
    od, rd, _ = workspace(tmp_path)
    out = run_checker(od, rd, EXAMPLES)
    assert out.returncode == 0, out.stdout
    assert "example_orders=1 example_receipts=1" in out.stdout


def test_checker_fails_on_a_duplicate_task_id_or_digest(tmp_path):
    o = good_order()
    od, rd, _ = workspace(tmp_path, [o])
    write(os.path.join(od, "FIXTURE-TASK-001-copy.json"), o)
    out = run_checker(od, rd)
    assert out.returncode == 1
    assert "already used" in out.stdout


def test_checker_fails_on_an_invalid_order_or_receipt(tmp_path):
    o = good_order(scientific_status_change_authorized=True)
    od, rd, _ = workspace(tmp_path, [o], [good_receipt(o, scientific_status_change="CHANGED")])
    out = run_checker(od, rd)
    assert out.returncode == 1
    assert "only permitted value is false" in out.stdout
    assert "FW-NO-RECEIPT-PROMOTION" in out.stdout
    assert out.stdout.strip().splitlines()[-1].endswith("problems=2")


def test_checker_fails_on_an_unreadable_file(tmp_path):
    od, rd, _ = workspace(tmp_path)
    with open(os.path.join(rd, ("a" * 64) + ".json"), "w") as f:
        f.write("{not json")
    out = run_checker(od, rd)
    assert out.returncode == 1 and "unreadable" in out.stdout


def test_checker_refuses_duplicate_keys_in_the_file_text(tmp_path):
    """The bytes are the record: a text whose first "status" says ACCEPTED is
    not made honest by a parser that keeps the last one."""
    o = good_order()
    r = good_receipt(o)
    text = RR.to_json(r).replace('"schema": "q0.bridge.run_receipt/v1"',
                                 '"status": "ACCEPTED",\n  "schema": "q0.bridge.run_receipt/v1"', 1)
    assert text.count('"status"') == 2
    od, rd, _ = workspace(tmp_path, [o])
    write(os.path.join(rd, r["idempotency_key"] + ".json"), None, text=text)
    out = run_checker(od, rd)
    assert out.returncode == 1 and "duplicate key 'status'" in out.stdout, out.stdout
    with pytest.raises(ValueError):
        RR.load_receipt(os.path.join(rd, r["idempotency_key"] + ".json"))


def test_strict_loader_refuses_nan(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        C.load_json_strict(str(p))
    assert json.load(open(p)) is not None       # the lenient parser would have taken it


@pytest.mark.parametrize("variant", ["indent4", "unsorted", "no_newline", "compact", "crlf"])
def test_checker_refuses_non_canonical_bytes(tmp_path, variant):
    o = good_order()
    r = good_receipt(o)
    canon = RR.to_json(r)
    text = {
        "indent4": json.dumps(r, indent=4, sort_keys=True) + "\n",
        "unsorted": json.dumps({k: r[k] for k in reversed(list(r))}, indent=2) + "\n",
        "no_newline": canon.rstrip("\n"),
        "compact": json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n",
        "crlf": canon.replace("\n", "\r\n"),
    }[variant]
    assert text != canon
    od, rd, _ = workspace(tmp_path, [o])
    write(os.path.join(rd, r["idempotency_key"] + ".json"), None, text=text)
    out = run_checker(od, rd)
    assert out.returncode == 1 and "not the canonical serialisation" in out.stdout, out.stdout


@pytest.mark.parametrize("case", ["receipt_in_orders", "order_in_receipts", "wrong_schema",
                                  "no_schema", "not_an_object", "upper_extension",
                                  "stray_file", "subdirectory", "unknown_in_examples"])
def test_checker_skips_nothing_silently(tmp_path, case):
    """Everything in a live directory is accounted for; nothing is ignored."""
    o = good_order()
    od, rd, ed = workspace(tmp_path, [o], examples=[load(EX_ORDER), load(EX_RECEIPT)])
    if case == "receipt_in_orders":
        write(os.path.join(od, "smuggled.json"), executed_receipt(o, status=RR.ACCEPTED))
        fragment = "every file under the orders directory is a work order"
    elif case == "order_in_receipts":
        write(os.path.join(rd, "FIXTURE-TASK-001.json"), o)
        fragment = "record under the receipts directory"
    elif case == "wrong_schema":
        write(os.path.join(od, "FIXTURE-TASK-002.json"),
              good_order(task_id="FIXTURE-TASK-002", schema="q0.bridge.work_order/v2"))
        fragment = "every file under the orders directory is a work order"
    elif case == "no_schema":
        o2 = good_order(task_id="FIXTURE-TASK-002")
        del o2["schema"]
        write(os.path.join(od, "FIXTURE-TASK-002.json"), WO.with_digest(o2))
        fragment = "every file under the orders directory is a work order"
    elif case == "not_an_object":
        write(os.path.join(rd, ("b" * 64) + ".json"), None, text="[]\n")
        fragment = "every file under the receipts directory is a receipt"
    elif case == "upper_extension":
        r = good_receipt(o)
        write(os.path.join(rd, r["idempotency_key"] + ".JSON"), r)
        fragment = "extension is not lowercase .json"
    elif case == "stray_file":
        write(os.path.join(rd, "notes.txt"), None, text="hello\n")
        fragment = "not a record file"
    elif case == "subdirectory":
        os.makedirs(os.path.join(od, "2026"))
        write(os.path.join(od, "2026", "FIXTURE-TASK-002.json"), good_order(task_id="FIXTURE-TASK-002"))
        fragment = "a subdirectory"
    else:
        write(os.path.join(ed, "held.json"), {"schema": "q0.bridge.something/v1"})
        fragment = "neither a work order nor a run receipt"
    out = run_checker(od, rd, ed)
    assert out.returncode == 1 and fragment in out.stdout, (case, out.stdout)


def test_checker_counts_readme_files_as_skipped_and_nothing_else(tmp_path):
    o = good_order()
    od, rd, _ = workspace(tmp_path, [o], [good_receipt(o)])
    write(os.path.join(od, "README.md"), None, text="# orders\n")
    write(os.path.join(rd, "README.md"), None, text="# receipts\n")
    out = run_checker(od, rd)
    assert out.returncode == 0, out.stdout
    assert "skipped=2" in out.stdout


def test_checker_flags_a_held_delivery_without_its_first_receipt(tmp_path):
    r1, r2 = executed_receipt(), executed_receipt(**{"verification.tests_passed": 4})
    RR.store_receipt(r1, str(tmp_path / "receipts"))
    held = RR.store_receipt(r2, str(tmp_path / "receipts"), now=TS).path
    od, rd, _ = workspace(tmp_path, [executable_order()])
    os.remove(os.path.join(rd, r1["idempotency_key"] + ".json"))
    out = run_checker(od, rd)
    assert out.returncode == 1 and "has no first receipt" in out.stdout
    # a held file whose contents equal the first is not a conflict
    write(os.path.join(rd, r1["idempotency_key"] + ".json"), r1)
    h = load(held)
    h["delivered"] = r1
    h["held_body_sha256"] = r1["body_sha256"]
    write(held, h)
    out = run_checker(od, rd)
    assert out.returncode == 1 and "identical retry is not a conflict" in out.stdout


def test_checker_flags_a_misnamed_held_delivery(tmp_path):
    r1, r2 = executed_receipt(), executed_receipt(**{"verification.tests_passed": 4})
    RR.store_receipt(r1, str(tmp_path / "receipts"))
    held = RR.store_receipt(r2, str(tmp_path / "receipts"), now=TS).path
    od, rd, _ = workspace(tmp_path, [executable_order()])
    os.rename(held, os.path.join(rd, "conflict.json"))
    out = run_checker(od, rd)
    assert out.returncode == 1 and "must be named" in out.stdout


def test_checker_writes_nothing(tmp_path):
    o = good_order()
    bad = good_order(task_id="FIXTURE-TASK-BAD", scientific_status_change_authorized=True)
    od, rd, ed = workspace(tmp_path, [o, bad], [good_receipt(o), good_receipt(bad)],
                           examples=[load(EX_ORDER), load(EX_RECEIPT), good_receipt(o)])
    before = snapshot(str(tmp_path))
    out = run_checker(od, rd, ed)
    assert out.returncode == 1
    assert snapshot(str(tmp_path)) == before
    assert sorted(os.listdir(tmp_path)) == ["examples", "orders", "receipts"]
    # and on the repository's own directories
    before = {d: snapshot(os.path.join(BRIDGE, d)) for d in ("orders", "receipts", "examples")}
    subprocess.run([sys.executable, CHECKER], cwd=ROOT, capture_output=True)
    assert {d: snapshot(os.path.join(BRIDGE, d)) for d in before} == before


def test_checker_module_has_no_write_call():
    """STRUCTURAL: tools/bridge_check.py opens files for reading only."""
    modes, attrs, imports = _open_modes(CHECKER)
    assert all(m in (None, "r", "rb") for m in modes), modes
    assert not (attrs & BANNED_ATTRS), attrs & BANNED_ATTRS
    assert "shutil" not in imports


def test_checker_reports_a_null_key_receipt_instead_of_crashing(tmp_path):
    """End to end: garbage in a live directory is a problem line and a summary,
    never a traceback."""
    o = executable_order()
    r = executed_receipt(o, **{"review.independence_credit": 1,
                               "review.technical_verdict": RR.AMEND_REQUIRED,
                               "review.organizational_independence_record_id": REVIEW_ID})
    r["idempotency_key"] = None
    r = rehash_receipt_body(r)
    od, rd, _ = workspace(tmp_path, [o])
    write(os.path.join(rd, ("c" * 64) + ".json"), r)
    out = run_checker(od, rd)
    assert out.returncode == 1
    assert out.stderr == "", out.stderr
    assert "idempotency_key None is not a sha256" in out.stdout
    assert out.stdout.strip().splitlines()[-1].startswith("orders=1 receipts=1")
    assert "independence_credit 1" not in out.stdout   # notes are printed for valid records only


def test_checker_detects_edits_and_deletions_against_git_head(tmp_path):
    """A forger who recomputes body_sha256 defeats the hash; git is the second line."""
    repo, odir, rdir, _ = git_repo(tmp_path)
    o = executable_order()
    write(os.path.join(odir, f"{o['task_id']}.json"), o)
    r_keep = executed_receipt(o)
    r_edit = executed_receipt(o, run_id="FIXTURE-RUN-1")
    p_keep = RR.store_receipt(r_keep, rdir).path
    p_edit = RR.store_receipt(r_edit, rdir).path
    git_commit(repo, "receipts")

    def problems():
        return BC.check(odir, rdir, None, root=str(repo), check_git=True)[0]

    assert problems() == []
    edited = load(p_edit)
    edited["verification"]["tests_passed"] = 1000
    write(p_edit, RR.finalize(edited))                # self-consistent hash and key
    assert any("differs from its git HEAD content" in p for p in problems())
    os.remove(p_keep)
    assert any("deleted from the working tree" in p for p in problems())


def test_checker_detects_committed_rewrites_and_removals_in_history(tmp_path):
    """On a pull-request head, HEAD equals the working tree. The freeze is
    history: a rewrite or removal that was committed still fails."""
    repo, odir, rdir, edir = git_repo(tmp_path)
    o = executable_order()
    write(os.path.join(odir, f"{o['task_id']}.json"), o)
    r1 = executed_receipt(o)
    p1 = RR.store_receipt(r1, rdir).path
    for name in ("work_order.example.json", "run_receipt.example.json"):
        write(os.path.join(edir, name), load(os.path.join(EXAMPLES, name)))
    git_commit(repo, "order, receipt, examples")

    def problems():
        return BC.check(odir, rdir, edir, root=str(repo), check_git=True)[0]

    assert problems() == []
    # 1. a conflicting receipt written over the first, then committed
    r2 = executed_receipt(o, **{"verification.tests_passed": 1000,
                                "review.technical_verdict": RR.PASS_TECHNICAL,
                                "review.authorship_exposure": "author-side, fixture"})
    assert r2["idempotency_key"] == r1["idempotency_key"]
    write(p1, r2)
    git_commit(repo, "rewrite the receipt")
    found = [p for p in problems() if "rewritten or removed in commit" in p]
    assert found and os.path.basename(p1) in found[0], problems()
    # 2. the receipt deleted, then committed
    os.remove(p1)
    git_commit(repo, "delete the receipt")
    assert any("rewritten or removed in commit" in p and os.path.basename(p1) in p
               for p in problems())
    # 3. an order edited in place, re-digested, committed
    o2 = WO.with_digest({**o, "limits": {"wall_time_seconds": 999999, "memory_mib": 999999,
                                         "network_policy": "UNRESTRICTED"}})
    write(os.path.join(odir, f"{o['task_id']}.json"), o2)
    git_commit(repo, "widen the order")
    assert any("rewritten or removed in commit" in p and f"{o['task_id']}.json" in p
               for p in problems())
    # 4. an example edited in place, re-digested, committed
    ex = load(EX_ORDER)
    ex2 = WO.with_digest({**ex, "scope": {**ex["scope"], "allowed_paths": ["research/"]}})
    write(os.path.join(edir, "work_order.example.json"), ex2)
    exr = load(EX_RECEIPT)
    write(os.path.join(edir, "run_receipt.example.json"),
          RR.finalize({**exr, "work_order_digest": ex2["work_order_digest"]}))
    git_commit(repo, "edit the examples")
    assert any("rewritten or removed in commit" in p and "work_order.example.json" in p
               for p in problems())
    # the history check is what caught them: HEAD equals the working tree throughout
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                          capture_output=True, text=True).stdout.strip() == ""


def test_checker_rejects_an_unknown_flag():
    out = subprocess.run([sys.executable, CHECKER, "--promote"], capture_output=True, text=True)
    assert out.returncode == 2


def test_checker_summary_line_is_last_and_one_line(tmp_path):
    od, rd, _ = workspace(tmp_path)
    out = run_checker(od, rd)
    lines = out.stdout.strip().splitlines()
    assert lines == ["orders=0 receipts=0 held=0 example_orders=0 example_receipts=0 "
                     "skipped=0 notes=0 problems=0"]


# ---------------------------------------------------------------------------
# 17. what the code says about itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    os.path.join(BRIDGE, "__init__.py"), os.path.join(BRIDGE, "common.py"),
    os.path.join(BRIDGE, "work_order.py"), os.path.join(BRIDGE, "run_receipt.py"),
    CHECKER, __file__,
])
def test_every_module_docstring_says_what_it_does_not_establish(path):
    doc = ast.get_docstring(ast.parse(open(path, encoding="utf-8").read())) or ""
    assert "not establish" in doc.lower() or "does not mean" in doc.lower(), path
    assert "not deployed" in doc.lower(), path


def test_agents_md_points_and_does_not_legislate():
    text = open(os.path.join(ROOT, "AGENTS.md"), encoding="utf-8").read()
    # The original bridge navigation stays short. The two later, explicitly
    # linked operational addenda do not silently replace that bridge contract.
    prefix, separator, addenda = text.partition("\n## Support withdrawal and scheduling — 2026-09-21 operational addendum")
    assert separator, "missing current withdrawal navigation"
    lines = prefix.strip().splitlines()
    assert len(lines) <= 45, len(lines)
    for heading, link in (
        ("Read [OP-WITHDRAWAL-20260921-v1.0]", "governance/withdrawal/PROTOCOL.md"),
        ("## Governance rollout compatibility — scoped correction v1.1",
         "governance/rollout/OP-ROLLOUT-AUDIT-20260921-v1.1.md"),
    ):
        assert heading in addenda and f"({link})" in addenda
        assert os.path.isfile(os.path.join(ROOT, link)), link
    for needle in ("CLAUDE.md", "180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8",
                   "1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc", "DG-EXEC-20260918-49291487",
                   "NOT DEPLOYED", "engine/bridge/", "99_DO_NOT_OPEN", "No status moves",
                   "enforcement"):
        assert needle in text, needle


def _text(path):
    return open(path, encoding="utf-8").read()


def test_no_text_claims_a_verification_this_repository_cannot_do():
    """LABELLING: the sentences the adversaries flagged stay out."""
    readme = _text(os.path.join(BRIDGE, "README.md"))
    agents = _text(os.path.join(ROOT, "AGENTS.md"))
    rr = _text(os.path.join(BRIDGE, "run_receipt.py"))
    wo = _text(os.path.join(BRIDGE, "work_order.py"))
    init = _text(os.path.join(BRIDGE, "__init__.py"))
    for text, name in ((readme, "README.md"), (rr, "run_receipt.py")):
        assert "can never set it" not in text, name
        assert "cannot verify an acceptance" in text, name
    for text, name in ((readme, "README.md"), (agents, "AGENTS.md")):
        assert "a valid work order is what authorizes work" not in text, name
        assert "cannot" in text and "confers nothing" in text, name
    assert "would authorize a bounded task" not in init
    assert "authorization of a bounded task" in init
    # the "pinned execution input" sentence is the handoff's, not governance's
    assert "governance/GIT_ADAPTATION.md``: \"a GitHub" not in wo
    assert "DRIVE_GITHUB_EXECUTION_HANDOFF.md" in wo and "not ``governance/GIT_ADAPTATION.md``" in wo
    gov = _text(os.path.join(ROOT, "governance", "GIT_ADAPTATION.md"))
    assert "pinned execution input" not in gov
    # the problem text for ACCEPTED says transcribed, not verified
    r = accepted_receipt(accepted_by_drive_record=None)
    msg = [p for p in RR.validate_run_receipt(r) if "ACCEPTED" in p][0]
    assert "cannot verify" in msg and "transcribed" in msg
