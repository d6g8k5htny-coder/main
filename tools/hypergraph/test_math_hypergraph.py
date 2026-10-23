"""Tests for the math artifact hypergraph. Organization only: credit 0, OBL OPEN."""
import importlib.util, json, sys
from pathlib import Path
import pytest

pytest.importorskip("hyperdb")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("math_hypergraph", HERE / "math_hypergraph.py")
mh = importlib.util.module_from_spec(spec); sys.modules["math_hypergraph"] = mh; spec.loader.exec_module(mh)


@pytest.fixture(scope="module")
def built():
    return mh.build()


def test_counts(built):
    hg, G = built
    g = mh.summary(mh.gaps(hg, G))
    assert g["units"] == 344 and hg.num_e == 344
    assert (g["sourced"], g["no_source"]) == (211, 133)
    assert (g["implemented"], g["no_impl"]) == (296, 48)
    assert g["impl_placeholder_only"] == 30
    assert g["bound_units"] == 14
    assert g["graph_nodes"] == 49 and g["unbound_nodes_per_527"] == 28
    assert g["nodes_touched_by_no_unit"] == 13
    assert g["role_problems"] == 0


def test_ledger_header_matches_table():
    text = mh.LEDGER.read_text()
    assert "**211**" in text and "**133**" in text
    missing = sum(1 for r in mh.parse_ledger() if r["src"].upper().startswith("MISSING"))
    assert missing == 133


def test_every_edge_fail_closed(built):
    hg, _ = built
    for e in hg.all_e:
        d = hg.e(e)
        assert d["credit"] == 0 and d["obl"] == "OPEN"


def test_validator_flags_missing_unit_role(built):
    hg, _ = built
    hg2 = mh.HypergraphDB()
    hg2.add_v("a", {"type": "source_file"}); hg2.add_v("b", {"type": "implementation"})
    hg2.add_e(("a", "b"), {"row": 999, "roles": {"a": "source"}})
    probs = mh.validate(hg2)
    assert any("missing role unit" in p["problem"] for p in probs)
    assert any("member without role b" in p["problem"] for p in probs)


def test_reach(built):
    hg, _ = built
    q = "node:math.d1-component.congruence-erratum"
    one = mh.reach(hg, q, 1)
    assert any(v.startswith("unit:") for v in one)
    assert one <= mh.reach(hg, q, 2)
    with pytest.raises(KeyError):
        mh.reach(hg, "node:does-not-exist", 1)


def test_isolated_vertex_workaround(built):
    hg, G = built
    iso = [v for v in hg.all_v if not hg.degree_v(v)]
    assert iso  # untouched graph nodes exist
    assert mh.nbr_vertices(hg, iso[0]) == set()
    assert mh.reach(hg, iso[0], 2) == set()


def test_shared_impl(built):
    hg, _ = built
    top = mh.shared_impl(hg, 3)
    assert len(top) == 3 and top[0][1] >= top[-1][1] >= 1


def test_cli(capsys):
    assert mh.main(["gaps"]) == 0
    assert json.loads(capsys.readouterr().out)["sourced"] == 211
    assert mh.main(["validate"]) == 0
    capsys.readouterr()
    assert mh.main(["reach", "node:nope"]) == 2


def test_committed_outputs_current(built, tmp_path):
    hg, G = built
    mh.write_outputs(hg, G, tmp_path)
    for f in ("math_hypergraph.json", "query_results.json"):
        assert (tmp_path / f).read_text() == (mh.DATA / f).read_text(), f"{f} stale: rerun build"
