

class TestZeroCrossCheckRejects:
    """Negative controls: the cross-check must flag planted defects (no script change needed)."""

    def _run(self, monkeypatch, tmp_path, edit):
        pytest.importorskip("flint")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import cross_check_zeros_arb as cc
        text = open(os.path.join(ROOT, "data", "riemann_zeros", "first_100_zeros.json")).read()
        bad = tmp_path / "zeros.json"
        bad.write_text(edit(text))
        monkeypatch.setattr(cc, "DATA", bad)
        return cc.cross_check()

    def test_rejects_last_digit_change(self, monkeypatch, tmp_path):
        r = self._run(monkeypatch, tmp_path,
                      lambda t: t.replace("150.05352042078488035", "150.05352042078488036"))
        assert r["summary"]["all_within_half_unit"] is False
        assert [z["n"] for z in r["zeros"] if not z["within_half_unit"]] == [53]

    def test_rejects_swapped_pair(self, monkeypatch, tmp_path):
        a, b = "69.546401711173979253", "72.067157674481907583"
        r = self._run(monkeypatch, tmp_path,
                      lambda t: t.replace(a, "@").replace(b, a).replace("@", b))
        assert r["summary"]["bracket_count_mismatches"] == [16]
