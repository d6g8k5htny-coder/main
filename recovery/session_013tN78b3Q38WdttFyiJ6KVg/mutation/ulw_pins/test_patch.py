

class VerifyPinsShapeControls(unittest.TestCase):
    """One injected defect per case; each must fail with its own diagnostic."""

    def run_case(self, change, message, after=None):
        with tempfile.TemporaryDirectory(prefix="ulw-test-") as root:
            gitlinks, modules, pins_entries = consistent_parts()
            change(gitlinks, modules, pins_entries)
            pins = build_fixture(root, gitlinks, modules, pins_entries)
            if after is not None:
                after(root, pins)
            proc = run_checker(root, pins)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn(message, proc.stdout)

    def test_pins_shape_defects_fail(self):
        def set_schema(root, pins):
            with open(pins) as fh:
                doc = json.load(fh)
            doc["schema"] = "ulw.pins.v0"
            with open(pins, "w") as fh:
                json.dump(doc, fh)

        def clear(g, m, p):
            g.clear(); m.clear(); p.clear()

        cases = {
            "wrong schema": (lambda g, m, p: None, "pins.json: schema is 'ulw.pins.v0', expected 'ulw.pins.v1'",
                             set_schema),
            "empty repos": (clear, "pins.json: 'repos' must be a non-empty list", None),
            "missing key": (lambda g, m, p: p[0].pop("note"), "pins.json: entry 0 missing 'note'", None),
            "uppercase sha": (lambda g, m, p: p[0].update(pinned_sha="A" * 40),
                              "pins.json: repos/a: pinned_sha '%s' is not a full lowercase 40-hex SHA" % ("A" * 40),
                              None),
            "short sha": (lambda g, m, p: p[0].update(pinned_sha=SHA_A[:12]),
                          "pins.json: repos/a: pinned_sha '%s' is not a full lowercase 40-hex SHA" % SHA_A[:12],
                          None),
            "string flag": (lambda g, m, p: p[1].update(opt_in="true"),
                            "pins.json: repos/b: 'opt_in' must be a boolean", None),
            "duplicate path": (lambda g, m, p: p.append(dict(entry("a", SHA_A), name="a2")),
                               "pins.json: duplicate path 'repos/a'", None),
        }
        for name, (change, message, after) in cases.items():
            with self.subTest(case=name):
                self.run_case(change, message, after)

    def test_gitmodules_defects_fail(self):
        def orphan_section(root, pins):
            with open(os.path.join(root, ".gitmodules"), "a") as fh:
                fh.write('[submodule "orphan"]\n\turl = https://example.invalid/o.git\n')

        def remove_gitmodules(root, pins):
            os.remove(os.path.join(root, ".gitmodules"))

        cases = {
            "undeclared submodule": (lambda g, m, p: m.pop("repos/a"), "repos/a: not declared in .gitmodules", None),
            "url mismatch": (lambda g, m, p: m["repos/a"].update(url="https://example.invalid/other.git"),
                             "repos/a: .gitmodules url 'https://example.invalid/other.git' != pins url "
                             "'https://example.invalid/a.git'", None),
            "module not in pins": (lambda g, m, p: m.update({"repos/c": {"url": "https://example.invalid/c.git",
                                                                          "branch": "main"}}),
                                   "repos/c: declared in .gitmodules but not in pins.json", None),
            "section without path": (lambda g, m, p: None, ".gitmodules: submodule 'orphan' has no path",
                                     orphan_section),
            "missing .gitmodules": (lambda g, m, p: None, ".gitmodules is missing", remove_gitmodules),
        }
        for name, (change, message, after) in cases.items():
            with self.subTest(case=name):
                self.run_case(change, message, after)

    def test_git_failure_is_reported(self):
        with tempfile.TemporaryDirectory(prefix="ulw-test-") as root:
            pins = os.path.join(root, "pins.json")
            with open(pins, "w") as fh:
                json.dump({"schema": "ulw.pins.v1", "repos": [entry("a", SHA_A)]}, fh)
            proc = run_checker(root, pins)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("git ls-tree -r -z HEAD failed", proc.stdout)
