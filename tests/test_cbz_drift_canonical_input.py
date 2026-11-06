"""R3 canonical-input regression controls using real local Git repositories.

The original missing-registry witness method below is retained verbatim from
OpenAI/Codex chat 01a103c8's test_cbz_drift_adversarial_01a103c8.py, lines 76-87,
commit 9a5d313bc3a2fc880044498b8e48288d53437ad4 (blob c8894d8ef40cd1b9a6ea7ebfb2a18f08e3183a81).
The complete original six-case qualification suite remains at that commit.
Additional controls cover diagnostics, explicit historical refs and the real CLI.
These fixtures establish engineering behavior only, not federation/G7 closure.
"""
import json
import pathlib
import subprocess
import sys

from test_cbz_drift import ENV, TOOL, FedCase, codes


class CanonicalInputTests(FedCase):
    def declare(self, path, **extra):
        spec = dict(repository='fed/reg', path=path, **extra)
        workspace = {
            'repositories': [{'full_name': 'fed/' + n} for n in ('ws', 'a', 'b', 'reg')],
            'canonical_inputs': {'repository_registry': spec},
        }
        self.fed.write_seed(extra={'workspace/repositories.json': json.dumps(workspace)})

    def assert_missing(self, report, ref, path):
        matches = [f for f in report['findings'] if f['level'] == 'BLOCK' and f['code'] == 'INPUT-MISSING']
        self.assertEqual(len(matches), 1, report['findings'])
        finding = matches[0]
        self.assertEqual(finding['label'], 'repository_registry')
        self.assertEqual(finding['repo'], 'fed/reg')
        self.assertEqual(finding['ref'], ref)
        self.assertEqual(finding['path'], path)
        for identity in ('repository_registry', 'fed/reg', ref, path):
            self.assertIn(identity, finding['message'])
        self.assertEqual(report['blocking'], 1)

    def cli(self):
        out = pathlib.Path(self.tmp.name) / 'report.json'
        mode = ['-O'] if sys.flags.optimize else []
        proc = subprocess.run(
            [sys.executable, '-B', *mode, str(TOOL / 'crawl.py'), '--seed', 'fed/ws',
             '--search', self.tmp.name, '--json', str(out)], capture_output=True, env=ENV)
        self.assertEqual(proc.stderr, b'', proc.stderr)
        return proc, json.loads(out.read_text())

    def test_absent_declared_canonical_registry_blocks_instead_of_zero_coverage(self):
        self.fed.build()
        registry = {
            "repositories": [{"full_name": "fed/" + n} for n in ("ws", "a", "b", "reg")],
            "canonical_inputs": {"repository_registry": {
                "repository": "fed/reg", "path": "absent-registry.json",
                "observed_blob": "0" * 40,
            }},
        }
        self.fed.write_seed(extra={"workspace/repositories.json": json.dumps(registry)})
        report = self.fed.crawl()
        self.assertGreater(report["blocking"], 0, "Declared registry disappeared without blocking the gate")

    def test_missing_declared_input_without_observed_blob_identifies_selected_ref(self):
        self.fed.build()
        self.declare('absent-registry.json')
        self.assert_missing(self.fed.crawl(), self.fed.pins['reg'], 'absent-registry.json')

    def test_missing_input_at_tip_uses_tip_identity(self):
        self.fed.build()
        tip = self.fed.commit('reg', {}, remove=('registry.json',))
        self.assert_missing(self.fed.crawl(at='tips'), tip, 'registry.json')

    def test_missing_input_at_override_uses_override_identity(self):
        self.fed.build()
        candidate = self.fed.commit('reg', {}, remove=('registry.json',))
        self.assert_missing(self.fed.crawl(overrides={'fed/reg': candidate}), candidate, 'registry.json')

    def test_explicit_missing_historical_input_is_not_rebound_to_present_tip(self):
        self.fed.build()
        historical = self.fed.commit('reg', {}, remove=('registry.json',))
        self.fed.commit('reg', {'registry.json': json.dumps(self.fed.registry)})
        self.declare('registry.json', ref=historical)
        self.assert_missing(self.fed.crawl(at='tips'), historical, 'registry.json')

    def test_present_historical_input_is_checked_after_current_path_disappears(self):
        self.fed.build()
        historical = self.fed.pins['reg']
        current = self.fed.commit('reg', {}, remove=('registry.json',))
        self.declare('registry.json', ref=historical)
        report = self.fed.crawl(at='tips')
        self.assertEqual(codes(report, 'BLOCK'), [])
        self.assertEqual(report['stats']['registry_artifacts_verified'], 1)
        self.assertEqual(next(r['ref'] for r in report['repositories'] if r['repository'] == 'fed/reg'), current)

    def test_present_declared_registry_cli_preserves_artifact_coverage(self):
        self.fed.build()
        proc, report = self.cli()
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(report['blocking'], 0)
        self.assertEqual(report['stats']['registry_artifacts_verified'], 1)
        self.assertEqual(report['stats']['lean_files'], 2)

    def test_missing_declared_registry_cli_returns_one_with_identity(self):
        self.fed.build()
        self.declare('absent-registry.json')
        proc, report = self.cli()
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assert_missing(report, self.fed.pins['reg'], 'absent-registry.json')
        self.assertIn(b'BLOCK INPUT-MISSING', proc.stdout)
        for identity in ('repository_registry', 'fed/reg', self.fed.pins['reg'], 'absent-registry.json'):
            self.assertIn(identity.encode(), proc.stdout)

    def test_never_declared_optional_discovery_files_remain_optional(self):
        self.fed.commit('ws', {'README.md': 'No optional discovery manifests declared.\n'})
        report = self.fed.crawl()
        self.assertEqual(codes(report, 'BLOCK'), [])
        self.assertEqual(report['blocking'], 0)
        self.assertEqual(report['stats'].get('registry_artifacts_verified', 0), 0)
        self.assertEqual([r['repository'] for r in report['repositories']], ['fed/ws'])
