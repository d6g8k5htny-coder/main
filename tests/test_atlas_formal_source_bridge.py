"""Run the real offline assembler against frozen inputs and adversarial changes."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/'tools/atlas_formal_source_bridge.py'
PROFILES={
    '8228':('8228e896c728d95ab94391899320e9d44ab7cdfc',17521,
            '1a09020f0706abd506824f54fca76df9daa533749350b9ab6b818e0bbe9ed744',15),
    'a22f':('a22f06a3a7b9ab451e9e344d7b2345d2e2ab0553',18104,
            '38eefbfdd84658cd0f6e18bd7d714458a7522b1021f73b1075e7972a63b9bcb0',13),
}

class SourceBridgeCLI(unittest.TestCase):
    profile='8228'
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        profile_root=ROOT/'tests/fixtures/atlas_formal_bridge_profiles'
        manifest_raw=(profile_root/(self.profile+'-manifest.json')).read_bytes()
        self.assertEqual(hashlib.sha256(manifest_raw).hexdigest(),PROFILES[self.profile][2])
        manifest=json.loads(manifest_raw)
        paths=list(manifest['files'])+['formal/manifest.json',
            'docs/site/dependency-source/GRAPH.json','docs/site/atlas-formal-evidence-data.mjs',
            'tests/fixtures/atlas_formal_projection.json']
        paths += [str(p.relative_to(ROOT)) for folder in [
            ROOT/'tests/fixtures/atlas_formal_build_3b39'] for p in folder.iterdir() if p.is_file()]
        paths += ['docs/site/evidence/side24-main-3b39/'+name for name in ['receipt.json','elaborated-types.log','review.json','review.md','verified-execution.zip']]
        if self.profile=='a22f':paths += ['docs/site/evidence/side24-main-3b39/source-bridge-8228.json']
        for relative in set(paths):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if relative in {'tools/formal_gate_check.py','tests/test_formal_gate.py'}:
                # Frozen profiles retain their exact old controls as literal data;
                # the current validator runs, never these historical Python bytes.
                source=profile_root/(Path(relative).name+'.txt')
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),manifest['files'][relative])
            elif relative in {'formal/UniversalLaw.lean','formal/SCOPE.md','formal/reviews/TEMPLATE_alignment_review.json'}:
                # Historical bytes frozen when the live package grew its P15 modules (main#740);
                # identical across the 3b39, 8228 and a22f profiles.
                source=ROOT/'tests/fixtures/atlas_formal_build_3b39'/(Path(relative).name+'.txt')
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),manifest['files'][relative])
            else:
                source=ROOT/relative
            shutil.copyfile(source,target)
        (self.root/'formal/manifest.json').write_bytes(manifest_raw)
        for name in ['README.md','REVIEW_LANE.md']:
            source=(ROOT/'tests/fixtures/atlas_formal_build_3b39'/(name+'.txt')
                    if self.profile=='8228' else profile_root/('a22f-'+name+'.txt'))
            (self.root/'formal'/name).write_bytes(source.read_bytes())
        self.snapshot=json.loads((ROOT/'tests/fixtures/atlas_formal_projection.json').read_text())
        formal=next(s for s in self.snapshot['sources'] if s['repository']=='d6g8k5htny-coder/main' and s['path']=='formal/manifest.json')
        formal.update(commit=('93869215a82da60f014093e2ac28503865ed0aa3' if self.profile=='8228'
                              else '55961773d31a7b2b9906d8be73a26ad44c364dd6'),
                      blob=PROFILES[self.profile][0],bytes=PROFILES[self.profile][1])
        self.input=self.root/'current-projection.json'

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self,*extra):
        self.input.write_text(json.dumps(self.snapshot))
        mode=['-O'] if sys.flags.optimize else []
        return subprocess.run([sys.executable,'-B',*mode,str(TOOL),'--root',str(self.root),
                               '--snapshot',str(self.input),*extra],capture_output=True,text=True)

    def test_current_bridge_reproduces_original_evidence_and_records_only_two_control_changes(self):
        result=self.run_cli();self.assertEqual(result.returncode,0,result.stderr)
        record=json.loads((self.root/('docs/site/evidence/side24-main-3b39/source-bridge-'+self.profile+'.json')).read_text())
        self.assertEqual(record['projection_manifest']['sha256'],PROFILES[self.profile][2])
        self.assertEqual(record['historical_artifact']['manifest_sha256'],'8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090')
        self.assertEqual(record['historical_artifact']['receipt_sha256'],'1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0')
        self.assertEqual(set(record['changed_control_files']),{'tools/formal_gate_check.py','tests/test_formal_gate.py'})
        self.assertEqual(len(record['identical_formal_files']),PROFILES[self.profile][3])
        if self.profile=='a22f':
            self.assertEqual(set(record['changed_documentation_files']),{'formal/README.md','formal/REVIEW_LANE.md'})
            self.assertEqual(set(record['changed_target_annotations']),{'UniversalLaw.Side24.d3_below_d2','UniversalLaw.Side24.endpoints_in_unit_tenth'})
            self.assertNotIn('UniversalLaw.Side24.pairing_sum_six',record['changed_target_annotations'])
        self.assertTrue(record['historical_evidence_reproduced_exactly'])
        self.assertEqual(record['scientific_effect'],'NONE')
        before=(self.root/'docs/site/atlas-formal-source-bridge-data.mjs').read_bytes()
        checked=self.run_cli('--check');self.assertEqual(checked.returncode,0,checked.stderr)
        self.assertEqual((self.root/'docs/site/atlas-formal-source-bridge-data.mjs').read_bytes(),before)

    def test_corrupt_generated_record_is_refused_by_check(self):
        initial=self.run_cli();self.assertEqual(initial.returncode,0,initial.stderr)
        path=self.root/('docs/site/evidence/side24-main-3b39/source-bridge-'+self.profile+'.json')
        path.write_bytes(path.read_bytes()+b' ')
        result=self.run_cli('--check');self.assertNotEqual(result.returncode,0)
        self.assertIn('generated bridge differs',result.stderr)

    def test_another_observation_cannot_overwrite_an_existing_qualified_record(self):
        initial=self.run_cli();self.assertEqual(initial.returncode,0,initial.stderr)
        path=self.root/('docs/site/evidence/side24-main-3b39/source-bridge-'+self.profile+'.json')
        original=path.read_bytes()
        self.snapshot['graph']['identity']['commit']='f'*40
        for item in self.snapshot['sources']:
            if item['repository']=='d6g8k5htny-coder/Math-':item['commit']='f'*40
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('immutable bridge record differs',result.stderr)
        self.assertEqual(path.read_bytes(),original)

    def test_crlf_altered_derivation_is_refused_without_rewriting_its_bytes(self):
        initial=self.run_cli();self.assertEqual(initial.returncode,0,initial.stderr)
        path=self.root/('docs/site/evidence/side24-main-3b39/source-bridge-'+self.profile+'.json')
        altered=path.read_bytes().replace(b'\n',b'\r\n');path.write_bytes(altered)
        for arguments,message in [(('--check',),'generated bridge differs'),
                                  ((),'immutable bridge record differs')]:
            with self.subTest(arguments=arguments):
                result=self.run_cli(*arguments)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)
                self.assertEqual(path.read_bytes(),altered)

    def test_every_changed_registered_file_is_refused_without_publishing_a_bridge(self):
        manifest=json.loads((self.root/'formal/manifest.json').read_text())
        for relative in manifest['files']:
            with self.subTest(relative=relative):
                path=self.root/relative;original=path.read_bytes();path.write_bytes(original+b'\n')
                result=self.run_cli();self.assertNotEqual(result.returncode,0)
                self.assertIn('source hash mismatch: '+relative,result.stderr)
                self.assertFalse((self.root/'docs/site/atlas-formal-source-bridge-data.mjs').exists())
                path.write_bytes(original)

    def test_resigned_new_manifest_is_not_an_approved_bridge(self):
        path=self.root/'formal/manifest.json';m=json.loads(path.read_text());m['targets'][0]['title']='Changed statement'
        path.write_text(json.dumps(m))
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('unrecognized projection manifest',result.stderr)

    def test_changed_historical_control_data_is_not_executed_or_accepted(self):
        path=self.root/'tests/fixtures/atlas_formal_build_3b39/formal_gate_check.py.txt'
        path.write_text('raise RuntimeError("fixture must never execute")\n')
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('historical input identity',result.stderr)

    def test_changed_original_evidence_or_archive_is_refused(self):
        for relative in ['docs/site/atlas-formal-evidence-data.mjs',
                         'docs/site/evidence/side24-main-3b39/verified-execution.zip']:
            with self.subTest(relative=relative):
                path=self.root/relative;original=path.read_bytes();path.write_bytes(original+b'\n')
                result=self.run_cli();self.assertNotEqual(result.returncode,0)
                self.assertIn('identity',result.stderr);path.write_bytes(original)

    def test_duplicate_manifest_inventory_is_not_silently_selected(self):
        formal=next(s for s in self.snapshot['sources'] if s['repository']=='d6g8k5htny-coder/main' and s['path']=='formal/manifest.json')
        self.snapshot['sources'].append(dict(formal))
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('projection manifest inventory',result.stderr)

    def test_changed_graph_scope_does_not_reuse_a_document_fingerprint(self):
        self.snapshot['graph']['data']['nodes']['math.side24-coefficient']['scope']='Full proof'
        result=self.run_cli();self.assertNotEqual(result.returncode,0)
        self.assertIn('graph record',result.stderr)

class CorrectedManifestBridgeCLI(SourceBridgeCLI):
    profile='a22f'

if __name__=='__main__':unittest.main()
