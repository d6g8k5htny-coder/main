"""Prospective hosted controls; no mathematical imports."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from tools import periodic_pilot_custody as custody

SUCCESS = """
import argparse,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--verify')
a=p.parse_args()
if a.output:
 d=pathlib.Path(a.output);d.mkdir()
 (d/'observations.json').write_bytes(b'{"synthetic":true}\\n')
 (d/'RUN.json').write_bytes(b'{"scientific_effect":"NONE"}\\n')
 sys.stdout.buffer.write(b'pilot\\x00\\xff\\n')
else:
 sys.stdout.buffer.write(b'verified\\n')
"""

class PilotCustodyTests(unittest.TestCase):
    def setUp(self):
        self.sandbox=tempfile.TemporaryDirectory();self.addCleanup(self.sandbox.cleanup)
        self.base=Path(self.sandbox.name)
        self.checkout=self.base/'checkout';self.checkout.mkdir()
        self.temp=self.base/'runner-temp';self.temp.mkdir()
        self.env={'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'github-hosted',
            'GITHUB_REPOSITORY':'d6g8k5htny-coder/main','GITHUB_SHA':'a'*40,
            'GITHUB_RUN_ID':'12345','GITHUB_RUN_ATTEMPT':'2','GITHUB_JOB':'pilot',
            'GITHUB_SERVER_URL':'https://github.com',
            'GITHUB_WORKFLOW_REF':'d6g8k5htny-coder/main/.github/workflows/periodic-h0-pilot.yml@refs/heads/main',
            'GITHUB_WORKFLOW_SHA':'b'*40}
        for name in custody.SOURCE_PATHS:
            p=self.checkout/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'# synthetic fixture\n')
        self.pilot=self.checkout/'experiments/periodic_h0/run_pilot.py'
        self.pilot.write_text(SUCCESS)
    def initialize(self,temp=None):
        return custody.initialize(self.checkout,temp or self.temp,self.env)
    def seal(self,root,temp=None,status='success',outcome='success'):
        return custody.seal(root,self.checkout,temp or self.temp,self.env,status,outcome)
    def test_success_has_raw_bytes_outputs_native_identity_and_both_commands(self):
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        r=self.seal(root)
        self.assertEqual(r['custody_exit'],0);self.assertTrue(r['pilot_success'])
        self.assertEqual(r['output_completeness'],'complete')
        x=json.loads((root/'execution.json').read_text())
        self.assertEqual([c['phase'] for c in x['commands']],['pilot','verify'])
        self.assertEqual((root/'logs/pilot.stdout.bin').read_bytes(),b'pilot\x00\xff\n')
        for name in custody.OUTPUT_NAMES:
            self.assertEqual((root/'outputs'/name).read_bytes(),
                             (self.temp/'periodic-h0-replay'/name).read_bytes())
        self.assertEqual(r['identity']['run_id'],'12345')
        self.assertEqual(r['identity']['run_attempt'],'2')
        self.assertEqual(r['identity']['source_sha'],'a'*40)
        self.assertEqual(r['scientific_effect'],'NONE')
    def test_failed_pilot_preserves_nonzero_raw_stderr_and_skips_verify(self):
        self.pilot.write_text("import sys\nsys.stderr.buffer.write(b'bad\\x00\\xff');sys.exit(23)\n")
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),23)
        x=json.loads((root/'execution.json').read_text())
        self.assertEqual(len(x['commands']),1);self.assertEqual(x['commands'][0]['returncode'],23)
        self.assertEqual(x['verify'],'SKIPPED_PILOT_FAILURE')
        r=self.seal(root,status='failure',outcome='failure')
        self.assertFalse(r['pilot_success']);self.assertEqual(r['output_completeness'],'absent')
        self.assertEqual((root/'logs/pilot.stderr.bin').read_bytes(),b'bad\x00\xff')
    def test_verify_failure_is_not_greenwashed(self):
        self.pilot.write_text(SUCCESS.replace("sys.stdout.buffer.write(b'verified\\n')","sys.exit(31)"))
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),31)
        r=self.seal(root,status='failure',outcome='failure')
        self.assertFalse(r['pilot_success']);self.assertEqual(r['output_completeness'],'complete')
    def test_both_full_pipes_are_drained_with_bounded_raw_prefixes_and_full_hashes(self):
        root=self.initialize()
        x=custody.capture_command([sys.executable,'-c',
            "import sys;sys.stdout.buffer.write(b'x'*65536);sys.stderr.buffer.write(b'y'*65536)"],
            root/'logs','large',limit=4096)
        self.assertEqual(x['returncode'],0);self.assertTrue(x['overflow'])
        for stream,char in [('stdout',b'x'),('stderr',b'y')]:
            row=x['streams'][stream]
            self.assertEqual(row['bytes'],65536);self.assertEqual(row['retained_bytes'],4096)
            self.assertEqual(row['sha256'],hashlib.sha256(char*65536).hexdigest())
            self.assertEqual((root/'logs'/('large.'+stream+'.bin')).read_bytes(),char*4096)
    def test_successful_child_with_log_overflow_is_custody_failure(self):
        self.pilot.write_text(SUCCESS.replace("sys.stdout.buffer.write(b'pilot\\x00\\xff\\n')",
                                              "sys.stdout.buffer.write(b'x'*(1048576+1))"))
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),custody.CUSTODY_FAILURE)
        r=self.seal(root,status='failure',outcome='failure')
        self.assertFalse(r['pilot_success']);self.assertIn('stream_overflow',r['holds'])
    def test_invalid_identity_creates_no_directory(self):
        for key,value in [('GITHUB_RUN_ATTEMPT','0'),('GITHUB_SHA','../escape'),
            ('GITHUB_REPOSITORY','other/repo'),('RUNNER_ENVIRONMENT','self-hosted'),
            ('GITHUB_WORKFLOW_REF','other/repo/x@refs/heads/main')]:
            with self.subTest(key=key),self.assertRaises(custody.CustodyError):
                custody.initialize(self.checkout,self.temp,dict(self.env,**{key:value}))
        self.assertEqual(list(self.temp.iterdir()),[])
    def test_existing_custody_is_preserved(self):
        root=self.initialize();(root/'sentinel').write_bytes(b'keep')
        with self.assertRaises(custody.CustodyError):self.initialize()
        self.assertEqual((root/'sentinel').read_bytes(),b'keep')
    def test_existing_output_refuses_without_deletion_or_execution(self):
        out=self.temp/'periodic-h0-replay';out.mkdir();(out/'sentinel').write_bytes(b'keep')
        (out/'observations.json').write_bytes(b'unrelated pre-existing bytes')
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),custody.CUSTODY_FAILURE)
        self.assertEqual((out/'sentinel').read_bytes(),b'keep')
        self.assertFalse((root/'logs/pilot.stdout.bin').exists())
        r=self.seal(root,status='failure',outcome='failure')
        self.assertEqual(r['output_completeness'],'not_owned')
        self.assertFalse((root/'upload/outputs/observations.json').exists())
    def test_another_run_attempt_cannot_reuse_custody(self):
        root=self.initialize()
        with self.assertRaises(custody.CustodyError):
            custody.run_pilot(root,self.checkout,self.temp,dict(self.env,GITHUB_RUN_ATTEMPT='3'))
    def test_source_drift_refuses_before_execution(self):
        root=self.initialize();self.pilot.write_text("raise RuntimeError('must not execute')\n")
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),custody.CUSTODY_FAILURE)
        self.assertFalse((root/'logs/pilot.stdout.bin').exists())
        r=self.seal(root,status='failure',outcome='failure')
        self.assertIn('source_drift',r['holds']);self.assertFalse(r['pilot_success'])
    def test_post_execution_drift_causes_custody_hold(self):
        root=self.initialize();self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        self.pilot.write_text('# changed\n');r=self.seal(root)
        self.assertIn('source_drift',r['holds']);self.assertEqual(r['custody_exit'],custody.CUSTODY_FAILURE)
    def test_interrupted_output_claim_retains_partial_bytes(self):
        root=self.initialize()
        (root/'execution-start.json').write_text(json.dumps({'new_output_absent':True}))
        out=self.temp/'periodic-h0-replay';out.mkdir()
        (out/'observations.json').write_bytes(b'{"unfinished":')
        r=self.seal(root,status='failure',outcome='skipped')
        self.assertFalse(r['pilot_success']);self.assertEqual(r['output_completeness'],'partial')
        self.assertEqual(r['execution'],'NOT_STARTED_OR_INTERRUPTED')
        self.assertEqual((root/'outputs/observations.json').read_bytes(),b'{"unfinished":')
        self.assertEqual(r['outputs']['RUN.json']['state'],'missing')
    def test_output_symlink_and_hardlink_are_never_followed(self):
        for link in ['symlink','hardlink']:
            with self.subTest(link=link):
                temp=self.base/('temp-'+link);temp.mkdir();root=self.initialize(temp)
                (root/'execution-start.json').write_text(json.dumps({'new_output_absent':True}))
                out=temp/'periodic-h0-replay';out.mkdir()
                secret=self.base/('secret-'+link);secret.write_bytes(b'do not publish')
                p=out/'observations.json'
                if link=='symlink':p.symlink_to(secret)
                else:os.link(secret,p)
                r=self.seal(root,temp,status='failure',outcome='failure')
                self.assertEqual(r['outputs']['observations.json']['state'],'unsafe')
                self.assertFalse((root/'outputs/observations.json').exists())
                self.assertEqual(secret.read_bytes(),b'do not publish')
    def test_oversize_retains_only_prefix_without_full_digest_claim(self):
        root=self.initialize()
        (root/'execution-start.json').write_text(json.dumps({'new_output_absent':True}))
        out=self.temp/'periodic-h0-replay';out.mkdir()
        with (out/'observations.json').open('wb') as f:f.truncate(custody.OUTPUT_LIMIT+1)
        r=self.seal(root,status='failure',outcome='failure');row=r['outputs']['observations.json']
        self.assertEqual(row['state'],'oversized');self.assertIsNone(row['sha256'])
        self.assertEqual(row['retained_bytes'],custody.OUTPUT_LIMIT)
        self.assertEqual((root/'outputs/observations.json').stat().st_size,custody.OUTPUT_LIMIT)
        self.assertEqual(r['custody_exit'],custody.CUSTODY_FAILURE)
    def test_unexpected_nested_output_is_metadata_only(self):
        root=self.initialize()
        (root/'execution-start.json').write_text(json.dumps({'new_output_absent':True}))
        out=self.temp/'periodic-h0-replay';out.mkdir()
        (out/'unexpected').mkdir();(out/'unexpected/private').write_bytes(b'do not descend')
        r=self.seal(root,status='failure',outcome='failure')
        self.assertIn('unexpected_output_members',r['holds'])
        self.assertEqual(r['unexpected_members'],['unexpected'])
        self.assertFalse((root/'outputs/unexpected').exists())
    def test_second_execution_or_seal_cannot_rewrite_history(self):
        root=self.initialize();self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        with self.assertRaises(custody.CustodyError):custody.run_pilot(root,self.checkout,self.temp,self.env)
        self.seal(root);raw=(root/'receipt.json').read_bytes()
        with self.assertRaises(custody.CustodyError):self.seal(root)
        self.assertEqual((root/'receipt.json').read_bytes(),raw)
    def test_spawn_failure_is_retained(self):
        root=self.initialize()
        x=custody.capture_command([str(self.base/'nonexistent')],root/'logs','spawn')
        self.assertEqual(x['returncode'],127);self.assertIsNotNone(x['spawn_error'])
        self.assertEqual(x['streams']['stdout']['bytes'],0)
    def test_changed_or_linked_raw_stream_is_excluded_from_upload(self):
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        stream=root/'logs/pilot.stdout.bin'
        stream.unlink()
        secret=self.base/'unrelated-secret';secret.write_bytes(b'never publish')
        stream.symlink_to(secret)
        r=self.seal(root)
        self.assertIn('unsafe_stream:pilot.stdout.bin',r['holds'])
        self.assertFalse((root/'upload/logs/pilot.stdout.bin').exists())
        self.assertEqual(secret.read_bytes(),b'never publish')
    def test_upload_stage_has_only_explicit_guarded_members_and_manifest(self):
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        (root/'unrelated-private').write_bytes(b'never publish')
        r=self.seal(root);self.assertTrue(r['pilot_success'])
        self.assertFalse((root/'upload/unrelated-private').exists())
        manifest=json.loads((root/'upload/artifact-manifest.json').read_text())
        actual=sorted(str(p.relative_to(root/'upload')) for p in (root/'upload').rglob('*') if p.is_file())
        self.assertEqual(actual,sorted([x['path'] for x in manifest['members']]+['artifact-manifest.json']))
        for row in manifest['members']:
            raw=(root/'upload'/row['path']).read_bytes()
            self.assertEqual(len(raw),row['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
    def test_source_symlink_is_refused_with_initialization_error_and_no_read_retry(self):
        source=self.checkout/'experiments/periodic_h0/experiment.py'
        source.unlink()
        unrelated=self.base/'unrelated-source';unrelated.write_bytes(b'never read as source')
        source.symlink_to(unrelated)
        with self.assertRaises(custody.CustodyError):self.initialize()
        root=self.temp/'periodic-h0-custody-12345-2'
        self.assertTrue((root/'initialization-error.json').is_file())
        r=self.seal(root,status='failure',outcome='skipped')
        self.assertIn('source_initialization_incomplete',r['holds'])
        self.assertIsNone(r['source_after'])
        self.assertFalse((root/'upload/sources-before.json').exists())
        self.assertEqual(unrelated.read_bytes(),b'never read as source')

    def test_zero_child_exits_cannot_hide_missing_pilot_output(self):
        self.pilot.write_text("pass\n")
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        r=self.seal(root)
        self.assertFalse(r['pilot_success'])
        self.assertEqual(r['output_completeness'],'absent')
        self.assertIn('successful_execution_missing_complete_outputs',r['holds'])
        self.assertEqual(r['custody_exit'],custody.CUSTODY_FAILURE)
    def test_regular_raw_stream_tampering_is_excluded(self):
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        (root/'logs/pilot.stdout.bin').write_bytes(b'changed')
        r=self.seal(root)
        self.assertIn('changed_stream:pilot.stdout.bin',r['holds'])
        self.assertFalse((root/'upload/logs/pilot.stdout.bin').exists())
    def test_signal_exit_preserves_shell_failure_semantics(self):
        self.pilot.write_text("import os,signal\nos.kill(os.getpid(),signal.SIGTERM)\n")
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),143)
        x=json.loads((root/'execution.json').read_text())
        self.assertEqual(x['commands'][0]['returncode'],-15)
        self.assertEqual(x['verify'],'SKIPPED_PILOT_FAILURE')

    def test_workflow_failure_and_always_retention_contract(self):
        text=(Path(__file__).resolve().parents[1]/'.github/workflows/periodic-h0-pilot.yml').read_text()
        for literal in ['timeout-minutes: 5','cancel-in-progress: true',
            'persist-credentials: false','if: ${{ always() }}',
            'if-no-files-found: error','retention-days: 14',
            'periodic-h0-custody-${{ github.run_id }}-${{ github.run_attempt }}-${{ github.sha }}']:
            self.assertIn(literal,text)
        for literal in ['continue-on-error:','|| true','rm -']:self.assertNotIn(literal,text)


class S1MutantGapTests(unittest.TestCase):
    """S1 (#475): pin agent 9 N1 (a)-(c). Kept apart from S2's OSError tests."""
    setUp=PilotCustodyTests.setUp
    initialize=PilotCustodyTests.initialize
    seal=PilotCustodyTests.seal
    def _succeed(self):
        root=self.initialize()
        self.assertEqual(custody.run_pilot(root,self.checkout,self.temp,self.env),0)
        return root
    def test_s1a_clean_seal_with_non_success_status_is_not_pilot_success(self):
        for status,outcome in [('failure','success'),('cancelled','success'),
                               ('success','failure'),('success','skipped')]:
            with self.subTest(status=status,outcome=outcome):
                temp=self.base/('s1a-'+status+'-'+outcome);temp.mkdir()
                root=self.initialize(temp)
                self.assertEqual(custody.run_pilot(root,self.checkout,temp,self.env),0)
                r=self.seal(root,temp,status=status,outcome=outcome)
                self.assertEqual(r['holds'],[]);self.assertEqual(r['custody_exit'],0)
                self.assertEqual(r['output_completeness'],'complete')
                self.assertFalse(r['pilot_success'])
    def test_s1b_same_size_stream_tampering_is_held(self):
        root=self._succeed()
        path=root/'logs/pilot.stdout.bin';raw=path.read_bytes()
        path.write_bytes(bytes(b^0x01 for b in raw[:1])+raw[1:])
        self.assertEqual(len(path.read_bytes()),len(raw))
        r=self.seal(root)
        self.assertIn('changed_stream:pilot.stdout.bin',r['holds'])
        self.assertEqual(r['custody_exit'],custody.CUSTODY_FAILURE)
        self.assertFalse((root/'upload/logs/pilot.stdout.bin').exists())
    def test_s1b_same_size_tampering_after_inventory_refuses_stage(self):
        root=self._succeed()
        execution=json.loads((root/'execution.json').read_text())
        streams,problems=custody._stream_inventory(root,execution)
        self.assertEqual(problems,[])
        path=root/'logs/pilot.stdout.bin';raw=path.read_bytes()
        path.write_bytes(bytes(b^0x01 for b in raw[:1])+raw[1:])
        receipt={'identity':{},'raw_streams':streams,'outputs':{}}
        with self.assertRaises(custody.CustodyError):
            custody._prepare_artifact(root,receipt)
    def test_s1c_newline_or_cr_in_workflow_ref_is_refused(self):
        ref=self.env['GITHUB_WORKFLOW_REF']
        for bad in [ref+'\n',ref+'\r',ref+'\nx',ref[:-4]+'\r\nmain']:
            with self.subTest(ref=repr(bad)),self.assertRaises(custody.CustodyError):
                custody.context_from_env(dict(self.env,GITHUB_WORKFLOW_REF=bad))
        self.assertEqual(list(self.temp.iterdir()),[])

if __name__=='__main__':unittest.main()
