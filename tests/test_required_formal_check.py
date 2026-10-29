"""Negative controls for current-run required formal checks; stdlib only."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/required_formal_check.py'
SPEC = importlib.util.spec_from_file_location('required_check', SCRIPT)
M = importlib.util.module_from_spec(SPEC)
if SCRIPT.exists():
    SPEC.loader.exec_module(M)

class RequiredCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {'GITHUB_SHA': 'a'*40, 'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main',
                    'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2'}
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text('{"source":"fixture"}\n')
        self.log = self.root/'build.log'; self.log.write_text('fixture build log\n')
        self.receipt = self.root/'receipt.json'
        self.record = {'checked_commit': self.env['GITHUB_SHA'],
            'repository': self.env['GITHUB_REPOSITORY'], 'workflow_run_id': '123',
            'formalization_status': 'kernel-checked', 'scientific_effect':'NONE',
            'manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            'logs':{'build.log':hashlib.sha256(self.log.read_bytes()).hexdigest()}}
        self.write_receipt()
        self.needs = {'checks': {'result':'success','outputs':{'checked_commit':'a'*40}},
                      'formal': {'result':'success','outputs':{
                          'checked_commit':'a'*40,'repository':self.env['GITHUB_REPOSITORY'],
                          'run_id':'123','run_attempt':'2','receipt_sha256':'b'*64}}}

    def write_receipt(self): self.receipt.write_text(json.dumps(self.record))
    def binding(self): return M.bind_receipt(self.receipt,self.manifest,self.env,'a'*40)
    def decision(self): return M.aggregate(self.needs,self.env)

    def test_implementation_exists(self):
        self.assertTrue(hasattr(M,'aggregate'), 'required-check bridge is missing')
    def test_success(self):
        result=self.decision()
        self.assertEqual(result['checked_commit'],'a'*40)
        self.assertEqual(result['scientific_effect'],'NONE')
        self.assertEqual(result['conclusion'],'success')
    def test_every_unsuccessful_result_refused(self):
        for job in ('formal','checks'):
            for result in ('failure','cancelled','skipped','neutral','timed_out','pending','',None,True):
                with self.subTest(job=job,result=result):
                    prior=self.needs[job]['result'];self.needs[job]['result']=result
                    with self.assertRaises(ValueError):self.decision()
                    self.needs[job]['result']=prior
    def test_missing_dependency(self):
        del self.needs['formal']
        with self.assertRaises(ValueError):self.decision()
    def test_extra_dependency(self):
        self.needs['other']={'result':'success'}
        with self.assertRaises(ValueError):self.decision()
    def test_each_output_required(self):
        for field in tuple(self.needs['formal']['outputs']):
            old=self.needs['formal']['outputs'].pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):self.decision()
            self.needs['formal']['outputs'][field]=old
    def test_formal_commit_drift(self):
        self.needs['formal']['outputs']['checked_commit']='c'*40
        with self.assertRaises(ValueError):self.decision()
    def test_existing_checks_commit_drift(self):
        self.needs['checks']['outputs']['checked_commit']='c'*40
        with self.assertRaises(ValueError):self.decision()
    def test_changed_test_merge_commit(self):
        self.env['GITHUB_SHA']='c'*40
        with self.assertRaises(ValueError):self.decision()
    def test_repository_substitution(self):
        self.needs['formal']['outputs']['repository']='d6g8k5htny-coder/Math-'
        with self.assertRaises(ValueError):self.decision()
    def test_stale_run(self):
        self.needs['formal']['outputs']['run_id']='122'
        with self.assertRaises(ValueError):self.decision()
    def test_stale_attempt(self):
        self.needs['formal']['outputs']['run_attempt']='1'
        with self.assertRaises(ValueError):self.decision()
    def test_no_numeric_identity_coercion(self):
        self.env['GITHUB_SHA']=int('1'*40)
        with self.assertRaises(ValueError):self.decision()
    def test_bad_digest(self):
        for value in ('main','B'*64,1,'b'*63):
            self.needs['formal']['outputs']['receipt_sha256']=value
            with self.subTest(value=value),self.assertRaises(ValueError):self.decision()
    def test_binding_preserves_receipt(self):
        before=self.receipt.read_bytes(); result=self.binding()
        self.assertEqual(self.receipt.read_bytes(),before)
        self.assertEqual(result['receipt_sha256'],hashlib.sha256(before).hexdigest())
        self.assertEqual(result['run_attempt'],'2')
    def test_unproved_receipt_refused(self):
        self.record['formalization_status']='specified';self.write_receipt()
        with self.assertRaises(ValueError):self.binding()
    def test_receipt_drift(self):
        self.record['checked_commit']='c'*40;self.write_receipt()
        with self.assertRaises(ValueError):self.binding()
    def test_manifest_drift(self):
        self.manifest.write_text('changed')
        with self.assertRaises(ValueError):self.binding()
    def test_log_tamper(self):
        self.log.write_text('changed')
        with self.assertRaises(ValueError):self.binding()
    def test_empty_logs_refused(self):
        self.record['logs']={};self.write_receipt()
        with self.assertRaises(ValueError):self.binding()
    def test_symlink_log_refused(self):
        self.log.unlink();other=self.root/'other';other.write_text('fixture build log\n');self.log.symlink_to(other)
        with self.assertRaises(ValueError):self.binding()
    def test_traversal_log_refused(self):
        self.record['logs']={'../build.log':'b'*64};self.write_receipt()
        with self.assertRaises(ValueError):self.binding()
    def test_duplicate_json_keys_refused(self):
        with self.assertRaises(ValueError):M.strict_json('{"formal":{},"formal":{}}')
    def test_actual_cli_failure(self):
        env={**os.environ,**self.env,'REQUIRED_FORMAL_NEEDS':json.dumps(self.needs)}
        good=subprocess.run([sys.executable,'-B','-S',str(SCRIPT),'aggregate'],env=env,capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        self.needs['formal']['result']='failure';env['REQUIRED_FORMAL_NEEDS']=json.dumps(self.needs)
        bad=subprocess.run([sys.executable,'-B','-S',str(SCRIPT),'aggregate'],env=env,capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)
        self.assertNotIn('"conclusion": "success"',bad.stdout)

class WiringTests(unittest.TestCase):
    def setUp(self):
        self.is_main=(ROOT/'.github/workflows/workspace-landing.yml').exists()
        self.parent_name='workspace-landing.yml' if self.is_main else 'downstream-gate.yml'
        self.child_name='formal-verification.yml' if self.is_main else 'formal-lean.yml'
        self.parent=(ROOT/'.github/workflows'/self.parent_name).read_text()
        self.child=(ROOT/'.github/workflows'/self.child_name).read_text()
    def test_parent_required_name_is_aggregate(self):
        name='verify' if self.is_main else 'math-downstream-gates'
        self.assertIn('  required:\n    name: '+name+'\n    if: ${{ always() }}\n    needs: [checks, formal]\n',self.parent)
    def test_same_revision_reusable_workflow(self):
        self.assertIn('    uses: ./.github/workflows/'+self.child_name,self.parent)
        self.assertIn('  workflow_call:\n',self.child)
        self.assertNotIn('pull_request:',self.child)
        self.assertNotIn('push:',self.child)
        self.assertNotIn('head.sha',self.child)
    def test_parent_no_path_filters_or_skip_override(self):
        self.assertIn('  pull_request:',self.parent)
        self.assertNotIn('paths:',self.parent)
        self.assertNotIn('continue-on-error:',self.parent+self.child)
        self.assertNotIn('pull_request_target:',self.parent+self.child)
    def test_required_calls_validator(self):
        self.assertIn('python3 -B -S tools/required_formal_check.py aggregate',self.parent)
        self.assertIn('REQUIRED_FORMAL_NEEDS: ${{ toJSON(needs) }}',self.parent)
    def test_outputs_are_bound_to_executed_receipt(self):
        self.assertIn('python3 -B -S tools/required_formal_check.py receipt',self.child)
        for key in ('checked_commit','repository','run_id','run_attempt','receipt_sha256'):
            self.assertIn(key+':',self.child)
    def test_synthetic_control_uses_no_proof_edits(self):
        self.assertIn('python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v',self.parent)
        self.assertIn('python3 -B -O -S -m unittest discover -s tests -p test_required_formal_check.py -v',self.parent)

if __name__=='__main__':unittest.main()
