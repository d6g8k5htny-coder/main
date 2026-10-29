"""Bounded complete engineering discovery; original public package caps stay fixed."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('intake_discovery',ROOT/'tools/public_intake_check.py')
intake=importlib.util.module_from_spec(spec);spec.loader.exec_module(intake)
REPO='d6g8k5htny-coder/main'

class API:
    def __init__(self,pr,rows):
        self.pr=pr;self.rows=rows;self.calls=[];self.pages={};self.reads=0;self.final_change=None
    def get(self,route):
        self.calls.append(route)
        if route==f'/repos/{REPO}/pulls/7':
            self.reads+=1;result=copy.deepcopy(self.pr)
            if self.reads==2 and self.final_change: self.final_change(result)
            return result
        if '/files?per_page=100&page=' in route:
            page=int(route.rsplit('=',1)[1]);return copy.deepcopy(self.pages.get(page,self.rows[(page-1)*100:page*100]))
        raise AssertionError('payload/source API must not be accessed by discovery')

class DiscoveryTests(unittest.TestCase):
    def setup_rows(self,count=1458):
        self.pr={'number':7,'state':'open','changed_files':count,'head':{'sha':'a'*40,'repo':{'full_name':REPO}},
                 'base':{'sha':'b'*40,'ref':'main','repo':{'full_name':REPO}},'labels':[]}
        self.event={'number':7,'repository':{'full_name':REPO},'pull_request':copy.deepcopy(self.pr)}
        rows=[{'filename':f'cbz/evidence/path-{i:04d}.txt','status':'added','sha':'c'*40} for i in range(count)]
        self.api=API(self.pr,rows)
    def check(self): return intake.check(self.event,self.api,REPO)
    def engineering(self):
        try: result=self.check()
        except ValueError as error: self.fail('ordinary engineering discovery rejected: '+str(error))
        self.assertEqual(result['lane'],'maintainer-engineering');self.assertEqual(result['scientific_effect'],'NONE')
        return result
    def file_calls(self): return [p for p in self.api.calls if '/files?' in p]

    def test_actual1458_path_engineering_shape_is_fully_discovered(self):
        self.setup_rows();self.engineering();self.assertEqual(len(self.file_calls()),15);self.assertTrue(self.file_calls()[-1].endswith('page=15'));self.assertEqual(self.api.reads,2)
    def test_api_ceiling3000_paths_requires_all30pages(self):
        self.setup_rows(3000);self.engineering();self.assertEqual(len(self.file_calls()),30)
    def test_above_api_ceiling_is_refused_without_file_reads(self):
        self.setup_rows(3001)
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(self.file_calls(),[])
    def test_boolean_changed_count_is_not_an_integer_count(self):
        self.setup_rows();self.pr['changed_files']=True
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(self.file_calls(),[])
    def test_fork501_keeps_early_public_intake_cap(self):
        self.setup_rows(501);self.pr['head']['repo']['full_name']='visitor/fork'
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(self.file_calls(),[])
    def test_labeled501_keeps_early_public_intake_cap(self):
        self.setup_rows(501);self.pr['labels']=[{'name':'results-for-review'}]
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(self.file_calls(),[])
    def test_late_incoming_routes_to_original_intake_cap_after_full_discovery(self):
        self.setup_rows();self.api.rows[-1]['filename']='incoming/hidden/RESULT.md'
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(len(self.file_calls()),15)
    def test_late_rename_from_incoming_cannot_hide_in_engineering(self):
        self.setup_rows();self.api.rows[-1].update(status='renamed',previous_filename='incoming/hidden/RESULT.md')
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(len(self.file_calls()),15)
    def test_501st_incoming_row_does_not_bypass_original500_cap(self):
        self.setup_rows(501);self.api.rows[-1]['filename']='incoming/hidden/RESULT.md'
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): self.check()
        self.assertEqual(len(self.file_calls()),6)
    def test_original_snapshot_api_keeps500_default(self):
        self.setup_rows(501)
        with self.assertRaisesRegex(ValueError,'unsupported changed file count'): intake.snapshot(self.pr,REPO)
    def test_all_public_package_and_byte_caps_are_unchanged(self):
        self.assertEqual((intake.MAX_FILES,intake.MAX_PACKAGE_FILES,intake.MAX_FILE_BYTES,intake.MAX_PACKAGE_BYTES,intake.MAX_RESPONSE_BYTES),(500,50,262144,2097152,8388608))
    def test_truncated_middle_page_is_refused(self):
        self.setup_rows();self.api.pages[8]=self.api.rows[700:799]
        with self.assertRaisesRegex(ValueError,'incomplete file listing'): self.check()
    def test_overfull_page_cannot_compensate_for_short_page(self):
        self.setup_rows(1458);self.api.pages[1]=self.api.rows[:99];self.api.pages[2]=self.api.rows[99:200]
        with self.assertRaisesRegex(ValueError,'incomplete file listing'): self.check()
    def test_duplicate_path_on_last_page_is_refused(self):
        self.setup_rows();self.api.rows[-1]['filename']=self.api.rows[0]['filename']
        with self.assertRaisesRegex(ValueError,'duplicate changed path'): self.check()
    def test_changed_head_is_refused_after_all_pages(self):
        self.setup_rows();self.api.final_change=lambda pr:pr['head'].update(sha='d'*40)
        with self.assertRaisesRegex(ValueError,'changed during verification'): self.check()
        self.assertEqual(len(self.file_calls()),15)
    def test_changed_base_is_refused_after_all_pages(self):
        self.setup_rows();self.api.final_change=lambda pr:pr['base'].update(sha='d'*40)
        with self.assertRaisesRegex(ValueError,'changed during verification'): self.check()
    def test_changed_label_is_refused_after_all_pages(self):
        self.setup_rows();self.api.final_change=lambda pr:pr['labels'].append({'name':'results-for-review'})
        with self.assertRaisesRegex(ValueError,'changed during verification'): self.check()
    def test_changed_count_is_refused_after_all_pages(self):
        self.setup_rows();self.api.final_change=lambda pr:pr.update(changed_files=1459)
        with self.assertRaisesRegex(ValueError,'changed during verification'): self.check()
    def test_event_head_mismatch_refused_before_path_reads(self):
        self.setup_rows();self.event['pull_request']['head']['sha']='d'*40
        with self.assertRaisesRegex(ValueError,'changed since event'): self.check()
        self.assertEqual(self.file_calls(),[])

if __name__=='__main__': unittest.main()
