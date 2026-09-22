"""Author-side regressions; they do not provide nonauthor review credit."""
from fractions import Fraction as F
from pathlib import Path
import copy
import csv
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

HERE=Path(__file__).resolve().parents[1]/'engine/solver_pilots/h3_20260921'
sys.path.insert(0,str(HERE))
import h3_scalar_arithmetic as a
import h3_scalar_certificate as c
import h3_solver as r


class ExactArithmetic(unittest.TestCase):
    def test_float_rejected(self):
        with self.assertRaises(TypeError):a.I(.1)
    def test_bool_rejected(self):
        with self.assertRaises(TypeError):a.I(True)
    def test_empty_interval(self):
        with self.assertRaises(ValueError):a.I(2,1)
    def test_division_by_zero_interval(self):
        with self.assertRaises(ZeroDivisionError):a.I(1)/a.I(-1,1)
    def test_negative_interval_division(self):
        x=a.I(1,2)/a.I(-4,-2)
        self.assertLessEqual(x.lo,-1);self.assertGreaterEqual(x.hi,F(-1,4))
    def test_square_crosses_zero(self):
        x=a.I(-2,3)**2;self.assertEqual(x.lo,0);self.assertEqual(x.hi,9)
    def test_odd_power_signed(self):
        x=a.I(-2,3)**3;self.assertEqual((x.lo,x.hi),(-8,27))
    def test_sqrt_exact(self):
        x=a.sqrt(a.I(4));self.assertEqual((x.lo,x.hi),(2,2))
    def test_sqrt_irrational_bounds(self):
        x=a.sqrt(a.I(2));self.assertLessEqual(x.lo*x.lo,2);self.assertGreaterEqual(x.hi*x.hi,2)
    def test_sqrt_negative_refusal(self):
        with self.assertRaises(ValueError):a.sqrt(a.I(-1,2))
    def test_signed_rounding(self):
        x=a.I.bound(F(-1,3),F(1,3));self.assertLessEqual(x.lo,F(-1,3));self.assertGreaterEqual(x.hi,F(1,3))
    def test_pi_coarse_rational_bounds(self):
        p=a.pi();self.assertGreater(p.lo,F(3141592653589793,10**15));self.assertLess(p.hi,F(3141592653589794,10**15))
    def test_exp_zero(self):
        e=a.exp_negative_point(0);self.assertEqual((e.lo,e.hi),(1,1))
    def test_exp_one_known_rational_bracket(self):
        e=a.exp_negative_point(1);self.assertGreater(e.lo,F('0.3678794411714423'));self.assertLess(e.hi,F('0.3678794411714424'))
    def test_exp_out_of_domain(self):
        with self.assertRaises(ValueError):a.exp_negative_point(51)
    def test_cdf_zero(self):
        v=a.cdf_point(0);self.assertEqual((v.lo,v.hi),(F(1,2),F(1,2)))
    def test_cdf_symmetry(self):
        x=a.cdf_point(F(3,2));y=a.cdf_point(F(-3,2))
        self.assertLessEqual((x+y).lo,1);self.assertGreaterEqual((x+y).hi,1)
    def test_cdf_bracket(self):
        x=a.cdf_point(1);self.assertGreater(x.lo,F('0.84134474606854'));self.assertLess(x.hi,F('0.84134474606855'))
    def test_cdf_short_series_rejected(self):
        with self.assertRaises(ValueError):a.cdf_point(2,terms=5)
    def test_cdf_unbounded_domain_rejected(self):
        with self.assertRaises(ValueError):a.cdf_point(9)
    def test_display_is_outward(self):
        d=a.outward_decimal(a.I(F(-1,3),F(1,3)),6)
        self.assertLessEqual(F(d['lo']),F(-1,3));self.assertGreaterEqual(F(d['hi']),F(1,3))
    def test_hermite_known(self):self.assertEqual(c.hermite(4),[3,0,-6,0,1])
    def test_moment_tails_retained(self):
        m,report=c.moments();self.assertGreater(F(report['raw_derivative_errors']['8']),0)
        self.assertLess(m[8].lo,105);self.assertGreater(m[8].hi,105)
    def test_gram_energy_budget(self):
        m,_=c.moments();Q,report=c.gram_and_energy(m,F(1,20))
        self.assertEqual(Q,F(1266466,160083));self.assertLess(Q,8)
        self.assertTrue(all(F(x['lo'])>0 for x in report['shifted_gram_pivots']))


class ScientificDirection(unittest.TestCase):
    def test_wide_crossing_not_counterexample(self):
        self.assertEqual(r.classify_enclosure('1','4','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'),'INCONCLUSIVE_REFINE_ENCLOSURE')
    def test_upper_below_floor_is_conflict(self):
        self.assertIn('CONTRADICTION',r.classify_enclosure('1','1.6','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_lower_above_ceiling_is_conflict(self):
        self.assertIn('CONTRADICTION',r.classify_enclosure('3.5','4','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_lower_formula_cannot_refute_true_quantity(self):
        self.assertEqual(r.classify_enclosure('0','1','1.7','3.49','SUFFICIENT_LOWER_BOUND'),'WRONG_QUANTITY_FOR_REFUTATION')
    def test_equal_endpoint_not_strict_violation(self):
        self.assertEqual(r.classify_enclosure('1','1.7','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'),'INCONCLUSIVE_REFINE_ENCLOSURE')
    def test_compatible_point_not_universal_proof(self):
        self.assertIn('DOMAIN_ONLY',r.classify_enclosure('2','3','1.7','3.49','TARGET_QUANTITY_ENCLOSURE'))
    def test_no_float_parameter_in_proof(self):
        with self.assertRaises(ValueError):c.prove(epsilon=.206)
    def test_two_axial_tails_required(self):
        with self.assertRaises(ValueError):c.prove(axial_tails=F(1))
    def test_two_difference_tails_required(self):
        with self.assertRaises(ValueError):c.prove(difference_tails=F(1))
    def test_no_understated_beta(self):
        with self.assertRaises(ValueError):c.prove(beta_scale=F(1,8))
    def test_no_understated_variance_loss(self):
        with self.assertRaises(ValueError):c.prove(variance_loss=F(1,8))
    def test_radius_scope_refusal(self):
        with self.assertRaises(ValueError):c.prove(radius=F(1,10))
    def test_zero_radius_not_endpoint_claim(self):
        with self.assertRaises(ValueError):c.prove(radius=F(0))
    def test_invalid_parameter_refusal(self):
        with self.assertRaises(ValueError):c.prove(epsilon=F(1))
    def test_search_is_diagnostic(self):
        x=r.search();self.assertEqual(x['evaluated'],30351)
        self.assertEqual(x['status'],'NONCERTIFYING_PARAMETER_SEARCH')
        self.assertEqual((x['epsilon'],x['gap_half']),('103/500','9/100'))
    def test_malformed_interval_refused(self):
        with self.assertRaises(ValueError):r.classify_enclosure('2','1','0','3','TARGET_QUANTITY_ENCLOSURE')


class Provenance(unittest.TestCase):
    def sample_context(self):return json.loads((HERE/'CONTEXT.json').read_text())
    def test_context_denial_before_cache(self):
        x=self.sample_context();x['source_usable_for_candidate']=False
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})
    def test_context_new_hash_new_key(self):
        x=self.sample_context();y=dict(x,basis='same candidate; additional observation')
        self.assertNotEqual(c.digest(r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})),c.digest(r.recipe({'archive_sha256':r.ARCHIVE_SHA},y,{})))
    def test_context_source_mismatch(self):
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':'0'*64},self.sample_context(),{})
    def test_independence_cannot_be_imported(self):
        x=self.sample_context();x['organizational_independence_credit']=1
        with self.assertRaises(ValueError):r.recipe({'archive_sha256':r.ARCHIVE_SHA},x,{})
    def test_local_cache_hit_no_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            x,tag,_=r.cached({'input':1},d,lambda:{'answer':2})
            y,tag2,_=r.cached({'input':1},d,lambda:self.fail('must hit'))
            self.assertEqual(x,y);self.assertIn('NOT_NEW_EVIDENCE',tag2)
    def test_cache_tampering_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,key=r.cached({'input':1},d,lambda:{'answer':2})
            p=Path(d)/(key+'.json');v=json.loads(p.read_text());v['payload']['answer']=3;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):r.cached({'input':1},d,lambda:{'answer':2})
    def test_cold_cache_disagreement_detected(self):
        with tempfile.TemporaryDirectory() as d:
            r.cached({'input':1},d,lambda:{'answer':2})
            with self.assertRaises(ValueError):r.cached({'input':1},d,lambda:{'answer':3},cold=True)
    def test_changed_inputs_recompute(self):
        with tempfile.TemporaryDirectory() as d:
            r.cached({'input':1},d,lambda:2)
            y,tag,_=r.cached({'input':2},d,lambda:3)
            self.assertEqual((y,tag),(3,'MISS_COMPUTED'))
    def test_dependency_propagation(self):
        self.assertEqual(r.affected_nodes(['midpoint']),['coefficient','determinant','midpoint'])
    def test_unknown_lineage_node_rejected(self):
        with self.assertRaises(ValueError):r.affected_nodes(['fake'])
    def test_archive_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip';p.write_bytes(b'not the source')
            with self.assertRaises(ValueError):r.source_check(p,self.sample_context())


class ExactCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=c.prove()
    def test_new_floor(self):
        v=self.result['values']['derived_lower_coefficient']
        self.assertGreater(F(v['lo']),F(1747,1000))
        self.assertLess(F(v['hi']),F(7,4))
    def test_scope_is_preserved(self):
        self.assertEqual(self.result['conclusion']['endpoint_at_rmax'],'1747/400000')
        self.assertFalse(self.result['scientific_status_changed'])
        self.assertEqual(self.result['organizational_independence_credit'],0)
    def test_positive_budget_before_product(self):
        for key in ('positive_bracket','p_axial','p_difference'):
            self.assertGreater(F(self.result['values'][key]['lo']),0)
    def test_formula_root_changes_with_upstream_data(self):
        x=copy.deepcopy(self.result);x['parameters']['epsilon']='1/5'
        a=r.formula_lineage(self.result,{'archive_sha256':r.ARCHIVE_SHA});b=r.formula_lineage(x,{'archive_sha256':r.ARCHIVE_SHA})
        self.assertNotEqual(a['root'],b['root']);self.assertEqual(a['spreadsheet_cells_used'],[])
    def test_unattainable_target_rejected(self):
        with self.assertRaises(ValueError):c.prove(target=F(7,4))
    def test_original_parameters_still_clear_original_target(self):
        x=c.prove(epsilon=F(1,5),gap_half=F(3,25),target=F(17,10))
        self.assertGreater(F(x['values']['derived_lower_coefficient']['lo']),F(17,10))


class SourceAdmission(unittest.TestCase):
    """Synthetic metadata controls, never a second admission authority."""
    @classmethod
    def setUpClass(cls):
        cls.certificate = c.prove()
        cls.original_root = r.admission.ROOT
        cls.original_record = r.admission.ADMISSION
        cls.record_bytes = cls.original_record.read_bytes()
        cls.record = json.loads(cls.record_bytes)
        cls.archive_bytes = (cls.original_root/r.admission.ARCHIVE).read_bytes()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.ad = r.admission
        self.context = json.loads((HERE/'CONTEXT.json').read_text())
        self.record_path = self.root/'admission.json'
        self.record_path.write_bytes(self.record_bytes)
        (self.root/'sources').mkdir()
        self.custody_path=self.root/'sources/H3_CURRENT_SOURCE_CUSTODY_V1.json'
        self.custody_path.write_bytes((HERE/'sources/H3_CURRENT_SOURCE_CUSTODY_V1.json').read_bytes())
        self.exclusions_path = self.root/'quarantine/EXCLUSIONS.json'
        self.exclusions_path.parent.mkdir()
        self.write_exclusions([])
        (self.root/'drive/source_map').mkdir(parents=True)
        carrier = self.record['upstream']['carrier']
        link = 'https://drive.google.com/file/d/'+carrier['drive_id']+'/view'
        self.inventory = [dict(id=carrier['drive_id'], bytes=carrier['bytes'], sha256=carrier['sha256'],
                               context='RESEARCH_SOURCE_CHECK_STATUS', access_status='ARCHIVE_INDEXED',
                               path='01_ACTIVE_RESEARCH_PACKAGES/current/source.zip', link=link)]
        self.payloads = [{'Source name':name, 'SHA-256':value['sha256'], 'Bytes':str(value['bytes']),
                          'Conversion':'TEXT_READING_COPY', 'Context':'RESEARCH_SOURCE_CHECK_STATUS',
                          'Scope holds':'[]', 'Original source link':link}
                         for name,value in self.record['upstream']['members'].items()]
        self.write_inventory(); self.write_payloads()
        self.archive = self.root/'source.zip'; self.archive.write_bytes(self.archive_bytes)
        self.context_path = self.root/'context.json'; self.context_path.write_text(json.dumps(self.context))
        self.enterContext(mock.patch.object(self.ad,'ROOT',self.root))
        self.enterContext(mock.patch.object(self.ad,'HERE',self.root))
        self.enterContext(mock.patch.object(self.ad,'ADMISSION',self.record_path))
        self.ad.metadata_gate(self.context)  # Each negative starts from an admitted fixture.

    def write_exclusions(self,rows):
        self.exclusions_path.write_text(json.dumps({'exclusions':rows}))

    def write_inventory(self):
        (self.root/'drive/inventory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.inventory))

    def write_payloads(self):
        with (self.root/'drive/source_map/Payloads.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.payloads[0]));writer.writeheader();writer.writerows(self.payloads)

    def denial(self,context=None):
        # Fails if a denial opens even ZIP metadata or reads the opaque archive.
        original=self.ad.read_bounded
        def read(path,limit):
            self.assertNotEqual(Path(path),self.archive,'denied archive bytes were read')
            return original(path,limit)
        with mock.patch.object(self.ad,'read_bounded',side_effect=read), mock.patch.object(zipfile.ZipFile,'open') as opened:
            with self.assertRaises((ValueError,OSError)):
                self.ad.source_check(self.archive,self.context if context is None else context)
            opened.assert_not_called()

    def exclusion(self,**changes):
        row={'kind':'archive_member','carrier_id':self.ad.ANCESTOR_ID,'member_path':'h3_floor/PROOF.md'}
        row.update(changes);self.write_exclusions([row])

    def main(self,out,**kwargs):
        argv=['h3_solver.py','--source-archive',str(self.archive),'--context',str(self.context_path),'--out',str(out),
              '--cache',str(self.root/'cache')]
        with mock.patch.object(sys,'argv',argv), mock.patch.object(r,'prove',**kwargs) as compute, mock.patch('builtins.print'):
            r.main()
        return compute

    def test_only_manifest_and_allowlisted_body_opened(self):
        seen=[];original=zipfile.ZipFile.open
        def opened(z,name,*args,**kwargs):
            seen.append(name.filename if isinstance(name,zipfile.ZipInfo) else name)
            return original(z,name,*args,**kwargs)
        with mock.patch.object(zipfile.ZipFile,'open',opened):
            result=self.ad.source_check(self.archive,self.context)
        self.assertEqual(seen,['MANIFEST.json','h3_floor/PROOF.md'])
        self.assertEqual(result['analytic_bodies_read'],self.ad.BODY_ALLOWLIST)
        self.assertNotIn('verified_manifest_leaves',result)

    def test_caller_denial_precedes_every_metadata_read(self):
        self.context['source_usable_for_candidate']=False
        with mock.patch.object(self.ad,'read_bounded') as read:
            with self.assertRaises(ValueError):self.ad.source_check(self.archive,self.context)
            read.assert_not_called()

    def test_unknown_context_hold_field_refused_before_metadata(self):
        self.context['new_hold']=True
        with mock.patch.object(self.ad,'read_bounded') as read:
            with self.assertRaises(ValueError):self.ad.source_check(self.archive,self.context)
            read.assert_not_called()
    def test_missing_context_basis_refused(self):
        del self.context['basis'];self.denial()
    def test_missing_custody_receipt_refused(self):
        self.custody_path.unlink();self.denial()
    def test_changed_custody_receipt_refused(self):
        self.custody_path.write_text('{}');self.denial()
    def test_unsupported_source_url_query_refused(self):
        self.payloads[0]['Original source link']+='?id=other';self.write_payloads();self.denial()
    def test_source_url_fragment_refused(self):
        self.payloads[0]['Original source link']+='#other';self.write_payloads();self.denial()
    def test_supported_drivesdk_query_keeps_same_identity(self):
        self.payloads[0]['Original source link']+='?usp=drivesdk';self.write_payloads()
        self.ad.source_check(self.archive,self.context)

    def test_context_false_scope_refused(self):
        self.context['scope']='all angles; promoted theorem';self.denial()
    def test_context_cannot_import_independence_bool(self):
        self.context['organizational_independence_credit']=False;self.denial()
    def test_context_revalidation_cannot_be_disabled(self):
        self.context['current_use_requires_revalidation']=False;self.denial()
    def test_missing_admission_refused_before_archive(self):
        self.record_path.unlink();self.denial()
    def test_changed_admission_refused_before_archive(self):
        obj=json.loads(self.record_bytes);obj['eligibility']='HELD';self.record_path.write_text(json.dumps(obj));self.denial()
    def test_extra_allowlisted_body_cannot_be_asserted(self):
        obj=json.loads(self.record_bytes);obj['body_allowlist']['rn_n6/PROOF.md']=dict(bytes=1,sha256='0'*64)
        self.record_path.write_text(json.dumps(obj));self.denial()
    def test_duplicate_admission_key_refused(self):
        self.record_path.write_bytes(b'{"schema":"first","schema":"second"}');self.denial()
    def test_symlink_admission_refused(self):
        other=self.root/'other.json';self.record_path.rename(other);self.record_path.symlink_to(other);self.denial()
    def test_missing_exclusions_refused(self):
        self.exclusions_path.unlink();self.denial()
    def test_duplicate_exclusion_key_refused(self):
        self.exclusions_path.write_text('{"exclusions":[],"exclusions":[]}');self.denial()
    def test_unknown_exclusion_kind_refused(self):
        self.exclusion(kind='unrecognized');self.denial()
    def test_ancestor_drive_hold_refused(self):
        self.exclusion(kind='drive_object',member_path='delivery');self.denial()
    def test_ancestor_hash_hold_refused(self):
        self.exclusion(payload_sha256=self.record['ancestor']['sha256'],member_path='alias');self.denial()
    def test_raw_archive_hash_hold_refused(self):
        self.exclusion(payload_sha256=self.ad.ARCHIVE_ID['sha256'],member_path='alias');self.denial()
    def test_nested_member_path_hold_refused(self):
        self.exclusion(member_path=self.ad.ARCHIVE+'!/h3_floor/PROOF.md');self.denial()
    def test_proof_hash_hold_cannot_be_hidden_by_other_carrier(self):
        self.exclusion(payload_sha256=self.ad.BODY_ALLOWLIST['h3_floor/PROOF.md']['sha256'],carrier_id='other');self.denial()
    def test_upstream_carrier_hold_refused(self):
        self.exclusion(kind='drive_object',carrier_id=self.inventory[0]['id']);self.denial()
    def test_upstream_member_hold_refused(self):
        self.exclusion(carrier_id=self.inventory[0]['id'],member_path=self.payloads[0]['Source name']);self.denial()
    def test_unrelated_held_leaf_is_not_opened_or_an_overbroad_denial(self):
        self.exclusion(member_path='rn_n6/unrelated.py',payload_sha256='0'*64)
        result=self.ad.source_check(self.archive,self.context)
        self.assertEqual(list(result['analytic_bodies_read']),['h3_floor/PROOF.md'])
    def test_missing_upstream_carrier_refused(self):
        self.inventory=[];self.write_inventory();self.denial()
    def test_ambiguous_upstream_carrier_refused(self):
        self.inventory.append(dict(self.inventory[0]));self.write_inventory();self.denial()
    def test_upstream_access_denial_refused(self):
        self.inventory[0]['access_status']='NO_ACCESS';self.write_inventory();self.denial()
    def test_upstream_legacy_path_refused(self):
        self.inventory[0]['path']='02_LEGACY_Q0_ARCHIVE/source.zip';self.write_inventory();self.denial()
    def test_upstream_context_denial_refused(self):
        self.inventory[0]['context']='QUARANTINE_OR_HISTORY';self.write_inventory();self.denial()
    def test_upstream_carrier_identity_drift_refused(self):
        self.inventory[0]['sha256']='0'*64;self.write_inventory();self.denial()
    def test_upstream_member_identity_drift_refused(self):
        self.payloads[0]['Bytes']='16700';self.write_payloads();self.denial()
    def test_upstream_scope_hold_refused(self):
        self.payloads[0]['Scope holds']='[{"key":"NEW_HOLD"}]';self.write_payloads();self.denial()
    def test_upstream_ambiguous_member_refused(self):
        self.payloads.append(dict(self.payloads[0]));self.write_payloads();self.denial()
    def test_upstream_missing_member_refused(self):
        self.payloads.pop();self.write_payloads();self.denial()
    def test_upstream_context_multiplicity_refused(self):
        self.payloads[0]['Context']='MULTIPLE_CONTEXTS';self.write_payloads();self.denial()
    def test_upstream_wrong_source_link_refused(self):
        self.payloads[0]['Original source link']='https://drive.google.com/file/d/other/view';self.write_payloads();self.denial()
    def test_duplicate_csv_header_refused(self):
        p=self.root/'drive/source_map/Payloads.csv';p.write_text('SHA-256,SHA-256\na,a\n');self.denial()
    def test_duplicate_inventory_key_refused(self):
        (self.root/'drive/inventory.jsonl').write_text('{"id":"a","id":"b"}\n');self.denial()
    def test_csv_short_row_refused(self):
        p=self.root/'drive/source_map/Payloads.csv';p.write_text(p.read_text()+'a,b\n');self.denial()
    def test_metadata_size_bound_precedes_archive(self):
        self.exclusions_path.write_bytes(b' '*(self.ad.MAX_METADATA+1));self.denial()
    def test_archive_mutation_after_valid_gate_refused(self):
        self.archive.write_bytes(b'not the source')
        with mock.patch.object(zipfile.ZipFile,'open') as opened:
            with self.assertRaises(ValueError):self.ad.source_check(self.archive,self.context)
            opened.assert_not_called()

    def test_policy_drift_after_manifest_prevents_proof_read(self):
        original=zipfile.ZipFile.read;seen=[]
        def read(z,name,*args,**kwargs):
            seen.append(name);value=original(z,name,*args,**kwargs)
            if name=='MANIFEST.json':self.exclusion()
            return value
        with mock.patch.object(zipfile.ZipFile,'read',read):
            with self.assertRaises(ValueError):self.ad.source_check(self.archive,self.context)
        self.assertEqual(seen,['MANIFEST.json'])

    def test_current_metadata_in_recipe_changes_key(self):
        first=self.ad.source_check(self.archive,self.context)
        self.inventory[0]['observation']='fresh metadata snapshot';self.write_inventory()
        second=self.ad.source_check(self.archive,self.context)
        self.assertNotEqual(c.digest(r.recipe(first,self.context,{})),c.digest(r.recipe(second,self.context,{})))

    def test_warm_cache_rechecks_eligibility(self):
        first=self.root/'first.json';self.main(first,return_value=self.certificate)
        self.exclusion()
        with mock.patch.object(zipfile.ZipFile,'open') as opened:
            with self.assertRaises(ValueError):self.main(self.root/'denied.json',side_effect=AssertionError('must not compute'))
            opened.assert_not_called()
        self.assertFalse((self.root/'denied.json').exists())

    def test_eligible_warm_hit_retains_trusted_local_label(self):
        self.main(self.root/'first.json',return_value=self.certificate)
        compute=self.main(self.root/'second.json',side_effect=AssertionError('must not compute'))
        compute.assert_not_called()
        result=json.loads((self.root/'second.json').read_text())
        self.assertEqual(result['cache'],'LOCAL_TRUSTED_HIT_NOT_NEW_EVIDENCE')
        self.assertFalse(result['automatic_promotion'])

    def test_drift_during_compute_never_publishes(self):
        def compute(**kwargs):self.exclusion();return self.certificate
        with self.assertRaises(ValueError):self.main(self.root/'denied.json',side_effect=compute)
        self.assertFalse((self.root/'denied.json').exists())

    def test_cli_real_cold_run_and_denied_retry(self):
        # Relocated minimal checkout: fixture metadata, unchanged canonical ZIP,
        # and the actual CLI. No system checkout or native source export is edited.
        destination=self.root/'engine/solver_pilots/h3_20260921';destination.mkdir(parents=True)
        for name in ('h3_solver.py','h3_source_admission.py','h3_scalar_arithmetic.py','h3_scalar_certificate.py'):
            (destination/name).write_bytes((HERE/name).read_bytes())
        (destination/'sources').mkdir()
        (destination/'sources/H3_AUTHORED_NESTED_SOURCE_V1.json').write_bytes(self.record_bytes)
        (destination/'sources/H3_CURRENT_SOURCE_CUSTODY_V1.json').write_bytes(
            (HERE/'sources/H3_CURRENT_SOURCE_CUSTODY_V1.json').read_bytes())
        cmd=[sys.executable]+(['-O'] if sys.flags.optimize else [])+[str(destination/'h3_solver.py'),
             '--source-archive',str(self.archive),'--context',str(self.context_path),
             '--out',str(self.root/'cli.json'),'--cold','--expected-certificate-sha',c.digest(self.certificate)]
        done=subprocess.run(cmd,capture_output=True,text=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(json.loads((self.root/'cli.json').read_text())['certificate'],self.certificate)
        self.exclusion();cmd[cmd.index('--out')+1]=str(self.root/'cli-denied.json')
        denied=subprocess.run(cmd,capture_output=True,text=True,timeout=30,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(denied.returncode,2,denied.stderr)
        self.assertFalse((self.root/'cli-denied.json').exists())


class ActualSourceAdmission(unittest.TestCase):
    def test_current_repository_metadata_and_exact_source(self):
        context=json.loads((HERE/'CONTEXT.json').read_text())
        result=r.source_check(r.admission.ROOT/r.admission.ARCHIVE,context)
        self.assertEqual(result['analytic_bodies_read'],r.admission.BODY_ALLOWLIST)
        self.assertIn('upstream_metadata',result['admission'])


if __name__=='__main__':unittest.main()
