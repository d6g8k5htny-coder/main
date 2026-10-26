"""Source-only semantic RED preparation; never executed by this author.

Root must load these exact bytes and the four availability modules through its
frozen loader. Default-None stubs intentionally fail real behavior assertions.
"""
import unittest
from copy import deepcopy
from fractions import Fraction as Q
import field
import field_certificate as fc
import prediction
import run_falsifier as run

A=Q(1,65536)
KS=(8,12,15,16)
EVENT={'id':'count','lifetime_upper':Q(1,4),'separation_upper':None,'birth_upper':None}

def cell(qa=Q(0),qb=Q(1),za=Q(0),zb=Q(1),epsilon=1):
    return dict(Qa=qa,Qb=qb,Za=za,Zb=zb,epsilon=epsilon)

def grid(nonzero):
    out=[0]*25
    for i,v in nonzero.items():
        out[i]=v
    return out

def rows(low,high):
    out=[]
    for sign in (-1,1):
        out.append({'cell':cell(qb=Q(1,2),epsilon=sign),'event_bounds':{'count':low}})
        out.append({'cell':cell(qa=Q(1,2),epsilon=sign),'event_bounds':{'count':high}})
    return out

def result():
    return {'status':'FINITE_WINDOW_CONSISTENT','full_declared_campaign':True,
            'rows':[{'status':'FINITE_WINDOW_CONSISTENT','qualified':True}]}

class FalsifierTests(unittest.TestCase):
    def call(self,fn,*args,**kwargs):
        try:
            got=fn(*args,**kwargs)
        except NotImplementedError as error:
            self.fail('Feature unavailable: '+fn.__name__+': '+str(error))
        self.assertIsNotNone(got,'Feature unavailable: '+fn.__name__)
        return got

    def test_exact_control(self):
        got=self.call(field.control,Q(1,2),Q(1,2),1)
        self.assertEqual(got['u'],3*A**2/4)
        self.assertEqual(got['v'],255*A**3/1024)
        other=self.call(field.control,Q(1,2),Q(1,2),-1)
        self.assertEqual(other['u'],got['u'])
        self.assertEqual(other['v'],-got['v'])

    def test_actual_core_nodal_values(self):
        c=self.call(field.control,Q(1,2),Q(1,2),1)
        for x,y,want in ((Q(1),Q(0),Q(383,1024)),(Q(-1),Q(0),Q(-127,1024)),(Q(0),Q(2),Q(-4))):
            with self.subTest(x=x,y=y):
                self.assertEqual(self.call(field.scaled_core_value,x,y,c),want)

    def test_whole_exterior_is_seed(self):
        c=self.call(field.control,Q(1,2),Q(1,2),1)
        self.assertEqual(self.call(field.full_residual_bounds,Q(1,2),Q(0),c),(Q(-1,4),Q(-1,4)))

    def test_whole_collar_is_exact_seed(self):
        c=self.call(field.control,Q(1,2),Q(1,2),-1)
        self.assertEqual(self.call(field.full_residual_bounds,Q(1,8),Q(0),c),(Q(-1,64),Q(-1,64)))

    def test_flat_step_exact_points(self):
        for z in (Q(0),Q(1,2),Q(1)):
            self.assertEqual(self.call(field.step_bounds,z),(z,z))

    def test_both_signs_normalize(self):
        for k in KS:
            for sign in (-1,1):
                self.assertEqual(self.call(field.cell_mass,cell(epsilon=sign),k),Q(1,2))

    def test_distinct_law_weights(self):
        for k,want in ((8,Q(1,512)),(12,Q(1,8192)),(15,Q(1,65536)),(16,Q(1,131072))):
            self.assertEqual(self.call(field.cell_mass,cell(zb=Q(1,2)),k),want)
            self.assertEqual(self.call(field.cell_mass,cell(qb=Q(1,2)),k),Q(1,64))

    def test_boundary_strip_has_positive_mass(self):
        for k in KS:
            mass=sum(self.call(field.cell_mass,cell(qb=Q(1,2),epsilon=s),k) for s in (-1,1))
            self.assertEqual(mass,Q(1,32))

    def test_invalid_controls_rejected(self):
        for c,k in ((cell(qa=Q(3,4),qb=Q(1,4)),8),(cell(qb=Q(1,3)),8),(cell(epsilon=0),8),(cell(epsilon=True),8),(cell(),9)):
            with self.subTest(c=c,k=k):
                with self.assertRaises(ValueError):
                    field.cell_mass(c,k)

    def test_physical_metric_not_chart_gap(self):
        got=self.call(field.physical_mark,(Q(-1),Q(-1)),(Q(1),Q(1)))
        self.assertGreater(got['separation'][0],3*A/4)
        self.assertLess(got['separation'][1],4*A/5)
        back=self.call(field.physical_mark,(Q(1),Q(1)),(Q(-1),Q(-1)))
        self.assertEqual(got['separation'],back['separation'])
        self.assertEqual(got['direction'],-back['direction'])

    def test_overlapping_locations_do_not_resolve(self):
        got=self.call(field.physical_mark,(Q(-1),Q(1)),(Q(0),Q(2)))
        self.assertFalse(got['resolved'])

    def test_actual_elder_selection(self):
        got=self.call(fc.observe_grid,grid({11:9,12:3,13:7}),5)
        self.assertEqual(got['barcode']['intervals'],[[7,3]])
        self.assertEqual(got['barcode']['birth_vertices'],[13])
        self.assertEqual(got['barcode']['essential'],[9])
        self.assertEqual(got['barcode']['essential_vertices'],[11])
        self.assertEqual(got['barcode']['zero_count'],23)

    def test_global_bypass_beats_local_middle(self):
        got=self.call(fc.observe_grid,grid({6:5,7:5,8:5,11:9,12:3,13:7}),5)
        self.assertEqual(got['barcode']['intervals'],[[7,5]])
        self.assertEqual(got['barcode']['birth_vertices'],[13])
        self.assertTrue(got['independent_bfs_verified'])

    def test_hidden_bar_is_preserved(self):
        got=self.call(fc.observe_grid,grid({0:6,11:9,12:3,13:7}),5)
        self.assertEqual(got['barcode']['intervals'],[[6,0],[7,3]])
        self.assertEqual(got['barcode']['birth_vertices'],[0,13])
        self.assertEqual(got['barcode']['zero_count'],22)

    def test_periodic_seam_route(self):
        got=self.call(fc.observe_grid,grid({10:9,13:7,14:5}),5)
        self.assertEqual(got['barcode']['intervals'],[[7,5]])
        self.assertEqual(got['barcode']['essential_vertices'],[10])

    def witness(self):
        values=grid({6:5,7:5,8:5,11:9,12:3,13:7})
        barcode={'intervals':[[7,5]],'birth_vertices':[13],'essential':[9],'essential_vertices':[11],'zero_count':23}
        witness={'birth_vertex':13,'death_level':5,'older_vertex':11,'path':[13,8,7,6,11],'cut_vertices':[13]}
        return values,barcode,witness

    def test_valid_actual_path_and_cut(self):
        values,barcode,w=self.witness()
        self.assertTrue(self.call(fc.verify_merge_witness,values,5,barcode,w))

    def test_path_nonedge_jump_rejected(self):
        values,barcode,w=self.witness()
        w['path']=[13,11]
        self.assertFalse(self.call(fc.verify_merge_witness,values,5,barcode,w))

    def test_cut_must_be_exhaustive(self):
        values,barcode,w=self.witness()
        w['cut_vertices']=[]
        self.assertFalse(self.call(fc.verify_merge_witness,values,5,barcode,w))

    def test_wrong_death_fails_plus_one_connectivity(self):
        values,barcode,w=self.witness()
        barcode['intervals']=[[7,3]]
        w['death_level']=3
        self.assertFalse(self.call(fc.verify_merge_witness,values,5,barcode,w))

    def test_path_must_reach_older_vertex(self):
        values,barcode,w=self.witness()
        w['older_vertex']=13
        w['path']=[13]
        self.assertFalse(self.call(fc.verify_merge_witness,values,5,barcode,w))

    def test_all_three_uniform_root_regions_required(self):
        c=cell(Q(511,1024),Q(513,1024),Q(511,1024),Q(513,1024))
        roots=self.call(fc.uniform_roots,c)
        self.assertTrue(self.call(fc.verify_root_cover,c,roots))
        bad=deepcopy(roots)
        bad['intervals']=roots['intervals'][:1]+roots['intervals'][2:]
        self.assertFalse(self.call(fc.verify_root_cover,c,bad))

    def test_boundary_root_cell_unresolved(self):
        got=self.call(fc.uniform_roots,cell(qb=Q(1,2),za=Q(1,4),zb=Q(3,4)))
        self.assertEqual(got['status'],'UNRESOLVED_ROOTS')

    def test_exact_integer_nodal_floor(self):
        c=self.call(field.control,Q(1,2),Q(1,2),1)
        for x,y,want in ((Q(1),Q(0),383*2**70),(Q(-1),Q(0),-127*2**70),(Q(0),Q(2),-4*2**80),(Q(8),Q(0),-16*2**80)):
            self.assertEqual(self.call(fc.integer_sample,x,y,c),want)

    def coupled_fixture(self):
        width=Q(1,2**24)
        c=cell(Q(1,2),Q(1,2)+width,Q(1,2),Q(1,2)+width)
        roots=self.call(fc.uniform_roots,c)
        g=self.call(fc.build_grid,c,roots)
        observation=self.call(fc.observe_grid,g['values'],8)
        return c,roots,g,observation

    def test_full64_actual_grid_and_floor_boundary(self):
        c,roots,g,observation=self.coupled_fixture()
        self.assertEqual(len(g['values']),64)
        self.assertEqual(g['axis_y_index'],4)
        for i in range(8):
            for j in range(8):
                if i in (0,7) or j in (0,7):
                    self.assertEqual(g['values'][i*8+j],-16*2**80)
        self.assertTrue(observation['independent_bfs_verified'])

    def test_whole_cell_coupling_uses_observed_saddle_witness(self):
        c,roots,g,observation=self.coupled_fixture()
        coupling=self.call(fc.couple_cell,c,roots,g,observation)
        self.assertEqual(coupling['status'],'RESOLVED')
        self.assertEqual(coupling['saddle_root_index'],1)
        self.assertGreater(coupling['endpoint_error_scaled'],Q(1,2**80))
        self.assertTrue(self.call(fc.verify_coupling_certificate,c,roots,g,observation,coupling))

    def test_forged_coupling_error_is_rejected(self):
        c,roots,g,observation=self.coupled_fixture()
        coupling=self.call(fc.couple_cell,c,roots,g,observation)
        bad=deepcopy(coupling)
        bad['endpoint_error_scaled']=Q(0)
        self.assertFalse(self.call(fc.verify_coupling_certificate,c,roots,g,observation,bad))

    def test_forged_saddle_location_is_rejected(self):
        c,roots,g,observation=self.coupled_fixture()
        coupling=self.call(fc.couple_cell,c,roots,g,observation)
        bad=deepcopy(coupling)
        bad['saddle_root_index']=0
        self.assertFalse(self.call(fc.verify_coupling_certificate,c,roots,g,observation,bad))

    def test_changed_actual_node_invalidates_coupling(self):
        c,roots,g,observation=self.coupled_fixture()
        coupling=self.call(fc.couple_cell,c,roots,g,observation)
        bad=deepcopy(g)
        bad['values'][12]+=1
        self.assertFalse(self.call(fc.verify_coupling_certificate,c,roots,bad,observation,coupling))

    def event_certificate(self,life,r=(Q(0),Q(0))):
        return {'status':'RESOLVED','lifetime_scaled':life,'separation_over_A':r,'centered_birth_scaled':(Q(-1),Q(-1))}

    def test_lifetime_event_uses_full_interval(self):
        for interval,want in (((Q(1,8),Q(1,4)),(1,1)),((Q(1,4),Q(3,8)),(0,1)),((Q(3,8),Q(1,2)),(0,0)),((Q(1,4),Q(1,4)),(1,1))):
            self.assertEqual(self.call(run.classify_event,self.event_certificate(interval),EVENT),want)

    def test_joint_event_uses_physical_bounds(self):
        event=dict(EVENT,separation_upper=Q(1,16))
        for interval,want in (((Q(1,32),Q(1,16)),(1,1)),((Q(1,32),Q(3,32)),(0,1)),((Q(3,32),Q(1,8)),(0,0))):
            self.assertEqual(self.call(run.classify_event,self.event_certificate((Q(1,8),Q(1,8)),interval),event),want)

    def test_false_conjunction_dominates_unknown_mark(self):
        event=dict(EVENT,separation_upper=Q(1,16))
        self.assertEqual(self.call(run.classify_event,self.event_certificate((Q(1,2),Q(3,4)),(Q(0),Q(1))),event),(0,0))

    def test_unresolved_mass_never_dropped(self):
        got=self.call(run.aggregate_cells,rows((0,1),(0,0)),[EVENT])
        for k in KS:
            self.assertEqual(got['law_bounds'][k]['count'],(Q(0),Q(1,32)))
            self.assertEqual(got['total_mass'][k],Q(1))

    def test_no_success_only_renormalization(self):
        got=self.call(run.aggregate_cells,rows((1,1),(0,1)),[EVENT])
        for k in KS:
            self.assertEqual(got['law_bounds'][k]['count'],(Q(1,32),Q(1)))

    def test_partition_has_no_gaps_or_duplicate_leaves(self):
        cells=[row['cell'] for row in rows((0,0),(0,0))]
        self.assertTrue(self.call(run.validate_partition,cells))
        self.assertFalse(self.call(run.validate_partition,cells[:-1]))
        self.assertFalse(self.call(run.validate_partition,cells+[cells[0]]))

    def test_broad_overlap_is_inconclusive(self):
        self.assertEqual(self.call(run.compare_row,(Q(0),Q(1)),(Q(1,4),Q(1,3)),True),'INCONCLUSIVE_PRECISION')

    def test_precise_positive_overlap(self):
        self.assertEqual(self.call(run.compare_row,(Q(1,4),Q(33,128)),(Q(1,4),Q(1,3)),True),'FINITE_WINDOW_CONSISTENT')

    def test_disjoint_bounds_need_qualification(self):
        obs,pred=(Q(0),Q(1,8)),(Q(1,4),Q(1,3))
        self.assertEqual(self.call(run.compare_row,obs,pred,True),'FINITE_WINDOW_FALSIFIED_CANDIDATE')
        self.assertEqual(self.call(run.compare_row,obs,pred,False),'INCONCLUSIVE_COUPLING')

    def test_zero_rule_has_no_hidden_tolerance(self):
        for obs,pred,want in (((Q(0),Q(0)),(Q(0),Q(0)),'FINITE_WINDOW_CONSISTENT_ZERO'),((Q(0),Q(1,1024)),(Q(0),Q(0)),'INCONCLUSIVE_PRECISION'),((Q(0),Q(0)),(Q(0),Q(1,1024)),'INCONCLUSIVE_PRECISION')):
            self.assertEqual(self.call(run.compare_row,obs,pred,True),want)

    def test_source_withdrawal_reaches_nested_rows(self):
        got=self.call(run.withdraw_acceptance,result(),'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertEqual(got['status'],'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertFalse(got['full_declared_campaign'])
        self.assertFalse(got['rows'][0]['qualified'])
        self.assertEqual(got['rows'][0]['status'],'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertEqual(got['rows'][0]['computed_status_before_withdrawal'],'FINITE_WINDOW_CONSISTENT')

    def test_late_budget_withdraws_prior_acceptance(self):
        got=self.call(run.withdraw_acceptance,result(),'INCONCLUSIVE_BUDGET')
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        self.assertFalse(got['full_declared_campaign'])
        self.assertFalse(got['rows'][0]['qualified'])

    def final_inputs(self):
        before={'field.py':{'bytes':1,'sha256':'a'*64}}
        terminal={'observed':True,'returncode':0,'owned_child_pid':123,'waited':True}
        budget={'exhausted':False,'elapsed_seconds':1,'wall_limit_seconds':120}
        return before,deepcopy(before),terminal,budget

    def test_after_read_error_withdraws_and_is_retained(self):
        before,after,terminal,budget=self.final_inputs()
        after['field.py']={'error':'OSError: fixture read failed'}
        got=self.call(run.finalize_attempt,result(),before,after,terminal,budget)
        self.assertEqual(got['status'],'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertFalse(got['rows'][0]['qualified'])
        self.assertEqual(got['source_after'],after)

    def test_missing_owned_child_terminal_cannot_pass(self):
        before,after,terminal,budget=self.final_inputs()
        terminal={'observed':False,'owned_child_pid':123,'waited':False}
        got=self.call(run.finalize_attempt,result(),before,after,terminal,budget)
        self.assertEqual(got['status'],'TERMINAL_UNOBSERVED')
        self.assertFalse(got['full_declared_campaign'])

    def test_nonzero_owned_child_exit_cannot_pass(self):
        before,after,terminal,budget=self.final_inputs()
        terminal['returncode']=1
        got=self.call(run.finalize_attempt,result(),before,after,terminal,budget)
        self.assertEqual(got['status'],'CHILD_FAILED')
        self.assertFalse(got['rows'][0]['qualified'])

    def test_final_posthash_budget_is_inside_limit(self):
        before,after,terminal,budget=self.final_inputs()
        budget['exhausted']=True
        got=self.call(run.finalize_attempt,result(),before,after,terminal,budget)
        self.assertEqual(got['status'],'INCONCLUSIVE_BUDGET')
        self.assertFalse(got['full_declared_campaign'])

if __name__=='__main__':
    unittest.main()
