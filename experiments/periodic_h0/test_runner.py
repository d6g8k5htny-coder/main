import copy
import json
from pathlib import Path
import unittest
import run_pilot

class RunnerTests(unittest.TestCase):
    def config(self):
        return json.loads(Path(__file__).with_name('pilot_config.json').read_text())

    def test_bad_or_dependent_design_rejected_before_execution(self):
        cases=[('seeds',[1,1]),('grids',[64,64]),('cutoffs',[24,12]),
               ('seeds',[94000,94001]),('covariance_replicates',1),('grids',[48]),
               ('side',25),('covariance_index_lags',[[1,0]]),('bin_edges',[.2,.1]),('seeds',[True,2])]
        self.assertTrue(hasattr(run_pilot,'validate_config'),'Config validation is not yet implemented')
        for key,value in cases:
            c=self.config();c[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):run_pilot.validate_config(c)

    def test_saved_observations_are_auditable_and_tampering_rejected(self):
        self.assertTrue(hasattr(run_pilot,'verify_observations'),'Output audit is not yet implemented')
        p=Path(__file__).parent/'results/pilot32/observations.json'
        data=json.loads(p.read_text())
        self.assertTrue(run_pilot.verify_observations(data))
        for mutate in ('count','essential','summary','duplicate','covariance','target','spectrum','control','missing_samples'):
            d=copy.deepcopy(data)
            if mutate=='count':d['records'][0]['counts'][0]+=1
            elif mutate=='essential':d['records'][0]['essential']=[]
            elif mutate=='summary':d['summary'][0]['bins'][0]['mean_mass']+=1
            elif mutate=='duplicate':d['records'][1]=copy.deepcopy(d['records'][0])
            elif mutate=='covariance':d['covariance_empirical']['mean_products'][0]+=1
            elif mutate=='target':d['covariance_empirical']['finite_target'][0]+=1
            elif mutate=='spectrum':d['spectral_diagnostics']['24']['variance_retained']=.5
            elif mutate=='control':d['controls']['mutants_rejected']['wrong_elder']=False
            else:del d['covariance_empirical']['point_samples']
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):run_pilot.verify_observations(d)

if __name__=='__main__':unittest.main()
