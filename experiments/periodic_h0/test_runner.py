import copy
import hashlib
import subprocess
import sys
import tempfile
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

    def test_custom_config_basename_cannot_hide_executed_source(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d); config_path=directory/'run_pilot.py'
            c=self.config();c.update(seeds=[1,2],grids=[8],cutoffs=[2],
                covariance_seeds_start=100,covariance_replicates=2,
                covariance_grid=8,covariance_cutoff=2,covariance_index_lags=[[0,0]])
            config_path.write_text(json.dumps(c))
            runner=Path(run_pilot.__file__)
            subprocess.run([sys.executable,'-B',str(runner),'--config',str(config_path),
                '--output',str(directory/'output')],check=True,capture_output=True,text=True)
            receipt=json.loads((directory/'output/RUN.json').read_text())
            self.assertEqual(set(receipt['sources']),{'runner','model','controls','config','dependencies'})
            for role,path in [('runner',runner),('config',config_path)]:
                self.assertEqual(receipt['sources'][role]['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(receipt['sources'][role]['bytes'],len(path.read_bytes()))
            self.assertNotEqual(receipt['sources']['runner']['sha256'],receipt['sources']['config']['sha256'])
            self.assertEqual(receipt['source_receipt_schema'],2)

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
