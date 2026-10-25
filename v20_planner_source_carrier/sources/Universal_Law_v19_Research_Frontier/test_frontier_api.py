"""Live frontier API boundaries: owner actions, scope, quarantine and real jobs."""
from pathlib import Path
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request


class FrontierAPITests(unittest.TestCase):
    def setUp(self):
        from frontier_server import FrontierServer, FrontierHandler
        self.temp=tempfile.TemporaryDirectory(prefix='ul-frontier-api-')
        self.root=Path(self.temp.name)
        self.server=FrontierServer(('127.0.0.1',0),FrontierHandler,self.root,
                                  toolchain=self.root/'unavailable-toolchain')
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.server.stop_services();self.server.shutdown();self.server.server_close()
        self.thread.join(timeout=5);self.temp.cleanup()

    def get(self,path):
        with urllib.request.urlopen(self.base+path,timeout=15) as response:
            return json.load(response)

    def post(self,path,data,token=True):
        headers={'Content-Type':'application/json'}
        if token:headers['X-UL-Token']=self.server.token
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)

    def test_frontier_state_preserves_math_and_scoped_capabilities(self):
        state=self.get('/api/frontier/state')
        self.assertEqual(state['version'],'19.0')
        self.assertEqual(state['math']['canonical_records'],104)
        self.assertEqual(state['math']['prerequisites'],108)
        self.assertEqual(state['math']['full_theorem_certificates'],0)
        self.assertEqual(len(state['conductor']['ir']['nodes']),104)
        self.assertEqual(state['capabilities']['zk'],'unconfigured')

    def test_owner_page_and_isolation_headers(self):
        with urllib.request.urlopen(self.base+'/frontier') as response:
            self.assertEqual(response.headers['Cross-Origin-Opener-Policy'],'same-origin')
            self.assertEqual(response.headers['Cross-Origin-Embedder-Policy'],'require-corp')
            body=response.read().decode()
        self.assertIn(self.server.token,body)
        self.assertNotIn('%%LOCAL_TOKEN%%',body)

    def test_missing_owner_token_and_submitted_code_are_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/frontier/run-math',{},token=False)
        self.assertEqual(error.exception.code,403)
        for data in ({'command':['sh','-c','true']},{'profile_id':'by sorry'},{'ir':{'certified':True}}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post('/api/frontier/run-math',data)
            self.assertEqual(error.exception.code,400)
        self.assertEqual(self.get('/api/frontier/state')['jobs'],[])

    def test_plan_is_shared_and_question_bound(self):
        plan=self.post('/api/frontier/plan',{'target_ids':['D1-A'],'budget':3})
        state=self.get('/api/frontier/state')
        self.assertTrue(state['conductor']['tasks'])
        self.assertEqual(state['math']['full_theorem_certificates'],0)

    def test_contradiction_is_preserved_without_becoming_mathematical_truth(self):
        state=self.get('/api/frontier/state')
        n=next(n for n in state['conductor']['ir']['nodes'] if n['id']=='D1-A')
        payload={'proposition':n['statement'],'scope':n['scope'],'source':'owner:test',
                 'record_id':n['id'],'input_fingerprint':n['input_fingerprint']}
        self.post('/api/frontier/belief',dict(payload,polarity='support'))
        self.post('/api/frontier/belief',dict(payload,polarity='refute'))
        after=self.get('/api/frontier/state')
        self.assertTrue(any(b['state']=='both' for b in after['beliefs']['beliefs']))
        self.assertEqual(after['math']['full_theorem_certificates'],0)
        self.assertFalse(self.server.coord.snapshot()['changed_ids'])

    def test_binary_positions_bind_real_ir_and_are_not_synthetic(self):
        from frontier_gpu import decode_positions
        with urllib.request.urlopen(self.base+'/api/frontier/positions') as response:
            frame=decode_positions(response.read())
        state=self.get('/api/frontier/state')
        self.assertEqual(frame['count'],104)
        self.assertFalse(frame['synthetic'])
        self.assertEqual(frame['graph_sha256'],state['conductor']['ir']['ir_sha256'])

    def test_math_request_is_durable_deduplicated_and_blocked_without_toolchain(self):
        self.server.start_services()
        first=self.post('/api/frontier/run-math',{'event_id':'test-math'})
        again=self.post('/api/frontier/run-math',{'event_id':'test-math'})
        self.assertEqual(first['id'],again['id'])
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            state=self.get('/api/frontier/state')
            current=next(j for j in state['jobs'] if j['id']==first['id'])
            if current['status'] not in ('queued','running'):break
            time.sleep(.05)
        self.assertEqual(current['status'],'blocked')
        self.assertEqual(state['math']['full_theorem_certificates'],0)

    def test_exact_adversarial_probe_refutes_only_the_generic_false_bound(self):
        result=self.post('/api/frontier/redteam',{'power':4,'claimed_upper_bound':[131,1]})
        self.assertTrue(result['verification']['valid'])
        self.assertEqual(result['probe']['value'],[132,1])
        self.assertFalse(result['verification']['actual_field_applicable'])
        inconclusive=self.post('/api/frontier/redteam',{'power':4,'claimed_upper_bound':[132,1]})
        self.assertEqual(inconclusive['probe']['status'],'inconclusive')
        self.assertFalse(inconclusive['verification']['valid'])
        self.assertFalse(self.server.coord.snapshot()['changed_ids'])
        self.assertEqual(self.server.coord.snapshot()['full_theorem_certificates'],0)

    def test_cross_origin_and_unknown_authority_fields_are_rejected(self):
        for headers in ({'Origin':'https://attacker.invalid'}, {'Host':'attacker.invalid'}):
            req=urllib.request.Request(self.base+'/api/frontier/run-math',data=b'{}',headers={
                'Content-Type':'application/json','X-UL-Token':self.server.token,**headers})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req,timeout=5)
            self.assertEqual(error.exception.code,403)
        for path,data in [('/api/frontier/claim',{'task_id':'x','effects':['write_checker']}),
                          ('/api/frontier/belief',{'proposition':'P','scope':'S','polarity':'support','source':'a','certified':True}),
                          ('/api/frontier/annotations/merge',{'state':{'certified':True}})]:
            with self.assertRaises(urllib.error.HTTPError) as error:self.post(path,data)
            self.assertEqual(error.exception.code,400)
        self.assertEqual(self.get('/api/frontier/state')['jobs'],[])

    def test_frontier_event_stream_delivers_real_bound_state(self):
        with urllib.request.urlopen(self.base+'/api/frontier/events',timeout=15) as response:
            self.assertEqual(response.headers['Content-Type'],'text/event-stream')
            for _ in range(6):
                line=response.readline()
                if line.startswith(b'data: '):
                    state=json.loads(line[6:]);break
            else:self.fail('no frontier event snapshot')
        self.assertEqual(state['math']['canonical_records'],104)
        self.assertEqual(len(state['conductor']['ir']['nodes']),104)
        self.assertEqual(state['math']['full_theorem_certificates'],0)

    def test_annotations_cannot_modify_scientific_status(self):
        self.post('/api/frontier/annotations',{'annotation_id':'note1','record_id':'D1-A',
            'text':'Check exact scope before using this result.','tag':'note'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post('/api/frontier/annotations',{'annotation_id':'note2','record_id':'D1-A',
                'text':'unsafe','certified':True})
        self.assertEqual(error.exception.code,400)
        self.assertEqual(self.server.coord.snapshot()['full_theorem_certificates'],0)
        self.assertFalse(self.server.coord.snapshot()['changed_ids'])


if __name__=='__main__':unittest.main()
