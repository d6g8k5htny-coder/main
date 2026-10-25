import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        from workspace import WorkspaceServer, WorkspaceHandler
        self.tmp = tempfile.TemporaryDirectory()
        self.server = WorkspaceServer(('127.0.0.1', 0), WorkspaceHandler,
                                      Path(self.tmp.name), watch=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.stop_services()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def test_current_math_and_director_are_one_live_view(self):
        with urllib.request.urlopen(self.url + '/api/workspace') as response:
            data = json.load(response)
        self.assertEqual(data['version'], '18.0')
        self.assertEqual(len(data['engine']['records']), 104)
        self.assertEqual(data['formal']['full_theorem_certificates'], 0)
        self.assertTrue(data['director']['tasks'])

    def test_registered_milestones_supply_the_literature_need_vocabulary(self):
        targets = self.server.need_targets(self.server.director.snapshot())
        self.assertIn('D5-second-factorial', targets)
        self.assertIn('Borel-KR', targets)

    def test_ui_and_assets_are_local_and_unknown_assets_are_404(self):
        with urllib.request.urlopen(self.url + '/') as response:
            self.assertIn(b'Universal', response.read())
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + '/assets/../../etc/passwd')
        self.assertEqual(error.exception.code, 404)

    def test_write_requires_local_token_and_does_not_mutate_math(self):
        req = urllib.request.Request(self.url + '/api/director/claim', data=b'{}',
                                     headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        self.assertEqual(error.exception.code, 403)

    def test_bad_origin_is_rejected(self):
        req = urllib.request.Request(self.url + '/api/workspace', headers={'Origin': 'https://other.example'})
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        self.assertEqual(error.exception.code, 403)

    def test_sse_emits_current_workspace(self):
        with urllib.request.urlopen(self.url + '/api/workspace/events', timeout=5) as response:
            lines = [response.readline().decode() for _ in range(4)]
        self.assertTrue(any(x.startswith('data: ') for x in lines))
        self.assertTrue(any(x.startswith('event: ') for x in lines))

    def test_lease_expiry_changes_stream_cursor_without_a_new_journal_event(self):
        task = self.server.director.create_example_task()
        claim = self.server.director.claim(task['task_id'], 'agent:expiring', lease_seconds=1)
        before = self.server.cursor()
        time.sleep(max(0, claim['expires_at'] - time.time()) + .03)
        self.assertNotEqual(before, self.server.cursor())


if __name__ == '__main__':
    unittest.main()
