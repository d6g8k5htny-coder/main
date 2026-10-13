"""Skeptic addendum: the finder's literal 'cite the PR HEAD itself' case is a SHA-1 fixed point
(IDENTITY.json would have to contain the SHA of the commit whose tree contains IDENTITY.json) and
cannot be produced in real Git. The REALIZABLE variant: commit P (an ancestor of HEAD, authored in
the fork) adds incoming/<id>/output.json; HEAD adds RESULT.md + IDENTITY.json citing P as a
'd6g8k5htny-coder/main' source. The PR diff (merge-base..HEAD) still shows output.json as 'added'.
On the e495837 checker this is green with verified_sources=1; on origin/main it is refused."""
import os, unittest
from harness import Base, REPO, HEAD, BASE, sha

PARENT = 'f' * 40
EXPECT_GREEN = os.environ.get('EXPECT', 'green') == 'green'


class ParentSelfCitation(Base):
    def test_parent_commit_copy_of_own_output_cited_as_main_source(self):
        own = self.raw['output.json']
        self.manifest['sources'] = [{'repository': REPO, 'path': 'incoming/example/output.json', 'commit': PARENT, 'sha256': sha(own)}]
        self.save()
        self.install_source(REPO, PARENT, 'incoming/example/output.json', own, reachable_from=())
        if EXPECT_GREEN:
            result = self.run_check()
            self.assertEqual((result['lane'], result['verified_sources']), ('incoming', 1))
            self.assertEqual(self.reachability_routes_requested(), [])
        else:
            self.reject('not reachable')

    def test_literal_head_self_citation_is_a_hash_cycle(self):
        # Demonstrate the impossibility: try to build a real git commit whose IDENTITY.json names
        # its own SHA. Any candidate SHA written into the file changes the file, hence the tree,
        # hence the commit SHA. Ten rounds of iteration never converge (and cannot, absent a preimage).
        import hashlib, json
        def git_obj(kind, body): return hashlib.sha1(kind + b' ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        candidate = '0' * 40
        for _ in range(10):
            identity = json.dumps({'commit': candidate}).encode()
            blob_sha = git_obj(b'blob', identity)
            tree = b'100644 IDENTITY.json\0' + bytes.fromhex(blob_sha)
            tree_sha = git_obj(b'tree', tree)
            commit = ('tree %s\nauthor a <a> 0 +0000\ncommitter a <a> 0 +0000\n\nx\n' % tree_sha).encode()
            new = git_obj(b'commit', commit)
            self.assertNotEqual(new, candidate)
            candidate = new


if __name__ == '__main__':
    unittest.main()
