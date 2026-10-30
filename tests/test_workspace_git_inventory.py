"""Real-Git census tests: discovery, native state, failure boundaries and privacy."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unittest

TOOL=Path(__file__).resolve().parents[1]/'tools/workspace_git_inventory.py'


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'workspace';self.root.mkdir()

    def git(self,path,*args):
        return subprocess.check_output(['git','-C',str(path),*args],stderr=subprocess.PIPE)

    def repo(self,name):
        p=self.root/name;p.mkdir(parents=True);self.git(p,'init','-q')
        self.git(p,'config','user.name','Census Test');self.git(p,'config','user.email','census@example.invalid')
        (p/'tracked').write_text('initial\n');self.git(p,'add','tracked');self.git(p,'commit','-qm','initial')
        return p

    def census(self,**options):
        spec=importlib.util.spec_from_file_location('git_inventory',TOOL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.inventory(self.root,**options)

    def test_nested_linked_state_refs_stash_and_redacted_remotes(self):
        repo=self.repo('primary');self.git(repo,'branch','other');self.git(repo,'tag','v1')
        self.git(repo,'tag','-am','annotated test','annotated')
        (repo/'tracked').write_text('stashed\n');self.git(repo,'stash','push','-qm','retained stash')
        self.git(repo,'remote','add','origin','https://name:password@example.invalid/repo.git?token=secret')
        self.git(repo,'remote','add','scp','private-user@example.invalid:repo.git')
        linked=self.root/('linked'+chr(10)+'worktree');self.git(repo,'worktree','add','-qb','linked-branch',str(linked))
        self.git(linked,'update-ref','refs/worktree/census','HEAD')
        nested=self.repo('primary/nested');dependency=self.repo('primary/node_modules/package')
        (repo/'tracked').write_text('dirty\n');name='untracked'+chr(10)+'name'+chr(9)+'and spaces';(repo/name).write_text('new')
        before=(repo/'.git/index').read_bytes()
        metadata={str(p.relative_to(repo/'.git')):p.read_bytes() for p in (repo/'.git').rglob('*') if p.is_file()}
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        roots={r['path']:r for r in d['worktrees']}
        for p in [repo,linked,nested,dependency]:self.assertIn(str(p),roots)
        self.assertEqual(len(d['stores']),3)
        self.assertEqual(roots[str(linked)]['branch'],'refs/heads/linked-branch')
        self.assertTrue(any(r['name']=='refs/worktree/census'for r in roots[str(linked)]['local_refs']))
        status=roots[str(repo)]['status'];self.assertTrue(any(r['path']==name and r['xy']=='??'for r in status))
        self.assertTrue(any(r['path']=='tracked'and'M'in r['xy']for r in status))
        store=next(s for s in d['stores']if s['path']==str(repo/'.git'))
        refs={r['name']:r['object']for r in store['refs']};self.assertIn('refs/heads/other',refs);self.assertIn('refs/tags/v1',refs)
        tag=next(r for r in store['refs']if r['name']=='refs/tags/annotated')
        self.assertEqual(tag['type'],'tag');self.assertEqual(tag['peeled_object'],self.git(repo,'rev-parse','HEAD').decode().strip())
        self.assertEqual(len(store['stashes']),1)
        raw=json.dumps(d);self.assertNotIn('password',raw);self.assertNotIn('token=secret',raw)
        self.assertIn('REDACTED',raw);self.assertEqual((repo/'.git/index').read_bytes(),before)
        self.assertNotIn('private-user',raw)
        self.assertEqual(metadata,{str(p.relative_to(repo/'.git')):p.read_bytes() for p in (repo/'.git').rglob('*') if p.is_file()})

    def test_submodule_bare_store_shallow_and_internal_symlink(self):
        source=self.repo('source');parent=self.repo('parent')
        subpath='sub'+chr(10)+'module'
        self.git(parent,'-c','protocol.file.allow=always','submodule','add','-q','--name','module',str(source),subpath)
        bare=self.root/'bare.git';subprocess.check_call(['git','clone','-q','--bare',str(source),str(bare)])
        shallow=self.root/'shallow';subprocess.check_call(['git','clone','-q','--depth','1',source.as_uri(),str(shallow)])
        (self.root/'alias').symlink_to(parent,target_is_directory=True)
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        work=next(r for r in d['worktrees']if r['path']==str(parent))
        self.assertEqual(work['submodules'][0]['path'],subpath);self.assertTrue(work['submodules'][0]['initialized'])
        self.assertTrue(next(s for s in d['stores']if s['path']==str(shallow/'.git'))['shallow'])
        self.assertTrue(any(s['path']==str(bare)and s['bare']for s in d['stores']))
        self.assertTrue(any(s['path']==str(self.root/'alias')for s in d['symlinks']))

    def test_broken_marker_and_external_directory_boundary_are_not_silent(self):
        self.repo('good');broken=self.root/'broken';broken.mkdir();(broken/'.git').write_text('gitdir: /does/not/exist\n')
        cycle=self.root/'cyclic-marker';cycle.mkdir();(cycle/'.git').symlink_to(cycle/'.git')
        external=Path(self.temp.name)/'outside';external.mkdir();(self.root/'external').symlink_to(external,target_is_directory=True)
        d=self.census();self.assertFalse(d['complete']);self.assertTrue(d['errors'])
        self.assertTrue(any(e['kind']=='external_directory_symlink'for e in d['errors']))
        self.assertIn(str(broken),d['discovered_roots'])

    def test_cli_private_output_and_nonzero_incomplete(self):
        self.repo('good');bad=self.root/'broken';bad.mkdir();(bad/'.git').write_text('invalid\n')
        out=Path(self.temp.name)/'inventory.json'
        r=subprocess.run(['python3','-S',str(TOOL),'--root',str(self.root),'--output',str(out)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertTrue(out.exists(),r.stderr)
        self.assertEqual(stat.S_IMODE(out.stat().st_mode),0o600)
        self.assertFalse(json.loads(out.read_bytes())['complete'])

    def test_hanging_git_is_killed_and_reported(self):
        self.repo('repo');fake=Path(self.temp.name)/'git-hang'
        fake.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n');fake.chmod(0o700)
        d=self.census(git=str(fake),command_timeout=.1,total_timeout=3)
        self.assertFalse(d['complete']);self.assertTrue(any(e['kind']=='timeout'for e in d['errors']))

    def test_closed_streams_do_not_leave_a_hanging_child(self):
        self.repo('repo');fake=Path(self.temp.name)/'git-closed';pidfile=Path(self.temp.name)/'pid'
        fake.write_text('#!/usr/bin/env python3\nimport os,time,pathlib\npathlib.Path('+repr(str(pidfile))+').write_text(str(os.getpid()))\nos.close(1)\nos.close(2)\ntime.sleep(30)\n')
        fake.chmod(0o700)
        try:
            d=self.census(git=str(fake),command_timeout=.1,total_timeout=3)
            self.assertFalse(d['complete']);self.assertTrue(any(e['kind']=='timeout'for e in d['errors']))
            with self.assertRaises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)
        finally:
            if pidfile.exists():
                try:os.killpg(int(pidfile.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass

    def test_rename_arbitrary_path_bytes_and_symlink_cycle(self):
        repo=self.repo('repo');old='old'+chr(10)+'name';new='new'+chr(9)+os.fsdecode(b'\xff')
        (repo/old).write_text('content');self.git(repo,'add',old);self.git(repo,'commit','-qm','path')
        self.git(repo,'mv',old,new)
        (self.root/'cycle-a').symlink_to(self.root/'cycle-b');(self.root/'cycle-b').symlink_to(self.root/'cycle-a')
        d=self.census();self.assertFalse(d['complete']);self.assertTrue(any(e['kind']=='filesystem'for e in d['errors']))
        r=next(w for w in d['worktrees']if w['path']==str(repo))
        renamed=next(r for r in r['status']if'R'in r['xy'])
        self.assertEqual((renamed['path'],renamed['original_path']),(new,old))
        self.assertEqual(json.loads(json.dumps(d))['worktrees'],d['worktrees'])

    def test_status_does_not_execute_configured_clean_filter(self):
        repo=self.repo('filtered');marker=Path(self.temp.name)/'FILTER-RAN'
        script=Path(self.temp.name)/'filter.py';script.write_text('import pathlib,sys\npathlib.Path('+repr(str(marker))+').write_text("bad")\nsys.stdout.write(sys.stdin.read())\n')
        self.git(repo,'config','filter.audit.clean','python3 '+str(script))
        (repo/'.gitattributes').write_text('tracked filter=audit\n');(repo/'tracked').write_text('changed\n')
        d=self.census();self.assertTrue(d['complete'],d['errors']);self.assertFalse(marker.exists())

        nested_source=self.repo('nested-source')
        (nested_source/'.gitattributes').write_text('tracked filter=nested-audit\n')
        self.git(nested_source,'add','.gitattributes')
        self.git(nested_source,'commit','-qm','configure nested filter')

        source=self.repo('source')
        self.git(source,'-c','protocol.file.allow=always','submodule','add','-q',
                 '--name','nested',str(nested_source),'nested')
        self.git(source,'commit','-qm','add nested submodule')
        self.git(source,'submodule','deinit','-f','nested')

        parent=self.repo('parent')
        self.git(parent,'-c','protocol.file.allow=always','submodule','add','-q',
                 '--name','module',str(source),'module')
        self.git(parent,'commit','-qm','add external-worktree submodule')
        self.git(parent,'submodule','deinit','-f','module')

        linked=Path(self.temp.name)/'external-linked'
        self.git(parent,'worktree','add','-qb','external-linked',str(linked))
        self.git(linked,'-c','protocol.file.allow=always','submodule',
                 'update','--init','--recursive')
        child=linked/'module'
        nested=child/'nested'
        markers=[Path(self.temp.name)/'SUBMODULE-FILTER-RAN',
                 Path(self.temp.name)/'NESTED-SUBMODULE-FILTER-RAN']
        for path,name,marker in [(child,'audit',markers[0]),
                                 (nested,'nested-audit',markers[1])]:
            script=Path(self.temp.name)/(name+'-filter.py')
            script.write_text(
                'import pathlib,sys\n'
                'pathlib.Path('+repr(str(marker))+').write_text("ran")\n'
                'sys.stdout.buffer.write(sys.stdin.buffer.read())\n')
            command=shlex.quote(sys.executable)+' '+shlex.quote(str(script))
            self.git(path,'config','filter.'+name+'.clean',command)
            self.git(path,'config','filter.'+name+'.required','true')
            (path/'tracked').write_text('modified\n')

        self.git(linked,'status','--porcelain=v1','-z','--ignore-submodules=none')
        self.assertTrue(markers[0].exists(),
                        'Native parent status did not run the direct submodule filter')
        markers[0].unlink()
        self.git(child,'status','--porcelain=v1','-z','--ignore-submodules=none')
        self.assertTrue(markers[1].exists(),
                        'Native child status did not run the nested submodule filter')
        for marker in markers:
            if marker.exists():
                marker.unlink()

        spec=importlib.util.spec_from_file_location('git_inventory_preflight',TOOL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        reader=module.Git('git',10,30,32*1024*1024)
        settings,submodules=module._preflight_submodule_settings(
            reader,{str(linked.resolve()):str(parent/'.git')},[])
        self.assertTrue(reader.configuration_complete,reader.errors)
        self.assertIn(str(linked.resolve()),submodules)
        self.assertIn(str(child.resolve()),submodules)
        self.assertIn(str(nested.resolve()),submodules)
        self.assertIn('filter.audit.clean=',settings)
        self.assertIn('filter.nested-audit.clean=',settings)

        d=self.census();self.assertTrue(d['complete'],d['errors'])
        self.assertFalse(any(marker.exists() for marker in markers),
                         'Collector executed an external submodule clean filter')

    def test_output_limit_and_total_deadline_are_explicit(self):
        self.repo('repo');fake=Path(self.temp.name)/'git-noisy'
        fake.write_text('#!/usr/bin/env python3\nimport sys\nsys.stdout.write("x"*100000)\n');fake.chmod(0o700)
        d=self.census(git=str(fake),max_command_bytes=128)
        self.assertFalse(d['complete']);self.assertTrue(any(e['kind']=='output_limit'for e in d['errors']))
        d=self.census(total_timeout=0)
        self.assertFalse(d['complete']);self.assertTrue(any(e['kind']=='deadline'for e in d['errors']))

    def test_metadata_directory_alias_does_not_escape_git_exclusion(self):
        repo=self.repo('repo');hidden=repo/'.git/objects/hidden';hidden.mkdir()
        self.git(hidden,'init','-q');(self.root/'objects-alias').symlink_to(repo/'.git/objects',target_is_directory=True)
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        self.assertNotIn(str(hidden),d['discovered_roots'])

    def test_unborn_and_detached_heads_are_observations_not_errors(self):
        empty=self.root/'unborn';empty.mkdir();self.git(empty,'init','-q')
        repo=self.repo('detached');self.git(repo,'checkout','-q','--detach')
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        by={w['path']:w for w in d['worktrees']}
        self.assertTrue(by[str(empty)]['unborn']);self.assertIsNone(by[str(empty)]['head'])
        self.assertTrue(by[str(repo)]['detached']);self.assertIsNone(by[str(repo)]['branch'])

    def test_equals_filter_keys_are_suppressed_in_direct_and_recursive_status(self):
        source=self.repo('source');parent=self.repo('parent')
        self.git(parent,'-c','protocol.file.allow=always','submodule','add','-q',str(source),'module')
        child=parent/'module';markers=[]
        for number,(repo,name) in enumerate([(parent,'eq=name'),(child,'eq==name')]):
            marker=Path(self.temp.name)/('FILTER-'+str(number));markers.append(marker)
            script=Path(self.temp.name)/('filter-'+str(number)+'.py')
            script.write_text('import pathlib,sys\npathlib.Path('+repr(str(marker))+').write_text("executed")\nsys.stdout.buffer.write(sys.stdin.buffer.read())\n')
            self.git(repo,'config','filter.'+name+'.clean',shlex.quote(sys.executable)+' '+shlex.quote(str(script)))
            self.git(repo,'config','filter.'+name+'.required','true')
            (repo/'.gitattributes').write_text('tracked filter='+name+'\n')
            self.assertEqual(self.git(repo,'check-attr','-z','filter','--','tracked').split(b'\0')[2],name.encode())
            (repo/'tracked').write_text('changed\n')
        # Native parent status exercises both its own and recursive child filter.
        self.git(parent,'status','--porcelain=v1','-z','--ignore-submodules=none')
        self.assertTrue(all(p.exists()for p in markers),'Native recursive positive control did not run both filters')
        for p in markers:p.unlink()
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        self.assertFalse(any(p.exists()for p in markers),'Collector executed an equals-named filter')
        out=Path(self.temp.name)/'equals-cli.json'
        result=subprocess.run([sys.executable,'-S','-B',str(TOOL),'--root',str(self.root),'--output',str(out)],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(out.read_bytes())['complete'])
        self.assertFalse(any(p.exists()for p in markers),'CLI executed an equals-named filter')
        # A failed filter-key census must refuse status, without an unsafe fallback.
        status_marker=Path(self.temp.name)/'STATUS-RAN';wrapper=Path(self.temp.name)/'git-config-failure'
        native=shutil.which('git')
        wrapper.write_text('#!/usr/bin/env python3\nimport os,pathlib,sys\na=sys.argv[1:]\nif "config" in a and "--name-only" in a:sys.exit(7)\nif "status" in a:pathlib.Path('+repr(str(status_marker))+').write_text("unsafe")\nos.execv('+repr(native)+',['+repr(native)+',*a])\n')
        wrapper.chmod(0o700)
        rejected=self.census(git=str(wrapper));self.assertFalse(rejected['complete'])
        self.assertTrue(any(e['kind']=='unsafe_configuration'for e in rejected['errors']))
        self.assertFalse(status_marker.exists());self.assertFalse(any(p.exists()for p in markers))
        # Simulate a Git implementation that ignores the separate config transport.
        ignored=Path(self.temp.name)/'git-ignores-config-environment'
        ignored.write_text('#!/usr/bin/env python3\nimport os,pathlib,sys\na=sys.argv[1:]\nif "status" in a:pathlib.Path('+repr(str(status_marker))+').write_text("unsafe")\nfor key in list(os.environ):\n if key=="GIT_CONFIG_COUNT" or key.startswith(("GIT_CONFIG_KEY_","GIT_CONFIG_VALUE_")):os.environ.pop(key)\nos.execv('+repr(native)+',['+repr(native)+',*a])\n')
        ignored.chmod(0o700)
        unsupported=self.census(git=str(ignored));self.assertFalse(unsupported['complete'])
        self.assertTrue(any(e['kind']=='unsafe_configuration'for e in unsupported['errors']))
        self.assertFalse(status_marker.exists());self.assertFalse(any(p.exists()for p in markers))

    def test_ordinary_head_objects_assets_do_not_hide_nested_repositories(self):
        outer=self.repo('outer');assets=outer/'assets';assets.mkdir()
        (assets/'HEAD').write_text('ordinary asset, not Git metadata\n');(assets/'objects').mkdir()
        nested=self.repo('outer/assets/nested');deep=self.repo('outer/assets/objects/deep')
        alias=self.root/'objects-alias';alias.symlink_to(assets/'objects',target_is_directory=True)
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        self.assertTrue({str(outer),str(nested),str(deep)}.issubset(d['discovered_roots']))
        self.assertNotIn(str(assets),d['discovered_roots'])
        self.assertTrue({str(nested),str(deep)}.issubset({w['path']for w in d['worktrees']}))
        self.assertNotIn('not_followed',next(s for s in d['symlinks']if s['path']==str(alias)))
        out=Path(self.temp.name)/'assets-cli.json'
        result=subprocess.run([sys.executable,'-S','-B',str(TOOL),'--root',str(self.root),'--output',str(out)],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue({str(nested),str(deep)}.issubset(json.loads(out.read_bytes())['discovered_roots']))

    def test_remote_redaction_preserves_all_filesystem_identity_fields(self):
        self.root=self.root/'root?token=root-value';self.root.mkdir()
        repo=self.repo('repo?token=first');second=self.repo('repo?token=second')
        names=['untracked?token=first','untracked?token=second','line\nname\t'+os.fsdecode(b'\xff')]
        for name in names:(repo/name).write_text('fixture\n')
        self.git(repo,'remote','add','origin','https://private-user:private-password@example.invalid/repo?token=remote-secret')
        linked=self.root/'linked?token=worktree';self.git(repo,'worktree','add','-qb','linked',str(linked))
        alias=self.root/'alias?token=link';alias.symlink_to(repo,target_is_directory=True)
        d=self.census();self.assertTrue(d['complete'],d['errors'])
        self.assertEqual(d['root'],str(self.root))
        self.assertTrue({str(repo),str(second),str(linked)}.issubset(d['discovered_roots']))
        work=next(w for w in d['worktrees']if w['path']==str(repo))
        self.assertEqual({r['path']for r in work['status']},set(names))
        self.assertEqual(work['git_dir'],str(repo/'.git'));self.assertEqual(work['common_store'],str(repo/'.git'))
        store=next(s for s in d['stores']if s['path']==str(repo/'.git'))
        self.assertTrue({str(repo),str(linked)}.issubset({r['worktree']for r in store['registered_worktrees']}))
        link=next(s for s in d['symlinks']if s['path']==str(alias));self.assertEqual(link['target'],str(repo))
        raw=json.dumps(d);self.assertNotIn('private-password',raw);self.assertNotIn('remote-secret',raw)
        self.assertNotIn('private-user',raw);self.assertIn('REDACTED',raw)
        out=Path(self.temp.name)/'paths-cli.json'
        result=subprocess.run([sys.executable,'-S','-B',str(TOOL),'--root',str(self.root),'--output',str(out)],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        cli=json.loads(out.read_bytes());self.assertEqual(cli['root'],str(self.root))
        self.assertEqual(next(w for w in cli['worktrees']if w['path']==str(repo))['status'],work['status'])
        bad=self.root/'bad?token=error-secret';bad.mkdir();(bad/'.git').write_text('invalid\n')
        external=Path(self.temp.name)/'outside?token=external-value';external.mkdir()
        (self.root/'external?token=link').symlink_to(external,target_is_directory=True)
        incomplete=self.census();self.assertFalse(incomplete['complete'])
        error=next(e for e in incomplete['errors']if e['path']==str(bad))
        self.assertNotIn('error-secret',error['stderr']);self.assertIn('REDACTED',error['stderr'])
        self.assertEqual(next(e for e in incomplete['errors']if e['kind']=='external_directory_symlink')['target'],str(external))


if __name__=='__main__':unittest.main()
