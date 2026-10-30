#!/usr/bin/env python3
"""Federation drift crawler (CBZ-0 addendum: dynamic federation drift crawler).

Discovers the federation at run time from a seed repository and blocks on drift between
records that claim the same identity. Nothing about the topology is hardcoded: targets
come from the seed's `.gitmodules`, gitlinks, `pins.json` and
`workspace/repositories.json`, from any repository registry those name under
`canonical_inputs` (artifact `repository` keys are resolved through the registry's own
`repositories` map), and from every `lake-manifest.json` dependency whose URL resolves to
a repository of the federation's owners. This is a breadth-first fixed point: the records
of every reached repository are read in turn until nothing new appears. Lake dependencies
outside the owners named by those records (for example mathlib) are counted as external
edges and not crawled. A rename, addition or removal in those records changes the crawl
with no code change.

Findings. BLOCK means two records that claim the same identity disagree, or something the
crawl must read is missing or malformed; the exit status is then 1. INFO means a record
differs from a moving tip (a stale pin, an artifact that has evolved since its recorded
commit, different Lean toolchains); it never fails the run.

  PIN-SOURCES     .gitmodules, gitlink, pins.json and repositories.json disagree   BLOCK
  PIN-MISSING     a pinned commit is not present in its repository                 BLOCK
  PIN-OFFLINE     a pinned commit is not an ancestor of the repository's default    BLOCK
  PIN-STALE       a pinned commit is behind the default tip                         INFO
  UNRESOLVED      a discovered repository cannot be read (opt-in ones are INFO)     BLOCK
  REGISTRY-BYTES  a registry artifact's bytes or SHA-256 differ at its commit       BLOCK
  REGISTRY-EVOLVED the artifact differs (or is gone) at the crawled ref             INFO
  INPUT-OBSERVED  a canonical input's observed blob differs at the crawled ref      INFO
  INPUT-MISSING   a declared canonical input is absent at its selected ref         BLOCK
  LEAN-DRIFT      one fully qualified Lean declaration has different interfaces     BLOCK
                  in two crawled locations (see lean_decls for the exact domain)
  LEAN-SYNTAX     a Lean file could not be lexed or its scopes do not balance       BLOCK
  LAKE-DEP        a lake dependency on a crawled repository pins another rev        INFO
  PIN-DIVERGENT   two reached repositories pin one repository differently           INFO
  FETCH-FAILED    --fetch could not refresh a repository, so its tips are not current BLOCK
  TOOLCHAIN       Lean packages that share declarations use different toolchains   INFO
  MALFORMED       a manifest exists but cannot be parsed                            BLOCK

Repositories are read from local git object stores only, found by matching each
candidate directory's `origin` URL. `--fetch` refreshes each repository from origin
first, so a default tip is the remote's current one (failure is FETCH-FAILED), and fetches
missing commits; `--clone-into DIR` clones a discovered repository that no search directory
holds (so a CI job needs only the seed). The crawl reads each repository at one ref: the seed at `--seed-ref`, every
pinned repository at its pin (`--at pins`, the default) or at its default tip
(`--at tips`), and `--override NAME=REV` replaces one repository's ref (to test a
candidate head against the rest of the federation).

Standard library only. Exit: 0 no BLOCK, 1 BLOCK findings, 2 usage or internal error.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_decls  # noqa: E402

BLOCK, INFO = 'BLOCK', 'INFO'


class CrawlError(RuntimeError):
    pass


def repo_key(url_or_name):
    """Normalize a GitHub-style URL or owner/name to a lowercase 'owner/name' key."""
    s = url_or_name.strip()
    s = re.sub(r'\.git/?$', '', s).rstrip('/')
    m = re.search(r'(?:^|[:/])([^/:@\s]+)/([^/:@\s]+)$', s)
    if m:
        return ('%s/%s' % (m.group(1), m.group(2))).lower()
    raise CrawlError('cannot read a repository identity from %r' % url_or_name)


def git(gitdir, *args, check=True, data=None):
    r = subprocess.run(['git', '-C', str(gitdir), *args], input=data, capture_output=True)
    if check and r.returncode:
        raise CrawlError('git %s failed in %s: %s' % (' '.join(args), gitdir, r.stderr.decode(errors='replace').strip()))
    return r


class Repo:
    def __init__(self, key, gitdir, fetch=False):
        self.key, self.gitdir, self.fetch = key, gitdir, fetch
        self._blob_cache = {}
        self.fetch_failed = None
        self.shallow = git(gitdir, 'rev-parse', '--is-shallow-repository').stdout.decode().strip() == 'true'
        if fetch:
            # refresh, so that a default tip read later is the remote's current one, not a cached ref
            args = ['fetch', '-q'] + (['--unshallow'] if self.shallow else []) + ['origin']
            r = git(gitdir, *args, check=False)
            if r.returncode:
                self.fetch_failed = r.stderr.decode(errors='replace').strip()[-300:] or 'git fetch failed'
            else:
                git(gitdir, 'remote', 'set-head', 'origin', '--auto', check=False)
            self.shallow = git(gitdir, 'rev-parse', '--is-shallow-repository').stdout.decode().strip() == 'true'

    def has_commit(self, rev):
        if git(self.gitdir, 'cat-file', '-e', rev + '^{commit}', check=False).returncode == 0:
            return True
        if self.fetch and re.fullmatch(r'[0-9a-f]{40}', rev):
            git(self.gitdir, 'fetch', '-q', 'origin', rev, check=False)
            return git(self.gitdir, 'cat-file', '-e', rev + '^{commit}', check=False).returncode == 0
        return False

    def resolve(self, rev):
        return git(self.gitdir, 'rev-parse', '--verify', '-q', rev + '^{commit}').stdout.decode().strip()

    def default_tip(self):
        r = git(self.gitdir, 'symbolic-ref', '-q', 'refs/remotes/origin/HEAD', check=False)
        ref = r.stdout.decode().strip() if r.returncode == 0 else ''
        for cand in ([ref] if ref else []) + ['refs/remotes/origin/main', 'refs/remotes/origin/master', 'HEAD']:
            r = git(self.gitdir, 'rev-parse', '--verify', '-q', cand + '^{commit}', check=False)
            if r.returncode == 0:
                return r.stdout.decode().strip()
        raise CrawlError('no default tip in %s' % self.gitdir)

    def where(self):
        if self.fetch:
            return ' (also not fetchable from origin)'
        return ' (local object store%s; rerun with --fetch to consult origin)' % (', shallow clone' if self.shallow else '')

    def is_ancestor(self, a, b):
        return git(self.gitdir, 'merge-base', '--is-ancestor', a, b, check=False).returncode == 0

    def tree(self, rev):
        """{path: (mode, type, sha)} for every entry at rev, recursively."""
        out = git(self.gitdir, 'ls-tree', '-r', '-z', '--full-tree', rev).stdout.decode()
        entries = {}
        for rec in out.split('\0'):
            if rec:
                meta, path = rec.split('\t', 1)
                mode, typ, sha = meta.split()
                entries[path] = (mode, typ, sha)
        return entries

    def blob(self, sha):
        if sha not in self._blob_cache:
            self._blob_cache[sha] = git(self.gitdir, 'cat-file', 'blob', sha).stdout
        return self._blob_cache[sha]

    def read(self, rev, path):
        r = git(self.gitdir, 'cat-file', 'blob', '%s:%s' % (rev, path), check=False)
        return r.stdout if r.returncode == 0 else None


def find_local_repos(search_dirs):
    """{key: gitdir} for git repositories directly inside each search directory."""
    found = {}
    for d in search_dirs:
        for child in sorted(pathlib.Path(d).iterdir()):
            if not child.is_dir():
                continue
            if git(child, 'rev-parse', '--git-dir', check=False).returncode:
                continue
            if git(child, 'rev-parse', '--show-toplevel', check=False).stdout.decode().strip() != str(child.resolve()):
                continue
            r = git(child, 'remote', 'get-url', 'origin', check=False)
            if r.returncode == 0:
                try:
                    found.setdefault(repo_key(r.stdout.decode()), child)
                except CrawlError:
                    pass
    return found


def parse_gitmodules(text):
    mods, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in '#;':
            continue
        m = re.fullmatch(r'\[submodule\s+"([^"]+)"\]', line)
        if m:
            cur = mods.setdefault(m.group(1), {})
            continue
        m = re.fullmatch(r'([A-Za-z]+)\s*=\s*(.*)', line)
        if m and cur is not None:
            cur[m.group(1).lower()] = m.group(2).strip()
        else:
            raise CrawlError('unparsed .gitmodules line: %r' % raw)
    return mods


class Crawl:
    def __init__(self, local, at='pins', overrides=None, fetch=False, clone_into=None):
        self.local, self.at, self.overrides, self.fetch = local, at, dict(overrides or {}), fetch
        self.clone_into, self.urls = clone_into, {}
        self.findings, self.repos, self.refs = [], {}, {}
        self.pins, self.optional, self.edges = {}, set(), []
        self.stats = collections.Counter()
        self.discovered, self.canonical_inputs, self.pin_records = set(), [], []
        self.owners, self.lake_records = set(), []

    def add(self, level, code, msg, **where):
        self.findings.append(dict(where, level=level, code=code, message=msg))

    def open(self, key, why):
        if key in self.repos:
            return self.repos[key]
        gitdir = self.local.get(key)
        if gitdir is None and self.clone_into:
            gitdir = self.clone(key)
        if gitdir is None:
            self.repos[key] = None
            self.add(INFO if key in self.optional else BLOCK, 'UNRESOLVED',
                     'repository %s (%s) is not available locally' % (key, why), repo=key)
            return None
        self.repos[key] = Repo(key, gitdir, self.fetch)
        if self.repos[key].fetch_failed:
            self.add(BLOCK, 'FETCH-FAILED', 'could not refresh %s from origin, so its tips are not current: %s' % (
                key, self.repos[key].fetch_failed), repo=key)
        return self.repos[key]

    def clone(self, key):
        """Clone a discovered repository that no search directory holds; None if that fails."""
        url = self.urls.get(key) or 'https://github.com/%s.git' % key
        dest = pathlib.Path(self.clone_into) / key.replace('/', '__')
        if not dest.exists():
            r = subprocess.run(['git', 'clone', '-q', '--no-checkout', url, str(dest)], capture_output=True)
            if r.returncode:
                return None
        try:
            if repo_key(git(dest, 'remote', 'get-url', 'origin').stdout.decode()) != key:
                return None
        except CrawlError:
            return None
        self.local[key] = dest
        return dest

    def ref_for(self, key, repo):
        if key in self.refs:
            return self.refs[key]
        if key in self.overrides:
            rev = self.overrides[key]
            if not repo.has_commit(rev):
                raise CrawlError('override %s=%s does not name a commit' % (key, rev))
            ref = repo.resolve(rev)
        elif self.at == 'pins' and key in self.pins and repo.has_commit(self.pins[key]):
            ref = self.pins[key]
        else:
            ref = repo.default_tip()
        self.refs[key] = ref
        return ref

    def json_at(self, repo, ref, path):
        data = repo.read(ref, path)
        if data is None:
            return None
        try:
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as exc:
            self.add(BLOCK, 'MALFORMED', '%s is not valid JSON: %s' % (path, exc), repo=repo.key, ref=ref, path=path)
            return None

    # ------------------------------------------------------------------ topology
    def seed(self, seed_key, seed_ref):
        repo = self.open(seed_key, 'seed')
        if repo is None:
            raise CrawlError('seed repository %s is not available locally' % seed_key)
        if seed_ref:
            self.overrides.setdefault(seed_key, seed_ref)
        self.seed_key, self.seed_ref = seed_key, self.ref_for(seed_key, repo)
        self.owners.add(seed_key.split('/')[0])

    def close_graph(self):
        """Breadth-first fixed point over every record that names a repository.

        Each reached repository's own `.gitmodules`, gitlinks, `pins.json`,
        `workspace/repositories.json`, canonical inputs and lake manifests are read, and
        what they name is reached in turn, until nothing new appears. A repository is read
        at one ref: an override, else the first pin met in breadth-first order (`--at pins`),
        else its default tip. Lake dependencies outside the federation's owners (those of
        the seed and of every repository its records list) are recorded as external edges
        and not crawled.
        """
        queue, done = collections.deque([(self.seed_key, 'seed')]), set()
        while queue:
            key, why = queue.popleft()
            if key in done:
                continue
            done.add(key)
            repo = self.open(key, why)
            if repo is None:
                continue
            for new in self.expand(key, repo):
                queue.append((new, 'listed by %s' % key))
            for new in self.lake_targets(key, repo):
                queue.append((new, 'lake dependency of %s' % key))
            for new in self.registry_targets(key):
                queue.append((new, 'registry artifact via %s' % key))
        for key, ref, lm, pkg, dep in self.lake_records:
            drepo = self.repos.get(dep)
            if drepo is None:
                continue
            if not drepo.has_commit(pkg['rev']):
                self.add(BLOCK, 'PIN-MISSING', '%s:%s pins %s at %s, which is not in that repository%s' % (
                    key, lm, dep, pkg['rev'], drepo.where()), repo=key, path=lm, target=dep, pin=pkg['rev'])
            elif pkg['rev'] != self.ref_for(dep, drepo):
                self.add(INFO, 'LAKE-DEP', '%s:%s pins %s at %s; crawl reads it at %s' % (
                    key, lm, dep, pkg['rev'][:12], self.ref_for(dep, drepo)[:12]), repo=key, path=lm)

    def lake_targets(self, key, repo):
        ref = self.ref_for(key, repo)
        tree = repo.tree(ref)
        out = []
        for lm in sorted(p for p in tree if p.rsplit('/', 1)[-1] == 'lake-manifest.json' and '/.lake/' not in '/' + p):
            doc = self.json_at(repo, ref, lm)
            for pkg in (doc or {}).get('packages', []) if isinstance(doc, dict) else []:
                if not (isinstance(pkg, dict) and pkg.get('type') == 'git' and pkg.get('url') and pkg.get('rev')):
                    continue
                try:
                    dep = repo_key(pkg['url'])
                except CrawlError:
                    continue
                if dep.split('/')[0] not in self.owners:
                    self.stats['lake_external_edges'] += 1
                    continue
                self.urls.setdefault(dep, pkg['url'])
                self.edges.append((key, dep, 'lake'))
                self.lake_records.append((key, ref, lm, pkg, dep))
                out.append(dep)
        return out

    def registry_targets(self, key):
        """Repositories named by artifacts of the registries that `key` declares."""
        out = []
        for owner, label, spec in self.canonical_inputs:
            if owner != key or not isinstance(spec, dict) or 'repository' not in spec or 'path' not in spec:
                continue
            try:
                rkey = repo_key(spec['repository'])
            except CrawlError:
                continue
            out.append(rkey)
            rrepo = self.open(rkey, 'canonical input %s of %s' % (label, key))
            if rrepo is None:
                continue
            ref = spec.get('ref') or self.ref_for(rkey, rrepo)
            if not rrepo.has_commit(ref):
                continue
            doc = self.json_at(rrepo, ref, spec['path']) if spec['path'] in rrepo.tree(ref) else None
            if isinstance(doc, dict) and isinstance(doc.get('repositories'), dict) and isinstance(doc.get('artifacts'), list):
                used = {a.get('repository') for a in doc['artifacts'] if isinstance(a, dict)}
                for name, ent in sorted(doc['repositories'].items()):
                    if name in used and isinstance(ent, dict) and ent.get('full_name'):
                        try:
                            out.append(repo_key(ent['full_name']))
                        except CrawlError:
                            pass
        return out

    def expand(self, owner, repo):
        """Read one repository's own pin and listing records; return the repositories they name."""
        seed_key = owner
        ref = self.ref_for(owner, repo)
        tree = repo.tree(ref)
        records = collections.defaultdict(dict)   # key -> {source: value}

        gm = repo.read(ref, '.gitmodules')
        if gm is not None:
            try:
                mods = parse_gitmodules(gm.decode('utf-8'))
            except (CrawlError, UnicodeDecodeError) as exc:
                self.add(BLOCK, 'MALFORMED', '.gitmodules: %s' % exc, repo=seed_key, ref=ref, path='.gitmodules')
                mods = {}
            for name, m in sorted(mods.items()):
                if 'url' not in m or 'path' not in m:
                    self.add(BLOCK, 'MALFORMED', 'submodule %s lacks url or path' % name, repo=seed_key, path='.gitmodules')
                    continue
                key = repo_key(m['url'])
                self.urls.setdefault(key, m['url'])
                records[key]['gitmodules'] = m['path']
                if m.get('update') == 'none':
                    self.optional.add(key)
                ent = tree.get(m['path'])
                if ent is None or ent[1] != 'commit':
                    self.add(BLOCK, 'PIN-SOURCES', 'submodule %s has no gitlink at %s' % (name, m['path']),
                             repo=seed_key, ref=ref, path=m['path'])
                else:
                    records[key]['gitlink'] = ent[2]
        for path, (mode, typ, sha) in sorted(tree.items()):
            if typ == 'commit' and not any(r.get('gitmodules') == path for r in records.values()):
                self.add(BLOCK, 'PIN-SOURCES', 'gitlink %s has no .gitmodules entry' % path, repo=seed_key, ref=ref, path=path)

        pins = self.json_at(repo, ref, 'pins.json')
        if pins is not None:
            if not isinstance(pins, dict) or not isinstance(pins.get('repos'), list):
                self.add(BLOCK, 'MALFORMED', 'pins.json has no repos list', repo=seed_key, path='pins.json')
            else:
                for ent in pins['repos']:
                    try:
                        key = repo_key(ent['url'])
                        sha, path = ent['pinned_sha'], ent['path']
                        self.urls.setdefault(key, ent['url'])
                    except (KeyError, TypeError, CrawlError) as exc:
                        self.add(BLOCK, 'MALFORMED', 'pins.json entry %r: %s' % (ent, exc), repo=seed_key, path='pins.json')
                        continue
                    if 'pins' in records[key]:
                        self.add(BLOCK, 'PIN-SOURCES', 'pins.json lists %s twice' % key, repo=seed_key, path='pins.json')
                    records[key]['pins'] = sha
                    records[key]['pins_path'] = path
                    if ent.get('opt_in'):
                        self.optional.add(key)

        reg = self.json_at(repo, ref, 'workspace/repositories.json')
        inputs = {}
        if reg is not None:
            for ent in reg.get('repositories', []) if isinstance(reg, dict) else []:
                try:
                    records[repo_key(ent['full_name'])]['listed'] = True
                except (KeyError, TypeError, CrawlError) as exc:
                    self.add(BLOCK, 'MALFORMED', 'repositories.json entry %r: %s' % (ent, exc), repo=seed_key)
            inputs = reg.get('canonical_inputs', {}) if isinstance(reg, dict) else {}

        named = []
        for key, rec in sorted(records.items()):
            if key == seed_key:
                continue
            named.append(key)
            self.discovered.add(key)
            self.owners.add(key.split('/')[0])
            srcs = [s for s in ('gitmodules', 'pins', 'listed') if s in rec]
            have = set(srcs)
            if gm is not None and pins is not None and ('gitmodules' in have) != ('pins' in have):
                self.add(BLOCK, 'PIN-SOURCES', '%s is in %s but not in %s' % (
                    key, 'gitmodules' if 'gitmodules' in have else 'pins.json',
                    'pins.json' if 'gitmodules' in have else '.gitmodules'), repo=seed_key, target=key)
            if reg is not None and (have & {'gitmodules', 'pins'}) and 'listed' not in have:
                self.add(BLOCK, 'PIN-SOURCES', '%s is pinned but not listed in workspace/repositories.json' % key,
                         repo=seed_key, target=key)
            if 'gitlink' in rec and 'pins' in rec and rec['gitlink'] != rec['pins']:
                self.add(BLOCK, 'PIN-SOURCES', '%s: gitlink %s but pins.json %s' % (key, rec['gitlink'][:12], rec['pins'][:12]),
                         repo=seed_key, target=key)
            if 'gitmodules' in rec and 'pins_path' in rec and rec['gitmodules'] != rec['pins_path']:
                self.add(BLOCK, 'PIN-SOURCES', '%s: .gitmodules path %s but pins.json path %s' % (key, rec['gitmodules'], rec['pins_path']),
                         repo=seed_key, target=key)
            pin = rec.get('pins') or rec.get('gitlink')
            if pin:
                self.pin_records.append((seed_key, key, pin))
                first = self.pins.setdefault(key, pin)
                if first != pin:
                    self.add(INFO, 'PIN-DIVERGENT', '%s pins %s at %s; an earlier record pins %s, which the crawl reads' % (
                        seed_key, key, pin[:12], first[:12]), repo=seed_key, target=key, pin=pin)
            self.edges.append((seed_key, key, '+'.join(srcs)))
        for label, spec in sorted(inputs.items()) if isinstance(inputs, dict) else []:
            self.canonical_inputs.append((seed_key, label, spec))
        return named

    def check_pins(self):
        for parent, key, pin in sorted(set(self.pin_records)):
            repo = self.repos.get(key)
            if repo is None:
                continue
            if not repo.has_commit(pin):
                self.add(BLOCK, 'PIN-MISSING', 'pinned commit %s is not in %s%s' % (pin, key, repo.where()), repo=key, pin=pin)
                continue
            tip = repo.default_tip()
            if pin == tip:
                continue
            if repo.shallow:
                self.add(BLOCK, 'PIN-OFFLINE', 'cannot decide whether pin %s is on the default line: %s is a shallow clone '
                         '(rerun with --fetch)' % (pin[:12], repo.gitdir), repo=key, pin=pin, tip=tip)
            elif not repo.is_ancestor(pin, tip):
                self.add(BLOCK, 'PIN-OFFLINE', 'pinned commit %s is not an ancestor of the default tip %s' % (pin[:12], tip[:12]),
                         repo=key, pin=pin, tip=tip)
            else:
                n = git(repo.gitdir, 'rev-list', '--count', '%s..%s' % (pin, tip)).stdout.decode().strip()
                self.add(INFO, 'PIN-STALE', 'pin %s is %s commits behind the default tip %s' % (pin[:12], n, tip[:12]),
                         repo=key, pin=pin, tip=tip)

    # ------------------------------------------------------------------ registries
    def check_registries(self):
        for owner, label, spec in self.canonical_inputs:
            if not isinstance(spec, dict) or 'repository' not in spec or 'path' not in spec:
                continue
            key = repo_key(spec['repository'])
            repo = self.open(key, 'canonical input %s' % label)
            if repo is None:
                continue
            ref = spec.get('ref')
            if ref and not repo.has_commit(ref):
                self.add(BLOCK, 'PIN-MISSING', 'canonical input %s ref %s is not in %s%s' % (label, ref, key, repo.where()), repo=key)
                continue
            ref = ref or self.ref_for(key, repo)
            ent = repo.tree(ref).get(spec['path'])
            if spec.get('observed_blob') and (ent is None or ent[2] != spec['observed_blob']):
                self.add(INFO, 'INPUT-OBSERVED', 'canonical input %s: %s at %s is %s, recorded %s' % (
                    label, spec['path'], ref[:12], ent[2][:12] if ent else 'absent', spec['observed_blob'][:12]),
                    repo=key, ref=ref, path=spec['path'])
            if ent is None:
                self.add(BLOCK, 'INPUT-MISSING', 'canonical input %s is absent at %s@%s:%s' % (
                    label, key, ref, spec['path']), label=label, repo=key, ref=ref, path=spec['path'])
                continue
            doc = self.json_at(repo, ref, spec['path'])
            if isinstance(doc, dict) and isinstance(doc.get('artifacts'), list) and isinstance(doc.get('repositories'), dict):
                self.check_artifacts(repo, ref, spec['path'], doc)

    def check_artifacts(self, reg_repo, reg_ref, reg_path, doc):
        names = {}
        for name, ent in doc['repositories'].items():
            try:
                names[name] = repo_key(ent['full_name'])
            except (KeyError, TypeError, CrawlError) as exc:
                self.add(BLOCK, 'MALFORMED', 'registry repository %r: %s' % (name, exc), repo=reg_repo.key, path=reg_path)
        trees = {}
        for art in doc['artifacts']:
            where = dict(repo=reg_repo.key, path=reg_path, artifact=art.get('key'))
            try:
                key = names[art['repository']]
                commit, path, size, digest = art['commit'], art['path'], int(art['bytes']), art['sha256']
            except (KeyError, TypeError, ValueError) as exc:
                self.add(BLOCK, 'MALFORMED', 'artifact %r: %s' % (art.get('key'), exc), **where)
                continue
            repo = self.open(key, 'registry artifact %s' % art.get('key'))
            if repo is None:
                continue
            self.stats['registry_artifacts'] += 1
            if not repo.has_commit(commit):
                self.add(BLOCK, 'PIN-MISSING', 'artifact %s: commit %s is not in %s%s' % (art.get('key'), commit, key, repo.where()), **where)
                continue
            if (key, commit) not in trees:
                trees[(key, commit)] = repo.tree(commit)
            ent = trees[(key, commit)].get(path)
            data = repo.blob(ent[2]) if ent and ent[1] == 'blob' else None
            if data is None:
                self.add(BLOCK, 'REGISTRY-BYTES', 'artifact %s: %s is absent at %s' % (art.get('key'), path, commit[:12]), **where)
                continue
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                self.add(BLOCK, 'REGISTRY-BYTES', 'artifact %s: %s at %s has %d bytes / sha256 %s, recorded %d / %s' % (
                    art.get('key'), path, commit[:12], len(data), hashlib.sha256(data).hexdigest()[:16], size, digest[:16]), **where)
                continue
            self.stats['registry_artifacts_verified'] += 1
            ref = self.ref_for(key, repo)
            if (key, ref) not in trees:
                trees[(key, ref)] = repo.tree(ref)
            now = trees[(key, ref)].get(path)
            if now is None or now[2] != ent[2]:
                self.stats['registry_artifacts_evolved'] += 1
                self.add(INFO, 'REGISTRY-EVOLVED', 'artifact %s: %s %s at the crawled ref %s' % (
                    art.get('key'), path, 'is absent' if now is None else 'differs', ref[:12]), **where)

    # ------------------------------------------------------------------ Lean
    def crawled(self):
        for key in sorted(k for k, r in self.repos.items() if r is not None):
            yield key, self.repos[key], self.ref_for(key, self.repos[key])

    def check_lean(self):
        decls = collections.defaultdict(list)
        packages = {}
        for key, repo, ref in list(self.crawled()):
            tree = repo.tree(ref)
            roots = sorted({p.rsplit('/', 1)[0] if '/' in p else '' for p in tree
                            if p.rsplit('/', 1)[-1] in ('lakefile.lean', 'lakefile.toml')})
            for root in roots:
                tc = repo.read(ref, (root + '/' if root else '') + 'lean-toolchain')
                packages[(key, root)] = tc.decode(errors='replace').strip() if tc else None
            for path, (mode, typ, sha) in sorted(tree.items()):
                if typ != 'blob' or not path.endswith('.lean') or '/.lake/' in '/' + path:
                    continue
                pkg = max((r for r in roots if r == '' or path.startswith(r + '/')), key=len, default=None)
                self.stats['lean_files'] += 1
                try:
                    found = lean_decls.declarations(repo.blob(sha).decode('utf-8'), unit='%s:%s' % (key, path))
                except (lean_decls.LeanSyntaxError, UnicodeDecodeError) as exc:
                    self.add(BLOCK, 'LEAN-SYNTAX', str(exc), repo=key, ref=ref, path=path)
                    continue
                for d in found:
                    self.stats['lean_declarations'] += 1
                    if d['private']:
                        continue
                    decls[d['name']].append({'repo': key, 'ref': ref, 'path': path, 'line': d['line'],
                                             'package': pkg, 'kind': d['kind'], 'interface': d['interface']})
        shared = 0
        coupled = collections.defaultdict(set)
        for name, locs in sorted(decls.items()):
            if len(locs) < 2:
                continue
            shared += 1
            pk = {(l['repo'], l['package']) for l in locs if l['package'] is not None}
            for a in pk:
                coupled[a] |= pk - {a}
            variants = collections.OrderedDict()
            for l in locs:
                variants.setdefault((l['kind'], l['interface']), []).append(l)
            if len(variants) > 1:
                self.add(BLOCK, 'LEAN-DRIFT', '%s has %d different interfaces across %d locations' % (name, len(variants), len(locs)),
                         name=name, variants=[{'kind': k, 'interface': i, 'locations': ['%s@%s:%s:%d' % (
                             l['repo'], l['ref'][:12], l['path'], l['line']) for l in ls]} for (k, i), ls in variants.items()])
        self.stats['lean_shared_names'] = shared
        for (a, b) in sorted({tuple(sorted((x, y))) for x, ys in coupled.items() for y in ys}):
            if packages.get(a) != packages.get(b):
                self.add(INFO, 'TOOLCHAIN', 'packages %s:%s and %s:%s share declarations but use %s and %s' % (
                    a[0], a[1] or '.', b[0], b[1] or '.', packages.get(a), packages.get(b)))

    def report(self):
        return {
            'schema': 'cbz.drift.v1',
            'seed': {'repository': self.seed_key, 'ref': self.seed_ref},
            'mode': self.at,
            'repositories': [{'repository': k, 'ref': self.refs.get(k), 'resolved': r is not None,
                              'optional': k in self.optional, 'pin': self.pins.get(k)} for k, r in sorted(self.repos.items())],
            'edges': [{'from': a, 'to': b, 'via': v} for a, b, v in self.edges],
            'stats': dict(sorted(self.stats.items())),
            'blocking': sum(f['level'] == BLOCK for f in self.findings),
            'findings': self.findings,
        }


def run(seed, search, at='pins', seed_ref=None, overrides=None, fetch=False, clone_into=None):
    local = find_local_repos(search)
    c = Crawl(local, at=at, overrides=overrides, fetch=fetch, clone_into=clone_into)
    c.seed(repo_key(seed) if '/' in seed else _match_name(seed, local), seed_ref)
    c.close_graph()
    c.check_pins()
    c.check_registries()
    c.check_lean()
    return c.report()


def _match_name(name, local):
    hits = [k for k in local if k.split('/')[1] == name.lower()]
    if len(hits) != 1:
        raise CrawlError('seed %r matches %d local repositories' % (name, len(hits)))
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seed', required=True, help='owner/name (or a unique name) of the seed repository')
    ap.add_argument('--search', action='append', required=True, help='directory whose subdirectories are local clones')
    ap.add_argument('--seed-ref', help='ref of the seed to read (default: its default tip)')
    ap.add_argument('--at', choices=('pins', 'tips'), default='pins')
    ap.add_argument('--override', action='append', default=[], metavar='OWNER/NAME=REV')
    ap.add_argument('--fetch', action='store_true', help='git fetch commits missing locally')
    ap.add_argument('--clone-into', metavar='DIR', help='clone discovered repositories that no --search directory holds')
    ap.add_argument('--json', metavar='FILE', help='write the full report here')
    args = ap.parse_args(argv)
    try:
        ov = {}
        for o in args.override:
            k, _, v = o.partition('=')
            if not v:
                raise CrawlError('override %r is not NAME=REV' % o)
            ov[repo_key(k) if '/' in k else k.lower()] = v
        if args.clone_into:
            pathlib.Path(args.clone_into).mkdir(parents=True, exist_ok=True)
        rep = run(args.seed, args.search, at=args.at, seed_ref=args.seed_ref, overrides=ov, fetch=args.fetch,
                  clone_into=args.clone_into)
    except CrawlError as exc:
        print('drift crawler error: %s' % exc, file=sys.stderr)
        return 2
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n')
    for f in rep['findings']:
        if f['level'] == BLOCK:
            print('BLOCK %s %s' % (f['code'], f['message']))
    info = collections.Counter(f['code'] for f in rep['findings'] if f['level'] == INFO)
    print('crawled %d repositories from %s@%s (%s): %d BLOCK, INFO %s; stats %s' % (
        sum(r['resolved'] for r in rep['repositories']), rep['seed']['repository'], rep['seed']['ref'][:12],
        rep['mode'], rep['blocking'], dict(sorted(info.items())) or '{}', rep['stats']))
    return 1 if rep['blocking'] else 0


if __name__ == '__main__':
    sys.exit(main())
