import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

// The museum-conditionals workflow runs only when a pull request touches its
// path filter. These checks keep that filter in step with the files its
// conditional suites and replay tool actually load.
const root=path.resolve(fileURLToPath(new URL('..',import.meta.url)));
const workflowPath='.github/workflows/museum-conditionals.yml';
const workflow=fs.readFileSync(path.join(root,workflowPath),'utf8');

// The filter as it stood before tests/fixtures/museum_fixture.mjs was added.
const filterBeforeFixture=['docs/site/**','tools/museum_conditionals.mjs','tests/test_museum*.mjs','tests/fixtures/cumulative_transfer_source.txt','.github/workflows/museum-conditionals.yml'];

function pullRequestPaths(text){
  const lines=text.split('\n'),start=lines.indexOf('  pull_request:');
  assert.notEqual(start,-1,'Missing pull_request trigger');
  assert.equal(lines[start+1],'    paths:','pull_request must start with its paths filter');
  const paths=[];
  for(const line of lines.slice(start+2)){
    const m=/^      - '([^']+)'$/.exec(line);
    if(!m)break;
    paths.push(m[1]);
  }
  return paths;
}

// GitHub filter patterns: ** crosses directories, * stays within one segment.
// Refuse any other pattern character so a later edit cannot be misread here.
function matcher(patterns){
  const regexes=patterns.map(pattern=>{
    assert.doesNotMatch(pattern,/[?+[\]!]/,`Unsupported filter character in ${pattern}`);
    const body=pattern.split('**').map(part=>part.split('*').map(s=>s.replace(/[.^$|(){}\\]/g,'\\$&')).join('[^/]*')).join('.*');
    return new RegExp(`^${body}$`);
  });
  return file=>regexes.some(regex=>regex.test(file));
}

// Relative imports and import.meta.url reads, followed transitively from the
// conditional and identity suites and the replay tool, which only this job
// runs. The museum frontend and geometry suites and the shop frontend suite it
// also runs are run by public-shop on every pull request to main, so they are
// not roots here.
function localClosure(starts){
  const seen=new Set(),stack=[...starts];
  const refs=[/\bfrom\s*['"](\.{1,2}\/[^'"]+)['"]/g,/\bimport\s*['"](\.{1,2}\/[^'"]+)['"]/g,/\bimport\s*\(\s*['"](\.{1,2}\/[^'"]+)['"]/g,/new URL\(\s*['"](\.{1,2}\/[^'"]+)['"]\s*,\s*import\.meta\.url\s*\)/g];
  while(stack.length){
    const file=stack.pop();
    if(seen.has(file))continue;
    seen.add(file);
    if(!/\.m?js$/.test(file))continue;
    const text=fs.readFileSync(path.join(root,file),'utf8');
    for(const pattern of refs)for(const [,ref] of text.matchAll(pattern)){
      const target=path.posix.normalize(path.posix.join(path.posix.dirname(file),ref.split('?')[0]));
      if(target.endsWith('/')||!fs.statSync(path.join(root,target),{throwIfNoEntry:false})?.isFile())continue;
      stack.push(target);
    }
  }
  return [...seen].sort();
}

const filter=pullRequestPaths(workflow);
const matches=matcher(filter);

test('the pull_request filter keeps every earlier path and adds the shared museum fixture',()=>{
  assert.deepEqual(filter,[...filterBeforeFixture.slice(0,4),'tests/fixtures/museum_fixture.mjs',filterBeforeFixture[4]]);
});

test('a fixture-only change was missed before and now runs the conditional suites',()=>{
  assert.equal(matcher(filterBeforeFixture)('tests/fixtures/museum_fixture.mjs'),false);
  assert.equal(matches('tests/fixtures/museum_fixture.mjs'),true);
});

test('existing matching paths stay matched and unrelated paths stay excluded',()=>{
  for(const file of ['docs/site/museum.mjs','docs/site/dependency-source/GRAPH.json','tools/museum_conditionals.mjs','tests/test_museum_frontend.mjs','tests/fixtures/cumulative_transfer_source.txt',workflowPath])assert.equal(matches(file),true,file);
  for(const file of ['README.md','tools/museum_check.py','tests/test_reader_links.py','tests/fixtures/other_fixture.mjs','tests/nested/test_museum_extra.mjs','tests/test_museum_extra/nested.mjs','docs/site.md'])assert.equal(matches(file),false,file);
});

test('every local file the conditional suites and replay tool load triggers the workflow',()=>{
  const roots=fs.readdirSync(path.join(root,'tests')).filter(name=>/^test_museum_(conditionals.*|identity)\.mjs$/.test(name)).map(name=>`tests/${name}`);
  assert.ok(roots.includes('tests/test_museum_conditionals_startup.mjs')&&roots.includes('tests/test_museum_identity.mjs'));
  const closure=localClosure([...roots,'tools/museum_conditionals.mjs']);
  assert.ok(closure.includes('tests/fixtures/museum_fixture.mjs'),'The closure should reach the shared fixture');
  assert.ok(closure.includes('docs/site/conditionals.mjs'),'The closure should reach the site module');
  for(const file of ['tests/fixtures/cumulative_transfer_source.txt','docs/site/museum.html','docs/site/museum.mjs'])assert.ok(closure.includes(file),`The closure should reach ${file} (import.meta.url read or dynamic import)`);
  assert.deepEqual(closure.filter(file=>!matches(file)),[]);
});
