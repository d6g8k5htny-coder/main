// Q4 limits lens -- FINDING probe (minor, test coverage): no negative control pins that startMuseum
// actually routes the manifest and source requests through boundedFetch with its production bounds.
// The two boundedFetch tests call boundedFetch directly with explicit tiny bounds (12 bytes, 15 ms);
// every startMuseum test serves complete in-memory Responses, which an unbounded path also accepts.
// Method: copy the cd32303 museum files into a scratch tree (the extracted tree is never modified),
// apply one mutation at a time to the COPY of museum.mjs, run the unmodified cd32303
// tests/test_museum_frontend.mjs against it, and count failures.
// No network: the tests use in-process Response stubs only.
// Usage: node p6_bound_wiring_mutants.mjs <path-to-cd32303-tree>
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
const root = process.argv[2];
const here = path.dirname(new URL(import.meta.url).pathname);
const src = fs.readFileSync(path.join(root, 'docs/site/museum.mjs'), 'utf8');

const UNBOUNDED = `async(url,fetcher=fetch)=>{const r=await fetcher(url,{credentials:'omit',redirect:'error'});return {ok:r.ok,status:r.status,bytes:r.ok?new Uint8Array(await r.arrayBuffer()):new Uint8Array()};}`;
const mutants = {
  'M0 unmodified copy (baseline)': s => s,
  'M1 manifest fetch unbounded (no 256 KiB cap, no deadline)': s => s.replace("await boundedFetch('museum.json',fetcher,256*1024)", `await (${UNBOUNDED})('museum.json',fetcher)`),
  'M2 source fetches unbounded (cachedFetch bypasses boundedFetch)': s => s.replace('cache.set(key,boundedFetch(url,fetcher))', `cache.set(key,(${UNBOUNDED})(url,fetcher))`),
  'M3 M1+M2 together': s => mutants['M2 source fetches unbounded (cachedFetch bypasses boundedFetch)'](mutants['M1 manifest fetch unbounded (no 256 KiB cap, no deadline)'](s)),
  'M4 production defaults raised to 1 GiB / 1 day': s => s.replace('maxBytes=2*1024*1024,timeoutMs=20000', 'maxBytes=1024*1024*1024,timeoutMs=86400000').replace("boundedFetch('museum.json',fetcher,256*1024)", "boundedFetch('museum.json',fetcher,1024*1024*1024)"),
};

const work = fs.mkdtempSync(path.join(here, 'mutants-'));
try {
  for (const [name, mutate] of Object.entries(mutants)) {
    const mutated = mutate(src);
    if (name[1] !== '0' && mutated === src) { console.log(`${name}: MUTATION DID NOT APPLY`); continue; }
    const dir = fs.mkdtempSync(path.join(work, 'm-'));
    fs.mkdirSync(path.join(dir, 'docs/site'), {recursive: true});
    fs.mkdirSync(path.join(dir, 'tests'));
    for (const f of ['core.mjs', 'geometry.mjs', 'museum.html', 'museum.json']) fs.copyFileSync(path.join(root, 'docs/site', f), path.join(dir, 'docs/site', f));
    fs.writeFileSync(path.join(dir, 'docs/site/museum.mjs'), mutated);
    fs.copyFileSync(path.join(root, 'tests/test_museum_frontend.mjs'), path.join(dir, 'tests/test_museum_frontend.mjs'));
    const env = {...process.env}; delete env.MUSEUM_FIXTURE; delete env.MUSEUM_MATH_ROOT;
    const r = spawnSync(process.execPath, ['--test', path.join(dir, 'tests/test_museum_frontend.mjs')], {encoding: 'utf8', env});
    const count = k => (r.stdout.match(new RegExp(`^# ${k} (\\d+)`, 'm')) || [])[1];
    console.log(`${name}: exit=${r.status} pass=${count('pass')} fail=${count('fail')} skipped=${count('skipped')}`);
  }
} finally { fs.rmSync(work, {recursive: true, force: true}); }
