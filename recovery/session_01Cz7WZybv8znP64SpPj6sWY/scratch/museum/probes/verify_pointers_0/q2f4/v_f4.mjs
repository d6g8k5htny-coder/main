// Q2-F4 verifier 1 (independent of probes/pointers/*). Imports the cd32303 museum.mjs read-only, stubs ALL network,
// serves only the hashlib-verified fixture (build_fixture.py) plus explicitly listed hypothetical bytes.
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url)); const CD = path.resolve(HERE, '../../../cd32303');
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const FX = JSON.parse(fs.readFileSync(path.join(HERE, 'fixture.json'), 'utf8'));
const MANIFEST_TEXT = fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8');
const fresh = () => JSON.parse(MANIFEST_TEXT);
const txt = u => Buffer.from(FX.sources[u], 'base64').toString('utf8');
const m0 = fresh(); const INDEX = txt(m0.index_source.url), STATUS = txt(m0.status_source.url);
const validate = m => { try { museum.validateBoundManifest(m, INDEX, STATUS); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; } };
// minimal DOM
class El { constructor(t){this.tagName=t.toUpperCase();this.children=[];this.attributes={};this._t='';this.className='';}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent)].filter(Boolean).join(' | ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} }
async function render(m, extra = {}) {
  const ids = ['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit']; const nodes = Object.fromEntries(ids.map(i => [i, new El('div')]));
  const doc = {createElement: t => new El(t), getElementById: i => nodes[i] ?? null};
  const fetcher = async u => { const k = String(u);
    if (k === 'museum.json') return new Response(JSON.stringify(m));
    if (Object.hasOwn(extra, k)) return new Response(extra[k]);
    if (Object.hasOwn(FX.sources, k)) return new Response(Buffer.from(FX.sources[k], 'base64'));
    return new Response('absent', {status: 404}); };
  await museum.startMuseum({document: doc, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const cards = {}; for (const host of nodes['claim-cards'].children) { const a = host.children[0]; if (a?.attributes?.['data-object-id']) cards[a.attributes['data-object-id']] = a; }
  return {state: nodes['museum-state'].textContent, cards};
}
const walk = (n, o = []) => { if (n?.tagName) { o.push(n); n.children.forEach(c => walk(c, o)); } return o; };
const reviewView = card => { if (!card) return 'CARD NOT RENDERED (refused at fetch)';
  const ns = walk(card), ps = ns.filter(n => n.tagName === 'P').map(n => n._t);
  const lab = ps.find(p => /Review pointer in the verified index|Byte-verified review|No separate byte-frozen/.test(p));
  const a = ns.find(n => n.tagName === 'A' && n._t === 'Open review ↗');
  const extra = ps.filter(p => /OPEN \/ NOT LANDED|Unmerged review pointer/.test(p));
  return JSON.stringify({label: lab, open_review: a?.href ?? null, also_shown: extra}); };
// mirror of museum.mjs:52, to say WHICH clause matched
const clauses = (q, s) => ({'](path)': q.includes(`](${s.path})`), html_url: q.includes(s.html_url), '/path) suffix': q.includes(`/${s.path})`)});
const rebind = (d, repository, commit, p = d.path) => { const e = p.split('/').map(encodeURIComponent).join('/');
  return {...d, repository, commit, path: p, url: `https://raw.githubusercontent.com/${repository}/${commit}/${e}`, html_url: `https://github.com/${repository}/blob/${commit}/${e}`}; };
const depoint = r => { delete r.pointer_only; delete r.review_url; delete r.review_notice; return r; };
async function run(label, m, id, extra = {}) {
  const v = validate(m); let line = `${label}\n    validateBoundManifest: ${v}`;
  if (v === 'PASS') { const r = await render(m, extra); line += `\n    card ${id}: ${reviewView(r.cards[id])}\n    state: ${r.state}`; }
  console.log(line);
}
console.log('fixture index/status digests:', crypto.createHash('sha256').update(INDEX).digest('hex') === m0.index_source.sha256, crypto.createHash('sha256').update(STATUS).digest('hex') === m0.status_source.sha256);
await run('[baseline] committed museum.json, d5-pin', fresh(), 'd5-pin-neighborhoods-open');
await run('[baseline] committed museum.json, d5-all-height', fresh(), 'd5-all-height-annulus');
// (a1)
{ const m = fresh(); const q = m.claims[12].scope_quote; depoint(m.claims[12].review);
  console.log('(a) d5-pin STATUS quote:', q); console.log('    clause hits for review=PROOF_INDEX@d6628da:', JSON.stringify(clauses(q, m.claims[12].review)));
  await run('(a1) d5-pin: pointer_only/review_url/review_notice removed (descriptor stays PROOF_INDEX@d6628da)', m, 'd5-pin-neighborhoods-open'); }
// (a2) REAL other revisions of PROOF_INDEX.md, real bytes from git
for (const [c, e] of Object.entries(FX.proof_index_revisions)) {
  const m = fresh(); const r = rebind(depoint(m.claims[12].review), e.repository, c); Object.assign(r, {blob: e.blob, bytes: e.bytes, sha256: e.sha256});
  m.claims[12].review = r;
  await run(`(a2) d5-pin review := PROOF_INDEX.md@${c.slice(0, 8)} (REAL git bytes, blob ${e.blob.slice(0, 8)}, ${e.bytes} B; snapshot pins e4cb1d94/12708 B)`, m, 'd5-pin-neighborhoods-open', {[r.url]: Buffer.from(e.b64, 'base64')}); }
// (b) other commit / repo for a reviewed card
{ const m = fresh(); const o = m.claims[3].review; const r = rebind(o, 'd6g8k5htny-coder/Math-', 'bebb77ae9fe3b22c425a979b780fc99a72c8e761'); m.claims[3].review = r;
  await run('(b1) d5-all-height review at Math-@bebb77ae (real commit; SAME blob 559d7224 as at d6628da)', m, 'd5-all-height-annulus', {[r.url]: Buffer.from(FX.sources[o.url], 'base64')}); }
{ const m = fresh(); const o = m.claims[3].review; const r = rebind(o, 'd6g8k5htny-coder/query-', 'f'.repeat(40)); m.claims[3].review = r;
  await run('(b2) d5-all-height review at query-@ffff.. (hypothetical identity, bytes served = genuine review bytes)', m, 'd5-all-height-annulus', {[r.url]: Buffer.from(FX.sources[o.url], 'base64')}); }
// (c) suffix clause, bare REVIEW.md
{ const m = fresh(); const forged = 'Hypothetical bytes at a hypothetical identity.\n'; const r = rebind(m.claims[3].review, 'd6g8k5htny-coder/query-', 'e'.repeat(40), 'REVIEW.md');
  Object.assign(r, {bytes: Buffer.byteLength(forged), sha256: crypto.createHash('sha256').update(forged).digest('hex')}); m.claims[3].review = r;
  console.log('(c) clause hits:', JSON.stringify(clauses(m.claims[3].scope_quote, r)));
  await run('(c) d5-all-height review := query-@eeee../REVIEW.md (hypothetical bytes, self-consistent digest)', m, 'd5-all-height-annulus', {[r.url]: forged}); }
// (d) proof fills review slot
{ const m = fresh(); m.claims[0].review = structuredClone(m.claims[0].proof); await run('(d) d2 review := d2 proof descriptor', m, 'd2-lifetime-remainder'); }
// controls
{ const m = fresh(); m.claims[3].review = structuredClone(m.claims[4].review); await run('(ctrl) d5-all-height review := d5-height-window review', m, 'd5-all-height-annulus'); }
for (const i of [0, 1, 2, 9, 11]) { const m = fresh(); depoint(m.claims[i].review); console.log(`(ctrl) ${m.claims[i].id}: pointer fields removed -> ${validate(m)}`); }
