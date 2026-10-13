// Verifier 2 of Q2-F4. Independent harness: imports the cd32303 museum.mjs read-only, serves only the
// fixture bytes (re-verified by v0_fixture_integrity.py) plus explicitly declared hypothetical files,
// and disables the network. Usage: node v1_q2f4.mjs <cd32303 root> <verified-inputs.json>
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
const [CD, FIX] = process.argv.slice(2).map(p => path.resolve(p));
globalThis.fetch = async u => { throw Error('NETWORK DISABLED: ' + u); };
const museum = await import(pathToFileURL(path.join(CD, 'docs/site/museum.mjs')));
const fixtures = JSON.parse(fs.readFileSync(FIX, 'utf8'));
const base = () => JSON.parse(fs.readFileSync(path.join(CD, 'docs/site/museum.json'), 'utf8'));
const bytesOf = url => Buffer.from(fixtures[url], 'base64');
const M0 = base();
const INDEX = bytesOf(M0.index_source.url).toString('utf8'), STATUS = bytesOf(M0.status_source.url).toString('utf8');
const sha = b => createHash('sha256').update(b).digest('hex');
const blob = b => createHash('sha1').update(Buffer.concat([Buffer.from(`blob ${b.length}\0`), b])).digest('hex');
const pinAt = (d, repository, commit, filePath, content) => {
  const p = filePath.split('/').map(encodeURIComponent).join('/');
  return {...d, repository, commit, path: filePath, blob: blob(content), bytes: content.length, sha256: sha(content),
    url: `https://raw.githubusercontent.com/${repository}/${commit}/${p}`, html_url: `https://github.com/${repository}/blob/${commit}/${p}`};
};
const validate = m => { try { museum.validateBoundManifest(m, INDEX, STATUS); return 'PASS'; } catch (e) { return 'REFUSED: ' + e.message; } };
class El { constructor(t){this.tag=t.toUpperCase();this.kids=[];this.attrs={};this._t='';this.className='';}
  set textContent(v){this._t=String(v);this.kids=[];} get textContent(){return [this._t,...this.kids.map(k=>k.textContent)].filter(Boolean).join(' ');}
  append(...k){this.kids.push(...k);} replaceChildren(...k){this._t='';this.kids=k;} setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(){} }
const all = (n, o = []) => { o.push(n); n.kids.forEach(k => all(k, o)); return o; };
async function render(m, extra = {}) {
  const html = fs.readFileSync(path.join(CD, 'docs/site/museum.html'), 'utf8');
  const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(x => [x[2], new El(x[1])]));
  const document = {createElement: t => new El(t), getElementById: id => nodes.get(id) ?? null};
  const fetcher = async u => { u = String(u);
    if (u === 'museum.json') return new Response(JSON.stringify(m));
    if (Object.hasOwn(extra, u)) return new Response(extra[u]);
    if (Object.hasOwn(fixtures, u)) return new Response(bytesOf(u));
    return new Response('absent', {status: 404}); };
  await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const cards = {};
  for (const host of nodes.get('claim-cards').kids) { const a = host.kids[0]; if (a?.attrs['data-object-id']) cards[a.attrs['data-object-id']] = all(a); else cards['refused:' + host.textContent.slice(0, 50)] = []; }
  return {state: nodes.get('museum-state').textContent, cards};
}
function reviewView(nodes) {
  const ps = nodes.filter(n => n.tag === 'P').map(n => n._t);
  const label = ps.find(t => /^(Review pointer in the verified index|Byte-verified review|No separate byte-frozen review)/.test(t));
  const a = nodes.find(n => n.tag === 'A' && n._t === 'Open review ↗');
  return {label, open_review: a?.href ?? null, amend_text: ps.includes('OPEN / NOT LANDED — the claimed closure remains open.'),
          unmerged_notice: ps.includes('Unmerged review pointer only. Its target is not fetched or adopted by this viewer.')};
}
const clauses = (q, s) => ({rel: q.includes(`](${s.path})`), html_url: q.includes(s.html_url), suffix: q.includes(`/${s.path})`)});
async function show(tag, m, id, extra) {
  const v = validate(m); console.log(`${tag}\n   validateBoundManifest: ${v}`);
  if (v !== 'PASS') return;
  const r = await render(m, extra);
  console.log(`   card ${id}: ${r.cards[id] ? JSON.stringify(reviewView(r.cards[id])) : 'NOT RENDERED (refused at fetch)'}`);
  console.log(`   museum-state: ${r.state}`);
}

console.log('baseline:', validate(M0));
{ const r = await render(M0); for (const id of ['d2-lifetime-remainder','d5-all-height-annulus','d5-pin-neighborhoods-open']) console.log(`   baseline card ${id}: ${JSON.stringify(reviewView(r.cards[id]))}`); }

// (a) d5-pin review: drop the pointer fields. Which sourceContainsPath clause matches, and on what link?
{ const m = base(), r = m.claims[12].review; delete r.pointer_only; delete r.review_url; delete r.review_notice;
  const q = m.claims[12].scope_quote;
  console.log(`\n(a) d5-pin quote clauses for review path ${r.path}: ${JSON.stringify(clauses(q, r))}; links in quote: ${JSON.stringify([...q.matchAll(/\]\(([^)]+)\)/g)].map(x => x[1]))}`);
  await show('(a) d5-pin-neighborhoods-open: pointer_only/review_url/review_notice removed', m, 'd5-pin-neighborhoods-open'); }

// (b) same review path, another Math- commit, DIFFERENT bytes with a self-consistent digest (stand-in for a later
//     revision of the file). The pinned index's relative link denotes Math-@d6628da's bytes, not these.
const REAL = bytesOf(M0.claims[3].review.url);
const REVISED = Buffer.from(REAL.toString('utf8').replace(/ACCEPT/g, 'AMEND'));
console.log(`\n(b) revised bytes differ from Math-@d6628da bytes: ${sha(REVISED) !== sha(REAL)} (ACCEPT occurrences ${(REAL.toString().match(/ACCEPT/g)||[]).length} -> ${(REVISED.toString().match(/ACCEPT/g)||[]).length})`);
for (const [repo, commit] of [['d6g8k5htny-coder/Math-', '1'.repeat(40)], ['d6g8k5htny-coder/query-', '2'.repeat(40)]]) {
  const m = base(); const d = pinAt(m.claims[3].review, repo, commit, m.claims[3].review.path, REVISED); m.claims[3].review = d;
  console.log(`   clauses: ${JSON.stringify(clauses(m.claims[3].scope_quote, d))}`);
  await show(`(b) d5-all-height review re-pinned to ${repo}@${commit.slice(0, 8)} (index_source is Math-@${M0.index_source.commit.slice(0, 8)})`, m, 'd5-all-height-annulus', {[d.url]: REVISED});
}
// (b-control) the SAME change applied to the proof is refused (line 75 ties proof to index_source repository/commit)
{ const m = base(); const P = bytesOf(m.claims[3].proof.url); m.claims[3].proof = pinAt(m.claims[3].proof, 'd6g8k5htny-coder/Math-', '1'.repeat(40), m.claims[3].proof.path, P);
  await show('(b-control) d5-all-height PROOF re-pinned to Math-@11111111 (same bytes)', m, 'd5-all-height-annulus'); }

// (c) path "REVIEW.md": only the '/${path})' suffix clause can match
{ const m = base(); const forged = Buffer.from('Hypothetical file at repository root.\n');
  const d = pinAt(m.claims[3].review, 'd6g8k5htny-coder/query-', '3'.repeat(40), 'REVIEW.md', forged); m.claims[3].review = d;
  console.log(`\n(c) clauses: ${JSON.stringify(clauses(m.claims[3].scope_quote, d))}`);
  await show('(c) d5-all-height review = query-@33333333:REVIEW.md', m, 'd5-all-height-annulus', {[d.url]: forged}); }

// (d) the claim's own proof descriptor in its review slot
{ const m = base(); m.claims[0].review = structuredClone(m.claims[0].proof);
  console.log(`\n(d) clauses: ${JSON.stringify(clauses(m.claims[0].scope_quote, m.claims[0].review))}`);
  await show('(d) d2 review := d2 proof descriptor', m, 'd2-lifetime-remainder'); }

// controls: removing pointer fields on the other five pointer cards
console.log('');
for (const i of [0, 1, 2, 9, 11]) { const m = base(), r = m.claims[i].review; delete r.pointer_only; delete r.review_url; delete r.review_notice;
  console.log(`(control) ${m.claims[i].id}: pointer fields removed -> ${validate(m)}`); }
{ const m = base(); m.claims[3].review = structuredClone(m.claims[4].review); console.log(`(control) d5-all-height review := d5-height-window review -> ${validate(m)}`); }
