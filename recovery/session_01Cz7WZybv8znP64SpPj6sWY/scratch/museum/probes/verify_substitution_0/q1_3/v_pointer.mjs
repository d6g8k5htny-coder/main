// Verifier 1 / Q1-3. Independent of probes/substitution/common.mjs. Read-only against cd32303 bytes.
// No network: global fetch throws; every source served to startMuseum is identity-checked by the
// code under test (core.mjs verifiedBytes: byte count + SHA-256), and the index/status bytes are
// additionally checked here against the shipped museum.json pins before use.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
globalThis.fetch = async u => { throw Error('NETWORK BLOCKED ' + u); };
const S = '<SCRATCH>/museum';
const T = `${S}/cd32303`, HERE = `${S}/probes/verify_substitution_0/q1_3`, MATH = `${S}/probes/verify_substitution_0/math`;
const sha = b => createHash('sha256').update(b).digest('hex');
const MOD = process.argv[2] === 'patched' ? `${HERE}/patched/museum.mjs` : `${T}/docs/site/museum.mjs`;
console.log('module under test:', MOD.replace(S, '$S'), 'sha256', sha(fs.readFileSync(MOD)).slice(0, 16));
const museum = await import(MOD);
const shippedRaw = fs.readFileSync(`${T}/docs/site/museum.json`, 'utf8');
const shipped = () => JSON.parse(shippedRaw);
const m0 = shipped();
const indexB = fs.readFileSync(`${MATH}/PROOF_INDEX.md`), statusB = fs.readFileSync(`${T}/STATUS.md`);
if (sha(indexB) !== m0.index_source.sha256 || sha(statusB) !== m0.status_source.sha256) throw Error('fixture identity mismatch');
const index = indexB.toString('utf8'), status = statusB.toString('utf8');
const v = m => { try { museum.validateBoundManifest(m, index, status); return 'ACCEPTED'; } catch (e) { return 'REFUSED (' + e.message + ')'; } };
console.log('baseline shipped museum.json:', v(shipped()));

const ISS = 'https://github.com/d6g8k5htny-coder/main/issues/';
const cases = [
  ['d3-side24-coefficient', ISS + '99', 'CONTROL: unrelated issue, not a prefix'],
  ['d3-side24-coefficient', ISS + '65#issuecomment-5841779222', 'other link in the same bullet (delimited)'],
  ['d3-side24-coefficient', ISS + '6', 'issue #6, prefix of #65'],
  ['d3-side24-coefficient', ISS + '65#issuecomment-58412', 'truncated comment id'],
  ['d2-lifetime-remainder', ISS + '6', 'issue #6, prefix of #67'],
  ['d2-lifetime-remainder', ISS + '67#issuecomment-5841270276', 'other comment in the same bullet (delimited)'],
  ['d4-fixed-remote-rn', ISS + '7', 'issue #7, prefix of #76'],
  ['p15-demand-one-counterexample', ISS + '6', 'issue #6, prefix of #67'],
  ['d1-parent-selection-open', ISS + '6', 'AMEND D1: issue #6, prefix of #63'],
  ['d1-parent-selection-open', ISS + '63', 'AMEND D1: task issue #63 itself (delimited link in row)'],
];
for (const [id, url, label] of cases) {
  const m = shipped(); const c = m.claims.find(x => x.id === id);
  c.review.review_url = url;
  console.log(`[${id}] ${label}: review_url=${url.replace(ISS, 'main/issues/')} | "](url)" in quote=${c.scope_quote.includes('](' + url + ')')} | substring=${c.scope_quote.includes(url)} -> ${v(m)}`);
}

// pointer_only strictness: validateDescriptor uses ===true, validateBoundManifest/renderClaim use truthiness.
const ptr = (po, url, notice) => ({...m0.index_source, pointer_only: po, review_url: url, ...(notice === undefined ? {} : {review_notice: notice})});
for (const [po, url] of [[true, ''], [1, ''], ['yes', ''], [1, 'reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md']]) {
  let r; try { museum.validateDescriptor(ptr(po, url)); r = 'passes'; } catch (e) { r = 'throws ' + e.message; }
  const m = shipped(); m.claims.find(x => x.id === 'd5-two-scale').review = ptr(po, url);
  console.log(`validateDescriptor({pointer_only:${JSON.stringify(po)}, review_url:${JSON.stringify(url)}, no notice}) ${r}; d5-two-scale manifest -> ${v(m)}`);
}

// ---- render through startMuseum with a minimal DOM and local pinned bytes ----
class El {
  constructor(t) { this.tagName = t.toUpperCase(); this.children = []; this.attributes = {}; this._t = ''; }
  set textContent(x) { this._t = String(x); this.children = []; } get textContent() { return this._t + this.children.map(c => c.textContent ?? '').join(''); }
  append(...c) { this.children.push(...c); } replaceChildren(...c) { this._t = ''; this.children = c; }
  setAttribute(k, x) { this.attributes[k] = String(x); } getAttribute(k) { return this.attributes[k] ?? null; } addEventListener() {}
}
function doc() {
  const nodes = new Map(['museum-state', 'claim-cards', 'lifetime-fixture', 'packet-cards', 'active-exhibit'].map(i => [i, new El('div')]));
  return {createElement: t => new El(t), createElementNS: (_n, t) => new El(t), createTextNode: x => ({textContent: x}), getElementById: i => nodes.get(i) ?? null, head: new El('head')};
}
const walk = (n, f) => { if (n && typeof n === 'object') { f(n); for (const c of n.children ?? []) walk(c, f); } };
const local = u => {
  const g = String(u).match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  if (!g) return null; const p = decodeURIComponent(g[3]);
  const f = g[1] === 'main' ? `${T}/${p}` : `${MATH}/${p}`;
  return fs.existsSync(f) ? fs.readFileSync(f) : null;
};
async function render(m, id) {
  const d = doc();
  const fetcher = async u => { if (u === 'museum.json') return new Response(JSON.stringify(m)); const b = local(u); return b ? new Response(b) : new Response('x', {status: 404}); };
  await museum.startMuseum({document: d, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
  const root = d.getElementById('claim-cards'); let card = null, refused = null;
  walk(root, n => { if (!card && n.attributes?.['data-object-id'] === id) card = n; });
  if (!card) { walk(root, n => { if (/refused/.test(n.className || '') && n.textContent.includes('Unavailable')) refused = (refused || []).concat(n.textContent); }); return {state: d.getElementById('museum-state').textContent, refused}; }
  const texts = [], links = []; walk(card, n => { if (n._t) texts.push(n._t); if (n.tagName === 'A') links.push([n._t, n.href]); });
  return {cls: card.attributes['data-object-class'], review: texts.filter(t => /review|pointer|byte-frozen/i.test(t)), reviewLink: links.find(([l]) => /review/i.test(l)), emptyP: (() => { let k = 0; walk(card, n => { if (n.tagName === 'P' && n._t === 'undefined') k++; }); return k; })()};
}
for (const [id, mut, label] of [
  ['d3-side24-coefficient', c => { c.review.review_url = ISS + '6'; }, 'd3 review_url -> main/issues/6'],
  ['d1-parent-selection-open', c => { c.review.review_url = ISS + '6'; }, 'AMEND D1 review_url -> main/issues/6'],
  ['d5-two-scale', c => { c.review = ptr(1, ''); }, 'd5-two-scale review := pointer_only:1, review_url:"" (no notice)'],
]) {
  const m = shipped(); mut(m.claims.find(x => x.id === id));
  console.log(`RENDER [${label}]`, JSON.stringify(await render(m, id)));
}
console.log('RENDER [baseline d5-two-scale]', JSON.stringify(await render(shipped(), 'd5-two-scale')));
