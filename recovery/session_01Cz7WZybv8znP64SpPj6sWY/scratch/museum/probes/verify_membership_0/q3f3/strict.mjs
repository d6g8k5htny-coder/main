// strict.mjs <museum.json> <cd32303 root> <objs.git>: run cd32303 docs/site/museum.mjs startMuseum() with the
// network stubbed. Emulates raw.githubusercontent.com/<repo>/<commit>/<path>: main URLs are answered ONLY from
// `git show <commit>:<path>` (404 if the commit lacks the path); Math- URLs from the byte-verified fixture
// (museum.mjs re-verifies size, blob and sha256 of every byte it is given, so the fixture cannot pass bad bytes).
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
const [manifestPath, cdRoot, gitDir] = process.argv.slice(2);
const mathFixture = new URL('../../membership/fixture/Math-/', import.meta.url).pathname;
const museum = await import('file://' + cdRoot + '/docs/site/museum.mjs');
const manifestText = fs.readFileSync(manifestPath, 'utf8'), manifest = JSON.parse(manifestText);
const log = [];
const fetcher = async (url) => {
  url = String(url);
  if (url === 'museum.json') return new Response(manifestText);
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  if (!m) { log.push('404 ' + url); return new Response('', {status: 404}); }
  const [, repo, commit, enc] = m, path = decodeURIComponent(enc);
  let bytes = null;
  if (repo === 'main') { try { bytes = execFileSync('git', ['--git-dir=' + gitDir, 'show', commit + ':' + path], {stdio: ['ignore', 'pipe', 'ignore']}); } catch { bytes = null; } }
  else { const f = mathFixture + path; bytes = fs.existsSync(f) ? fs.readFileSync(f) : null; }
  if (repo === 'main') log.push(`${bytes ? 200 : 404} main@${commit.slice(0, 8)}:${path}`);
  return bytes ? new Response(bytes) : new Response('', {status: 404});
};
class El { constructor(t){this.tagName=t;this.children=[];this._t='';this.attributes={};}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent)].filter(Boolean).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._t='';this.children=c;} setAttribute(k,v){this.attributes[k]=v;} addEventListener(){} }
const nodes = new Map(['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'].map(id => [id, new El('div')]));
const document = {createElement: t => new El(t), createTextNode: v => ({textContent: String(v)}), getElementById: id => nodes.get(id) ?? null};
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('geometry not loaded'); }});
const html = fs.readFileSync(cdRoot + '/docs/site/museum.html', 'utf8');
console.log(`strict render: status_source.commit=${manifest.status_source.commit.slice(0,8)} packet.result.commit=${manifest.packets[0].result.commit.slice(0,8)}`);
console.log('museum-state: ' + nodes.get('museum-state').textContent);
console.log('main fetches: ' + log.join(' | '));
const pc = nodes.get('packet-cards').textContent.replace(/\s+/g, ' ');
console.log('packet card shows: ' + (pc.match(/Packet — not STATUS.*?Commit [0-9a-f]{40}/)?.[0] ?? pc.slice(0, 200)));
console.log('static section text: ' + html.match(/These are pinned landed submissions\.[^<]*/)[0].slice(0, 36) + '...');
console.log('refused cards: ' + (nodes.get('claim-cards').textContent.match(/Unavailable:[^.]*/g) ?? []).length);
