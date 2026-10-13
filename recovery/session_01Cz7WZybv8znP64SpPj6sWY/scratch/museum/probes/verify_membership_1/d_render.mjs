// d_render.mjs <checkout> <mode>: run cd32303 docs/site/museum.mjs startMuseum() on <checkout>/docs/site/museum.json
// with a stubbed fetcher (no network). Main-repository URLs .../main/<commit>/<path>:
//   mode "harness": read_source semantics (git show <commit>:<path> in <checkout>, else the working-tree file)
//                   = the bytes museum_check.py puts in MUSEUM_FIXTURE under that URL;
//   mode "strict":  raw.githubusercontent.com semantics: only git show <commit>:<path>, else 404.
// Math- URLs: verified mathfix/ files in both modes.
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
const V = new URL('.', import.meta.url).pathname;
const museum = await import(new URL('../../cd32303/docs/site/museum.mjs', import.meta.url));
const [ck, mode] = process.argv.slice(2);
const manifestText = fs.readFileSync(`${ck}/docs/site/museum.json`, 'utf8');
const show = (c, p) => { try { return execFileSync('git', ['-C', ck, 'show', `${c}:${p}`], {stdio: ['ignore', 'pipe', 'ignore']}); } catch { return null; } };
const log = [];
const fetcher = async url => {
  url = String(url);
  if (url === 'museum.json') return new Response(manifestText);
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  if (!m) return new Response('', {status: 404});
  const [, repo, c, p] = [m[0], m[1], m[2], decodeURIComponent(m[3])];
  let b = null;
  if (repo === 'Math-') { const f = `${V}mathfix/${p}`; b = fs.existsSync(f) ? fs.readFileSync(f) : null; }
  else { b = show(c, p); if (!b && mode === 'harness' && fs.existsSync(`${ck}/${p}`)) b = fs.readFileSync(`${ck}/${p}`); }
  if (p.startsWith('incoming/')) log.push(`${b ? 200 : 404} main@${c.slice(0, 8)}:${p}`);
  return b ? new Response(b) : new Response('', {status: 404});
};
class El { constructor(t){this.tagName=t;this.children=[];this._t='';this.attributes={};}
  set textContent(v){this._t=String(v);this.children=[];} get textContent(){return [this._t,...this.children.map(c=>c.textContent)].filter(Boolean).join(' ');}
  append(...c){this.children.push(...c.map(x=>typeof x==='string'?{textContent:x}:x));} replaceChildren(...c){this._t='';this.children=[];this.append(...c);}
  setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const ids = ['museum-state','claim-cards','lifetime-fixture','packet-cards','active-exhibit'];
const nodes = new Map(ids.map(i => [i, new El('div')]));
const document = {createElement: t => new El(t), createTextNode: v => ({textContent: String(v)}), getElementById: i => nodes.get(i) ?? null};
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
const sq = s => s.replace(/\s+/g, ' ').trim();
console.log(`[${mode}] packet requests: ${log.join(' | ')}`);
console.log(`[${mode}] museum-state: ${sq(nodes.get('museum-state').textContent)}`);
console.log(`[${mode}] packet-cards: ${sq(nodes.get('packet-cards').textContent).slice(0, 300)}`);
