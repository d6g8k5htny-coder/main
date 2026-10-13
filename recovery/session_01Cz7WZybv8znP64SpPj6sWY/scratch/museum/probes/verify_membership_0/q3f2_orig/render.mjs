// render.mjs <museum.json> <mode>: run the cd32303 museum.mjs startMuseum() on a manifest, network stubbed.
// mode "harness": serve every URL from verified bytes keyed by URL, exactly as museum_check.py's
//   MUSEUM_FIXTURE does (bytes come from read_source, i.e. with the working-tree fallback).
// mode "strict": emulate raw.githubusercontent.com/<repo>/<commit>/<path>: main-repository URLs are
//   answered ONLY from `git show <commit>:<path>` in the read-only main object store (404 otherwise).
//   Math- URLs (commit d6628da, not in the local shallow clone) are served from the verified fixture.
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
const HERE = new URL('.', import.meta.url).pathname;
const CD = new URL('file://<SCRATCH>/museum/cd32303/');
const museum = await import(new URL('docs/site/museum.mjs', CD));
const [manifestPath, mode] = process.argv.slice(2);
const manifestText = fs.readFileSync(manifestPath, 'utf8');
const manifest = JSON.parse(manifestText);
const fixtureFile = url => { // repository/commit/path -> fixture/<repo>/<path> (bytes verified by build_fixture.py)
  const m = url.match(/^https:\/\/raw\.githubusercontent\.com\/d6g8k5htny-coder\/(main|Math-)\/([0-9a-f]{40})\/(.+)$/);
  return m && {repo: m[1], commit: m[2], path: decodeURIComponent(m[3])};
};
const gitShow = (commit, path) => { try { return execFileSync('git', [`--git-dir=${HERE}objstore.git`, 'show', `${commit}:${path}`], {stdio: ['ignore', 'pipe', 'ignore'], env: {...process.env, GIT_OPTIONAL_LOCKS: '0'}}); } catch { return null; } };
const fetched = [];
const fetcher = async url => {
  url = String(url);
  if (url === 'museum.json') return new Response(manifestText);
  const t = fixtureFile(url); if (!t) return new Response('', {status: 404});
  let bytes;
  if (mode === 'strict' && t.repo === 'main') bytes = gitShow(t.commit, t.path);
  else { const f = `${HERE}fixture/${t.repo}/${t.path}`; bytes = fs.existsSync(f) ? fs.readFileSync(f) : null; }
  fetched.push(`${bytes ? 200 : 404} ${t.repo}@${t.commit.slice(0, 8)}:${t.path}`);
  return bytes ? new Response(bytes) : new Response('', {status: 404});
};
class Element { constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this._text='';this.id='';}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent??String(c)).join(' ');}
  append(...c){this.children.push(...c);} replaceChildren(...c){this._text='';this.children=c;} setAttribute(k,v){this.attributes[k]=String(v);} addEventListener(){} }
const html = fs.readFileSync(new URL('docs/site/museum.html', CD), 'utf8');
const nodes = new Map([...html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"/gi)].map(m => [m[2], new Element(m[1])]));
const document = {createElement: t => new Element(t), createTextNode: v => ({textContent: v}), getElementById: id => nodes.get(id) ?? null};
await museum.startMuseum({document, fetcher, search: '', geometryLoader: () => { throw Error('no geometry'); }});
console.log(`[${mode}] manifest status_source.commit=${manifest.status_source.commit.slice(0,8)} packets=${JSON.stringify(manifest.packets.map(p => p.id + '@' + p.result.commit.slice(0,8)))}`);
console.log(`[${mode}] museum-state: ${nodes.get('museum-state').textContent}`);
console.log(`[${mode}] packet-cards: ${nodes.get('packet-cards').textContent.replace(/\s+/g, ' ').slice(0, 400)}`);
console.log(`[${mode}] packet requests: ${fetched.filter(f => f.includes('incoming/')).join(' | ')}`);
