#!/usr/bin/env python3
"""Source-bound acquired-excerpt save/citation regression; engineering only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import traceback
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'universal-law-reading-list/v1'

@contextmanager
def serve(site):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try: yield f'http://127.0.0.1:{server.server_port}/'
    finally: server.shutdown(); server.server_close(); worker.join()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, default=ROOT/'docs/site')
    ap.add_argument('--origin')
    ap.add_argument('--browser', default=shutil.which('chromium'))
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [], 'page_errors': [],
              'driver_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'scientific_effect': 'NONE',
              'scope': 'Actual admitted observation; local driver hash alone does not establish public served bytes.'}
    def run(origin, browser):
        from playwright.sync_api import expect
        report['origin'] = origin
        # Bind replay to the expected served implementation, including admission
        # and the one storage/citation controller. Hash decoded UTF-8 file bytes.
        paths=['workspace.html','library-discovery.mjs','research-desk.mjs','current.mjs',
               'current-model.mjs','current.json','current-meta.json','research-desk-store.mjs',
               'reading-list-citation.mjs','library-source-search.mjs']
        binding_context=browser.new_context()
        binding_context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
        binding_page=binding_context.new_page()
        binding={'case':'served-source-binding','files':{}};report['cases'].append(binding)
        try:
            binding_page.goto(origin+'index.html')
            fetched=binding_page.evaluate('''async(paths)=>await Promise.all(paths.map(async path=>{
              const response=await fetch(path,{cache:'no-store'});
              if(!response.ok)throw Error(path+': '+response.status);
              return [path,await response.text()];
            }))''',paths)
            for path,body in fetched:
                actual=body.encode('utf-8');expected=(args.site/path).read_bytes()
                identity={'bytes':len(actual),'sha256':sha256(actual).hexdigest()}
                binding['files'][path]=identity
                assert actual==expected, 'Served bytes differ from --site: '+path
            binding['passed']=True
        except Exception:binding['passed']=False;binding['error']=traceback.format_exc()
        finally:binding_context.close()
        for width, height, theme in [(390,844,'dark'), (1440,1000,'light')]:
            context = browser.new_context(viewport={'width':width,'height':height}, color_scheme=theme)
            context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
            page = context.new_page(); page.set_default_timeout(8000)
            page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            row = {'case':'excerpt-source-save-'+str(width),'width':width,'theme':theme}; report['cases'].append(row)
            try:
                page.goto(origin+'workspace.html#library-current')
                field=page.locator('#current-research .current-search'); select=page.locator('#library-collection')
                expect(field).to_be_enabled(); expect(select).to_be_visible()
                raw, meta=page.evaluate("async()=>await Promise.all([fetch('current.json').then(r=>r.text()),fetch('current-meta.json').then(r=>r.json())])")
                assert len(raw.encode())==meta['bytes'] and sha256(raw.encode()).hexdigest()==meta['sha256']
                data=json.loads(raw)
                # Literal text query returns two excerpts of the same actual source.
                query='hess'
                excerpts=[e for e in data['equations'] if query in (e['text']+' '+e['repository']+' '+e['path']).lower()]
                assert len(excerpts)==2 and len({(e['repository'],e['commit'],e['path']) for e in excerpts})==1
                excerpt=excerpts[0]
                source=next(s for s in data['sources'] if all(s[k]==excerpt[k] for k in ['repository','commit','path']))
                row['observation']={'bytes':meta['bytes'],'sha256':meta['sha256'],'source':source,'excerpt_sha256':[e['sha256'] for e in excerpts]}
                field.fill(query);select.select_option('text')
                results=page.locator('#library-text-results > li');expect(results).to_have_count(2)
                assert page.locator('#desk-save-current').is_hidden()
                buttons=results.locator('.desk-save-source')
                expect(buttons, 'Matched acquired excerpts must offer the existing source-save control').to_have_count(2)
                for i,e in enumerate(excerpts):
                    assert sha256(e['text'].encode()).hexdigest()==e['sha256']
                    assert e['sha256'] in results.nth(i).locator('details').text_content()
                buttons.first.focus();page.keyboard.press('Enter')
                expect(page.locator('#desk-count')).to_have_text('1 saved item')
                expect(buttons.first).to_be_disabled();expect(buttons.last).to_be_disabled()
                saved=json.loads(page.evaluate('(key)=>localStorage.getItem(key)',SCHEMA))['entries']
                assert len(saved)==1;entry=saved[0]
                for key in ['repository','commit','path','blob','bytes']:assert entry[key]==source[key],(key,entry,source)
                assert 'sha256' not in entry and entry['blob'] not in [e['sha256'] for e in excerpts]
                reading=page.locator('#reading-list');reading.evaluate('(e)=>e.open=true')
                saved_row=page.locator('#desk-entries > li');expect(saved_row).to_have_count(1)
                href='https://github.com/'+source['repository']+'/blob/'+source['commit']+'/'+quote(source['path'],safe="/~!*'()")
                expect(saved_row.locator(':scope > div > a').first).to_have_attribute('href',href)
                saved_row.locator('.desk-remove').click()
                expect(page.locator('#desk-count')).to_have_text('0 saved items')
                expect(buttons.first).to_be_enabled();expect(buttons.last).to_be_enabled()
                page.locator('#desk-undo').click()
                expect(page.locator('#desk-count')).to_have_text('1 saved item')
                expect(buttons.first).to_be_disabled();expect(buttons.last).to_be_disabled()
                # Cross-tab updates use the same pre-existing storage controller.
                other=context.new_page();other.goto(page.url)
                expect(other.locator('#desk-count')).to_have_text('1 saved item')
                other.locator('#reading-list').evaluate('(e)=>e.open=true')
                other.locator('.desk-remove').click()
                expect(page.locator('#desk-count')).to_have_text('0 saved items');expect(buttons.first).to_be_enabled()
                other.locator('#desk-undo').click()
                expect(page.locator('#desk-count')).to_have_text('1 saved item');expect(buttons.last).to_be_disabled();other.close()
                # Existing whole-file identity is exported, never the excerpt digest.
                with page.expect_download() as download:
                    page.locator('#desk-export').click()
                exported=json.loads(Path(download.value.path()).read_text())
                assert exported['entries']==json.loads(page.evaluate('(key)=>localStorage.getItem(key)',SCHEMA))['entries']
                row['export_entry']=exported['entries'][0]
                # Source reference receives the exact whole-file source identity.
                page.locator('.desk-cite-source').click()
                expect(page.locator('#reference-repository')).to_have_value(source['repository'].split('/')[1])
                expect(page.locator('#reference-commit')).to_have_value(source['commit'])
                expect(page.locator('#reference-path')).to_have_value(source['path'])
                expect(page.locator('#reference-sha256')).to_have_value('')
                assert json.loads(page.locator('#reference-json').text_content())['verification']=='not_performed'
                page.go_back();expect(select).to_have_value('text');expect(field).to_have_value(query)
                expect(results.locator('.desk-save-source').first).to_be_disabled()
                page.reload();expect(select).to_have_value('text');expect(field).to_have_value(query)
                expect(results.locator('.desk-save-source').last).to_be_disabled()
                # Appended excerpts of this source immediately share saved state.
                field.fill(source['path']);expect(results).to_have_count(10)
                initial=results.first.locator(':scope > a').get_attribute('href')
                page.locator('#library-text-more').focus();page.keyboard.press('Enter')
                expect(results).to_have_count(20);expect(results.nth(10).locator(':scope > a')).to_be_focused()
                expect(results.nth(10).locator('.desk-save-source')).to_be_disabled()
                assert results.first.locator(':scope > a').get_attribute('href')==initial
                # Collection changes retain the existing Files save/citation flow.
                select.select_option('files');expect(page.locator('.current-results > li')).to_have_count(1)
                expect(page.locator('.current-results .desk-save-source')).to_be_disabled()
                select.select_option('text');expect(results).to_have_count(10)
                expect(results.locator('.desk-save-source').first).to_be_disabled()
                page.screenshot(path=str(args.output/(row['case']+'.png')))
                row['passed']=True
            except Exception:
                row['passed']=False;row['error']=traceback.format_exc()
                page.screenshot(path=str(args.output/(row['case']+'-failed.png')))
            finally:context.close()
        context=browser.new_context()
        page=context.new_page();page.set_default_timeout(8000)
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        fixture={'case':'exact-match-controller-negative-controls',
                 'scope':'Isolated engineering controller inputs based on actual source; no admission/acquisition claim.'}
        report['cases'].append(fixture)
        try:
            page.goto(origin+'index.html')
            fixture['results']=page.evaluate("""async()=>{
              const {startResearchDesk}=await import('./research-desk.mjs');
              const data=await fetch('current.json').then(r=>r.json());
              const e=data.equations[0],source=data.sources.find(s=>['repository','commit','path'].every(k=>s[k]===e[k]));
              if(!source)throw Error('Actual excerpt source missing');
              const changed=[e,{...e,commit:'a'.repeat(40)},{...e,path:e.path+'.unmatched'},{...e,repository:'d6g8k5htny-coder/Math-'}];
              const doc=document.implementation.createHTMLDocument('Isolated controller negative controls');
              doc.body.innerHTML='<div id="research-desk"><p id="desk-message"></p><details id="reading-list"><summary>Reading list</summary><ul id="desk-entries"></ul></details><p id="desk-count"></p><p id="desk-empty"></p><input id="desk-import"><button id="desk-undo"></button><button id="desk-export"></button><button id="desk-clear"></button><button id="desk-save-current"></button><button id="desk-save-catalog"></button><a id="library-current-link"></a><input id="search" disabled></div><section id="current-research"><details class="current-source-details"><details><summary>Sources</summary><label>Search<input class="current-search"></label><p class="current-count"></p><ul class="current-results"></ul><button class="current-more">More files</button></details></details></section>';
              startResearchDesk(doc,window,Promise.resolve({...data,sources:[source],equations:changed}));
              await new Promise(resolve=>setTimeout(resolve,30));
              const select=doc.getElementById('library-collection');select.value='text';select.dispatchEvent(new Event('change'));
              const rows=[...doc.querySelectorAll('#library-text-results > li')];
              return rows.map((row,i)=>({index:i,buttons:row.querySelectorAll('.desk-save-source').length,href:row.querySelector(':scope > a').href}));
            }""")
            assert [r['buttons'] for r in fixture['results']]==[1,0,0,0],fixture['results']
            fixture['passed']=True
        except Exception:fixture['passed']=False;fixture['error']=traceback.format_exc()
        finally:context.close()
        # Exercise duplicate inventory through the unchanged hash-bound production
        # admission path. These are synthetic engineering inputs, never sources.
        for variant in ['conflicting', 'identical', 'unsafe-byte-count']:
            context=browser.new_context(viewport={'width':1440,'height':1000})
            page=context.new_page();page.set_default_timeout(8000)
            page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
            duplicate_case={'case':'duplicate-source-'+variant,
                            'scope':'Synthetic inventory through unchanged production hash-bound admission; engineering only.'}
            report['cases'].append(duplicate_case)
            try:
                raw=(args.site/'current.json').read_bytes();data=json.loads(raw)
                excerpt=next(e for e in data['equations'] if 'hess' in (e['text']+' '+e['repository']+' '+e['path']).lower())
                source=next(s for s in data['sources'] if all(s[k]==excerpt[k] for k in ['repository','commit','path']))
                duplicate=dict(source)
                if variant=='conflicting':duplicate.update(blob='a'*40,bytes=source['bytes']+1)
                elif variant=='unsafe-byte-count':duplicate['bytes']=9007199254740992
                data['sources'].append(duplicate)
                mutated=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
                html=(args.site/'workspace.html').read_bytes().replace(sha256(raw).hexdigest().encode(),sha256(mutated).hexdigest().encode()).replace(str(len(raw)).encode(),str(len(mutated)).encode())
                duplicate_case.update(original_source=source,duplicate_source=duplicate,
                                      original_payload_sha256=sha256(raw).hexdigest(),synthetic_payload_sha256=sha256(mutated).hexdigest())
                base_path=urlsplit(origin).path
                def synthetic(route):
                    if not route.request.url.startswith(origin):route.abort();return
                    path=urlsplit(route.request.url).path
                    if path==base_path+'workspace.html':route.fulfill(status=200,content_type='text/html',body=html)
                    elif path==base_path+'current.json':route.fulfill(status=200,content_type='application/json',body=mutated)
                    else:route.continue_()
                context.route('**/*',synthetic)
                page.goto(origin+'workspace.html?source-q=hess&library-collection=text#library-current')
                select=page.locator('#library-collection');field=page.locator('#current-research .current-search')
                expect(select).to_have_value('text');expect(field).to_be_enabled()
                expect(page.locator('.current-state')).to_contain_text('Verified source observation:')
                notice=page.locator('.current-acquisition-notice')
                expect(notice).to_contain_text('invalid_acquisition_identity' if variant=='unsafe-byte-count' else 'duplicate_inventory_identity')
                duplicate_case['acquisition_notice']=notice.text_content()
                expect(page.locator('#library-text-results > li')).to_have_count(2)
                duplicate_case['text_save_controls']=page.locator('#library-text-results .desk-save-source').count()
                expect(page.locator('#library-text-results .desk-save-source'),
                       'Ambiguous whole-source identity must never be offered for saving').to_have_count(0)
                field.fill(source['path']);expect(page.locator('#library-text-results > li')).to_have_count(10)
                page.locator('#library-text-more').click();expect(page.locator('#library-text-results > li')).to_have_count(20)
                expect(page.locator('#library-text-results .desk-save-source')).to_have_count(0)
                select.select_option('files');expect(page.locator('.current-results > li')).to_have_count(2)
                expect(page.locator('.current-results .desk-save-source')).to_have_count(0)
                duplicate_case['files_save_controls']=page.locator('.current-results .desk-save-source').count()
                expect(page.locator('#desk-count')).to_have_text('0 saved items')
                # Ambiguity blocks only this key; a different unique source keeps
                # the existing save/storage identity and remains usable.
                unique=next(s for s in data['sources'] if s['path']!=source['path'])
                field.fill(unique['repository']+' '+unique['path']);expect(page.locator('.current-results > li')).to_have_count(1)
                expect(page.locator('.current-results .desk-save-source')).to_have_count(1)
                page.locator('.current-results .desk-save-source').click()
                expect(page.locator('#desk-count')).to_have_text('1 saved item')
                saved=json.loads(page.evaluate('(key)=>localStorage.getItem(key)',SCHEMA))['entries']
                assert len(saved)==1 and all(saved[0][key]==unique[key] for key in ['repository','commit','path','blob','bytes'])
                duplicate_case['unique_saved_entry']=saved[0]
                duplicate_case['passed']=True
                page.screenshot(path=str(args.output/(duplicate_case['case']+'.png')))
            except Exception:
                duplicate_case['passed']=False;duplicate_case['error']=traceback.format_exc()
                page.screenshot(path=str(args.output/(duplicate_case['case']+'-failed.png')))
            finally:context.close()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=args.browser,chromium_sandbox=True)
        report['browser']=browser.version
        try:
            if args.origin:run(args.origin.rstrip('/')+'/',browser)
            else:
                with serve(args.site) as origin:run(origin,browser)
        finally:browser.close()
    report['passed']=sum(c['passed'] for c in report['cases']);report['failed']=len(report['cases'])-report['passed']
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['browser','passed','failed','page_errors']}))
    return int(bool(report['failed'] or report['page_errors']))

if __name__=='__main__':raise SystemExit(main())
