#!/usr/bin/env python3
"""Real current-first Library/excerpt journeys; engineering verification only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tempfile
import threading
import traceback
from urllib.parse import parse_qs, quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

@contextmanager
def serve(site):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try: yield f'http://127.0.0.1:{server.server_port}/'
    finally: server.shutdown(); server.server_close(); worker.join()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, default=ROOT/'docs/site')
    ap.add_argument('--origin')
    ap.add_argument('--browser', default=shutil.which('chromium'))
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [], 'page_errors': [],
              'source_scope': 'Local driver identity is not independent proof of served bytes.',
              'driver_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'scientific_effect': 'NONE'}
    def run(origin, browser):
        from playwright.sync_api import expect
        report['origin'] = origin
        for width, height, theme in [(320,800,'dark'), (390,844,'light'), (1440,1000,'light')]:
            c = browser.new_context(viewport={'width':width,'height':height},color_scheme=theme)
            c.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
            p = c.new_page(); p.set_default_timeout(10000)
            p.on('pageerror', lambda e: report['page_errors'].append(str(e)))
            row = {'case':'discovery-'+str(width), 'width':width, 'theme':theme}; report['cases'].append(row)
            try:
                p.goto(origin+'index.html'); link=p.locator('a.launch-card').filter(has=p.locator('#card-find'))
                expect(link).to_have_attribute('href','workspace.html#library-current'); link.click()
                field=p.locator('#current-research .current-search'); select=p.locator('#library-collection')
                expect(select).to_be_visible(); expect(field).to_be_enabled()
                raw, metadata = p.evaluate("async()=>await Promise.all([fetch('current.json').then(r=>r.text()),fetch('current-meta.json').then(r=>r.json())])")
                raw_bytes=raw.encode('utf-8'); assert len(raw_bytes)==metadata['bytes']
                assert sha256(raw_bytes).hexdigest()==metadata['sha256']
                admitted=json.loads(raw); row['independently_read_input']={'bytes':len(raw_bytes),'sha256':metadata['sha256'],'excerpts':len(admitted['equations'])}
                p.wait_for_timeout(80); box=field.bounding_box(); row['input_box_after_route']=box
                assert box['y']>=44 and box['y']+box['height']<=height,box
                assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
                field.fill('gradients'); expect(p.locator('.current-results > li')).to_have_count(0)
                select.focus(); select.select_option('text')
                expect(select).to_be_focused(); expect(p.locator('#library-text-results > li')).to_have_count(2)
                expect(p.locator('#library-text-count')).to_contain_text('2 matching excerpts')
                assert p.locator('#desk-save-current').is_hidden()
                text=p.locator('#library-text-results > li').first
                assert 'gradients' in text.inner_text().lower()
                expected=[e for e in admitted['equations'] if 'gradients' in (e['text']+' '+e['repository']+' '+e['path']).lower()]
                assert len(expected)==2
                for index, excerpt in enumerate(expected):
                    actual=p.locator('#library-text-results > li').nth(index)
                    assert sha256(excerpt['text'].encode('utf-8')).hexdigest()==excerpt['sha256']
                    fragment=f"#L{excerpt['line_start']}"+(f"-L{excerpt['line_end']}" if excerpt['line_end']!=excerpt['line_start'] else '')
                    href='https://github.com/'+excerpt['repository']+'/blob/'+excerpt['commit']+'/'+quote(excerpt['path'],safe="/~!*'()")+fragment
                    expect(actual.locator('a').first).to_have_attribute('href',href)
                    assert actual.locator('pre').text_content()==excerpt['text'][:280]+('…' if len(excerpt['text'])>280 else '')
                    identity=actual.locator('details').text_content()
                    assert excerpt['commit'] in identity and excerpt['sha256'] in identity
                href=text.locator('a').first.get_attribute('href')
                row['first_excerpt_href']=href
                share=p.locator('#library-source-link').get_attribute('href'); assert 'library-collection=text' in share
                p.goto(share); expect(p.locator('#library-collection')).to_have_value('text')
                expect(p.locator('#library-text-results > li')).to_have_count(2)
                p.locator('#library-source-clear').click(); expect(field).to_be_focused()
                expect(p.locator('#library-text-results > li')).to_have_count(10)
                p.evaluate("window.__oldTextRows=[...document.querySelectorAll('#library-text-results > li')];window.__oldTextRows[0].querySelector('details').open=true")
                more=p.locator('#library-text-more'); more.focus(); p.keyboard.press('Enter')
                expect(p.locator('#library-text-results > li')).to_have_count(20)
                expect(p.locator('#library-text-results > li').nth(10).locator('a').first).to_be_focused()
                assert p.evaluate("window.__oldTextRows.every((n,i)=>document.querySelectorAll('#library-text-results > li')[i]===n)&&window.__oldTextRows[0].querySelector('details').open")
                outside=p.locator('#library-current-link'); outside.focus()
                more.evaluate('(button)=>button.click()')
                expect(p.locator('#library-text-results > li')).to_have_count(30);expect(outside).to_be_focused()
                assert p.evaluate("window.__oldTextRows.every((n,i)=>document.querySelectorAll('#library-text-results > li')[i]===n)")
                field.fill('<img src=x onerror=alert(1)>'); expect(p.locator('#library-text-results > li')).to_have_count(0)
                assert p.locator('#library-text-results img').count()==0
                expect(p.locator('#library-text-count')).to_contain_text('0 matching excerpts')
                p.locator('#library-collection').select_option('files'); expect(p.locator('#desk-save-current')).to_be_visible()
                field.fill('side24 coefficient'); assert p.locator('.current-results > li').count()>0
                p.screenshot(path=str(a.output/(row['case']+'.png')))
                row['passed']=True
            except Exception: row['passed']=False; row['error']=traceback.format_exc()
            finally: c.close()

            def extra(name, exercise):
                context=browser.new_context(viewport={'width':width,'height':height},color_scheme=theme)
                context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
                page=context.new_page();page.set_default_timeout(10000)
                page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
                result={'case':name+'-'+str(width),'width':width,'theme':theme};report['cases'].append(result)
                try:
                    exercise(page,result);result['passed']=True
                    page.screenshot(path=str(a.output/(result['case']+'.png')))
                except Exception:
                    result['passed']=False;result['error']=traceback.format_exc()
                    try:page.screenshot(path=str(a.output/(result['case']+'-failed.png')))
                    except Exception:pass
                finally:context.close()

            def files_and_saves(page,result):
                page.goto(origin+'workspace.html#library-current')
                field=page.locator('#current-research .current-search');expect(field).to_be_enabled()
                expect(page.locator('#library-collection')).to_have_value('files')
                field.fill('proof');expect(page.locator('.current-results > li')).to_have_count(50)
                identity=page.locator('.current-results > li').filter(has=page.locator('details')).first
                assert identity.count()==1
                identity.locator('summary').first.click();expect(identity.locator('details').first).to_have_attribute('open','')
                saved_href=identity.locator(':scope > a').get_attribute('href')
                page.evaluate("window.__fileRows=[...document.querySelectorAll('.current-results > li')];window.__identity=document.querySelector('.current-results details[open]')")
                more=page.locator('.current-more');more.focus();page.keyboard.press('Enter')
                expect(page.locator('.current-results > li')).to_have_count(100)
                expect(page.locator('.current-results > li').nth(50).locator(':scope > a')).to_be_focused()
                assert page.evaluate("window.__fileRows.every((n,i)=>document.querySelectorAll('.current-results > li')[i]===n)&&window.__identity.open")
                result['retained_file_rows']=50;result['acquisition_disclosure_retained']=True
                identity.locator('.desk-save-source').click();expect(page.locator('#desk-count')).to_have_text('1 saved item')
                page.locator('#desk-save-current').click();expect(page.locator('#desk-count')).to_have_text('2 saved items')
                reading=page.locator('#reading-list')
                if not reading.evaluate('(e)=>e.open'):reading.locator(':scope > summary').click()
                source_row=page.locator('#desk-entries > li').filter(has=page.locator('.desk-cite-source'))
                expect(source_row.locator(':scope > div > a').first).to_have_attribute('href',saved_href)
                search_row=page.locator('#desk-entries > li').filter(has_text='Current source search: proof')
                search_href=search_row.locator(':scope > div > a').first.get_attribute('href')
                parsed=parse_qs(urlsplit(search_href).query);assert parsed['source-q']==['proof'] and 'library-collection' not in parsed
                storage=page.evaluate("localStorage.getItem('universal-law-reading-list/v1')")
                entries=json.loads(storage)['entries'];entry=next(x for x in entries if x['kind']=='source')
                source_row.locator('.desk-cite-source').focus();page.keyboard.press('Enter')
                expect(page.locator('#reference-repository')).to_have_value(entry['repository'].split('/')[1])
                expect(page.locator('#reference-commit')).to_have_value(entry['commit'])
                expect(page.locator('#reference-path')).to_have_value(entry['path'])
                expect(page.locator('#reference-sha256')).to_have_value(entry.get('sha256',''))
                assert json.loads(page.locator('#reference-json').text_content())['verification']=='not_performed'
                page.go_back();expect(page.locator('#desk-count')).to_have_text('2 saved items')
                assert page.evaluate("localStorage.getItem('universal-law-reading-list/v1')")==storage
                if not page.locator('#reading-list').evaluate('(e)=>e.open'):page.locator('#reading-list > summary').click()
                page.locator('#desk-entries > li').filter(has_text='Current source search: proof').locator(':scope > div > a').first.click()
                expect(page.locator('#library-collection')).to_have_value('files');expect(field).to_have_value('proof')
                expect(page.locator('.current-results > li')).to_have_count(50)
                result['source_saved_and_citation_identity_exact']=True;result['legacy_saved_search_reopens_files']=True
            extra('files-append-save-reference',files_and_saves)

            def history_and_focus(page,result):
                keep='literal#note + x&y';query=' Δ + a&b#c?d%20 '
                page.goto(origin+'workspace.html?keep='+quote(keep,safe='')+'#library-current')
                field=page.locator('#current-research .current-search');select=page.locator('#library-collection')
                expect(select).to_be_visible();field.fill(query);select.select_option('text')
                share=page.locator('#library-source-link').get_attribute('href');params=parse_qs(urlsplit(share).query)
                assert params['source-q']==[query] and params['keep']==[keep] and params['library-collection']==['text']
                page.goto(share);expect(field).to_have_value(query);expect(select).to_have_value('text')
                expect(page.locator('#library-text-count')).to_contain_text('0 matching excerpts')
                field.fill('gradients');expect(page.locator('#library-text-results > li')).to_have_count(2)
                first=page.evaluate('location.href');page.evaluate("location.hash='reading-list'")
                field.fill('nu_bar');select.select_option('files');second=page.evaluate('location.href')
                page.go_back();expect(page).to_have_url(first);expect(field).to_have_value('gradients');expect(select).to_have_value('text')
                expect(page.locator('#library-text-results > li')).to_have_count(2)
                page.go_forward();expect(page).to_have_url(second);expect(field).to_have_value('nu_bar');expect(select).to_have_value('files')
                expect(page.locator('.current-results > li')).to_have_count(0)
                field.fill('gradients');select.select_option('text');expect(page.locator('#library-text-results > li')).to_have_count(2)
                page.locator('#library-text-results > li a').first.focus()
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.delete('library-collection');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(select).to_have_value('files');expect(select).to_be_focused()
                field.fill('proof');expect(page.locator('.current-results > li')).to_have_count(50)
                page.locator('.current-results > li > a').first.focus()
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.set('library-collection','text');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(select).to_have_value('text');expect(select).to_be_focused()
                field.fill('gradients');expect(page.locator('#library-text-results > li')).to_have_count(2)
                page.locator('#library-text-results > li a').first.focus()
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.set('source-q','nu_bar');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(field).to_have_value('nu_bar');expect(page.locator('#library-text-results > li')).to_have_count(5)
                expect(select).to_have_value('text');expect(select).to_be_focused()
                select.select_option('files');field.fill('proof')
                expect(page.locator('.current-results > li')).to_have_count(50)
                page.locator('.current-results > li > a').first.focus()
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.set('source-q','coefficients');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(field).to_have_value('coefficients');expect(select).to_have_value('files');expect(select).to_be_focused()
                outside=page.locator('#library-current-link');outside.focus()
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.delete('library-collection');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(select).to_have_value('files');expect(outside).to_be_focused()
                assert parse_qs(urlsplit(page.url).query)['keep']==[keep]
                result['native_history_literal_query_and_unrelated_params']=True
                result['hidden_result_focus_repaired_both_directions']=True;result['outside_focus_retained']=True
                result['same_collection_query_history_focus_repaired']=True
            extra('history-literal-hidden-focus',history_and_focus)

            def focused_more_history(page,result,collection):
                page.goto(origin+'workspace.html#library-current')
                field=page.locator('#current-research .current-search');select=page.locator('#library-collection')
                expect(select).to_be_visible()
                if collection=='text':select.select_option('text')
                field.fill('proof' if collection=='files' else '')
                more=page.locator('.current-more' if collection=='files' else '#library-text-more')
                expect(more).to_be_visible();more.focus();expect(more).to_be_focused()
                page.evaluate("window.__ownedMore=document.activeElement")
                page.evaluate("()=>{const u=new URL(location.href);u.searchParams.set('source-q','gradients');history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}")
                expect(field).to_have_value('gradients');expect(more).to_be_hidden()
                assert page.evaluate('window.__ownedMore.isConnected')
                expect(select).to_be_focused()
                result['previous_more_remained_connected']=True
                result['hidden_more_history_focus_repaired']=True
            extra('files-hidden-more-history',lambda page,result:focused_more_history(page,result,'files'))
            extra('text-hidden-more-history',lambda page,result:focused_more_history(page,result,'text'))

        for javascript in [True,False]:
            context=browser.new_context(viewport={'width':390,'height':844},java_script_enabled=javascript)
            def refused(route):
                if not route.request.url.startswith(origin):route.abort()
                elif javascript and urlsplit(route.request.url).path.endswith('/current.json'):route.fulfill(status=503,body='Unavailable')
                else:route.continue_()
            context.route('**/*',refused)
            page=context.new_page();page.set_default_timeout(10000)
            page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
            row={'case':'source-failure-no-ghost-search' if javascript else 'no-script-static-fallback'};report['cases'].append(row)
            try:
                page.goto(origin+'workspace.html#library-current')
                if javascript:expect(page.locator('.current-state')).to_contain_text('Unavailable')
                expect(page.locator('.current-search')).to_be_disabled()
                assert page.locator('#library-collection').count()==0 and page.locator('#library-text-results').count()==0
                expect(page.locator('#desk-save-current')).to_be_disabled()
                assert page.locator('#library-static-sources').get_attribute('href')=='nodes/index.html#sources'
                row['passed']=True
            except Exception:row['passed']=False;row['error']=traceback.format_exc()
            finally:context.close()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.browser,chromium_sandbox=True); report['browser']=browser.version
        try:
            if a.origin: run(a.origin.rstrip('/')+'/',browser)
            else:
                with serve(a.site) as origin: run(origin,browser)
        finally: browser.close()
    report['passed']=sum(x['passed'] for x in report['cases']); report['failed']=len(report['cases'])-report['passed']
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['browser','passed','failed','page_errors']}))
    return int(bool(report['failed'] or report['page_errors']))
if __name__=='__main__': raise SystemExit(main())
