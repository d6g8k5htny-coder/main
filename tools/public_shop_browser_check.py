"""Run scoped headless Chromium Library and reader source-card checks against docs served on loopback.

Requires tests/browser-requirements.txt and the runner’s packaged Google Chrome.
Screenshots are evidence for inspection, not automatic visual certification.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256, file_digest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import importlib.metadata
import json
import os
import mimetypes
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]

@contextmanager
def local_docs(warm_release=False):
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            requests.append(self.path)
            url=urlsplit(self.path)
            path=Path(self.translate_path(url.path))
            if warm_release and url.path == '/site/predecessor.html':
                data=re.sub(r'[?&]site-release=[0-9a-f]{64}', '', (ROOT/'docs/site/dependencies.html').read_text()).encode()
                self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            if warm_release and not url.query and path.suffix in ('.mjs','.js','.css') and path.is_file():
                # Controlled predecessor reproduces the actual former 30-result cap.
                # No routing interception: Chromium's HTTP cache remains enabled.
                text=re.sub(r'[?&]site-release=[0-9a-f]{64}', '', path.read_text())
                if path.name == 'dependencies.mjs':
                    text=text.replace('searchResults.replaceChildren(...results.map(', 'searchResults.replaceChildren(...results.slice(0,30).map(')
                data=text.encode();self.send_response(200)
                self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/javascript')
                self.send_header('Cache-Control','public, max-age=31536000, immutable')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            super().do_GET()
    server=ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler,directory=str(ROOT/"docs")))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        origin=f"http://127.0.0.1:{server.server_port}/site/"
        yield (origin,requests) if warm_release else origin
    finally:
        server.shutdown();server.server_close();thread.join()


def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_warm_asset_release(browser, expect, result, output):
    with local_docs(warm_release=True) as (origin, requests):
        context=browser.new_context(viewport={"width":1200,"height":900},color_scheme="light")
        page=context.new_page();page.set_default_timeout(45000);errors=[]
        page.on('pageerror', lambda error:errors.append(str(error)))
        try:
            for _ in range(2):
                page.goto(origin+'predecessor.html')
                page.locator('#dependency-search').fill('math')
                expect(page.locator('#search-results > li')).to_have_count(30)
            require(requests.count('/site/dependencies.mjs')==1,'Predecessor was not reused from Chromium HTTP cache')
            page.goto(origin+'dependencies.html')
            page.locator('#dependency-search').fill('math')
            expect(page.locator('#search-results > li')).to_have_count(39)
            for name in ('dependencies.mjs','dependency-model.mjs','dependencies.css','brand.css'):
                require(any(urlsplit(url).path=='/site/'+name and re.fullmatch('[0-9a-f]{64}',parse_qs(urlsplit(url).query).get('site-release',[''])[0]) for url in requests), 'Missing versioned request: '+name)
            require(not errors, 'Warm-release page errors: '+str(errors))
            result['page_errors']=errors
            result['steps'].extend(['controlled predecessor shows 30 entries with HTTP cache enabled', 'second predecessor visit reuses cached module', 'new entry, transitive module and styles select release URLs', 'same browser then renders all 39 entries'])
            shot=output/'warm-asset-release.png';page.screenshot(path=str(shot))
            result['screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
        finally: context.close()


def check_flow(page, origin, expect, result):
    def ready():
        expect(page.locator("#search")).to_be_enabled(timeout=45000)
        expect(page.locator("#counts > article")).to_have_count(3)
        expect(page.locator("#coefficient-state")).to_contain_text("SHA-256 and byte count verified")
        expect(page.locator("#query-note")).to_contain_text("performs no live tip check")

    page.goto(origin+"workspace.html?q=PROOF.md&repository=Math-&path=coefficients%2Fside24_v1%2F&from=research#inventory")
    ready()
    expect(page.locator("#search")).to_have_value("PROOF.md")
    expect(page.locator("#repository-filter")).to_have_value("Math-")
    expect(page.locator("#path-filter")).to_have_value("coefficients/side24_v1/")
    expect(page.locator("#catalog tr")).to_have_count(1)
    expect(page.locator("#more")).to_be_hidden()
    expect(page.locator("#catalog a")).to_contain_text("coefficients/side24_v1/PROOF.md")
    link=page.locator("#catalog-link").get_attribute("href")
    require(set(parse_qs(urlsplit(link).query))=={"q","repository","path"},"Search link has unexpected parameters")
    require(urlsplit(link).fragment=="inventory","Search link lost inventory fragment")
    result["steps"].append("combined bookmark restores exact source; share link omits unrelated parameters")

    page.locator("#search").fill("nothing-matches-this-example")
    expect(page.locator("#catalog tr")).to_have_count(0)
    expect(page.locator("#more")).to_be_hidden()
    expect(page.locator("#inventory-state")).to_contain_text("clear filters")
    page.locator("#search").press("Tab")
    expect(page.locator("#catalog-clear")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator("#catalog-link")).to_be_focused()
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Enter")
    expect(page.locator("#search")).to_be_focused()
    expect(page.locator("#catalog tr")).to_have_count(50)
    expect(page.locator("#more")).to_be_visible()
    page.locator("#more").click()
    expect(page.locator("#catalog tr")).to_have_count(100)
    expect(page.locator("#repository-filter")).to_have_value("")
    expect(page.locator("#path-filter")).to_have_value("")
    result["steps"].append("zero-result recovery, Tab order and keyboard Clear focus")

    page.locator("#search").fill("SIDE24")
    expect(page.locator("#inventory-state")).to_contain_text("96 matches")
    page.locator("#catalog-link").click()
    ready()
    require(parse_qs(urlsplit(page.url).query)=={"q":["SIDE24"]},"Search link did not restore SIDE24 only")
    page.reload();ready()
    expect(page.locator("#search")).to_have_value("SIDE24")
    expect(page.locator("#inventory-state")).to_contain_text("96 matches")
    result["steps"].append("explicit search link and refresh retain results")

    # Real fragment navigation creates a history entry; typing replaces it.
    page.get_by_role("link",name="Recorded status",exact=True).click()
    page.locator("#search").fill("EC-014")
    require(urlsplit(page.url).fragment=="board","Board navigation was overwritten")
    page.go_back()
    expect(page.locator("#search")).to_have_value("SIDE24")
    require(urlsplit(page.url).fragment=="inventory","Back navigation lost inventory fragment")
    page.go_forward()
    expect(page.locator("#search")).to_have_value("EC-014")
    require(urlsplit(page.url).fragment=="board","Board navigation was overwritten")
    result["steps"].append("Back/Forward restore filters and preserve section navigation")

    # A compact, meaningful final view keeps the controls and source row legible.
    page.locator("#repository-filter").select_option("Math-")
    page.locator("#path-filter").fill("coefficients/side24_v1/")
    page.locator("#search").fill("PROOF.md")
    expect(page.locator("#catalog tr")).to_have_count(1)
    expect(page.locator("#more")).to_be_hidden()
    page.locator("#inventory").scroll_into_view_if_needed()
    result["layout"]=page.evaluate("""() => ({viewport: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      tableWidth: document.querySelector('.table-wrap').scrollWidth,
      tableViewport: document.querySelector('.table-wrap').clientWidth})""")
    require(result["layout"]["documentWidth"]<=result["layout"]["viewport"],f"Document overflow: {result['layout']}")
    result["steps"].append("no document overflow; source table measured separately")



def check_latest_work_flow(page, origin, expect, result, output):
    # Static navigation must remain usable if remote source fetching is unavailable.
    remote_requests=[]
    def refuse_remote(route):
        remote_requests.append(route.request.url)
        route.abort()
    page.route("https://raw.githubusercontent.com/**",refuse_remote)
    try:
        page.goto(origin+"index.html")
        entry=page.get_by_role("link",name="Read the latest public work →",exact=True)
        entry.focus();page.keyboard.press("Enter")
        expect(page.locator("#latest-heading")).to_have_text("Latest public work")
        expect(page.locator("#latest-work")).to_be_focused()
        def fragment_focus_indicator(selector):
            style=page.locator(selector).evaluate("el => { const s=getComputedStyle(el); return {width:s.outlineWidth, style:s.outlineStyle, color:s.outlineColor, offset:s.outlineOffset, visible:el.matches(':focus-visible')}; }")
            expected_color="rgb(16, 85, 141)" if result["color_scheme"]=="light" else "rgb(233, 198, 117)"
            require(style=={"width":"3px","style":"solid","color":expected_color,"offset":"4px","visible":True},f"Missing authored fragment focus: {selector}: {style}")
            return style
        result["latest_focus_indicator"]=fragment_focus_indicator("#latest-work")
        expect(page.locator("#latest-work time")).to_have_attribute("datetime","2026-10-03T18:00:00Z")
        expect(page.locator(".latest-chain > li")).to_have_count(3)
        expect(page.locator("#actual-bars")).to_contain_text("P, CAP, E1, E2, REC, CUB and ELDER")
        expect(page.locator("#strict-bar-coefficient")).to_contain_text("integration in progress at this cut")
        expect(page.locator("#actual-bars a[data-source-kind=pinned]")).to_have_count(3)
        expect(page.locator("#strict-bar-coefficient a[data-source-kind=pinned]")).to_have_count(2)
        expect(page.locator(".latest-boundary")).to_contain_text("Conjecture 7 remains open")
        expect(page.locator(".latest-card").first).to_contain_text("Landed at this cut")
        expect(page.locator(".latest-card").first).to_contain_text("I1–I4")
        issue_card=page.locator(".latest-card").nth(1)
        expect(issue_card).to_contain_text("C81–C82 concern the soft model")
        expect(issue_card).to_contain_text("C83–C84 control actual/contact field quantities")
        expect(issue_card.get_by_role("link",name="C84: coupled transverse-sector transfer",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5960630207")
        expect(issue_card).to_contain_text("same weighted observable")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Latest work entry overflow")
        shot=output/f'{result["case"]}-latest-entry.png'
        page.screenshot(path=str(shot))
        result["latest_entry_screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
        full=output/f'{result["case"]}-latest-full.png'
        page.locator("#latest-work").screenshot(path=str(full))
        result["latest_full_screenshot"]={"path":full.name,"sha256":sha256(full.read_bytes()).hexdigest()}
        summary=page.locator(".latest-identities summary")
        summary.focus();page.keyboard.press("Enter")
        expect(page.locator(".latest-identities")).to_have_attribute("open","")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded source identities overflow")
        page.keyboard.press("Enter");page.keyboard.press("Enter");page.keyboard.press("Enter")
        expect(summary).to_be_focused()
        require(page.locator(".latest-identities").get_attribute("open") is None,"Identity disclosure did not close")
        newer=page.get_by_role("link",name="upstream routes below",exact=True)
        newer.focus();page.keyboard.press("Enter")
        expect(page.locator("#newer-work")).to_be_focused()
        result["upstream_focus_indicator"]=fragment_focus_indicator("#newer-work")
        expect(page.locator("#newer-work")).to_contain_text("Mutable upstream navigation")
        page.go_back()
        require(urlsplit(page.url).fragment=="latest-work","Back lost latest-work anchor")
        page.go_forward()
        require(urlsplit(page.url).fragment=="newer-work","Forward lost upstream anchor")
        require(not remote_requests,"Static latest-work reading route unexpectedly fetched a remote source")
        result["latest-source-requests"]=remote_requests
        result["steps"].extend(["Home keyboard route reaches dated latest-work anchor with focus", "landed source packets and issue-only scopes remain visible", "source identities repeatedly open/close without overflow", "upstream navigation and Back/Forward preserve anchors", "static reading path remains usable with remote source requests refused"])
    finally:
        page.unroute("https://raw.githubusercontent.com/**",refuse_remote)


def check_curvature_export_flow(page, origin, expect, result, output):
    page.goto(origin+'explore.html?s=-1&R=1&unrelated=do-not-cite#peaks')
    download=page.get_by_role('button',name='Download curvature SVG',exact=True)
    expect(download).to_be_enabled()
    started=datetime.now(timezone.utc)
    download.focus()
    with page.expect_download() as pending:
        page.keyboard.press('Enter')
    exported=output/f'{result["case"]}-curvature.svg'
    pending.value.save_as(str(exported))
    data=exported.read_bytes()
    require(b'<!DOCTYPE' not in data and b'<!ENTITY' not in data,'SVG includes an external/entity declaration')
    root=ElementTree.fromstring(data)
    ns={'s':'http://www.w3.org/2000/svg'}
    metadata=json.loads(root.find('s:metadata',ns).text)
    require(metadata['schema']=='universal-law/curvature-teaching-figure/v1','Wrong figure metadata schema')
    require(metadata['params']=={'s':-1,'R':1} and metadata['eigenvalues']==[-2,0] and metadata['classification']=='flat','SVG metadata differs from displayed boundary state')
    require(metadata['mode']=='teaching_model' and metadata['verification']=='not_performed','Figure implies research verification')
    source=metadata['source']
    require(source['repository']=='d6g8k5htny-coder/Math-' and source['commit']=='9d7b6802424fb4715b31999066aafca8ee2f3cca' and source['path']=='coefficients/side24_v1/PROOF.md','Figure lost pinned source identity')
    require(source['blob']=='44b66f04f89fcd87383b3603fa69f1feb64cdddd' and source['bytes']==10272 and source['sha256']=='c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769','Figure changed recorded source bytes')
    require(source['url'].startswith('https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/PROOF.md#'),'Figure source URL is mutable or omitted section')
    permalink=urlsplit(metadata['permalink'])
    require(permalink.scheme=='https' and permalink.netloc=='d6g8k5htny-coder.github.io' and permalink.path=='/main/site/explore.html' and parse_qs(permalink.query)=={'s':['-1'],'R':['1']} and permalink.fragment=='peaks','Figure citation copied host or unrelated URL fields')
    generated=datetime.fromisoformat(metadata['generated_at'].replace('Z','+00:00'))
    require(started.timestamp()-2 <= generated.timestamp() <= datetime.now(timezone.utc).timestamp()+2,'Figure generation time is stale or invalid')
    allowed={'svg','title','desc','metadata','rect','polygon','line','text','circle','g'}
    for node in root.iter():
        require(node.tag.startswith('{http://www.w3.org/2000/svg}') and node.tag.rsplit('}',1)[-1] in allowed,'SVG includes unsupported content')
        for key,value in node.attrib.items():
            require(not key.lower().startswith('on') and key.rsplit('}',1)[-1] not in ('href','style','class') and not re.search(r'(?:url|var)\s*\(',value,re.I),'SVG carries executable, external or unresolved presentation')
    markers=[node for node in root.findall('.//s:circle',ns) if float(node.get('r','0'))==9]
    require(len(markers)==1 and float(markers[0].get('cx'))==241 and float(markers[0].get('cy'))==205,'Exported marker differs from the actual boundary geometry')
    require(root.find('s:title',ns) is not None and root.find('s:desc',ns) is not None and root.get('aria-labelledby'),'SVG lost accessible title/description')
    text=' '.join(root.itertext())
    require('teaching' in text.lower() and 'non-certifying' in text and 'Gaussian' in text,'Standalone figure omits teaching limits')
    standalone=page.context.new_page()
    try:
        standalone.set_viewport_size({'width':750,'height':650})
        standalone.goto(exported.resolve().as_uri())
        expect(standalone.locator('svg')).to_be_visible()
        require(standalone.locator('circle').count()==len(markers),'Standalone SVG lost marker')
        clipped=standalone.locator('svg text').evaluate_all("nodes=>nodes.map(node=>({text:node.textContent,box:node.getBBox()})).filter(({box})=>box.x < 0 || box.y < 0 || box.x+box.width > 600 || box.y+box.height > 535).map(({text})=>text)")
        require(not clipped,f'Standalone SVG clips visible labels or provenance: {clipped}')
        shot=output/f'{result["case"]}-curvature-export.png'
        standalone.screenshot(path=str(shot))
        result['curvature_export_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
    finally: standalone.close()
    result['curvature_svg']={'path':exported.name,'sha256':sha256(data).hexdigest(),'metadata':metadata}
    page.context.grant_permissions(['clipboard-read','clipboard-write'],origin=origin.rstrip('/'))
    copy=page.get_by_role('button',name='Copy figure citation',exact=True)
    copy.focus();page.keyboard.press('Enter')
    expect(page.locator('#curvature-export-status')).to_contain_text('Copied figure citation')
    require(page.evaluate('navigator.clipboard.readText()')==page.locator('#curvature-citation-text').input_value(),'Figure citation clipboard differs from displayed text')
    require('unrelated' not in page.locator('#curvature-citation-text').input_value(),'Citation leaked unrelated URL fields')
    page.add_init_script("const mode=sessionStorage.getItem('curvature-clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:mode==='defer'?{writeText:text=>new Promise(resolve=>{window.curvaturePending={text,resolve}})}:undefined})")
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','defer')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks')
    copy.click();expect(copy).to_be_disabled()
    center=page.locator('#curvature-center');center.focus();page.keyboard.press('Home')
    expect(center).to_have_value('-3')
    expect(copy).to_be_disabled()
    page.evaluate('window.curvaturePending.resolve()')
    expect(copy).to_be_enabled()
    require('Copied figure citation' not in page.locator('#curvature-export-status').text_content(),'Old copy claimed success for newly displayed settings')
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','deny')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks');copy.click()
    expect(page.locator('#curvature-export-status')).to_contain_text('manually')
    manual=page.locator('#curvature-figure-citation summary')
    if page.locator('#curvature-figure-citation').get_attribute('open') is None:
        manual.press('Enter')
    expect(page.locator('#curvature-citation-text')).to_be_visible()
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','absent')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks')
    expect(copy).to_be_disabled();expect(download).to_be_enabled()
    require(bool(page.locator('#curvature-citation-text').input_value()),'Manual citation missing without clipboard API')
    page.evaluate("sessionStorage.removeItem('curvature-clipboard-test')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks')
    require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Figure actions overflowed document')
    page.locator('#curvature-figure-citation summary').press('Enter')
    expect(page.locator('#curvature-citation-text')).to_be_visible()
    shot=output/f'{result["case"]}-curvature-export-controls.png'
    page.locator('#peaks').screenshot(path=str(shot))
    result['curvature_controls_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
    result['steps'].append('Actual keyboard SVG download parsed/reopened with exact displayed geometry, concrete presentation and teaching/source metadata; real citation clipboard, denial/absence and pending settings change checked')


def check_source_card_flow(page, origin, expect, result, output):
    check_latest_work_flow(page,origin,expect,result,output)
    result["reader_entry_screenshots"]=[]
    for entry in ["index","explore","cite","reproduce","formal"]:
        page.goto(origin+entry+".html")
        expect(page.locator("h1")).to_have_count(1)
        if entry == "formal":
            jump=page.get_by_role('link',name='Inspect the exact coverage and its limits ↓',exact=True)
            jump.focus();page.keyboard.press('Enter')
            expect(page.locator('#formal-coverage')).to_be_focused()
            require(page.locator('#formal-coverage').evaluate('(el)=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),'Formal coverage jump lacks authored focus outline')
            summary=page.locator('#formal-coverage-details summary')
            summary.focus();page.keyboard.press('Enter')
            expect(page.locator('#formal-coverage-details')).to_have_attribute('open','')
            expect(page.locator('#formal-coverage-table tbody tr')).to_have_count(9)
            expect(page.locator('#formal-coverage-table tbody code')).to_have_count(13)
            expect(page.locator('#formal-coverage')).to_contain_text('not a current formalization inventory')
            expect(page.locator('.boundary')).to_contain_text('PENDING')
            region=page.get_by_role('region',name='Exact scalar companion scope table',exact=True)
            region.focus();page.keyboard.press('ArrowRight')
            require(region.evaluate('(el)=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),'Formal table region lacks authored focus outline')
            page.wait_for_function("()=>{const el=document.querySelector('#formal-coverage .table-wrap');return el.scrollWidth<=el.clientWidth||el.scrollLeft>0}")
            require(region.evaluate('(el)=>el.scrollWidth <= el.clientWidth || el.scrollLeft > 0'),'Overflowing scope table did not keyboard-scroll')
            require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Formal scope overflowed document')
            shot=output/f'{result["case"]}-formal-coverage.png'
            page.locator('#formal-coverage').screenshot(path=str(shot))
            result['formal_coverage_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
            summary.focus();page.keyboard.press('Enter')
            require(page.locator('#formal-coverage-details').get_attribute('open') is None,'Formal scope did not close')
            expect(summary).to_be_focused()
            static=page.context.browser.new_context(java_script_enabled=False)
            try:
                fallback=static.new_page();fallback.goto(origin+'formal.html#formal-coverage')
                fallback.locator('#formal-coverage-details summary').press('Enter')
                expect(fallback.locator('#formal-coverage-details')).to_have_attribute('open','')
                expect(fallback.locator('#formal-coverage-table tbody tr')).to_have_count(9)
                expect(fallback.locator('#formal-coverage-table')).to_be_visible()
            finally:
                static.close()
            result['steps'].append('Formal snapshot exposes nine verbatim scope rows/13 declarations; keyboard disclosure, table scroll, close focus, no document overflow and JavaScript-off access checked')
        if entry == "explore":
            page.goto(origin+"explore.html?s=-1&R=1&r=0.25&region=remote&objects=#peaks")
            center=page.locator("#curvature-center")
            expect(center).to_be_enabled();expect(center).to_have_value("-1")
            expect(page.locator("#eigenvalue-second")).to_have_text("0.0")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            expect(page.locator("#pin-gap")).to_have_text("0.015625")
            expect(page.locator('input[name="region"][value="remote"]')).to_be_checked()
            expect(page.locator("#palette-kind")).to_contain_text("Empty groups")
            center.focus();page.keyboard.press("Home")
            expect(center).to_have_value("-3")
            require("s=-3" in page.url,"Keyboard change missing from URL")
            page.go_back();expect(center).to_have_value("-1")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            page.go_forward();expect(center).to_have_value("-3")
            page.get_by_role("button",name="Reset curvatures",exact=True).click()
            expect(center).to_have_value("-2")
            page.get_by_role("button",name="Compare half distance",exact=True).click()
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            page.locator('#object-three').check()
            shared=page.locator("#explore-state-link").get_attribute("href")
            require("objects=3" in shared and "r=0.5" in shared,"Share link missed controls")
            require(shared.endswith("#peaks"),"Share link lost section")
            page.goto(shared)
            expect(center).to_have_value("-2")
            expect(page.locator('#object-three')).to_be_checked()
            expect(page.locator('#object-one')).not_to_be_checked()
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            page.get_by_role("button",name="Restore all three",exact=True).click()
            expect(page.locator("#palette-kind")).to_have_text("One object has no place")
            page.goto(origin+"explore.html?s=NaN&R=2&r=0&region=other&objects=1,1")
            expect(page.locator("#explore-state-status")).to_contain_text("invalid")
            expect(center).to_have_value("-2")
            expect(page.locator("#curvature-spread")).to_have_value("2")
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            expect(page.locator("#palette-kind")).to_have_text("One object has no place")
            # Static readers get an explicit boundary instead of a false restored state.
            static=page.context.browser.new_context(java_script_enabled=False)
            try:
                fallback=static.new_page()
                fallback.goto(shared)
                expect(fallback.locator("#curvature-center")).to_be_disabled()
                expect(fallback.locator("#explore-state-link")).to_be_hidden()
                expect(fallback.locator("#explore-state-status")).to_contain_text("starting examples")
                expect(fallback.locator('#curvature-export-svg')).to_be_disabled()
                expect(fallback.locator('#curvature-copy-citation')).to_be_disabled()
            finally:
                static.close()
            page.goto(origin+"explore.html?s=-1&R=1&r=0.25&region=remote&objects=1,3#peaks")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            page.locator("#explore-state-link").scroll_into_view_if_needed()
            result["steps"].append("Explore URLs restore all controls and derived explanations; keyboard edits, reset, reload, Back/Forward, malformed fields and JavaScript-off fallback checked")
            check_curvature_export_flow(page,origin,expect,result,output)
        if entry == "cite":
            commit="a414e77d6278a5d9ce6aa6e2bdca6146b048f27a"
            digest=sha256((ROOT/"CITATION.cff").read_bytes()).hexdigest()
            page.locator("#reference-commit").fill(commit)
            page.locator("#reference-path").fill("CITATION.cff")
            page.locator("#reference-sha256").fill(digest)
            page.get_by_role("button",name="Build reference",exact=True).click()
            expect(page.locator("#reference-link")).to_have_attribute("href",f"https://github.com/d6g8k5htny-coder/main/blob/{commit}/CITATION.cff")
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            copy=page.get_by_role("button",name="Copy source reference",exact=True)
            copy.focus();page.keyboard.press("Enter")
            expect(page.locator("#reference-status")).to_contain_text("Copied source reference")
            require(page.evaluate("navigator.clipboard.readText()") == page.locator("#reference-output").text_content(),"Source clipboard differs from visible reference")
            page.get_by_role("button",name="Copy JSON",exact=True).click()
            expect(page.locator("#reference-status")).to_contain_text("Copied JSON")
            record=json.loads(page.evaluate("navigator.clipboard.readText()"))
            require(record["commit"]==commit and record["sha256"]==digest and record["verification"]=="not_performed","JSON reference lost source or verification boundary")
            shared=page.locator("#reference-share").get_attribute("href")
            page.locator("#reference-path").fill("README.md")
            expect(page.locator("#reference-actions")).to_be_hidden()
            expect(page.locator("#reference-output")).to_be_empty()
            page.locator("#reference-sha256").fill("")
            page.get_by_role("button",name="Build reference",exact=True).click()
            page.go_back();expect(page.locator("#reference-path")).to_have_value("CITATION.cff")
            expect(page.locator("#reference-sha256")).to_have_value(digest)
            page.go_forward();expect(page.locator("#reference-path")).to_have_value("README.md")
            expect(page.locator("#reference-sha256")).to_have_value("")
            page.goto(shared);expect(page.locator("#reference-path")).to_have_value("CITATION.cff")
            expect(page.locator("#reference-output")).to_contain_text(digest)
            page.goto(origin+"cite.html?repo=main&commit=main#reference-builder")
            expect(page.locator("#reference-status")).to_contain_text("Shared reference unavailable")
            expect(page.locator("#reference-output")).to_be_empty()
            expect(page.locator("#reference-actions")).to_be_hidden()
            page.goto(origin+f"cite.html?repo=main&commit={commit}&path=%FF#reference-builder")
            expect(page.locator("#reference-status")).to_contain_text("malformed URL encoding")
            expect(page.locator("#reference-output")).to_be_empty()
            page.add_init_script("const mode=sessionStorage.getItem('reference-clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:mode==='defer'?{writeText:text=>new Promise(resolve=>{window.referencePending={text,resolve}})}:undefined})")
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','defer')")
            page.goto(shared);copy.click()
            expect(copy).to_be_disabled()
            page.locator("#reference-path").fill("README.md")
            page.locator("#reference-sha256").fill("")
            page.get_by_role("button",name="Build reference",exact=True).click()
            expect(page.get_by_role("button",name="Copy JSON",exact=True)).to_be_disabled()
            page.evaluate("window.referencePending.resolve()")
            expect(copy).to_be_enabled()
            expect(page.locator("#reference-status")).to_contain_text("Previous copy finished")
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','deny')")
            page.goto(shared);copy.click()
            expect(page.locator("#reference-status")).to_contain_text("copy it manually")
            expect(copy).to_be_enabled()
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','unsupported')")
            page.reload();expect(copy).to_be_disabled()
            expect(page.locator("#reference-output")).to_contain_text(digest)
            page.evaluate("sessionStorage.removeItem('reference-clipboard-test')")
            page.reload();expect(copy).to_be_enabled()
            page.get_by_text("Inspect structured source identity (JSON)",exact=True).click()
            expect(page.locator("#reference-json")).to_be_visible()
            page.locator("#reference-actions").scroll_into_view_if_needed()
            static=page.context.browser.new_context(java_script_enabled=False)
            try:
                fallback=static.new_page();fallback.goto(shared)
                expect(fallback.locator("#reference-builder")).to_be_hidden()
                expect(fallback.get_by_role("heading",name="Build a source link without JavaScript")).to_be_visible()
            finally:
                static.close()
            result["steps"].append("Cite keyboard copies exact text/JSON with supplied digest; edit invalidation, share/reload, Back/Forward, invalid links, clipboard refusal and JavaScript-off fallback checked")
        if entry == "reproduce":
            recorded=[]
            for row in page.locator('#coefficient-enclosures tbody tr').all():
                recorded.append({'dimension':row.locator('th').inner_text(),'lower':row.locator('code').nth(0).inner_text(),'upper':row.locator('code').nth(1).inner_text()})
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            for format in ['text','json','latex']:
                button=page.locator('#copy-coefficients-'+format)
                button.focus();page.keyboard.press('Enter')
                expect(page.locator('#copy-coefficients-status')).to_contain_text('Copied')
                copied=page.evaluate('navigator.clipboard.readText()')
                require(copied==page.locator('#coefficient-export-'+format).text_content(),'Coefficient clipboard differs from recorded export')
                for row in recorded:
                    require(row['lower'] in copied and row['upper'] in copied,'Decimal enclosure lost precision')
                    if format=='text':
                        require(f"{row['lower']} < c_{row['dimension']},24 < {row['upper']}" in copied,'Text export weakened the strict SIDE24 statement')
                    if format=='latex':
                        require('\\['+row['lower']+' < c_{'+row['dimension']+',24} < '+row['upper']+'\\]' in copied,'LaTeX export lacks strict display math')
                if format=='json':
                    record=json.loads(copied)
                    require(record['intervals']==recorded and record['execution']=='not_performed' and record['source_state']=='historical_pinned','Coefficient export lost identity or scope')
                    require(record['bounds']=='strict' and record['coefficient']=='c_{d,24}','JSON export lost strict SIDE24 identity')
            page.get_by_text('Inspect or manually copy interval exports',exact=True).click()
            expect(page.locator('#coefficient-export-json')).to_be_visible()
            require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Expanded interval export overflow')
            shot=output/f'{result["case"]}-coefficient-exports.png'
            page.locator('#coefficient-exports').screenshot(path=str(shot))
            result['coefficient_export_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
            result['steps'].append('Exact historical coefficient endpoints copied as text/JSON/LaTeX with pinned source and no execution claim')
            commands=page.locator("#reproduction-commands").inner_text()
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            copy=page.get_by_role("button",name="Copy all commands",exact=True)
            expect(copy).to_be_enabled();copy.focus();page.keyboard.press("Enter")
            expect(page.locator("#copy-commands-status")).to_contain_text("Copied all commands")
            require(page.evaluate("navigator.clipboard.readText()") == commands,"Clipboard differs from visible pinned commands")
            expect(page.locator("#copy-commands-status")).to_contain_text("not run")
            page.add_init_script("const mode=sessionStorage.getItem('clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:undefined})")
            page.evaluate("sessionStorage.setItem('clipboard-test','deny')")
            page.reload();copy.click()
            expect(page.locator("#copy-commands-status")).to_contain_text("Select the command block")
            expect(copy).to_be_enabled()
            page.locator('#copy-coefficients-json').click()
            expect(page.locator('#copy-coefficients-status')).to_contain_text('manually')
            expect(page.locator('#copy-coefficients-json')).to_be_enabled()
            expect(page.locator('#coefficient-export-json')).to_have_text(json.dumps(record,indent=2))
            page.evaluate("sessionStorage.setItem('clipboard-test','unsupported')")
            page.reload();expect(copy).to_be_disabled()
            expect(page.locator("#reproduction-commands")).to_have_text(commands)
            expect(page.locator("#copy-commands-status")).to_contain_text("Select the command block")
            expect(page.locator('#copy-coefficients-json')).to_be_disabled()
            expect(page.locator('#copy-coefficients-status')).to_contain_text('manually')
            page.evaluate("sessionStorage.removeItem('clipboard-test')")
            result["steps"].append("Reproduction keyboard copy matches exact visible commands; clipboard refusal and unsupported API preserve manual fallback without execution")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),f"Reader entry overflow: {entry}")
        shot=output/f'{result["case"]}-{entry}.png'
        page.screenshot(path=str(shot))
        result["reader_entry_screenshots"].append({"page":entry,"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    result["steps"].append("Home, Explore, Cite, Reproduce and Formal entries have one main heading and no document overflow")
    # Enter through the public reading path, not a fabricated application state.
    page.goto(origin+"research.html")
    page.get_by_role("link",name="Inspect its review and exact scope",exact=True).click()
    card=page.locator("#d2-lifetime-remainder")
    expect(card).to_be_visible(timeout=45000)
    expect(page.locator("#museum-state")).to_contain_text("displayed source bytes verified",timeout=45000)
    expect(card).to_be_focused()
    readable=card.locator(".source-quote-readable")
    expect(readable.get_by_role("link",name="R1–R4 review",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276")
    expect(readable.get_by_role("link",name="R5/R6 delta review",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206")
    expect(readable.locator("code")).to_have_text("O(1)")
    expect(readable).to_contain_text("Numerical constants/radii, a second coefficient, and RN/24-jet closure are outside this verdict.")
    original=card.locator(".source-quote-original")
    expect(original).to_be_hidden()
    # Every quoted character remains available through a native keyboard control.
    summary=card.locator("summary").filter(has_text="Exact source quote")
    summary.focus();page.keyboard.press("Enter")
    expect(summary).to_be_focused();expect(original).to_be_visible()
    manifest=json.loads((ROOT/"docs/site/museum.json").read_text())
    require(original.text_content()==manifest["claims"][0]["scope_quote"],"Original quoted source changed")
    page.keyboard.press("Enter");expect(original).to_be_hidden()
    page.keyboard.press("Enter");expect(original).to_be_visible()
    page.keyboard.press("Enter");expect(original).to_be_hidden()
    expect(card.locator(".object-class")).to_have_text("ACCEPT-scoped")
    expect(card).to_contain_text("Hashes, browser controls and replay output do not change status.")
    card.scroll_into_view_if_needed()
    result["layout"]=page.evaluate("""() => ({viewport: innerWidth, documentWidth: document.documentElement.scrollWidth})""")
    require(result["layout"]["documentWidth"]<=result["layout"]["viewport"],f"Source card overflow: {result['layout']}")
    result["steps"].extend(["Research route reaches verified D2 card with focus", "both exact review comments are interactive and qualifiers preserved", "keyboard disclosure repeated open/close preserves exact original quote", "source class and acceptance limits retained; no document overflow"])


def check_reader_tools_flow(page, origin, expect, result, output):
    result["screenshots"]=[]
    page.goto(origin+"research.html")
    measure=page.get_by_role("link",name="synthetic counts-to-density guide",exact=True)
    expect(measure).to_have_attribute("href","measure.html")
    measure.click()
    expect(page.locator("h1")).to_have_text("From counts to density.")
    expect(page.locator("#measure-controls")).to_be_enabled()
    expect(page.locator("#measure-mass")).to_have_text("0.5")
    expect(page.locator("#measure-density")).to_have_text("0.153846")
    source=page.locator("#measure-source a")
    source_href=source.get_attribute("href")
    expected_source="https://github.com/d6g8k5htny-coder/main/blob/e60629edde151c27541847b61672e38965752b77/docs/research-translation/20260930/EXPERIMENT.md#1-freeze-the-observable-before-generating-data"
    require(source_href==expected_source,"Measurement source identity drifted")
    page.locator("#measure-count").fill("1152")
    expect(page.locator("#measure-mass")).to_have_text("1")
    expect(page.locator("#measure-density")).to_have_text("0.307692")
    page.locator("#measure-lower").fill("0")
    expect(page.locator("#measure-error")).to_contain_text("greater than zero")
    expect(page.locator("#measure-lower")).to_have_attribute("aria-invalid","true")
    page.locator("#measure-reset").click()
    expect(page.locator("#measure-mass")).to_have_text("0.5")
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Measurement guide overflow")
    shot=output/f'{result["case"]}-measure.png';page.screenshot(path=str(shot),full_page=True)
    result["screenshots"].append({"page":"measure","path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    result["steps"].append("Research route opens the synthetic measurement guide; normalization, invalid edge and exact source cut checked")

    page.goto(origin+"research.html")
    dependencies=page.get_by_role("link",name="dated claim-dependency snapshot",exact=True)
    expect(dependencies).to_have_attribute("href","dependencies.html")
    dependencies.click()
    expect(page.locator("#load-status")).to_contain_text("Pinned Math commit 7858329974e2",timeout=45000)
    expect(page.locator("#node-count")).to_have_text("49")
    expect(page.locator("#edge-count")).to_have_text("55")
    expect(page.locator("#unresolved-count")).to_have_text("15")
    expect(page.locator(".boundary")).to_contain_text("Dated read-only snapshot")
    expect(page.get_by_role("link",name="Provenance record",exact=True)).to_have_attribute("href","dependency-source/PROVENANCE.json")
    page.locator("#dependency-search").fill("math")
    expect(page.locator("#search-results > li")).to_have_count(39)
    expect(page.locator("#search-results")).to_contain_text("math.side24-coefficient")
    expect(page.locator("#search-results")).to_contain_text("math.uniform-matrix-cap-lifetime")
    result["steps"].append("broad dependency search exposes all 39 matching nodes")
    page.locator("#classification-filter").select_option("PROVED_REVIEWED")
    require(0 < page.locator("#search-results > li").count() < 39,"Classification must intersect search")
    require(parse_qs(urlsplit(page.url).query)=={"q":["math"],"classification":["PROVED_REVIEWED"]},"Filters missing from saved URL")
    page.reload()
    expect(page.locator("#classification-filter")).to_have_value("PROVED_REVIEWED",timeout=45000)
    expect(page.locator("#dependency-search")).to_have_value("math")
    page.locator("#clear-filters").click()
    expect(page.locator("#search-results > li")).to_have_count(49)
    page.locator("#dependency-search").fill("reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md")
    expect(page.locator("#search-results > li")).to_have_count(1)
    result_button=page.locator("#search-results button")
    page.keyboard.press("Tab")
    expect(page.locator("#classification-filter")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator("#clear-filters")).to_be_focused()
    page.keyboard.press("Tab")
    expect(result_button).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.locator("#detail-heading")).to_have_text("math.rn-fixed-annulus-window")
    expect(page.locator("#node-detail")).to_be_focused()
    expect(page.locator("#object-scope")).to_contain_text("compact positive gaps")
    expect(page.locator("#object-notes")).to_contain_text("No global D5/RN/JETMOD")
    expect(page.locator("#evidence-body")).to_contain_text("Record linked")
    expect(page.locator("#evidence-body")).to_contain_text("Not recorded")
    page.locator("#object-audit summary").click()
    expect(page.locator("#node-metadata")).to_contain_text("Review source")
    expect(page.locator("#node-metadata")).to_contain_text("reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md")
    expect(page.locator("#dependency-paths > li")).to_have_count(1)
    expect(page.locator("#dependency-paths > li")).to_contain_text("math.rn-count-interface")
    expect(page.locator("#dependency-paths")).not_to_contain_text("hist.CH-LIFT")
    page.locator("#dependency-search").fill("math.lifetime-remainder")
    remainder_result=page.locator(
        '#search-results button:has(strong:text-is("math.lifetime-remainder"))'
    )
    expect(remainder_result).to_have_count(1)
    remainder_result.click()
    expected_remainder_source="https://github.com/d6g8k5htny-coder/Math-/blob/7858329974e28be79f29b22644370084ff43da4f/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md"
    expected_remainder_review="https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206"
    expect(page.locator(f'#node-metadata a[href="{expected_remainder_source}"]')).to_have_count(1)
    expect(page.locator(f'#node-metadata a[href="{expected_remainder_review}"]')).to_have_count(1)
    result["steps"].append("required dependency path and actionable source/review metadata checked")
    page.goto(origin+"dependencies.html?node=math.rn-fixed-annulus-window#node-detail")
    require(parse_qs(urlsplit(page.url).query)=={"node":["math.rn-fixed-annulus-window"]},"Saved dependency selection missing")
    page.reload()
    expect(page.locator("#detail-heading")).to_have_text("math.rn-fixed-annulus-window",timeout=45000)
    expect(page.locator("#node-detail")).to_be_focused()
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Dependency viewer overflow")
    shot=output/f'{result["case"]}-dependencies.png';page.screenshot(path=str(shot),full_page=True)
    result["screenshots"].append({"page":"dependencies","path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    for fragment in ("object-evidence", "object-audit"):
        page.goto(origin+"dependencies.html?node=math.rn-fixed-annulus-window#"+fragment)
        expect(page.locator("#"+fragment)).to_be_focused(timeout=45000)
        page.reload()
        expect(page.locator("#"+fragment)).to_be_focused(timeout=45000)
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Object deep-link overflow")
    page.locator("#object-audit summary").click()
    expect(page.locator("#node-record")).to_contain_text('"scientific_status_unchanged": true')
    page.goto(origin+"dependencies.html?node=missing.node#node-detail")
    expect(page.locator("#detail-heading")).to_have_text("Saved selection unavailable",timeout=45000)
    expect(page.locator("#selection-error")).to_contain_text("does not match this source snapshot")
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Dependency refusal overflow")
    result["steps"].append("Pinned 49/55/15 graph, review-source search, post-layout saved-link focus, invalid-ID refusal and narrow overflow checked")
    page.goto(origin+"dependencies.html?classification=ACCEPT_ALL")
    expect(page.locator("#filter-error")).to_contain_text("Unknown classification",timeout=45000)
    expect(page.locator("#search-results > li")).to_have_count(0)
    page.locator("#clear-filters").click()
    expect(page.locator("#filter-error")).to_be_hidden()
    expect(page.locator("#search-results > li")).to_have_count(49)
    result["steps"].append("Saved exact-classification filters, all-records recovery, scope/evidence boundaries and native audit disclosure checked")


def check_visitor_recovery(page, origin, expect, result):
    if result["case"] == "query-refusal":
        config=json.loads((ROOT/"docs/site/config.json").read_text())
        hits=[]
        def refuse_query(route):
            hits.append(route.request.url)
            route.fulfill(status=503,headers={"Access-Control-Allow-Origin":"*"},body="")
        page.route(config["query"]["url"],refuse_query)
        page.goto(origin+"workspace.html#board")
        expect(page.locator("#query-note")).to_contain_text("Source unavailable (503). No result inferred.",timeout=45000)
        expect(page.locator("#custody-note")).to_contain_text("9 byte-copy imports landed",timeout=45000)
        expect(page.locator("#query-identity")).to_be_empty()
        require(len(hits)==1,"Expected exactly one refused query request")
        result["steps"].append("query 503 ends loading without erasing verified import information")
    else:
        manifest=json.loads((ROOT/"docs/site/museum.json").read_text())
        held=[]
        page.route(manifest["claims"][0]["proof"]["url"],lambda route:held.append(route))
        page.goto(origin+"museum.html#d3-side24-coefficient",wait_until="domcontentloaded")
        expect(page.locator("#claim-cards")).to_contain_text("Verifying",timeout=45000)
        deadline=time.monotonic()+45
        while not held and time.monotonic()<deadline:
            page.wait_for_timeout(10)  # Pump Playwright events until the route callback ran.
        require(len(held)==1,"The delayed source was not intercepted before reader interaction")
        page.get_by_role("link",name="Reproduce",exact=True).first.focus()
        page.keyboard.press("Tab")
        focused=page.get_by_role("link",name="Cite",exact=True).first
        expect(focused).to_be_focused()
        held[0].fulfill(response=held[0].fetch())
        expect(page.locator("#museum-state")).to_contain_text("displayed source bytes verified",timeout=45000)
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        expect(focused).to_be_focused()
        require(page.evaluate("Math.abs(document.querySelector('#d3-side24-coefficient').getBoundingClientRect().top)>100"),"Delayed claim completion pulled the reader back")
        result["steps"].append("keyboard navigation during museum verification keeps focus and reading position")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    report={"scientific_effect":"NONE","browser_run_started":False,"cases":[],"limitations":["Headless Chromium only; mobile viewport emulation, not physical devices","Screenshots require inspection; not full accessibility certification","Local checkout preview, not deployed Pages evidence"]}
    try:
        report["checked_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        if os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"]!=report["checked_commit"]:
            raise ValueError("Runner commit differs from GITHUB_SHA")
        report["run_id"]=os.environ.get("GITHUB_RUN_ID")
        report["run_attempt"]=os.environ.get("GITHUB_RUN_ATTEMPT")
        report["pr_head"]=os.environ.get("BROWSER_PR_HEAD")
        report["tree"]=subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=ROOT,text=True).strip()
        report["playwright"]=importlib.metadata.version("playwright")
        from playwright.sync_api import sync_playwright, expect
        with local_docs() as origin, sync_playwright() as playwright:
            executable=Path("/opt/google/chrome/chrome")
            require(executable.is_file(),"The runner’s packaged Chrome binary is missing")
            report["browser_channel"]="chrome"
            report["browser_executable"]=str(executable)
            with executable.open("rb") as binary:
                report["browser_executable_sha256"]=file_digest(binary,"sha256").hexdigest()
            report["runner_image_os"]=os.environ.get("ImageOS")
            report["runner_image_version"]=os.environ.get("ImageVersion")
            browser=playwright.chromium.launch(channel="chrome",chromium_sandbox=True)
            try:
                report["browser_run_started"]=True;report["browser"]=browser.version
                warm={"case":"warm-asset-release","steps":[],"passed":False};report["cases"].append(warm)
                try:
                    check_warm_asset_release(browser,expect,warm,output);warm["passed"]=True
                except Exception: warm["error"]=traceback.format_exc()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_flow(page,origin,expect,result)
                            require(not errors,f"Page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors
                            shot=output/f"{label}.png"
                            try:
                                page.locator("#inventory").screenshot(path=str(shot),timeout=10000)
                                result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                            except Exception:
                                result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                            context.close()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'source-{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_source_card_flow(page,origin,expect,result,output)
                            require(not errors,f"Source page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors;shot=output/f"{label}.png"
                            try:
                                page.locator("#d2-lifetime-remainder").screenshot(path=str(shot),timeout=10000)
                                result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                            except Exception:
                                result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                            context.close()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'reader-tools-{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_reader_tools_flow(page,origin,expect,result,output)
                            require(not errors,f"Reader tool page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors
                            context.close()
                for label in ["query-refusal","museum-reader-intent"]:
                    result={"case":label,"steps":[],"passed":False};report["cases"].append(result)
                    context=browser.new_context(viewport={"width":390,"height":844},color_scheme="light",reduced_motion="reduce")
                    page=context.new_page();page.set_default_timeout(15000);errors=[]
                    page.on("pageerror",lambda error:errors.append(str(error)))
                    try:
                        check_visitor_recovery(page,origin,expect,result)
                        require(not errors,f"Visitor recovery page errors: {errors}")
                        result["passed"]=True
                    except Exception:
                        result["error"]=traceback.format_exc()
                    finally:
                        result["page_errors"]=errors;shot=output/f"{label}.png"
                        try:
                            page.screenshot(path=str(shot))
                            result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                        except Exception:
                            result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                        context.close()
                result={"case":"inventory-refusal","steps":[],"passed":False};report["cases"].append(result)
                context=browser.new_context(viewport={"width":390,"height":844},color_scheme="dark")
                page=context.new_page();refusal_errors=[];route_hits=[]
                page.on("pageerror",lambda error:refusal_errors.append(str(error)))
                try:
                    config=json.loads((ROOT/"docs/site/config.json").read_text())
                    def refuse_inventory(route):
                        route_hits.append(route.request.url)
                        route.fulfill(status=503,headers={"Access-Control-Allow-Origin":"*"},body="")
                    page.route(config["inventory"]["url"],refuse_inventory)
                    page.goto(origin+"workspace.html?q=SIDE24#inventory")
                    expect(page.locator("#inventory-state")).to_contain_text("Source unavailable (503). No result inferred.",timeout=45000)
                    require(len(route_hits)==1,"Expected exactly one refused inventory request")
                    for selector in ["#search","#repository-filter","#path-filter","#catalog-clear"]:
                        expect(page.locator(selector)).to_be_disabled()
                    require(page.locator("#catalog-link").get_attribute("href") is None,"Unavailable catalog has an active search link")
                    expect(page.locator("#catalog tr")).to_have_count(0)
                    expect(page.locator("#more")).to_be_hidden()
                    require(not refusal_errors,f"Refusal page errors: {refusal_errors}")
                    result["steps"].append("pinned inventory 503 keeps filters/link unavailable and no rows")
                    result["passed"]=True
                except Exception:
                    result["error"]=traceback.format_exc()
                finally:
                    result["page_errors"]=refusal_errors;result["refused_requests"]=route_hits
                    shot=output/"inventory-refusal.png"
                    try:
                        page.locator("#inventory").screenshot(path=str(shot),timeout=10000)
                        result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                    except Exception:
                        result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                    context.close()
            finally:
                browser.close()
        report["passed"]=len(report["cases"])==16 and all(case["passed"] for case in report["cases"])
    except Exception:
        report["passed"]=False;report["error"]=traceback.format_exc()
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
