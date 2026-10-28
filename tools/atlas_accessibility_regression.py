"""Actual pointer/keyboard regressions for the scoped Atlas accessibility fixes.

This is not a conformance checker. It uses the unchanged application and CSP;
only the explicitly named source-fetch failure is injected.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright,expect
from atlas_formal_evidence_browser_check import identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();site=args.site.resolve();args.output.mkdir(parents=True,exist_ok=False)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*unused):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(site)))
    threading.Thread(target=server.serve_forever,daemon=True).start();origin=f'http://127.0.0.1:{server.server_port}'
    report={'scope':'Rendered pointer/keyboard/semantic regressions; no fullWCAG or spokenAT certification.',
            'driver':identity(Path(__file__)),'payload':identity(site/'current.json'),
            'served_files':{str(x.relative_to(site)):identity(x) for x in sorted(site.rglob('*')) if x.is_file()},'cases':[]}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path='/usr/bin/chromium',chromium_sandbox=True,
                                      args=['--enable-unsafe-swiftshader','--disable-background-networking'])
            report['browser']=browser.version
            for width,scheme in [(320,'light'),(320,'dark'),(390,'light'),(390,'dark')]:
                row={'width':width,'scheme':scheme,'checks':[],'failures':[]};report['cases'].append(row)
                def check(name,ok,detail=None):
                    row['checks'].append({'name':name,'passed':bool(ok),'detail':detail})
                    if not ok:row['failures'].append(name)
                context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce')
                page=context.new_page()
                try:
                    page.goto(origin+'/dependencies.html?atlas-view=graph')
                    expect(page.locator('#atlas-view')).to_be_enabled()
                    page.locator('footer').scroll_into_view_if_needed()
                    geometry=page.locator('footer').evaluate('''footer=>{
                      const links=[...footer.querySelectorAll('a')],boxes=links.map(e=>{const r=e.getBoundingClientRect();return{text:e.textContent,x:r.x,y:r.y,right:r.right,bottom:r.bottom,width:r.width,height:r.height}}),overlaps=[];
                      for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){const a=boxes[i],b=boxes[j],w=Math.min(a.right,b.right)-Math.max(a.x,b.x),h=Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y);if(w>.5&&h>.5)overlaps.push({a:a.text,b:b.text,width:w,height:h})}
                      return{boxes,overlaps};}''')
                    check('Footer targets do not overlap',not geometry['overlaps'],geometry)
                    check('Footer targets are at least44px high',all(b['height']>=43.99 for b in geometry['boxes']),geometry['boxes'])
                    target=page.locator('footer > p a').first
                    point=target.evaluate('''e=>{const q=document.createRange();q.selectNodeContents(e);const r=q.getBoundingClientRect();return{x:r.x+r.width/2,y:r.bottom-2,text:e.textContent}}''')
                    page.evaluate('''()=>{window.auditClicked=null;document.addEventListener('click',e=>{window.auditClicked=e.target.closest('a')?.textContent;e.preventDefault()},{capture:true,once:true})}''')
                    page.mouse.click(point['x'],point['y']);clicked=page.evaluate('window.auditClicked')
                    check('Clicking lower visible footer text activates its own link',clicked==point['text'],{'expected':point['text'],'actual':clicked})
                    page.locator('footer').screenshot(path=str(args.output/f'footer-{width}-{scheme}.png'))
                    check('Workbench has an accessible level2 heading',page.get_by_role('heading',level=2,name='Explore claims and evidence').count()==1)
                    check('Named action toolbars have group semantics',page.locator('.atlas-toolbar[aria-label]').evaluate_all('(els)=>els.every(e=>e.getAttribute("role")==="group")'))
                    page.locator('#atlas-search').fill('.')
                    page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed()
                    expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
                    nodes=page.locator('.atlas-space > details > ul').first
                    relations=page.locator('.atlas-space > details > ul').nth(1)
                    check('Relationship alternative has an explicit accessible name',relations.get_attribute('aria-label')=='Recorded directed relationships, scrollable')
                    check('Relationship alternative has an explicit keyboard stop',relations.get_attribute('tabindex')=='0')
                    nodes.locator('button').last.focus();page.keyboard.press('Tab')
                    check('Tab reaches relationship alternative',relations.evaluate('(e)=>e===document.activeElement'))
                    before=relations.evaluate('(e)=>e.scrollTop');page.keyboard.press('ArrowDown')
                    page.wait_for_function('(before)=>document.activeElement.scrollTop>before',arg=before,timeout=1500)
                    check('Arrow keys scroll relationships',relations.evaluate('(e)=>e.scrollTop')>before)
                    check('Relationship focus uses the3px brand indicator',relations.evaluate('(e)=>e.matches(":focus-visible")&&parseFloat(getComputedStyle(e).outlineWidth)>=3'))
                    page.screenshot(path=str(args.output/f'relationship-focus-{width}-{scheme}.png'))
                    # Use a fresh context: an already admitted page may correctly
                    # retain its verified observation when a later fetch fails.
                    fault=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce')
                    try:
                        fault.route('**/current.json*',lambda route:route.abort())
                        refused=fault.new_page();refused.goto(origin+'/dependencies.html?atlas-view=graph')
                        expect(refused.locator('#atlas-status')).to_contain_text('Unavailable')
                        stage=refused.locator('#atlas-stage')
                        check('Source-failure graph keeps a valid namedregion',stage.get_attribute('role')=='region' and bool(stage.get_attribute('aria-label')))
                        refused.screenshot(path=str(args.output/f'source-failure-{width}-{scheme}.png'))
                    finally:fault.close()
                except Exception as exc:row['failures'].append(str(exc))
                row['passed']=not row['failures'];context.close()
                (args.output/'accessibility-regression.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.close()
    finally:server.shutdown();server.server_close()
    print(json.dumps({'passed':sum(r['passed'] for r in report['cases']),'total':len(report['cases'])}))
    if any(not r['passed'] for r in report['cases']):raise SystemExit(1)


if __name__=='__main__':main()
