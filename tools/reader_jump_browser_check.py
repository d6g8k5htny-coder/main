#!/usr/bin/env python3
"""Native persistent navigation; emulated browser checks, not spoken AT evidence."""
import argparse,json,shutil,traceback
from pathlib import Path
from research_desk_browser_check import serve,ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--browser',default=shutil.which('chromium'));ap.add_argument('--baseline',action='store_true');ap.add_argument('--origin');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from playwright.sync_api import sync_playwright,expect
    report={'cases':[],'page_errors':[],'limits':'Headless Chromium/Linux, emulated widths; no physical/Safari/spoken AT/full WCAG claim.'}
    def run(origin,browser):
        for width,scheme in [(1440,'light'),(1440,'dark'),(390,'light'),(390,'dark'),(320,'dark')]:
            c=browser.new_context(viewport={'width':width,'height':844},color_scheme=scheme,reduced_motion='reduce');c.route('https://**/*',lambda route:route.fallback() if route.request.url.startswith(origin) else route.abort());p=c.new_page();p.set_default_timeout(5000);p.on('pageerror',lambda e:report['page_errors'].append(str(e)))
            row={'case':f'persistent-keyboard-{width}-{scheme}'};report['cases'].append(row)
            try:
                p.goto(origin+'workspace.html');assert p.locator('.reader-jump').count()==1,'Persistent Go to navigation is absent'
                menu=p.locator('.reader-jump');summary=menu.locator('summary');p.evaluate('scrollTo(0,3000)');box=summary.bounding_box();assert box and 0<=box['y'] and box['y']+box['height']<=844
                summary.focus();p.keyboard.press('Space');expect(menu).to_have_attribute('open','')
                nav=menu.locator('nav');box=nav.bounding_box();assert box and box['x']>=0 and box['y']>=0 and box['x']+box['width']<=width and box['y']+box['height']<=844
                if not args.baseline:
                    menu.locator('a').first.focus();p.keyboard.press('Escape');assert not menu.evaluate('(e)=>e.open');expect(summary).to_be_focused()
                    summary.press('Space');p.mouse.click(1,1);assert not menu.evaluate('(e)=>e.open')
                    summary.press('Space');link=menu.get_by_role('link',name='Compare snapshots',exact=True);link.focus();p.keyboard.press('Enter');expect(p).to_have_url(origin+'compare.html');assert not p.locator('.reader-jump').evaluate('(e)=>e.open')
                    p.locator('.reader-jump > summary').click();p.locator('.reader-jump').get_by_role('link',name='Reading list',exact=True).click();expect(p).to_have_url(origin+'workspace.html#reading-list');expect(p.locator('#reading-list')).to_have_attribute('open','')
                    p.emulate_media(media='print');expect(p.locator('.reader-jump')).to_be_hidden();p.emulate_media(media='screen')
                    assert p.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    p.locator('.reader-jump > summary').click();p.screenshot(path=str(args.output/f'menu-{width}-{scheme}.png'))
                row['passed']=True
            except Exception:row['passed']=False;row['error']=traceback.format_exc()
            c.close()
        if not args.baseline:
            for scripted in [True,False]:
                for scheme in ['light','dark']:
                    row={'case':f'toolbar-focus-{scheme}-'+('scripted' if scripted else 'native-no-js')};report['cases'].append(row)
                    c=browser.new_context(viewport={'width':320,'height':800},color_scheme=scheme,java_script_enabled=scripted);c.route('https://**/*',lambda route:route.fallback() if route.request.url.startswith(origin) else route.abort());p=c.new_page();p.set_default_timeout(5000);p.on('pageerror',lambda e:report['page_errors'].append(str(e)))
                    try:
                        p.goto(origin+('workspace.html' if scripted else 'formal.html'))
                        menu=p.locator('.reader-jump');summary=menu.locator('summary')
                        summary.focus();expect(summary).to_be_focused()
                        p.keyboard.press('Shift+Tab');assert not menu.evaluate('(e)=>e.contains(document.activeElement)')
                        assert p.evaluate('document.activeElement.matches(":focus-visible")')
                        assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1','The tools strip remains available during outside focus'
                        assert menu.evaluate('(e)=>getComputedStyle(e).pointerEvents')!='none'
                        p.keyboard.press('Tab');expect(summary).to_be_focused();assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1'
                        p.keyboard.press('Space');expect(menu).to_have_attribute('open','');p.keyboard.press('Tab');expect(menu.locator('a').first).to_be_focused();assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1'
                        p.keyboard.press('Shift+Tab');expect(summary).to_be_focused();p.keyboard.press('Space');assert not menu.evaluate('(e)=>e.open')
                        summary.evaluate('(e)=>e.blur()');assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1'
                        if scripted:
                            p.locator('#library-current-link').click()
                            field=p.locator('.current-search');field.click();expect(field).to_be_focused();assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1'
                            field.evaluate('(e)=>e.blur()');assert menu.evaluate('(e)=>getComputedStyle(e).opacity')=='1'
                        row['passed']=True
                    except Exception:row['passed']=False;row['error']=traceback.format_exc()
                    c.close()
            row={'case':'native-no-js-and-script-free-formal'};report['cases'].append(row)
            c=browser.new_context(viewport={'width':390,'height':844},java_script_enabled=False);p=c.new_page()
            try:
                p.goto(origin+'formal.html');assert '<script' not in p.content().lower();summary=p.locator('.reader-jump > summary');summary.focus();p.keyboard.press('Space');expect(p.locator('.reader-jump nav')).to_be_visible();p.locator('.reader-jump').get_by_role('link',name='Compare snapshots',exact=True).click();expect(p).to_have_url(origin+'compare.html');p.locator('.reader-jump > summary').click();expect(p.locator('.reader-jump nav')).to_be_visible();row['passed']=True
            except Exception:row['passed']=False;row['error']=traceback.format_exc()
            c.close()
    with sync_playwright() as pw:
        b=pw.chromium.launch(executable_path=args.browser,chromium_sandbox=True);report['browser']=b.version
        if args.origin:run(args.origin,b)
        else:
            with serve() as origin:run(origin,b)
        b.close()
    report['passed']=sum(c['passed'] for c in report['cases']);report['failed']=len(report['cases'])-report['passed'];(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['browser','passed','failed','page_errors']}));return 1 if report['failed'] or report['page_errors'] else 0
if __name__=='__main__':raise SystemExit(main())
