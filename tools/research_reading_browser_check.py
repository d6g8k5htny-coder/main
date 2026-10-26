#!/usr/bin/env python3
"""Exercise source-preserving reading navigation; no scientific/AT verdict."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import subprocess
import traceback
from research_desk_browser_check import serve
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
CUTS = {
    'reading-cut-20261010': ['reading-cut-20261010-details'],
    'reading-cut-20261009': ['reading-cut-20261009-details'],
    'reading-cut-20261007': ['reading-cut-20261007-details'],
    'shrinking-bin-sampling': ['sampling-details'],
    'pair-endpoint-rate': ['pair-endpoint-details'],
    'reading-addendum': ['bar-occurrence-density-details', 'designated-pair-failure-details'],
    'latest-work': ['latest-work-details'],
}

def run(origin, browser, out, report):
    def case(name, fn):
        row = {'case': name}; report['cases'].append(row)
        try: fn(); row['passed'] = True
        except Exception:
            row.update(passed=False, error=traceback.format_exc()); raise
    def context(**kw):
        c = browser.new_context(reduced_motion='reduce', **kw)
        c.route('https://**', lambda r: r.abort())
        return c
    def page(c, fragment=''):
        p = c.new_page(); p.set_default_timeout(12000)
        p.on('pageerror', lambda e: report['page_errors'].append(str(e)))
        p.goto(origin + 'research.html' + fragment); return p
    for width, scheme in [(1440, 'light'), (1440, 'dark'), (390, 'light'), (320, 'dark')]:
        def journey():
            c = context(viewport={'width': width, 'height': 900}, color_scheme=scheme)
            p = page(c)
            expect(p.locator('.reading-jump')).to_have_count(7)
            expect(p.locator('#reading-archive')).not_to_have_attribute('open', '')
            q = p.locator('#reading-questions a[href="#reading-cut-20261010"]')
            q.focus(); p.keyboard.press('Enter')
            expect(p.locator('#reading-cut-20261010')).to_be_focused()
            expect(p.locator('#reading-archive')).to_have_attribute('open', '')
            for cut, sources in CUTS.items():
                nav = p.locator(f'#{cut} > .reading-jump')
                expect(nav.locator('a')).to_have_count(2 + len(sources))
                assert nav.locator('a').evaluate_all('(aa)=>aa.every(a=>a.getAttribute("href").startsWith("#"))')
                for source in sources:
                    link = nav.locator(f'a[href="#{source}"]')
                    link.focus(); p.keyboard.press('Enter')
                    expect(p.locator(f'#{source}')).to_have_attribute('open', '')
                    expect(p.locator(f'#{source}')).to_be_focused()
                    assert p.evaluate('location.hash') == '#' + source
                    p.reload()
                    expect(p.locator(f'#{source}')).to_have_attribute('open', '')
                back = p.locator(f'#{cut} > .reading-return a')
                back.focus(); p.keyboard.press('Enter')
                expect(p.locator('#reading-questions')).to_be_focused()
            newest = p.locator('#reading-cut-20261010 > .reading-jump')
            newest.locator('a[href="#reading-guide"]').focus(); p.keyboard.press('Enter')
            expect(p.locator('#reading-guide')).to_have_attribute('open', '')
            p.go_back(); p.wait_for_function('location.hash === "#reading-questions"')
            expect(p.locator('#reading-questions')).to_be_visible()
            p.go_forward(); expect(p.locator('#reading-guide')).to_have_attribute('open', '')
            newest.locator('a[href="#reading-cut-20261010-details"]').click()
            p.reload(); expect(p.locator('#reading-cut-20261010-details')).to_have_attribute('open', '')
            # Re-initialization cannot duplicate controls or overwrite a source ID.
            p.evaluate('''async () => { const script=[...document.scripts].find(s=>s.src.includes('research-reading.mjs')); const m=await import(script.src); m.installReadingNavigation(document); }''')
            expect(p.locator('.reading-jump')).to_have_count(7)
            assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
            for link in newest.locator('a').all():
                box = link.bounding_box(); assert box['width'] >= 44 and box['height'] >= 44, box
            p.locator('#reading-cut-20261010 > .section-heading').evaluate('(e)=>e.scrollIntoView({block:"start"})')
            p.screenshot(path=str(out / f'reading-{width}-{scheme}.png'))
            p.emulate_media(media='print')
            expect(newest).not_to_be_visible()
            expect(p.locator('#reading-cut-20261010 h2')).to_be_visible()
            c.close()
        case(f'keyboard-native-history-all-cuts-{width}-{scheme}', journey)
    def nojs():
        c = context(java_script_enabled=False, viewport={'width':320,'height':900})
        p=page(c,'#reading-cut-20261010-details')
        expect(p.locator('#reading-archive')).to_have_attribute('open','')
        expect(p.locator('.reading-jump')).to_have_count(0)
        expect(p.locator('#reading-guide a[href="#reading-cut-20261010"]')).to_have_count(1)
        assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
        c.close()
    case('no-javascript-preserved-static-reading-map',nojs)
    def missing():
        c=context(); p=page(c,'#reading-cut-20261010')
        p.evaluate('''async()=>{document.querySelectorAll('.reading-jump,.reading-return').forEach(n=>n.remove()); document.getElementById('reading-cut-20261010-details').remove(); const s=[...document.scripts].find(s=>s.src.includes('research-reading.mjs')); const m=await import(s.src); m.installReadingNavigation(document);}''')
        expect(p.locator('#reading-cut-20261010 > .reading-jump')).to_have_count(0)
        expect(p.locator('.reading-jump')).to_have_count(6)
        expect(p.locator('#reading-cut-20261010 h2')).to_be_visible()
        c.close()
    case('missing-source-disclosure-omits-only-affected-enhancement',missing)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--browser',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--origin');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report={'started_at':datetime.now(timezone.utc).isoformat(),'cases':[],'page_errors':[], 'browser_sha256':sha256(Path(args.browser).read_bytes()).hexdigest(),'scope':'engineering reading navigation only; no scientific or spoken-AT verdict', 'local_product_sha256':{n:sha256((ROOT/'docs/site'/n).read_bytes()).hexdigest() for n in ['research.html','research-reading.mjs','research-reading.css','reader-shell.mjs']}, 'source_identity_note':'Git HEAD is checkout ancestry. Local hashes bind working bytes for local serving, and are comparison context only for --origin; published bytes require independent readback.'}
    try:
        try: report['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except subprocess.CalledProcessError:report['source_commit']=None
        with sync_playwright() as pw:
            b=pw.chromium.launch(executable_path=args.browser,headless=True,chromium_sandbox=True)
            report['browser_version']=b.version
            if args.origin:run(args.origin.rstrip('/')+'/',b,args.output,report)
            else:
                with serve() as origin:run(origin,b,args.output,report)
            b.close()
        assert not report['page_errors'],report['page_errors']
        report['passed']=True
    except Exception:report.update(passed=False,error=traceback.format_exc())
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'output':str(args.output)}))
    raise SystemExit(0 if report['passed'] else 1)
