"""Run scoped headless Chromium Library and reader source-card checks against docs served on loopback.

Requires tests/browser-requirements.txt and the runner’s packaged Google Chrome.
Screenshots are evidence for inspection, not automatic visual certification.
"""
from contextlib import contextmanager
from functools import partial
from hashlib import sha256, file_digest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import traceback
from urllib.parse import parse_qs, urlsplit

ROOT=Path(__file__).resolve().parents[1]

@contextmanager
def local_docs():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
    server=ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler,directory=str(ROOT/"docs")))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/site/"
    finally:
        server.shutdown();server.server_close();thread.join()


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
        expect(page.locator("#latest-work time")).to_have_attribute("datetime","2026-10-02T21:18:12Z")
        expect(page.locator(".latest-chain > li")).to_have_count(3)
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


def check_source_card_flow(page, origin, expect, result, output):
    check_latest_work_flow(page,origin,expect,result,output)
    result["reader_entry_screenshots"]=[]
    for entry in ["index","explore","cite","reproduce","formal"]:
        page.goto(origin+entry+".html")
        expect(page.locator("h1")).to_have_count(1)
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
        report["passed"]=len(report["cases"])==9 and all(case["passed"] for case in report["cases"])
    except Exception:
        report["passed"]=False;report["error"]=traceback.format_exc()
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
