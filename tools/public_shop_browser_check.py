"""Run scoped headless Chromium Library checks against docs served on loopback.

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
        report["passed"]=len(report["cases"])==5 and all(case["passed"] for case in report["cases"])
    except Exception:
        report["passed"]=False;report["error"]=traceback.format_exc()
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
