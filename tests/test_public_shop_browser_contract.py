"""Stdlib contract checks for the optional browser-QA runner, not browser evidence."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class BrowserCheckContract(unittest.TestCase):
    def test_browser_job_is_read_only_and_retains_failure_artifacts(self):
        workflow=(ROOT/'.github/workflows/public-shop.yml').read_text()
        self.assertIn('  browser-smoke:',workflow)
        self.assertNotIn('pull_request_target:',workflow)
        self.assertIn('contents: read',workflow)
        block=workflow.split('  browser-smoke:',1)[1]
        self.assertIn('persist-credentials: false',block)
        self.assertIn('if: always()',block)
        self.assertIn('path: browser-evidence',block)
        self.assertIn('timeout-minutes: 10',block)
        self.assertNotIn('playwright install',block, 'Use the runner’s supported packaged browser sandbox')
        self.assertIn('runs-on: ubuntu-24.04',block)
        self.assertNotIn('secrets.',block)
        self.assertNotIn('deploy',block)
        for test in ('tests/test_measurement_model.mjs','tests/test_dependency_viewer.mjs'):
            self.assertIn(test,workflow)

    def test_runner_is_loopback_only_with_a_four_case_matrix(self):
        path=ROOT/'tools/public_shop_browser_check.py'
        self.assertTrue(path.is_file(),'Browser runner is missing')
        source=path.read_text();ast.parse(source)
        self.assertIn('("127.0.0.1", 0)',source)
        self.assertNotIn('0.0.0.0',source)
        for item in ['1200','900','390','844','"light"','"dark"','go_back','go_forward','to_be_focused','scrollWidth','screenshot','checked_commit','browser.version']:
            self.assertIn(item,source)
        self.assertIn('channel="chrome",chromium_sandbox=True',source)
        self.assertIn('browser_executable_sha256',source)
        self.assertNotIn('assert ',source, 'CLI checks must not disappear under Python -O')
        self.assertIn('Access-Control-Allow-Origin',source)
        self.assertIn('Source unavailable (503)',source)
        self.assertIn('len(route_hits)==1',source)
        self.assertNotIn('ignore_https_errors',source)
        self.assertNotIn('--no-sandbox',source)

    def test_browser_dependencies_are_separate_and_version_pinned(self):
        path=ROOT/'tests/browser-requirements.txt'
        self.assertTrue(path.is_file(),'Optional browser dependencies are missing')
        requirements=[line for line in path.read_text().splitlines() if line and not line.startswith('#')]
        self.assertEqual(len(requirements),4)
        self.assertTrue(all('==' in line and '://' not in line for line in requirements))
        self.assertIn('playwright==1.62.0',requirements)

class ReaderSourceBrowserContract(unittest.TestCase):
    def test_clipboard_denial_fixture_reinitializes_after_same_document_history(self):
        module=ast.parse((ROOT/'tools/public_shop_browser_check.py').read_text())
        exports=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                     and node.name=='check_other_teaching_exports')
        flow=next(node for node in exports.body if isinstance(node,ast.For)
                  and isinstance(node.target,ast.Tuple))
        index=next(i for i,node in enumerate(flow.body) if isinstance(node,ast.Expr)
                   and isinstance(node.value,ast.Call) and node.value.args
                   and isinstance(node.value.args[0],ast.Constant)
                   and "setItem('teaching-clipboard-test','deny')" in str(node.value.args[0].value))
        class DocumentFixture:
            url='same-state'; stored_mode='defer'; installed_mode='defer'
            def evaluate(self,script):
                self.stored_mode='deny'
            def goto(self,url):
                # Returning to the same fragment does not install init scripts.
                if url!=self.url:
                    self.url=url; self.installed_mode=self.stored_mode
            def reload(self):
                self.installed_mode=self.stored_mode
        page=DocumentFixture()
        exec(compile(ast.Module(body=flow.body[index:index+2],type_ignores=[]),'<clipboard-fixture>','exec'),
             {'page':page,'start':page.url})
        self.assertEqual(page.installed_mode,'deny','The denial case retained the deferred clipboard from Back navigation')

    def test_pin_source_enumeration_preserves_the_downloaded_artifact_path(self):
        # Execute the actual metadata-enumeration loop: it must not clobber
        # the Path later used to reopen and record the downloaded SVG.
        module=ast.parse((ROOT/'tools/public_shop_browser_check.py').read_text())
        exports=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                     and node.name=='check_other_teaching_exports')
        read_svg=next(node for node in exports.body if isinstance(node,ast.FunctionDef)
                      and node.name=='read_svg')
        reopen=next(node for node in ast.walk(read_svg) if isinstance(node,ast.Call)
                    and isinstance(node.func,ast.Attribute) and node.func.attr=='resolve')
        path_name=reopen.func.value.id
        source_loop=next(node for node in ast.walk(read_svg) if isinstance(node,ast.For)
                         and isinstance(node.target,ast.Tuple)
                         and [item.id for item in node.target.elts[1:]]==['blob','size','digest'])
        artifact=Path('/tmp/downloaded-teaching-figure.svg')
        scope={path_name:artifact,'expected_sources':[]}
        exec(compile(ast.Module(body=[source_loop],type_ignores=[]),'<source-enumeration>','exec'),scope)
        self.assertEqual(scope[path_name],artifact,'Source identity loop overwrote the downloaded SVG path')
        self.assertEqual(len(scope['expected_sources']),2)

    def test_other_teaching_exports_are_reached_in_the_four_source_cases(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        module=ast.parse(source)
        functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
        self.assertIn('check_other_teaching_exports',functions)
        calls=[node for node in ast.walk(functions['check_source_card_flow'])
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and node.func.id=='check_other_teaching_exports']
        self.assertEqual(len(calls),1,'The real four-viewport flow must reach export checks')
        workflow=(ROOT/'.github/workflows/public-shop.yml').read_text()
        for name in ('teaching_export','pin_export','palette_export'):
            self.assertIn(f'tests/test_{name}.mjs',workflow)

    def test_bibtex_disabled_checks_keep_the_hidden_button_addressable(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        module=ast.parse(source)
        flow=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                  and node.name=='check_source_card_flow')
        bindings=[node for node in ast.walk(flow) if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=='bib'
                          for target in node.targets)]
        self.assertEqual(len(bindings),1)
        call=bindings[0].value
        self.assertIsInstance(call,ast.Call)
        self.assertEqual(call.func.attr,'locator',
                         'Role locators omit the hidden actions after an edit or invalid URL')
        self.assertEqual(call.args[0].value,'#reference-copy-bibtex')

    def test_new_reader_controls_are_reached_by_the_existing_browser_flows(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        module=ast.parse(source)
        functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
        for parent,helper in [('check_source_card_flow','check_curvature_preset_flow'),
                              ('check_reader_tools_flow','check_recorded_context_flow')]:
            with self.subTest(helper=helper):
                self.assertIn(helper,functions,'A source review is not actual hosted control coverage')
                calls=[node for node in ast.walk(functions[parent]) if isinstance(node,ast.Call)
                       and isinstance(node.func,ast.Name) and node.func.id==helper]
                self.assertEqual(len(calls),1,'The helper must be reached by the existing four-case flow')
        self.assertIn('len(report["cases"])==18',source)

    def test_bibtex_browser_checks_compare_real_clipboard_and_keep_failure_paths(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        module=ast.parse(source)
        flow=next(node for node in module.body if isinstance(node,ast.FunctionDef)
                  and node.name=='check_source_card_flow')
        flow_source=ast.get_source_segment(source,flow)
        for token in ['BibTeX clipboard differs from visible template',
                      '#reference-copy-bibtex','Previous copy finished',
                      'Pending BibTeX copy differs from visible template',
                      'copy it manually','referencePending.resolve()',
                      'bibtex_with_digest']:
            self.assertIn(token,flow_source)

    def test_existing_runner_covers_source_quote_and_keyboard_disclosure(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in ['check_source_card_flow','source-quote-readable','source-quote-original','Exact source quote','issuecomment-5841270276','len(report["cases"])==18']:
            self.assertIn(token,source)

    def test_reader_tools_have_desktop_and_mobile_browser_evidence(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in [
            'check_reader_tools_flow', 'measure.html', 'dependencies.html',
            'Pinned Math commit 7858329974e2', 'math.rn-fixed-annulus-window',
            'Saved selection unavailable', 'to_be_focused', 'post-layout saved-link focus',
            'https://github.com/d6g8k5htny-coder/main/blob/e60629edde151c27541847b61672e38965752b77/docs/research-translation/20260930/EXPERIMENT.md#1-freeze-the-observable-before-generating-data',
            'https://github.com/d6g8k5htny-coder/Math-/blob/7858329974e28be79f29b22644370084ff43da4f/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md',
            'https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206',
            '#dependency-paths > li',
            'strong:text-is("math.lifetime-remainder")',
            'required dependency path and actionable source/review metadata',
            'dated claim-dependency snapshot', 'page.keyboard.press("Tab")',
            'page.keyboard.press("Enter")',
            'documentElement.scrollWidth',
            'len(report["cases"])==18',
        ]:
            self.assertIn(token,source)

    def test_rendered_layout_cases_read_what_the_browser_drew(self):
        # Stylesheet parsing misses compound selectors, nesting, !important and longhands; these cases read computed and rendered outcomes.
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        module=ast.parse(source)
        functions={node.name:node for node in module.body if isinstance(node,ast.FunctionDef)}
        main=ast.get_source_segment(source,functions['main'])
        for name in ('check_rendered_focus_rings','check_rendered_text_layout'):
            self.assertIn(name,functions)
            self.assertIn(name,main)
        for token in [":focus-visible","getBoundingClientRect","getClientRects","overflowX","overflowY","#conditional-route","figure.museum-visual",
                      "ec014","remote","annulus","p15","blockquote","text-spacing-override.css","line-height:1.5","letter-spacing:.12em",
                      "word-spacing:.16em","margin-bottom:2em",".evidence-table th, .evidence-table td:nth-child(2)","hist.CH-LIFT",
                      "math.d5-component.punctured-pin-proof","scrollWidth","clientWidth"]:
            self.assertIn(token,source)
        self.assertIn('len(report["cases"])==18',source)

if __name__=='__main__':unittest.main()
