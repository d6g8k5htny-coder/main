"""Instrument every `ok &= ...` in selected checker functions and report the first-failing lines, in the executed worktree."""
import ast, importlib.util, json, pathlib, sys, types
WT = pathlib.Path("<SCRATCH>/wtdry")
def instrument(path, funcs):
    src = (WT / path).read_text(); tree = ast.parse(src); lines = src.splitlines()
    class T(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name not in funcs: return node
            new_body = []
            for st in node.body:
                new_body.append(st)
                if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id == "ok":
                    rep = ast.parse(f"_report({node.name!r}, {st.lineno}, ok)").body[0]
                    new_body.append(rep)
                if isinstance(st, ast.For):        # instrument ok &= inside simple for loops too
                    for inner in st.body:
                        pass
            node.body = new_body
            return node
    tree = T().visit(tree); ast.fix_missing_locations(tree)
    mod = types.ModuleType(path); mod.__file__ = str(WT / path)
    seen = set()
    def _report(fn, ln, ok):
        if not ok and (fn, ln) not in seen:
            seen.add((fn, ln)); print("   FAIL %s line %d: %s" % (fn, ln, lines[ln - 1].strip()[:150]))
    mod._report = _report
    sys.argv = ["x"]
    exec(compile(tree, str(WT / path), "exec"), mod.__dict__)
    return mod
import os
os.chdir(WT)
print("== reconciliation_check (executed state)")
m = instrument("reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py", {"check_transitions"})
print("   check_transitions ->", m.check_transitions(WT))
print("== alignment_check (executed state)")
m2 = instrument("reviews/register_alignment_20260930/alignment_check.py", {"check_baseline", "check_transitions", "check_gate", "check_propagation", "check_d4_duplicate"})
spec = m2.load_spec(WT)
print("   check_baseline ->", m2.check_baseline(WT, spec))
graph = json.loads((WT / m2.GRAPH).read_text()); new = m2.apply(spec, graph)
try: print("   check_transitions ->", m2.check_transitions(WT, spec))
except Exception as ex: print("   check_transitions raised", repr(ex)[:200])
try: print("   check_gate ->", m2.check_gate(WT, spec, new))
except Exception as ex: print("   check_gate raised", repr(ex)[:200])
try: print("   check_propagation ->", m2.check_propagation(WT, spec, new))
except Exception as ex: print("   check_propagation raised", repr(ex)[:200])
try: print("   check_d4_duplicate ->", m2.check_d4_duplicate(WT, spec))
except Exception as ex: print("   check_d4_duplicate raised", repr(ex)[:200])
