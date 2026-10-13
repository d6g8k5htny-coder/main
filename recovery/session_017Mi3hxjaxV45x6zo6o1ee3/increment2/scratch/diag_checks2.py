import ast, importlib.util, json, pathlib, sys, types, os
WT = pathlib.Path("<SCRATCH>/wtdry")
def instrument(path, funcs):
    src = (WT / path).read_text(); tree = ast.parse(src); lines = src.splitlines()
    class T(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            for field in ("body", "orelse"):
                body = getattr(node, field, None)
                if isinstance(body, list) and body and any(isinstance(s, ast.stmt) for s in body):
                    nb = []
                    for st in body:
                        nb.append(st)
                        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id == "ok":
                            nb.append(ast.parse(f"_report({st.lineno})").body[0])
                    setattr(node, field, nb)
            return node
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in funcs:
            T().visit(fn)
    ast.fix_missing_locations(tree)
    mod = types.ModuleType("m"); mod.__file__ = str(WT / path)
    state = {"prev": True}
    def _report(ln):
        import inspect
        ok = inspect.currentframe().f_back.f_locals.get("ok")
        if state["prev"] and not ok:
            print("   FIRST-FAIL line %d: %s" % (ln, lines[ln - 1].strip()[:140]))
        state["prev"] = bool(ok)
    mod._report = _report; mod._state = state
    sys.argv = ["x"]; exec(compile(tree, str(WT / path), "exec"), mod.__dict__)
    return mod, state
os.chdir(WT)
print("== reconciliation_check.check_transitions")
m, st = instrument("reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py", {"check_transitions"})
print("   ->", m.check_transitions(WT))
print("== alignment_check")
m2, st2 = instrument("reviews/register_alignment_20260930/alignment_check.py", {"check_baseline", "check_transitions", "check_d4_duplicate", "check_gate"})
spec = m2.load_spec(WT)
st2["prev"] = True; print("   check_baseline ->", m2.check_baseline(WT, spec))
st2["prev"] = True; r = m2.check_transitions(WT, spec); print("   check_transitions ->", r[0])
st2["prev"] = True; print("   check_d4_duplicate ->", m2.check_d4_duplicate(WT, spec))
graph = json.loads((WT / m2.GRAPH).read_text()); new = m2.apply(spec, graph)
try:
    m2.load_hard_gate(WT).validate_graph_fail_closed(new); print("   apply(live) validates: OK")
except Exception as ex: print("   apply(live) validate FAILS:", repr(ex)[:160])
