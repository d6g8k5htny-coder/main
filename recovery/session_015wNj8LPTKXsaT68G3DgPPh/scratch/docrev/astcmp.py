"""Usage: astcmp.py OLD NEW  -> exit 0 iff the two Python sources have identical ASTs after removing docstrings
(comments are not in the AST at all). Prints which docstrings differ."""
import ast, sys
def strip(src):
    t = ast.parse(src); docs = {}
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
           and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], 'value', None), ast.Constant) \
           and isinstance(n.body[0].value.value, str):
            docs[getattr(n, 'name', '<module>')] = n.body[0].value.value
            n.body = n.body[1:] or [ast.Pass()]
    return ast.dump(t, include_attributes=False), docs
a, da = strip(open(sys.argv[1]).read()); b, db = strip(open(sys.argv[2]).read())
changed = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
print('AST (docstrings removed) identical:', a == b, '| docstrings changed:', changed)
sys.exit(0 if a == b else 1)
