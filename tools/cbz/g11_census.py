#!/usr/bin/env python3
"""CBZ-0 gate G11 census: stub scan and tool documentation check (stdlib only).

Reads the exact blob bytes of one git revision and reports:

  stub      a Python function whose body, after any docstring, is only `pass`,
            `...` or `raise NotImplementedError`, or is the docstring alone.
            @abstractmethod is contextual only in an immediate imported ABC subclass;
            @overload requires a complete overload group and a concrete implementation;
            immediate Protocol methods are contextual; ABC or exception inheritance alone
            is not an exemption.
  marker    TODO, FIXME, XXX, TBD or STUB (upper case) or "not implemented" in
            a code comment. In Python the comment is found by tokenizing; in
            .mjs/.js/.cjs/.sh a comment is found by a quote-parity heuristic.
            The same words in string data are recorded as mentions only.
  noop      a workflow step whose `run:` does nothing: `true`, `:` or `exit 0`.
  tools     every executable source under tools/ (not tests, not __init__.py),
            with tracked Markdown documentation and statically reachable test
            or workflow references. References are not proof of runtime execution.

Every path has an area. Only `tooling` (tools/, .github/) and `site`
(docs/site/ outside vendor/) are production; findings elsewhere are reported
and never fail. A production file that is not UTF-8, or Python that does not
parse, fails closed.

Exit status: 0 when nothing in production fails, 1 when something does, 2 when
the revision cannot be read. The census is evidence for a reviewer, never a
G11 certificate.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import posixpath
import shlex
import io
import json
import re
import subprocess
import sys
import tokenize

MARKER = re.compile(r"\b(?:TODO|FIXME|XXX|TBD|STUB)\b|\b(?i:not implemented)\b")
CODE_SUFFIXES = (".py", ".mjs", ".js", ".cjs", ".sh")
SCANNED = CODE_SUFFIXES + (".md", ".yml", ".yaml")
LEGIT_DECORATORS = {"abstractmethod", "overload"}
LEGIT_BASES = {"Protocol", "ABC"}
NOOP_RUN = {"true", ":", "exit 0"}
PRODUCTION = {"tooling", "site"}
_UNRESOLVED_STATIC = object()


class _StaticAlternatives(tuple):
    """Resolved mutually exclusive values whose active branch is not statically known."""


def _static_alternatives(values):
    distinct = []
    for value in values:
        if not any(value == prior for prior in distinct):
            distinct.append(value)
    return distinct[0] if len(distinct) == 1 else _StaticAlternatives(distinct)


def _static_add(left, right):
    left_values = left if isinstance(left, _StaticAlternatives) else (left,)
    right_values = right if isinstance(right, _StaticAlternatives) else (right,)
    results = []
    for left_value in left_values:
        for right_value in right_values:
            if isinstance(left_value, list) and isinstance(right_value, list):
                results.append(left_value + right_value)
            elif isinstance(left_value, str) and isinstance(right_value, str):
                results.append(left_value + right_value)
            else:
                return None
    return _static_alternatives(results)


def area(path):
    name = path.rsplit("/", 1)[-1]
    if (path.startswith("tests/") or "/tests/" in path or name == "conftest.py"
            or name.startswith("test_") or name.endswith("_test.py") or ".test." in name):
        return "tests"
    if path.startswith("tools/") or path.startswith(".github/"):
        return "tooling"
    if path.startswith("docs/site/vendor/"):
        return "vendor"
    if path.startswith("docs/site/"):
        return "site"
    if path.startswith("experiments/"):
        return "research"
    return "evidence"


def _decorator_name(node):
    while isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _base_name(node):
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""

def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return base + "." + node.attr if base else ""
    return ""


def _target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return set()


def _scope_bindings(scope):
    """Names bound in one lexical scope; nested scope bodies are kept separate."""
    names = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = scope.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            names.add(arg.arg)
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)

    def visit(node, root=False):
        if not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return
        if not root and isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(scope, True)
    return names


def _import_bindings(tree):
    modules, symbols, shadowed = {}, {}, set()
    for node in tree.body:
        if type(node) is ast.Import:
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                modules[bound] = alias.name if alias.asname else alias.name.split(".", 1)[0]
        elif type(node) is ast.ImportFrom and node.level == 0:
            for alias in node.names:
                symbols[alias.asname or alias.name] = (node.module or "", alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            shadowed.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                shadowed.update(_target_names(target))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            shadowed.update(_target_names(node.target))
    for name in shadowed:
        modules.pop(name, None)
        symbols.pop(name, None)
    return {"modules": modules, "symbols": symbols, "shadowed": shadowed}


def _resolve_import(node, imports):
    if isinstance(node, ast.Name):
        if node.id in imports["shadowed"]:
            return None
        return imports["symbols"].get(node.id)
    dotted = _dotted_name(node)
    if not dotted:
        return None
    parts = dotted.split(".")
    if parts[0] in imports["shadowed"]:
        return None
    module = imports["modules"].get(parts[0])
    if module is None or len(parts) < 2:
        return None
    resolved = module + "." + ".".join(parts[1:])
    return tuple(resolved.rsplit(".", 1))


def _resolved_import_name(node, imports, scope_shadowed=()):
    dotted = _dotted_name(node)
    if not dotted:
        return None
    parts = dotted.split(".")
    if parts[0] in imports["shadowed"] or parts[0] in scope_shadowed:
        return None
    if parts[0] in imports["modules"]:
        return ".".join([imports["modules"][parts[0]], *parts[1:]])
    if parts[0] in imports["symbols"]:
        module, symbol = imports["symbols"][parts[0]]
        return ".".join([module, symbol, *parts[1:]])
    return None


def _empty_import_bindings():
    return {"modules": {}, "symbols": {}, "shadowed": set()}


def _update_local_import_bindings(node, bindings):
    """Apply one reachable function-local binding in execution order."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            bindings["modules"][bound] = alias.name if alias.asname else alias.name.split(".", 1)[0]
            bindings["symbols"].pop(bound, None)
            bindings["shadowed"].discard(bound)
        return
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        for alias in node.names:
            bound = alias.asname or alias.name
            bindings["symbols"][bound] = (node.module or "", alias.name)
            bindings["modules"].pop(bound, None)
            bindings["shadowed"].discard(bound)
        return
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    for target in targets:
        for name in _target_names(target):
            bindings["modules"].pop(name, None)
            bindings["symbols"].pop(name, None)
            bindings["shadowed"].add(name)


def _combined_import_bindings(module, local, scope_shadowed):
    modules = dict(module["modules"])
    symbols = dict(module["symbols"])
    locally_bound = set(local["modules"]) | set(local["symbols"])
    modules.update(local["modules"])
    symbols.update(local["symbols"])
    shadowed = set(module["shadowed"]) | set(local["shadowed"])
    shadowed.update(set(scope_shadowed) - locally_bound)
    for name in shadowed:
        modules.pop(name, None)
        symbols.pop(name, None)
    return {"modules": modules, "symbols": symbols, "shadowed": shadowed}, ()


def _resolve_scoped_import(node, imports, scope_shadowed=()):
    target = node
    while isinstance(target, (ast.Call, ast.Subscript)):
        target = target.func if isinstance(target, ast.Call) else target.value
    dotted = _dotted_name(target)
    binding = dotted.split(".", 1)[0] if dotted else ""
    if binding and binding in scope_shadowed:
        return None
    return _resolve_import(target, imports)


def _trusted_decorator(node, imports, scope_shadowed=()):
    resolved = _resolve_scoped_import(node, imports, scope_shadowed)
    return resolved in {("abc", "abstractmethod"), ("typing", "overload"),
                        ("typing_extensions", "overload")}


def _overload_declaration(node, imports, scope_shadowed=()):
    return (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_resolve_scoped_import(dec, imports, scope_shadowed) in {
            ("typing", "overload"), ("typing_extensions", "overload")
        } for dec in node.decorator_list)
    )


def _complete_overload_group(func, siblings, imports, scope_shadowed=()):
    if not _overload_declaration(func, imports, scope_shadowed):
        return False
    try:
        index = next(i for i, node in enumerate(siblings) if node is func)
    except StopIteration:
        return False
    start = index
    while (start > 0 and isinstance(siblings[start - 1], (ast.FunctionDef, ast.AsyncFunctionDef))
           and siblings[start - 1].name == func.name
           and _overload_declaration(siblings[start - 1], imports, scope_shadowed)):
        start -= 1
    end = start
    while (end < len(siblings) and isinstance(siblings[end], (ast.FunctionDef, ast.AsyncFunctionDef))
           and siblings[end].name == func.name
           and _overload_declaration(siblings[end], imports, scope_shadowed)):
        end += 1
    if not any(siblings[i] is func for i in range(start, end)) or end >= len(siblings):
        return False
    implementation = siblings[end]
    return (
        isinstance(implementation, (ast.FunctionDef, ast.AsyncFunctionDef))
        and implementation.name == func.name
        and not _overload_declaration(implementation, imports, scope_shadowed)
        and _stub_kind(implementation.body) is None
    )


def _trusted_abstract_base(node, imports, scope_shadowed=()):
    resolved = _resolve_scoped_import(node, imports, scope_shadowed)
    if resolved == ("abc", "ABC"):
        return "ABC"
    if resolved in {("typing", "Protocol"), ("typing_extensions", "Protocol")}:
        return "Protocol"
    return None


def _builtin_exception_base(node, imports, scope_shadowed=()):
    resolved = _resolve_scoped_import(node, imports, scope_shadowed)
    name = None
    if resolved and resolved[0] == "builtins":
        name = resolved[1]
    elif (isinstance(node, ast.Name) and node.id in builtins.__dict__
          and node.id not in imports["shadowed"] and node.id not in scope_shadowed):
        name = node.id
    if not name:
        return False
    candidate = getattr(builtins, name, None)
    return isinstance(candidate, type) and issubclass(candidate, BaseException)


def _http_request_handler_base(node, imports, scope_shadowed=()):
    return _resolve_scoped_import(node, imports, scope_shadowed) in {
        ("http.server", "BaseHTTPRequestHandler"),
        ("http.server", "SimpleHTTPRequestHandler"),
        ("http.server", "CGIHTTPRequestHandler"),
    }




def _stub_kind(body):
    rest = body
    if rest and isinstance(rest[0], ast.Expr) and isinstance(rest[0].value, ast.Constant) \
            and isinstance(rest[0].value.value, str):
        rest = rest[1:]
        if not rest:
            return "docstring only"
    if len(rest) != 1:
        return None
    stmt = rest[0]
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return "ellipsis"
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return "raise NotImplementedError"
    return None


def _legit_context(func, classes, imports, scope_shadowed=(), siblings=()):
    for dec in func.decorator_list:
        if _trusted_decorator(dec, imports, scope_shadowed):
            resolved = _resolve_scoped_import(dec, imports, scope_shadowed)
            if resolved == ("abc", "abstractmethod"):
                if classes and any(_trusted_abstract_base(base, imports, scope_shadowed) == "ABC"
                                   for base in classes[-1].bases):
                    return "@abstractmethod"
            elif (resolved in {("typing", "overload"), ("typing_extensions", "overload")}
                  and _complete_overload_group(func, siblings, imports, scope_shadowed)):
                return "@overload"
    # Only the immediate class can supply method context.
    for cls in classes[-1:]:
        for base in cls.bases:
            base_kind = _trusted_abstract_base(base, imports, scope_shadowed)
            if base_kind == "Protocol":
                return "method of a Protocol class"
            if base_kind == "ABC":
                continue
            if _builtin_exception_base(base, imports, scope_shadowed):
                continue
            if (_http_request_handler_base(base, imports, scope_shadowed)
                    and func.name == "log_message"):
                return "log_message override that silences http.server request logging"
    return None


def python_stubs(tree, imports=None):
    found = []
    if imports is None:
        imports = _import_bindings(tree)

    def visit(node, classes, scope_shadowed, siblings):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, classes + [child], scope_shadowed | _scope_bindings(child), child.body)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = _stub_kind(child.body)
                if kind:
                    found.append({"line": child.lineno, "name": child.name, "kind": kind,
                                  "context": _legit_context(child, classes, imports, scope_shadowed, siblings)})
                visit(child, [], scope_shadowed | _scope_bindings(child), child.body)
            else:
                child_body = getattr(child, "body", None)
                visit(child, classes, scope_shadowed,
                      child_body if isinstance(child_body, list) else siblings)
    visit(tree, [], set(), tree.body)
    return found


def python_words(raw):
    markers, mentions = [], []
    for tok in tokenize.tokenize(io.BytesIO(raw).readline):
        if tok.type == tokenize.COMMENT:
            target = markers
        elif tok.type == tokenize.STRING or tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            target = mentions
        else:
            continue
        for m in MARKER.finditer(tok.string):
            target.append({"line": tok.start[0] + tok.string[:m.start()].count("\n"), "word": m.group(0)})
    return markers, mentions


def _comment_start(line, opener):
    # Index where a comment opens outside quotes, by quote parity; -1 if none.
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif line.startswith(opener, i) and (opener != "#" or i == 0 or line[i - 1].isspace()):
            return i
        i += 1
    return -1


def text_words(text, path):
    markers, mentions = [], []
    opener = "#" if path.endswith(".sh") else "//"
    in_block = False
    for number, line in enumerate(text.splitlines(), 1):
        comment_parts, data_parts = [], []
        if opener == "//":
            rest = line
            while rest:
                if in_block:
                    end = rest.find("*/")
                    comment_parts.append(rest if end < 0 else rest[:end])
                    if end < 0:
                        rest = ""
                    else:
                        in_block, rest = False, rest[end + 2:]
                    continue
                line_at, block_at = _comment_start(rest, "//"), _comment_start(rest, "/*")
                starts = [p for p in (line_at, block_at) if p >= 0]
                if not starts:
                    data_parts.append(rest)
                    break
                first = min(starts)
                data_parts.append(rest[:first])
                if first == line_at:
                    comment_parts.append(rest[first:])
                    break
                in_block, rest = True, rest[first + 2:]
        else:
            at = _comment_start(line, "#")
            data_parts.append(line if at < 0 else line[:at])
            if at >= 0:
                comment_parts.append(line[at:])
        for part in comment_parts:
            markers.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(part))
        for part in data_parts:
            mentions.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(part))
    return markers, mentions



def _yaml_comment_start(line):
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if quote == '"' and ch == "\\":
                i += 2
                continue
            if quote == "'" and ch == "'" and i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return i
        i += 1
    return -1


def yaml_words(text):
    markers, mentions = [], []
    for number, line in enumerate(text.splitlines(), 1):
        at = _yaml_comment_start(line)
        data = line if at < 0 else line[:at]
        comment = "" if at < 0 else line[at + 1:]
        mentions.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(data))
        markers.extend({"line": number, "word": m.group(0)} for m in MARKER.finditer(comment))
    return markers, mentions


def _strip_shell_comment(line):
    at = _comment_start(line, "#")
    return line if at < 0 else line[:at]


def _is_step_run(lines, index):
    line = lines[index]
    indent = len(line) - len(line.lstrip())
    list_run = line.lstrip().startswith("- ")
    parents = []
    limit = indent
    for prior in reversed(lines[:index]):
        stripped = prior.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        prior_indent = len(prior) - len(stripped)
        if prior_indent >= limit:
            continue
        list_item = stripped.startswith("- ")
        content = stripped[2:].lstrip() if list_item else stripped
        key = re.match(r"^([A-Za-z0-9_-]+):", content)
        if key:
            parents.append((key.group(1), list_item))
            limit = prior_indent
    if list_run and not parents:
        return True
    try:
        steps_at = next(i for i, (key, _) in enumerate(parents) if key == "steps")
    except StopIteration:
        return False
    return list_run or (steps_at > 0 and parents[0][1])


def _workflow_commands(text):
    found, lines = [], text.splitlines()
    for index, line in enumerate(lines):
        number = index + 1
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*?)\s*$", line)
        if not m:
            continue
        if not _is_step_run(lines, index):
            continue
        value = m.group(2).strip()
        if value.startswith("#"):
            value = ""
        if not value or value[0] in "|>":
            key_column = line.index("run:")
            body = []
            for nxt in lines[number:]:
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= key_column:
                    break
                body.append(nxt.strip())
            if value and value[0] in "|>":
                commands = [b for b in body if b and not b.startswith("#")]
                commands = [_strip_shell_comment(b).strip() for b in commands]
                commands = [b for b in commands if b]
            else:
                commands = []
        else:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            commands = [_strip_shell_comment(value).strip()]
            commands = [_strip_shell_comment(c).strip() for c in commands]
            commands = [c for c in commands if c]
        found.append((number, commands))
    return found


def noop_steps(text):
    found = []
    for number, commands in _workflow_commands(text):
        if not commands:
            found.append({"line": number, "run": "<empty>"})
        if commands and all(c in NOOP_RUN for c in commands):
            found.append({"line": number, "run": " ; ".join(commands)})
    return found


def tool_paths(files):
    return sorted(p for p in files if p.startswith("tools/") and p.endswith(CODE_SUFFIXES)
                  and area(p) == "tooling" and not p.endswith("__init__.py"))


def _module(path):
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _names_tool(text, path, unique):
    # The full path always counts. The module name and the basename count only
    # when no other tracked file shares the basename, so neither is ambiguous.
    pattern = r"\b" + re.escape(path) + r"\b"
    if unique:
        module = re.escape(_module(path))
        pattern += (r"|\b(?:import|from)\s+(?:[\w.]+\.)?" + module + r"\b"
                    r"|\bfrom\s+[\w.]+\s+import\s+(?:\([^)]*|[^\n]*)\b" + module + r"\b"
                    r"|\b" + re.escape(path.rsplit("/", 1)[-1]) + r"\b")
    return re.search(pattern, text) is not None



def _tool_for_module(module, tools, unique):
    if not module:
        return None
    module = module.strip(".")
    candidate = module.replace(".", "/") + ".py"
    if candidate in tools:
        return candidate
    basename = module.rsplit(".", 1)[-1]
    if unique.get(basename):
        for path in tools:
            if path.rsplit("/", 1)[-1] == basename + ".py":
                return path
    return None


def _tool_for_token(token, tools):
    token = token.strip().strip("'\"").replace("\\", "/")
    while token.startswith("./"):
        token = token[2:]
    if token in tools:
        return token
    for path in tools:
        if token.endswith("/" + path):
            return path
    return None


def _literal_bool(node):
    try:
        return bool(ast.literal_eval(node))
    except (ValueError, TypeError):
        return None


def _function_is_generator(function):
    """Return whether a function body yields, without entering nested scopes."""
    found = False

    class Finder(ast.NodeVisitor):
        def visit_Yield(self, node):
            nonlocal found
            found = True

        visit_YieldFrom = visit_Yield

        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            return

        def visit_Lambda(self, node):
            return

    finder = Finder()
    for statement in function.body:
        finder.visit(statement)
    return found


def _literal_comparison(left, operator, right):
    """Evaluate a comparison only when both operands are inert literals."""
    try:
        left_value = ast.literal_eval(left)
        right_value = ast.literal_eval(right)
        if isinstance(operator, ast.Eq):
            return left_value == right_value
        if isinstance(operator, ast.NotEq):
            return left_value != right_value
        if isinstance(operator, ast.Lt):
            return left_value < right_value
        if isinstance(operator, ast.LtE):
            return left_value <= right_value
        if isinstance(operator, ast.Gt):
            return left_value > right_value
        if isinstance(operator, ast.GtE):
            return left_value >= right_value
        if isinstance(operator, ast.Is):
            return left_value is right_value
        if isinstance(operator, ast.IsNot):
            return left_value is not right_value
        if isinstance(operator, ast.In):
            return left_value in right_value
        if isinstance(operator, ast.NotIn):
            return left_value not in right_value
    except (ValueError, TypeError):
        pass
    return None


def _is_main_guard(node):
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and node.left.id == "__name__" and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq) and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == "__main__")


def _is_none_guard(node):
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
            and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is None)


def _branch_assigned_names(body):
    """Return names possibly rebound or mutated without entering nested scopes."""
    names = set()

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            names.add(node.name)
            return

        def visit_Lambda(self, node):
            return

        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)

        def visit_Import(self, node):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)

        def visit_Call(self, node):
            mutated = _python_mutated_root_name(node)
            if mutated:
                names.add(mutated)
            self.generic_visit(node)

        def _target_root(self, target):
            root = target
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)

        def visit_Assign(self, node):
            for target in node.targets:
                self._target_root(target)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            self._target_root(node.target)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            self._target_root(node.target)
            self.generic_visit(node)

        def visit_Delete(self, node):
            for target in node.targets:
                self._target_root(target)
            self.generic_visit(node)

    collector = Collector()
    for statement in body:
        collector.visit(statement)
    return names


def _reachable_python_nodes(tree, mode):
    functions, methods = {}, {}

    def register(body, owner=None):
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if owner is None:
                    functions.setdefault(statement.name, statement)
                else:
                    methods.setdefault((owner, statement.name), statement)
            elif isinstance(statement, ast.ClassDef):
                register(statement.body, statement.name)

    register(tree.body)
    test_classes = set()
    if mode == "tests":
        imports = _import_bindings(tree)
        for statement in tree.body:
            if (isinstance(statement, ast.ClassDef)
                    and any(_resolve_import(base, imports) == ("unittest", "TestCase")
                            for base in statement.bases)):
                test_classes.add(statement.name)

    nodes, queue, active_calls = [], [], set()
    active_functions = {}
    function_scope_bindings = {
        id(node): _scope_bindings(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_scope_globals = {
        id(node): _python_scope_global_names(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_scope_nonlocals = {
        id(node): _python_scope_nonlocal_names(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_scope_parents = _python_function_parent_scopes(tree)

    def statement_sequence_terminates(body):
        return any(statement_terminates(statement) for statement in body)

    def statement_terminates(statement):
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            return statement_sequence_terminates(statement.finalbody)
        if not isinstance(statement, ast.If):
            return False
        if _is_main_guard(statement.test):
            branch = statement.body if mode == "script" else statement.orelse
        else:
            truth = _literal_bool(statement.test)
            if truth is None:
                return False
            branch = statement.body if truth else statement.orelse
        return statement_sequence_terminates(branch)

    def active_function_destination(scope, name):
        if name in function_scope_globals.get(scope, set()):
            return None
        if name in function_scope_nonlocals.get(scope, set()):
            parent = function_scope_parents.get(scope)
            while parent is not None:
                if name in function_scope_bindings.get(parent, set()):
                    return parent
                parent = function_scope_parents.get(parent)
        return scope

    def execute(function, owner, activation_line):
        identity = (id(function), owner)
        if identity in active_calls:
            return
        active_calls.add(identity)
        try:
            visit_statement_sequence(function.body, owner, id(function), activation_line)
        finally:
            active_calls.remove(identity)

    def visit_statement_sequence(body, owner, scope, activation_line):
        for statement in body:
            visit(statement, owner, scope, activation_line)
            if statement_terminates(statement):
                break

    def visit(node, owner=None, scope=None, activation_line=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A definition is not a call: only decorators/defaults execute here.
            for decorator in node.decorator_list:
                visit(decorator, owner, scope, activation_line)
            for default in node.args.defaults:
                visit(default, owner, scope, activation_line)
            for default in node.args.kw_defaults:
                if default is not None:
                    visit(default, owner, scope, activation_line)
            if (owner is None and isinstance(node, ast.FunctionDef)
                    and not _function_is_generator(node)):
                active_functions[(scope, node.name)] = node
            if (isinstance(node, ast.FunctionDef)
                    and not _function_is_generator(node)
                    and mode == "tests" and owner in test_classes
                    and (node.name.startswith("test_")
                         or node.name in {"setUp", "tearDown", "setUpClass", "tearDownClass"})):
                queue.append((node, owner, float("inf")))
            return
        if isinstance(node, ast.Lambda):
            # Lambda construction evaluates defaults, but the body remains
            # deferred until an invocation is established.
            for default in node.args.defaults:
                visit(default, owner, scope, activation_line)
            for default in node.args.kw_defaults:
                if default is not None:
                    visit(default, owner, scope, activation_line)
            return
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                visit(decorator, owner, scope, activation_line)
            for base in node.bases:
                visit(base, owner, scope, activation_line)
            for keyword in node.keywords:
                visit(keyword.value, owner, scope, activation_line)
            for statement in node.body:
                visit(statement, node.name, scope, activation_line)
            return
        if isinstance(node, ast.If):
            visit(node.test, owner, scope, activation_line)
            truth = None
            if _is_main_guard(node.test):
                branch = node.body if mode == "script" else node.orelse
            else:
                truth = _literal_bool(node.test)
                if truth is None:
                    if not _is_none_guard(node.test):
                        return
                    names = (_branch_assigned_names(node.body)
                             | _branch_assigned_names(node.orelse))
                    for alternative in (node.body, node.orelse):
                        if names:
                            marker = ast.Delete(
                                targets=[ast.Name(id=name, ctx=ast.Del())
                                         for name in sorted(names)])
                            marker.lineno = (getattr(alternative[0], "lineno", node.lineno)
                                             if alternative else node.lineno)
                            execution_line = (marker.lineno if activation_line is None
                                              else activation_line)
                            nodes.append((marker, scope, execution_line))
                        visit_statement_sequence(
                            alternative, owner, scope, activation_line)
                    if names:
                        marker = ast.Delete(
                            targets=[ast.Name(id=name, ctx=ast.Del())
                                     for name in sorted(names)])
                        marker.lineno = getattr(node, "end_lineno", node.lineno)
                        execution_line = (marker.lineno if activation_line is None
                                          else activation_line)
                        nodes.append((marker, scope, execution_line))
                    return
                else:
                    branch = node.body if truth else node.orelse
            visit_statement_sequence(branch, owner, scope, activation_line)
            return
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                visit(node.iter, owner, scope, activation_line)
            else:
                visit(node.test, owner, scope, activation_line)
            return

        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            # The try body executes, and a finally body executes regardless of
            # whether an exception is raised. Without sound exception-flow
            # matching, neither a handler nor the no-exception else body is
            # definite execution evidence.
            visit_statement_sequence(node.body, owner, scope, activation_line)
            possible_effects = set().union(
                *(_branch_assigned_names(handler.body) for handler in node.handlers),
                _branch_assigned_names(node.orelse))
            if possible_effects:
                marker = ast.Delete(
                    targets=[ast.Name(id=name, ctx=ast.Del())
                             for name in sorted(possible_effects)])
                boundary_statements = [
                    *(handler.body[0] for handler in node.handlers if handler.body),
                    *(node.orelse[:1]),
                    *(node.finalbody[:1]),
                ]
                marker.lineno = min(
                    (getattr(statement, "lineno", node.lineno)
                     for statement in boundary_statements),
                    default=getattr(node, "end_lineno", node.lineno))
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            visit_statement_sequence(node.finalbody, owner, scope, activation_line)
            return

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            # Assignment targets are rebound only after their value expressions
            # finish. A helper invoked by the RHS still resolves the prior
            # function binding.
            if isinstance(node, ast.AugAssign):
                visit(node.target, owner, scope, activation_line)
            if getattr(node, "value", None) is not None:
                visit(node.value, owner, scope, activation_line)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in _target_names(target):
                    active_functions.pop(
                        (active_function_destination(scope, name), name), None)
            execution_line = (getattr(node, "lineno", 0)
                              if activation_line is None else activation_line)
            nodes.append((node, scope, execution_line))
            return

        if isinstance(node, ast.Delete):
            targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                       else [node.target])
            for target in targets:
                for name in _target_names(target):
                    active_functions.pop(
                        (active_function_destination(scope, name), name), None)

        if isinstance(node, ast.IfExp):
            # Conditional expressions evaluate the condition first and only
            # one arm. Unknown conditions do not establish either arm.
            visit(node.test, owner, scope, activation_line)
            truth = _literal_bool(node.test)
            if truth is not None:
                visit(node.body if truth else node.orelse,
                      owner, scope, activation_line)
            return

        if isinstance(node, ast.BoolOp):
            # Boolean operands execute left-to-right and later operands are
            # reachable only when the preceding literal truth values require
            # evaluation. An unknown predecessor is not definite execution.
            for index, value in enumerate(node.values):
                visit(value, owner, scope, activation_line)
                if index == len(node.values) - 1:
                    break
                truth = _literal_bool(value)
                if truth is None:
                    break
                if isinstance(node.op, ast.And) and not truth:
                    break
                if isinstance(node.op, ast.Or) and truth:
                    break
            return

        if isinstance(node, ast.Compare):
            # Chained comparisons evaluate one comparator at a time and stop
            # after the first false result. Unknown comparison truth does not
            # establish execution of a later comparator.
            visit(node.left, owner, scope, activation_line)
            left = node.left
            for operator, comparator in zip(node.ops, node.comparators):
                visit(comparator, owner, scope, activation_line)
                truth = _literal_comparison(left, operator, comparator)
                if truth is not True:
                    break
                left = comparator
            return

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)):
            # Comprehension bodies are deferred until iteration. The outermost
            # iterable expression is evaluated at construction; a generator
            # expression is otherwise entirely deferred. For eager
            # comprehensions, only a statically non-empty inert iterable with
            # literal-true guards establishes one body execution.
            if not node.generators:
                return
            first = node.generators[0]
            visit(first.iter, owner, scope, activation_line)
            if isinstance(node, ast.GeneratorExp):
                return
            try:
                iterable = ast.literal_eval(first.iter)
            except (ValueError, TypeError):
                return
            if not iterable or any(_literal_bool(guard) is not True for guard in first.ifs):
                return
            if len(node.generators) != 1:
                return
            comprehension_scope = id(node)
            if isinstance(node, ast.DictComp):
                visit(node.key, owner, comprehension_scope, activation_line)
                visit(node.value, owner, comprehension_scope, activation_line)
            else:
                visit(node.elt, owner, comprehension_scope, activation_line)
            return

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # Binary operands are evaluated left-to-right before addition.
            # Capture a resolved named/list operand before effects in the
            # later operand can rebind its source name.
            for index, operand in enumerate((node.left, node.right)):
                visit(operand, owner, scope, activation_line)
                marker = ast.Expr(value=operand)
                marker.lineno = getattr(operand, "lineno", getattr(node, "lineno", 0))
                marker._g11_binary_id = id(node)
                marker._g11_binary_index = index
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            return

        if isinstance(node, (ast.List, ast.Tuple)):
            # Sequence elements are evaluated left-to-right. Preserve each
            # static value before a later element's helper effects can rebind
            # its source name.
            for index, element in enumerate(node.elts):
                value_node = element.value if isinstance(element, ast.Starred) else element
                visit(value_node, owner, scope, activation_line)
                marker = ast.Expr(value=value_node)
                marker.lineno = getattr(value_node, "lineno", getattr(node, "lineno", 0))
                marker._g11_sequence_id = id(node)
                marker._g11_sequence_index = index
                marker._g11_sequence_starred = isinstance(element, ast.Starred)
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            return

        if isinstance(node, ast.Call):
            # Python evaluates the callee and every argument before invoking
            # the enclosing call. Helper effects in those expressions must be
            # visible to process attribution at the call itself.
            visit(node.func, owner, scope, activation_line)
            for index, argument in enumerate(node.args):
                visit(argument, owner, scope, activation_line)
                marker = ast.Expr(value=argument)
                marker.lineno = getattr(argument, "lineno", getattr(node, "lineno", 0))
                marker._g11_call_id = id(node)
                marker._g11_argument_index = index
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            for keyword in node.keywords:
                visit(keyword.value, owner, scope, activation_line)
            execution_line = (getattr(node, "lineno", 0)
                              if activation_line is None else activation_line)
            nodes.append((node, scope, execution_line))
            target = node.func
            if isinstance(target, ast.Name):
                function = active_functions.get((scope, target.id))
                if (function is None and scope is not None
                        and target.id not in function_scope_bindings.get(scope, set())):
                    function = active_functions.get((None, target.id))
                if function is not None:
                    execute(function, None, execution_line)
            elif (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                  and target.value.id in {"self", "cls"} and owner):
                method = methods.get((owner, target.attr))
                if (isinstance(method, ast.FunctionDef)
                        and not _function_is_generator(method)):
                    execute(method, owner, execution_line)
            return

        execution_line = (getattr(node, "lineno", 0)
                          if activation_line is None else activation_line)
        nodes.append((node, scope, execution_line))
        for child in ast.iter_child_nodes(node):
            visit(child, owner, scope, activation_line)

    for statement in tree.body:
        visit(statement)
    while queue:
        function, owner, activation_line = queue.pop()
        execute(function, owner, activation_line)
    return nodes


def _python_static_value(node, local, module, source_path, imports, scope_shadowed=()):
    """Resolve only path/command values whose bytes are explicit in the AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return source_path
        if node.id in local:
            return local[node.id]
        if node.id in scope_shadowed:
            return None
        return module.get(node.id)
    if isinstance(node, ast.Attribute):
        if _resolved_import_name(node, imports, scope_shadowed) == "sys.executable":
            return "<python>"
        value = _python_static_value(node.value, local, module, source_path, imports, scope_shadowed)
        if isinstance(value, str) and node.attr == "parent":
            return posixpath.dirname(value)
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
            and node.value.attr == "parents" and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, int):
        value = _python_static_value(node.value.value, local, module, source_path, imports, scope_shadowed)
        if not isinstance(value, str) or node.slice.value < 0:
            return None
        current = value
        for _ in range(node.slice.value + 1):
            parent = posixpath.dirname(current)
            if parent == current:
                return None
            current = parent
        return current
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _python_static_value(node.left, local, module, source_path, imports, scope_shadowed)
        right = _python_static_value(node.right, local, module, source_path, imports, scope_shadowed)
        if isinstance(left, str) and isinstance(right, str):
            return posixpath.normpath(posixpath.join(left, right))
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _python_static_value(node.left, local, module, source_path, imports, scope_shadowed)
        right = _python_static_value(node.right, local, module, source_path, imports, scope_shadowed)
        return _static_add(left, right)
    if isinstance(node, ast.IfExp):
        truth = _literal_bool(node.test)
        if truth is not None:
            selected = node.body if truth else node.orelse
            return _python_static_value(selected, local, module, source_path, imports,
                                        scope_shadowed)
        body = _python_static_value(node.body, local, module, source_path, imports,
                                    scope_shadowed)
        other = _python_static_value(node.orelse, local, module, source_path, imports,
                                     scope_shadowed)
        if body is None or other is None:
            return None
        return _static_alternatives((body, other))
    if isinstance(node, (ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                value = _python_static_value(element.value, local, module, source_path,
                                             imports, scope_shadowed)
                if not isinstance(value, list):
                    values.append(_UNRESOLVED_STATIC)
                    continue
                values.extend(value)
                continue
            value = _python_static_value(element, local, module, source_path, imports, scope_shadowed)
            if value is None:
                values.append(_UNRESOLVED_STATIC)
                continue
            values.append(value)
        return values
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        for value_node in node.values:
            value = _python_static_value(value_node, local, module, source_path, imports, scope_shadowed)
            if value is None:
                return None
            if value:
                return value
        return value if node.values else None
    if not isinstance(node, ast.Call):
        return None
    call_name = _dotted_name(node.func)
    resolved_call = _resolved_import_name(node.func, imports, scope_shadowed)
    builtin_str = (call_name == "str" and "str" not in imports["shadowed"]
                   and "str" not in scope_shadowed)
    if builtin_str and node.args:
        return _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)
    if resolved_call == "pathlib.Path" and node.args:
        values = [_python_static_value(arg, local, module, source_path, imports, scope_shadowed)
                  for arg in node.args]
        if values and all(isinstance(value, str) for value in values):
            return posixpath.normpath(posixpath.join(*values))
        return None
    if resolved_call in {"os.path.join", "posixpath.join"}:
        values = [_python_static_value(arg, local, module, source_path, imports, scope_shadowed)
                  for arg in node.args]
        if values and all(isinstance(value, str) for value in values):
            return posixpath.normpath(posixpath.join(*values))
    if resolved_call in {"os.path.abspath", "os.path.realpath", "posixpath.abspath",
                         "posixpath.realpath"} and node.args:
        return _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)
    if resolved_call in {"os.path.dirname", "posixpath.dirname"} and node.args:
        value = _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)
        return posixpath.dirname(value) if isinstance(value, str) else None
    if resolved_call == "os.environ.get" and len(node.args) >= 2:
        return _python_static_value(node.args[1], local, module, source_path, imports, scope_shadowed)
    if isinstance(node.func, ast.Attribute):
        base = _python_static_value(node.func.value, local, module, source_path, imports, scope_shadowed)
        if node.func.attr in {"resolve", "absolute"}:
            return base
        if node.func.attr == "with_name" and isinstance(base, str) and node.args:
            name = _python_static_value(node.args[0], local, module, source_path, imports, scope_shadowed)
            if isinstance(name, str):
                return posixpath.join(posixpath.dirname(base), name)
    return None


def _python_sequence_static_value(node, evaluated_elements, local, module,
                                  source_path, imports, scope_shadowed=()):
    """Resolve a sequence from values captured at each element evaluation."""
    missing = object()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = evaluated_elements.get(("binary", id(node), 0), missing)
        if left is missing:
            left = _python_sequence_static_value(
                node.left, evaluated_elements, local, module, source_path,
                imports, scope_shadowed)
        right = evaluated_elements.get(("binary", id(node), 1), missing)
        if right is missing:
            right = _python_sequence_static_value(
                node.right, evaluated_elements, local, module, source_path,
                imports, scope_shadowed)
        return _static_add(left, right)
    if not isinstance(node, (ast.List, ast.Tuple)):
        return _python_static_value(
            node, local, module, source_path, imports, scope_shadowed)
    values = []
    for index, element in enumerate(node.elts):
        value_node = element.value if isinstance(element, ast.Starred) else element
        value = evaluated_elements.get((id(node), index), missing)
        if value is missing:
            value = _python_sequence_static_value(
                value_node, evaluated_elements, local, module, source_path,
                imports, scope_shadowed)
        if isinstance(element, ast.Starred):
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(_UNRESOLVED_STATIC)
        else:
            values.append(_UNRESOLVED_STATIC if value is None else value)
    return values


_MUTATING_LIST_METHODS = {"append", "clear", "extend", "insert", "pop", "remove",
                          "reverse", "sort", "__delitem__", "__iadd__", "__imul__",
                          "__setitem__"}


def _python_mutated_root_name(node):
    """Return the directly mutated aggregate root for a supported list method call."""
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
            or node.func.attr not in _MUTATING_LIST_METHODS):
        return None
    root = node.func.value
    while isinstance(root, (ast.Subscript, ast.Attribute)):
        root = root.value
    return root.id if isinstance(root, ast.Name) else None


def _python_scope_global_names(function):
    """Return names declared global in one function, excluding nested scopes."""
    names = set()

    def visit(node):
        if node is not function and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Global):
            names.update(node.names)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(function)
    return names


def _python_scope_nonlocal_names(function):
    """Return names declared nonlocal in one function, excluding nested scopes."""
    names = set()

    def visit(node):
        if node is not function and isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Nonlocal):
            names.update(node.names)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(function)
    return names


def _python_function_parent_scopes(tree):
    """Map each function scope to its nearest enclosing function scope."""
    parents = {}

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            parents[id(node)] = self.stack[-1] if self.stack else None
            self.stack.append(id(node))
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Collector().visit(tree)
    return parents


def _invalidate_python_binding(name, local, module, scope, global_names=(),
                               destination_for_name=None):
    destination = (destination_for_name(name) if destination_for_name is not None
                   else module if scope is None or name in global_names else local)
    prior = destination.get(name, module.get(name))
    if isinstance(prior, list):
        # Previously evaluated arguments hold the same list object. Mark that
        # object unresolved so later in-place mutations cannot retain stale
        # command provenance.
        prior[:] = [_UNRESOLVED_STATIC]
        environments = (destination,) if destination is module else (destination, module)
        for environment in environments:
            for bound_name, bound in list(environment.items()):
                if bound is prior:
                    environment[bound_name] = _UNRESOLVED_STATIC
    destination[name] = _UNRESOLVED_STATIC


def _bind_python_value(node, value, local, module, scope, global_names=(),
                       destination_for_name=None):
    targets = []
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]

    def destination(name):
        if destination_for_name is not None:
            return destination_for_name(name)
        return module if scope is None or name in global_names else local

    def bind(target, target_value):
        if isinstance(target, ast.Subscript):
            root = target.value
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name):
                _invalidate_python_binding(
                    root.id, local, module, scope, global_names,
                    destination_for_name)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(target_value, list) or len(target.elts) != len(target_value):
                for name in _target_names(target):
                    destination(name).pop(name, None)
                return
            for element, element_value in zip(target.elts, target_value):
                bind(element, element_value)
            return
        for name in _target_names(target):
            if target_value is None:
                destination(name).pop(name, None)
            else:
                destination(name)[name] = target_value

    for target in targets:
        bind(target, value)


def _tool_values(value, tools):
    if isinstance(value, _StaticAlternatives):
        choices = [_tool_values(choice, tools) for choice in value]
        return set.intersection(*choices) if choices else set()
    values = value if isinstance(value, list) else [value]
    return {_tool_for_token(item, tools) for item in values if isinstance(item, str)} - {None}


def _command_tool_values(value, tools):
    if isinstance(value, _StaticAlternatives):
        choices = [_command_tool_values(choice, tools) for choice in value]
        return set.intersection(*choices) if choices else set()
    if not isinstance(value, list) or not value:
        return _tool_values(value, tools)
    first = value[0]
    python_interpreters = {"<python>", "python", "python3", "python3.11"}
    shell_interpreters = {"bash", "sh"}
    interpreters = python_interpreters | shell_interpreters
    if first not in interpreters:
        return _tool_values(first, tools)
    arguments = value[1:]
    if first in python_interpreters:
        terminating_options = {"-", "-c", "-m"}
        operand_options = {"-W", "-X", "--check-hash-based-pycs"}
        attached_options = ("-W", "-X")
    else:
        terminating_options = {"-c", "-s"}
        operand_options = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
        attached_options = ()
    consume_operand = False
    parsing_options = True
    for token in arguments:
        if not isinstance(token, str):
            return set()
        if consume_operand:
            consume_operand = False
            continue
        if parsing_options and token == "--":
            parsing_options = False
            continue
        if parsing_options and token in terminating_options:
            return set()
        if parsing_options and token in operand_options:
            consume_operand = True
            continue
        if parsing_options and any(token.startswith(option) and token != option
                                   for option in attached_options):
            continue
        if parsing_options and token.startswith(("-", "+")):
            continue
        path = _tool_for_token(token, tools)
        return {path} if path else set()
    return set()


def _python_execution_wrappers(tree, imports):
    """Return reachable-call argument positions forwarded to a trusted process API."""
    wrappers = {}
    process_calls = {"subprocess.run", "subprocess.call", "subprocess.Popen",
                     "subprocess.check_call", "subprocess.check_output", "os.system"}
    path_calls = {"importlib.util.spec_from_file_location": 1, "runpy.run_path": 0}

    def wrapper_positions(function):
        parameters = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args)]
        positions = set()
        rebound = set()
        aliases = {parameter: parameter for parameter in parameters}
        shadowed = _scope_bindings(function)
        local_imports = _empty_import_bindings()

        def invalidate(name):
            parameter = aliases.get(name)
            if parameter is not None:
                rebound.add(parameter)

        def update_target(target, value=None):
            if isinstance(target, (ast.Subscript, ast.Attribute)):
                root = target
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    root = root.value
                if isinstance(root, ast.Name):
                    invalidate(root.id)
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    update_target(element)
                return
            if not isinstance(target, ast.Name):
                return
            if target.id in parameters:
                rebound.add(target.id)
            aliases.pop(target.id, None)
            if (isinstance(value, ast.Name) and value.id in aliases
                    and target.id not in parameters):
                aliases[target.id] = aliases[value.id]

        def visit_body(body):
            for statement in body:
                if visit(statement):
                    return True
            return False

        def visit(node):
            if node is not function and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                          ast.ClassDef, ast.Lambda)):
                return
            if isinstance(node, ast.If):
                truth = _literal_bool(node.test)
                if truth is None:
                    if not _is_none_guard(node.test):
                        return False
                    body_terminates = visit_body(node.body)
                    other_terminates = visit_body(node.orelse)
                    return bool(node.body and node.orelse
                                and body_terminates and other_terminates)
                branch = node.body if truth else node.orelse
                return visit_body(branch)
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                return False
            if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                body_terminates = visit_body(node.body)
                possible_effects = set().union(
                    *(_branch_assigned_names(handler.body)
                      for handler in node.handlers),
                    _branch_assigned_names(node.orelse))
                for name in possible_effects:
                    invalidate(name)
                finally_terminates = visit_body(node.finalbody)
                return finally_terminates or (body_terminates and not node.handlers)
            if isinstance(node, ast.IfExp):
                visit(node.test)
                truth = _literal_bool(node.test)
                if truth is not None:
                    visit(node.body if truth else node.orelse)
                return False
            if isinstance(node, ast.BoolOp):
                for index, value in enumerate(node.values):
                    visit(value)
                    if index == len(node.values) - 1:
                        break
                    truth = _literal_bool(value)
                    if truth is None:
                        break
                    if isinstance(node.op, ast.And) and not truth:
                        break
                    if isinstance(node.op, ast.Or) and truth:
                        break
                return False
            if isinstance(node, ast.Compare):
                visit(node.left)
                left = node.left
                for operator, comparator in zip(node.ops, node.comparators):
                    visit(comparator)
                    truth = _literal_comparison(left, operator, comparator)
                    if truth is not True:
                        break
                    left = comparator
                return False
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp)):
                if not node.generators:
                    return False
                first = node.generators[0]
                visit(first.iter)
                if isinstance(node, ast.GeneratorExp):
                    return False
                try:
                    iterable = ast.literal_eval(first.iter)
                except (ValueError, TypeError):
                    return False
                if (not iterable
                        or any(_literal_bool(guard) is not True for guard in first.ifs)
                        or len(node.generators) != 1):
                    return False
                target_names = _target_names(first.target)
                saved_aliases = {name: aliases[name] for name in target_names
                                 if name in aliases}
                for name in target_names:
                    aliases.pop(name, None)
                try:
                    if isinstance(node, ast.DictComp):
                        visit(node.key)
                        visit(node.value)
                    else:
                        visit(node.elt)
                finally:
                    aliases.update(saved_aliases)
                return False
            if isinstance(node, ast.Call):
                # Python evaluates the callee, positional arguments and
                # keyword values before invoking the enclosing call. A nested
                # mutation can therefore invalidate forwarding before the
                # process API receives its argument.
                visit(node.func)
                for argument in node.args:
                    visit(argument)
                for keyword in node.keywords:
                    visit(keyword.value)
                mutated_root = _python_mutated_root_name(node)
                if mutated_root:
                    invalidate(mutated_root)
                _update_local_import_bindings(node, local_imports)
                active_imports, active_shadowed = _combined_import_bindings(
                    imports, local_imports, shadowed)
                if node.args:
                    resolved = _resolved_import_name(
                        node.func, active_imports, active_shadowed)
                    forwarded = node.args[0] if resolved in process_calls else None
                    if resolved in path_calls and len(node.args) > path_calls[resolved]:
                        forwarded = node.args[path_calls[resolved]]
                    if isinstance(forwarded, ast.Name):
                        parameter = aliases.get(forwarded.id)
                        if parameter is not None and parameter not in rebound:
                            positions.add(parameters.index(parameter))
                return False
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    update_target(target, node.value)
            elif isinstance(node, ast.AnnAssign):
                update_target(node.target, node.value)
            elif isinstance(node, ast.AugAssign):
                update_target(node.target)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    update_target(target)
            mutated_root = _python_mutated_root_name(node)
            if mutated_root:
                invalidate(mutated_root)
            _update_local_import_bindings(node, local_imports)
            active_imports, active_shadowed = _combined_import_bindings(
                imports, local_imports, shadowed)
            for child in ast.iter_child_nodes(node):
                visit(child)
            return isinstance(node, (ast.Return, ast.Raise))

        visit_body(function.body)
        return positions

    def register(body, containing_scope=None):
        for statement in body:
            if isinstance(statement, ast.FunctionDef):
                if _function_is_generator(statement):
                    continue
                positions = wrapper_positions(statement)
                key = (containing_scope, statement.name)
                if positions:
                    wrappers[key] = (positions, statement.lineno)
                else:
                    wrappers.pop(key, None)
                register(statement.body, id(statement))
            elif isinstance(statement, ast.ClassDef):
                register(statement.body, id(statement))

    register(tree.body)
    return wrappers


def python_tool_references(text, source_path, tools, unique, mode="script"):
    references = set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return references
    imports = _import_bindings(tree)
    wrappers = _python_execution_wrappers(tree, imports)
    scope_shadowed = {id(node): _scope_bindings(node) for node in ast.walk(tree)
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    scope_shadowed.update({
        id(node): set().union(*(_target_names(generator.target)
                                for generator in node.generators))
        for node in ast.walk(tree)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp)) and node.generators
    })
    scope_globals = {id(node): _python_scope_global_names(node) for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    scope_nonlocals = {id(node): _python_scope_nonlocal_names(node) for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    scope_parents = _python_function_parent_scopes(tree)
    module_bindings, scope_bindings, scope_imports = {}, {}, {}
    evaluated_arguments = {}
    evaluated_sequence_elements = {}
    missing_argument = object()
    reachable_nodes = sorted(_reachable_python_nodes(tree, mode), key=lambda item: item[2])
    wrapper_reassignments = {}

    def binding_destination_scope(scope, name):
        if scope is None or name in scope_globals.get(scope, set()):
            return None
        if name in scope_nonlocals.get(scope, set()):
            parent = scope_parents.get(scope)
            while parent is not None:
                owns_binding = (
                    name in scope_shadowed.get(parent, set())
                    and name not in scope_globals.get(parent, set())
                    and name not in scope_nonlocals.get(parent, set()))
                if owns_binding:
                    return parent
                parent = scope_parents.get(parent)
        return scope

    def binding_destination(scope, name):
        destination_scope = binding_destination_scope(scope, name)
        return (module_bindings if destination_scope is None
                else scope_bindings.setdefault(destination_scope, {}))

    wrapper_destination_scope = binding_destination_scope

    for node, scope, execution_line in reachable_nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for name in _target_names(target):
                destination_scope = wrapper_destination_scope(scope, name)
                key = (destination_scope, name)
                if key in wrappers:
                    wrapper_reassignments.setdefault(key, []).append(
                        (execution_line, node.lineno))
    for node, scope, execution_line in reachable_nodes:
        local = scope_bindings.setdefault(scope, {}) if scope is not None else module_bindings
        global_names = scope_globals.get(scope, set())
        local_imports = scope_imports.setdefault(scope, _empty_import_bindings())
        destination_for_name = lambda name: binding_destination(scope, name)
        if scope is not None:
            _update_local_import_bindings(node, local_imports)
        active_imports, _ = _combined_import_bindings(
            imports, local_imports, scope_shadowed.get(scope, set()))
        locally_imported = set(local_imports["modules"]) | set(local_imports["symbols"])
        static_shadowed = (scope_shadowed.get(scope, set())
                           - locally_imported - global_names)
        if hasattr(node, "_g11_binary_id"):
            evaluated_sequence_elements[(
                "binary", node._g11_binary_id, node._g11_binary_index)] = (
                    _python_sequence_static_value(
                        node.value, evaluated_sequence_elements, local,
                        module_bindings, source_path, active_imports,
                        static_shadowed))
            continue
        if hasattr(node, "_g11_sequence_id"):
            value = _python_sequence_static_value(
                node.value, evaluated_sequence_elements, local, module_bindings,
                source_path, active_imports, static_shadowed)
            if node._g11_sequence_starred and isinstance(value, list):
                value = list(value)
            evaluated_sequence_elements[(
                node._g11_sequence_id, node._g11_sequence_index)] = value
            continue
        if hasattr(node, "_g11_call_id"):
            evaluated_arguments[(node._g11_call_id, node._g11_argument_index)] = (
                _python_sequence_static_value(
                    node.value, evaluated_sequence_elements, local,
                    module_bindings, source_path, active_imports,
                    static_shadowed))
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value_node = (ast.BinOp(left=node.target, op=node.op, right=node.value)
                          if isinstance(node, ast.AugAssign) else node.value)
            value = _python_sequence_static_value(
                value_node, evaluated_sequence_elements, local, module_bindings,
                source_path, active_imports, static_shadowed)
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                destination = destination_for_name(node.target.id)
                prior = destination.get(node.target.id, module_bindings.get(node.target.id))
                if isinstance(prior, list) and isinstance(value, list):
                    prior[:] = value
                    value = prior
            _bind_python_value(
                node, value, local, module_bindings, scope, global_names,
                destination_for_name)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                names = _target_names(target)
                if not names and isinstance(target, (ast.Subscript, ast.Attribute)):
                    root = target
                    while isinstance(root, (ast.Subscript, ast.Attribute)):
                        root = root.value
                    if isinstance(root, ast.Name):
                        names = {root.id}
                for name in names:
                    _invalidate_python_binding(
                        name, local, module_bindings, scope, global_names,
                        destination_for_name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                path = _tool_for_module(alias.name, tools, unique)
                if path:
                    references.add(path)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            path = _tool_for_module(module, tools, unique)
            if path:
                references.add(path)
            else:
                for alias in node.names:
                    path = _tool_for_module((module + "." + alias.name).strip("."), tools, unique)
                    if path:
                        references.add(path)
        elif isinstance(node, ast.Call):
            def argument_value(index):
                value = evaluated_arguments.get((id(node), index), missing_argument)
                if value is not missing_argument:
                    return value
                return _python_sequence_static_value(
                    node.args[index], evaluated_sequence_elements, local,
                    module_bindings, source_path, active_imports,
                    static_shadowed)

            mutated_root = _python_mutated_root_name(node)
            if mutated_root:
                _invalidate_python_binding(
                    mutated_root, local, module_bindings, scope, global_names,
                    destination_for_name)
            call_name = _resolved_import_name(node.func, active_imports, static_shadowed)
            if isinstance(node.func, ast.Name):
                key = (scope, node.func.id)
                wrapper = wrappers.get(key)
                if wrapper is None and node.func.id not in static_shadowed:
                    key = (None, node.func.id)
                    wrapper = wrappers.get(key)
                call_order = (execution_line, node.lineno)
                wrapper_order = (wrapper[1], -1) if wrapper is not None else None
                reassigned = (wrapper is not None and any(
                    wrapper_order < order < call_order
                    for order in wrapper_reassignments.get(key, ())))
                positions = (wrapper[0] if wrapper is not None
                             and not reassigned and call_order > wrapper_order else ())
                for position in positions:
                    if position < len(node.args):
                        value = argument_value(position)
                        references.update(_command_tool_values(value, tools))
            if call_name == "importlib.util.spec_from_file_location" and len(node.args) >= 2:
                value = argument_value(1)
                references.update(_tool_values(value, tools))
                continue
            if call_name == "runpy.run_path" and node.args:
                value = argument_value(0)
                references.update(_tool_values(value, tools))
                continue
            if call_name in {"subprocess.run", "subprocess.call", "subprocess.Popen",
                             "subprocess.check_call", "subprocess.check_output", "os.system",
                             "runpy.run_path"}:
                for index, arg in enumerate(node.args):
                    value = argument_value(index)
                    if isinstance(value, (list, _StaticAlternatives)):
                        references.update(_command_tool_values(value, tools))
                        continue
                    if isinstance(arg, (ast.List, ast.Tuple)):
                        tokens = ([elt.value for elt in arg.elts]
                                  if all(isinstance(elt, ast.Constant)
                                         and isinstance(elt.value, str) for elt in arg.elts)
                                  else [])
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        try:
                            tokens = shlex.split(arg.value)
                        except ValueError:
                            tokens = []
                    else:
                        tokens = []
                    references.update(_command_tool_values(tokens, tools))
    return references


def _strip_js_comments(text):
    out = []
    i = 0
    quote = None
    line_comment = False
    block_comment = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                out.extend((" ", " "))
                block_comment = False
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
            continue
        if quote:
            if quote == "`":
                if ch == "\\" and i + 1 < len(text):
                    out.extend((" ", " "))
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                out.append("\n" if ch == "\n" else " ")
                i += 1
                continue
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'" + chr(96):
            quote = ch
            out.append(" " if ch == chr(96) else ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            out.extend((" ", " "))
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            out.extend((" ", " "))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def js_tool_references(text, source_path, tools):
    references = set()
    visible = _strip_js_comments(text)
    pattern = re.compile(r"(?m)^\s*import\s+(?:[^'\";]+?\s+from\s+)?['\"]([^'\"]+)['\"]\s*;?")
    for match in pattern.finditer(visible):
        specifier = match.group(1)
        if specifier.startswith("."):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), specifier))
        else:
            target = specifier
        if target in tools:
            references.add(target)
    bindings = {}
    url_binding = re.compile(
        r"\b(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:fileURLToPath\s*\(\s*)?new\s+URL\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*"
        r"import\.meta\.url\s*\)\s*\)?")
    for match in url_binding.finditer(visible):
        specifier = match.group(3)
        if not specifier.startswith("."):
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), specifier))
        if target in tools:
            searchable, _function_scopes = _js_function_scopes(visible)
            function_blocks = {(scope[2], scope[3]) for scope in _function_scopes}
            declaration_blocks = [block for block in _js_block_scopes(searchable)
                                  if block not in function_blocks
                                  and block[0] < match.start() < block[1]]
            declaration_block = (max(declaration_blocks, key=lambda item: item[0])
                                 if match.group(1) != "var" and declaration_blocks
                                 else None)
            bindings.setdefault(match.group(2), []).append(
                (match.end(), target, _js_function_scope_key(visible, match.start()),
                 declaration_block))
    child_names = set()
    named_import = re.compile(
        r"\bimport\s*\{([^}]*)\}\s*from\s*['\"](?:node:)?child_process['\"]")
    allowed_child_calls = {"spawnSync", "spawn", "execFileSync", "execFile"}
    for match in named_import.finditer(visible):
        for item in match.group(1).split(","):
            parts = re.split(r"\s+as\s+", item.strip())
            if parts and parts[0] in allowed_child_calls:
                child_names.add(parts[-1])
    for call_start, callee, arguments in _js_call_arguments(visible, child_names):
        activation_points = _js_call_activation_points(visible, call_start)
        if not activation_points:
            continue
        if _js_parameter_shadows(visible, call_start, callee):
            continue
        if _js_local_declaration_shadows(visible, call_start, callee):
            continue
        call_scope = _js_function_scope_key(visible, call_start)
        identifiers = _js_expression_identifiers(arguments)
        for name, declarations in bindings.items():
            if _js_parameter_shadows(visible, call_start, name):
                continue
            available = [declaration for declaration in declarations
                         if declaration[0] <= call_start
                         and (declaration[2] is None or declaration[2] == call_scope)
                         and (declaration[3] is None
                              or declaration[3][0] < call_start < declaration[3][1])]
            if name not in identifiers or not available:
                continue
            declaration_end, target, declaration_scope, _declaration_block = available[-1]
            if _js_local_declaration_shadows(
                    visible, call_start, name, declaration_end):
                continue
            assignment = re.compile(
                r"(?<![\w$.])" + re.escape(name) +
                r"(?![\w$])\s*(?:=(?!=|>)|\+\+|--|(?:\+|-|\*|/|%|&|\||\^|<<|>>)\s*=)")
            cutoffs = ({call_start} if declaration_scope is not None
                       else activation_points)
            if any(not assignment.search(visible[declaration_end:cutoff])
                   for cutoff in cutoffs if cutoff >= declaration_end):
                references.add(target)
    return references


def _js_expression_identifiers(text):
    """Return identifiers used as expressions, excluding strings and property names."""
    masked = []
    quote = None
    escaped = False
    for char in text:
        if quote:
            masked.append("\n" if char == "\n" else " ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'", "`"}:
            quote = char
            masked.append(" ")
        else:
            masked.append(char)
    visible = "".join(masked)
    identifiers = set()
    pattern = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)(?![\w$])")
    for match in pattern.finditer(visible):
        before = visible[:match.start()].rstrip()
        after = visible[match.end():].lstrip()
        if (before and before[-1] == ".") or (after and after[0] == ":"):
            continue
        identifiers.add(match.group(1))
    return identifiers


def _js_call_arguments(text, names):
    if not names:
        return []
    masked = []
    quote = None
    escaped = False
    for char in text:
        if quote:
            masked.append("\n" if char == "\n" else " ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'", "`"}:
            quote = char
            masked.append(" ")
        else:
            masked.append(char)
    searchable = "".join(masked)
    starts = re.compile(
        r"(?<![\w$.])(" + "|".join(map(re.escape, sorted(names))) + r")(?![\w$])\s*\(")
    arguments = []
    for match in starts.finditer(searchable):
        start = match.end()
        depth, quote, escaped = 1, None, False
        for index in range(start, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {'"', "'", "`"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    arguments.append((match.start(), match.group(1), text[start:index]))
                    break
    return arguments


def _js_matching_delimiter(searchable, start, opening, closing):
    if start >= len(searchable) or searchable[start] != opening:
        return None
    depth = 0
    for index in range(start, len(searchable)):
        if searchable[index] == opening:
            depth += 1
        elif searchable[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _js_top_level_parts(text, delimiter=","):
    """Split JavaScript binding syntax without entering nested patterns."""
    parts = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if char in depths:
            depths[char] += 1
        elif char in closing:
            opening = closing[char]
            depths[opening] = max(0, depths[opening] - 1)
        elif char == delimiter and not any(depths.values()):
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _js_top_level_index(text, needle):
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if char in depths:
            depths[char] += 1
        elif char in closing:
            opening = closing[char]
            depths[opening] = max(0, depths[opening] - 1)
        elif char == needle and not any(depths.values()):
            return index
    return -1


def _js_binding_names(pattern):
    """Return identifiers bound by one JavaScript parameter pattern."""
    pattern = pattern.strip()
    if pattern.startswith("..."):
        pattern = pattern[3:].strip()
    assignment = _js_top_level_index(pattern, "=")
    if assignment >= 0:
        pattern = pattern[:assignment].strip()
    if re.fullmatch(r"[A-Za-z_$][\w$]*", pattern):
        return {pattern}
    if len(pattern) >= 2 and pattern[0] == "[" and pattern[-1] == "]":
        return set().union(*(_js_binding_names(part)
                             for part in _js_top_level_parts(pattern[1:-1])))
    if len(pattern) >= 2 and pattern[0] == "{" and pattern[-1] == "}":
        names = set()
        for part in _js_top_level_parts(pattern[1:-1]):
            part = part.strip()
            if not part:
                continue
            if part.startswith("..."):
                names.update(_js_binding_names(part))
                continue
            colon = _js_top_level_index(part, ":")
            names.update(_js_binding_names(part[colon + 1:] if colon >= 0 else part))
        return names
    return set()


def _js_parameter_bindings(parameters):
    return set().union(*(_js_binding_names(part)
                         for part in _js_top_level_parts(parameters)))


def _js_function_scopes(text):
    """Return masked text and bounded block-bodied function/arrow scopes."""
    searchable = []
    quote = None
    escaped = False
    for char in text:
        if quote:
            searchable.append("\n" if char == "\n" else " ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'", "`"}:
            quote = char
            searchable.append(" ")
        else:
            searchable.append(char)
    searchable = "".join(searchable)

    open_scopes = []
    for function in re.finditer(r"\bfunction\b", searchable):
        open_paren = searchable.find("(", function.end())
        if open_paren < 0:
            continue
        header = searchable[function.end():open_paren]
        if "{" in header or ";" in header:
            continue
        close_paren = _js_matching_delimiter(searchable, open_paren, "(", ")")
        if close_paren is None:
            continue
        open_brace = close_paren + 1
        while open_brace < len(searchable) and searchable[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(searchable) or searchable[open_brace] != "{":
            continue
        open_scopes.append((open_paren + 1, close_paren, open_brace))

    arrow = re.compile(r"(?:\(([^()]*)\)|([A-Za-z_$][\w$]*))\s*=>\s*\{")
    for match in arrow.finditer(searchable):
        param_start = match.start(1) if match.group(1) is not None else match.start(2)
        param_end = match.end(1) if match.group(1) is not None else match.end(2)
        open_scopes.append((param_start, param_end, match.end() - 1))

    scopes = []
    for param_start, param_end, open_brace in open_scopes:
        close_brace = _js_matching_delimiter(searchable, open_brace, "{", "}")
        if close_brace is not None:
            scopes.append((param_start, param_end, open_brace, close_brace))
    return searchable, scopes


def _js_function_scope_key(text, position):
    _searchable, scopes = _js_function_scopes(text)
    containing = [(open_brace, close_brace) for _start, _end, open_brace, close_brace in scopes
                  if open_brace < position < close_brace]
    return max(containing, key=lambda item: item[0]) if containing else None


def _js_block_scopes(searchable):
    """Return balanced lexical brace ranges from already masked JavaScript."""
    stack = []
    scopes = []
    for index, char in enumerate(searchable):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            scopes.append((stack.pop(), index))
    return scopes


def _js_call_activation_points(text, position):
    """Return module-root activation points for one reachable process call."""
    searchable, scopes = _js_function_scopes(text)
    trusted_test_names = set()
    for match in re.finditer(
            r"\bimport\s+([A-Za-z_$][\w$]*)\s+from\s+['\"]node:test['\"]", text):
        trusted_test_names.add(match.group(1))
    for match in re.finditer(
            r"\bimport\s*\{([^}]*)\}\s*from\s*['\"]node:test['\"]", text):
        for item in match.group(1).split(","):
            parts = re.split(r"\s+as\s+", item.strip())
            if parts and parts[0] == "test":
                trusted_test_names.add(parts[-1])

    def in_literal_false_block(candidate):
        pattern = re.compile(r"\bif\s*\(\s*false\s*\)\s*\{")
        for match in pattern.finditer(searchable):
            open_brace = match.end() - 1
            close_brace = _js_matching_delimiter(searchable, open_brace, "{", "}")
            if close_brace is not None and open_brace < candidate < close_brace:
                return True
        return False

    def containing_named_function(candidate):
        containing = sorted(
            (scope for scope in scopes if scope[2] < candidate < scope[3]),
            key=lambda item: item[2], reverse=True)
        if not containing:
            return None
        param_start, _param_end, open_brace, close_brace = containing[0]
        header = re.search(
            r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\($", searchable[:param_start])
        if header is None:
            prefix = searchable[:open_brace + 1]
            for name in trusted_test_names:
                callback = re.compile(
                    r"(?<![\w$.])" + re.escape(name) +
                    r"(?![\w$])\s*\([^{};]*=>\s*\{$")
                match = callback.search(prefix)
                if match:
                    call_start = match.start()
                    if (_js_parameter_shadows(searchable, call_start, name)
                            or _js_local_declaration_shadows(searchable, call_start, name)):
                        return ()
                    return ("test-callback", call_start, open_brace, close_brace, name)
            return ()
        return header.group(1), header.start(1), open_brace, close_brace

    def binding_active(name, definition_end, definition_scope, activation):
        if activation <= definition_end:
            return True
        changed = re.compile(
            r"(?<![\w$.])" + re.escape(name) +
            r"(?![\w$])\s*(?:=(?!=|>)|\+\+|--|(?:\+|-|\*|/|%|&|\||\^|<<|>>)\s*=)"
            r"|\b(?:const|let|var|function|class)\s+" + re.escape(name) +
            r"(?![\w$])")
        for match in changed.finditer(searchable, definition_end, activation):
            if _js_function_scope_key(searchable, match.start()) == definition_scope:
                return False
        return True

    def reachable(candidate, seen, binding_guards=()):
        if in_literal_false_block(candidate):
            return set()
        function = containing_named_function(candidate)
        if function is None:
            return ({candidate} if all(binding_active(*guard, candidate)
                                       for guard in binding_guards) else set())
        if not function:
            return set()
        if len(function) == 5:
            _kind, call_start, open_brace, close_brace, name = function
            identity = (name, open_brace, close_brace)
            if identity in seen:
                return set()
            return reachable(call_start, seen | {identity}, binding_guards)
        name, definition_start, open_brace, close_brace = function
        identity = (name, open_brace, close_brace)
        if identity in seen:
            return set()
        definition_scope = _js_function_scope_key(searchable, definition_start)
        guard = (name, close_brace + 1, definition_scope)
        calls = re.compile(
            r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])\s*\(")
        activations = set()
        for call in calls.finditer(searchable):
            if call.start() == definition_start:
                continue
            if open_brace < call.start() < close_brace:
                continue
            if _js_helper_call_shadowed(
                    searchable, call.start(), name, definition_start):
                continue
            activations.update(reachable(
                call.start(), seen | {identity}, binding_guards + (guard,)))
        return activations

    return reachable(position, set())


def _js_call_is_reachable(text, position):
    """Bound child-process calls to literal-live blocks and invoked helpers."""
    return bool(_js_call_activation_points(text, position))


def _js_parameter_shadows(text, position, name):
    """Return true when a containing function binds the imported callee name."""
    searchable, scopes = _js_function_scopes(text)

    for param_start, param_end, open_brace, close_brace in scopes:
        if not (open_brace < position < close_brace):
            continue
        if name in _js_parameter_bindings(searchable[param_start:param_end]):
            return True
    return False


def _js_local_declaration_shadows(text, position, name,
                                  after_position=None):
    """Return true for a declaration in a containing lexical block."""
    searchable, function_scopes = _js_function_scopes(text)
    blocks = _js_block_scopes(searchable)
    containing = sorted(
        (block for block in blocks if block[0] < position < block[1]),
        key=lambda item: item[0], reverse=True)
    if not containing:
        return False
    direct_pattern = re.compile(
        r"\b(?:function|class)\s+" + re.escape(name) + r"(?![\w$])")
    for block in containing:
        for declaration in direct_pattern.finditer(searchable, block[0] + 1, block[1]):
            if after_position is not None and declaration.start() < after_position:
                continue
            declaration_blocks = [candidate for candidate in blocks
                                  if candidate[0] < declaration.start() < candidate[1]]
            declaration_block = (max(declaration_blocks, key=lambda item: item[0])
                                 if declaration_blocks else None)
            if declaration_block == block:
                return True
        variable_pattern = re.compile(r"\b(?:const|let|var)\b")
        for declaration in variable_pattern.finditer(searchable, block[0] + 1, block[1]):
            if after_position is not None and declaration.start() < after_position:
                continue
            declaration_blocks = [candidate for candidate in blocks
                                  if candidate[0] < declaration.start() < candidate[1]]
            declaration_block = (max(declaration_blocks, key=lambda item: item[0])
                                 if declaration_blocks else None)
            if declaration_block != block:
                continue
            start = declaration.end()
            simple = re.match(r"\s*([A-Za-z_$][\w$]*)", searchable[start:block[1]])
            if simple is not None and simple.group(1) == name:
                return True
            depths = {"(": 0, "[": 0, "{": 0}
            closing = {")": "(", "]": "[", "}": "{"}
            end = block[1]
            for index in range(start, block[1]):
                char = searchable[index]
                if char in depths:
                    depths[char] += 1
                elif char in closing:
                    opening = closing[char]
                    depths[opening] = max(0, depths[opening] - 1)
                elif char == ";" and not any(depths.values()):
                    end = index
                    break
            bindings = set().union(*(
                _js_binding_names(part)
                for part in _js_top_level_parts(searchable[start:end])
            ))
            if name in bindings:
                return True
    return False


def _js_helper_call_shadowed(text, position, name, definition_start):
    """Resolve a helper call against the nearest lexical declaration scope."""
    if _js_parameter_shadows(text, position, name):
        return True
    searchable, scopes = _js_function_scopes(text)
    containing = sorted(
        (scope for scope in scopes if scope[2] < position < scope[3]),
        key=lambda item: item[2], reverse=True)
    pattern = re.compile(
        r"\b(?:const|let|var|function|class)\s+(" + re.escape(name) +
        r")(?![\w$])")
    for scope in containing:
        declarations = []
        for declaration in pattern.finditer(searchable, scope[2] + 1, scope[3]):
            declaration_scopes = [candidate for candidate in scopes
                                  if candidate[2] < declaration.start() < candidate[3]]
            declaration_scope = (max(declaration_scopes, key=lambda item: item[2])
                                 if declaration_scopes else None)
            if declaration_scope == scope:
                declarations.append(declaration)
        if declarations:
            return not any(declaration.start(1) == definition_start
                           for declaration in declarations)
    return False


def _heredoc_opener(command):
    match = re.search(r"<<(?P<strip>-?)(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))",
                      command)
    if not match:
        return None
    delimiter = match.group("single") or match.group("double") or match.group("bare")
    return delimiter, match.group("strip") == "-"


def workflow_tool_references(text, tools):
    references = set()
    for _number, commands in _workflow_commands(text):
        heredoc = None
        for command in commands:
            if heredoc:
                delimiter, strip_tabs = heredoc
                candidate = command.lstrip("\t") if strip_tabs else command
                if candidate.strip() == delimiter:
                    heredoc = None
                continue
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = []
            if tokens:
                if tokens[0] in {"python", "python3", "python3.11", "bash", "sh"}:
                    if "-c" not in tokens[1:]:
                        for index, token in enumerate(tokens[1:], 1):
                            if token == "-m" and index + 1 < len(tokens):
                                module_path = tokens[index + 1].replace(".", "/") + ".py"
                                if module_path in tools:
                                    references.add(module_path)
                                break
                            if token.startswith("-"):
                                continue
                            path = _tool_for_token(token, tools)
                            if path:
                                references.add(path)
                            break
                else:
                    path = _tool_for_token(tokens[0], tools)
                    if path:
                        references.add(path)
            opener = _heredoc_opener(command)
            if opener:
                heredoc = opener
    return references


def _markdown_visible(text):
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def tool_report(files, texts, all_paths):
    tools = tool_paths(files)
    basenames = {}
    for p in all_paths:
        name = p.rsplit("/", 1)[-1]
        basenames[name] = basenames.get(name, 0) + 1
    unique = {p: basenames[p.rsplit("/", 1)[-1]] == 1 for p in tools}
    unique_modules = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0]: unique[p] for p in tools}
    docs = [p for p in texts if p.endswith(".md")]
    exercisers = [p for p in texts if area(p) == "tests" or
                  (p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")))]
    direct, tool_dependencies = {p: [] for p in tools}, {}
    for path in tools:
        tool_dependencies[path] = python_tool_references(texts.get(path, ""), path, tools, unique_modules,
                                                         mode="script")
    for exerciser in exercisers:
        if exerciser.endswith(".py"):
            refs = python_tool_references(texts[exerciser], exerciser, tools, unique_modules, mode="tests")
        elif exerciser.endswith((".mjs", ".js", ".cjs")):
            refs = js_tool_references(texts[exerciser], exerciser, tools)
        else:
            refs = workflow_tool_references(texts[exerciser], tools)
        for path in refs:
            direct[path].append(exerciser)
    report = {}
    for path in tools:
        documented = [d for d in docs if _names_tool(_markdown_visible(texts[d]), path, unique[path])]
        direct_paths = direct[path]
        docstring = None
        if path.endswith(".py") and path in texts:
            try:
                docstring = bool(ast.get_docstring(ast.parse(files[path])))
            except (SyntaxError, ValueError):
                docstring = False
        report[path] = {"documented_by": documented[:5], "documented": bool(documented),
                        "referenced_by": direct_paths[:5], "referenced": bool(direct_paths),
                        "module_docstring": docstring}
    changed = True
    while changed:
        changed = False
        for path in tools:
            if report[path]["referenced"] or path not in texts:
                continue
            for other in tools:
                if other != path and report[other]["referenced"] and other in texts and path in tool_dependencies.get(other, set()):
                    report[path]["referenced"] = True
                    report[path]["referenced_by"] = ["via " + other]
                    changed = True
                    break
    return report


def census(files, all_paths=None):
    """Census of a {path: bytes} tree. Pure: no I/O. `all_paths` lists every
    tracked path (default: the keys of `files`) for basename ambiguity."""
    findings, texts = [], {}
    for path in sorted(files):
        raw = files[path]
        kind = area(path)
        if not path.endswith(SCANNED):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"path": path, "area": kind, "class": "undecodable", "line": 0, "detail": "not UTF-8"})
            continue
        texts[path] = text
        if path.endswith(".py"):
            try:
                tree = ast.parse(raw, filename=path)
                markers, mentions = python_words(raw)
            except (SyntaxError, ValueError, tokenize.TokenError) as exc:
                findings.append({"path": path, "area": kind, "class": "unparseable", "line": 0, "detail": str(exc)[:120]})
                continue
            imports = _import_bindings(tree)
            for stub in python_stubs(tree, imports):
                findings.append({"path": path, "area": kind, "class": "stub" if not stub["context"] else "stub-context",
                                 "line": stub["line"], "detail": "%s: %s%s" % (stub["name"], stub["kind"],
                                                                              "; " + stub["context"] if stub["context"] else "")})
        elif path.endswith((".mjs", ".js", ".cjs", ".sh")):
            markers, mentions = text_words(text, path)
        elif path.endswith((".yml", ".yaml")):
            markers, mentions = yaml_words(text)
        else:
            markers, mentions = [], []
        findings.extend({"path": path, "area": kind, "class": "marker", "line": m["line"], "detail": m["word"]} for m in markers)
        findings.extend({"path": path, "area": kind, "class": "mention", "line": m["line"], "detail": m["word"]} for m in mentions)
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
            findings.extend({"path": path, "area": kind, "class": "noop", "line": n["line"], "detail": n["run"]}
                            for n in noop_steps(text))
    tools = tool_report(files, texts, sorted(all_paths if all_paths is not None else files))
    failing = [f for f in findings if f["area"] in PRODUCTION
               and f["class"] in {"stub", "marker", "noop", "undecodable", "unparseable"}]
    failing_tools = sorted(p for p, t in tools.items() if not (t["documented"] and t["referenced"]))
    return {"findings": findings, "tools": tools, "failing": failing, "failing_tools": failing_tools,
            "status": "FAIL" if failing or failing_tools else "PASS"}


def read_tree(root, rev):
    listing = subprocess.run(["git", "-C", root, "ls-tree", "-r", "-z", "--full-tree", rev],
                             check=True, capture_output=True).stdout
    entries, skipped, tracked = [], [], []
    for item in listing.split(b"\0"):
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, sha = meta.split()
        path = path.decode("utf-8", "surrogateescape")
        if path.endswith(SCANNED) and (kind != b"blob" or mode == b"120000"):
            raise ValueError("in-scope non-regular tracked path %s (mode %s, type %s)" %
                             (path, mode.decode(), kind.decode()))
        if kind == b"blob" and mode != b"120000":
            if path.endswith(SCANNED):
                entries.append((path, sha.decode()))
            tracked.append(path)
        else:
            skipped.append({"path": path, "mode": mode.decode(), "type": kind.decode()})
    batch = subprocess.run(["git", "-C", root, "cat-file", "--batch"], check=True, capture_output=True,
                           input=b"".join(sha.encode() + b"\n" for _p, sha in entries)).stdout
    files, at = {}, 0
    for path, sha in entries:
        end = batch.index(b"\n", at)
        header = batch[at:end].split()
        if len(header) != 3 or header[0].decode() != sha:
            raise ValueError("unexpected git cat-file header for " + path)
        size = int(header[2])
        files[path] = batch[end + 1:end + 1 + size]
        at = end + 1 + size + 1
    return files, skipped, tracked


def main(argv=None):
    parser = argparse.ArgumentParser(description="CBZ-0 G11 census: stub scan and tool documentation check.")
    parser.add_argument("--root", default=".", help="git repository (default: current directory)")
    parser.add_argument("--rev", default="HEAD", help="revision to read (default: HEAD)")
    parser.add_argument("--json", help="write the full report to this file")
    args = parser.parse_args(argv)
    try:
        commit = subprocess.run(["git", "-C", args.root, "rev-parse", "--verify", args.rev + "^{commit}"],
                                check=True, capture_output=True, text=True).stdout.strip()
        files, skipped, tracked = read_tree(args.root, commit)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print("G11_CENSUS_ERROR: cannot read revision %s: %s" % (args.rev, exc), file=sys.stderr)
        return 2
    report = census(files, tracked)
    report.update(commit=commit, files=len(files), tracked=len(tracked), skipped=skipped)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
    for f in report["failing"]:
        print("%s:%d: %s %s (%s)" % (f["path"], f["line"], f["area"], f["class"], f["detail"]))
    for path in report["failing_tools"]:
        t = report["tools"][path]
        print("%s: tool %s" % (path, ", ".join(w for w, ok in (("undocumented", t["documented"]),
                                                              ("unexercised", t["referenced"])) if not ok)))
    counts = {}
    for f in report["findings"]:
        counts[f["class"]] = counts.get(f["class"], 0) + 1
    print("G11_CENSUS_%s: commit %s, %d files, %d tools, findings %s, failing %d, failing tools %d"
          % (report["status"], commit, len(files), len(report["tools"]), json.dumps(counts, sort_keys=True),
             len(report["failing"]), len(report["failing_tools"])))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
