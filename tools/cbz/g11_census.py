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
_UNKNOWN_AGGREGATE_EFFECT = "@g11-unknown-aggregate-mutation"


class _StaticAlternatives(tuple):
    """Resolved mutually exclusive values whose active branch is not statically known."""


class _StaticTuple(tuple):
    """Resolved tuple elements with immutable alias semantics."""


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
            elif (isinstance(left_value, _StaticTuple)
                  and isinstance(right_value, _StaticTuple)):
                results.append(_StaticTuple((*left_value, *right_value)))
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
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            shadowed.update(_target_names(node.target))
        elif isinstance(node, ast.AugAssign):
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
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target] if node.value is not None else []
    elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        # Synthetic Delete nodes carry fail-closed invalidations for names
        # that a possible control-flow path may rebind.  They must invalidate
        # trusted import identities as well as static values and wrappers.
        targets = node.targets
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
    for number, commands in _workflow_reference_commands(text):
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


def _python_call_signature_binds(function, call, implicit_positional=0):
    """Prove that a statically shaped call can enter this function body."""
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        return False
    if any(keyword.arg is None for keyword in call.keywords):
        return False
    positional_only = [argument.arg for argument in function.args.posonlyargs]
    positional_or_keyword = [argument.arg for argument in function.args.args]
    positional = positional_only + positional_or_keyword
    supplied = implicit_positional + len(call.args)
    if supplied > len(positional) and function.args.vararg is None:
        return False
    bound = set(positional[:min(supplied, len(positional))])
    keyword_only = {argument.arg for argument in function.args.kwonlyargs}
    seen_keywords = set()
    for keyword in call.keywords:
        name = keyword.arg
        if name in seen_keywords:
            return False
        seen_keywords.add(name)
        if name in positional_only:
            if function.args.kwarg is None:
                return False
            continue
        if name in positional_or_keyword or name in keyword_only:
            if name in bound:
                return False
            bound.add(name)
        elif function.args.kwarg is None:
            return False
    required_positional = set(
        positional[:len(positional) - len(function.args.defaults)])
    required_keyword_only = {
        argument.arg
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults)
        if default is None
    }
    return required_positional <= bound and required_keyword_only <= bound


def _python_call_argument_bindings(function, call, implicit_positional=0):
    """Map supported explicit call arguments to their formal parameter names."""
    positional = ([argument.arg for argument in function.args.posonlyargs]
                  + [argument.arg for argument in function.args.args])
    bindings = []
    for index, _argument in enumerate(call.args):
        formal_index = implicit_positional + index
        if formal_index < len(positional):
            bindings.append((positional[formal_index], index))
    bindable_keywords = (
        {argument.arg for argument in function.args.args}
        | {argument.arg for argument in function.args.kwonlyargs})
    for keyword in call.keywords:
        if keyword.arg in bindable_keywords:
            bindings.append((keyword.arg, ("keyword", keyword.arg)))
    return bindings


def _python_function_defaults(function):
    """Return formal names paired with their definition-time default nodes."""
    positional = ([argument.arg for argument in function.args.posonlyargs]
                  + [argument.arg for argument in function.args.args])
    positional_defaults = zip(
        positional[len(positional) - len(function.args.defaults):],
        function.args.defaults)
    keyword_defaults = (
        (argument.arg, default)
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults)
        if default is not None)
    return [*positional_defaults, *keyword_defaults]


def _python_call_aggregate_bindings(function, call, implicit_positional=0):
    """Describe default, variadic-positional and variadic-keyword bindings."""
    positional_only = [argument.arg for argument in function.args.posonlyargs]
    positional_or_keyword = [argument.arg for argument in function.args.args]
    positional = positional_only + positional_or_keyword
    fixed_explicit = max(0, len(positional) - implicit_positional)
    bound = set(positional[:implicit_positional])
    bound.update(positional[implicit_positional:
                            implicit_positional + min(len(call.args), fixed_explicit)])
    keyword_only = {argument.arg for argument in function.args.kwonlyargs}
    for keyword in call.keywords:
        if keyword.arg in positional_or_keyword or keyword.arg in keyword_only:
            bound.add(keyword.arg)

    defaults = [(formal, id(function), formal)
                for formal, _node in _python_function_defaults(function)
                if formal not in bound]
    varargs = None
    if function.args.vararg is not None:
        keys = list(range(fixed_explicit, len(call.args)))
        varargs = (function.args.vararg.arg, id(call), keys)
    kwargs = None
    if function.args.kwarg is not None:
        fixed_keywords = set(positional_or_keyword) | keyword_only
        entries = [(keyword.arg, ("keyword", keyword.arg))
                   for keyword in call.keywords
                   if keyword.arg not in fixed_keywords
                   or keyword.arg in positional_only]
        kwargs = (function.args.kwarg.arg, id(call), entries)
    return defaults, varargs, kwargs


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


def _literal_iterable_nonemptiness(node):
    """Return literal iterable nonemptiness, or None when runtime-dependent."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if type(value) not in {list, tuple, set, dict, str, bytes}:
        return False
    return bool(value)


def _literal_nonempty_iterable(node):
    """Return whether a bounded literal is proven to iterate at least once."""
    return _literal_iterable_nonemptiness(node) is True


def _comprehension_possible_effect_expressions(node):
    """Return deferred eager-comprehension expressions that may execute."""
    expressions = []
    for index, generator in enumerate(node.generators):
        if index:
            expressions.append(generator.iter)
        expressions.extend(generator.ifs)
    if isinstance(node, ast.DictComp):
        expressions.extend((node.key, node.value))
    else:
        expressions.append(node.elt)
    return expressions


def _branch_assigned_names(body, call_effects=None):
    """Return names possibly rebound or mutated without entering nested scopes."""
    names = set()

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)
            for expression in node.decorator_list:
                self.visit(expression)
            for expression in node.args.defaults:
                self.visit(expression)
            for expression in node.args.kw_defaults:
                if expression is not None:
                    self.visit(expression)
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            names.add(node.name)
            for expression in node.decorator_list:
                self.visit(expression)
            for expression in node.bases:
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)
            for statement in node.body:
                self.visit(statement)
            return

        def visit_Lambda(self, node):
            for expression in node.args.defaults:
                self.visit(expression)
            for expression in node.args.kw_defaults:
                if expression is not None:
                    self.visit(expression)
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
            if call_effects is not None:
                names.update(call_effects(node))
            self.generic_visit(node)

        def _target_root(self, target):
            if isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    self._target_root(element)
                return
            if isinstance(target, ast.Starred):
                self._target_root(target.value)
                return
            root = target
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)

        def visit_For(self, node):
            self._target_root(node.target)
            self.generic_visit(node)

        visit_AsyncFor = visit_For

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


def _branch_mutated_names(body, call_effects=None):
    """Return aggregate roots possibly mutated without entering nested scopes."""
    names = set()

    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for expression in node.decorator_list:
                self.visit(expression)
            for expression in node.args.defaults:
                self.visit(expression)
            for expression in node.args.kw_defaults:
                if expression is not None:
                    self.visit(expression)
            return

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            for expression in node.decorator_list:
                self.visit(expression)
            for expression in node.bases:
                self.visit(expression)
            for keyword in node.keywords:
                self.visit(keyword.value)
            for statement in node.body:
                self.visit(statement)
            return

        def visit_Lambda(self, node):
            for expression in node.args.defaults:
                self.visit(expression)
            for expression in node.args.kw_defaults:
                if expression is not None:
                    self.visit(expression)
            return

        def visit_Call(self, node):
            mutated = _python_mutated_root_name(node)
            if mutated:
                names.add(mutated)
            if call_effects is not None:
                names.update(call_effects(node))
            self.generic_visit(node)

        def visit_Assign(self, node):
            for target in node.targets:
                self._mutation_target(target, include_name=False)
            self.visit(node.value)

        def visit_AnnAssign(self, node):
            self._mutation_target(node.target, include_name=False)
            if node.value is not None:
                self.visit(node.value)

        def visit_AugAssign(self, node):
            self._mutation_target(node.target, include_name=True)
            self.visit(node.value)

        def visit_Delete(self, node):
            for target in node.targets:
                self._mutation_target(target, include_name=False)

        def _mutation_target(self, target, include_name):
            if isinstance(target, (ast.Tuple, ast.List)):
                for element in target.elts:
                    self._mutation_target(element, include_name)
                return
            if isinstance(target, ast.Starred):
                self._mutation_target(target.value, include_name)
                return
            if isinstance(target, (ast.Subscript, ast.Attribute)):
                root = target
                nested_attribute = False
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    if isinstance(root, ast.Attribute):
                        nested_attribute = True
                    root = root.value
                if isinstance(root, ast.Name):
                    names.add(root.id)
                if (isinstance(target, ast.Subscript) and nested_attribute
                        or include_name and isinstance(target, ast.Attribute)):
                    names.add(_UNKNOWN_AGGREGATE_EFFECT)
                self.visit(target)
                return
            if include_name and isinstance(target, ast.Name):
                names.add(target.id)

    collector = Collector()
    for statement in body:
        collector.visit(statement)
    return names


def _exception_possible_effects(node):
    """Return names a non-definite handler or else path may replace."""
    return set().union(
        *(_branch_assigned_names(handler.body) for handler in node.handlers),
        _branch_assigned_names(node.orelse),
        {handler.name for handler in node.handlers if handler.name})


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

    nodes, active_calls = [], set()
    lifecycle = {}
    active_functions = {}
    captured_default_effects = {}
    function_scope_bindings = {
        id(node): _scope_bindings(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    function_scope_globals = {
        id(node): _python_scope_global_names(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    function_scope_nonlocals = {
        id(node): _python_scope_nonlocal_names(node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function_scope_parents = _python_function_parent_scopes(tree)

    def branch_effect_summary(body, scope, seen=frozenset()):
        """Return possible writes and object mutations, including helpers."""
        summaries = {}
        call_mutations = {}

        def helper_summary(call):
            if id(call) in summaries:
                return summaries[id(call)]
            if not isinstance(call.func, ast.Name):
                return set(), set()
            function = active_functions.get((scope, call.func.id))
            if (function is None and scope is not None
                    and call.func.id not in function_scope_bindings.get(scope, set())):
                function = active_functions.get((None, call.func.id))
            if (function is None or id(function) in seen
                    or not _python_call_signature_binds(function, call)):
                return set(), set()
            effects, mutations, helper_call_mutations = branch_effect_summary(
                function.body, id(function), seen | {id(function)})
            local_only = (function_scope_bindings.get(id(function), set())
                          - function_scope_globals.get(id(function), set())
                          - function_scope_nonlocals.get(id(function), set()))
            visible_effects = effects - local_only
            visible_mutations = mutations - local_only
            arguments = dict(_python_call_argument_bindings(function, call))
            formal_aliases = {
                argument.arg: {argument.arg}
                for argument in (function.args.posonlyargs + function.args.args
                                 + function.args.kwonlyargs)
            }
            if function.args.vararg is not None:
                formal_aliases[function.args.vararg.arg] = {
                    function.args.vararg.arg}
            if function.args.kwarg is not None:
                formal_aliases[function.args.kwarg.arg] = {
                    function.args.kwarg.arg}
            mutation_origins = {}
            class FormalAliasCollector(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    return

                visit_AsyncFunctionDef = visit_FunctionDef
                visit_Lambda = visit_FunctionDef
                visit_ClassDef = visit_FunctionDef

                def _record_mutation_name(self, name):
                    origins = formal_aliases.get(name)
                    if origins:
                        mutation_origins.setdefault(name, set()).update(origins)

                def _record_mutation_target(self, target, include_name=False):
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for element in target.elts:
                            self._record_mutation_target(element, include_name)
                        return
                    if isinstance(target, ast.Starred):
                        self._record_mutation_target(target.value, include_name)
                        return
                    if isinstance(target, (ast.Subscript, ast.Attribute)):
                        root = target
                        while isinstance(root, (ast.Subscript, ast.Attribute)):
                            root = root.value
                        if isinstance(root, ast.Name):
                            self._record_mutation_name(root.id)
                        return
                    if include_name and isinstance(target, ast.Name):
                        self._record_mutation_name(target.id)

                def _clear_alias_target(self, target):
                    if isinstance(target, ast.Name):
                        formal_aliases.pop(target.id, None)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for element in target.elts:
                            self._clear_alias_target(element)
                    elif isinstance(target, ast.Starred):
                        self._clear_alias_target(target.value)

                def _alias_expression_roots(self, value):
                    if isinstance(value, ast.Name):
                        return set(formal_aliases.get(value.id, ()))
                    if isinstance(value, ast.Call):
                        children = [*value.args,
                                    *(keyword.value for keyword in value.keywords)]
                    elif isinstance(value, ast.Subscript):
                        children = [value.value]
                    else:
                        children = ast.iter_child_nodes(value)
                    return set().union(*(
                        self._alias_expression_roots(child)
                        for child in children))

                def _bind_alias_target(self, target, value):
                    if isinstance(target, ast.Name):
                        roots = self._alias_expression_roots(value)
                        if not roots:
                            formal_aliases.pop(target.id, None)
                        else:
                            formal_aliases[target.id] = roots
                        return
                    if (isinstance(target, (ast.Tuple, ast.List))
                            and isinstance(value, (ast.Tuple, ast.List))
                            and len(target.elts) == len(value.elts)
                            and not any(isinstance(element, ast.Starred)
                                        for element in target.elts)):
                        for target_element, value_element in zip(
                                target.elts, value.elts):
                            self._bind_alias_target(
                                target_element, value_element)
                        return
                    self._clear_alias_target(target)

                def visit_Call(self, node):
                    for mutated_name in helper_call_mutations.get(id(node), ()):
                        self._record_mutation_name(mutated_name)
                    mutated = _python_mutated_root_name(node)
                    if mutated:
                        self._record_mutation_name(mutated)
                    self.generic_visit(node)

                def visit_Assign(self, node):
                    self.visit(node.value)
                    for target in node.targets:
                        self._record_mutation_target(target)
                        self.visit(target)
                        self._bind_alias_target(target, node.value)

                def visit_AnnAssign(self, node):
                    if node.value is not None:
                        self.visit(node.value)
                    self._record_mutation_target(node.target)
                    self.visit(node.target)
                    if node.value is not None:
                        self._bind_alias_target(node.target, node.value)

                def visit_AugAssign(self, node):
                    self._record_mutation_target(node.target, include_name=True)
                    self.generic_visit(node)

                def visit_Delete(self, node):
                    for target in node.targets:
                        self._record_mutation_target(target)
                    self.generic_visit(node)

                def visit_If(self, node):
                    self.visit(node.test)
                    truth = _literal_bool(node.test)
                    if truth is not None:
                        for statement in (node.body if truth else node.orelse):
                            self.visit(statement)
                        return
                    before = {
                        name: set(origins)
                        for name, origins in formal_aliases.items()}
                    branches = []
                    for statements in (node.body, node.orelse):
                        formal_aliases.clear()
                        formal_aliases.update({
                            name: set(origins)
                            for name, origins in before.items()})
                        for statement in statements:
                            self.visit(statement)
                        branches.append({
                            name: set(origins)
                            for name, origins in formal_aliases.items()})
                    formal_aliases.clear()
                    for name in set().union(*(branch.keys() for branch in branches)):
                        origins = set().union(*(
                            branch.get(name, set()) for branch in branches))
                        if origins:
                            formal_aliases[name] = origins

            alias_collector = FormalAliasCollector()
            for statement in function.body:
                alias_collector.visit(statement)
            defaults = dict(_python_function_defaults(function))
            for mutated_name in mutations & local_only:
                formals = set(formal_aliases.get(mutated_name, {mutated_name}))
                formals.update(mutation_origins.get(mutated_name, set()))
                for formal in formals:
                    key = arguments.get(formal)
                    argument = None
                    if isinstance(key, int) and key < len(call.args):
                        argument = call.args[key]
                    elif isinstance(key, tuple) and key[0] == "keyword":
                        argument = next(
                            (keyword.value for keyword in call.keywords
                             if keyword.arg == key[1]), None)
                    elif key is None and formal in defaults:
                        token = f"@g11-default:{id(function)}:{formal}"
                        captured_default_effects[token] = (id(function), formal)
                        visible_effects.add(token)
                        visible_mutations.add(token)
                        continue
                    while isinstance(argument, (ast.Subscript, ast.Attribute)):
                        argument = argument.value
                    if isinstance(argument, ast.Name):
                        visible_effects.add(argument.id)
                        visible_mutations.add(argument.id)
            summaries[id(call)] = visible_effects, visible_mutations
            call_mutations[id(call)] = set(visible_mutations)
            return summaries[id(call)]

        effects = _branch_assigned_names(
            body, lambda call: helper_summary(call)[0])
        mutations = _branch_mutated_names(
            body, lambda call: helper_summary(call)[1])
        if _UNKNOWN_AGGREGATE_EFFECT in mutations:
            effects.add(_UNKNOWN_AGGREGATE_EFFECT)
        return effects, mutations, call_mutations

    def branch_possible_effects(body, scope, seen=frozenset()):
        """Return visible effects from an unproven path, including helpers."""
        return branch_effect_summary(body, scope, seen)[0]

    def possible_effect_marker(names, lineno):
        """Build an invalidation marker for names and captured defaults."""
        marker = ast.Delete(targets=[
            ast.Name(id=name, ctx=ast.Del())
            for name in sorted(names)
            if (name not in captured_default_effects
                and name != _UNKNOWN_AGGREGATE_EFFECT)])
        marker._g11_default_mutations = sorted({
            captured_default_effects[name]
            for name in names
            if name in captured_default_effects})
        marker._g11_unknown_aggregate_mutation = (
            _UNKNOWN_AGGREGATE_EFFECT in names)
        marker.lineno = lineno
        return marker

    def statement_sequence_terminates(body):
        return any(statement_terminates(statement) for statement in body)

    def statement_terminates(statement):
        if isinstance(statement, (ast.Return, ast.Raise)):
            return True
        if isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            return (statement_sequence_terminates(statement.finalbody)
                    or (not statement.handlers
                        and statement_sequence_terminates(statement.body)))
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

    def execute(function, owner, activation_line, call=None, implicit_positional=0):
        identity = (id(function), owner)
        if identity in active_calls:
            return
        active_calls.add(identity)
        try:
            for key in [key for key in active_functions if key[0] == id(function)]:
                active_functions.pop(key, None)
            marker = ast.Pass()
            marker.lineno = (getattr(function, "lineno", 0)
                             if activation_line is None else activation_line)
            marker._g11_scope_start = True
            marker._g11_argument_bindings = (
                [(formal, id(call), key) for formal, key in
                 _python_call_argument_bindings(
                     function, call, implicit_positional)]
                if call is not None else [])
            if call is not None:
                (marker._g11_default_bindings,
                 marker._g11_vararg_binding,
                 marker._g11_kwarg_binding) = _python_call_aggregate_bindings(
                     function, call, implicit_positional)
            else:
                marker._g11_default_bindings = []
                marker._g11_vararg_binding = None
                marker._g11_kwarg_binding = None
            nodes.append((marker, id(function), marker.lineno))
            visit_statement_sequence(function.body, owner, id(function), activation_line)
        finally:
            active_calls.remove(identity)

    def visit_statement_sequence(body, owner, scope, activation_line):
        for statement in body:
            visit(statement, owner, scope, activation_line)
            if statement_terminates(statement):
                break

    def visit_assignment_target_effects(target, owner, scope, activation_line):
        """Visit only expressions evaluated while storing into *target*."""
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                visit_assignment_target_effects(
                    element, owner, scope, activation_line)
            return
        if isinstance(target, ast.Starred):
            visit_assignment_target_effects(
                target.value, owner, scope, activation_line)
            return
        if isinstance(target, ast.Attribute):
            visit(target.value, owner, scope, activation_line)
            return
        if isinstance(target, ast.Subscript):
            visit(target.value, owner, scope, activation_line)
            visit(target.slice, owner, scope, activation_line)

    def visit(node, owner=None, scope=None, activation_line=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A definition is not a call: only decorators/defaults execute here.
            for decorator in node.decorator_list:
                visit(decorator, owner, scope, activation_line)
            for formal, default in _python_function_defaults(node):
                visit(default, owner, scope, activation_line)
                marker = ast.Expr(value=default)
                marker.lineno = getattr(default, "lineno", getattr(node, "lineno", 0))
                marker._g11_default_id = (id(node), formal)
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            if (owner is None and isinstance(node, ast.FunctionDef)
                    and not _function_is_generator(node)):
                key = (scope, node.name)
                active_functions.pop(key, None)
                if not node.decorator_list:
                    active_functions[key] = node
            if (isinstance(node, ast.FunctionDef)
                    and not _function_is_generator(node)
                    and mode == "tests" and owner in test_classes
                    and (node.name.startswith("test_")
                         or node.name in {"setUp", "tearDown", "setUpClass", "tearDownClass"})):
                roles = lifecycle.setdefault(
                    owner, {"tests": [], "setUp": None, "tearDown": None,
                            "setUpClass": None, "tearDownClass": None})
                if node.name.startswith("test_"):
                    roles["tests"].append(node)
                else:
                    roles[node.name] = node
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
            class_scope = id(node)
            marker = ast.Pass()
            marker.lineno = (getattr(node, "lineno", 0)
                             if activation_line is None else activation_line)
            marker._g11_scope_start = True
            nodes.append((marker, class_scope, marker.lineno))
            for statement in node.body:
                visit(statement, node.name, class_scope, activation_line)
            return
        if isinstance(node, ast.If):
            visit(node.test, owner, scope, activation_line)
            truth = None
            if _is_main_guard(node.test):
                branch = node.body if mode == "script" else node.orelse
            else:
                truth = _literal_bool(node.test)
                if truth is None:
                    names = (branch_possible_effects(node.body, scope)
                             | branch_possible_effects(node.orelse, scope))
                    if not _is_none_guard(node.test):
                        if names:
                            marker = possible_effect_marker(
                                names, getattr(node, "end_lineno", node.lineno))
                            visit(marker, owner, scope, activation_line)
                        return
                    for alternative in (node.body, node.orelse):
                        if names:
                            marker = possible_effect_marker(
                                names,
                                (getattr(alternative[0], "lineno", node.lineno)
                                 if alternative else node.lineno))
                            visit(marker, owner, scope, activation_line)
                        visit_statement_sequence(
                            alternative, owner, scope, activation_line)
                    if names:
                        marker = possible_effect_marker(
                            names, getattr(node, "end_lineno", node.lineno))
                        visit(marker, owner, scope, activation_line)
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
            possible_effects = branch_possible_effects([node], scope)
            if possible_effects:
                marker = possible_effect_marker(
                    possible_effects, getattr(node, "end_lineno", node.lineno))
                visit(marker, owner, scope, activation_line)
            return

        if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            # The try body executes, and a finally body executes regardless of
            # whether an exception is raised. Without sound exception-flow
            # matching, neither a handler nor the no-exception else body is
            # definite execution evidence.
            visit_statement_sequence(node.body, owner, scope, activation_line)
            possible_effects = set().union(
                *(branch_possible_effects(handler.body, scope)
                  for handler in node.handlers),
                branch_possible_effects(node.orelse, scope),
                {handler.name for handler in node.handlers if handler.name})
            if possible_effects:
                boundary_statements = [
                    *(handler.body[0] for handler in node.handlers if handler.body),
                    *(node.orelse[:1]),
                    *(node.finalbody[:1]),
                ]
                marker = possible_effect_marker(possible_effects, min(
                    (getattr(statement, "lineno", node.lineno)
                     for statement in boundary_statements),
                    default=getattr(node, "end_lineno", node.lineno)))
                visit(marker, owner, scope, activation_line)
            visit_statement_sequence(node.finalbody, owner, scope, activation_line)
            return

        if isinstance(node, ast.Assign):
            # Evaluate the RHS once, then interleave each target's evaluation
            # and store.  Later chained targets may call helpers that observe
            # or mutate an object already stored into an earlier target.
            visit(node.value, owner, scope, activation_line)
            marker = ast.copy_location(ast.Expr(value=node.value), node.value)
            marker._g11_assignment_value_id = id(node)
            marker._g11_assignment_roots = node.targets
            execution_line = (getattr(node, "lineno", 0)
                              if activation_line is None else activation_line)
            nodes.append((marker, scope, execution_line))
            def store_assignment_target(target, root_target):
                # Unpacking stores proceed left-to-right.  A later target may
                # call a helper that observes or mutates an earlier store, so
                # do not defer every element store until the whole target's
                # evaluation has completed.  Each unpacking level captures
                # its direct elements when that level is reached: the outer
                # level is fixed before its first store, while a nested level
                # still observes in-place mutations by earlier outer stores.
                if isinstance(target, (ast.Tuple, ast.List)):
                    unpack = ast.copy_location(
                        ast.Expr(value=node.value), target)
                    unpack._g11_assignment_unpack_id = id(node)
                    unpack._g11_assignment_unpack_target = target
                    unpack._g11_assignment_unpack_is_root = target is root_target
                    nodes.append((unpack, scope, execution_line))
                    for element in target.elts:
                        store_assignment_target(element, root_target)
                    return
                visit_assignment_target_effects(
                    target, owner, scope, activation_line)
                for name in _target_names(target):
                    active_functions.pop(
                        (active_function_destination(scope, name), name), None)
                store = ast.copy_location(
                    ast.Assign(targets=[target], value=node.value,
                               type_comment=node.type_comment), node)
                store._g11_captured_assignment_id = id(node)
                if root_target is not target:
                    store._g11_captured_assignment_root = root_target
                    store._g11_captured_assignment_target_id = id(target)
                nodes.append((store, scope, execution_line))

            for target in node.targets:
                store_assignment_target(target, target)
            return

        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            # Assignment targets are rebound only after their value expressions
            # finish. A helper invoked by the RHS still resolves the prior
            # function binding.
            if isinstance(node, ast.AugAssign):
                visit(node.target, owner, scope, activation_line)
                if isinstance(node.target, ast.Name):
                    # Python captures an augmented-assignment target before
                    # evaluating the RHS.  Preserve that object even when an
                    # RHS helper rebinds the target name.
                    marker = ast.Expr(value=node.target)
                    marker.lineno = getattr(node.target, "lineno", node.lineno)
                    marker._g11_augassign_target_id = id(node)
                    execution_line = (marker.lineno if activation_line is None
                                      else activation_line)
                    nodes.append((marker, scope, execution_line))
            if getattr(node, "value", None) is not None:
                visit(node.value, owner, scope, activation_line)
            elif isinstance(node, ast.AnnAssign):
                return
            targets = [node.target]
            if not isinstance(node, ast.AugAssign):
                for target in targets:
                    visit_assignment_target_effects(
                        target, owner, scope, activation_line)
            for target in targets:
                for name in _target_names(target):
                    active_functions.pop(
                        (active_function_destination(scope, name), name), None)
            execution_line = (getattr(node, "lineno", 0)
                              if activation_line is None else activation_line)
            nodes.append((node, scope, execution_line))
            return

        if isinstance(node, ast.NamedExpr):
            # Assignment expressions evaluate their value before rebinding.
            visit(node.value, owner, scope, activation_line)
            for name in _target_names(node.target):
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
            else:
                possible_effects = branch_possible_effects(
                    (node.body, node.orelse), scope)
                if possible_effects:
                    marker = possible_effect_marker(
                        possible_effects, getattr(node, "end_lineno", node.lineno))
                    visit(marker, owner, scope, activation_line)
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
                    possible_effects = branch_possible_effects(
                        node.values[index + 1:], scope)
                    if possible_effects:
                        marker = possible_effect_marker(
                            possible_effects,
                            getattr(node, "end_lineno", node.lineno))
                        visit(marker, owner, scope, activation_line)
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
            for index, (operator, comparator) in enumerate(
                    zip(node.ops, node.comparators)):
                visit(comparator, owner, scope, activation_line)
                truth = _literal_comparison(left, operator, comparator)
                if truth is None:
                    possible_effects = branch_possible_effects(
                        node.comparators[index + 1:], scope)
                    if possible_effects:
                        marker = possible_effect_marker(
                            possible_effects,
                            getattr(node, "end_lineno", node.lineno))
                        visit(marker, owner, scope, activation_line)
                    break
                if not truth:
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
            nonempty = _literal_iterable_nonemptiness(first.iter)
            supported = (nonempty is True
                         and isinstance(first.target, ast.Name)
                         and all(_literal_bool(guard) is True
                                 for guard in first.ifs)
                         and len(node.generators) == 1)
            if supported:
                comprehension_scope = id(node)
                if isinstance(node, ast.DictComp):
                    visit(node.key, owner, comprehension_scope, activation_line)
                    visit(node.value, owner, comprehension_scope, activation_line)
                else:
                    visit(node.elt, owner, comprehension_scope, activation_line)
                return
            if nonempty is False:
                return
            possible_effects = branch_possible_effects(
                _comprehension_possible_effect_expressions(node), scope)
            if possible_effects:
                marker = possible_effect_marker(
                    possible_effects, getattr(node, "end_lineno", node.lineno))
                visit(marker, owner, scope, activation_line)
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
            # visible to process attribution at the call itself, but the
            # callable identity is captured before those argument effects.
            visit(node.func, owner, scope, activation_line)
            target = node.func
            captured_function = None
            captured_method = None
            if isinstance(target, ast.Name):
                captured_function = active_functions.get((scope, target.id))
                if (captured_function is None and scope is not None
                        and target.id not in function_scope_bindings.get(scope, set())):
                    captured_function = active_functions.get((None, target.id))
            elif (isinstance(target, ast.Attribute)
                  and isinstance(target.value, ast.Name)
                  and target.value.id in {"self", "cls"} and owner):
                captured_method = methods.get((owner, target.attr))
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
                marker = ast.Expr(value=keyword.value)
                marker.lineno = getattr(
                    keyword.value, "lineno", getattr(node, "lineno", 0))
                marker._g11_call_id = id(node)
                marker._g11_argument_key = ("keyword", keyword.arg)
                execution_line = (marker.lineno if activation_line is None
                                  else activation_line)
                nodes.append((marker, scope, execution_line))
            execution_line = (getattr(node, "lineno", 0)
                              if activation_line is None else activation_line)
            nodes.append((node, scope, execution_line))
            if isinstance(target, ast.Name):
                if (captured_function is not None
                        and _python_call_signature_binds(captured_function, node)):
                    execute(captured_function, None, execution_line, node)
            elif (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                  and target.value.id in {"self", "cls"} and owner):
                if (isinstance(captured_method, ast.FunctionDef)
                        and not _function_is_generator(captured_method)
                        and _python_call_signature_binds(
                            captured_method, node, implicit_positional=1)):
                    execute(captured_method, owner, execution_line, node,
                            implicit_positional=1)
            return

        execution_line = (getattr(node, "lineno", 0)
                          if activation_line is None else activation_line)
        nodes.append((node, scope, execution_line))
        for child in ast.iter_child_nodes(node):
            visit(child, owner, scope, activation_line)

    for statement in tree.body:
        visit(statement)
    activation_line = max(
        (getattr(node, "end_lineno", getattr(node, "lineno", 0))
         for node in ast.walk(tree)), default=0)
    for owner in sorted(lifecycle):
        roles = lifecycle[owner]
        if roles["setUpClass"] is not None:
            activation_line += 1
            execute(roles["setUpClass"], owner, activation_line)
        for test in sorted(roles["tests"], key=lambda function: function.name):
            if roles["setUp"] is not None:
                activation_line += 1
                execute(roles["setUp"], owner, activation_line)
            activation_line += 1
            execute(test, owner, activation_line)
            if roles["tearDown"] is not None:
                activation_line += 1
                execute(roles["tearDown"], owner, activation_line)
        if roles["tearDownClass"] is not None:
            activation_line += 1
            execute(roles["tearDownClass"], owner, activation_line)
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
                if not isinstance(value, (list, _StaticTuple)):
                    values.append(_UNRESOLVED_STATIC)
                    continue
                values.extend(value)
                continue
            value = _python_static_value(element, local, module, source_path, imports, scope_shadowed)
            if value is None:
                values.append(_UNRESOLVED_STATIC)
                continue
            values.append(value)
        return _StaticTuple(values) if isinstance(node, ast.Tuple) else values
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
        value = _python_static_value(
            node.args[0], local, module, source_path, imports, scope_shadowed)
        # The scanner represents supported paths as strings already.  Preserve
        # only those values: converting an aggregate command produces one
        # display string, not an argv sequence with executable semantics.
        return value if isinstance(value, str) else None
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
            if isinstance(value, (list, _StaticTuple)):
                values.extend(value)
            else:
                values.append(_UNRESOLVED_STATIC)
        else:
            values.append(_UNRESOLVED_STATIC if value is None else value)
    return _StaticTuple(values) if isinstance(node, ast.Tuple) else values


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


def _python_class_parent_scopes(tree):
    """Map each class scope to its nearest enclosing function, if any."""
    parents = {}

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(id(node))
            self.generic_visit(node)
            self.functions.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            parents[id(node)] = self.functions[-1] if self.functions else None
            self.generic_visit(node)

    Collector().visit(tree)
    return parents


def _invalidate_python_value(value, environments, recursive=False):
    """Poison one captured aggregate and every binding sharing its identity."""
    if not isinstance(value, (list, _StaticTuple, dict)):
        return
    seen = set()

    def poison(current):
        if (not isinstance(current, (list, _StaticTuple, dict))
                or id(current) in seen):
            return
        seen.add(id(current))
        if recursive:
            children = current.values() if isinstance(current, dict) else current
            for child in list(children):
                poison(child)
        if isinstance(current, list):
            current[:] = [_UNRESOLVED_STATIC]

    poison(value)
    for environment in environments:
        for bound_name, bound in list(environment.items()):
            if id(bound) in seen:
                environment[bound_name] = _UNRESOLVED_STATIC


def _invalidate_python_binding(name, local, module, scope, global_names=(),
                               destination_for_name=None, recursive=False):
    destination = (destination_for_name(name) if destination_for_name is not None
                   else module if scope is None or name in global_names else local)
    prior = destination.get(name, module.get(name))
    environments = ((destination,) if destination is module
                    else (destination, module))
    _invalidate_python_value(prior, environments, recursive)
    destination[name] = _UNRESOLVED_STATIC


def _python_assignment_target_values(pattern, value):
    """Map one unpacking level to values captured when that level starts."""
    values = {}

    def unresolved(target):
        for element in pattern.elts:
            values[id(element)] = _UNRESOLVED_STATIC

    if not isinstance(pattern, (ast.Tuple, ast.List)):
        return values
    if not isinstance(value, (list, _StaticTuple)):
        unresolved(pattern)
        return values
    starred = [index for index, element in enumerate(pattern.elts)
               if isinstance(element, ast.Starred)]
    if not starred:
        if len(pattern.elts) != len(value):
            unresolved(pattern)
            return values
        element_values = list(value)
    elif len(starred) == 1 and len(value) >= len(pattern.elts) - 1:
        star = starred[0]
        trailing = len(pattern.elts) - star - 1
        element_values = [
            *value[:star],
            list(value[star:len(value) - trailing]),
            *(value[len(value) - trailing:] if trailing else []),
        ]
    else:
        unresolved(pattern)
        return values
    for element, element_value in zip(pattern.elts, element_values):
        values[id(element)] = element_value
    return values


def _bind_python_value(node, value, local, module, scope, global_names=(),
                       destination_for_name=None):
    targets = []
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]

    def destination(name):
        if destination_for_name is not None:
            return destination_for_name(name)
        return module if scope is None or name in global_names else local

    def bind(target, target_value):
        if isinstance(target, ast.Subscript):
            root = target.value
            recursive = isinstance(root, (ast.Subscript, ast.Attribute))
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name):
                _invalidate_python_binding(
                    root.id, local, module, scope, global_names,
                    destination_for_name, recursive=recursive)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if (not isinstance(target_value, (list, _StaticTuple))
                    or len(target.elts) != len(target_value)):
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
    values = value if isinstance(value, (list, _StaticTuple)) else [value]
    return {_tool_for_token(item, tools) for item in values if isinstance(item, str)} - {None}


def _command_tool_values(value, tools):
    if isinstance(value, _StaticAlternatives):
        choices = [_command_tool_values(choice, tools) for choice in value]
        return set.intersection(*choices) if choices else set()
    if not isinstance(value, (list, _StaticTuple)) or not value:
        return _tool_values(value, tools)
    first = value[0]
    python_interpreters = {"<python>", "python", "python3", "python3.11"}
    shell_interpreters = {"bash", "sh"}
    interpreters = python_interpreters | shell_interpreters
    if not isinstance(first, str):
        return set()
    if first not in interpreters:
        return _tool_values(first, tools)
    arguments = value[1:]
    if first in python_interpreters:
        terminating_options = {"-", "-c", "-m"}
        operand_options = {"-W", "-X", "--check-hash-based-pycs"}
        attached_options = ("-W", "-X")
        flag_options = {"-b", "-bb", "-B", "-d", "-E", "-I", "-i", "-O",
                        "-OO", "-P", "-q", "-R", "-s", "-S", "-u", "-v", "-x"}
    else:
        terminating_options = {"-c", "-s"}
        operand_options = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
        attached_options = ()
        flag_options = set()
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
        if (parsing_options and first in python_interpreters
                and token not in terminating_options
                and (token.startswith("-c") or token.startswith("-m"))):
            return set()
        if parsing_options and token in operand_options:
            consume_operand = True
            continue
        if parsing_options and any(token.startswith(option) and token != option
                                   for option in attached_options):
            continue
        if parsing_options and token in flag_options:
            continue
        if parsing_options and token.startswith(("-", "+")):
            return set()
        path = _tool_for_token(token, tools)
        return {path} if path else set()
    return set()


def _subprocess_uses_supported_argv(node):
    """Return whether subprocess uses the bounded ordinary-argv semantics."""
    if len(node.args) != 1:
        return False
    for keyword in node.keywords:
        if keyword.arg is None:
            return False
        if (keyword.arg == "executable"
                and not (isinstance(keyword.value, ast.Constant)
                         and keyword.value.value is None)):
            return False
        if (keyword.arg == "shell"
                and not (isinstance(keyword.value, ast.Constant)
                         and keyword.value.value in (False, None))):
            return False
    return True


def _python_execution_wrappers(tree, imports):
    """Return reachable-call argument positions forwarded to a trusted process API."""
    wrappers = {}
    subprocess_calls = {"subprocess.run", "subprocess.call", "subprocess.Popen",
                         "subprocess.check_call", "subprocess.check_output"}
    process_calls = subprocess_calls | {"os.system"}
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
                        for name in _branch_assigned_names([node]):
                            invalidate(name)
                        return False
                    body_terminates = visit_body(node.body)
                    other_terminates = visit_body(node.orelse)
                    return bool(node.body and node.orelse
                                and body_terminates and other_terminates)
                branch = node.body if truth else node.orelse
                return visit_body(branch)
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                visit(node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test)
                for name in _branch_assigned_names([node]):
                    invalidate(name)
                return False
            if isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                body_terminates = visit_body(node.body)
                possible_effects = _exception_possible_effects(node)
                for name in possible_effects:
                    invalidate(name)
                finally_terminates = visit_body(node.finalbody)
                return finally_terminates or (body_terminates and not node.handlers)
            if isinstance(node, ast.IfExp):
                visit(node.test)
                truth = _literal_bool(node.test)
                if truth is not None:
                    visit(node.body if truth else node.orelse)
                else:
                    for name in _branch_assigned_names((node.body, node.orelse)):
                        invalidate(name)
                return False
            if isinstance(node, ast.BoolOp):
                for index, value in enumerate(node.values):
                    visit(value)
                    if index == len(node.values) - 1:
                        break
                    truth = _literal_bool(value)
                    if truth is None:
                        for name in _branch_assigned_names(node.values[index + 1:]):
                            invalidate(name)
                        break
                    if isinstance(node.op, ast.And) and not truth:
                        break
                    if isinstance(node.op, ast.Or) and truth:
                        break
                return False
            if isinstance(node, ast.Compare):
                visit(node.left)
                left = node.left
                for index, (operator, comparator) in enumerate(
                        zip(node.ops, node.comparators)):
                    visit(comparator)
                    truth = _literal_comparison(left, operator, comparator)
                    if truth is None:
                        for name in _branch_assigned_names(
                                node.comparators[index + 1:]):
                            invalidate(name)
                        break
                    if not truth:
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
                nonempty = _literal_iterable_nonemptiness(first.iter)
                supported = (nonempty is True
                             and isinstance(first.target, ast.Name)
                             and all(_literal_bool(guard) is True
                                     for guard in first.ifs)
                             and len(node.generators) == 1)
                if not supported:
                    if nonempty is not False:
                        for name in _branch_assigned_names(
                                _comprehension_possible_effect_expressions(node)):
                            invalidate(name)
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
                    forwarded = (node.args[0] if resolved in process_calls
                                 and (resolved not in subprocess_calls
                                      or _subprocess_uses_supported_argv(node))
                                 else None)
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
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                update_target(node.target, node.value)
            elif isinstance(node, ast.AugAssign):
                update_target(node.target)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    update_target(target)
            elif isinstance(node, ast.NamedExpr):
                visit(node.value)
                update_target(node.target, node.value)
                return False
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
                key = (containing_scope, statement.name)
                if _function_is_generator(statement) or statement.decorator_list:
                    wrappers.pop(key, None)
                    continue
                positions = wrapper_positions(statement)
                if positions:
                    wrappers[key] = (positions, statement.lineno, statement)
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
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.ClassDef))}
    scope_nonlocals = {id(node): _python_scope_nonlocal_names(node) for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    scope_parents = _python_function_parent_scopes(tree)
    class_scope_parents = _python_class_parent_scopes(tree)
    module_bindings, scope_bindings, scope_imports = {}, {}, {}
    evaluated_arguments = {}
    evaluated_defaults = {}
    evaluated_assignment_values = {}
    evaluated_assignment_target_values = {}
    evaluated_sequence_elements = {}
    evaluated_augassign_targets = {}
    missing_argument = object()
    # Reachability traversal already emits semantic evaluation order. Sorting
    # by source line reorders multiline call effects ahead of their arguments.
    reachable_nodes = _reachable_python_nodes(tree, mode)
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
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                                 ast.Delete, ast.NamedExpr)):
            continue
        targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                   else [node.target])
        for target in targets:
            for name in _target_names(target):
                destination_scope = wrapper_destination_scope(scope, name)
                key = (destination_scope, name)
                if key in wrappers:
                    wrapper_reassignments.setdefault(key, []).append(
                        (execution_line, node.lineno))
    for node, scope, execution_line in reachable_nodes:
        if getattr(node, "_g11_unknown_aggregate_mutation", False):
            environments = (module_bindings, *scope_bindings.values())
            for environment in environments:
                for value in list(environment.values()):
                    _invalidate_python_value(
                        value, environments, recursive=True)
        for default_id in getattr(node, "_g11_default_mutations", ()):
            captured = evaluated_defaults.get(default_id, missing_argument)
            if captured is not missing_argument:
                _invalidate_python_value(
                    captured, (module_bindings, *scope_bindings.values()),
                    recursive=True)
        if hasattr(node, "_g11_scope_start"):
            scope_bindings[scope] = {}
            scope_imports[scope] = _empty_import_bindings()
            for formal, call_id, argument_key in getattr(
                    node, "_g11_argument_bindings", ()):
                value = evaluated_arguments.get(
                    (call_id, argument_key), missing_argument)
                if value is not missing_argument:
                    scope_bindings[scope][formal] = value
            for formal, function_id, default_name in getattr(
                    node, "_g11_default_bindings", ()):
                value = evaluated_defaults.get(
                    (function_id, default_name), missing_argument)
                if value is not missing_argument:
                    scope_bindings[scope][formal] = value
            vararg = getattr(node, "_g11_vararg_binding", None)
            if vararg is not None:
                formal, call_id, argument_keys = vararg
                values = [evaluated_arguments.get(
                    (call_id, key), missing_argument) for key in argument_keys]
                if all(value is not missing_argument for value in values):
                    scope_bindings[scope][formal] = _StaticTuple(values)
            kwarg = getattr(node, "_g11_kwarg_binding", None)
            if kwarg is not None:
                formal, call_id, entries = kwarg
                values = {name: evaluated_arguments.get(
                    (call_id, key), missing_argument) for name, key in entries}
                if all(value is not missing_argument for value in values.values()):
                    scope_bindings[scope][formal] = values
            continue
        local = scope_bindings.setdefault(scope, {}) if scope is not None else module_bindings
        global_names = scope_globals.get(scope, set())
        local_imports = scope_imports.setdefault(scope, _empty_import_bindings())
        destination_for_name = lambda name: binding_destination(scope, name)
        if scope is not None:
            _update_local_import_bindings(node, local_imports)
        visible_module_bindings = module_bindings
        visible_imports = imports
        visible_shadowed = scope_shadowed.get(scope, set())
        if scope is not None and scope not in class_scope_parents:
            enclosing_scopes = []
            parent_scope = scope_parents.get(scope)
            while parent_scope is not None:
                enclosing_scopes.append(parent_scope)
                parent_scope = scope_parents.get(parent_scope)
            if enclosing_scopes:
                visible_module_bindings = dict(module_bindings)
                for parent_scope in reversed(enclosing_scopes):
                    owned = (scope_shadowed.get(parent_scope, set())
                             - scope_globals.get(parent_scope, set())
                             - scope_nonlocals.get(parent_scope, set()))
                    for name in owned - global_names:
                        visible_module_bindings[name] = scope_bindings.get(
                            parent_scope, {}).get(name, _UNRESOLVED_STATIC)
        if scope in class_scope_parents:
            visible_shadowed = set()
            parent_scope = class_scope_parents[scope]
            if parent_scope is not None:
                visible_module_bindings = dict(module_bindings)
                visible_module_bindings.update(scope_bindings.get(parent_scope, {}))
                visible_imports, _ = _combined_import_bindings(
                    imports,
                    scope_imports.get(parent_scope, _empty_import_bindings()),
                    scope_shadowed.get(parent_scope, set()))
        active_imports, _ = _combined_import_bindings(
            visible_imports, local_imports, visible_shadowed)
        locally_imported = set(local_imports["modules"]) | set(local_imports["symbols"])
        static_shadowed = ((visible_shadowed | set(local_imports["shadowed"]))
                           - locally_imported - global_names
                           - scope_nonlocals.get(scope, set()))
        if hasattr(node, "_g11_default_id"):
            evaluated_defaults[node._g11_default_id] = (
                _python_sequence_static_value(
                    node.value, evaluated_sequence_elements, local,
                    visible_module_bindings, source_path, active_imports,
                    static_shadowed))
            continue
        if hasattr(node, "_g11_assignment_value_id"):
            assignment_value = _python_sequence_static_value(
                node.value, evaluated_sequence_elements, local,
                visible_module_bindings, source_path, active_imports,
                static_shadowed)
            evaluated_assignment_values[node._g11_assignment_value_id] = assignment_value
            evaluated_assignment_target_values[
                node._g11_assignment_value_id] = {}
            continue
        if hasattr(node, "_g11_assignment_unpack_id"):
            assignment_id = node._g11_assignment_unpack_id
            target = node._g11_assignment_unpack_target
            target_values = evaluated_assignment_target_values.setdefault(
                assignment_id, {})
            target_value = (
                evaluated_assignment_values.get(
                    assignment_id, _UNRESOLVED_STATIC)
                if node._g11_assignment_unpack_is_root
                else target_values.get(id(target), _UNRESOLVED_STATIC))
            target_values.update(
                _python_assignment_target_values(target, target_value))
            continue
        if hasattr(node, "_g11_augassign_target_id"):
            evaluated_augassign_targets[node._g11_augassign_target_id] = (
                _python_sequence_static_value(
                    node.value, evaluated_sequence_elements, local,
                    visible_module_bindings, source_path, active_imports,
                    static_shadowed))
            continue
        if hasattr(node, "_g11_binary_id"):
            evaluated_sequence_elements[(
                "binary", node._g11_binary_id, node._g11_binary_index)] = (
                    _python_sequence_static_value(
                        node.value, evaluated_sequence_elements, local,
                        visible_module_bindings, source_path, active_imports,
                        static_shadowed))
            continue
        if hasattr(node, "_g11_sequence_id"):
            value = _python_sequence_static_value(
                node.value, evaluated_sequence_elements, local, visible_module_bindings,
                source_path, active_imports, static_shadowed)
            if node._g11_sequence_starred and isinstance(value, (list, _StaticTuple)):
                value = list(value)
            evaluated_sequence_elements[(
                node._g11_sequence_id, node._g11_sequence_index)] = value
            continue
        if hasattr(node, "_g11_call_id"):
            argument_key = (node._g11_argument_key
                            if hasattr(node, "_g11_argument_key")
                            else node._g11_argument_index)
            evaluated_arguments[(node._g11_call_id, argument_key)] = (
                _python_sequence_static_value(
                    node.value, evaluated_sequence_elements, local,
                    visible_module_bindings, source_path, active_imports,
                    static_shadowed))
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            augmented_prior = None
            if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
                augmented_prior = evaluated_augassign_targets.get(id(node))
                right = _python_sequence_static_value(
                    node.value, evaluated_sequence_elements, local,
                    visible_module_bindings, source_path, active_imports,
                    static_shadowed)
                value = _static_add(augmented_prior, right)
            elif hasattr(node, "_g11_captured_assignment_id"):
                value = evaluated_assignment_values.get(
                    node._g11_captured_assignment_id, _UNRESOLVED_STATIC)
                if hasattr(node, "_g11_captured_assignment_target_id"):
                    value = evaluated_assignment_target_values.get(
                        node._g11_captured_assignment_id, {}).get(
                        node._g11_captured_assignment_target_id,
                        _UNRESOLVED_STATIC)
            else:
                value_node = (ast.BinOp(left=node.target, op=node.op, right=node.value)
                              if isinstance(node, ast.AugAssign) else node.value)
                value = _python_sequence_static_value(
                    value_node, evaluated_sequence_elements, local,
                    visible_module_bindings, source_path, active_imports,
                    static_shadowed)
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                destination = destination_for_name(node.target.id)
                prior = (augmented_prior if augmented_prior is not None
                         else destination.get(
                             node.target.id,
                             visible_module_bindings.get(node.target.id)))
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
                    visible_module_bindings, source_path, active_imports,
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
                             and not reassigned and call_order > wrapper_order
                             and _python_call_signature_binds(wrapper[2], node) else ())
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
            subprocess_calls = {"subprocess.run", "subprocess.call", "subprocess.Popen",
                                 "subprocess.check_call", "subprocess.check_output"}
            if call_name in subprocess_calls and not _subprocess_uses_supported_argv(node):
                continue
            if call_name in subprocess_calls | {"os.system", "runpy.run_path"}:
                for index, arg in enumerate(node.args):
                    value = argument_value(index)
                    if isinstance(value, (list, _StaticTuple, _StaticAlternatives)):
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


def _js_builtin_name_unshadowed(text, position, name):
    """Require a bounded unshadowed global built-in identifier."""
    if (_js_parameter_shadows(text, position, name)
            or _js_local_declaration_shadows(text, position, name)):
        return False
    searchable, _scopes = _js_function_scopes(text)
    blocks = _js_block_scopes(searchable)

    import_from = re.compile(r"\bimport\s+([^;'\"]+?)\s+from\s*['\"]")
    for imported in import_from.finditer(text):
        clause = imported.group(1).strip()
        default = re.match(r"([A-Za-z_$][\w$]*)", clause)
        if default is not None and default.group(1) == name:
            return False
        namespace = re.search(r"\*\s+as\s+([A-Za-z_$][\w$]*)", clause)
        if namespace is not None and namespace.group(1) == name:
            return False
        named = re.search(r"\{([^}]*)\}", clause)
        if named is not None:
            for item in named.group(1).split(","):
                parts = re.split(r"\s+as\s+", item.strip())
                if parts and parts[-1] == name:
                    return False

    def at_module_root(candidate):
        return not any(start < candidate < end for start, end in blocks)

    declaration = re.compile(
        r"\b(?:const|let|var|function|class)\s+" + re.escape(name) +
        r"(?![\w$])")
    if any(at_module_root(match.start()) for match in declaration.finditer(searchable)):
        return False
    assignment = re.compile(
        r"(?<![\w$.])" + re.escape(name) +
        r"(?![\w$])\s*(?:=(?!=|>)|\+\+|--|(?:\+|-|\*|/|%|&|\||\^|<<|>>)\s*=)")
    if any(at_module_root(match.start())
           for match in assignment.finditer(searchable, 0, position)):
        return False
    return True


def _js_binding_or_property_written(segment, name, mutable_object=False):
    """Return whether a URL binding or a value rooted at it may be mutated."""
    segment, _scopes = _js_function_scopes(segment)
    root = r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])"
    if mutable_object:
        alias_assignment = (
            r"(?<![\w$.])(?:(?:const|let|var)\s+)?"
            r"[A-Za-z_$][\w$]*\s*=(?!=|>)\s*" + root +
            r"\s*(?=;|,|\n|$)")
        if re.search(alias_assignment, segment):
            return True
    update = r"(?:=(?!=|>)|\+\+|--|(?:\+|-|\*|/|%|&|\||\^|<<|>>)\s*=)"
    if re.search(root + r"\s*" + update, segment):
        return True
    property_root = (root +
                     r"\s*(?:(?:\.\s*[A-Za-z_$][\w$]*)|(?:\[[^\]\n]+\]))+")
    if re.search(r"\bdelete\s+" + property_root, segment):
        return True
    return bool(re.search(
        property_root + r"\s*(?:" + update + r"|\()",
        segment))


def _js_supported_initializer_complete(text, end):
    """Require the matched URL/path expression to be the whole initializer."""
    tail = text[end:]
    horizontal = len(tail) - len(tail.lstrip(" \t\r"))
    if horizontal == len(tail):
        return True
    next_char = tail[horizontal]
    if next_char in ";,":
        return True
    if next_char != "\n":
        return False
    remainder = tail[horizontal:].lstrip()
    if not remainder:
        return True
    if remainder.startswith(("in ", "in\t", "instanceof ", "instanceof\t")):
        return False
    return remainder[0] not in ".?[(`+-*/%&|^<>=!"


def js_tool_references(text, source_path, tools):
    references = set()
    visible = _strip_js_comments(text)
    searchable, function_scopes = _js_function_scopes(visible)
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
    url_converters = {}
    url_import = re.compile(
        r"\bimport\s*\{([^}]*)\}\s*from\s*['\"]node:url['\"]")
    for imported in url_import.finditer(visible):
        if searchable[imported.start():imported.start() + 6] != "import":
            continue
        for item in imported.group(1).split(","):
            parts = re.split(r"\s+as\s+", item.strip())
            if parts and parts[0] == "fileURLToPath":
                url_converters[parts[-1]] = imported.end()
    url_binding = re.compile(
        r"\b(?P<kind>const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:(?P<converter>[A-Za-z_$][\w$]*)\s*\(\s*)?"
        r"new\s+(?P<constructor>URL)\s*\(\s*['\"](?P<specifier>[^'\"]+)['\"]\s*,\s*"
        r"import\.meta\.url\s*\)(?(converter)\s*\))")
    for match in url_binding.finditer(visible):
        if (searchable[match.start("kind"):match.end("kind")]
                != match.group("kind")):
            continue
        if not _js_supported_initializer_complete(visible, match.end()):
            continue
        if not _js_builtin_name_unshadowed(
                visible, match.start("constructor"), match.group("constructor")):
            continue
        if not _js_call_activation_points(visible, match.start("kind")):
            continue
        converter = match.group("converter")
        if converter:
            import_end = url_converters.get(converter)
            if import_end is None or import_end > match.start():
                continue
            changed = re.compile(
                r"(?<![\w$.])" + re.escape(converter) +
                r"(?![\w$])\s*(?:=(?!=|>)|\+\+|--|(?:\+|-|\*|/|%|&|\||\^|<<|>>)\s*=)"
                r"|\b(?:const|let|var|function|class)\s+" +
                re.escape(converter) + r"(?![\w$])")
            if (changed.search(visible[import_end:match.start()])
                    or _js_parameter_shadows(
                        visible, match.start("converter"), converter)
                    or _js_local_declaration_shadows(
                        visible, match.start("converter"), converter, import_end)):
                continue
        specifier = match.group("specifier")
        if not specifier.startswith("."):
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), specifier))
        if target in tools:
            function_blocks = {(scope[2], scope[3]) for scope in function_scopes}
            declaration_blocks = [block for block in _js_block_scopes(searchable)
                                  if block not in function_blocks
                                  and block[0] < match.start() < block[1]]
            declaration_block = (max(declaration_blocks, key=lambda item: item[0])
                                 if match.group("kind") != "var" and declaration_blocks
                                 else None)
            bindings.setdefault(match.group("name"), []).append(
                (match.end(), target, _js_function_scope_key(visible, match.start()),
                 declaration_block, converter is None))
    child_names = set()
    named_import = re.compile(
        r"\bimport\s*\{([^}]*)\}\s*from\s*['\"](?:node:)?child_process['\"]")
    allowed_child_calls = {"spawnSync", "spawn", "execFileSync", "execFile"}
    for match in named_import.finditer(visible):
        if searchable[match.start():match.start() + 6] != "import":
            continue
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
        identifiers = _js_direct_child_argument_identifiers(arguments)
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
            (declaration_end, target, declaration_scope, _declaration_block,
             mutable_object) = available[-1]
            if _js_local_declaration_shadows(
                    visible, call_start, name, declaration_end):
                continue
            cutoffs = ({call_start} if declaration_scope is not None
                       else activation_points)
            if any(not _js_binding_or_property_written(
                       visible[declaration_end:cutoff], name, mutable_object)
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


def _js_direct_child_argument_identifiers(text):
    """Return URL names in command/file or literal argv positions only."""
    arguments = []
    start = 0
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
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
            continue
        if char in "([{":
            stack.append(char)
            continue
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return set()
            stack.pop()
            continue
        if char == "," and not stack:
            arguments.append(text[start:index].strip())
            start = index + 1
    if quote or stack:
        return set()
    arguments.append(text[start:].strip())

    identifier = re.compile(r"[A-Za-z_$][\w$]*")
    identifiers = set()
    if arguments and identifier.fullmatch(arguments[0]):
        identifiers.add(arguments[0])
    if len(arguments) > 1:
        argv = arguments[1]
        if argv.startswith("[") and argv.endswith("]"):
            inner = argv[1:-1]
            part_start = 0
            nested = []
            quote = None
            escaped = False
            parts = []
            for index, char in enumerate(inner):
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
                elif char in "([{":
                    nested.append(char)
                elif char in pairs:
                    if not nested or nested[-1] != pairs[char]:
                        return set()
                    nested.pop()
                elif char == "," and not nested:
                    parts.append(inner[part_start:index].strip())
                    part_start = index + 1
            if quote or nested:
                return set()
            parts.append(inner[part_start:].strip())
            identifiers.update(part for part in parts if identifier.fullmatch(part))
    return identifiers


def _js_call_arguments(text, names):
    if not names:
        return []
    searchable = _js_mask_noncode_literals(text)
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


def _js_regex_follows_control_header(searchable, close_paren):
    """Return whether a slash after `)` begins a control-body statement."""
    depth = 1
    cursor = close_paren - 1
    while cursor >= 0:
        if searchable[cursor] == ")":
            depth += 1
        elif searchable[cursor] == "(":
            depth -= 1
            if depth == 0:
                prefix = searchable[:cursor]
                return bool(re.search(r"\b(?:if|for|while|with)\s*$", prefix))
        cursor -= 1
    return False


def _js_mask_noncode_literals(text, preserve_string_delimiters=False):
    """Mask strings and bounded regular-expression literals, preserving offsets."""
    masked = list(text)
    index = 0
    while index < len(text):
        char = text[index]
        if char in {'"', "'", "`"}:
            quote = char
            cursor = index
            escaped = False
            while cursor < len(text):
                current = text[cursor]
                closing = (cursor > index and not escaped and current == quote)
                masked[cursor] = (
                    current if (preserve_string_delimiters
                                and (cursor == index or closing))
                    else ("\n" if current == "\n" else " "))
                if cursor > index:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == quote:
                        cursor += 1
                        break
                cursor += 1
            index = cursor
            continue
        if char == "/" and text[index:index + 2] not in {"//", "/*"}:
            previous = index - 1
            while previous >= 0 and text[previous].isspace():
                previous -= 1
            prefix = text[:index]
            can_start = (previous < 0 or text[previous] in "=([{,:;!?&|"
                         or re.search(r"\b(?:return|throw|case)\s*$", prefix)
                         or (text[previous] == ")"
                             and _js_regex_follows_control_header(text, previous)))
            if can_start:
                cursor = index + 1
                escaped = False
                character_class = False
                close = None
                while cursor < len(text) and text[cursor] not in "\r\n":
                    current = text[cursor]
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == "[":
                        character_class = True
                    elif current == "]":
                        character_class = False
                    elif current == "/" and not character_class:
                        close = cursor
                        break
                    cursor += 1
                if close is not None:
                    cursor = close + 1
                    while cursor < len(text) and text[cursor].isalpha():
                        cursor += 1
                    for position in range(index, cursor):
                        masked[position] = ("\n" if text[position] == "\n" else " ")
                    index = cursor
                    continue
        index += 1
    return "".join(masked)


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


def _js_newline_continues_expression(searchable, newline):
    """Recognize bounded token transitions where JavaScript cannot insert a semicolon."""
    left = newline - 1
    while left >= 0 and searchable[left] in " \t\r":
        left -= 1
    right = newline + 1
    while right < len(searchable) and searchable[right] in " \t\r\n":
        right += 1
    if left < 0 or right >= len(searchable):
        return False
    if searchable[right] in "([.?+-*/%&|^:,":
        return True
    if searchable[left] in "([{.,?:=+-*/%&|^!~<>":
        return True
    return False


def _js_bounded_expression_end(searchable, start, comma_ends=True):
    """Return the end of one supported delimiter-balanced JavaScript expression."""
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index in range(start, len(searchable)):
        char = searchable[index]
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack:
                return index
            if stack[-1] != pairs[char]:
                return index
            stack.pop()
        elif not stack and (char == ";" or (comma_ends and char == ",")):
            return index
        elif (not stack and char == "\n"
              and not _js_newline_continues_expression(searchable, index)):
            return index
    return len(searchable)


def _js_bounded_statement_end(searchable, start):
    """Return the exclusive end of one supported JavaScript statement."""
    while start < len(searchable) and searchable[start].isspace():
        start += 1
    if start >= len(searchable):
        return start
    if searchable[start] == "{":
        close = _js_matching_delimiter(searchable, start, "{", "}")
        return None if close is None else close + 1
    if re.match(r"if\b", searchable[start:]):
        condition_start = start + 2
        while (condition_start < len(searchable)
               and searchable[condition_start].isspace()):
            condition_start += 1
        if (condition_start >= len(searchable)
                or searchable[condition_start] != "("):
            return None
        condition_end = _js_matching_delimiter(
            searchable, condition_start, "(", ")")
        if condition_end is None:
            return None
        consequent_end = _js_bounded_statement_end(searchable, condition_end + 1)
        if consequent_end is None:
            return None
        alternate_start = consequent_end
        while (alternate_start < len(searchable)
               and searchable[alternate_start].isspace()):
            alternate_start += 1
        if re.match(r"else\b", searchable[alternate_start:]):
            return _js_bounded_statement_end(searchable, alternate_start + 4)
        return consequent_end
    if re.match(
            r"(?:for|while|do|switch|try|with|function|class|const|let|var|"
            r"return|throw|break|continue|debugger)\b", searchable[start:]):
        return None
    end = _js_bounded_expression_end(searchable, start, comma_ends=False)
    return end + 1 if end < len(searchable) and searchable[end] == ";" else end


def _js_function_scopes(text):
    """Return masked text and bounded function/arrow body scopes."""
    searchable = _js_mask_noncode_literals(text)

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

    concise_scopes = []
    for arrow in re.finditer(r"=>", searchable):
        left = arrow.start() - 1
        while left >= 0 and searchable[left].isspace():
            left -= 1
        if left < 0:
            continue
        if searchable[left] == ")":
            depth = 1
            open_paren = left - 1
            while open_paren >= 0:
                if searchable[open_paren] == ")":
                    depth += 1
                elif searchable[open_paren] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                open_paren -= 1
            if open_paren < 0:
                continue
            param_start, param_end = open_paren + 1, left
        else:
            identifier = re.search(r"([A-Za-z_$][\w$]*)$", searchable[:left + 1])
            if identifier is None:
                continue
            param_start, param_end = identifier.start(1), identifier.end(1)
        body_start = arrow.end()
        while body_start < len(searchable) and searchable[body_start].isspace():
            body_start += 1
        if body_start < len(searchable) and searchable[body_start] == "{":
            open_scopes.append((param_start, param_end, body_start))
        else:
            body_end = _js_bounded_expression_end(searchable, body_start)
            if body_start < body_end:
                concise_scopes.append(
                    (param_start, param_end, body_start - 1, body_end))

    # Class and object-literal method bodies are deferred just like functions.
    # Invocation through property lookup is outside the bounded reachability
    # model, so retain their scopes only to refuse module-execution credit.
    method_searchable = _js_mask_noncode_literals(
        text, preserve_string_delimiters=True)
    method = re.compile(
        r"(?<![\w$])(?P<name>[A-Za-z_$][\w$]*|\[[^\]\r\n]*\]|"
        r"'\s*'|\"\s*\"|"
        r"(?:0[xX][0-9A-Fa-f](?:_?[0-9A-Fa-f])*|"
        r"0[bB][01](?:_?[01])*|0[oO][0-7](?:_?[0-7])*|"
        r"(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|"
        r"\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?))\s*\(")
    control_names = {
        "if", "for", "while", "switch", "catch", "with", "function"
    }
    for match in method.finditer(method_searchable):
        if match.group("name") in control_names:
            continue
        prefix = method_searchable[
            max(0, match.start("name") - 32):match.start("name")]
        if re.search(r"\bfunction\s*$", prefix):
            continue
        open_paren = match.end() - 1
        close_paren = _js_matching_delimiter(searchable, open_paren, "(", ")")
        if close_paren is None:
            continue
        open_brace = close_paren + 1
        while open_brace < len(searchable) and searchable[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(searchable) or searchable[open_brace] != "{":
            continue
        open_scopes.append((open_paren + 1, close_paren, open_brace))

    # Multiline and nested computed names require balanced scanning; the
    # common single-line form is already covered by the regular expression.
    for computed_start, char in enumerate(searchable):
        if char != "[":
            continue
        computed_end = _js_matching_delimiter(
            searchable, computed_start, "[", "]")
        if (computed_end is None
                or not _js_computed_name_needs_balanced_fallback(
                    searchable, computed_start, computed_end)):
            continue
        open_paren = computed_end + 1
        while (open_paren < len(searchable)
               and searchable[open_paren].isspace()):
            open_paren += 1
        if open_paren >= len(searchable) or searchable[open_paren] != "(":
            continue
        close_paren = _js_matching_delimiter(
            searchable, open_paren, "(", ")")
        if close_paren is None:
            continue
        open_brace = close_paren + 1
        while open_brace < len(searchable) and searchable[open_brace].isspace():
            open_brace += 1
        if open_brace >= len(searchable) or searchable[open_brace] != "{":
            continue
        open_scopes.append((open_paren + 1, close_paren, open_brace))

    scopes = []
    for param_start, param_end, open_brace in open_scopes:
        close_brace = _js_matching_delimiter(searchable, open_brace, "{", "}")
        if close_brace is not None:
            scopes.append((param_start, param_end, open_brace, close_brace))
    scopes.extend(concise_scopes)
    return searchable, scopes


def _js_function_scope_key(text, position):
    _searchable, scopes = _js_function_scopes(text)
    containing = [(open_brace, close_brace) for _start, _end, open_brace, close_brace in scopes
                  if open_brace < position < close_brace]
    return max(containing, key=lambda item: item[0]) if containing else None


def _js_computed_name_needs_balanced_fallback(searchable, start, end):
    """Return whether a computed member name bypasses the simple regex."""
    interior = searchable[start + 1:end]
    return "[" in interior or "\r" in interior or "\n" in interior


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


def _js_in_import_export_specifier(searchable, position):
    """Return whether a token is inside an import/export ``{...}`` list."""
    for open_brace, close_brace in _js_block_scopes(searchable):
        if (open_brace < position < close_brace
                and re.search(
                    r"(?:\bexport|\bimport(?:\s+[A-Za-z_$][\w$]*\s*,)?)\s*$",
                              searchable[:open_brace])):
            return True
    return False


def _js_expression_keyword_at(searchable, index, keyword):
    """Recognize a class/function expression keyword, not a member name."""
    if not searchable.startswith(keyword, index):
        return False
    if index and re.match(r"[\w$#]", searchable[index - 1]):
        return False
    end = index + len(keyword)
    if end < len(searchable) and re.match(r"[\w$]", searchable[end]):
        return False
    if searchable[:index].rstrip()[-1:] == ".":
        return False
    if _js_in_import_export_specifier(searchable, index):
        return False
    following = searchable[end:].lstrip()[:1]
    if following in {":", "=", ";", ",", "}"}:
        return False
    if keyword == "class" and following == "(":
        return False
    return True


def _js_function_body_open(searchable, start):
    """Locate a function-expression body after its ``function`` keyword."""
    stack = []
    closing = {")": "(", "]": "["}
    for index in range(start, len(searchable)):
        char = searchable[index]
        if char in "([":
            stack.append(char)
        elif char == "{":
            if not stack:
                return index
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return None
            stack.pop()
        elif char == ";" and not stack:
            return None
    return None


def _js_class_body_open(searchable, start):
    """Locate a class body without mistaking nested heritage braces for it."""
    stack = []
    closing = {")": "(", "]": "[", "}": "{"}
    index = start
    while index < len(searchable):
        char = searchable[index]
        if not stack:
            nested_keyword = next((keyword for keyword in ("class", "function")
                                   if _js_expression_keyword_at(
                                       searchable, index, keyword)), None)
            if nested_keyword is not None:
                nested_start = index + len(nested_keyword)
                nested_open = (_js_class_body_open(searchable, nested_start)
                               if nested_keyword == "class"
                               else _js_function_body_open(
                                   searchable, nested_start))
                if nested_open is None:
                    return None
                nested_close = _js_matching_delimiter(
                    searchable, nested_open, "{", "}")
                if nested_close is None:
                    return None
                index = nested_close + 1
                continue
        if char in "([":
            stack.append(char)
        elif char == "{":
            if not stack:
                return index
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return None
            stack.pop()
        elif char == ";" and not stack:
            return None
        index += 1
    return None


def _js_class_ranges(searchable):
    """Return every balanced class range, or ``None`` for malformed input."""
    ranges = []
    for class_match in re.finditer(r"\bclass\b", searchable):
        if not _js_expression_keyword_at(
                searchable, class_match.start(), "class"):
            continue
        class_open = _js_class_body_open(searchable, class_match.end())
        if class_open is None:
            return None
        class_close = _js_matching_delimiter(
            searchable, class_open, "{", "}")
        if class_close is None:
            return None
        ranges.append((class_match, class_open, class_close))
    return ranges


def _js_static_block_scopes(searchable):
    """Return class static blocks that execute during class definition."""
    scopes = []
    class_ranges = _js_class_ranges(searchable)
    if class_ranges is None:
        return scopes
    for _class_match, class_open, class_close in class_ranges:
        for static in re.finditer(
                r"\bstatic\s*\{", searchable[class_open + 1:class_close]):
            static_start = class_open + 1 + static.start()
            static_open = class_open + 1 + static.end() - 1
            depth = 0
            for char in searchable[class_open + 1:static_start]:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            if depth != 0:
                continue
            static_close = _js_matching_delimiter(
                searchable, static_open, "{", "}")
            if static_close is not None and static_close < class_close:
                scopes.append((static_open, static_close))
    return scopes


def _js_short_circuit_call_reachable(searchable, position):
    """Bound a call reached through one literal short-circuit operator."""
    boundary = max(searchable.rfind(token, 0, position) for token in (";", "{", "}"))
    prefix = searchable[boundary + 1:position]
    operators = list(re.finditer(r"&&|\|\||\?\?", prefix))
    if not operators:
        return True
    if len(operators) != 1:
        return False
    operator = operators[0]
    if prefix[operator.end():].strip(" \t\r\n("):
        return False
    literal = re.search(
        r"\b(true|false|null|undefined)\s*$", prefix[:operator.start()])
    if literal is None:
        return False
    if operator.group() == "??":
        return literal.group(1) in {"null", "undefined"}
    return ((literal.group(1) == "true" and operator.group() == "&&")
            or (literal.group(1) == "false" and operator.group() == "||"))


def _js_optional_call_argument_reachable(searchable, position):
    """Fail closed for expressions suppressed by an unproven optional chain."""
    def segment_end(start):
        index = start
        while index < len(searchable) and searchable[index].isspace():
            index += 1
        identifier = re.match(r"[A-Za-z_$][\w$]*", searchable[index:])
        if identifier is not None:
            return index + identifier.end()
        if index < len(searchable) and searchable[index] in "[(":
            opening = searchable[index]
            closing = ")" if opening == "(" else "]"
            matched = _js_matching_delimiter(
                searchable, index, opening, closing)
            return None if matched is None else matched + 1
        return None

    for optional in re.finditer(r"\?\.", searchable):
        chain_end = segment_end(optional.end())
        if chain_end is None:
            continue
        while True:
            index = chain_end
            while index < len(searchable) and searchable[index].isspace():
                index += 1
            if searchable.startswith("?.", index):
                next_end = segment_end(index + 2)
            elif index < len(searchable) and searchable[index] == ".":
                next_end = segment_end(index + 1)
            elif index < len(searchable) and searchable[index] in "[(":
                next_end = segment_end(index)
            else:
                break
            if next_end is None or next_end <= chain_end:
                break
            chain_end = next_end
        if optional.start() < position < chain_end:
            return False
    return True


def _js_in_deferred_instance_field(searchable, position, text=None):
    """Return whether a call is in an unsupported instance field initializer."""
    member_searchable = (_js_mask_noncode_literals(
        text, preserve_string_delimiters=True) if text is not None else searchable)
    class_ranges = _js_class_ranges(searchable)
    if class_ranges is None:
        return True
    for _class_match, open_brace, close_brace in class_ranges:
        if not (open_brace < position < close_brace):
            continue

        def at_class_member_level(index):
            depth = 0
            for char in searchable[open_brace + 1:index]:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            return depth == 0

        identifier_escape = (
            r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]+\})")
        identifier_name = (
            r"(?:(?:[#$_]|[^\W\d]|" + identifier_escape + r")"
            r"(?:[\w$#]|" + identifier_escape + r")*)")
        member = re.compile(
            r"(?:^|[;{}\r\n])\s*(?!static\b(?!\s*=))(?:accessor\s+)?"
            r"(?:" + identifier_name + r"|\[[^\]\r\n]*\]|"
            r"'[^'\r\n]*'|\"[^\"\r\n]*\"|"
            r"(?:0[xX][0-9A-Fa-f](?:_?[0-9A-Fa-f])*|"
            r"0[bB][01](?:_?[01])*|0[oO][0-7](?:_?[0-7])*|"
            r"(?:\d(?:_?\d)*(?:\.(?:\d(?:_?\d)*)?)?|"
            r"\.\d(?:_?\d)*)(?:[eE][+-]?\d(?:_?\d)*)?))"
            r"\s*=")
        for field in member.finditer(member_searchable, open_brace, position):
            value_start = field.end()
            if not at_class_member_level(value_start):
                continue
            value_end = _js_bounded_expression_end(
                searchable, value_start, comma_ends=False)
            if value_start <= position < value_end:
                return True
        # Multiline and nested computed names require balanced scanning; the
        # common single-line form is already covered by the regular expression.
        computed_member = re.compile(
            r"(?:^|[;{}\r\n])\s*(?!static\b)(?:accessor\s+)?(?=\[)")
        for prefix in computed_member.finditer(
                searchable, open_brace, position):
            computed_start = prefix.end()
            computed_end = _js_matching_delimiter(
                searchable, computed_start, "[", "]")
            if (computed_end is None
                    or not _js_computed_name_needs_balanced_fallback(
                        searchable, computed_start, computed_end)):
                continue
            equals = computed_end + 1
            while equals < len(searchable) and searchable[equals].isspace():
                equals += 1
            if equals >= len(searchable) or searchable[equals] != "=":
                continue
            value_start = equals + 1
            if not at_class_member_level(value_start):
                continue
            value_end = _js_bounded_expression_end(
                searchable, value_start, comma_ends=False)
            if value_start <= position < value_end:
                return True
    return False


def _js_ternary_call_reachable(searchable, position):
    """Bound calls in conditional-expression arms to a proven literal arm."""
    start = max(searchable.rfind(token, 0, position) for token in (";", "{", "}")) + 1
    end_candidates = [candidate for token in (";", "}")
                      if (candidate := searchable.find(token, position)) >= 0]
    end = min(end_candidates) if end_candidates else len(searchable)
    segment = searchable[start:end]
    candidate = position - start
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    questions = []
    pairs = []
    for index, char in enumerate(segment):
        if char in depths:
            depths[char] += 1
            continue
        if char in closing:
            opening = closing[char]
            depths[opening] = max(0, depths[opening] - 1)
            continue
        depth = tuple(depths.values())
        if char == "?":
            previous = segment[index - 1] if index else ""
            following = segment[index + 1] if index + 1 < len(segment) else ""
            if previous == "?" or following in {"?", "."}:
                continue
            questions.append((index, depth))
        elif char == ":":
            for question_index in range(len(questions) - 1, -1, -1):
                if questions[question_index][1] == depth:
                    question, _question_depth = questions.pop(question_index)
                    pairs.append((question, index))
                    break
    controlling = [(question, colon) for question, colon in pairs
                   if question < candidate]
    if not controlling:
        return not any(question < candidate for question, _depth in questions)
    for question, colon in controlling:
        literal = re.search(r"\b(true|false)\s*$", segment[:question])
        if literal is None:
            return False
        in_consequent = candidate < colon
        if in_consequent != (literal.group(1) == "true"):
            return False
    return True


def _js_call_activation_points(text, position):
    """Return module-root activation points for one reachable process call."""
    searchable, scopes = _js_function_scopes(text)
    static_block_scopes = _js_static_block_scopes(searchable)
    trusted_test_names = set()
    for match in re.finditer(
            r"\bimport\s+([A-Za-z_$][\w$]*)\s+from\s+['\"]node:test['\"]", text):
        if searchable[match.start():match.start() + 6] != "import":
            continue
        trusted_test_names.add(match.group(1))
    for match in re.finditer(
            r"\bimport\s*\{([^}]*)\}\s*from\s*['\"]node:test['\"]", text):
        if searchable[match.start():match.start() + 6] != "import":
            continue
        for item in match.group(1).split(","):
            parts = re.split(r"\s+as\s+", item.strip())
            if parts and parts[0] == "test":
                trusted_test_names.add(parts[-1])

    def in_unselected_or_unknown_branch(candidate):
        pattern = re.compile(r"\bif\s*\(")
        for match in pattern.finditer(searchable):
            condition_start = searchable.find("(", match.start(), match.end())
            condition_end = _js_matching_delimiter(
                searchable, condition_start, "(", ")")
            if condition_end is None:
                continue
            condition = searchable[condition_start + 1:condition_end].strip()
            truth = (condition == "true" if condition in {"true", "false"}
                     else None)
            consequent_start = condition_end + 1
            while (consequent_start < len(searchable)
                   and searchable[consequent_start].isspace()):
                consequent_start += 1
            consequent_end = _js_bounded_statement_end(
                searchable, consequent_start)
            if consequent_end is None:
                continue
            if consequent_start <= candidate < consequent_end:
                if truth is not True:
                    return True
                continue
            alternate_start = consequent_end
            while (alternate_start < len(searchable)
                   and searchable[alternate_start].isspace()):
                alternate_start += 1
            alternate = re.match(r"else\b", searchable[alternate_start:])
            if alternate is None:
                continue
            alternate_body_start = alternate_start + alternate.end()
            while (alternate_body_start < len(searchable)
                   and searchable[alternate_body_start].isspace()):
                alternate_body_start += 1
            alternate_end = _js_bounded_statement_end(
                searchable, alternate_body_start)
            if (alternate_end is not None
                    and alternate_body_start <= candidate < alternate_end):
                if truth is not False:
                    return True
                continue
        return False

    def in_unproven_loop(candidate):
        """Reject loop bodies unless bounded syntax proves a first iteration."""
        pattern = re.compile(r"\b(?P<kind>while|for)\s*\(")
        for match in pattern.finditer(searchable):
            header_start = searchable.find("(", match.start(), match.end())
            header_end = _js_matching_delimiter(
                searchable, header_start, "(", ")")
            if header_end is None:
                continue
            body_start = header_end + 1
            while (body_start < len(searchable)
                   and searchable[body_start].isspace()):
                body_start += 1
            body_end = _js_bounded_statement_end(searchable, body_start)
            if body_end is None or not (body_start <= candidate < body_end):
                continue
            header = searchable[header_start + 1:header_end].strip()
            if match.group("kind") == "while":
                proven = header == "true"
            else:
                parts = _js_top_level_parts(header, delimiter=";")
                proven = (len(parts) == 3
                          and parts[1].strip() in {"", "true"})
            if not proven:
                return True
        return False

    def after_unconditional_loop_termination(candidate):
        """Reject calls after a direct break/continue in a containing loop."""
        pattern = re.compile(r"\b(?:while|for)\s*\(")
        token = re.compile(r"\b(?:break|continue)\b|[(){}\[\]]")
        closing = {")": "(", "]": "[", "}": "{"}
        for match in pattern.finditer(searchable):
            header_start = searchable.find("(", match.start(), match.end())
            header_end = _js_matching_delimiter(
                searchable, header_start, "(", ")")
            if header_end is None:
                continue
            body_start = header_end + 1
            while (body_start < len(searchable)
                   and searchable[body_start].isspace()):
                body_start += 1
            if body_start >= len(searchable) or searchable[body_start] != "{":
                continue
            body_end = _js_matching_delimiter(
                searchable, body_start, "{", "}")
            if body_end is None or not (body_start < candidate < body_end):
                continue
            depths = {"(": 0, "[": 0, "{": 0}
            for item in token.finditer(searchable, body_start + 1, candidate):
                value = item.group()
                if value in depths:
                    depths[value] += 1
                elif value in closing:
                    opening = closing[value]
                    depths[opening] = max(0, depths[opening] - 1)
                elif not any(depths.values()):
                    return True
        return False

    def in_unproven_switch(candidate):
        """Fail closed for calls inside unsupported switch selection."""
        for match in re.finditer(r"\bswitch\s*\(", searchable):
            header_start = searchable.find("(", match.start(), match.end())
            header_end = _js_matching_delimiter(
                searchable, header_start, "(", ")")
            if header_end is None:
                continue
            body_start = header_end + 1
            while (body_start < len(searchable)
                   and searchable[body_start].isspace()):
                body_start += 1
            if body_start >= len(searchable) or searchable[body_start] != "{":
                continue
            body_end = _js_matching_delimiter(
                searchable, body_start, "{", "}")
            if body_end is not None and body_start < candidate < body_end:
                return True
        return False

    def in_unproven_catch(candidate):
        """Fail closed for calls in handlers without a proven throw."""
        for match in re.finditer(r"\bcatch\b", searchable):
            body_start = match.end()
            while body_start < len(searchable) and searchable[body_start].isspace():
                body_start += 1
            if body_start < len(searchable) and searchable[body_start] == "(":
                header_end = _js_matching_delimiter(
                    searchable, body_start, "(", ")")
                if header_end is None:
                    continue
                body_start = header_end + 1
                while (body_start < len(searchable)
                       and searchable[body_start].isspace()):
                    body_start += 1
            if body_start >= len(searchable) or searchable[body_start] != "{":
                continue
            body_end = _js_matching_delimiter(
                searchable, body_start, "{", "}")
            if body_end is not None and body_start < candidate < body_end:
                return True
        return False

    def after_unconditional_termination(candidate):
        """Reject calls after direct or completed-block return/throw."""
        containing_execution_scopes = [
            (scope[2], scope[3]) for scope in scopes
            if scope[2] < candidate < scope[3] and searchable[scope[2]] == "{"]
        containing_execution_scopes.extend(
            scope for scope in static_block_scopes
            if scope[0] < candidate < scope[1])
        if not containing_execution_scopes:
            return False
        function_open, function_close = max(
            containing_execution_scopes, key=lambda item: item[0])
        all_blocks = _js_block_scopes(searchable)
        blocks = [(function_open, function_close)]
        blocks.extend(
            block for block in all_blocks
            if function_open < block[0] < candidate < block[1] < function_close)
        for block_open, block_close in all_blocks:
            if not (function_open < block_open < block_close < candidate < function_close):
                continue
            parents = [block for block in all_blocks
                       if block[0] < block_open < block_close < block[1]]
            parent = (max(parents, key=lambda item: item[0])
                      if parents else None)
            if parent != (function_open, function_close):
                continue
            previous = block_open - 1
            while previous > function_open and searchable[previous].isspace():
                previous -= 1
            if searchable[previous] not in "{;}":
                continue
            blocks.append((block_open, block_close))
        token = re.compile(r"\b(?:return|throw)\b|[(){}\[\]]")
        closing = {")": "(", "]": "[", "}": "{"}
        for block_open, block_close in blocks:
            depths = {"(": 0, "[": 0, "{": 0}
            for match in token.finditer(
                    searchable, block_open + 1, min(candidate, block_close)):
                value = match.group()
                if value in depths:
                    depths[value] += 1
                elif value in closing:
                    opening = closing[value]
                    depths[opening] = max(0, depths[opening] - 1)
                elif not any(depths.values()):
                    operand_end = _js_bounded_expression_end(
                        searchable, match.end(), comma_ends=False)
                    if candidate < operand_end:
                        continue
                    return True
        return False

    def after_unproven_await(candidate):
        """Reject continuation after an await the bounded model cannot prove."""
        candidate_scope = _js_function_scope_key(searchable, candidate)
        start = candidate_scope[0] + 1 if candidate_scope is not None else 0
        for match in re.compile(r"\bawait\b").finditer(
                searchable, start, candidate):
            if _js_function_scope_key(searchable, match.start()) == candidate_scope:
                return True
        return False

    def containing_named_function(candidate):
        containing = sorted(
            (scope for scope in scopes
             if (scope[0] <= candidate < scope[1]
                 or scope[2] < candidate < scope[3])),
            key=lambda item: item[2], reverse=True)
        if not containing:
            return None
        param_start, param_end, open_brace, close_brace = containing[0]
        header = re.search(
            r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\($", searchable[:param_start])
        if header is None:
            prefix = searchable[:open_brace + 1]
            for name in trusted_test_names:
                callback = re.compile(
                    r"(?<![\w$.])" + re.escape(name) +
                    r"(?![\w$])\s*\([^{};]*=>\s*(?:\{)?$")
                match = callback.search(prefix)
                if match:
                    call_start = match.start()
                    if (_js_parameter_shadows(searchable, call_start, name)
                            or _js_local_declaration_shadows(searchable, call_start, name)):
                        return ()
                    return ("test-callback", call_start, open_brace, close_brace, name)
            return ()
        return (header.group(1), header.start(1), param_start, param_end,
                open_brace, close_brace)

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
        if not _js_optional_call_argument_reachable(searchable, candidate):
            return set()
        if _js_in_deferred_instance_field(searchable, candidate, text):
            return set()
        if not _js_short_circuit_call_reachable(searchable, candidate):
            return set()
        if not _js_ternary_call_reachable(searchable, candidate):
            return set()
        if in_unselected_or_unknown_branch(candidate):
            return set()
        if in_unproven_loop(candidate):
            return set()
        if after_unconditional_loop_termination(candidate):
            return set()
        if in_unproven_switch(candidate):
            return set()
        if in_unproven_catch(candidate):
            return set()
        if after_unconditional_termination(candidate):
            return set()
        if after_unproven_await(candidate):
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
        name, definition_start, param_start, param_end, open_brace, close_brace = function
        identity = (name, open_brace, close_brace)
        if identity in seen:
            return set()
        definition_scope = _js_function_scope_key(searchable, definition_start)
        guard = (name, close_brace + 1, definition_scope)
        calls = re.compile(
            r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])\s*\(")
        activations = set()
        default_parameter = None
        if param_start <= candidate < param_end:
            relative = candidate - param_start
            cursor = 0
            for index, part in enumerate(
                    _js_top_level_parts(searchable[param_start:param_end])):
                if cursor <= relative <= cursor + len(part):
                    default_parameter = index
                    break
                cursor += len(part) + 1
        for call in calls.finditer(searchable):
            if call.start() == definition_start:
                continue
            if open_brace < call.start() < close_brace:
                continue
            if _js_helper_call_shadowed(
                    searchable, call.start(), name, definition_start):
                continue
            if default_parameter is not None:
                call_open = call.end() - 1
                call_close = _js_matching_delimiter(
                    searchable, call_open, "(", ")")
                if call_close is None:
                    continue
                argument_text = searchable[call_open + 1:call_close]
                arguments = ([] if not argument_text.strip()
                             else _js_top_level_parts(argument_text))
                if (default_parameter < len(arguments)
                        and arguments[default_parameter].strip() != "undefined"):
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
                                  after_position=None,
                                  permitted_declaration=None):
    """Return true for a declaration in a containing lexical block."""
    searchable, function_scopes = _js_function_scopes(text)
    blocks = _js_block_scopes(searchable)
    containing = sorted(
        (block for block in blocks if block[0] < position < block[1]),
        key=lambda item: item[0], reverse=True)
    if not containing:
        return False
    direct_pattern = re.compile(
        r"\b(?:function|class)\s+(" + re.escape(name) + r")(?![\w$])")
    for block in containing:
        for declaration in direct_pattern.finditer(searchable, block[0] + 1, block[1]):
            if after_position is not None and declaration.start() < after_position:
                continue
            declaration_blocks = [candidate for candidate in blocks
                                  if candidate[0] < declaration.start() < candidate[1]]
            declaration_block = (max(declaration_blocks, key=lambda item: item[0])
                                 if declaration_blocks else None)
            if declaration_block == block:
                if declaration.start(1) == permitted_declaration:
                    continue
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
                if start + simple.start(1) == permitted_declaration:
                    continue
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
    if _js_local_declaration_shadows(
            text, position, name, permitted_declaration=definition_start):
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


def _unique_workflow_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON workflow key")
        result[key] = value
    return result


def _reject_workflow_json_constant(_value):
    raise ValueError("nonstandard JSON workflow constant")


def _strict_workflow_json(text):
    workflow = json.loads(text, object_pairs_hook=_unique_workflow_json_object,
                          parse_constant=_reject_workflow_json_constant)
    if not isinstance(workflow, dict) or not isinstance(workflow.get("jobs"), dict):
        raise ValueError("JSON workflow must have a jobs object")
    for job in workflow["jobs"].values():
        if not isinstance(job, dict):
            raise ValueError("JSON workflow job must be an object")
        if "steps" in job:
            if not isinstance(job["steps"], list):
                raise ValueError("JSON workflow steps must be a list")
            for step in job["steps"]:
                if not isinstance(step, dict):
                    raise ValueError("JSON workflow step must be an object")
                if "run" in step and not isinstance(step["run"], str):
                    raise ValueError("JSON workflow run must be a string")
    return workflow


def _workflow_reference_commands(text):
    """Extract only job step run strings from strict JSON or supported YAML."""
    if text.lstrip().startswith(("{", "[")):
        try:
            workflow = _strict_workflow_json(text)
        except (ValueError, TypeError):
            return []
        blocks = []
        for job in workflow["jobs"].values():
            if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
                continue
            for step in job["steps"]:
                if isinstance(step, dict) and isinstance(step.get("run"), str):
                    commands = [_strip_shell_comment(line.strip()).strip()
                                for line in step["run"].splitlines()]
                    blocks.append((0, [command for command in commands if command]))
        return blocks
    return _workflow_commands(text)


def _workflow_unwrap_env(tokens):
    """Recognize only env with explicit unset options before a command."""
    if not tokens or tokens[0] != "env":
        return tokens
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return tokens[index + 1:]
        if token in ("-u", "--unset"):
            if index + 1 >= len(tokens) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tokens[index + 1]):
                return []
            index += 2
            continue
        if token.startswith("--unset="):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token[len("--unset="):]):
                return []
            index += 1
            continue
        if token.startswith("-") or "=" in token:
            return []
        return tokens[index:]
    return []


def workflow_tool_references(text, tools):
    references = set()
    for _number, commands in _workflow_reference_commands(text):
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
            if tokens and tokens[0] == "env":
                tokens = _workflow_unwrap_env(tokens)
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
            if path.startswith(".github/workflows/") and text.lstrip().startswith(("{", "[")):
                try:
                    _strict_workflow_json(text)
                except (ValueError, TypeError):
                    findings.append({"path": path, "area": kind, "class": "unparseable",
                                     "line": 0, "detail": "invalid or unsupported JSON workflow"})
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
