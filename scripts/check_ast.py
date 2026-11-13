#!/usr/bin/env python3
"""Two rules ruff does not have, decided on the syntax tree. scripts/check_tree.sh runs this.

    scripts/check_ast.py FILE...    one line per finding, in ruff's concise form; exit 1 if any,
                                    2 on an error of this script (a file it cannot read or parse)
    scripts/check_ast.py --rules    the rule codes, comma-separated

Needs Python 3.11 (`except*`).

TREE001  a `mark.xfail` without `raises=`: any error then passes as the expected failure.
TREE002  an import guard: a `try` with an import statement in its body (the bodies of nested
         functions and classes apart) and a handler for ImportError that does not end in `raise`,
         or the same import under `suppress(ImportError)`. Reported on the `except` or `with` line.

A `# noqa` that names one of the two on a line that does not break it is a finding as well
(`unused-noqa`): ruff's RUF100 does not look at a code ruff does not own.
"""
import ast
import io
import os
import re
import sys
import tokenize
import traceback

RULES = {
    "TREE001": "`xfail` mark without `raises=`",
    "TREE002": "import guard: a handler for ImportError that does not end in `raise`",
}
MISSING_MODULE = {"ImportError", "ModuleNotFoundError"}
OWN_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
# A comment that holds the word, in any case, names every rule code found anywhere in it. That is
# looser than ruff reads a directive. The form a real violation must carry is strict and lives in
# check_tree.sh, beside the issue number it checks.
NOQA = re.compile(r"\bnoqa\b", re.IGNORECASE)


def last_names(expr):
    """The last name of each dotted name in an expression or a tuple of them: `builtins.ImportError`
    gives ImportError, `contextlib.suppress` gives suppress."""
    items = expr.elts if isinstance(expr, ast.Tuple) else [expr]
    return {i.id if isinstance(i, ast.Name) else i.attr for i in items if isinstance(i, (ast.Name, ast.Attribute))}


def holds_import(statements):
    pending = list(statements)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        if not isinstance(node, OWN_SCOPE):
            pending.extend(ast.iter_child_nodes(node))
    return False


def violations(tree):
    """(line, column, rule) for each violation of the two rules."""
    found = []
    with_raises = {
        id(n.func) for n in ast.walk(tree)
        if isinstance(n, ast.Call) and any(k.arg == "raises" for k in n.keywords)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "xfail":
            owner = node.value   # `pytest.mark` or a bare `mark`
            on_mark = isinstance(owner, ast.Attribute) and owner.attr == "mark" or isinstance(owner, ast.Name) and owner.id == "mark"
            if on_mark and id(node) not in with_raises:
                found.append((node.lineno, node.col_offset + 1, "TREE001"))
        elif isinstance(node, (ast.Try, ast.TryStar)) and holds_import(node.body):
            for handler in node.handlers:
                if handler.type is not None and last_names(handler.type) & MISSING_MODULE and not isinstance(handler.body[-1], ast.Raise):
                    found.append((handler.lineno, handler.col_offset + 1, "TREE002"))
        elif isinstance(node, (ast.With, ast.AsyncWith)) and holds_import(node.body):
            calls = [item.context_expr for item in node.items if isinstance(item.context_expr, ast.Call)]
            if any(last_names(c.func) == {"suppress"} and any(last_names(a) & MISSING_MODULE for a in c.args) for c in calls):
                found.append((node.lineno, node.col_offset + 1, "TREE002"))   # one statement, one violation
    return found


def noqa_codes(source):
    """(line, column, code) for each of this script's codes a `# noqa:` comment names."""
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and NOQA.search(token.string):
            for code in RULES:
                if re.search(rf"\b{code}\b", token.string):
                    yield token.start[0], token.start[1] + 1, code


def main(argv):
    if argv == ["--rules"]:
        print(",".join(RULES))
        return 0
    rc = 0
    for path in argv:
        shown = os.path.normpath(path)   # `./training/x.py` prints as ruff prints it
        with tokenize.open(path) as f:   # decodes as Python does: a BOM, a coding line
            source = f.read()
        broken = violations(ast.parse(source, filename=path))
        for line, column, rule in sorted(broken):
            print(f"{shown}:{line}:{column}: {rule} {RULES[rule]}")
        on_line = {(line, rule) for line, _, rule in broken}
        for line, column, code in noqa_codes(source):
            if (line, code) not in on_line:
                print(f"{shown}:{line}:{column}: unused-noqa {code} is not broken on this line")
                rc = 1
        rc = rc or bool(broken)
    return int(rc)


def exit_2(kind, error, trace):
    """An error of this script exits 2. Python's own exit for it is 1, which here means findings:
    the caller would keep the lines printed so far and take the files after the error as clean."""
    traceback.print_exception(kind, error, trace)
    sys.stdout.flush()
    os._exit(2)


if __name__ == "__main__":
    sys.excepthook = exit_2
    sys.exit(main(sys.argv[1:]))
