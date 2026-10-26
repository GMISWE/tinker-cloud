#!/usr/bin/env bash
# Tree rules ruff cannot see; each one a move or a doc edit breaks silently.
#
#   scripts/check_tree.sh                       # this repo
#   scripts/check_tree.sh --paths-in FILE...    # rule 1 only, on files elsewhere
#                                               # (the monorepo's specs name paths here)
#
#   1. every `tests/<...>.py` path named in a .md/.py/.sh/.yml/.toml file exists
#   2. every directory holding a test_*.py is a package (__init__.py), so
#      same-named test files in two suites import under distinct names
#   3. no test imports a sibling test module (`from test_x import`): shared
#      helpers live in tests/datums.py, tests/fakes/ or a conftest
set -euo pipefail
rc=0
# a path of this repo starts at `tests/`; `other_repo/tests/x.py` is somebody else's
PATH_RE='(?<![\w/])tests/[A-Za-z0-9_/]+\.py'
check_paths() {   # stdin: file:line:path (-H: grep names the file even for a single input)
  while IFS=: read -r file line path; do
    [ -e "$path" ] || { echo "$file:$line: $path does not exist" >&2; rc=1; }
  done
}
if [ "${1:-}" = --paths-in ]; then
  shift
  files=(); for f in "$@"; do files+=("$(realpath "$f")"); done
  cd "$(dirname "$0")/.."
  check_paths < <(grep -HnoP "$PATH_RE" "${files[@]}" || true)
  exit $rc
fi
cd "$(dirname "$0")/.."
# plain find, no git: the pre-push hook runs this on a `git archive` tree
files() { find . \( -name .git -o -name .oss-scanner -o -name __pycache__ -o -name .pytest_cache \) -prune -o -type f \( "$@" \) -print0; }
check_paths < <(files -name '*.md' -o -name '*.py' -o -name '*.sh' -o -name '*.yml' -o -name '*.toml' | xargs -0 grep -HnoP "$PATH_RE" || true)
while IFS= read -r d; do
  [ -f "$d/__init__.py" ] || { echo "$d: holds tests but has no __init__.py" >&2; rc=1; }
done < <(find tests -name 'test_*.py' -print0 | xargs -0 -n1 dirname | sort -u)
if files -path './tests/*' -name '*.py' | xargs -0 grep -nE '^\s*(from|import) test_'; then
  echo "tests import a sibling test module; move the helper to tests/datums.py, tests/fakes/ or a conftest" >&2; rc=1
fi
exit $rc
