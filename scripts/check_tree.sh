#!/usr/bin/env bash
# Tree rules a plain `ruff check` cannot see.
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
#   4. every violation of a rule pyproject.toml's extend-select names, counted with every suppression
#      switched off, and of a rule of scripts/check_ast.py (the syntax tree decides those), is
#      suppressed on its own line by `# noqa: <rule>  (#<n>)` naming the issue of its path, and each
#      path holds exactly the count scripts/suppression_baseline.tsv gives it
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
# Rule 4. The tracked rules are the ones extend-select names, read from its line, so a rule selected
# later is tracked from that change on. The line is one line of full rule codes (three digits or
# more: a prefix such as "BLE" or "B9" selects rules this check could not name) or the check stops.
# RUF100 stays out: it reports suppressions, not code. check_ast.py names its own rules.
select_line=$(grep -E '^extend-select = ' pyproject.toml || true)   # no such line: the next test says so
[[ $select_line =~ ^extend-select\ =\ \[(\"[A-Z]+[0-9]{3,}\"(,\ )?)+\]$ ]] \
  || { echo "pyproject.toml: extend-select must be one line of full rule codes; found: ${select_line:-no such line}" >&2; exit 1; }
ruff_rules=$(grep -oE '[A-Z]+[0-9]+' <<< "$select_line" | grep -vx RUF100 | paste -sd, - || true)
[ -n "$ruff_rules" ] || { echo "pyproject.toml: extend-select names no rule besides RUF100, so there is no rule to track" >&2; exit 1; }
tracked=$ruff_rules,$(python3 scripts/check_ast.py --rules)
baseline=scripts/suppression_baseline.tsv
declare -A want issue have
while IFS=$'\t' read -r path count number || [ -n "$path" ]; do   # a last row with no newline is a row
  case $path in ''|'#'*) continue ;; esac
  [[ $count =~ ^[0-9]+$ && $number =~ ^[0-9]+$ ]] || { echo "$baseline: the row of $path needs a count and an issue number" >&2; exit 1; }
  want[$path]=$count; issue[$path]=$number; have[$path]=0
done < "$baseline"
# Every violation, however it is silenced: --ignore-noqa drops the comments, --isolated drops every
# config file (per-file ignores, excludes, nested configs), and naming each file passes ruff's default
# excludes and .gitignore. ruff exits 0 clean, 1 on violations, 2 on its own error (its documented codes).
mapfile -d '' -t sources < <(files -path './training/*' -name '*.py' -o -path './tests/*' -name '*.py')
ruff_violations=$(ruff check --isolated --no-cache --select "$ruff_rules" --ignore-noqa --output-format concise "${sources[@]}") \
  || [ $? -eq 1 ] || { echo "ruff could not count the violations of $ruff_rules" >&2; exit 1; }
# check_ast.py reads no suppression either; it exits 0 clean, 1 on findings and 2 on its own error
ast_violations=$(python3 scripts/check_ast.py "${sources[@]}") \
  || [ $? -eq 1 ] || { echo "scripts/check_ast.py could not check the files" >&2; exit 1; }
while IFS=: read -r file line _ report; do
  read -r rule _ <<< "$report"
  # A diagnostic that is no tracked rule fails as it stands: `unused-noqa` from check_ast.py, a `noqa`
  # of its rules on a line that breaks none; and ruff's `invalid-syntax` for a file its parser refuses
  # and Python's takes (check_ast.py stops the check on the others), whose violations cannot be counted.
  case ",$tracked," in
    *",$rule,"*) ;;
    *) echo "$file:$line:$report" >&2; rc=1; continue ;;
  esac
  path=""
  for p in "${!want[@]}"; do
    if [[ $file == "$p"/* ]] && [ ${#p} -gt ${#path} ]; then path=$p; fi
  done
  if [ -z "$path" ]; then echo "$file:$line: $rule under a path with no row in $baseline" >&2; rc=1; continue; fi
  have[$path]=$((have[$path] + 1))
  if ! sed -n "${line}p" "$file" | grep -qE "# noqa: ([A-Z]+[0-9]+, )*$rule(, [A-Z]+[0-9]+)* +\(#${issue[$path]}\)"; then
    echo "$file:$line: $rule must be suppressed on this line with its issue: # noqa: $rule  (#${issue[$path]})" >&2; rc=1
  fi
done < <(grep -E '^[^:]+:[0-9]+:[0-9]+: ' <<< "$ruff_violations"$'\n'"$ast_violations" || true)   # every diagnostic line; a clean tree gives none
for p in "${!want[@]}"; do
  if [ "${want[$p]}" -eq 0 ]; then echo "$baseline: the row of $p holds 0; delete it" >&2; rc=1
  elif [ "${have[$p]}" -ne "${want[$p]}" ]; then echo "$p: ${have[$p]} violations in the tree, ${want[$p]} in $baseline" >&2; rc=1; fi
done
exit $rc
