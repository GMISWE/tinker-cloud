#!/usr/bin/env bash
# Cases for rule 4 of scripts/check_tree.sh and for scripts/check_ast.py. Each case builds a small
# tree that passes, changes one thing and runs the check there. One line per case, `ok <case>` or
# `not ok <case>` with the check's output under it; exit 1 if any case is not ok.
#
#   scripts/check_tree_test.sh
#
# The cases read ruff's concise output, so they hold for the ruff that .github/workflows/lint.yml pins.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fails=0
# a handler ruff reports as BLE001 alone: it assigns, so S110 stays quiet
BROAD='try:\n    x = 1\nexcept Exception:%s\n    x = 2\n'
GUARD='try:\n    import zstandard\nexcept ImportError:%s\n    zstandard = None\n'

# A tree the check passes: this repository's pyproject.toml and scripts, one broad handler and one
# import guard under training/pkg (issue 1), one xfail mark without raises= under tests/unit (issue 2).
seed() {
  tree=$T/$1
  mkdir -p "$tree/scripts" "$tree/training/pkg" "$tree/tests/unit"
  cp "$root/pyproject.toml" "$tree/"
  cp "$root/scripts/check_tree.sh" "$root/scripts/check_ast.py" "$tree/scripts/"
  # shellcheck disable=SC2059  # the format is the constant; the argument is the comment
  printf "$BROAD" '  # noqa: BLE001  (#1)' > "$tree/training/pkg/a.py"
  # shellcheck disable=SC2059
  printf "$GUARD" '  # noqa: TREE002  (#1)' > "$tree/training/pkg/b.py"
  # tests/datums.py: the check's own text names it, and its first rule wants every named test path
  : > "$tree/tests/__init__.py"; : > "$tree/tests/unit/__init__.py"; : > "$tree/tests/datums.py"
  printf 'import pytest\n\n\n@pytest.mark.xfail(reason="r")  # noqa: TREE001  (#2)\ndef test_a():\n    assert False\n' > "$tree/tests/unit/test_a.py"
  printf 'training/pkg\t2\t1\ntests/unit\t1\t2\n' > "$tree/scripts/suppression_baseline.tsv"
}
# $1 case, $2 the exit code wanted, $3 a pattern the output holds (empty: any output), $4... the command
# that runs in the tree (default: the check)
expect() {
  local name=$1 want=$2 pattern=$3 out got; shift 3
  [ $# -gt 0 ] || set -- scripts/check_tree.sh
  out=$(cd "$tree" && "$@" 2>&1); got=$?
  if [ "$got" -eq "$want" ] && grep -qE -- "$pattern" <<< "$out"; then echo "ok $name"
  else echo "not ok $name"; echo "    exit $got, wanted $want and /$pattern/"; sed 's/^/    /' <<< "$out"; fails=1; fi
}
# $1 case, $2 a file to write under the tree, $3 its content (printf format), $4 exit, $5 pattern
seeded() {
  seed "$1"; mkdir -p "$(dirname "$tree/$2")"
  # shellcheck disable=SC2059
  printf "$3" > "$tree/$2"
  expect "$1" "$4" "$5"
}
untagged="must be suppressed on this line with its issue"
c=training/pkg/c.py

seed clean-tree; expect clean-tree 0 ''

# the suppression form and the count
# shellcheck disable=SC2059
seeded no-issue-tag "$c" "$(printf "$BROAD" '  # noqa: BLE001')" 1 "^$c:3: BLE001 $untagged"
# shellcheck disable=SC2059
seeded one-more "$c" "$(printf "$BROAD" '  # noqa: BLE001  (#1)')" 1 '^training/pkg: 3 violations in the tree, 2 in '
seed one-fewer; rm "$tree/training/pkg/b.py"; expect one-fewer 1 '^training/pkg: 1 violations in the tree, 2 in '
# shellcheck disable=SC2059
seeded no-row training/other/c.py "$(printf "$BROAD" '  # noqa: BLE001  (#1)')" 1 '^training/other/c.py:3: BLE001 under a path with no row'
seed zero-row; printf 'training/other\t0\t1\n' >> "$tree/scripts/suppression_baseline.tsv"; expect zero-row 1 'the row of training/other holds 0'
seed malformed-row; printf 'training/other\tmany\t1\n' >> "$tree/scripts/suppression_baseline.tsv"; expect malformed-row 1 'the row of training/other needs a count and an issue number'

# every way to silence ruff still counts
# shellcheck disable=SC2059
seeded blanket-noqa "$c" "$(printf "$BROAD" '  # noqa')" 1 "^$c:3: BLE001 $untagged"
# shellcheck disable=SC2059
seeded ruff-disable-range "$c" "# ruff: disable[BLE001]\n$(printf "$BROAD" '')# ruff: enable[BLE001]\n" 1 "^$c:4: BLE001 $untagged"
seed per-file-ignore
# shellcheck disable=SC2059
printf "$BROAD" '' > "$tree/$c"; sed -i 's|^\[tool.ruff.lint.per-file-ignores\]$|&\n"training/pkg/c.py" = ["BLE001"]|' "$tree/pyproject.toml"
grep -q '^"training/pkg/c.py" = ' "$tree/pyproject.toml" || { echo "not ok per-file-ignore: the seed did not land"; fails=1; }
expect per-file-ignore 1 "^$c:3: BLE001 $untagged"
seed nested-config; mkdir -p "$tree/training/pkg/sub"
# shellcheck disable=SC2059
printf "$BROAD" '' > "$tree/training/pkg/sub/c.py"; printf '[lint]\nignore = ["BLE001"]\n' > "$tree/training/pkg/sub/ruff.toml"
expect nested-config 1 "^training/pkg/sub/c.py:3: BLE001 $untagged"
seed exclude
# shellcheck disable=SC2059
printf "$BROAD" '' > "$tree/$c"; sed -i 's|^extend-exclude = \[|&"training/pkg/c.py", |' "$tree/pyproject.toml"
grep -q '^extend-exclude = \["training/pkg/c.py", ' "$tree/pyproject.toml" || { echo "not ok exclude: the seed did not land"; fails=1; }
expect exclude 1 "^$c:3: BLE001 $untagged"
# shellcheck disable=SC2059
seeded default-excluded-dir training/pkg/node_modules/c.py "$(printf "$BROAD" '')" 1 "^training/pkg/node_modules/c.py:3: BLE001 $untagged"
seed gitignored
# shellcheck disable=SC2059
printf "$BROAD" '' > "$tree/$c"; git -C "$tree" init -q; echo c.py > "$tree/.gitignore"
expect gitignored 1 "^$c:3: BLE001 $untagged"

# the rule list and what ruff cannot count
seed added-rule-bare-noqa; sed -i 's|^extend-select = \["B904",|extend-select = ["B904", "E722",|' "$tree/pyproject.toml"
printf 'try:\n    x = 1\nexcept:  # noqa: E722\n    x = 2\n' > "$tree/$c"
expect added-rule-bare-noqa 1 "^$c:3: E722 $untagged"
seeded unparseable "$c" 'x = (\n' 1 'scripts/check_ast.py could not check the files'
# a file that is no text: ruff warns and passes, check_ast.py stops, and the guard after it is not taken as counted
seed undecodable; printf 'x = "\377"\n' > "$tree/training/pkg/b2.py"
# shellcheck disable=SC2059
printf "$GUARD" '' > "$tree/$c"
expect undecodable 1 'scripts/check_ast.py could not check the files'
seed malformed-select; sed -i 's|^extend-select = \["B904",|extend-select = ["B9",|' "$tree/pyproject.toml"
expect malformed-select 1 'extend-select must be one line of full rule codes'

# TREE001: an xfail mark without raises=
t=tests/unit/test_c; t=$t.py   # in two steps: the check's first rule reads a whole test path here as a claim that it exists
seeded xfail-bare "$t" 'import pytest\n\n\n@pytest.mark.xfail\ndef test_c():\n    assert False\n' 1 "^$t:4: TREE001 $untagged"
seeded xfail-no-raises "$t" 'import pytest\n\n\n@pytest.mark.xfail(strict=True, reason="r")\ndef test_c():\n    assert False\n' 1 "^$t:4: TREE001 $untagged"
seeded xfail-in-param "$t" 'import pytest\n\n\n@pytest.mark.parametrize("x", [pytest.param(1, marks=pytest.mark.xfail(reason="r"))])\ndef test_c(x):\n    assert x\n' 1 "^$t:4: TREE001 $untagged"
seeded xfail-with-raises-clean "$t" 'import pytest\n\n\n@pytest.mark.xfail(raises=AssertionError, reason="r")\ndef test_c():\n    pytest.xfail("r")\n' 0 ''

# TREE002: an import guard
# shellcheck disable=SC2059
seeded guard-importerror "$c" "$(printf "$GUARD" '')" 1 "^$c:3: TREE002 $untagged"
seeded guard-modulenotfound "$c" 'try:\n    from a import b\nexcept ModuleNotFoundError:\n    b = None\n' 1 "^$c:3: TREE002 $untagged"
seeded guard-tuple "$c" 'try:\n    import a\nexcept (AttributeError, ImportError):\n    a = None\n' 1 "^$c:3: TREE002 $untagged"
seeded guard-suppress "$c" 'import contextlib\n\nwith contextlib.suppress(ImportError):\n    import a\n' 1 "^$c:3: TREE002 $untagged"
seeded guard-except-star "$c" 'try:\n    import a\nexcept* ImportError:\n    a = None\n' 1 "^$c:3: TREE002 $untagged"
seeded guard-suppress-twice "$c" 'import contextlib\n\nwith contextlib.suppress(ImportError), contextlib.suppress(ModuleNotFoundError):  # noqa: TREE002  (#1)\n    import a\n' 1 '^training/pkg: 3 violations in the tree, 2 in '
seeded guard-nested "$c" 'import os\n\ntry:\n    if os.name:\n        import a\nexcept ImportError:\n    a = None\n' 1 "^$c:6: TREE002 $untagged"
seeded guard-in-nested-def-clean "$c" 'try:\n    def g():\n        import a\n        return a\nexcept ImportError:\n    g = None\n' 0 ''
seeded guard-reraise-clean "$c" 'try:\n    import a\nexcept ImportError as e:\n    raise RuntimeError("install a") from e\n' 0 ''
# shellcheck disable=SC2059
seeded guard-no-tag "$c" "$(printf "$GUARD" '  # noqa: TREE002')" 1 "^$c:3: TREE002 $untagged"
# shellcheck disable=SC2059
seeded guard-one-more "$c" "$(printf "$GUARD" '  # noqa: TREE002  (#1)')" 1 '^training/pkg: 3 violations in the tree, 2 in '

# a noqa for one of the two on a line that breaks neither; ruff's RUF100 does not see it
seeded stale-tree002-noqa "$c" 'x = 1  # noqa: TREE002  (#1)\n' 1 "^$c:1:.*unused-noqa TREE002"
seeded stale-noqa-no-space "$c" 'x = 1  #noqa: TREE002\n' 1 "^$c:1:.*unused-noqa TREE002"
seeded stale-noqa-spaced-colon "$c" 'x = 1  # noqa : TREE002\n' 1 "^$c:1:.*unused-noqa TREE002"
seeded stale-noqa-file-level "$c" '# ruff: noqa: TREE002\nx = 1\n' 1 "^$c:1:.*unused-noqa TREE002"
seeded stale-noqa-second-comment "$c" "$(printf "$BROAD" '  # noqa: BLE001  (#1)  # noqa: TREE002')" 1 "^$c:3:.*unused-noqa TREE002"
seeded stale-noqa-upper-case "$c" 'x = 1  # NOQA: TREE002\n' 1 "^$c:1:.*unused-noqa TREE002"
seeded stale-tree001-noqa "$t" 'x = 1  # noqa: TREE001  (#2)\n' 1 "^$t:1:.*unused-noqa TREE001"
# ruff takes the comments of the passing tree with RUF102 (an unknown code in a noqa) switched on,
# which no configured run does today: pyproject.toml's `external` is there for the day one does
seed ruff-accepts-tree-noqa
expect ruff-accepts-tree-noqa 0 'All checks passed' ruff check --no-cache --extend-select RUF102 training/ tests/
exit $fails
