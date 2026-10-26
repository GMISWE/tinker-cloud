#!/usr/bin/env bash
# Open a PR from a body file, verify the body landed, print the URL.
#
#   scripts/pr_open.sh <head-branch> <title> <body-file> [base=main]
#   gh pr checks <n> --watch        # then wait for CI
#
# A body passed inline can be mangled or refused on the way; a file is read
# back and compared, as scripts/pr_body.sh does for edits.
set -euo pipefail
BRANCH="$1"; TITLE="$2"; FILE="$3"; BASE="${4:-main}"
URL=$(gh pr create --base "$BASE" --head "$BRANCH" --title "$TITLE" --body-file "$FILE")
N="${URL##*/}"
LIVE="$(gh pr view "$N" --json body -q .body)"
if [ "$(printf '%s' "$LIVE" | tr -d '\r')" != "$(tr -d '\r' < "$FILE")" ]; then
  echo "PR #$N body differs from $FILE after create" >&2; exit 1
fi
echo "$URL"
