#!/usr/bin/env bash
# Set a PR body from a file and verify it landed.
#
#   scripts/pr_body.sh <pr-number> <body-file>
#
# `gh pr edit --body-file` can print only a GraphQL deprecation warning and
# change nothing; the REST PATCH is reliable and the read-back proves it.
set -euo pipefail
PR="$1"; FILE="$2"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
gh api -X PATCH "repos/$REPO/pulls/$PR" -F body=@"$FILE" --silent
LIVE="$(gh pr view "$PR" --json body -q .body)"
if [ "$(printf '%s' "$LIVE" | tr -d '\r')" != "$(tr -d '\r' < "$FILE")" ]; then
  echo "PR #$PR body differs from $FILE after PATCH" >&2; exit 1
fi
echo "PR #$PR body set ($(wc -l < "$FILE") lines)"
